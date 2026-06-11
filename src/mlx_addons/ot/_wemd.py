"""GPU Wavelet Earth Mover's Distance (WEMD) for MLX on Apple Silicon.

WEMD (Shirdhonkar & Jacobs 2008; Zelesko, Moscovich, Kileel & Singer 2022) approximates
the Wasserstein-1 distance between densities as a **weighted L1 distance in wavelet space**::

    W·x = concat over detail levels l of  2^(l*(1 + d/2)) * wavedecn(x / x.sum())_l
    WEMD(x, y) = || W·x - W·y ||_1

This is O(n) and — for smooth / overlapping densities (its design regime, e.g. cryo-EM maps
or molecular atom-densities) — tracks exact W1 closely, while remaining tractable in 2-D/3-D
where exact optimal transport is not. A DWT is just a 2-filter conv + stride-2 downsample, so
the multilevel n-D transform runs on the GPU via ``mx.conv1d`` (PyWavelets is used only to
fetch the wavelet's filter taps, not at transform time).

Public API:
    wavedecn      - GPU n-D multilevel discrete wavelet transform
    wemd_vectors  - weighted-wavelet vectors whose L1 distance is the WEMD
    pairwise_l1   - fused Metal pairwise L1 (Manhattan) distance matrix
    wemd_rdm      - full WEMD distance matrix between densities
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def _filters(wavelet: str):
    try:
        import pywt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "WEMD requires PyWavelets for the wavelet filter coefficients "
            "(`pip install PyWavelets`)."
        ) from e
    w = pywt.Wavelet(wavelet)
    lo = np.array(w.dec_lo, np.float32)[::-1].copy()      # flip: mx.conv1d is cross-correlation
    hi = np.array(w.dec_hi, np.float32)[::-1].copy()
    K = len(lo)
    return mx.array(lo).reshape(1, K, 1), mx.array(hi).reshape(1, K, 1), K


def _dwt1d_last(x, lo_f, hi_f, K):
    """Single-level 1-D DWT along the last axis of (B, L). pywt 'zero' convention."""
    B, L = x.shape
    xp = mx.pad(x, [(0, 0), (K - 1, K - 1)]).reshape(B, L + 2 * (K - 1), 1)
    cA = mx.conv1d(xp, lo_f)[:, 1::2, 0]
    cD = mx.conv1d(xp, hi_f)[:, 1::2, 0]
    return cA, cD


def _dwt_axis(x, lo_f, hi_f, K, axis):
    x = mx.swapaxes(x, axis, -1)
    shp = x.shape
    cA, cD = _dwt1d_last(x.reshape(-1, shp[-1]), lo_f, hi_f, K)
    L2 = cA.shape[-1]
    cA = cA.reshape(*shp[:-1], L2)
    cD = cD.reshape(*shp[:-1], L2)
    return mx.swapaxes(cA, axis, -1), mx.swapaxes(cD, axis, -1)


def _split_nd(vol, lo_f, hi_f, K):
    subs = {"": vol}
    for ax in range(vol.ndim):
        nxt = {}
        for key, arr in subs.items():
            a, d = _dwt_axis(arr, lo_f, hi_f, K, ax)
            nxt[key + "L"] = a
            nxt[key + "H"] = d
        subs = nxt
    return subs


def wavedecn(volume, wavelet: str = "coif3", level: int = 3):
    """GPU n-D multilevel discrete wavelet transform.

    Parameters
    ----------
    volume : (*grid) array (numpy or mx)
    wavelet : PyWavelets wavelet name (e.g. "coif3", "db2").
    level : number of decomposition levels.

    Returns
    -------
    details : list of dicts; ``details[l]`` holds the ``2**d - 1`` detail sub-bands at
        decomposition step ``l`` (``l = 0`` is the finest scale). Keys are strings over
        {'L','H'} per axis (e.g. 'LLH').
    approx : the all-low residual (coarsest approximation).
    """
    lo_f, hi_f, K = _filters(wavelet)
    approx = mx.array(np.ascontiguousarray(np.asarray(volume, np.float32))) if not isinstance(volume, mx.array) else volume
    d = approx.ndim
    details = []
    for _ in range(level):
        subs = _split_nd(approx, lo_f, hi_f, K)
        approx = subs.pop("L" * d)
        details.append(subs)
    return details, approx


def wemd_vectors(densities, wavelet: str = "coif3", level: int = 4, normalize: bool = True) -> np.ndarray:
    """Weighted-wavelet WEMD vectors for a batch of densities.

    The L1 distance between two returned vectors is the WEMD between the densities.

    Parameters
    ----------
    densities : iterable of identically-shaped non-negative n-D arrays.
    wavelet, level : wavelet name and number of levels.
    normalize : divide each density by its sum (mass normalization) first.

    Returns
    -------
    (N, M) np.ndarray of weighted-wavelet vectors.
    """
    lo_f, hi_f, K = _filters(wavelet)
    out = []
    d = np.ndim(densities[0])
    for x in densities:
        x = np.asarray(x, np.float32)
        if normalize:
            x = x / (x.sum() + 1e-12)
        approx = mx.array(x)
        parts = []
        for l in range(level):                            # l=0 finest -> coarse scales up-weighted
            subs = _split_nd(approx, lo_f, hi_f, K)
            approx = subs.pop("L" * d)
            wt = 2.0 ** (l * (1.0 + d / 2.0))
            for arr in subs.values():
                parts.append(wt * np.array(arr).ravel())
        out.append(np.concatenate(parts).astype(np.float32))
    return np.array(out)


_L1_SRC = """
    uint i = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    uint N = V_shape[0];
    uint M = V_shape[1];
    if (i >= N || j >= N) { return; }
    float s = 0.0f;
    for (uint k = 0; k < M; k++) {
        float dd = V[i * M + k] - V[j * M + k];
        s += (dd < 0.0f) ? -dd : dd;
    }
    rdm[i * N + j] = s;
"""
_l1_kernel = mx.fast.metal_kernel(
    name="pairwise_l1", input_names=["V"], output_names=["rdm"],
    source=_L1_SRC, ensure_row_contiguous=True,
)


def pairwise_l1(V) -> np.ndarray:
    """Fused N×N pairwise L1 (Manhattan) distance matrix of the rows of ``V`` (N, M).

    One Metal thread per (i, j) reduces over the M feature dimension — no (N, N, M)
    intermediate is materialized.
    """
    V = mx.array(np.ascontiguousarray(np.asarray(V, np.float32)))
    N = V.shape[0]
    (rdm,) = _l1_kernel(
        inputs=[V], grid=(N, N, 1), threadgroup=(8, 8, 1),
        output_shapes=[(N, N)], output_dtypes=[mx.float32],
    )
    mx.eval(rdm)
    return np.array(rdm)


def wemd_rdm(densities, wavelet: str = "coif3", level: int = 4, normalize: bool = True) -> np.ndarray:
    """Full N×N WEMD distance matrix between densities (GPU wavelet + fused L1)."""
    return pairwise_l1(wemd_vectors(densities, wavelet, level, normalize))
