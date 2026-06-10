"""Exact 1-D Wasserstein-1 (Earth Mover's Distance) on Apple Silicon via MLX.

For distributions on a shared, uniformly-spaced 1-D support the Wasserstein-1
distance has a closed form::

    W1(p, q) = ∫ |CDF_p - CDF_q| dx  =  dx · Σ_b |cumsum(p)_b - cumsum(q)_b|

so an all-pairs distance matrix is one ``cumsum`` followed by a pairwise L1 over
the CDFs. Two fused ``mx.fast.metal_kernel`` paths are provided:

* :func:`wasserstein1d_rdm` — full ``N×N`` distance matrix; one thread per ``(i, j)``
  pair writes the result directly, so only the ``N×N`` output is materialized (no
  ``(N, N, B)`` broadcast intermediate).
* :func:`wasserstein1d_neighbors` — a two-pass (count → CSR fill) kernel that returns
  only the pairs within ``cutoff``. The ``N×N`` matrix is **never** materialized, so
  memory is ``O(N + edges)`` instead of ``O(N²)``.

Inputs are ``(N, B)`` arrays of non-negative weights (histograms / profiles) over the
shared support; rows are normalized to sum to one unless ``normalize=False``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


# --- full pairwise RDM: one thread per (i, j) ------------------------------------
_RDM_SRC = """
    uint i = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    uint N = cdf_shape[0];
    uint B = cdf_shape[1];
    if (i >= N || j >= N) { return; }
    float s = 0.0f;
    for (uint b = 0; b < B; b++) {
        float d = cdf[i * B + b] - cdf[j * B + b];
        s += (d < 0.0f) ? -d : d;
    }
    rdm[i * N + j] = s * dx[0];
"""
_rdm_kernel = mx.fast.metal_kernel(
    name="wasserstein1d_rdm",
    input_names=["cdf", "dx"],
    output_names=["rdm"],
    source=_RDM_SRC,
    ensure_row_contiguous=True,
)

# --- O(N)-memory neighbor list: two passes (count, then CSR fill) ----------------
_COUNT_SRC = """
    uint i = thread_position_in_grid.x;
    uint N = cdf_shape[0];
    uint B = cdf_shape[1];
    if (i >= N) { return; }
    float c = cutoff[0]; float dxv = dx[0];
    int count = 0;
    for (uint j = 0; j < N; j++) {
        if (j == i) { continue; }
        float s = 0.0f;
        for (uint b = 0; b < B; b++) { float d = cdf[i*B+b]-cdf[j*B+b]; s += (d<0.0f)?-d:d; }
        if (s * dxv <= c) { count++; }
    }
    num_neighbors[i] = count;
"""
_FILL_SRC = """
    uint i = thread_position_in_grid.x;
    uint N = cdf_shape[0];
    uint B = cdf_shape[1];
    if (i >= N) { return; }
    float c = cutoff[0]; float dxv = dx[0];
    int base = offsets[i]; int pos = 0;
    for (uint j = 0; j < N; j++) {
        if (j == i) { continue; }
        float s = 0.0f;
        for (uint b = 0; b < B; b++) { float d = cdf[i*B+b]-cdf[j*B+b]; s += (d<0.0f)?-d:d; }
        if (s * dxv <= c) { indices[base + pos] = (int)j; pos++; }
    }
"""
_count_kernel = mx.fast.metal_kernel(
    name="wasserstein1d_nbr_count", input_names=["cdf", "dx", "cutoff"],
    output_names=["num_neighbors"], source=_COUNT_SRC, ensure_row_contiguous=True,
)
_fill_kernel = mx.fast.metal_kernel(
    name="wasserstein1d_nbr_fill", input_names=["cdf", "dx", "cutoff", "offsets"],
    output_names=["indices"], source=_FILL_SRC, ensure_row_contiguous=True,
)


def _cdf(weights, normalize: bool) -> tuple[mx.array, int]:
    W = mx.array(np.ascontiguousarray(np.asarray(weights, dtype=np.float32)))
    if normalize:
        W = W / mx.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return mx.cumsum(W, axis=1).astype(mx.float32), int(W.shape[0])


def wasserstein1d_rdm(weights, *, dx: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Full ``N×N`` 1-D Wasserstein-1 distance matrix.

    Parameters
    ----------
    weights : (N, B) array
        Non-negative weights of ``N`` distributions over a shared ``B``-bin support.
    dx : float
        Spacing between adjacent support points (uniform grid).
    normalize : bool
        If True (default) normalize each row to sum to one before computing.

    Returns
    -------
    (N, N) np.ndarray of Wasserstein-1 distances.
    """
    cdf, N = _cdf(weights, normalize)
    (rdm,) = _rdm_kernel(
        inputs=[cdf, mx.array([float(dx)], dtype=mx.float32)],
        grid=(N, N, 1), threadgroup=(16, 16, 1),
        output_shapes=[(N, N)], output_dtypes=[mx.float32],
    )
    mx.eval(rdm)
    return np.array(rdm)


def wasserstein1d_neighbors(
    weights, cutoff: float, *, dx: float = 1.0, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbors within ``cutoff`` Wasserstein-1 distance, in ``O(N + edges)`` memory.

    The full distance matrix is never materialized: a fused count pass sizes the CSR
    arrays, then a fill pass writes neighbor indices.

    Parameters
    ----------
    weights : (N, B) array of non-negative weights.
    cutoff : float
        Keep pairs with ``W1(i, j) <= cutoff`` (self excluded).
    dx, normalize : see :func:`wasserstein1d_rdm`.

    Returns
    -------
    offsets : (N + 1,) int32 CSR row offsets.
    indices : (edges,) int32 neighbor column indices; row ``i`` is
        ``indices[offsets[i]:offsets[i + 1]]``.
    """
    cdf, N = _cdf(weights, normalize)
    dxa = mx.array([float(dx)], dtype=mx.float32)
    cbuf = mx.array([float(cutoff)], dtype=mx.float32)
    num = _count_kernel(
        inputs=[cdf, dxa, cbuf], grid=(N, 1, 1), threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)], output_dtypes=[mx.int32],
    )[0]
    mx.eval(num)
    offsets = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(np.array(num), out=offsets[1:])
    total = int(offsets[-1])
    indices = _fill_kernel(
        inputs=[cdf, dxa, cbuf, mx.array(offsets)], grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(max(1, total),)], output_dtypes=[mx.int32],
    )[0]
    mx.eval(indices)
    return offsets, np.array(indices)[:total]
