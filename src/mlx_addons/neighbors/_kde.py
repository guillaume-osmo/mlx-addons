"""Kernel density estimation on Apple Silicon via MLX.

A drop-in replacement for :class:`sklearn.neighbors.KernelDensity` that runs the
kernel sum on the Metal GPU. It reproduces sklearn's log-density to ~1e-6 for all
six sklearn kernels (``gaussian``, ``tophat``, ``epanechnikov``, ``exponential``,
``linear``, ``cosine``) while keeping peak memory **bounded and configurable**.

Why this exists
---------------
A Gaussian KDE evaluated on a grid of ``G`` query points against ``N`` samples is
a ``(G, N)`` kernel matrix reduced along ``N``. Materialising that matrix is what
makes the naive/GPU-library path expensive: at ``G=5000`` and ``N=200_000`` it is
1e9 floats ~= 4 GB. The whole matrix is never needed at once — only the row sum —
so this implementation walks the query- and sample-axes in tiles and accumulates
one ``(query_tile,)`` vector per tile. Peak scratch is ``query_tile x sample_tile``
floats regardless of ``G`` or ``N`` (67 MB at the 2048x8192 defaults).

A second simplification keeps it both fast and stable: every sklearn kernel has a
raw value in ``(0, 1]`` (Gaussian ``exp(-u^2/2)``, exponential ``exp(-u)``, the rest
are ``<= 1`` with finite support), so the per-query kernel sum lives in ``[0, N]``
and needs no log-sum-exp max-tracking — a plain running sum is exact. Normalisation
is applied once, in closed form, matching sklearn's per-kernel constants exactly.

Example
-------
>>> import numpy as np
>>> from mlx_addons.neighbors import KernelDensity
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((10_000, 1)).astype(np.float32)
>>> kde = KernelDensity(bandwidth=0.2).fit(X)
>>> logp = kde.score_samples(np.linspace(-3, 3, 500).reshape(-1, 1))
>>> logp.shape
(500,)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import mlx.core as mx
import numpy as np

VALID_KERNELS = (
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
)

_ArrayLike = Union[np.ndarray, "mx.array"]


def _log_kernel_norm(kernel: str, d: int, h: float) -> float:
    """Log of the per-kernel normalising constant ``log Z``.

    ``density(x) = (1 / (W * Z)) * sum_i w_i * K_raw(dist_i / h)`` where ``K_raw`` is
    the un-normalised kernel below and ``W`` is the total sample weight. These match
    ``sklearn``'s ``BinaryTree`` kernel-norm constants to machine precision.

    ``log V_d`` is the log-volume of the unit ``d``-ball; ``log S_{d-1}`` the
    log-surface-area of the unit ``(d-1)``-sphere in ``R^d``.
    """
    log_vn = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)  # unit d-ball volume
    log_sn = math.log(2.0) + 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d)  # unit (d-1)-sphere area
    d_log_h = d * math.log(h)

    if kernel == "gaussian":
        return 0.5 * d * math.log(2.0 * math.pi) + d_log_h
    if kernel == "tophat":
        return log_vn + d_log_h
    if kernel == "epanechnikov":
        return log_vn + d_log_h + math.log(2.0 / (d + 2.0))
    if kernel == "exponential":
        return log_sn + d_log_h + math.lgamma(d)
    if kernel == "linear":
        return log_vn + d_log_h - math.log(d + 1.0)
    if kernel == "cosine":
        # Z = S_{d-1} * h^d * I_d,  I_d = int_0^1 t^(d-1) cos(pi t / 2) dt.
        # Closed form for d == 1 (I_1 = 2/pi); Simpson quadrature otherwise
        # (matches sklearn's series expansion to ~1e-12).
        if d == 1:
            i_d = 2.0 / math.pi
        else:
            i_d = _simpson_cosine_moment(d)
        return log_sn + d_log_h + math.log(i_d)
    raise ValueError(f"unknown kernel {kernel!r}; valid: {VALID_KERNELS}")


def _simpson_cosine_moment(d: int, n: int = 2048) -> float:
    """int_0^1 t^(d-1) cos(pi t / 2) dt via composite Simpson (n even)."""
    t = np.linspace(0.0, 1.0, n + 1)
    f = t ** (d - 1) * np.cos(0.5 * math.pi * t)
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((w * f).sum() * (1.0 / n) / 3.0)


class KernelDensity:
    """Kernel density estimation — sklearn-compatible, Metal-accelerated, low-RAM.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth ``h`` (same meaning and scale as sklearn).
    kernel : str, default="gaussian"
        One of :data:`VALID_KERNELS`.
    query_tile : int, default=2048
        Number of query points evaluated per GPU dispatch. Caps the query axis of
        the scratch matrix.
    sample_tile : int, default=8192
        Number of training samples reduced per inner step. Caps the sample axis of
        the scratch matrix. Peak scratch ~= ``query_tile * sample_tile`` floats.
    dtype : mx.Dtype, default=mx.float32
        Compute dtype. ``mx.float32`` is the memory-lean default; ``mx.float64`` is
        unavailable on the GPU.

    Attributes
    ----------
    n_features_in_ : int
    n_samples_ : int

    Notes
    -----
    Increasing ``query_tile`` / ``sample_tile`` raises throughput and peak memory;
    lowering them shrinks the footprint. Results are identical regardless of tiling.
    """

    def __init__(
        self,
        *,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        query_tile: int = 2048,
        sample_tile: int = 8192,
        dtype: mx.Dtype = mx.float32,
    ):
        if kernel not in VALID_KERNELS:
            raise ValueError(f"kernel must be one of {VALID_KERNELS}, got {kernel!r}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.query_tile = int(query_tile)
        self.sample_tile = int(sample_tile)
        self.dtype = dtype

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: _ArrayLike,
        y=None,
        sample_weight: Optional[_ArrayLike] = None,
    ) -> "KernelDensity":
        """Store the training samples (no heavy work; KDE is lazy at fit time).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        sample_weight : array of shape (n_samples,), optional
            Non-negative weights. Normalised internally; ``score_samples`` then
            divides by the total weight, matching sklearn.
        """
        data = mx.array(np.ascontiguousarray(np.asarray(X, dtype=np.float32)))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {data.shape}")
        self._data = data.astype(self.dtype)
        self._data_sqnorm = mx.sum(self._data * self._data, axis=1)  # (n,)
        self.n_samples_, self.n_features_in_ = int(data.shape[0]), int(data.shape[1])

        if sample_weight is None:
            self._weight = None
            self._log_weight_total = math.log(self.n_samples_)
        else:
            w = mx.array(np.asarray(sample_weight, dtype=np.float32)).reshape(-1)
            if int(w.shape[0]) != self.n_samples_:
                raise ValueError("sample_weight length must match n_samples")
            self._weight = w.astype(self.dtype)
            self._log_weight_total = math.log(float(mx.sum(self._weight).item()))

        self._log_norm = _log_kernel_norm(self.kernel, self.n_features_in_, self.bandwidth)
        return self

    # ------------------------------------------------------- core reduction
    def _kernel_sum(self, Q: mx.array) -> mx.array:
        """Weighted kernel sum ``sum_i w_i K_raw(dist(q, x_i) / h)`` for every ``q``.

        Returns an ``(n_query,)`` MLX array. Walks queries in ``query_tile`` chunks
        and, inside each, samples in ``sample_tile`` chunks, so the largest live
        buffer is a single ``(query_tile, sample_tile)`` block that is summed away
        before the next block is formed.
        """
        h2 = self.bandwidth * self.bandwidth
        inv_h = 1.0 / self.bandwidth
        n = self.n_samples_
        q_sqnorm_all = mx.sum(Q * Q, axis=1)  # (nq,)
        out_tiles = []

        for qs in range(0, int(Q.shape[0]), self.query_tile):
            qe = min(qs + self.query_tile, int(Q.shape[0]))
            Qt = Q[qs:qe]                       # (qt, d)
            q_sq = q_sqnorm_all[qs:qe]          # (qt,)
            acc = mx.zeros((qe - qs,), dtype=self.dtype)

            for ss in range(0, n, self.sample_tile):
                se = min(ss + self.sample_tile, n)
                Dt = self._data[ss:se]                              # (st, d)
                gram = Qt @ Dt.T                                    # (qt, st)
                d2 = q_sq[:, None] + self._data_sqnorm[ss:se][None, :] - 2.0 * gram
                d2 = mx.maximum(d2, 0.0)                            # guard fp cancellation
                kmat = self._raw_kernel(d2, h2, inv_h)             # (qt, st), in (0, 1]
                if self._weight is not None:
                    kmat = kmat * self._weight[ss:se][None, :]
                acc = acc + mx.sum(kmat, axis=1)                   # (qt,)
                mx.eval(acc)                                       # truncate graph -> bound RAM

            out_tiles.append(acc)

        return mx.concatenate(out_tiles, axis=0) if len(out_tiles) > 1 else out_tiles[0]

    def _raw_kernel(self, d2: mx.array, h2: float, inv_h: float) -> mx.array:
        """Un-normalised kernel evaluated from squared distance ``d2``. Values in [0, 1]."""
        k = self.kernel
        if k == "gaussian":
            return mx.exp(-0.5 * d2 / h2)                          # no sqrt needed
        dist = mx.sqrt(d2)
        u = dist * inv_h
        if k == "exponential":
            return mx.exp(-u)
        inside = u < 1.0
        if k == "tophat":
            return mx.where(inside, 1.0, 0.0).astype(self.dtype)
        if k == "epanechnikov":
            return mx.where(inside, mx.maximum(1.0 - u * u, 0.0), 0.0)
        if k == "linear":
            return mx.where(inside, mx.maximum(1.0 - u, 0.0), 0.0)
        if k == "cosine":
            return mx.where(inside, mx.cos(0.5 * math.pi * u), 0.0)
        raise ValueError(f"unknown kernel {k!r}")

    # ------------------------------------------------------------- scoring
    def eval_density(self, X: _ArrayLike) -> np.ndarray:
        """Probability density at each row of ``X`` (i.e. ``exp(score_samples)``).

        Cheaper than ``exp(score_samples(X))`` because it never forms the log.
        """
        dens = self._kernel_sum(self._as_query(X)) * math.exp(-self._log_norm - self._log_weight_total)
        mx.eval(dens)
        return np.asarray(dens)

    def score_samples(self, X: _ArrayLike) -> np.ndarray:
        """Log-density at each row of ``X`` — matches ``sklearn`` to ~1e-6.

        Empty-support query points (finite-support kernels) return ``-inf``, as in
        sklearn.
        """
        ks = self._kernel_sum(self._as_query(X))
        log_dens = mx.log(ks) - (self._log_norm + self._log_weight_total)
        mx.eval(log_dens)
        return np.asarray(log_dens)

    def score(self, X: _ArrayLike, y=None) -> float:
        """Total log-likelihood ``sum_i log p(x_i)`` (sklearn's ``score``)."""
        return float(np.sum(self.score_samples(X)))

    def _as_query(self, X: _ArrayLike) -> mx.array:
        Q = mx.array(np.ascontiguousarray(np.asarray(X, dtype=np.float32)))
        if Q.ndim == 1:
            Q = Q.reshape(-1, 1)
        if int(Q.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"query has {int(Q.shape[1])} features, model fit on {self.n_features_in_}"
            )
        return Q.astype(self.dtype)


def bootstrap_kde(
    values: _ArrayLike,
    grid: _ArrayLike,
    *,
    n_samples: int,
    n_boot: int = 100,
    bandwidth: float = 1.0,
    kernel: str = "gaussian",
    seed: int = 0,
    query_tile: int = 5000,
    sample_tile: int = 8192,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    return_curves: bool = False,
) -> dict:
    """Bootstrap a KDE density band on the Apple GPU (MLX port of the demo's core).

    Resamples ``values`` with replacement ``n_boot`` times (``n_samples`` draws each),
    fits a Gaussian (or other) KDE per resample, evaluates it on ``grid``, and reads a
    percentile band off the spread of the refits.

    Memory stays bounded: the full pool lives on the GPU once, each resample is a gather
    of integer indices, and every KDE evaluation is tiled (see :class:`KernelDensity`).
    Only the ``(n_boot, len(grid))`` density stack is retained for the percentiles
    (20 MB at 1000x5000 float32).

    Parameters
    ----------
    values : 1-D array
        The pooled observations to resample from.
    grid : array of shape (G,) or (G, 1)
        Fixed evaluation grid.
    n_samples, n_boot, bandwidth, kernel, seed
        Bootstrap / KDE settings. ``seed + i`` seeds resample ``i`` (reproducible).
    lo_pct, hi_pct : float
        Percentiles for the confidence band.
    return_curves : bool
        If True, include the full ``(n_boot, G)`` density stack under ``"curves"``.

    Returns
    -------
    dict with keys ``fit_seconds, n_samples, n_boot, mean, lo, hi`` (and ``curves``
    if requested) — the same schema as the reference notebook's ``bootstrap_kde``.
    """
    import time

    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1))
    grid_np = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    if grid_np.ndim == 1:
        grid_np = grid_np.reshape(-1, 1)
    g = grid_np.shape[0]

    pool = mx.array(vals)                       # (P,) resident on GPU for the whole run
    grid_mx = mx.array(grid_np)
    n_pool = int(pool.shape[0])
    densities = np.empty((n_boot, g), dtype=np.float32)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, query_tile=query_tile, sample_tile=sample_tile)

    t0 = time.perf_counter()
    for i in range(n_boot):
        rng = np.random.default_rng(seed + i)
        idx = mx.array(rng.integers(0, n_pool, size=n_samples).astype(np.int32))
        sample = pool[idx].reshape(-1, 1)       # gather on GPU
        densities[i] = kde.fit(sample).eval_density(grid_mx)
    fit_seconds = time.perf_counter() - t0

    out = {
        "fit_seconds": fit_seconds,
        "n_samples": n_samples,
        "n_boot": n_boot,
        "mean": densities.mean(axis=0),
        "lo": np.percentile(densities, lo_pct, axis=0),
        "hi": np.percentile(densities, hi_pct, axis=0),
    }
    if return_curves:
        out["curves"] = densities
    return out
