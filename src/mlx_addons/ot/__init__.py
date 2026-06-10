"""
GPU-accelerated optimal transport for MLX on Apple Silicon.

Exact closed-form 1-D Wasserstein-1 (Earth Mover's Distance) via fused Metal kernels,
plus a batched entropic (Sinkhorn) solver for general ground costs.

Usage::

    import mlx.core as mx
    from mlx_addons.ot import wasserstein1d_rdm, wasserstein1d_neighbors, sinkhorn2_batch

    W = mx.random.uniform(shape=(2000, 64))           # 2000 histograms, 64 bins
    D = wasserstein1d_rdm(W, dx=1.0)                   # (2000, 2000) EMD matrix
    offsets, idx = wasserstein1d_neighbors(W, cutoff=0.1)   # O(N) neighbor list

    costs = sinkhorn2_batch(a, b, C, reg=0.05)        # batched entropic OT

Functions:
    wasserstein1d_rdm       - full N×N 1-D Wasserstein-1 distance matrix
    wasserstein1d_neighbors - O(N+edges) neighbor list within a distance cutoff
    sinkhorn2_batch         - batched entropic OT transport cost
"""

from ._wasserstein1d import wasserstein1d_rdm, wasserstein1d_neighbors
from ._sinkhorn import sinkhorn2_batch

__all__ = [
    "wasserstein1d_rdm",
    "wasserstein1d_neighbors",
    "sinkhorn2_batch",
]
