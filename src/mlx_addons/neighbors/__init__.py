"""Neighbor-based estimators built on MLX Metal kernels.

Currently exposes:

- :class:`KernelDensity` — sklearn-compatible KDE drop-in, Metal-accelerated,
  with bounded/configurable peak memory (tiled kernel reduction).
- :func:`bootstrap_kde` — GPU bootstrap density band (the compute core of the
  KDE "stylized facts" demo).

Usage::

    from mlx_addons.neighbors import KernelDensity, bootstrap_kde

    kde = KernelDensity(bandwidth=0.2).fit(X)     # X: (n_samples, n_features)
    logp = kde.score_samples(grid)                # matches sklearn to ~1e-6
    p = kde.eval_density(grid)                     # == exp(score_samples), cheaper
"""

from ._kde import KernelDensity, bootstrap_kde, VALID_KERNELS

__all__ = ["KernelDensity", "bootstrap_kde", "VALID_KERNELS"]
