"""Decomposition algorithms built on top of ``mlx_addons.linalg``.

Currently exposes:

- :class:`PCA` — randomized PCA, sklearn-compatible drop-in, Metal-accelerated.

Usage::

    from mlx_addons.decomposition import PCA

    pca = PCA(n_components=32).fit(X)      # X: (n_samples, n_features)
    Z = pca.transform(X_test)              # (n_samples, 32)
    X_hat = pca.inverse_transform(Z)       # (n_samples, n_features)
"""

from ._pca import PCA

__all__ = ["PCA"]
