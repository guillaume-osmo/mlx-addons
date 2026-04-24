"""Randomized PCA on Apple Silicon via MLX.

A thin wrapper around :func:`mlx_addons.linalg.randomized_svd` that adds the
standard sklearn ``PCA`` API: mean-centering, ``fit`` / ``transform`` /
``fit_transform`` / ``inverse_transform`` / ``explained_variance_`` /
``explained_variance_ratio_``.

Uses the Metal GPU matmul path of :func:`randomized_svd` for the heavy
projections, so fit + transform on ~10k × 2k inputs is dominated by a
single Metal matmul (< 100 ms on M3 Max) rather than the full SVD (> 4 s).

Example
-------
>>> from mlx_addons.decomposition import PCA
>>> import numpy as np
>>> X = np.random.RandomState(0).randn(10_000, 2048).astype(np.float32)
>>> pca = PCA(n_components=64, random_state=0).fit(X)
>>> Z = pca.transform(X)
>>> Z.shape
(10000, 64)
>>> # Inverse maps back to the original space (lossy at n_components < n_features)
>>> X_hat = pca.inverse_transform(Z)
>>> np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro') < 1.0
True
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..linalg import randomized_svd


class PCA:
    """Randomized principal-component analysis — sklearn-style, Metal-accelerated.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    n_oversamples : int, default=10
        Passed through to :func:`randomized_svd`.
    n_iter : int, default=4
        Power-iteration steps in the underlying randomized SVD. Higher values
        improve accuracy for slow-decay spectra at the cost of additional
        matmuls. ``n_iter=4`` matches sklearn's default for ``svd_solver='randomized'``.
    whiten : bool, default=False
        If True, divide transformed data by ``sqrt(explained_variance_)`` so
        that each output component has unit variance. Matches sklearn behaviour.
    random_state : int, optional
        Seed for the random projection used inside :func:`randomized_svd`.

    Attributes
    ----------
    components_ : (n_components, n_features) np.ndarray
        Principal axes (Vt from the thin SVD of centered X).
    singular_values_ : (n_components,) np.ndarray
    explained_variance_ : (n_components,) np.ndarray
    explained_variance_ratio_ : (n_components,) np.ndarray
    mean_ : (n_features,) np.ndarray
        Per-feature mean removed from X during ``fit``.
    n_samples_ : int
    n_features_in_ : int
    """

    def __init__(
        self,
        n_components: int,
        *,
        n_oversamples: int = 10,
        n_iter: int = 4,
        whiten: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X: Union[np.ndarray, "mx.array"], y=None) -> "PCA":
        """Fit PCA.

        ``X`` can be 2-D ``(n_samples, n_features)`` — standard sklearn behaviour —
        or 3-D ``(batch, n_samples, n_features)``, in which case one independent
        PCA is fit per batch element in a single Metal dispatch and all stateful
        attributes (``components_``, ``mean_``, ``explained_variance_`` …) grow
        a leading batch dimension.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim not in (2, 3):
            raise ValueError(f"PCA expects 2-D or 3-D input, got shape {X.shape}")
        self._batched = (X.ndim == 3)
        axis_samples = -2
        self.mean_ = X.mean(axis=axis_samples)                   # (n_features,) or (batch, n_features)
        Xc = X - np.expand_dims(self.mean_, axis=axis_samples)
        n_samples = X.shape[axis_samples]
        n_features = X.shape[-1]

        k = min(self.n_components, n_samples, n_features)
        _U, S, Vt = randomized_svd(
            Xc,
            n_components=k,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.components_ = Vt
        self.singular_values_ = S

        # sklearn convention: explained_variance_ = S**2 / (n_samples - 1)
        denom = max(n_samples - 1, 1)
        self.explained_variance_ = (S ** 2) / denom
        # Total variance per (batch, feature) — sum over samples axis.
        total_var = np.sum(Xc ** 2, axis=(axis_samples, -1)) / denom
        if self._batched:
            self.explained_variance_ratio_ = self.explained_variance_ / np.maximum(total_var, 1e-20)[:, None]
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / max(float(total_var), 1e-20)
        self.n_samples_ = n_samples
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Union[np.ndarray, "mx.array"]) -> np.ndarray:
        """Project X onto the fitted principal components.

        Mirrors the dimensionality used at ``fit`` time: 2-D fit → 2-D transform,
        3-D fit → 3-D transform. For 3-D the batch dim must match ``components_``'s.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        Xc = X - np.expand_dims(self.mean_, axis=-2)
        if self._batched:
            # (batch, n, n_features) @ (batch, n_features, k)
            Z = Xc @ np.swapaxes(self.components_, -2, -1)
        else:
            Z = Xc @ self.components_.T
        if self.whiten:
            ev = np.maximum(self.explained_variance_, 1e-20)
            Z = Z / np.expand_dims(np.sqrt(ev), axis=-2)
        return Z

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float32))
        if self.whiten:
            ev = np.maximum(self.explained_variance_, 1e-20)
            Z = Z * np.expand_dims(np.sqrt(ev), axis=-2)
        if self._batched:
            X = Z @ self.components_ + np.expand_dims(self.mean_, axis=-2)
        else:
            X = Z @ self.components_ + self.mean_
        return X
