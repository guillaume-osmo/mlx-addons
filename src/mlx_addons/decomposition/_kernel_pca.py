"""Kernel PCA on Apple Silicon via MLX.

Pipeline:

    1. K  = pairwise_kernel(X, X)               # Metal matmul
    2. K̃ = double-center K                      # elementwise
    3. eigh(K̃) → top-k components               # CPU eigh

Transforming new points:

    K_new = kernel(X_new, X_train)
    K̃_new = center K_new with train-fit row/col means
    Z     = K̃_new @ V / sqrt(eigenvalues)

This is a drop-in for ``sklearn.decomposition.KernelPCA`` for the most
common use cases (RBF / poly / linear / sigmoid kernels, fit / transform).

For very large ``n`` the full ``n × n`` kernel matrix becomes the bottleneck;
use :class:`Nystroem` instead.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from ._kernels import KernelName, pairwise_kernel


class KernelPCA:
    """Kernel Principal Component Analysis — sklearn-style.

    Parameters
    ----------
    n_components : int
    kernel : one of "linear", "rbf", "poly"/"polynomial", "sigmoid". Default "rbf".
    gamma, degree, coef0 : kernel parameters.
    remove_zero_eig : bool, default=True
        Drop components whose eigenvalue is effectively zero.

    Attributes
    ----------
    eigenvalues_ : (n_components,) np.ndarray
    eigenvectors_ : (n_train, n_components) np.ndarray
    X_fit_ : (n_train, n_features) np.ndarray
    """

    def __init__(
        self,
        n_components: int,
        *,
        kernel: KernelName = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        remove_zero_eig: bool = True,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.remove_zero_eig = remove_zero_eig

    @staticmethod
    def _center_K(K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Double-center a kernel matrix. Returns (K_centered, row_means, total_mean)."""
        row_means = K.mean(axis=0)                                 # (n,)
        total_mean = float(row_means.mean())
        col_means = K.mean(axis=1)                                 # (n,)
        Kc = K - row_means[None, :] - col_means[:, None] + total_mean
        return Kc, row_means, total_mean

    def fit(self, X: np.ndarray, y=None) -> "KernelPCA":
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        self.X_fit_ = X

        K = pairwise_kernel(
            X, kernel=self.kernel, gamma=self.gamma,
            degree=self.degree, coef0=self.coef0,
        )
        Kc, self._row_means, self._total_mean = self._center_K(K)

        # Symmetrize to guard tiny float asymmetry from the centering.
        Kc = 0.5 * (Kc + Kc.T)
        with mx.stream(mx.cpu):
            evals, evecs = mx.linalg.eigh(mx.array(Kc))
            mx.eval(evals, evecs)
        evals_np = np.array(evals, dtype=np.float32)
        evecs_np = np.array(evecs, dtype=np.float32)

        # eigh gives ascending — reverse to descending, optionally drop near-zeros.
        order = np.argsort(-evals_np)
        evals_np = evals_np[order]
        evecs_np = evecs_np[:, order]
        if self.remove_zero_eig:
            cutoff = 1e-10 * float(evals_np[0] if evals_np[0] > 0 else 1.0)
            nonzero = evals_np > cutoff
            evals_np = evals_np[nonzero]
            evecs_np = evecs_np[:, nonzero]

        k = min(self.n_components, evals_np.shape[0])
        self.eigenvalues_ = evals_np[:k]
        self.eigenvectors_ = evecs_np[:, :k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        K_new = pairwise_kernel(
            X, self.X_fit_,
            kernel=self.kernel, gamma=self.gamma,
            degree=self.degree, coef0=self.coef0,
        )
        # Double-center K_new using the train-fit statistics.
        col_means_new = K_new.mean(axis=1, keepdims=True)            # (n_new, 1)
        Kc_new = K_new - self._row_means[None, :] - col_means_new + self._total_mean
        # Projection: K̃_new @ V / sqrt(λ)
        return Kc_new @ (self.eigenvectors_ / np.sqrt(np.maximum(self.eigenvalues_, 1e-20))[None, :])

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)
