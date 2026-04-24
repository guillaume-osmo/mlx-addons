"""Nyström low-rank kernel approximation — sklearn-compatible, Metal-accelerated.

For a kernel ``K(x, y)``, sample ``n_components`` anchor points, compute
``K_mm`` (m×m) and ``K_xm`` (n×m), then approximate the feature map as

    φ(x) = K_xm  U Σ^{-1/2}        where K_mm = U Σ U^T

so that ``φ(x) φ(y)^T ≈ K(x, y)``. The ``(n × n)`` kernel matrix is never
materialized, making the approximation the go-to for large datasets.

This is a drop-in for ``sklearn.kernel_approximation.Nystroem`` on Apple
Silicon. Uses Metal matmul for ``K_xm`` / ``K_mm`` construction; the small
``m × m`` eigendecomposition runs on the MLX CPU stream via
``mx.linalg.eigh``.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from ._kernels import KernelName, pairwise_kernel


class Nystroem:
    """Low-rank kernel approximation via Nyström method.

    Parameters
    ----------
    n_components : int, default=100
        Number of anchor points (rank of the approximation).
    kernel : one of "linear", "rbf", "poly"/"polynomial", "sigmoid". Default "rbf".
    gamma : float, optional
        Kernel scale. Default ``1 / n_features``.
    degree, coef0 : polynomial / sigmoid params.
    random_state : int, optional
        Controls anchor selection.

    Attributes
    ----------
    components_ : (n_components, n_features) np.ndarray
        The sampled anchor points.
    normalization_ : (n_components, n_components) np.ndarray
        ``U Σ^{-1/2}`` so that ``transform(X) = K(X, components_) @ normalization_``.
    component_indices_ : (n_components,) np.ndarray

    Example
    -------
    >>> from mlx_addons.decomposition import Nystroem
    >>> approx = Nystroem(n_components=100, kernel='rbf').fit(X_train)
    >>> Z_train = approx.transform(X_train)     # (n_samples, 100)
    >>> Z_test  = approx.transform(X_test)
    >>> # Z_train @ Z_test.T ≈ K_rbf(X_train, X_test)
    """

    def __init__(
        self,
        n_components: int = 100,
        *,
        kernel: KernelName = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "Nystroem":
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        n_samples = X.shape[0]
        m = min(self.n_components, n_samples)

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_samples, size=m, replace=False)
        idx.sort()
        self.component_indices_ = idx
        self.components_ = X[idx]

        # K_mm via Metal
        K_mm = pairwise_kernel(
            self.components_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        # Symmetric eigendecomposition on the small m×m matrix (CPU stream).
        with mx.stream(mx.cpu):
            K_mm_mx = mx.array(K_mm)
            # Symmetrize to guard against tiny asymmetry from float round-off.
            K_mm_mx = 0.5 * (K_mm_mx + K_mm_mx.T)
            evals, evecs = mx.linalg.eigh(K_mm_mx)         # ascending
            mx.eval(evals, evecs)
        evals_np = np.array(evals, dtype=np.float32)
        evecs_np = np.array(evecs, dtype=np.float32)

        # Reverse to descending, drop tiny / negative eigenvalues.
        order = np.argsort(-evals_np)
        evals_np = evals_np[order]
        evecs_np = evecs_np[:, order]
        eps = 1e-10 * float(evals_np[0] if evals_np[0] > 0 else 1.0)
        keep = evals_np > eps
        evals_np = evals_np[keep]
        evecs_np = evecs_np[:, keep]

        self.normalization_ = evecs_np / np.sqrt(evals_np)[None, :]   # (m, k_eff)
        self._gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        K = pairwise_kernel(
            X, self.components_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        return K @ self.normalization_

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)
