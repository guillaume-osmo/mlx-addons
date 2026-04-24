"""Random projection on Apple Silicon — sklearn-compatible, Metal-accelerated.

Johnson-Lindenstrauss dimensionality reduction via random matrices. Two
classes, drop-in for ``sklearn.random_projection``:

- :class:`GaussianRandomProjection` — entries ~ N(0, 1/n_components).
- :class:`SparseRandomProjection` — Achlioptas / Li sparse random matrix
  with values in ``{-s, 0, +s}`` where ``s = sqrt(1/density/n_components)``.
  Despite the name the matrix is stored dense internally so the projection
  runs as one Metal matmul; "sparse" here refers to the *construction*
  being memory-efficient (only ~``density × d × k`` non-zeros are drawn)
  and the JL lemma still holding. A dedicated GPU sparse matmul kernel is
  future work — see ``TODO`` below.

Example
-------
>>> import numpy as np
>>> from mlx_addons.decomposition import GaussianRandomProjection
>>> X = np.random.RandomState(0).randn(10_000, 8192).astype(np.float32)
>>> rp = GaussianRandomProjection(n_components=64, random_state=0).fit(X)
>>> Z = rp.transform(X)
>>> Z.shape
(10000, 64)
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

import mlx.core as mx
import numpy as np


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int:
    """Minimum n_components so pairwise distances are preserved to ``(1 ± eps)``.

    Matches ``sklearn.random_projection.johnson_lindenstrauss_min_dim``.

    Formula: ``k >= 4 log(n) / (eps^2 / 2 - eps^3 / 3)``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not (0 < eps < 1):
        raise ValueError("eps must be in (0, 1)")
    denom = (eps ** 2) / 2.0 - (eps ** 3) / 3.0
    # sklearn uses truncation (int cast of the float), not ceil — keep parity.
    return int(4.0 * math.log(n_samples) / denom)


class _BaseRandomProjection:
    """Common fit/transform plumbing — subclasses only define _make_matrix."""

    components_: np.ndarray
    n_features_in_: int
    n_components_: int
    _sparse_csr: Optional[tuple]       # (indptr, indices, values) if stored sparse

    def transform(self, X: Union[np.ndarray, mx.array]) -> np.ndarray:
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X_np.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_np.shape[-1]} features, fit saw {self.n_features_in_}"
            )

        # Sparse path (if stored in CSR): compute Z = X @ R.T = (R @ X.T).T
        # via one CSR @ dense matmul on Metal.
        if getattr(self, "_sparse_csr", None) is not None:
            from ..linalg import csr_matmul
            indptr, indices, values = self._sparse_csr
            # (k, n_features) @ (n_features, n_samples) = (k, n_samples)
            Zt = csr_matmul(indptr, indices, values, X_np.T, M=self.n_components_)
            mx.eval(Zt)
            return np.array(Zt, dtype=np.float32).T

        # Dense path
        X_mx = mx.array(X_np)
        C_mx = mx.array(self.components_)
        Z = X_mx @ C_mx.T
        mx.eval(Z)
        return np.array(Z, dtype=np.float32)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def _resolve_n_components(self, n_samples: int, n_features: int) -> int:
        if self.n_components == "auto":
            return johnson_lindenstrauss_min_dim(n_samples, eps=self.eps)
        return int(self.n_components)


class GaussianRandomProjection(_BaseRandomProjection):
    """Gaussian random projection (entries i.i.d. ``N(0, 1/n_components)``).

    Parameters
    ----------
    n_components : int or "auto", default="auto"
        Target dimension. If ``"auto"``, ``johnson_lindenstrauss_min_dim(n_samples, eps)``.
    eps : float, default=0.1
        JL distortion parameter — only used when ``n_components='auto'``.
    random_state : int, optional

    Attributes
    ----------
    components_ : (n_components, n_features) np.ndarray
    n_features_in_ : int
    n_components_ : int
    """

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        *,
        eps: float = 0.1,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    def fit(self, X: Union[np.ndarray, mx.array], y=None) -> "GaussianRandomProjection":
        X = np.asarray(X)
        n_samples, n_features = X.shape
        k = self._resolve_n_components(n_samples, n_features)
        rng = np.random.default_rng(self.random_state)
        # Entries ~ N(0, 1/k); transpose to (k, n_features) for our convention.
        scale = 1.0 / math.sqrt(k)
        self.components_ = (rng.standard_normal((k, n_features)) * scale).astype(np.float32)
        self._sparse_csr = None
        self.n_features_in_ = n_features
        self.n_components_ = k
        return self


class SparseRandomProjection(_BaseRandomProjection):
    """Sparse random projection (Achlioptas / Li).

    Entries ``R[i, j]`` are drawn from::

        +s    with prob density/2
         0    with prob 1 - density
        -s    with prob density/2

    where ``s = sqrt(1 / (density * n_components))`` keeps ``E[R R^T] = I / n_components``.
    Johnson-Lindenstrauss guarantees still apply — see Li, Hastie, Church 2006.

    Parameters
    ----------
    n_components : int or "auto", default="auto"
    density : float or "auto", default="auto"
        Fraction of non-zero entries. ``"auto"`` uses ``1 / sqrt(n_features)``
        (Achlioptas' recommendation for a good JL-preserving, memory-light matrix).
    eps : float, default=0.1
    random_state : int, optional

    Notes
    -----
    The random matrix is constructed sparsely (only ~``density * d * k``
    values drawn) but currently **stored dense** for the Metal matmul. For
    very large ``n_features`` (≳ 100k), memory is still proportional to
    ``d × k × 4`` bytes — no win over ``GaussianRandomProjection``.

    TODO: once ``mlx_addons`` ships a CSR × dense Metal kernel, keep the
    matrix sparse end-to-end for another 3–10× on wide feature spaces.
    """

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        *,
        density: Union[float, Literal["auto"]] = "auto",
        eps: float = 0.1,
        store_sparse: Union[bool, Literal["auto"]] = "auto",
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.density = density
        self.eps = eps
        self.store_sparse = store_sparse
        self.random_state = random_state

    def _resolve_density(self, n_features: int) -> float:
        if self.density == "auto":
            return 1.0 / math.sqrt(max(n_features, 1))
        d = float(self.density)
        if not (0.0 < d <= 1.0):
            raise ValueError(f"density must be in (0, 1], got {d}")
        return d

    def fit(self, X: Union[np.ndarray, mx.array], y=None) -> "SparseRandomProjection":
        X = np.asarray(X)
        n_samples, n_features = X.shape
        k = self._resolve_n_components(n_samples, n_features)
        density = self._resolve_density(n_features)
        scale = math.sqrt(1.0 / (density * k))

        rng = np.random.default_rng(self.random_state)
        # Achlioptas construction: draw sign-or-zero per entry.
        if density >= 1.0:
            mask = np.ones((k, n_features), dtype=bool)
        else:
            mask = rng.random((k, n_features)) < density
        signs = rng.choice([-1.0, 1.0], size=(k, n_features)).astype(np.float32)
        R_dense = np.where(mask, signs * scale, np.float32(0.0))

        # Decide storage format. The dense path is faster at RP-typical
        # shapes (small ``k``, large ``n_samples``) because Metal's dense
        # GEMM beats a one-thread-per-output-cell SpMM when the output
        # (``k × n_samples``) is small-by-large. ``"auto"`` keeps the
        # dense default; pass ``store_sparse=True`` explicitly only if
        # the sparse layout has a downstream use (e.g. memory-constrained
        # or hand-off into a GNN-shaped SpMM).
        use_sparse = self.store_sparse
        if use_sparse == "auto":
            use_sparse = False

        if use_sparse:
            from ..linalg import csr_from_dense
            indptr, indices, values = csr_from_dense(R_dense)
            self._sparse_csr = (indptr, indices, values)
            self.components_ = R_dense                              # kept for API compat
        else:
            self._sparse_csr = None
            self.components_ = R_dense

        self.density_ = density
        self.n_features_in_ = n_features
        self.n_components_ = k
        return self
