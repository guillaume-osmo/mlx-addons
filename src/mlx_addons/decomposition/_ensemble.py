"""EnsembleRandomProjection — mix PCA + random projections into one fit/transform.

A composite feature-reduction operator: fits one ``PCA`` plus ``n_sparse``
independent ``SparseRandomProjection`` plus ``n_gaussian`` independent
``GaussianRandomProjection`` on the same input. ``transform`` returns the
stacked low-dim views with shape ``(n_members, n_samples, n_components)``.

Why mix PCA with random projection?

    - **PCA** captures the high-variance directions exactly → strong signal on
      large, well-conditioned datasets.
    - **Random projections** preserve pairwise distances uniformly (Johnson-
      Lindenstrauss) → more robust when PCA's top eigenvectors are noisy
      (ill-conditioned, near-degenerate spectra, small n).
    - **Averaging predictions** across the ensemble cancels the per-basis
      variance while keeping each member's strengths.

On MoleculeACE (ChemeleonSMD v5 fingerprints → PCA/RP(128) → TabICL-MLX),
measured on the first 10 targets:

    PCA alone                  : mean RMSE 0.6281
    SparseRP × 5 averaged      : mean RMSE 0.6239  (-0.004)
    PCA + 2 SRP + 2 GRP mix    : mean RMSE 0.6215  (-0.007) ← recommended
    2×PCA + 4 SRP + 4 GRP mix  : mean RMSE 0.6216  (-0.007)

Usage
-----

>>> from mlx_addons.decomposition import EnsembleRandomProjection
>>> ens = EnsembleRandomProjection(
...     n_components=128, n_pca=1, n_sparse=2, n_gaussian=2,
...     random_state=42,
... ).fit(X_all)         # X_all: (n_samples, n_features)
>>> Z = ens.transform(X_all)            # (5, n_samples, 128)
>>> # Typical pattern: fit a regressor on each member, average predictions
>>> preds = np.stack([model.fit(Z[i], y).predict(Z_test[i]) for i in range(len(Z))])
>>> final = preds.mean(axis=0)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ._pca import PCA
from ._random_projection import GaussianRandomProjection, SparseRandomProjection


class EnsembleRandomProjection:
    """Mixed PCA + RP ensemble feature reducer.

    Parameters
    ----------
    n_components : int
        Target dimension for every ensemble member.
    n_pca : int, default=1
        Number of PCA members (fit is deterministic — fitting more than one
        gives the same result, so default is 1; >1 is useful only if you want
        to weight PCA more heavily when averaging downstream predictions).
    n_sparse : int, default=2
        Number of ``SparseRandomProjection`` members with different seeds.
    n_gaussian : int, default=2
        Number of ``GaussianRandomProjection`` members with different seeds.
    density : float or "auto", default="auto"
        Passed through to each ``SparseRandomProjection``.
    random_state : int, optional
        Master seed. Each member gets ``random_state + i`` so ``fit`` is
        deterministic and reproducible given this single value.
    pca_kwargs, sparse_kwargs, gaussian_kwargs : dict, optional
        Extra kwargs forwarded to ``PCA``, ``SparseRandomProjection``, and
        ``GaussianRandomProjection`` respectively.

    Attributes
    ----------
    members_ : list of fitted (PCA | SparseRandomProjection | GaussianRandomProjection)
        Length ``n_pca + n_sparse + n_gaussian``.
    n_features_in_ : int
    n_components_ : int
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        n_pca: int = 1,
        n_sparse: int = 2,
        n_gaussian: int = 2,
        density: Union[float, str] = "auto",
        random_state: Optional[int] = 42,
        pca_kwargs: Optional[dict] = None,
        sparse_kwargs: Optional[dict] = None,
        gaussian_kwargs: Optional[dict] = None,
    ):
        if n_pca + n_sparse + n_gaussian == 0:
            raise ValueError("At least one of n_pca / n_sparse / n_gaussian must be > 0")
        self.n_components = n_components
        self.n_pca = n_pca
        self.n_sparse = n_sparse
        self.n_gaussian = n_gaussian
        self.density = density
        self.random_state = random_state
        self.pca_kwargs = pca_kwargs or {}
        self.sparse_kwargs = sparse_kwargs or {}
        self.gaussian_kwargs = gaussian_kwargs or {}

    def fit(self, X: np.ndarray, y=None) -> "EnsembleRandomProjection":
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim != 2:
            raise ValueError(f"EnsembleRandomProjection expects 2-D input, got {X.shape}")
        base_seed = self.random_state if self.random_state is not None else 0

        members: List = []
        # PCA (deterministic — duplicate fits add nothing, but we allow n_pca > 1
        # so a caller can reweight PCA in downstream averages).
        for i in range(self.n_pca):
            pca = PCA(
                n_components=self.n_components,
                random_state=base_seed,                 # PCA is deterministic here
                **self.pca_kwargs,
            ).fit(X)
            members.append(pca)

        # SparseRP members
        for i in range(self.n_sparse):
            srp = SparseRandomProjection(
                n_components=self.n_components,
                density=self.density,
                random_state=base_seed + 1 + i,
                **self.sparse_kwargs,
            ).fit(X)
            members.append(srp)

        # GaussianRP members
        for i in range(self.n_gaussian):
            grp = GaussianRandomProjection(
                n_components=self.n_components,
                random_state=base_seed + 1 + self.n_sparse + i,
                **self.gaussian_kwargs,
            ).fit(X)
            members.append(grp)

        self.members_ = members
        self.n_features_in_ = X.shape[1]
        self.n_components_ = self.n_components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return stacked projections of shape ``(n_members, n_samples, n_components)``."""
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        Zs = [m.transform(X) for m in self.members_]
        return np.stack(Zs, axis=0).astype(np.float32)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_members_(self) -> int:
        return len(self.members_)

    def member_kinds(self) -> List[str]:
        """Friendly labels for each member (``"pca" | "sparse" | "gaussian"``)."""
        return [type(m).__name__.replace("RandomProjection", "").lower() for m in self.members_]


# Convenience helper — not a class, just the pattern users typically want.

def ensemble_mean_predict(
    ens: EnsembleRandomProjection,
    fit_predict_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit one regressor per ensemble member and average predictions.

    Parameters
    ----------
    ens : fitted EnsembleRandomProjection
    fit_predict_fn : callable
        ``fit_predict_fn(Z_train, y_train, Z_test) -> y_pred`` — e.g. a
        lambda wrapping a sklearn-compatible regressor's ``fit`` + ``predict``.
    X_train, y_train, X_test : arrays

    Returns
    -------
    y_pred : (n_test,) np.ndarray — mean prediction across ensemble members.

    Example
    -------
    >>> from tabicl_mlx import TabICLRegressorMLX
    >>> def fp(Ztr, ytr, Zte):
    ...     m = TabICLRegressorMLX(n_estimators=8).fit(Ztr, ytr)
    ...     return m.predict(Zte).flatten()
    >>> y_pred = ensemble_mean_predict(ens, fp, X_tr, y_tr, X_te)
    """
    Z_tr = ens.transform(X_train)         # (M, n_tr, k)
    Z_te = ens.transform(X_test)          # (M, n_te, k)
    preds = np.stack([
        fit_predict_fn(Z_tr[i], y_train, Z_te[i]) for i in range(ens.n_members_)
    ], axis=0)
    return preds.mean(axis=0)
