"""Sanity benchmark for EnsembleRandomProjection on synthetic data.

The scientific finding that PCA + RP mix beats either alone was measured
on MoleculeACE (see mlx_addons README "Patterns & recipes" section). This
script reproduces the pattern on a structured synthetic regression problem,
so you can validate the mechanism before wiring it into your own pipeline.
"""

import time
import numpy as np

from mlx_addons.decomposition import (
    PCA,
    GaussianRandomProjection,
    SparseRandomProjection,
    EnsembleRandomProjection,
    ensemble_mean_predict,
)


def _make_regression(n, d, true_k, seed=0):
    """Low-rank design matrix + linear target, to see feature-map effects."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, true_k)).astype(np.float32)
    V = rng.standard_normal((true_k, d)).astype(np.float32)
    X = (U @ V + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _ridge_fit_predict(alpha=1.0):
    from sklearn.linear_model import Ridge

    def fp(Ztr, ytr, Zte):
        return Ridge(alpha=alpha).fit(Ztr, ytr).predict(Zte)
    return fp


def rmse(y, yhat):
    return float(np.sqrt(((y - yhat) ** 2).mean()))


if __name__ == "__main__":
    print("=" * 80)
    print("  Ensemble random projection vs single PCA / RP (synthetic regression)")
    print("=" * 80)

    X, y = _make_regression(n=800, d=300, true_k=30, seed=0)
    split = 600
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    ridge = _ridge_fit_predict(alpha=1.0)
    # Fit all feature maps on the TRAIN set (unsupervised)
    k = 64
    t0 = time.perf_counter()
    pca = PCA(n_components=k, random_state=0).fit(X_tr)
    y_pca = ridge(pca.transform(X_tr), y_tr, pca.transform(X_te))
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Single SparseRP
    srp = SparseRandomProjection(n_components=k, random_state=0).fit(X_tr)
    y_srp1 = ridge(srp.transform(X_tr), y_tr, srp.transform(X_te))
    t_srp1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Single GaussianRP
    grp = GaussianRandomProjection(n_components=k, random_state=0).fit(X_tr)
    y_grp1 = ridge(grp.transform(X_tr), y_tr, grp.transform(X_te))
    t_grp1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Ensemble: 1 PCA + 2 SRP + 2 GRP
    ens = EnsembleRandomProjection(
        n_components=k, n_pca=1, n_sparse=2, n_gaussian=2, random_state=0,
    ).fit(X_tr)
    y_ens = ensemble_mean_predict(ens, ridge, X_tr, y_tr, X_te)
    t_ens = time.perf_counter() - t0

    print(f"\n  Ridge on raw X   : RMSE={rmse(y_te, ridge(X_tr, y_tr, X_te)):.4f}")
    print(f"  PCA (k={k})         : RMSE={rmse(y_te, y_pca):.4f}   time {t_pca*1000:6.1f} ms")
    print(f"  SparseRP × 1 (k={k}): RMSE={rmse(y_te, y_srp1):.4f}   time {t_srp1*1000:6.1f} ms")
    print(f"  GaussRP  × 1 (k={k}): RMSE={rmse(y_te, y_grp1):.4f}   time {t_grp1*1000:6.1f} ms")
    print(f"  Ensemble (1+2+2)   : RMSE={rmse(y_te, y_ens):.4f}   time {t_ens*1000:6.1f} ms  "
          f"(5x regressor fits)")

    print("\nScaling: ensemble size sweep (n_sparse = n_gaussian = k_half, n_pca=1)")
    for half in [1, 2, 4, 8]:
        ens = EnsembleRandomProjection(
            n_components=k, n_pca=1, n_sparse=half, n_gaussian=half, random_state=0,
        ).fit(X_tr)
        y_pred = ensemble_mean_predict(ens, ridge, X_tr, y_tr, X_te)
        print(f"  n_pca=1 n_sparse={half} n_gaussian={half}  (ensemble of {1+2*half}): "
              f"RMSE={rmse(y_te, y_pred):.4f}")
