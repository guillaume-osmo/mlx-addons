"""Tests for EnsembleRandomProjection + ensemble_mean_predict."""

import numpy as np
import pytest

from mlx_addons.decomposition import (
    EnsembleRandomProjection,
    ensemble_mean_predict,
)
from mlx_addons.decomposition._ensemble import PCA, SparseRandomProjection, GaussianRandomProjection


class TestEnsembleRP:
    def test_default_recipe_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 100)).astype(np.float32)
        ens = EnsembleRandomProjection(
            n_components=32, n_pca=1, n_sparse=2, n_gaussian=2, random_state=0,
        ).fit(X)
        assert ens.n_members_ == 5
        assert ens.n_components_ == 32
        Z = ens.transform(X)
        assert Z.shape == (5, 300, 32)

    def test_member_kinds(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 40)).astype(np.float32)
        ens = EnsembleRandomProjection(
            n_components=16, n_pca=1, n_sparse=2, n_gaussian=3, random_state=0,
        ).fit(X)
        assert ens.member_kinds() == ["pca", "sparse", "sparse", "gaussian", "gaussian", "gaussian"]

    def test_all_members_are_fitted_types(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((80, 20)).astype(np.float32)
        ens = EnsembleRandomProjection(
            n_components=8, n_pca=2, n_sparse=1, n_gaussian=1, random_state=3,
        ).fit(X)
        assert isinstance(ens.members_[0], PCA)
        assert isinstance(ens.members_[1], PCA)
        assert isinstance(ens.members_[2], SparseRandomProjection)
        assert isinstance(ens.members_[3], GaussianRandomProjection)

    def test_reproducibility(self):
        """Same random_state → identical transforms."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 50)).astype(np.float32)
        a = EnsembleRandomProjection(
            n_components=8, n_sparse=2, n_gaussian=2, random_state=7,
        ).fit(X).transform(X)
        b = EnsembleRandomProjection(
            n_components=8, n_sparse=2, n_gaussian=2, random_state=7,
        ).fit(X).transform(X)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_different_seeds_differ(self):
        """Different random_state → different RP components (but same PCA)."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((100, 50)).astype(np.float32)
        a = EnsembleRandomProjection(
            n_components=8, n_pca=0, n_sparse=2, n_gaussian=2, random_state=0,
        ).fit(X).transform(X)
        b = EnsembleRandomProjection(
            n_components=8, n_pca=0, n_sparse=2, n_gaussian=2, random_state=100,
        ).fit(X).transform(X)
        # Shouldn't be bitwise-identical
        assert not np.allclose(a, b, atol=1e-3)

    def test_raises_on_empty_recipe(self):
        with pytest.raises(ValueError, match="must be > 0"):
            EnsembleRandomProjection(n_components=8, n_pca=0, n_sparse=0, n_gaussian=0)

    def test_fit_transform_matches_fit_then_transform(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((200, 40)).astype(np.float32)
        ens1 = EnsembleRandomProjection(n_components=16, random_state=9).fit(X)
        Z1 = ens1.transform(X)
        ens2 = EnsembleRandomProjection(n_components=16, random_state=9)
        Z2 = ens2.fit_transform(X)
        np.testing.assert_allclose(Z1, Z2, atol=1e-5)


class TestEnsembleMeanPredict:
    def test_linear_regressor_averages_correctly(self):
        """Sanity: ensemble_mean_predict of identity fit-predict gives mean of test views."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 30)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)
        ens = EnsembleRandomProjection(
            n_components=8, n_pca=1, n_sparse=1, n_gaussian=1, random_state=0,
        ).fit(X)

        # "Regressor" that just predicts the first feature → lets us check averaging
        def fit_predict(Ztr, ytr, Zte):
            return Zte[:, 0].copy()

        got = ensemble_mean_predict(ens, fit_predict, X, y, X)
        # Expected = mean(Z_member[:, 0]) across members
        Z = ens.transform(X)
        expected = Z[:, :, 0].mean(axis=0)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_sklearn_ridge_integration(self):
        """Full integration with sklearn Ridge — ensemble should beat a single RP."""
        pytest.importorskip("sklearn.linear_model")
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(42)
        # Low-rank structured data so the ensemble can actually help
        n, d, true_k = 500, 200, 20
        U = rng.standard_normal((n, true_k)).astype(np.float32)
        V = rng.standard_normal((true_k, d)).astype(np.float32)
        X = (U @ V + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
        w_true = rng.standard_normal(d).astype(np.float32)
        y = (X @ w_true + 0.1 * rng.standard_normal(n)).astype(np.float32)
        split = 350
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        ens = EnsembleRandomProjection(
            n_components=40, n_pca=1, n_sparse=2, n_gaussian=2, random_state=0,
        ).fit(X_tr)

        def fp(Ztr, ytr, Zte):
            return Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte)

        y_ens = ensemble_mean_predict(ens, fp, X_tr, y_tr, X_te)
        rmse_ens = float(np.sqrt(((y_ens - y_te) ** 2).mean()))

        # Single RP for reference
        single = GaussianRandomProjection(n_components=40, random_state=0).fit(X_tr)
        y_single = fp(single.transform(X_tr), y_tr, single.transform(X_te))
        rmse_single = float(np.sqrt(((y_single - y_te) ** 2).mean()))

        # Ensemble should generally beat a single RP (variance reduction)
        assert rmse_ens < rmse_single + 0.01, (
            f"ensemble {rmse_ens:.3f} did not beat single RP {rmse_single:.3f}"
        )
