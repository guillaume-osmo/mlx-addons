"""Tests for mlx_addons.decomposition.PCA."""

import numpy as np
import pytest

from mlx_addons.decomposition import PCA


class TestPCA:
    @pytest.mark.parametrize("shape,k", [
        ((500, 128), 32),
        ((1000, 64), 16),
        ((2000, 256), 48),
    ])
    def test_fit_transform_shape(self, shape, k):
        rng = np.random.default_rng(0)
        X = rng.standard_normal(shape).astype(np.float32)
        pca = PCA(n_components=k, random_state=0).fit(X)
        Z = pca.transform(X)
        assert Z.shape == (shape[0], k)
        assert pca.components_.shape == (k, shape[1])
        assert pca.mean_.shape == (shape[1],)

    def test_mean_centering(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 64)).astype(np.float32) + 5.0  # shifted
        pca = PCA(n_components=8, random_state=0).fit(X)
        np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-5)

    def test_explained_variance_matches_sklearn(self):
        """Top singular values match sklearn's PCA within relative tolerance."""
        sk = pytest.importorskip("sklearn.decomposition")
        rng = np.random.default_rng(2)
        X = rng.standard_normal((800, 80)).astype(np.float32)
        k = 16

        mine = PCA(n_components=k, random_state=0, n_iter=6).fit(X)
        skl = sk.PCA(n_components=k, random_state=0, svd_solver="full").fit(X)

        rel = np.abs(mine.explained_variance_ - skl.explained_variance_) / skl.explained_variance_
        assert rel.max() < 5e-2, f"max rel err = {rel.max():.3e}"

    def test_inverse_transform_recovers_full_rank(self):
        """If n_components == rank(X), inverse_transform recovers X exactly (up to float32)."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 8)).astype(np.float32)
        pca = PCA(n_components=8, random_state=0, n_iter=6).fit(X)
        Z = pca.transform(X)
        X_hat = pca.inverse_transform(Z)
        np.testing.assert_allclose(X_hat, X, atol=1e-3)

    def test_whiten(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((500, 32)).astype(np.float32)
        pca = PCA(n_components=10, whiten=True, random_state=0, n_iter=6).fit(X)
        Z = pca.transform(X)
        # Whitened components should have ~unit variance (sample variance)
        var = Z.var(axis=0, ddof=1)
        assert np.all(np.abs(var - 1.0) < 0.1), f"var={var}"

    def test_variance_ratio_ordered(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((400, 60)).astype(np.float32)
        pca = PCA(n_components=20, random_state=0).fit(X)
        # Monotonically non-increasing (svd returns sorted)
        ratios = pca.explained_variance_ratio_
        assert np.all(ratios[:-1] >= ratios[1:] - 1e-6)
        assert ratios.sum() <= 1.0 + 1e-5
