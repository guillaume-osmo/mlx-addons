"""Tests for Nystroem + KernelPCA + pairwise_kernel."""

import numpy as np
import pytest

from mlx_addons.decomposition import Nystroem, KernelPCA, pairwise_kernel


# ──────────────────────────────── pairwise_kernel ──

class TestPairwiseKernel:
    @pytest.mark.parametrize("kernel,kw", [
        ("linear", {}),
        ("rbf", {"gamma": 0.1}),
        ("rbf", {"gamma": None}),       # default gamma
        ("poly", {"gamma": 0.5, "degree": 3, "coef0": 1.0}),
        ("sigmoid", {"gamma": 0.01, "coef0": 0.5}),
    ])
    def test_matches_sklearn(self, kernel, kw):
        sk = pytest.importorskip("sklearn.metrics.pairwise")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 10)).astype(np.float32)
        Y = rng.standard_normal((30, 10)).astype(np.float32)

        K_ours = pairwise_kernel(X, Y, kernel=kernel, **kw)

        fn = {
            "linear": sk.linear_kernel,
            "rbf": sk.rbf_kernel,
            "poly": sk.polynomial_kernel,
            "sigmoid": sk.sigmoid_kernel,
        }[kernel]
        # sklearn API differences
        sk_kw = {}
        if "gamma" in kw and kw["gamma"] is not None:
            sk_kw["gamma"] = kw["gamma"]
        if kernel == "poly":
            sk_kw.update({"degree": kw["degree"], "coef0": kw["coef0"]})
        elif kernel == "sigmoid":
            sk_kw.update({"coef0": kw["coef0"]})
        if kernel == "linear":
            K_sk = sk.linear_kernel(X, Y)
        else:
            K_sk = fn(X, Y, **sk_kw)

        np.testing.assert_allclose(K_ours, K_sk, atol=1e-4, rtol=1e-4)

    def test_symmetric_when_Y_none(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 5)).astype(np.float32)
        K = pairwise_kernel(X, kernel="rbf", gamma=0.2)
        np.testing.assert_allclose(K, K.T, atol=1e-5)

    def test_rbf_diagonal_is_one(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        K = pairwise_kernel(X, kernel="rbf", gamma=0.3)
        np.testing.assert_allclose(np.diag(K), np.ones(30), atol=1e-5)


# ──────────────────────────────── Nystroem ──

class TestNystroem:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 10)).astype(np.float32)
        ny = Nystroem(n_components=50, kernel="rbf", gamma=0.1, random_state=0).fit(X)
        Z = ny.transform(X)
        assert Z.shape[0] == 200
        assert Z.shape[1] <= 50
        assert ny.components_.shape == (50, 10)

    def test_approximates_kernel(self):
        """Z Z^T should approximate the true RBF kernel; error shrinks with m."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((150, 8)).astype(np.float32)
        K_true = pairwise_kernel(X, kernel="rbf", gamma=0.2)

        errs = {}
        for m in [20, 60, 120]:
            ny = Nystroem(n_components=m, kernel="rbf", gamma=0.2, random_state=0).fit(X)
            Z = ny.transform(X)
            K_approx = Z @ Z.T
            errs[m] = np.linalg.norm(K_true - K_approx, "fro") / np.linalg.norm(K_true, "fro")
        # Monotone decrease + final error bounded.
        assert errs[20] >= errs[60] >= errs[120], errs
        assert errs[120] < 0.2, f"with m=120 anchors, relerr={errs[120]:.3e}"

    def test_sklearn_parity(self):
        """Feature-space inner products agree with sklearn's Nystroem within kernel-approx tol."""
        sk = pytest.importorskip("sklearn.kernel_approximation")
        rng = np.random.default_rng(2)
        X = rng.standard_normal((120, 6)).astype(np.float32)
        K_true = pairwise_kernel(X, kernel="rbf", gamma=0.15)

        ours = Nystroem(n_components=80, kernel="rbf", gamma=0.15, random_state=0).fit(X)
        theirs = sk.Nystroem(n_components=80, kernel="rbf", gamma=0.15, random_state=0).fit(X)

        relerr_ours = np.linalg.norm(K_true - ours.transform(X) @ ours.transform(X).T, "fro") / np.linalg.norm(K_true, "fro")
        relerr_theirs = np.linalg.norm(K_true - theirs.transform(X) @ theirs.transform(X).T, "fro") / np.linalg.norm(K_true, "fro")
        # Both should be in the same ballpark (different anchor samples → different approximations).
        assert abs(relerr_ours - relerr_theirs) < 0.1, (relerr_ours, relerr_theirs)


# ──────────────────────────────── KernelPCA ──

class TestKernelPCA:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 6)).astype(np.float32)
        kpca = KernelPCA(n_components=5, kernel="rbf", gamma=0.1).fit(X)
        Z = kpca.transform(X)
        assert Z.shape == (100, 5)

    def test_linear_kpca_equals_pca(self):
        """Linear KernelPCA should produce the same embedding (up to sign) as PCA on centered X."""
        from mlx_addons.decomposition import PCA
        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 8)).astype(np.float32)
        k = 4

        kpca = KernelPCA(n_components=k, kernel="linear").fit(X)
        Z_kpca = kpca.transform(X)

        pca = PCA(n_components=k, random_state=0, n_iter=6).fit(X)
        Z_pca = pca.transform(X)

        # Compare on pairwise distances (scale/sign invariant summary of the embedding)
        def pdist(Z):
            s = (Z[:, None, :] - Z[None, :, :]) ** 2
            return np.sqrt(s.sum(axis=-1))
        D_k = pdist(Z_kpca); D_p = pdist(Z_pca)
        relerr = np.linalg.norm(D_k - D_p) / np.linalg.norm(D_p)
        assert relerr < 0.05, f"pdist relerr = {relerr:.3e}"

    def test_sklearn_parity_rbf(self):
        """Top eigenvalues match sklearn's KernelPCA."""
        sk = pytest.importorskip("sklearn.decomposition")
        rng = np.random.default_rng(5)
        X = rng.standard_normal((150, 6)).astype(np.float32)

        ours = KernelPCA(n_components=10, kernel="rbf", gamma=0.1).fit(X)
        theirs = sk.KernelPCA(n_components=10, kernel="rbf", gamma=0.1).fit(X)
        # sklearn stores eigenvalues in descending order
        rel = np.abs(ours.eigenvalues_ - theirs.eigenvalues_) / np.maximum(theirs.eigenvalues_, 1e-8)
        assert rel.max() < 1e-3, f"max rel err on eigenvalues = {rel.max():.3e}"
