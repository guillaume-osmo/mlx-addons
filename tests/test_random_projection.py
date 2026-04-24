"""Tests for GaussianRandomProjection, SparseRandomProjection, csr_matmul."""

import numpy as np
import pytest

from mlx_addons.decomposition import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from mlx_addons.linalg import csr_matmul, csr_from_dense


class TestJLDim:
    def test_matches_sklearn(self):
        sk = pytest.importorskip("sklearn.random_projection")
        for n, eps in [(1000, 0.1), (10000, 0.2), (100, 0.05)]:
            assert johnson_lindenstrauss_min_dim(n, eps) == sk.johnson_lindenstrauss_min_dim(n, eps=eps)


class TestGaussianRP:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 200)).astype(np.float32)
        rp = GaussianRandomProjection(n_components=64, random_state=0).fit(X)
        Z = rp.transform(X)
        assert Z.shape == (500, 64)
        assert rp.components_.shape == (64, 200)
        assert rp.n_components_ == 64

    def test_jl_preserves_distances(self):
        """Pairwise distances should be approximately preserved (Johnson-Lindenstrauss)."""
        rng = np.random.default_rng(1)
        n, d = 150, 500
        X = rng.standard_normal((n, d)).astype(np.float32)
        # k from JL dim with eps=0.3 for tolerance on a modest n
        k = johnson_lindenstrauss_min_dim(n, eps=0.3)
        rp = GaussianRandomProjection(n_components=k, random_state=0).fit(X)
        Z = rp.transform(X)

        def pdist_sq(A):
            s = ((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=-1)
            return s
        D_orig = pdist_sq(X) + 1e-10
        D_proj = pdist_sq(Z)
        ratios = D_proj / D_orig
        # Strip diagonal (0/0)
        off_diag = ~np.eye(n, dtype=bool)
        ratios = ratios[off_diag]
        # Mean should be ≈ 1
        assert abs(float(ratios.mean()) - 1.0) < 0.1
        # Fraction of pairs with distortion within eps=0.3
        within = ((ratios > 0.7) & (ratios < 1.3)).mean()
        assert within > 0.9, f"only {within:.1%} of pairs within (1 ± 0.3)"

    def test_auto_n_components(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5000, 800)).astype(np.float32)
        rp = GaussianRandomProjection(n_components="auto", eps=0.2, random_state=0).fit(X)
        assert rp.n_components_ == johnson_lindenstrauss_min_dim(5000, 0.2)

    def test_sklearn_parity_shape(self):
        sk = pytest.importorskip("sklearn.random_projection")
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 400)).astype(np.float32)
        ours = GaussianRandomProjection(n_components=50, random_state=0).fit(X)
        theirs = sk.GaussianRandomProjection(n_components=50, random_state=0).fit(X)
        assert ours.components_.shape == theirs.components_.shape


class TestSparseRP:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 500)).astype(np.float32)
        rp = SparseRandomProjection(n_components=40, density=0.1, random_state=0).fit(X)
        Z = rp.transform(X)
        assert Z.shape == (200, 40)
        assert rp.density_ == 0.1

    def test_density_auto_is_one_over_sqrt_d(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 400)).astype(np.float32)
        rp = SparseRandomProjection(n_components=32, density="auto", random_state=0).fit(X)
        assert abs(rp.density_ - 1.0 / np.sqrt(400)) < 1e-10

    def test_values_are_ternary(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((50, 100)).astype(np.float32)
        rp = SparseRandomProjection(n_components=20, density=0.5, random_state=0).fit(X)
        # components_ contains {-s, 0, +s}
        uniq = np.unique(rp.components_)
        assert len(uniq) <= 3
        # Non-zero entries have |value| = sqrt(1 / (density * k))
        nonzero = rp.components_[rp.components_ != 0]
        import math
        expected = math.sqrt(1.0 / (0.5 * 20))
        np.testing.assert_allclose(np.abs(nonzero), expected, atol=1e-5)

    def test_sparse_path_matches_dense_path(self):
        """Enable the CSR path and verify the result equals the dense path."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 2500)).astype(np.float32)
        rp_dense = SparseRandomProjection(
            n_components=40, density=0.05, store_sparse=False, random_state=0,
        ).fit(X)
        rp_sparse = SparseRandomProjection(
            n_components=40, density=0.05, store_sparse=True, random_state=0,
        ).fit(X)
        Z_dense = rp_dense.transform(X)
        Z_sparse = rp_sparse.transform(X)
        np.testing.assert_allclose(Z_sparse, Z_dense, atol=1e-4)

    def test_jl_preserves_distances_sparse(self):
        rng = np.random.default_rng(8)
        n, d = 150, 500
        X = rng.standard_normal((n, d)).astype(np.float32)
        k = johnson_lindenstrauss_min_dim(n, eps=0.3)
        rp = SparseRandomProjection(n_components=k, density="auto", random_state=0).fit(X)
        Z = rp.transform(X)
        D_orig = ((X[:, None] - X[None]) ** 2).sum(axis=-1) + 1e-10
        D_proj = ((Z[:, None] - Z[None]) ** 2).sum(axis=-1)
        off_diag = ~np.eye(n, dtype=bool)
        ratios = (D_proj / D_orig)[off_diag]
        within = ((ratios > 0.7) & (ratios < 1.3)).mean()
        assert within > 0.9


class TestCSRMatmul:
    def test_csr_matmul_matches_dense(self):
        rng = np.random.default_rng(10)
        A = rng.standard_normal((300, 500)).astype(np.float32)
        # Mask to ~5% density
        A *= (rng.random(A.shape) < 0.05)
        B = rng.standard_normal((500, 64)).astype(np.float32)
        C_dense = A @ B

        indptr, indices, values = csr_from_dense(A)
        C_sparse = csr_matmul(indptr, indices, values, B, M=A.shape[0])
        np.testing.assert_allclose(np.array(C_sparse), C_dense, atol=1e-4)

    def test_csr_matmul_handles_empty_rows(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((50, 100)).astype(np.float32)
        A[5:15, :] = 0.0                                     # force empty rows
        B = rng.standard_normal((100, 8)).astype(np.float32)
        indptr, indices, values = csr_from_dense(A)
        C = np.array(csr_matmul(indptr, indices, values, B, M=50))
        np.testing.assert_allclose(C[5:15], 0.0, atol=1e-6)

    def test_csr_from_dense_roundtrip(self):
        rng = np.random.default_rng(12)
        A = rng.standard_normal((20, 40)).astype(np.float32)
        A *= (rng.random(A.shape) < 0.3)
        indptr, indices, values = csr_from_dense(A)
        # Rebuild dense from CSR and compare
        A_rebuilt = np.zeros_like(A)
        for i in range(20):
            for p in range(indptr[i], indptr[i + 1]):
                A_rebuilt[i, indices[p]] = values[p]
        np.testing.assert_array_equal(A_rebuilt, A)
