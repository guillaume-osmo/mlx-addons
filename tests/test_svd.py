"""Tests for mlx_addons.linalg.randomized_svd / TruncatedSVD."""

import numpy as np
import pytest

from mlx_addons.linalg import randomized_svd, TruncatedSVD


def _reconstruction_relerr(X, U, S, Vt):
    """Relative Frobenius error of the rank-k approximation."""
    X_hat = U @ np.diag(S) @ Vt
    return np.linalg.norm(X - X_hat, "fro") / np.linalg.norm(X, "fro")


class TestRandomizedSVD:
    @pytest.mark.parametrize("shape,k,true_rank", [
        ((100, 30), 6, 10),
        ((500, 128), 16, 32),
        ((1000, 64), 8, 16),
        ((2000, 256), 24, 48),
    ])
    def test_low_rank_recovery(self, shape, k, true_rank):
        """Recover a rank-``true_rank`` matrix from its top-``k`` singular triplets.

        The randomized algorithm is accurate for the top ``k`` triplets when the
        true spectrum has ``true_rank > k``; the reconstruction error of the
        top-k approximation is bounded by the (k+1)-th singular value.
        """
        n, m = shape
        rng = np.random.default_rng(0)
        U_true, _ = np.linalg.qr(rng.standard_normal((n, true_rank)).astype(np.float32))
        Vt_true, _ = np.linalg.qr(rng.standard_normal((m, true_rank)).astype(np.float32))
        Vt_true = Vt_true.T
        # Exponential spectrum so top-k dominates
        S_true = (10.0 * np.exp(-0.3 * np.arange(true_rank))).astype(np.float32)
        X = (U_true * S_true) @ Vt_true

        # Benchmark the rank-k reconstruction against the exact SVD's rank-k
        U_exact, S_exact, Vt_exact = np.linalg.svd(X, full_matrices=False)
        X_rank_k_exact = (U_exact[:, :k] * S_exact[:k]) @ Vt_exact[:k]
        exact_tail_err = np.linalg.norm(X - X_rank_k_exact, "fro") / np.linalg.norm(X, "fro")

        U, S, Vt = randomized_svd(X, n_components=k, n_iter=6, random_state=0)
        relerr = _reconstruction_relerr(X, U, S, Vt)

        # Randomized rSVD should approach the exact tail error within a small factor.
        assert relerr < max(exact_tail_err * 1.5, 1e-3), (
            f"shape={shape} k={k}: relerr={relerr:.3e}  exact_tail={exact_tail_err:.3e}"
        )

    @pytest.mark.parametrize("shape,k", [((500, 64), 8), ((1000, 128), 24)])
    def test_approx_matches_full_svd(self, shape, k):
        """Singular values recovered within small tolerance of full SVD."""
        n, m = shape
        rng = np.random.default_rng(1)
        X = rng.standard_normal(shape).astype(np.float32)
        S_true = np.linalg.svd(X, compute_uv=False)[:k]

        _U, S, _Vt = randomized_svd(X, n_components=k, n_iter=6, random_state=0)
        relerr = np.abs(S - S_true) / S_true
        assert relerr.max() < 5e-2, f"max |ΔS|/S = {relerr.max():.3e}"

    def test_sign_canonicalization(self):
        """Repeated runs with same seed produce identical signs on Vt."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 50)).astype(np.float32)
        _, _, Vt_a = randomized_svd(X, n_components=12, random_state=42)
        _, _, Vt_b = randomized_svd(X, n_components=12, random_state=42)
        np.testing.assert_allclose(Vt_a, Vt_b, atol=1e-6)

    def test_orthonormal_U(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 80)).astype(np.float32)
        U, _, _ = randomized_svd(X, n_components=20, random_state=0)
        gram = U.T @ U
        assert np.abs(gram - np.eye(20)).max() < 1e-3

    def test_batched_shape(self):
        rng = np.random.default_rng(31)
        X = rng.standard_normal((5, 300, 60)).astype(np.float32)
        U, S, Vt = randomized_svd(X, n_components=8, random_state=0)
        assert U.shape == (5, 300, 8)
        assert S.shape == (5, 8)
        assert Vt.shape == (5, 8, 60)

    def test_batched_matches_exact_sv(self):
        """Each batch element's top-k singular values match the exact SVD."""
        rng = np.random.default_rng(32)
        batch, n, m, true_rank, k = 4, 300, 60, 20, 12
        X_list, S_exact_list = [], []
        for _ in range(batch):
            U_true, _ = np.linalg.qr(rng.standard_normal((n, true_rank)).astype(np.float32))
            Vt_true, _ = np.linalg.qr(rng.standard_normal((m, true_rank)).astype(np.float32))
            Vt_true = Vt_true.T
            S_true = (10.0 * np.exp(-0.2 * np.arange(true_rank))).astype(np.float32)
            X_list.append((U_true * S_true) @ Vt_true)
            S_exact_list.append(np.linalg.svd(X_list[-1], compute_uv=False)[:k])
        X = np.stack(X_list)
        S_exact = np.stack(S_exact_list)

        _U, S, _Vt = randomized_svd(X, n_components=k, n_iter=6, random_state=0)
        relerr = np.abs(S - S_exact) / S_exact
        assert relerr.max() < 5e-2, f"max rel err = {relerr.max():.3e}"

    def test_batched_reconstruction_per_matrix(self):
        rng = np.random.default_rng(33)
        batch, n, m, true_rank, k = 6, 400, 80, 24, 10
        X_list = []
        for _ in range(batch):
            U_true, _ = np.linalg.qr(rng.standard_normal((n, true_rank)).astype(np.float32))
            Vt_true, _ = np.linalg.qr(rng.standard_normal((m, true_rank)).astype(np.float32))
            Vt_true = Vt_true.T
            S_true = (10.0 * np.exp(-0.3 * np.arange(true_rank))).astype(np.float32)
            X_list.append((U_true * S_true) @ Vt_true)
        X = np.stack(X_list)
        U, S, Vt = randomized_svd(X, n_components=k, n_iter=6, random_state=0)
        for i in range(batch):
            rec = U[i] @ np.diag(S[i]) @ Vt[i]
            relerr = np.linalg.norm(X[i] - rec, "fro") / np.linalg.norm(X[i], "fro")
            assert relerr < 0.1, f"batch {i}: relerr={relerr:.3e}"


class TestTruncatedSVDClass:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((400, 100)).astype(np.float32)
        svd = TruncatedSVD(n_components=16, random_state=0)
        Z = svd.fit_transform(X)
        assert Z.shape == (400, 16)
        assert svd.components_.shape == (16, 100)
        assert svd.singular_values_.shape == (16,)

    def test_transform_matches_projection(self):
        """svd.transform(X) equals X @ components_.T."""
        rng = np.random.default_rng(8)
        X_fit = rng.standard_normal((500, 60)).astype(np.float32)
        X_new = rng.standard_normal((50, 60)).astype(np.float32)
        svd = TruncatedSVD(n_components=10, random_state=0).fit(X_fit)
        np.testing.assert_allclose(
            svd.transform(X_new),
            X_new @ svd.components_.T,
            atol=1e-5,
        )

    def test_sklearn_parity_smoke(self):
        """Compare reconstruction quality against sklearn's TruncatedSVD (randomized).

        We don't require identical outputs (different basis rotations in the
        random projection), but the reconstruction residual should be similar.
        """
        sk = pytest.importorskip("sklearn.decomposition")
        rng = np.random.default_rng(11)
        X = rng.standard_normal((800, 120)).astype(np.float32)
        k = 24

        ours = TruncatedSVD(n_components=k, random_state=0).fit(X)
        theirs = sk.TruncatedSVD(n_components=k, algorithm="randomized", random_state=0, n_iter=4).fit(X)

        # Both should give low residual
        res_ours = np.linalg.norm(X - ours.transform(X) @ ours.components_, "fro") / np.linalg.norm(X, "fro")
        res_theirs = np.linalg.norm(X - theirs.transform(X) @ theirs.components_, "fro") / np.linalg.norm(X, "fro")
        assert abs(res_ours - res_theirs) < 0.02, (res_ours, res_theirs)
