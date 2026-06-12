"""Tests for mlx_addons.linalg eigensolver helpers."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import (
    gershgorin_bounds,
    batched_eigh,
    jacobi_eigh,
    gen_eigh,
    JACOBI_MAX_N,
)


def _make_symmetric(*shape, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(shape).astype(np.float32)
    return (A + np.swapaxes(A, -2, -1)) / 2


# ============================================================
# gershgorin_bounds
# ============================================================


class TestGershgorinBounds:
    @pytest.mark.parametrize("N", [3, 5, 10, 20, 50])
    def test_brackets_actual_spectrum(self, N):
        A = _make_symmetric(N, N, seed=N)
        actual = np.linalg.eigvalsh(A)
        lo, hi = gershgorin_bounds(mx.array(A))
        mx.eval(lo, hi)
        assert float(lo) <= actual.min() + 1e-5
        assert float(hi) >= actual.max() - 1e-5

    @pytest.mark.parametrize("batch", [1, 3, 8])
    def test_batched(self, batch):
        A = _make_symmetric(batch, 12, 12, seed=batch)
        lo, hi = gershgorin_bounds(mx.array(A))
        mx.eval(lo, hi)
        assert lo.shape == (batch,)
        assert hi.shape == (batch,)
        for i in range(batch):
            actual = np.linalg.eigvalsh(A[i])
            assert float(lo[i]) <= actual.min() + 1e-5
            assert float(hi[i]) >= actual.max() - 1e-5

    def test_diagonal_matrix_is_tight(self):
        # Diagonal matrices have R_i = 0 so Gershgorin is exact.
        diag = np.array([1.0, 3.0, -2.0, 5.0], dtype=np.float32)
        A = np.diag(diag)
        lo, hi = gershgorin_bounds(mx.array(A))
        mx.eval(lo, hi)
        assert float(lo) == pytest.approx(diag.min(), abs=1e-6)
        assert float(hi) == pytest.approx(diag.max(), abs=1e-6)


# ============================================================
# batched_eigh
# ============================================================


class TestBatchedEigh:
    @pytest.mark.parametrize("N", [3, 5, 10, 20])
    def test_parity_vs_numpy(self, N):
        A = _make_symmetric(N, N, seed=N)
        w, v = batched_eigh(mx.array(A))
        mx.eval(w, v)
        w_ref = np.linalg.eigvalsh(A)
        np.testing.assert_allclose(np.sort(np.array(w)), np.sort(w_ref), rtol=1e-4, atol=1e-4)

    def test_batched_shape(self):
        A = _make_symmetric(4, 8, 8, seed=1)
        w, v = batched_eigh(mx.array(A))
        mx.eval(w, v)
        assert w.shape == (4, 8)
        assert v.shape == (4, 8, 8)

    def test_eigenvectors_orthogonal(self):
        A = _make_symmetric(15, 15, seed=2)
        w, v = batched_eigh(mx.array(A))
        mx.eval(w, v)
        v_np = np.array(v)
        # V^T V should be identity.
        VtV = v_np.T @ v_np
        np.testing.assert_allclose(VtV, np.eye(15), atol=1e-4)

    def test_reconstruct_matrix(self):
        A = _make_symmetric(10, 10, seed=3)
        w, v = batched_eigh(mx.array(A))
        mx.eval(w, v)
        v_np = np.array(v)
        w_np = np.array(w)
        A_reco = v_np @ np.diag(w_np) @ v_np.T
        np.testing.assert_allclose(A_reco, A, atol=1e-4)


# ============================================================
# jacobi_eigh — batched cyclic Jacobi on Metal GPU
# ============================================================


class TestJacobiEigh:
    @pytest.mark.parametrize("N", [3, 5, 10, 16, 24, 32])
    def test_eigenvalues_match_numpy(self, N):
        rng = np.random.default_rng(N)
        A = rng.standard_normal((4, N, N)).astype(np.float32)
        A = (A + np.swapaxes(A, -2, -1)) / 2
        w, v = jacobi_eigh(mx.array(A))
        mx.eval(w, v)
        for i in range(4):
            w_ref = np.sort(np.linalg.eigvalsh(A[i]))
            np.testing.assert_allclose(np.sort(np.array(w[i])), w_ref, atol=1e-3)

    def test_ascending(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 12, 12)).astype(np.float32)
        A = (A + np.swapaxes(A, -2, -1)) / 2
        w, _ = jacobi_eigh(mx.array(A))
        mx.eval(w)
        w_np = np.array(w)
        for i in range(3):
            assert np.all(np.diff(w_np[i]) >= -1e-5), f"batch {i} not ascending"

    def test_eigenvectors_orthonormal(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((2, 16, 16)).astype(np.float32)
        A = (A + np.swapaxes(A, -2, -1)) / 2
        _, v = jacobi_eigh(mx.array(A))
        mx.eval(v)
        v_np = np.array(v)
        for i in range(2):
            VtV = v_np[i].T @ v_np[i]
            np.testing.assert_allclose(VtV, np.eye(16), atol=1e-3)

    def test_eigenvector_residual(self):
        # A V == V diag(w)
        rng = np.random.default_rng(3)
        A = rng.standard_normal((3, 20, 20)).astype(np.float32)
        A = (A + np.swapaxes(A, -2, -1)) / 2
        w, v = jacobi_eigh(mx.array(A))
        mx.eval(w, v)
        for i in range(3):
            res = A[i] @ np.array(v[i]) - np.array(v[i]) @ np.diag(np.array(w[i]))
            assert np.linalg.norm(res) < 1e-3

    def test_rejects_too_large_N(self):
        A = mx.zeros((1, JACOBI_MAX_N + 1, JACOBI_MAX_N + 1))
        with pytest.raises(ValueError, match=f"N <= {JACOBI_MAX_N}"):
            jacobi_eigh(A)

    def test_rejects_non_3d(self):
        A = mx.zeros((10, 10))
        with pytest.raises(ValueError, match="expects \\(B, N, N\\)"):
            jacobi_eigh(A)


# ============================================================
# gen_eigh — generalized symmetric eigenproblem F C = S C diag(w)
# ============================================================


def _make_spd(N, seed=0, batch=None):
    rng = np.random.default_rng(seed)
    shape = (batch, N, N) if batch is not None else (N, N)
    A = rng.standard_normal(shape).astype(np.float32)
    if batch is not None:
        S = np.einsum("bij,bkj->bik", A, A) + 2 * np.eye(N)
    else:
        S = A @ A.T + 2 * np.eye(N)
    return S.astype(np.float32)


def _make_sym(N, seed=0, batch=None):
    rng = np.random.default_rng(seed)
    shape = (batch, N, N) if batch is not None else (N, N)
    A = rng.standard_normal(shape).astype(np.float32)
    return ((A + np.swapaxes(A, -2, -1)) / 2).astype(np.float32)


class TestGenEigh:
    def test_reduces_to_standard_when_S_is_identity(self):
        F_np = _make_sym(15, seed=0)
        I = np.eye(15, dtype=np.float32)
        w, _ = gen_eigh(mx.array(F_np), mx.array(I))
        mx.eval(w)
        w_ref = np.linalg.eigvalsh(F_np)
        np.testing.assert_allclose(np.sort(np.array(w)), np.sort(w_ref), atol=1e-3)

    def test_residual_F_C_equals_S_C_diag_w(self):
        F_np = _make_sym(18, seed=1)
        S_np = _make_spd(18, seed=2)
        w, C = gen_eigh(mx.array(F_np), mx.array(S_np))
        mx.eval(w, C)
        C_np = np.array(C); w_np = np.array(w)
        residual = F_np @ C_np - S_np @ C_np @ np.diag(w_np)
        assert np.linalg.norm(residual) < 1e-3

    def test_S_orthonormal_eigenvectors(self):
        F_np = _make_sym(15, seed=3)
        S_np = _make_spd(15, seed=4)
        _, C = gen_eigh(mx.array(F_np), mx.array(S_np))
        mx.eval(C)
        C_np = np.array(C)
        ortho = C_np.T @ S_np @ C_np
        np.testing.assert_allclose(ortho, np.eye(15), atol=1e-3)

    @pytest.mark.parametrize("N", [3, 8, 16, 24, 32])
    def test_eigenvalues_vs_scipy(self, N):
        scipy = pytest.importorskip("scipy")
        from scipy.linalg import eigh as scipy_eigh
        F_np = _make_sym(N, seed=N)
        S_np = _make_spd(N, seed=N + 100)
        w, _ = gen_eigh(mx.array(F_np), mx.array(S_np))
        mx.eval(w)
        w_ref, _ = scipy_eigh(F_np.astype(np.float64), S_np.astype(np.float64))
        np.testing.assert_allclose(np.sort(np.array(w)), np.sort(w_ref), atol=1e-3)

    def test_batched(self):
        scipy = pytest.importorskip("scipy")
        from scipy.linalg import eigh as scipy_eigh
        B, N = 5, 12
        F_np = _make_sym(N, seed=10, batch=B)
        S_np = _make_spd(N, seed=11, batch=B)
        w, C = gen_eigh(mx.array(F_np), mx.array(S_np))
        mx.eval(w, C)
        for i in range(B):
            w_ref, _ = scipy_eigh(F_np[i].astype(np.float64), S_np[i].astype(np.float64))
            np.testing.assert_allclose(np.sort(np.array(w[i])), np.sort(w_ref), atol=1e-3)
            res = F_np[i] @ np.array(C[i]) - S_np[i] @ np.array(C[i]) @ np.diag(np.array(w[i]))
            assert np.linalg.norm(res) < 1e-3
