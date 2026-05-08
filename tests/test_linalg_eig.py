"""Tests for mlx_addons.linalg eigensolver helpers."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import gershgorin_bounds, batched_eigh


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
