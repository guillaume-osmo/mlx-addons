"""Tests for mlx_addons.linalg GPU Metal kernels."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import solve, cholesky, tril_solve, triu_solve
from mlx_addons.linalg import det, slogdet, logdet_spd


def _make_spd(batch, k, seed=42):
    """Create well-conditioned random SPD matrices."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((batch, k, k)).astype(np.float64)
    A = A @ np.transpose(A, (0, 2, 1)) + np.eye(k, dtype=np.float64) * 2.0
    return A


def _make_rhs(batch, k, m=1, seed=43):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, k, m)).astype(np.float64)


# ============================================================
# solve
# ============================================================


class TestSolve:
    @pytest.mark.parametrize("k", [3, 5, 8, 10, 15, 20, 30])
    def test_small_accuracy(self, k):
        """Metal kernel path (k <= 32): check residual ||Ax-b||/||b||."""
        A_np = _make_spd(100, k)
        b_np = _make_rhs(100, k)
        x = solve(mx.array(A_np.astype(np.float32)), mx.array(b_np.astype(np.float32)))
        mx.eval(x)
        x_np = np.array(x).astype(np.float64)
        residual = np.max(np.abs(A_np @ x_np - b_np)) / np.max(np.abs(b_np))
        assert residual < 1e-3, f"k={k}: residual {residual:.2e}"

    @pytest.mark.parametrize("k", [48, 64, 96, 128])
    def test_large_accuracy(self, k):
        """Blocked path (k > 32): check residual."""
        A_np = _make_spd(20, k)
        b_np = _make_rhs(20, k)
        x = solve(mx.array(A_np.astype(np.float32)), mx.array(b_np.astype(np.float32)))
        mx.eval(x)
        x_np = np.array(x).astype(np.float64)
        residual = np.max(np.abs(A_np @ x_np - b_np)) / np.max(np.abs(b_np))
        assert residual < 1e-2, f"k={k}: residual {residual:.2e}"

    def test_multi_rhs(self):
        A_np = _make_spd(50, 8)
        b_np = _make_rhs(50, 8, m=4)
        x = solve(mx.array(A_np.astype(np.float32)), mx.array(b_np.astype(np.float32)))
        mx.eval(x)
        assert x.shape == (50, 8, 4)
        x_np = np.array(x).astype(np.float64)
        residual = np.max(np.abs(A_np @ x_np - b_np)) / np.max(np.abs(b_np))
        assert residual < 1e-3


# ============================================================
# cholesky
# ============================================================


class TestCholesky:
    @pytest.mark.parametrize("k", [3, 5, 10, 20, 30])
    def test_small_reconstruction(self, k):
        """Metal path: ||A - L L^T|| / ||A||."""
        A_np = _make_spd(50, k)
        L = cholesky(mx.array(A_np.astype(np.float32)))
        mx.eval(L)
        L_np = np.array(L).astype(np.float64)
        rel_err = np.max(np.abs(A_np - L_np @ np.transpose(L_np, (0, 2, 1)))) / np.max(np.abs(A_np))
        assert rel_err < 1e-4, f"k={k}: rel_recon {rel_err:.2e}"

    @pytest.mark.parametrize("k", [48, 64, 128])
    def test_large_reconstruction(self, k):
        """Blocked path: ||A - L L^T|| / ||A||."""
        A_np = _make_spd(10, k)
        L = cholesky(mx.array(A_np.astype(np.float32)))
        mx.eval(L)
        L_np = np.array(L).astype(np.float64)
        rel_err = np.max(np.abs(A_np - L_np @ np.transpose(L_np, (0, 2, 1)))) / np.max(np.abs(A_np))
        assert rel_err < 1e-3, f"k={k}: rel_recon {rel_err:.2e}"

    def test_lower_triangular(self):
        A_np = _make_spd(10, 6)
        L = cholesky(mx.array(A_np.astype(np.float32)))
        mx.eval(L)
        L_np = np.array(L)
        for i in range(6):
            for j in range(i + 1, 6):
                assert np.all(np.abs(L_np[:, i, j]) < 1e-6)


# ============================================================
# triangular solve
# ============================================================


class TestTriangularSolve:
    def test_tril(self):
        A_np = _make_spd(30, 6)
        A = mx.array(A_np.astype(np.float32))
        L = cholesky(A)
        b = mx.array(_make_rhs(30, 6).astype(np.float32))
        x = tril_solve(L, b)
        mx.eval(x)
        residual = L @ x - b
        mx.eval(residual)
        assert np.max(np.abs(np.array(residual))) < 1e-3

    def test_triu(self):
        A_np = _make_spd(30, 6)
        A = mx.array(A_np.astype(np.float32))
        L = cholesky(A)
        b = mx.array(_make_rhs(30, 6).astype(np.float32))
        x = triu_solve(L, b)
        mx.eval(x)
        residual = L.swapaxes(-2, -1) @ x - b
        mx.eval(residual)
        assert np.max(np.abs(np.array(residual))) < 1e-3


# ============================================================
# det / slogdet / logdet_spd
# ============================================================


class TestDet:
    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_det_accuracy(self, n):
        A_np = _make_spd(10, n)
        det_ref = np.linalg.det(A_np)
        d = det(mx.array(A_np.astype(np.float32)))
        mx.eval(d)
        d_np = np.array(d).astype(np.float64)
        # Compare in log-space to handle large values
        log_err = np.max(np.abs(np.log(np.abs(d_np)) - np.log(np.abs(det_ref))))
        assert log_err < 0.01, f"n={n}: log|det| error {log_err:.2e}"

    @pytest.mark.parametrize("n", [3, 5, 10, 20, 50])
    def test_slogdet_accuracy(self, n):
        A_np = _make_spd(10, n)
        sign_ref, logdet_ref = np.linalg.slogdet(A_np)
        s, ld = slogdet(mx.array(A_np.astype(np.float32)))
        mx.eval(s, ld)
        s_np = np.array(s).astype(np.float64)
        ld_np = np.array(ld).astype(np.float64)
        assert np.all(s_np == sign_ref), f"n={n}: sign mismatch"
        err = np.max(np.abs(ld_np - logdet_ref))
        assert err < 0.1, f"n={n}: logdet error {err:.2e}"

    @pytest.mark.parametrize("n", [3, 5, 10, 20, 64, 128])
    def test_logdet_spd_accuracy(self, n):
        A_np = _make_spd(10, n)
        _, logdet_ref = np.linalg.slogdet(A_np)
        ld = logdet_spd(mx.array(A_np.astype(np.float32)))
        mx.eval(ld)
        ld_np = np.array(ld).astype(np.float64)
        err = np.max(np.abs(ld_np - logdet_ref))
        assert err < 0.1, f"n={n}: logdet_spd error {err:.2e}"
