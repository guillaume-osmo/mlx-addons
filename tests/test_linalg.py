"""Tests for mlx_addons.linalg GPU Metal kernels."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import solve, cholesky, tril_solve, triu_solve


def _make_spd(batch, k, seed=42):
    """Create random SPD matrices."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((batch, k, k)).astype(np.float32)
    A = A @ np.transpose(A, (0, 2, 1)) + np.eye(k, dtype=np.float32) * 1.0
    return mx.array(A)


def _make_rhs(batch, k, m=1, seed=42):
    rng = np.random.default_rng(seed + 1)
    return mx.array(rng.standard_normal((batch, k, m)).astype(np.float32))


class TestSolve:
    @pytest.mark.parametrize("k", [3, 5, 10, 15, 20, 30])
    def test_accuracy(self, k):
        A = _make_spd(100, k)
        b = _make_rhs(100, k)
        x = solve(A, b)
        x_ref = mx.linalg.solve(A, b, stream=mx.cpu)
        mx.eval(x, x_ref)
        err = np.max(np.abs(np.array(x) - np.array(x_ref)))
        assert err < 1e-3, f"k={k}: error {err:.2e}"

    def test_multi_rhs(self):
        A = _make_spd(50, 8)
        b = _make_rhs(50, 8, m=4)
        x = solve(A, b)
        mx.eval(x)
        assert x.shape == (50, 8, 4)
        # Verify A @ x ≈ b
        residual = A @ x - b
        mx.eval(residual)
        assert np.max(np.abs(np.array(residual))) < 1e-3

    def test_2d_rhs(self):
        A = _make_spd(20, 5)
        b = mx.array(np.random.randn(20, 5).astype(np.float32))
        x = solve(A, mx.expand_dims(b, -1))
        mx.eval(x)
        assert x.shape == (20, 5, 1)


class TestCholesky:
    @pytest.mark.parametrize("k", [3, 5, 10, 20])
    def test_accuracy(self, k):
        A = _make_spd(100, k)
        L = cholesky(A)
        L_ref = mx.linalg.cholesky(A, stream=mx.cpu)
        mx.eval(L, L_ref)
        err = np.max(np.abs(np.array(L) - np.array(L_ref)))
        assert err < 1e-3, f"k={k}: error {err:.2e}"

    def test_lower_triangular(self):
        A = _make_spd(10, 6)
        L = cholesky(A)
        mx.eval(L)
        L_np = np.array(L)
        # Check upper triangle is zero
        for i in range(6):
            for j in range(i + 1, 6):
                assert np.all(np.abs(L_np[:, i, j]) < 1e-6)

    def test_reconstruction(self):
        A = _make_spd(20, 8)
        L = cholesky(A)
        A_recon = L @ L.swapaxes(-2, -1)
        mx.eval(A_recon)
        # Relative error: |A - L L^T| / |A|
        err = np.max(np.abs(np.array(A) - np.array(A_recon)))
        scale = np.max(np.abs(np.array(A)))
        assert err / scale < 1e-4, f"Relative error {err/scale:.2e}"


class TestTriangularSolve:
    def test_tril(self):
        A = _make_spd(30, 6)
        L = cholesky(A)
        b = _make_rhs(30, 6)
        x = tril_solve(L, b)
        mx.eval(x)
        residual = L @ x - b
        mx.eval(residual)
        assert np.max(np.abs(np.array(residual))) < 1e-3

    def test_triu(self):
        A = _make_spd(30, 6)
        L = cholesky(A)
        b = _make_rhs(30, 6)
        x = triu_solve(L, b)
        mx.eval(x)
        residual = L.swapaxes(-2, -1) @ x - b
        mx.eval(residual)
        assert np.max(np.abs(np.array(residual))) < 1e-3
