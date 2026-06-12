"""Tests for mlx_addons.solvers — Pulay DIIS and the commutator residual."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.solvers import pulay_diis, commutator_error


# ============================================================
# commutator_error
# ============================================================


class TestCommutatorError:
    def test_zero_when_F_and_P_commute(self):
        # Diagonal F and P commute.
        F = np.diag([1.0, 2.0, 3.0]).astype(np.float32)
        P = np.diag([0.1, 0.5, 0.9]).astype(np.float32)
        e = commutator_error(mx.array(F), mx.array(P))
        mx.eval(e)
        assert np.allclose(np.array(e), 0.0, atol=1e-6)

    def test_nonzero_for_general_pair(self):
        rng = np.random.default_rng(0)
        F = rng.standard_normal((5, 5)).astype(np.float32)
        F = (F + F.T) / 2
        P = rng.standard_normal((5, 5)).astype(np.float32)
        P = (P + P.T) / 2
        e = commutator_error(mx.array(F), mx.array(P))
        mx.eval(e)
        # Verify it equals F @ P - P @ F numerically.
        expected = F @ P - P @ F
        np.testing.assert_allclose(np.array(e), expected, atol=1e-5)

    def test_batched(self):
        rng = np.random.default_rng(1)
        F = rng.standard_normal((4, 6, 6)).astype(np.float32)
        F = (F + np.swapaxes(F, -2, -1)) / 2
        P = rng.standard_normal((4, 6, 6)).astype(np.float32)
        P = (P + np.swapaxes(P, -2, -1)) / 2
        e = commutator_error(mx.array(F), mx.array(P))
        mx.eval(e)
        assert e.shape == (4, 6, 6)


# ============================================================
# pulay_diis
# ============================================================


class TestPulayDIIS:
    def test_returns_latest_when_history_too_short(self):
        rng = np.random.default_rng(0)
        F0 = mx.array(rng.standard_normal((5, 5)).astype(np.float32))
        e0 = mx.array(rng.standard_normal((5, 5)).astype(np.float32))
        F_extrap = pulay_diis([F0], [e0])
        mx.eval(F_extrap)
        np.testing.assert_allclose(np.array(F_extrap), np.array(F0))

    def test_extrapolation_shape(self):
        rng = np.random.default_rng(1)
        F_hist = [mx.array(rng.standard_normal((8, 8)).astype(np.float32)) for _ in range(4)]
        e_hist = [mx.array(rng.standard_normal((8, 8)).astype(np.float32)) for _ in range(4)]
        F_extrap = pulay_diis(F_hist, e_hist)
        mx.eval(F_extrap)
        assert F_extrap.shape == (8, 8)

    def test_max_history_truncation(self):
        rng = np.random.default_rng(2)
        # 10 vectors but max_history=3.
        F_hist = [mx.array(rng.standard_normal((6, 6)).astype(np.float32)) for _ in range(10)]
        e_hist = [mx.array(rng.standard_normal((6, 6)).astype(np.float32)) for _ in range(10)]
        F_extrap = pulay_diis(F_hist, e_hist, max_history=3)
        mx.eval(F_extrap)
        assert F_extrap.shape == (6, 6)

    def test_batched_extrapolation(self):
        rng = np.random.default_rng(3)
        F_hist = [mx.array(rng.standard_normal((4, 5, 5)).astype(np.float32)) for _ in range(3)]
        e_hist = [mx.array(rng.standard_normal((4, 5, 5)).astype(np.float32)) for _ in range(3)]
        F_extrap = pulay_diis(F_hist, e_hist)
        mx.eval(F_extrap)
        assert F_extrap.shape == (4, 5, 5)

    def test_accelerates_synthetic_fixed_point(self):
        """Apply DIIS to a contractive linear map and verify it accelerates
        convergence vs plain Picard iteration.

        Solve x = M x + b with M.shape (8, 8), spectral radius 0.85.
        Plain Picard converges in O(log(eps) / log(0.85)) iterations.
        DIIS should beat that.
        """
        rng = np.random.default_rng(42)
        # Build M with spectral radius 0.85.
        Q = np.linalg.qr(rng.standard_normal((8, 8)))[0]
        eigvals = rng.uniform(-0.85, 0.85, size=8)
        eigvals[0] = 0.85  # ensure radius
        M = Q @ np.diag(eigvals) @ Q.T
        b = rng.standard_normal(8)
        x_star = np.linalg.solve(np.eye(8) - M, b)
        # Reshape into (8, 1) "F-like" matrices for DIIS interface.
        # Treat each iterate x_n as F_n (ignoring physics meaning); residual
        # e_n = x_n - x_{n-1} (shape match).
        x = np.zeros(8)
        F_hist = []
        e_hist = []
        n_picard_iters = 0
        # First a few Picard steps to build history.
        for _ in range(3):
            x_new = M @ x + b
            F_hist.append(mx.array(x_new.reshape(8, 1).astype(np.float32)))
            e_hist.append(mx.array((x_new - x).reshape(8, 1).astype(np.float32)))
            x = x_new
            n_picard_iters += 1
        # Then DIIS-extrapolated steps.
        for _ in range(20):
            F_extrap = pulay_diis(F_hist, e_hist, max_history=6)
            mx.eval(F_extrap)
            x = np.array(F_extrap).reshape(8)
            x_new = M @ x + b
            F_hist.append(mx.array(x_new.reshape(8, 1).astype(np.float32)))
            e_hist.append(mx.array((x_new - x).reshape(8, 1).astype(np.float32)))
            x = x_new
            err = np.linalg.norm(x - x_star)
            if err < 1e-6:
                break
        assert np.linalg.norm(x - x_star) < 1e-6, (
            f"DIIS-accelerated Picard didn't converge: ||x - x*|| = "
            f"{np.linalg.norm(x - x_star)}"
        )
