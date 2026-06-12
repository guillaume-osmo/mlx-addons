"""Tests for mlx_addons.linalg density-matrix purification primitives."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import (
    batched_eigh,
    mcweeny_purify,
    sp2_purify,
)


def _make_fock_like(N, seed=0):
    """Random symmetric matrix with N(0,1) entries — Fock-like proxy."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)).astype(np.float32)
    return (A + A.T) / 2


def _make_structured_fock(N, n_occ, gap=2.0, seed=0):
    """Symmetric matrix with prescribed spectrum: n_occ negative + (N-n_occ)
    positive eigenvalues, separated by a gap. More realistic Fock proxy.
    """
    rng = np.random.default_rng(seed)
    eigvals = np.concatenate([
        np.linspace(-3.0, -gap / 2, n_occ),
        np.linspace(gap / 2, 5.0, N - n_occ),
    ]).astype(np.float32)
    Q = np.linalg.qr(rng.standard_normal((N, N)).astype(np.float32))[0]
    A = Q @ np.diag(eigvals) @ Q.T
    return (A + A.T) / 2  # symmetrize against float roundoff


def _eigh_density(F: np.ndarray, n_occ: int) -> np.ndarray:
    """Reference closed-shell projector P = C_occ @ C_occ.T (factor 2 NOT
    applied; matches purification convention of trace = n_occ).
    """
    _, V = np.linalg.eigh(F)
    return V[:, :n_occ] @ V[:, :n_occ].T


# ============================================================
# sp2_purify
# ============================================================


class TestSP2Purify:
    @pytest.mark.parametrize("N,n_occ", [(2, 1), (5, 2), (10, 3), (10, 7), (20, 5), (50, 10)])
    def test_idempotent_and_trace(self, N, n_occ):
        F = _make_fock_like(N, seed=N * n_occ)
        rho = sp2_purify(mx.array(F), n_occ)
        mx.eval(rho)
        tr = float(mx.trace(rho, axis1=-2, axis2=-1))
        idem_err = float(mx.linalg.norm(rho @ rho - rho))
        assert tr == pytest.approx(n_occ, abs=1e-4), f"trace {tr} != {n_occ}"
        assert idem_err < 1e-5, f"||rho^2 - rho||_F = {idem_err} not idempotent"

    @pytest.mark.parametrize("N,n_occ", [(10, 3), (20, 5), (30, 8)])
    def test_parity_vs_eigh_random(self, N, n_occ):
        F = _make_fock_like(N, seed=N + n_occ)
        rho = sp2_purify(mx.array(F), n_occ)
        mx.eval(rho)
        P_eig = _eigh_density(F, n_occ)
        diff = np.linalg.norm(np.array(rho) - P_eig, "fro")
        assert diff < 1e-4, f"||rho_sp2 - P_eigh||_F = {diff}"

    @pytest.mark.parametrize("N,n_occ", [(20, 5), (40, 12)])
    def test_parity_vs_eigh_structured(self, N, n_occ):
        F = _make_structured_fock(N, n_occ, gap=2.0, seed=N)
        rho = sp2_purify(mx.array(F), n_occ)
        mx.eval(rho)
        P_eig = _eigh_density(F, n_occ)
        diff = np.linalg.norm(np.array(rho) - P_eig, "fro")
        assert diff < 1e-4, f"||rho_sp2 - P_eigh||_F = {diff}"

    def test_batched(self):
        rng = np.random.default_rng(7)
        batch = np.stack([
            (lambda a: (a + a.T) / 2)(rng.standard_normal((20, 20)).astype(np.float32) * (i + 1))
            for i in range(4)
        ])
        F_b = mx.array(batch)
        rho_b = sp2_purify(F_b, 5)
        mx.eval(rho_b)
        trs = mx.trace(rho_b, axis1=-2, axis2=-1)
        for i in range(4):
            assert float(trs[i]) == pytest.approx(5.0, abs=1e-4)
            P_eig = _eigh_density(batch[i], 5)
            diff = np.linalg.norm(np.array(rho_b[i]) - P_eig, "fro")
            assert diff < 1e-4, f"batch[{i}]: diff = {diff}"

    def test_eigenvalue_range(self):
        F = _make_structured_fock(20, 5, seed=0)
        rho = sp2_purify(mx.array(F), 5)
        mx.eval(rho)
        eigs = np.linalg.eigvalsh(np.array(rho))
        # All eigenvalues should be very close to {0, 1}.
        assert eigs.min() > -1e-4
        assert eigs.max() < 1 + 1e-4
        # Exactly n_occ should be near 1.
        n_near_one = int(np.sum(eigs > 0.5))
        assert n_near_one == 5


# ============================================================
# mcweeny_purify (warm-start refinement)
# ============================================================


class TestMcWeenyPurify:
    def test_polishes_sp2_result(self):
        F = _make_structured_fock(20, 5, seed=10)
        rho_sp2 = sp2_purify(mx.array(F), 5)
        rho_mcw = mcweeny_purify(mx.array(F), rho_sp2)
        mx.eval(rho_mcw)
        tr = float(mx.trace(rho_mcw, axis1=-2, axis2=-1))
        idem = float(mx.linalg.norm(rho_mcw @ rho_mcw - rho_mcw))
        assert tr == pytest.approx(5.0, abs=1e-4)
        assert idem < 1e-5

    def test_warm_start_from_perturbed_projector(self):
        F = _make_structured_fock(20, 5, gap=3.0, seed=11)
        P_eig = _eigh_density(F, 5)
        rng = np.random.default_rng(0)
        # Tiny perturbation that preserves the count of eigenvalues > 0.5.
        noise = (rng.standard_normal((20, 20)) * 1e-4).astype(np.float32)
        noise = (noise + noise.T) / 2
        rho_warm = mx.array(P_eig + noise)
        rho_ref = mcweeny_purify(mx.array(F), rho_warm)
        mx.eval(rho_ref)
        tr = float(mx.trace(rho_ref, axis1=-2, axis2=-1))
        idem = float(mx.linalg.norm(rho_ref @ rho_ref - rho_ref))
        assert tr == pytest.approx(5.0, abs=1e-4)
        assert idem < 1e-5

    def test_does_not_degrade_already_idempotent(self):
        F = _make_structured_fock(20, 5, seed=12)
        P_eig = mx.array(_eigh_density(F, 5))
        rho_polish = mcweeny_purify(mx.array(F), P_eig)
        mx.eval(rho_polish)
        idem = float(mx.linalg.norm(rho_polish @ rho_polish - rho_polish))
        assert idem < 1e-5
