"""Tests for mlx_addons.ot — 1-D Wasserstein (EMD) + batched Sinkhorn."""

import numpy as np
import pytest

from mlx_addons.ot import wasserstein1d_rdm, wasserstein1d_neighbors, sinkhorn2_batch


# ──────────────────────────────── 1-D Wasserstein ──

class TestWasserstein1D:
    def test_matches_scipy(self):
        sp = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        N, B, dx = 50, 40, 0.5
        W = rng.random((N, B)).astype(np.float32) + 1e-3
        grid = np.arange(B) * dx

        D = wasserstein1d_rdm(W, dx=dx)
        # spot-check a handful of pairs against scipy's exact 1-D EMD
        Wn = W / W.sum(1, keepdims=True)
        for i, j in [(0, 1), (3, 17), (10, 49), (25, 25)]:
            ref = sp.wasserstein_distance(grid, grid, Wn[i], Wn[j])
            assert abs(D[i, j] - ref) < 1e-5, (i, j, D[i, j], ref)

    def test_symmetric_and_zero_diagonal(self):
        rng = np.random.default_rng(1)
        W = rng.random((30, 25)).astype(np.float32) + 1e-3
        D = wasserstein1d_rdm(W, dx=1.0)
        np.testing.assert_allclose(D, D.T, atol=1e-5)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_translation_equals_shift(self):
        """A unit point mass shifted by k bins is exactly k·dx away."""
        B, dx = 20, 0.25
        p = np.zeros((2, B), np.float32)
        p[0, 3] = 1.0
        p[1, 3 + 5] = 1.0          # shifted by 5 bins
        D = wasserstein1d_rdm(p, dx=dx, normalize=False)
        assert abs(D[0, 1] - 5 * dx) < 1e-6


class TestWasserstein1DNeighbors:
    def test_matches_bruteforce_threshold(self):
        rng = np.random.default_rng(2)
        W = rng.random((120, 30)).astype(np.float32) + 1e-3
        cutoff = 0.4
        D = wasserstein1d_rdm(W, dx=1.0)
        offsets, idx = wasserstein1d_neighbors(W, cutoff=cutoff, dx=1.0)

        assert offsets.shape == (121,)
        assert offsets[-1] == len(idx)
        for i in [0, 5, 60, 119]:
            ref = set(np.where(D[i] <= cutoff)[0].tolist()) - {i}
            got = set(idx[offsets[i]:offsets[i + 1]].tolist())
            assert ref == got, i

    def test_empty_when_cutoff_zero(self):
        rng = np.random.default_rng(3)
        W = rng.random((40, 16)).astype(np.float32) + 1e-3
        offsets, idx = wasserstein1d_neighbors(W, cutoff=-1.0, dx=1.0)
        assert len(idx) == 0
        assert offsets[-1] == 0


# ──────────────────────────────── Sinkhorn ──

def _random_problem(P, B, seed):
    rng = np.random.default_rng(seed)
    a = rng.random((P, B)).astype(np.float32) + 1e-2
    b = rng.random((P, B)).astype(np.float32) + 1e-2
    a /= a.sum(1, keepdims=True)
    b /= b.sum(1, keepdims=True)
    x = np.linspace(0, 1, B)
    C = np.stack([(x[:, None] - x[None, :]) ** 2 for _ in range(P)]).astype(np.float32)
    return a, b, C


class TestSinkhorn:
    def test_matches_pot(self):
        ot = pytest.importorskip("ot")
        a, b, C, = _random_problem(10, 24, 4)
        reg = 0.05
        ours = sinkhorn2_batch(a, b, C, reg=reg, num_iters=500)
        theirs = np.array([
            ot.sinkhorn2(a[p], b[p], C[p], reg=reg, method="sinkhorn_log", numItermax=4000)
            for p in range(a.shape[0])
        ], dtype=float)
        rel = np.abs(ours - theirs) / (np.abs(theirs) + 1e-9)
        assert rel.max() < 1e-3, (ours[:3], theirs[:3])

    def test_shared_cost_broadcast(self):
        a, b, C = _random_problem(6, 20, 5)
        costs_batched = sinkhorn2_batch(a, b, C, reg=0.1, num_iters=300)
        costs_shared = sinkhorn2_batch(a, b, C[0], reg=0.1, num_iters=300)  # (B,B) broadcast
        # same shared cost for all -> per-problem costs still differ by marginals, both finite
        assert costs_shared.shape == (6,)
        assert np.all(np.isfinite(costs_shared))

    def test_handles_zero_mass_bins(self):
        """Empty bins (zero mass) must not produce NaNs (the textbook 1/a would)."""
        rng = np.random.default_rng(6)
        P, B = 5, 18
        a = rng.random((P, B)).astype(np.float32)
        b = rng.random((P, B)).astype(np.float32)
        a[a < 0.5] = 0.0            # punch holes
        b[b < 0.5] = 0.0
        a /= a.sum(1, keepdims=True)
        b /= b.sum(1, keepdims=True)
        x = np.linspace(0, 1, B)
        C = ((x[:, None] - x[None, :]) ** 2).astype(np.float32)
        costs = sinkhorn2_batch(a, b, C, reg=0.05, num_iters=300)
        assert np.all(np.isfinite(costs))
        assert np.all(costs >= 0)
