"""Tests for mlx_addons.neighbors.KernelDensity (sklearn parity + low-RAM tiling)."""

import numpy as np
import pytest

from mlx_addons.neighbors import KernelDensity, VALID_KERNELS, bootstrap_kde


def _rel_density_diff(mine_log, ref_log, floor=1e-4):
    """Max |density difference| relative to the peak density, over points where the
    reference density is non-negligible.

    This is the metric that matters for a KDE curve (and the one the reference
    notebook uses to validate CPU vs GPU): the ``float32`` Metal path agrees with
    ``float64`` sklearn to ~1e-5 in the density domain. Pure log-density diffs are
    dominated by tail underflow (``exp(-170)`` -> 0 in float32) and are not
    physically meaningful.
    """
    ref, mine = np.exp(ref_log), np.exp(mine_log)
    peak = ref.max()
    mask = ref > floor * peak
    return float(np.max(np.abs(mine[mask] - ref[mask])) / peak)


class TestSklearnParity:
    @pytest.mark.parametrize("kernel", VALID_KERNELS)
    def test_logdensity_matches_sklearn_1d(self, kernel):
        sk = pytest.importorskip("sklearn.neighbors")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1500, 1)).astype(np.float32)
        Q = np.linspace(-3.5, 3.5, 500).reshape(-1, 1).astype(np.float32)
        h = 0.5
        ref = sk.KernelDensity(bandwidth=h, kernel=kernel).fit(X).score_samples(Q)
        mine = KernelDensity(bandwidth=h, kernel=kernel).fit(X).score_samples(Q)
        assert _rel_density_diff(mine, ref) < 1e-4

    @pytest.mark.parametrize("kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear"])
    def test_logdensity_matches_sklearn_2d(self, kernel):
        sk = pytest.importorskip("sklearn.neighbors")
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1200, 2)).astype(np.float32)
        Q = rng.standard_normal((300, 2)).astype(np.float32)
        h = 0.7
        ref = sk.KernelDensity(bandwidth=h, kernel=kernel).fit(X).score_samples(Q)
        mine = KernelDensity(bandwidth=h, kernel=kernel).fit(X).score_samples(Q)
        assert _rel_density_diff(mine, ref) < 1e-4

    @pytest.mark.parametrize("h", [0.05, 0.2, 1.0, 3.0])
    def test_bandwidth_sweep_matches_sklearn(self, h):
        sk = pytest.importorskip("sklearn.neighbors")
        rng = np.random.default_rng(2)
        X = rng.standard_normal((2000, 1)).astype(np.float32)
        Q = np.linspace(-4, 4, 400).reshape(-1, 1).astype(np.float32)
        ref = sk.KernelDensity(bandwidth=h, kernel="gaussian").fit(X).score_samples(Q)
        mine = KernelDensity(bandwidth=h, kernel="gaussian").fit(X).score_samples(Q)
        assert _rel_density_diff(mine, ref) < 1e-4

    def test_sample_weight_matches_sklearn(self):
        sk = pytest.importorskip("sklearn.neighbors")
        rng = np.random.default_rng(3)
        X = rng.standard_normal((800, 1)).astype(np.float32)
        w = rng.uniform(0.1, 2.0, size=800).astype(np.float32)
        Q = np.linspace(-3, 3, 200).reshape(-1, 1).astype(np.float32)
        ref = sk.KernelDensity(bandwidth=0.3).fit(X, sample_weight=w).score_samples(Q)
        mine = KernelDensity(bandwidth=0.3).fit(X, sample_weight=w).score_samples(Q)
        assert _rel_density_diff(mine, ref) < 1e-4


class TestTiling:
    def test_result_independent_of_tile_sizes(self):
        """The whole point of the low-RAM design: tiling must not change the answer."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((6000, 1)).astype(np.float32)
        Q = np.linspace(-3, 3, 3001).reshape(-1, 1).astype(np.float32)
        small = KernelDensity(bandwidth=0.2, query_tile=64, sample_tile=97).fit(X).eval_density(Q)
        big = KernelDensity(bandwidth=0.2, query_tile=8192, sample_tile=99999).fit(X).eval_density(Q)
        assert np.max(np.abs(small - big)) < 1e-6

    def test_eval_density_equals_exp_score(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((1000, 1)).astype(np.float32)
        Q = np.linspace(-3, 3, 500).reshape(-1, 1).astype(np.float32)
        kde = KernelDensity(bandwidth=0.25).fit(X)
        assert np.max(np.abs(kde.eval_density(Q) - np.exp(kde.score_samples(Q)))) < 1e-5


class TestDensityProperties:
    def test_integrates_to_one(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((4000, 1)).astype(np.float32)
        grid = np.linspace(-8, 8, 8000).reshape(-1, 1).astype(np.float32)
        dens = KernelDensity(bandwidth=0.3).fit(X).eval_density(grid)
        integral = np.trapz(dens, grid.ravel())
        assert abs(integral - 1.0) < 1e-2

    def test_1d_and_2d_input_accepted(self):
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal(500).astype(np.float32)         # 1-D input
        kde = KernelDensity(bandwidth=0.4).fit(X1)
        assert kde.n_features_in_ == 1
        assert kde.eval_density(np.linspace(-2, 2, 50)).shape == (50,)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            KernelDensity(bandwidth=-1.0)
        with pytest.raises(ValueError):
            KernelDensity(kernel="not_a_kernel")


class TestBootstrap:
    def test_bootstrap_shapes_and_band_ordering(self):
        rng = np.random.default_rng(8)
        vals = rng.standard_t(df=5, size=50_000).astype(np.float32) * 0.01
        grid = np.linspace(-0.05, 0.05, 400)
        res = bootstrap_kde(vals, grid, n_samples=5000, n_boot=40, bandwidth=0.002, seed=0)
        for key in ("fit_seconds", "n_samples", "n_boot", "mean", "lo", "hi"):
            assert key in res
        assert res["mean"].shape == res["lo"].shape == res["hi"].shape == (400,)
        # band must bracket the mean everywhere
        assert np.all(res["lo"] <= res["mean"] + 1e-6)
        assert np.all(res["hi"] >= res["mean"] - 1e-6)

    def test_bootstrap_is_reproducible(self):
        rng = np.random.default_rng(9)
        vals = rng.standard_normal(20_000).astype(np.float32)
        grid = np.linspace(-3, 3, 200)
        a = bootstrap_kde(vals, grid, n_samples=3000, n_boot=20, bandwidth=0.2, seed=42)
        b = bootstrap_kde(vals, grid, n_samples=3000, n_boot=20, bandwidth=0.2, seed=42)
        np.testing.assert_allclose(a["mean"], b["mean"], rtol=0, atol=0)
