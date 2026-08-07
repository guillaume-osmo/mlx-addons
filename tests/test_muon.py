"""Tests for mlx_addons.optimizers.Muon — SYRK-accelerated Newton-Schulz."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx.utils
import numpy as np
import pytest

from mlx_addons.optimizers import Muon, zeropower_via_newtonschulz5


def _rand(shape, seed=0, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _rel_err(got, ref):
    got = np.array(got, dtype=np.float64)
    ref = np.array(ref, dtype=np.float64)
    return np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-30)


def _mlx_ns5(X, steps=5):
    """Reference: the stock mlx.optimizers.Muon inner loop."""
    return mlx.optimizers.Muon(learning_rate=0.0)._zeropower_via_newtonschulz5(X, steps)


# ============================================================
# orthogonalization
# ============================================================


class TestNewtonSchulz:
    @pytest.mark.parametrize("shape", [(64, 64), (128, 32), (32, 128), (257, 129)])
    @pytest.mark.parametrize("steps", [1, 3, 5])
    def test_matches_mlx_below_thresholds(self, shape, steps):
        """Small shapes take the fallback path — must be bit-identical to MLX."""
        X = _rand(shape, seed=1)
        got, ref = zeropower_via_newtonschulz5(X, steps), _mlx_ns5(X, steps)
        mx.eval(got, ref)
        assert np.array_equal(np.array(got), np.array(ref))

    @pytest.mark.parametrize("shape", [(2048, 2048), (1024, 4096), (4096, 1024)])
    def test_matches_mlx_on_syrk_path(self, shape):
        """Large shapes take the SYRK path — equal up to summation order."""
        X = _rand(shape, seed=2)
        got, ref = zeropower_via_newtonschulz5(X), _mlx_ns5(X)
        mx.eval(got, ref)
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < 1e-4

    @pytest.mark.parametrize("shape", [(64, 64), (512, 2048), (2048, 512)])
    def test_output_is_approximately_orthogonal(self, shape):
        """Singular values are pushed toward 1 (the point of the iteration).

        Note this holds for well-conditioned inputs only. A square Gaussian is
        near-singular, and 5 quintic steps cannot lift its smallest singular
        value off the floor — MLX's own implementation behaves identically
        (both return ~0.002 for a 2048x2048 Gaussian), which is what
        ``test_matches_mlx_on_syrk_path`` pins down.
        """
        X = _rand(shape, seed=3)
        Y = zeropower_via_newtonschulz5(X, steps=5)
        mx.eval(Y)
        s = np.linalg.svd(np.array(Y, dtype=np.float64), compute_uv=False)
        assert 0.5 < s.min() and s.max() < 1.5, f"singular values in [{s.min()}, {s.max()}]"

    @pytest.mark.parametrize("shape", [(2048, 2048), (512, 2048)])
    def test_spectrum_matches_mlx(self, shape):
        """Same singular values as MLX, including the ill-conditioned case."""
        X = _rand(shape, seed=3)
        ours, ref = zeropower_via_newtonschulz5(X), _mlx_ns5(X)
        mx.eval(ours, ref)
        s_a = np.linalg.svd(np.array(ours, dtype=np.float64), compute_uv=False)
        s_b = np.linalg.svd(np.array(ref, dtype=np.float64), compute_uv=False)
        assert np.max(np.abs(s_a - s_b)) < 1e-3

    def test_transposed_shapes_agree(self):
        """Orthogonalizing X.T gives the transpose of orthogonalizing X."""
        X = _rand((128, 64), seed=4)
        a = zeropower_via_newtonschulz5(X)
        b = zeropower_via_newtonschulz5(X.T)
        mx.eval(a, b)
        assert _rel_err(a, b.T) < 1e-5

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
    def test_dtype_preserved(self, dtype):
        Y = zeropower_via_newtonschulz5(_rand((256, 128), seed=5, dtype=dtype))
        mx.eval(Y)
        assert Y.dtype == dtype

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            zeropower_via_newtonschulz5(mx.zeros((4, 4, 4)))


# ============================================================
# optimizer drop-in behaviour
# ============================================================


class TestMuonDropIn:
    def _model(self, seed=0):
        mx.random.seed(seed)
        return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8))

    def _grads(self, model, seed=7):
        X, y = _rand((32, 64), seed=seed), _rand((32, 8), seed=seed + 1)
        loss_fn = lambda m: mx.mean((m(X) - y) ** 2)  # noqa: E731
        return nn.value_and_grad(model, loss_fn)(model)[1]

    def test_is_subclass(self):
        assert issubclass(Muon, mlx.optimizers.Muon)

    def test_matches_mlx_muon_step_for_step(self):
        """Same params + same grads => identical updates (fallback path)."""
        ours, theirs = self._model(), self._model()
        opt_a = Muon(learning_rate=0.02, momentum=0.95)
        opt_b = mlx.optimizers.Muon(learning_rate=0.02, momentum=0.95)
        for _ in range(3):
            opt_a.update(ours, self._grads(ours))
            opt_b.update(theirs, self._grads(theirs))
            mx.eval(ours.parameters(), theirs.parameters())
        flat_a = dict(mlx.utils.tree_flatten(ours.parameters()))
        flat_b = dict(mlx.utils.tree_flatten(theirs.parameters()))
        assert flat_a.keys() == flat_b.keys()
        for key in flat_a:
            assert np.array_equal(np.array(flat_a[key]), np.array(flat_b[key])), key

    def test_training_reduces_loss(self):
        model = self._model()
        X, y = _rand((64, 64), seed=11), _rand((64, 8), seed=12)
        loss_fn = lambda m: mx.mean((m(X) - y) ** 2)  # noqa: E731
        opt = Muon(learning_rate=0.02, momentum=0.95)
        first = float(loss_fn(model))
        for _ in range(20):
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        assert float(loss_fn(model)) < first

    def test_handles_1d_and_4d_params(self):
        """Biases (1D) bypass orthogonalization; conv filters (4D) are reshaped."""
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(8, 4))
        opt = Muon(learning_rate=0.01)
        X = _rand((2, 8, 8, 3), seed=13)
        loss_fn = lambda m: mx.sum(m(X) ** 2)  # noqa: E731
        _, grads = nn.value_and_grad(model, loss_fn)(model)
        opt.update(model, grads)
        mx.eval(model.parameters())
        for p in mlx.utils.tree_flatten(model.parameters()):
            assert not mx.any(mx.isnan(p[1])).item()
