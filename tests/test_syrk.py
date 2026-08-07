"""Tests for mlx_addons.linalg.syrk — blocked symmetric rank-k update."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.linalg import syrk, gram
from mlx_addons.linalg._syrk import _num_blocks


def _rand(shape, seed=0, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _rel_err(got, ref):
    got = np.array(got, dtype=np.float64)
    ref = np.array(ref, dtype=np.float64)
    return np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-30)


# ============================================================
# correctness
# ============================================================


class TestCorrectness:
    @pytest.mark.parametrize("shape", [(64, 32), (128, 512), (300, 128), (1024, 256)])
    @pytest.mark.parametrize("blocks", [1, 2, 3, 4, 7])
    def test_matches_matmul(self, shape, blocks):
        """Forced blocking reproduces X @ X.T for any block count."""
        X = _rand(shape, seed=1)
        ref = X @ X.T
        got = syrk(X, blocks=blocks)
        mx.eval(ref, got)
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < 1e-5

    @pytest.mark.parametrize("shape", [(64, 32), (128, 512), (300, 128)])
    @pytest.mark.parametrize("blocks", [2, 3, 5])
    def test_trans_matches_matmul(self, shape, blocks):
        """trans=True reproduces X.T @ X."""
        X = _rand(shape, seed=2)
        ref = X.T @ X
        got = syrk(X, trans=True, blocks=blocks)
        mx.eval(ref, got)
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < 1e-5

    def test_gram_alias(self):
        X = _rand((256, 64), seed=3)
        a, b = gram(X, blocks=4), syrk(X, trans=True, blocks=4)
        mx.eval(a, b)
        assert _rel_err(a, b) == 0.0

    def test_uneven_blocks(self):
        """Dimensions that do not divide by the block count."""
        X = _rand((101, 37), seed=4)
        ref = X @ X.T
        got = syrk(X, blocks=8)
        mx.eval(ref, got)
        assert _rel_err(got, ref) < 1e-5

    def test_output_is_exactly_symmetric(self):
        """Off-diagonal blocks are mirrored, so symmetry is exact — not just
        within tolerance, unlike a plain X @ X.T."""
        X = _rand((256, 129), seed=5)
        A = syrk(X, blocks=4)
        mx.eval(A)
        assert np.array_equal(np.array(A), np.array(A).T)

    @pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
    def test_batched(self, batch_shape):
        X = _rand(batch_shape + (128, 64), seed=6)
        ref = mx.matmul(X, mx.swapaxes(X, -1, -2))
        got = syrk(X, blocks=4)
        mx.eval(ref, got)
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < 1e-5

    @pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
    def test_batched_trans(self, batch_shape):
        X = _rand(batch_shape + (128, 64), seed=7)
        ref = mx.matmul(mx.swapaxes(X, -1, -2), X)
        got = syrk(X, trans=True, blocks=4)
        mx.eval(ref, got)
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < 1e-5

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    def test_dtype_preserved(self, dtype):
        X = _rand((256, 128), seed=8, dtype=dtype)
        got = syrk(X, blocks=4)
        mx.eval(got)
        assert got.dtype == dtype

    def test_rank_deficient(self):
        """Low-rank input: result must still match the plain product."""
        U = _rand((256, 8), seed=9)
        V = _rand((8, 128), seed=10)
        X = U @ V
        ref = X @ X.T
        got = syrk(X, blocks=4)
        mx.eval(ref, got)
        assert _rel_err(got, ref) < 1e-5


# ============================================================
# dispatch heuristic
# ============================================================


class TestBlockPlanning:
    def test_small_inputs_fall_back(self):
        """Below the thresholds the automatic path must not block."""
        assert _num_blocks(512, 8192, None, 1024, 2048, 1024) == 1  # dim too small
        assert _num_blocks(8192, 256, None, 1024, 2048, 1024) == 1  # contraction too short

    def test_block_count_scales_with_dim(self):
        assert _num_blocks(2048, 4096, None, 1024, 2048, 1024) == 2
        assert _num_blocks(4096, 4096, None, 1024, 2048, 1024) == 4
        assert _num_blocks(8192, 4096, None, 1024, 2048, 1024) == 8

    def test_block_count_capped(self):
        assert _num_blocks(1 << 20, 4096, None, 1024, 2048, 1024) == 16

    def test_explicit_blocks_override_thresholds(self):
        assert _num_blocks(64, 8, 4, 1024, 2048, 1024) == 4

    def test_blocks_never_exceed_dim(self):
        assert _num_blocks(3, 8192, 16, 1024, 2048, 1024) == 3

    def test_auto_path_matches_matmul(self):
        """Default (auto) path on a shape large enough to actually block."""
        X = _rand((2048, 1024), seed=11)
        ref = X @ X.T
        got = syrk(X)
        mx.eval(ref, got)
        assert _rel_err(got, ref) < 1e-5

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            syrk(mx.zeros((8,)))
