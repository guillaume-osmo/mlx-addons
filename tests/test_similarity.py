"""Tests for mlx_addons.similarity."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_addons.similarity import (
    OnlineSingleLinkCluster,
    StreamingFingerprintBank,
    cosine_sim_batched,
    l2_normalize,
    max_cosine_to_set,
    max_tanimoto_to_set,
    tanimoto_binary_batched,
)


class TestCosine:
    def test_self_similarity_is_one(self):
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(4, 16).astype(np.float32))
        sims = cosine_sim_batched(x, x)
        mx.eval(sims)
        # Diagonal should be ~1
        diag = np.array(sims).diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5)

    def test_max_to_empty_set_is_zero(self):
        x = mx.array(np.random.randn(4, 16).astype(np.float32))
        empty = mx.zeros((0, 16))
        out = max_cosine_to_set(x, empty)
        mx.eval(out)
        assert np.array(out).shape == (4,)
        assert np.allclose(np.array(out), 0.0)

    def test_l2_normalize_unit_norm(self):
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(4, 16).astype(np.float32))
        x_n = l2_normalize(x)
        mx.eval(x_n)
        norms = np.linalg.norm(np.array(x_n), axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestTanimoto:
    def test_identical_fps_one(self):
        rng = np.random.RandomState(0)
        bits = mx.array((rng.rand(4, 32) > 0.5).astype(np.uint8))
        sims = tanimoto_binary_batched(bits, bits)
        mx.eval(sims)
        diag = np.array(sims).diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5)

    def test_disjoint_fps_zero(self):
        # Two fingerprints with no overlap → Tanimoto = 0
        a = mx.array(np.array([[1, 1, 0, 0]], dtype=np.uint8))
        b = mx.array(np.array([[0, 0, 1, 1]], dtype=np.uint8))
        sim = tanimoto_binary_batched(a, b)
        mx.eval(sim)
        assert float(sim[0, 0]) < 1e-5


class TestStreamingBank:
    def test_grow(self):
        bank = StreamingFingerprintBank(dim=8, init_capacity=4)
        assert len(bank) == 0
        rng = np.random.RandomState(0)
        for i in range(10):
            bank.add_batch(mx.array(rng.randn(1, 8).astype(np.float32)))
        mx.eval(bank.matrix)
        assert len(bank) == 10
        assert bank.matrix.shape == (10, 8)

    def test_round_trip_query(self):
        rng = np.random.RandomState(0)
        bank = StreamingFingerprintBank(dim=16)
        fps = mx.array(rng.randn(5, 16).astype(np.float32))
        bank.add_batch(fps)
        # Query with the same fps → max cosine should be 1.0
        max_sims = max_cosine_to_set(fps, bank.matrix)
        mx.eval(max_sims)
        assert np.allclose(np.array(max_sims), 1.0, atol=1e-5)


class TestOnlineSingleLinkCluster:
    def test_first_assignment(self):
        c = OnlineSingleLinkCluster(threshold=0.5)
        rng = np.random.RandomState(0)
        cid, sim = c.assign(rng.randn(16).astype(np.float32))
        assert cid == 0
        assert sim == 0.0
        assert c.n_clusters == 1

    def test_similar_vectors_same_cluster(self):
        c = OnlineSingleLinkCluster(threshold=0.7)
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        v_similar = np.array([0.9, 0.1, 0, 0], dtype=np.float32)  # cos sim ~0.99
        cid1, _ = c.assign(v); c.increment(cid1)
        cid2, _ = c.assign(v_similar); c.increment(cid2)
        assert cid1 == cid2
        assert c.n_clusters == 1

    def test_orthogonal_vectors_distinct_clusters(self):
        c = OnlineSingleLinkCluster(threshold=0.5)
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)  # orthogonal → cos = 0
        cid1, _ = c.assign(v1)
        cid2, _ = c.assign(v2)
        assert cid1 != cid2
        assert c.n_clusters == 2

    def test_capacity(self):
        c = OnlineSingleLinkCluster(threshold=0.5, max_per_cluster=2)
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        cid, _ = c.assign(v)
        assert not c.at_capacity(cid)
        c.increment(cid)
        c.increment(cid)
        assert c.at_capacity(cid)
