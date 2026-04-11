"""Tests for mlx_addons.knn GPU-accelerated nearest neighbors."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.knn import knn


class TestKNN:
    def test_basic(self):
        """KNN returns correct shapes."""
        np.random.seed(42)
        pos = mx.array(np.random.randn(500, 3).astype(np.float32))
        k = 8
        rnn, inn = knn(pos, k)
        mx.eval(rnn, inn)
        assert rnn.shape == (500, k)
        assert inn.shape == (500, k)

    def test_sorted_distances(self):
        """Distances should be sorted ascending per query."""
        np.random.seed(42)
        pos = mx.array(np.random.randn(200, 3).astype(np.float32))
        rnn, inn = knn(pos, k=8)
        mx.eval(rnn)
        rnn_np = np.array(rnn)
        for i in range(rnn_np.shape[0]):
            assert np.all(np.diff(rnn_np[i]) >= -1e-6), f"Row {i} not sorted"

    def test_accuracy_vs_brute_force(self):
        """KNN results should mostly match brute-force."""
        np.random.seed(42)
        N, dim, k = 300, 3, 8
        pos_np = np.random.randn(N, dim).astype(np.float32)
        pos = mx.array(pos_np)

        _, inn = knn(pos, k)
        mx.eval(inn)
        inn_np = np.array(inn)

        # Brute force
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_np)
        _, bf_indices = tree.query(pos_np, k=k + 1)
        bf_indices = bf_indices[:, 1:]  # exclude self

        # Check recall (allow some misses due to tree approximation)
        recalls = []
        for i in range(N):
            found = set(inn_np[i].tolist())
            expected = set(bf_indices[i].tolist())
            recalls.append(len(found & expected) / k)
        mean_recall = np.mean(recalls)
        assert mean_recall > 0.8, f"Mean recall {mean_recall:.2f} too low"

    def test_different_k(self):
        """Test various k values."""
        pos = mx.array(np.random.randn(100, 3).astype(np.float32))
        for k in [1, 4, 8, 16, 32]:
            rnn, inn = knn(pos, k)
            mx.eval(rnn, inn)
            assert rnn.shape == (100, k)
            assert inn.shape == (100, k)
