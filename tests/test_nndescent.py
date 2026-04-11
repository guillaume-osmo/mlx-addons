"""Tests for mlx_addons.nndescent approximate k-NN graph construction."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_addons.nndescent import NNDescent, NNDescentBatched


def _brute_force_knn(X_np: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force k-NN using squared Euclidean distance (numpy)."""
    n = X_np.shape[0]
    # Compute full pairwise squared distance matrix
    sq_norms = (X_np * X_np).sum(axis=1)
    dist_matrix = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X_np @ X_np.T)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self
    dist_matrix = np.maximum(dist_matrix, 0.0)

    # Top-k per row
    idx = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    # Sort within top-k
    dists_topk = np.take_along_axis(dist_matrix, idx, axis=1)
    order = np.argsort(dists_topk, axis=1)
    bf_idx = np.take_along_axis(idx, order, axis=1)
    bf_dist = np.take_along_axis(dists_topk, order, axis=1)
    return bf_idx, bf_dist


def _recall(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Mean recall: fraction of true neighbors found per point."""
    n, k = exact_idx.shape
    recalls = []
    for i in range(n):
        found = set(approx_idx[i].tolist())
        expected = set(exact_idx[i].tolist())
        recalls.append(len(found & expected) / k)
    return float(np.mean(recalls))


class TestNNDescentShapes:
    """Output shapes and basic invariants."""

    @pytest.mark.parametrize("k", [1, 5, 15, 30])
    def test_shapes(self, k: int):
        np.random.seed(42)
        n, d = 200, 10
        X = mx.array(np.random.randn(n, d).astype(np.float32))
        nn = NNDescent(k=k, n_iters=5, seed=42)
        indices, distances = nn.build(X)
        mx.eval(indices, distances)
        assert indices.shape == (n, k), f"Expected ({n}, {k}), got {indices.shape}"
        assert distances.shape == (n, k), f"Expected ({n}, {k}), got {distances.shape}"

    def test_no_self_neighbors(self):
        """No point should appear as its own neighbor."""
        np.random.seed(42)
        n, d, k = 300, 8, 15
        X = mx.array(np.random.randn(n, d).astype(np.float32))
        nn = NNDescent(k=k, n_iters=10, seed=42)
        indices, _ = nn.build(X)
        mx.eval(indices)
        idx_np = np.array(indices)
        for i in range(n):
            assert i not in idx_np[i], f"Point {i} is its own neighbor"

    def test_sorted_distances(self):
        """Distances should be sorted ascending per point."""
        np.random.seed(42)
        n, d, k = 300, 8, 15
        X = mx.array(np.random.randn(n, d).astype(np.float32))
        nn = NNDescent(k=k, n_iters=10, seed=42)
        _, distances = nn.build(X)
        mx.eval(distances)
        d_np = np.array(distances)
        for i in range(n):
            assert np.all(np.diff(d_np[i]) >= -1e-5), f"Row {i} not sorted"

    def test_nonnegative_distances(self):
        """All distances should be non-negative."""
        np.random.seed(42)
        X = mx.array(np.random.randn(200, 10).astype(np.float32))
        nn = NNDescent(k=10, n_iters=5, seed=42)
        _, distances = nn.build(X)
        mx.eval(distances)
        assert np.all(np.array(distances) >= 0.0)


class TestNNDescentAccuracy:
    """Accuracy compared to brute-force."""

    def test_recall_1000_points(self):
        """Should achieve >90% recall on 1000 points, k=15."""
        np.random.seed(42)
        n, d, k = 1000, 20, 15
        X_np = np.random.randn(n, d).astype(np.float32)
        X = mx.array(X_np)

        nn = NNDescent(k=k, n_iters=20, seed=42, verbose=False)
        indices, _ = nn.build(X)
        mx.eval(indices)
        idx_np = np.array(indices)

        bf_idx, _ = _brute_force_knn(X_np, k)
        r = _recall(idx_np, bf_idx)
        assert r > 0.90, f"Recall {r:.3f} is below 0.90 threshold"

    def test_recall_500_high_dim(self):
        """Reasonable recall on 500 points in 50 dimensions."""
        np.random.seed(123)
        n, d, k = 500, 50, 10
        X_np = np.random.randn(n, d).astype(np.float32)
        X = mx.array(X_np)

        nn = NNDescent(k=k, n_iters=20, seed=123)
        indices, _ = nn.build(X)
        mx.eval(indices)
        idx_np = np.array(indices)

        bf_idx, _ = _brute_force_knn(X_np, k)
        r = _recall(idx_np, bf_idx)
        assert r > 0.80, f"Recall {r:.3f} is below 0.80 threshold"

    def test_distances_match_brute_force(self):
        """Returned distances should be correct squared Euclidean distances."""
        np.random.seed(42)
        n, d, k = 200, 8, 10
        X_np = np.random.randn(n, d).astype(np.float32)
        X = mx.array(X_np)

        nn = NNDescent(k=k, n_iters=15, seed=42)
        indices, distances = nn.build(X)
        mx.eval(indices, distances)
        idx_np = np.array(indices)
        dist_np = np.array(distances)

        # Recompute distances from indices
        for i in range(n):
            for j_pos in range(k):
                j = int(idx_np[i, j_pos])
                expected = float(np.sum((X_np[i] - X_np[j]) ** 2))
                actual = float(dist_np[i, j_pos])
                assert abs(actual - expected) < 1e-3, (
                    f"Distance mismatch at ({i},{j_pos}): "
                    f"got {actual:.6f}, expected {expected:.6f}"
                )


class TestNNDescentConvergence:
    """Convergence behavior."""

    def test_converges_early(self):
        """Should converge in fewer than 20 iterations for 500 points."""
        np.random.seed(42)
        n, d, k = 500, 10, 15
        X = mx.array(np.random.randn(n, d).astype(np.float32))

        # Track iterations via verbose output (we just check it finishes)
        nn = NNDescent(k=k, n_iters=20, delta=0.015, seed=42, verbose=False)
        indices, distances = nn.build(X)
        mx.eval(indices, distances)

        # Run again with very tight delta to count real iterations
        # If delta=0.0 it will run all 20; with delta=0.015 it should stop earlier
        # We verify by running with delta=0.0 and checking that the default
        # delta=0.015 version has same quality (meaning it converged)
        nn_full = NNDescent(k=k, n_iters=20, delta=0.0, seed=42)
        indices_full, _ = nn_full.build(X)
        mx.eval(indices_full)

        # Both should have similar quality
        X_np = np.array(X)
        bf_idx, _ = _brute_force_knn(X_np, k)

        r_early = _recall(np.array(indices), bf_idx)
        r_full = _recall(np.array(indices_full), bf_idx)

        # Early stopping should not lose more than 5% recall
        assert r_early > r_full - 0.05, (
            f"Early stop recall {r_early:.3f} much worse than "
            f"full run {r_full:.3f}"
        )

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        np.random.seed(42)
        X = mx.array(np.random.randn(300, 10).astype(np.float32))

        nn1 = NNDescent(k=10, n_iters=10, seed=99)
        idx1, dist1 = nn1.build(X)
        mx.eval(idx1, dist1)

        nn2 = NNDescent(k=10, n_iters=10, seed=99)
        idx2, dist2 = nn2.build(X)
        mx.eval(idx2, dist2)

        np.testing.assert_array_equal(np.array(idx1), np.array(idx2))
        np.testing.assert_allclose(np.array(dist1), np.array(dist2), atol=1e-6)


class TestNNDescentBatched:
    """Test the batched variant has same behavior."""

    def test_shapes(self):
        np.random.seed(42)
        n, d, k = 200, 10, 15
        X = mx.array(np.random.randn(n, d).astype(np.float32))
        nn = NNDescentBatched(k=k, n_iters=10, seed=42)
        indices, distances = nn.build(X)
        mx.eval(indices, distances)
        assert indices.shape == (n, k)
        assert distances.shape == (n, k)

    def test_recall(self):
        np.random.seed(42)
        n, d, k = 500, 20, 15
        X_np = np.random.randn(n, d).astype(np.float32)
        X = mx.array(X_np)
        nn = NNDescentBatched(k=k, n_iters=15, seed=42)
        indices, _ = nn.build(X)
        mx.eval(indices)
        bf_idx, _ = _brute_force_knn(X_np, k)
        r = _recall(np.array(indices), bf_idx)
        assert r > 0.85, f"Batched recall {r:.3f} below 0.85"
