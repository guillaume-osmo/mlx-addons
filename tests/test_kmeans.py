"""Tests for mlx_addons.cluster.KMeans."""

import numpy as np
import pytest

from mlx_addons.cluster import KMeans


def _make_blobs(n_samples=600, n_features=4, n_centers=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_centers, n_features)).astype(np.float32)
    labels = rng.integers(0, n_centers, size=n_samples)
    X = centers[labels] + rng.standard_normal((n_samples, n_features)).astype(np.float32) * 0.3
    return X.astype(np.float32), labels.astype(np.int32), centers


class TestKMeans:
    def test_fit_shape(self):
        X, _, _ = _make_blobs()
        km = KMeans(n_clusters=4, random_state=0).fit(X)
        assert km.cluster_centers_.shape == (4, X.shape[1])
        assert km.labels_.shape == (X.shape[0],)
        assert km.inertia_ >= 0
        assert km.n_iter_ >= 1

    def test_recovers_known_centers(self):
        """On well-separated blobs, kmeans should recover the true cluster structure."""
        X, true_labels, true_centers = _make_blobs(n_centers=4)
        km = KMeans(n_clusters=4, random_state=0, n_init=5).fit(X)
        # Each predicted center should be close to one of the true centers.
        from scipy.spatial.distance import cdist
        dists = cdist(km.cluster_centers_, true_centers)
        # Greedy-matching: every predicted center matches a unique true center
        matched = set()
        for i in range(4):
            j = int(np.argmin(dists[i]))
            assert j not in matched, f"two predicted centers matched to the same true center {j}"
            matched.add(j)
            assert dists[i, j] < 1.0, f"predicted center {i} far from nearest true ({dists[i, j]:.3f})"

    def test_predict_new_points(self):
        X, _, _ = _make_blobs()
        km = KMeans(n_clusters=4, random_state=0).fit(X)
        rng = np.random.default_rng(10)
        X_new = X[:20] + rng.standard_normal(X[:20].shape).astype(np.float32) * 0.1
        labels_new = km.predict(X_new)
        assert labels_new.shape == (20,)
        assert labels_new.dtype == np.int32
        # Tight noise → predicted label should match the point's own original label.
        orig_labels = km.predict(X[:20])
        np.testing.assert_array_equal(labels_new, orig_labels)

    def test_transform_shape(self):
        X, _, _ = _make_blobs()
        km = KMeans(n_clusters=5, random_state=0).fit(X)
        D = km.transform(X)
        assert D.shape == (X.shape[0], 5)
        assert np.all(D >= 0)
        # transform() is Euclidean distance — its argmin should match labels_.
        np.testing.assert_array_equal(np.argmin(D, axis=1).astype(np.int32), km.labels_)

    def test_n_init_improves_or_matches_inertia(self):
        X, _, _ = _make_blobs(n_samples=400, n_centers=5)
        km_low = KMeans(n_clusters=5, n_init=1, random_state=0).fit(X)
        km_high = KMeans(n_clusters=5, n_init=5, random_state=0).fit(X)
        assert km_high.inertia_ <= km_low.inertia_ + 1e-4

    def test_sklearn_parity_inertia(self):
        """Our inertia should be comparable to sklearn's (different init → ±10%)."""
        sk = pytest.importorskip("sklearn.cluster")
        X, _, _ = _make_blobs(n_samples=500, n_centers=6)
        ours = KMeans(n_clusters=6, n_init=5, random_state=0).fit(X)
        theirs = sk.KMeans(n_clusters=6, n_init=5, random_state=0, init="k-means++").fit(X)
        # Allow for the fact that both are randomized — inertia within 15% of each other.
        ratio = ours.inertia_ / max(theirs.inertia_, 1e-12)
        assert 0.85 < ratio < 1.20, f"inertia ratio = {ratio:.3f}"

    def test_empty_cluster_reseeded(self):
        """Degenerate init (duplicate points) shouldn't crash and no output cluster is empty."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 4)).astype(np.float32)
        # Ask for more clusters than natural groupings — forces some close centers
        km = KMeans(n_clusters=8, n_init=1, random_state=0).fit(X)
        # No NaNs, no infs in centers
        assert np.isfinite(km.cluster_centers_).all()
        # Every cluster is represented (after our empty-cluster reseeding)
        unique = np.unique(km.labels_)
        assert len(unique) == 8, f"only {len(unique)}/8 clusters populated"
