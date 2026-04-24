"""K-Means on Apple Silicon — sklearn-compatible, Metal-accelerated.

Lloyd's algorithm with k-means++ initialization. The assignment step is
one batched distance via Metal matmul (``X @ C.T``) + ``argmin``; the
update step is a one-hot mask + ``one_hot.T @ X`` — also Metal matmul.
No custom kernels needed.

Drop-in for ``sklearn.cluster.KMeans`` for the ``fit`` / ``predict`` /
``fit_predict`` / ``transform`` API (Lloyd-only — no Elkan / mini-batch
yet, though the latter is trivial to add).

Example
-------
>>> import numpy as np
>>> from mlx_addons.cluster import KMeans
>>> X = np.random.RandomState(0).randn(10_000, 32).astype(np.float32)
>>> km = KMeans(n_clusters=16, random_state=0).fit(X)
>>> km.labels_.shape
(10000,)
>>> km.cluster_centers_.shape
(16, 32)
>>> km.predict(np.random.RandomState(1).randn(100, 32).astype(np.float32)).shape
(100,)
"""

from __future__ import annotations

from typing import Literal, Optional

import mlx.core as mx
import numpy as np


def _sq_dists_mx(X: mx.array, C: mx.array) -> mx.array:
    """Pairwise squared distances via one Metal matmul. Returns (n, k)."""
    Xn = mx.sum(X * X, axis=1, keepdims=True)
    Cn = mx.sum(C * C, axis=1, keepdims=True).T
    D = Xn + Cn - 2.0 * (X @ C.T)
    return mx.maximum(D, mx.zeros_like(D))


def _kmeans_plus_plus_init(
    X_np: np.ndarray, k: int, rng: np.random.Generator, n_local_trials: int = 2
) -> np.ndarray:
    """k-means++ seeding. Returns an (k, d) array of initial centroids.

    Uses n_local_trials candidates per step (like sklearn) to reduce variance.
    Distance computation runs through MLX via _sq_dists_mx, so even large
    corpora get the Metal matmul speedup during init.
    """
    n, d = X_np.shape
    X_mx = mx.array(X_np)
    centers = np.empty((k, d), dtype=np.float32)

    # First center: uniform sample
    i0 = int(rng.integers(n))
    centers[0] = X_np[i0]

    # Closest-distance-so-far vector
    c0 = mx.array(centers[0:1])
    dist_sq = _sq_dists_mx(X_mx, c0).reshape(n)                # (n,)
    mx.eval(dist_sq)
    dist_sq_np = np.array(dist_sq)

    for c in range(1, k):
        total = float(dist_sq_np.sum())
        if total <= 0:                                           # degenerate: all identical
            i = int(rng.integers(n))
            centers[c] = X_np[i]
            continue
        # Pick n_local_trials candidates proportional to dist_sq
        probs = dist_sq_np / total
        cand_ids = rng.choice(n, size=n_local_trials, replace=True, p=probs)
        # Evaluate: which candidate minimizes the post-add potential?
        cands_mx = mx.array(X_np[cand_ids])                      # (t, d)
        d_cands = _sq_dists_mx(X_mx, cands_mx)                   # (n, t)
        new_dist_np = np.minimum(dist_sq_np[:, None], np.array(d_cands))
        potentials = new_dist_np.sum(axis=0)                     # (t,)
        best = int(np.argmin(potentials))
        centers[c] = X_np[cand_ids[best]]
        dist_sq_np = new_dist_np[:, best]
    return centers


class KMeans:
    """K-means clustering — Lloyd's algorithm, Metal-accelerated.

    Parameters
    ----------
    n_clusters : int
    init : {"k-means++", "random"}, default "k-means++"
    n_init : int, default=3
        Run this many random inits and keep the best by inertia.
    max_iter : int, default=300
    tol : float, default=1e-4
        Relative Frobenius shift in centers below which we declare convergence.
    random_state : int, optional

    Attributes
    ----------
    cluster_centers_ : (n_clusters, n_features) np.ndarray
    labels_ : (n_samples,) np.ndarray[int32]
    inertia_ : float
        Sum of squared distances to the closest center (sklearn convention).
    n_iter_ : int
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centers(self, X_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, _d = X_np.shape
        k = self.n_clusters
        if self.init == "random" or k >= n:
            idx = rng.choice(n, size=min(k, n), replace=(k > n))
            return X_np[idx].copy()
        return _kmeans_plus_plus_init(X_np, k, rng)

    def _lloyd_one_run(self, X_mx: mx.array, X_np: np.ndarray, rng: np.random.Generator):
        C = mx.array(self._init_centers(X_np, rng))
        mx.eval(C)
        prev_shift = float("inf")
        for it in range(self.max_iter):
            # Assignment (one matmul on Metal)
            D = _sq_dists_mx(X_mx, C)
            labels = mx.argmin(D, axis=1)
            # Update via one-hot @ X (matmul)
            k = self.n_clusters
            OH = (labels[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
            counts = mx.sum(OH, axis=0)                         # (k,)
            sums = OH.T @ X_mx                                   # (k, d)
            C_new = sums / mx.maximum(counts[:, None], mx.array(1.0))
            # Handle empty clusters: reseed at the point with max inertia
            empty = counts == 0
            n_empty = int(mx.sum(empty.astype(mx.int32)).item())
            if n_empty > 0:
                # Replace each empty center with the point farthest from its centroid.
                # Quick CPU fallback — this is rare and slow in a worst case.
                mx.eval(C_new, labels)
                C_new_np = np.array(C_new)
                labels_np = np.array(labels)
                D_np = np.array(D)
                dist_to_assigned = D_np[np.arange(X_np.shape[0]), labels_np]
                far_order = np.argsort(-dist_to_assigned)
                empty_idx = np.where(np.array(empty))[0]
                for i, e in enumerate(empty_idx):
                    C_new_np[e] = X_np[far_order[i]]
                C_new = mx.array(C_new_np)

            # Convergence check (Frobenius norm of center shift, relative to center norm)
            shift_sq = mx.sum((C_new - C) ** 2)
            Cnorm_sq = mx.sum(C_new ** 2) + 1e-20
            rel_shift = mx.sqrt(shift_sq / Cnorm_sq)
            mx.eval(rel_shift, C_new)
            rel_shift_val = float(rel_shift.item())
            C = C_new
            if rel_shift_val < self.tol:
                it += 1
                break
            prev_shift = rel_shift_val

        # Final inertia using the converged centers
        D_final = _sq_dists_mx(X_mx, C)
        labels_final = mx.argmin(D_final, axis=1)
        min_d = mx.take_along_axis(D_final, labels_final[:, None], axis=1).reshape(-1)
        inertia = float(mx.sum(min_d).item())
        mx.eval(C, labels_final)
        return np.array(C), np.array(labels_final).astype(np.int32), inertia, it + 1

    def fit(self, X: np.ndarray, y=None, sample_weight=None) -> "KMeans":
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X_np.ndim != 2:
            raise ValueError(f"KMeans expects 2-D input, got shape {X_np.shape}")
        X_mx = mx.array(X_np)
        base_seed = self.random_state if self.random_state is not None else 0

        best_inertia = float("inf")
        best_C, best_labels, best_iters = None, None, 0
        for run in range(self.n_init):
            rng = np.random.default_rng(base_seed + run)
            C, labels, inertia, n_iter = self._lloyd_one_run(X_mx, X_np, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_C = C
                best_labels = labels
                best_iters = n_iter
        self.cluster_centers_ = best_C
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iters
        self.n_features_in_ = X_np.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        X_mx = mx.array(X_np)
        C_mx = mx.array(self.cluster_centers_)
        D = _sq_dists_mx(X_mx, C_mx)
        labels = mx.argmin(D, axis=1)
        mx.eval(labels)
        return np.array(labels).astype(np.int32)

    def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Distance from each sample to each centroid — sklearn convention."""
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        X_mx = mx.array(X_np)
        C_mx = mx.array(self.cluster_centers_)
        D = _sq_dists_mx(X_mx, C_mx)
        mx.eval(D)
        return np.sqrt(np.array(D))
