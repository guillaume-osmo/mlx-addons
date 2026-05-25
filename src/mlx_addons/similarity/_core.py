"""Vector similarity + online clustering — MLX-native implementations.

All operations work on `mlx.core.array`. For numpy inputs, wrap with `mx.array`.
"""
from __future__ import annotations

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError("pip install mlx") from e

import numpy as np


# ============================================================================
# Cosine similarity (dense float vectors)
# ============================================================================

def l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-9) -> mx.array:
    return x / (mx.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine_sim_batched(query: mx.array, reference: mx.array, eps: float = 1e-9) -> mx.array:
    """Returns (B, N) cosine similarity matrix.
    query: (B, D), reference: (N, D)."""
    q = l2_normalize(query, eps=eps)
    r = l2_normalize(reference, eps=eps)
    return q @ r.T


def max_cosine_to_set(query: mx.array, reference: mx.array) -> mx.array:
    """Returns (B,) — max cosine sim of each query against the reference set.
    If reference is empty, returns zeros (no neighbors → no similarity)."""
    if reference.shape[0] == 0:
        return mx.zeros((query.shape[0],))
    return cosine_sim_batched(query, reference).max(axis=-1)


# ============================================================================
# Tanimoto similarity (binary fingerprints, e.g. Morgan/ECFP)
# ============================================================================

def tanimoto_binary_batched(query: mx.array, reference: mx.array, eps: float = 1e-9) -> mx.array:
    """Pairwise Tanimoto for binary fingerprints. Returns (B, N).
    Inputs are bool or uint8/uint32 arrays; treated as 0/1 floats internally."""
    q = query.astype(mx.float32)
    r = reference.astype(mx.float32)
    inter = q @ r.T                                  # |A ∩ B|
    q_sum = q.sum(axis=-1, keepdims=True)            # |A|
    r_sum = r.sum(axis=-1, keepdims=True).T          # |B|
    union = q_sum + r_sum - inter
    return inter / (union + eps)


def max_tanimoto_to_set(query: mx.array, reference: mx.array) -> mx.array:
    if reference.shape[0] == 0:
        return mx.zeros((query.shape[0],))
    return tanimoto_binary_batched(query, reference).max(axis=-1)


# ============================================================================
# StreamingFingerprintBank — append-only MLX-resident matrix
# ============================================================================

class StreamingFingerprintBank:
    """Append-only MLX-resident (N, D) fingerprint matrix for fast novelty queries.

    Matches the 'accepted set grows monotonically' pattern of adaptive samplers.
    Doubles capacity when full (amortized O(1) append).
    """

    def __init__(self, dim: int, init_capacity: int = 1024, dtype=mx.float32):
        self.dim = dim
        self.dtype = dtype
        self._capacity = init_capacity
        self._buffer = mx.zeros((init_capacity, dim), dtype=dtype)
        self._n = 0

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2)
        new_buf = mx.zeros((new_cap, self.dim), dtype=self.dtype)
        if self._n > 0:
            new_buf[:self._n] = self._buffer[:self._n]
        self._buffer = new_buf
        self._capacity = new_cap

    def add_batch(self, fps: mx.array) -> None:
        n_new = fps.shape[0]
        if n_new == 0:
            return
        self._grow(self._n + n_new)
        self._buffer[self._n:self._n + n_new] = fps.astype(self.dtype)
        self._n += n_new

    @property
    def matrix(self) -> mx.array:
        return self._buffer[:self._n]

    def __len__(self) -> int:
        return self._n


# ============================================================================
# OnlineSingleLinkCluster — vector-agnostic online clustering in cosine space
# ============================================================================

class OnlineSingleLinkCluster:
    """Online single-link clustering on dense vectors using cosine similarity.

    Each new vector joins the closest centroid if cosine sim >= threshold;
    otherwise starts a new centroid. Tracks per-cluster counts so the caller
    can cap members per cluster (useful for diversity-bounded acceptance).

    Uses numpy internally for the centroid set since `assign` is called
    per-sample in tight loops where per-call MLX dispatch overhead dominates.
    For batched novelty queries against many candidates, use
    `cosine_sim_batched` / `max_cosine_to_set` on a StreamingFingerprintBank.
    """

    def __init__(self, threshold: float = 0.75, max_per_cluster: int = 20):
        self.threshold = threshold
        self.max_per_cluster = max_per_cluster
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    def _l2(self, x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def assign(self, fp) -> tuple[int, float]:
        """Returns (cluster_id, max_sim_to_existing_centroid).
        Always returns a valid cluster_id — creates a new cluster if needed."""
        if hasattr(fp, "shape") and not isinstance(fp, np.ndarray):
            fp = np.asarray(fp)
        fp_n = self._l2(fp.astype(np.float32))
        if not self._centroids:
            self._centroids.append(fp_n)
            self._counts.append(0)
            return 0, 0.0
        sims = np.array([float(fp_n @ c) for c in self._centroids])
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return best, float(sims[best])
        self._centroids.append(fp_n)
        self._counts.append(0)
        return len(self._centroids) - 1, float(sims[best])

    def increment(self, cluster_id: int) -> None:
        self._counts[cluster_id] += 1

    def at_capacity(self, cluster_id: int) -> bool:
        return self._counts[cluster_id] >= self.max_per_cluster

    @property
    def n_clusters(self) -> int:
        return len(self._centroids)

    def summary(self) -> dict:
        return {i: c for i, c in enumerate(self._counts)}
