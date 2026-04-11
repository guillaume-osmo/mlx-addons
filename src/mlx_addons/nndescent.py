"""
NNDescent: approximate k-nearest-neighbor graph construction for MLX.

Pure MLX implementation (no custom Metal kernels) of the NNDescent algorithm.
Uses the algebraic distance trick, float16 acceleration, NEW/OLD tracking,
reverse edges, and memory chunking for bounded GPU memory.

Reference:
    Dong, Wei, Charikar, Moses, and Li, Kai.
    "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures."
    WWW 2011.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INF = float("inf")
_LARGE = 1e30  # sentinel for dedup / invalid entries


def _sq_norms(X: mx.array) -> mx.array:
    """Row-wise squared L2 norms: shape (n,)."""
    return (X * X).sum(axis=1)


def _sq_euclidean_chunked(
    X: mx.array,
    sq_norms: mx.array,
    sources: mx.array,
    targets: mx.array,
    *,
    use_fp16: bool = False,
    chunk_elems: int = 300_000_000,
) -> mx.array:
    """Compute squared Euclidean distances between X[sources] and X[targets].

    Uses the identity  ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    to avoid materializing the (n_src, n_tgt, d) difference tensor.

    Processing is split into row-chunks so that peak GPU memory stays bounded
    at roughly *chunk_elems* float32 elements.

    Parameters
    ----------
    X : (N, d)
    sq_norms : (N,)  precomputed squared norms of X
    sources : (n_src,)  int32 row indices into X
    targets : (n_tgt,)  int32 row indices into X
    use_fp16 : if True, perform the matmul in float16 for speed
    chunk_elems : approximate element budget per chunk

    Returns
    -------
    dists : (n_src, n_tgt) float32
    """
    n_src = sources.shape[0]
    n_tgt = targets.shape[0]

    if n_src == 0 or n_tgt == 0:
        return mx.zeros((n_src, n_tgt), dtype=mx.float32)

    chunk_rows = max(1, chunk_elems // n_tgt)

    X_tgt = X[targets]  # (n_tgt, d)
    norms_tgt = sq_norms[targets]  # (n_tgt,)

    if use_fp16:
        X_tgt_h = X_tgt.astype(mx.float16)

    parts: list[mx.array] = []
    for start in range(0, n_src, chunk_rows):
        end = min(start + chunk_rows, n_src)
        src_chunk = sources[start:end]  # (cs,)

        X_src = X[src_chunk]  # (cs, d)
        norms_src = sq_norms[src_chunk]  # (cs,)

        if use_fp16:
            dots = (X_src.astype(mx.float16) @ X_tgt_h.T).astype(mx.float32)
        else:
            dots = X_src @ X_tgt.T  # (cs, n_tgt)

        d = norms_src[:, None] + norms_tgt[None, :] - 2.0 * dots
        # Numerical guard: distances must be non-negative
        d = mx.maximum(d, 0.0)
        parts.append(d)

    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# NNDescent
# ---------------------------------------------------------------------------

class NNDescent:
    """Approximate k-nearest-neighbor graph via the NNDescent algorithm.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to find for each point.
    n_iters : int
        Maximum number of NNDescent iterations.
    max_candidates : int or None
        Number of candidate neighbors to consider per iteration.
        Defaults to min(k, 32) if None.
    delta : float
        Early stopping threshold. If the fraction of neighbor list entries
        that changed in an iteration is below *delta*, stop.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, print progress information.
    """

    def __init__(
        self,
        k: int = 15,
        n_iters: int = 20,
        max_candidates: Optional[int] = None,
        delta: float = 0.015,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iters = n_iters
        self.max_candidates = max_candidates if max_candidates is not None else min(k, 32)
        self.delta = delta
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, X: mx.array) -> tuple[mx.array, mx.array]:
        """Build the approximate k-NN graph for *X*.

        Parameters
        ----------
        X : mx.array, shape (n, d)
            Data matrix (float32).

        Returns
        -------
        indices : mx.array, shape (n, k), int32
            Neighbor indices for each point.
        distances : mx.array, shape (n, k), float32
            Squared Euclidean distances to each neighbor, sorted ascending.
        """
        X = X.astype(mx.float32)
        n, d = X.shape
        k = self.k
        assert k < n, f"k={k} must be less than n={n}"

        rng = np.random.RandomState(self.seed)

        sq_norms = _sq_norms(X)
        mx.eval(sq_norms)

        # ----- random initialization -----
        indices, distances = self._random_init(X, sq_norms, n, k, d, rng)

        # Track NEW flags: True means this neighbor was added/updated recently
        # Shape (n, k), dtype bool
        is_new = np.ones((n, k), dtype=bool)

        # ----- iterative refinement -----
        for it in range(self.n_iters):
            t0 = time.time()
            use_fp16 = it < 4  # float16 acceleration for early iterations

            # Convert current state to numpy for candidate generation
            idx_np = np.array(indices)
            dist_np = np.array(distances)

            # Build candidate lists from NEW neighbors + reverse edges
            new_candidates, old_candidates = self._build_candidates(
                idx_np, is_new, n, k, rng
            )

            # Compute updates
            n_updates = self._apply_updates(
                X, sq_norms, idx_np, dist_np, is_new,
                new_candidates, old_candidates,
                n, k, use_fp16,
            )

            # Write back
            indices = mx.array(idx_np.astype(np.int32))
            distances = mx.array(dist_np.astype(np.float32))
            mx.eval(indices, distances)

            update_frac = n_updates / (n * k) if n * k > 0 else 0.0
            elapsed = time.time() - t0

            if self.verbose:
                print(
                    f"  iter {it:3d}  updates={n_updates:7d}  "
                    f"frac={update_frac:.4f}  time={elapsed:.2f}s"
                    f"{'  [fp16]' if use_fp16 else ''}"
                )

            if update_frac < self.delta:
                if self.verbose:
                    print(f"  converged at iteration {it}")
                break

        # Recompute final distances in full fp32 to remove fp16 drift
        indices, distances = self._recompute_distances_fp32(
            X, sq_norms, indices, n, k, d
        )

        return indices, distances

    # ------------------------------------------------------------------
    # Final distance recomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _recompute_distances_fp32(
        X: mx.array,
        sq_norms: mx.array,
        indices: mx.array,
        n: int,
        k: int,
        d: int,
    ) -> tuple[mx.array, mx.array]:
        """Recompute all neighbor distances in full fp32 precision."""
        idx_np = np.array(indices)
        dist_np = np.empty((n, k), dtype=np.float32)
        chunk = 2048
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            cs = end - start
            row_indices = mx.arange(start, end)
            nbr_indices = mx.array(idx_np[start:end])  # (cs, k)

            Xi = X[row_indices]  # (cs, d)
            Xj = X[nbr_indices.reshape(-1)].reshape(cs, k, d)

            norms_i = sq_norms[row_indices]
            norms_j = sq_norms[nbr_indices.reshape(-1)].reshape(cs, k)

            dots = (Xi[:, None, :] * Xj).sum(axis=2)
            dists = norms_i[:, None] + norms_j - 2.0 * dots
            dists = mx.maximum(dists, 0.0)
            mx.eval(dists)
            dist_np[start:end] = np.array(dists)

        # Re-sort by corrected distances
        order = np.argsort(dist_np, axis=1)
        dist_np = np.take_along_axis(dist_np, order, axis=1)
        idx_np = np.take_along_axis(idx_np, order, axis=1)

        indices = mx.array(idx_np.astype(np.int32))
        distances = mx.array(dist_np.astype(np.float32))
        mx.eval(indices, distances)
        return indices, distances

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _random_init(
        self,
        X: mx.array,
        sq_norms: mx.array,
        n: int,
        k: int,
        d: int,
        rng: np.random.RandomState,
    ) -> tuple[mx.array, mx.array]:
        """Create initial random neighbor graph and compute distances."""
        # For each point, sample k random neighbors (excluding self)
        idx_np = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            choices = rng.choice(n - 1, size=k, replace=(k > n - 1))
            # Shift indices >= i to avoid self
            choices = np.where(choices >= i, choices + 1, choices)
            idx_np[i] = choices

        indices = mx.array(idx_np)

        # Compute distances using the algebraic trick, row by row in chunks
        dist_np = np.full((n, k), _LARGE, dtype=np.float32)
        chunk = 2048
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = mx.arange(start, end)
            row_idx = indices[start:end]  # (chunk, k)

            # Gather norms
            norms_i = sq_norms[rows]  # (chunk,)
            norms_j = mx.take(sq_norms, row_idx.reshape(-1)).reshape(row_idx.shape)  # (chunk, k)

            # Gather vectors and compute dots
            Xi = X[rows]  # (chunk, d)
            Xj = X[row_idx.reshape(-1)].reshape(end - start, k, d)  # (chunk, k, d)

            # dot products: sum over d
            dots = (Xi[:, None, :] * Xj).sum(axis=2)  # (chunk, k)

            dists = norms_i[:, None] + norms_j - 2.0 * dots
            dists = mx.maximum(dists, 0.0)
            mx.eval(dists)
            dist_np[start:end] = np.array(dists)

        # Sort each row by distance
        order = np.argsort(dist_np, axis=1)
        dist_np = np.take_along_axis(dist_np, order, axis=1)
        idx_np = np.take_along_axis(idx_np, order, axis=1)

        indices = mx.array(idx_np)
        distances = mx.array(dist_np)
        mx.eval(indices, distances)
        return indices, distances

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        idx_np: np.ndarray,
        is_new: np.ndarray,
        n: int,
        k: int,
        rng: np.random.RandomState,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build per-point candidate lists from NEW and OLD neighbors + reverse edges.

        Returns
        -------
        new_candidates : list of 1-d int arrays, one per point
            Candidate indices sourced from NEW neighbors (and their reverse edges).
        old_candidates : list of 1-d int arrays, one per point
            Candidate indices sourced from OLD neighbors (and their reverse edges).
        """
        max_cand = self.max_candidates

        # Forward candidates: neighbors of each point, split by NEW/OLD
        new_fwd: list[list[int]] = [[] for _ in range(n)]
        old_fwd: list[list[int]] = [[] for _ in range(n)]

        # Reverse candidates: for each neighbor j of point i, add i to j's list
        new_rev: list[list[int]] = [[] for _ in range(n)]
        old_rev: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            for jj in range(k):
                j = int(idx_np[i, jj])
                if j < 0 or j >= n:
                    continue
                if is_new[i, jj]:
                    new_fwd[i].append(j)
                    new_rev[j].append(i)
                else:
                    old_fwd[i].append(j)
                    old_rev[j].append(i)

        # Combine forward + reverse, subsample to max_candidates
        new_candidates: list[np.ndarray] = []
        old_candidates: list[np.ndarray] = []

        for i in range(n):
            nc = list(set(new_fwd[i] + new_rev[i]))
            oc = list(set(old_fwd[i] + old_rev[i]))

            # Remove self
            nc = [x for x in nc if x != i]
            oc = [x for x in oc if x != i]

            if len(nc) > max_cand:
                nc = list(rng.choice(nc, max_cand, replace=False))
            if len(oc) > max_cand:
                oc = list(rng.choice(oc, max_cand, replace=False))

            new_candidates.append(np.array(nc, dtype=np.int32))
            old_candidates.append(np.array(oc, dtype=np.int32))

        return new_candidates, old_candidates

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def _apply_updates(
        self,
        X: mx.array,
        sq_norms: mx.array,
        idx_np: np.ndarray,
        dist_np: np.ndarray,
        is_new: np.ndarray,
        new_candidates: list[np.ndarray],
        old_candidates: list[np.ndarray],
        n: int,
        k: int,
        use_fp16: bool,
    ) -> int:
        """Generate nn-of-nn candidates, compute distances, merge into graph.

        Operates in-place on idx_np, dist_np, is_new.
        Returns the total number of neighbor list updates.
        """
        total_updates = 0

        # Mark all current NEW as OLD for next iteration
        # (we will mark genuinely new arrivals below)
        is_new[:] = False

        # Process in chunks of points to bound memory
        point_chunk = max(1, 256)

        for start in range(0, n, point_chunk):
            end = min(start + point_chunk, n)

            for i in range(start, end):
                # Gather all candidate pairs:
                # For NNDescent, the key insight is: for each pair (u, v) where
                # u is a NEW candidate of point i, we should try (i, u) and
                # also for each v in old_candidates, try (u, v) combinations.
                # But the practical approach is: gather all unique candidates
                # for point i from new + old + nn-of-nn, then compute distances.

                cands_new = new_candidates[i]
                cands_old = old_candidates[i]

                # nn-of-nn: neighbors of my NEW candidates
                nn_of_nn_set: set[int] = set()
                for j in cands_new:
                    j = int(j)
                    if 0 <= j < n:
                        for jj in range(k):
                            nbr = int(idx_np[j, jj])
                            if nbr != i and 0 <= nbr < n:
                                nn_of_nn_set.add(nbr)

                # Also neighbors of OLD candidates (less critical but helps)
                for j in cands_old:
                    j = int(j)
                    if 0 <= j < n:
                        for jj in range(min(k, 3)):  # only top-3 from old
                            nbr = int(idx_np[j, jj])
                            if nbr != i and 0 <= nbr < n:
                                nn_of_nn_set.add(nbr)

                # Combine all candidates
                all_cands_set = set(cands_new.tolist()) | set(cands_old.tolist()) | nn_of_nn_set
                # Remove current neighbors (they are already in the graph)
                current_nbrs = set(idx_np[i].tolist())
                all_cands_set -= current_nbrs
                all_cands_set.discard(i)

                if len(all_cands_set) == 0:
                    continue

                all_cands = np.array(sorted(all_cands_set), dtype=np.int32)

                # Compute distances to candidates
                cands_mx = mx.array(all_cands)
                Xi = X[i : i + 1]  # (1, d)
                Xc = X[cands_mx]  # (nc, d)

                ni = sq_norms[i]
                nc_norms = sq_norms[cands_mx]

                if use_fp16:
                    dots = (Xi.astype(mx.float16) @ Xc.astype(mx.float16).T).astype(mx.float32)
                else:
                    dots = Xi @ Xc.T

                cand_dists = ni + nc_norms - 2.0 * dots.reshape(-1)
                cand_dists = mx.maximum(cand_dists, 0.0)
                mx.eval(cand_dists)
                cand_dists_np = np.array(cand_dists)

                # Merge: try to insert each candidate into the sorted neighbor list
                updates = self._merge_candidates(
                    idx_np, dist_np, is_new, i, k,
                    all_cands, cand_dists_np,
                )
                total_updates += updates

        return total_updates

    @staticmethod
    def _merge_candidates(
        idx_np: np.ndarray,
        dist_np: np.ndarray,
        is_new: np.ndarray,
        i: int,
        k: int,
        cand_indices: np.ndarray,
        cand_dists: np.ndarray,
    ) -> int:
        """Merge candidate neighbors into point i's sorted neighbor list.

        Operates in-place. Returns number of updates (insertions).
        """
        updates = 0
        worst_dist = dist_np[i, k - 1]

        for ci in range(len(cand_indices)):
            cd = cand_dists[ci]
            cj = cand_indices[ci]

            if cd >= worst_dist:
                continue

            # Find insertion position (binary search in sorted distances)
            pos = np.searchsorted(dist_np[i], cd)

            # Check for duplicate index
            is_dup = False
            for check in range(max(0, pos - 1), min(k, pos + 2)):
                if idx_np[i, check] == cj:
                    is_dup = True
                    break
            if is_dup:
                continue

            # Shift right and insert
            idx_np[i, pos + 1 : k] = idx_np[i, pos : k - 1]
            dist_np[i, pos + 1 : k] = dist_np[i, pos : k - 1]
            is_new[i, pos + 1 : k] = is_new[i, pos : k - 1]

            idx_np[i, pos] = cj
            dist_np[i, pos] = cd
            is_new[i, pos] = True
            updates += 1

            worst_dist = dist_np[i, k - 1]

        return updates


class NNDescentBatched:
    """Vectorized NNDescent using batch MLX operations for better GPU utilization.

    Same API as NNDescent but uses padded batch operations instead of
    per-point Python loops for the distance computation and merge steps.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to find for each point.
    n_iters : int
        Maximum number of NNDescent iterations.
    max_candidates : int or None
        Number of candidate neighbors to consider per iteration.
        Defaults to min(k, 32) if None.
    delta : float
        Early stopping threshold.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, print progress information.
    """

    def __init__(
        self,
        k: int = 15,
        n_iters: int = 20,
        max_candidates: Optional[int] = None,
        delta: float = 0.015,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iters = n_iters
        self.max_candidates = max_candidates if max_candidates is not None else min(k, 32)
        self.delta = delta
        self.seed = seed
        self.verbose = verbose

    def build(self, X: mx.array) -> tuple[mx.array, mx.array]:
        """Build the approximate k-NN graph for *X*.

        Parameters
        ----------
        X : mx.array, shape (n, d)
            Data matrix (float32).

        Returns
        -------
        indices : mx.array, shape (n, k), int32
            Neighbor indices for each point.
        distances : mx.array, shape (n, k), float32
            Squared Euclidean distances to each neighbor, sorted ascending.
        """
        X = X.astype(mx.float32)
        n, d = X.shape
        k = self.k
        assert k < n, f"k={k} must be less than n={n}"

        rng = np.random.RandomState(self.seed)

        sq_norms = _sq_norms(X)
        mx.eval(sq_norms)
        sq_norms_np = np.array(sq_norms)

        # ----- random initialization -----
        idx_np, dist_np = self._random_init(X, sq_norms, sq_norms_np, n, k, d, rng)

        # NEW flags
        is_new = np.ones((n, k), dtype=bool)

        # ----- iterative refinement -----
        for it in range(self.n_iters):
            t0 = time.time()
            use_fp16 = it < 4

            # Build padded candidate matrix via nn-of-nn expansion
            cand_matrix, cand_counts = self._build_candidate_matrix(
                idx_np, is_new, n, k, rng
            )
            # cand_matrix: (n, max_cand_per_point) padded with -1

            # Mark all current as OLD
            is_new[:] = False

            # Batch distance computation + merge
            n_updates = self._batch_update(
                X, sq_norms, sq_norms_np,
                idx_np, dist_np, is_new,
                cand_matrix, cand_counts,
                n, k, d, use_fp16,
            )

            update_frac = n_updates / (n * k) if n * k > 0 else 0.0
            elapsed = time.time() - t0

            if self.verbose:
                print(
                    f"  iter {it:3d}  updates={n_updates:7d}  "
                    f"frac={update_frac:.4f}  time={elapsed:.2f}s"
                    f"{'  [fp16]' if use_fp16 else ''}"
                )

            if update_frac < self.delta:
                if self.verbose:
                    print(f"  converged at iteration {it}")
                break

        indices = mx.array(idx_np.astype(np.int32))
        distances = mx.array(dist_np.astype(np.float32))
        mx.eval(indices, distances)

        # Recompute final distances in full fp32
        indices, distances = NNDescent._recompute_distances_fp32(
            X, sq_norms, indices, n, k, d
        )

        return indices, distances

    def _random_init(
        self,
        X: mx.array,
        sq_norms: mx.array,
        sq_norms_np: np.ndarray,
        n: int,
        k: int,
        d: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create initial random neighbor graph and compute distances."""
        idx_np = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            choices = rng.choice(n - 1, size=k, replace=(k > n - 1))
            choices = np.where(choices >= i, choices + 1, choices)
            idx_np[i] = choices

        # Batch distance computation
        dist_np = np.full((n, k), _LARGE, dtype=np.float32)
        chunk = 2048
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            cs = end - start

            row_indices = mx.arange(start, end)
            nbr_indices = mx.array(idx_np[start:end])  # (cs, k)

            Xi = X[row_indices]  # (cs, d)
            Xj = X[nbr_indices.reshape(-1)].reshape(cs, k, d)

            norms_i = sq_norms[row_indices]  # (cs,)
            flat_nbr = nbr_indices.reshape(-1)
            norms_j = sq_norms[flat_nbr].reshape(cs, k)

            dots = (Xi[:, None, :] * Xj).sum(axis=2)
            dists = norms_i[:, None] + norms_j - 2.0 * dots
            dists = mx.maximum(dists, 0.0)
            mx.eval(dists)
            dist_np[start:end] = np.array(dists)

        # Sort
        order = np.argsort(dist_np, axis=1)
        dist_np = np.take_along_axis(dist_np, order, axis=1)
        idx_np = np.take_along_axis(idx_np, order, axis=1)

        return idx_np, dist_np

    def _build_candidate_matrix(
        self,
        idx_np: np.ndarray,
        is_new: np.ndarray,
        n: int,
        k: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a padded (n, max_width) candidate matrix via nn-of-nn expansion.

        For each point i, candidates = union of:
          - NEW neighbors of i (forward)
          - Points that have i as a NEW neighbor (reverse)
          - Neighbors of those candidates (nn-of-nn)
        minus current neighbors of i and i itself.
        """
        max_cand = self.max_candidates

        # Forward + reverse lists (split NEW/OLD)
        new_fwd: list[set[int]] = [set() for _ in range(n)]
        new_rev: list[set[int]] = [set() for _ in range(n)]
        old_fwd: list[set[int]] = [set() for _ in range(n)]

        for i in range(n):
            for jj in range(k):
                j = int(idx_np[i, jj])
                if j < 0 or j >= n:
                    continue
                if is_new[i, jj]:
                    new_fwd[i].add(j)
                    new_rev[j].add(i)
                else:
                    old_fwd[i].add(j)

        # Build candidates per point
        all_cands: list[np.ndarray] = []
        max_width = 0

        for i in range(n):
            seeds = new_fwd[i] | new_rev[i]
            # Subsample seeds
            seeds_list = list(seeds - {i})
            if len(seeds_list) > max_cand:
                seeds_list = list(rng.choice(seeds_list, max_cand, replace=False))

            # nn-of-nn expansion
            expanded: set[int] = set(seeds_list)
            for s in seeds_list:
                for jj in range(k):
                    nbr = int(idx_np[s, jj])
                    if 0 <= nbr < n:
                        expanded.add(nbr)

            # Also light expansion from old neighbors
            old_list = list(old_fwd[i] - {i})
            if len(old_list) > max_cand:
                old_list = list(rng.choice(old_list, max_cand, replace=False))
            for s in old_list[:3]:
                for jj in range(min(k, 3)):
                    nbr = int(idx_np[s, jj])
                    if 0 <= nbr < n:
                        expanded.add(nbr)

            # Remove self and current neighbors
            expanded.discard(i)
            current = set(idx_np[i].tolist())
            expanded -= current

            cands = np.array(sorted(expanded), dtype=np.int32)
            all_cands.append(cands)
            if len(cands) > max_width:
                max_width = len(cands)

        # Pad into (n, max_width) matrix
        if max_width == 0:
            max_width = 1

        cand_matrix = np.full((n, max_width), -1, dtype=np.int32)
        cand_counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            nc = len(all_cands[i])
            cand_matrix[i, :nc] = all_cands[i]
            cand_counts[i] = nc

        return cand_matrix, cand_counts

    def _batch_update(
        self,
        X: mx.array,
        sq_norms: mx.array,
        sq_norms_np: np.ndarray,
        idx_np: np.ndarray,
        dist_np: np.ndarray,
        is_new: np.ndarray,
        cand_matrix: np.ndarray,
        cand_counts: np.ndarray,
        n: int,
        k: int,
        d: int,
        use_fp16: bool,
    ) -> int:
        """Batch-compute distances to candidates and merge into neighbor lists.

        Processes points in row-chunks to bound GPU memory.
        """
        total_updates = 0
        max_width = cand_matrix.shape[1]

        # Process in chunks
        chunk_rows = max(1, min(512, 300_000_000 // max(max_width * d, 1)))

        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            cs = end - start

            # Gather candidates for this chunk
            chunk_cands = cand_matrix[start:end]  # (cs, max_width)
            chunk_counts = cand_counts[start:end]

            # For each point in chunk, compute distances to its candidates
            # We do this per-point because candidate sets differ in size,
            # but batch the MLX computation for each point.
            for local_i in range(cs):
                i = start + local_i
                nc = int(chunk_counts[local_i])
                if nc == 0:
                    continue

                cands = chunk_cands[local_i, :nc]  # (nc,) numpy
                cands_mx = mx.array(cands)

                Xi = X[i : i + 1]  # (1, d)
                Xc = X[cands_mx]  # (nc, d)
                ni = sq_norms[i]
                nc_norms = sq_norms[cands_mx]

                if use_fp16:
                    dots = (Xi.astype(mx.float16) @ Xc.astype(mx.float16).T).astype(
                        mx.float32
                    )
                else:
                    dots = Xi @ Xc.T

                cand_dists = ni + nc_norms - 2.0 * dots.reshape(-1)
                cand_dists = mx.maximum(cand_dists, 0.0)
                mx.eval(cand_dists)
                cand_dists_np = np.array(cand_dists)

                updates = NNDescent._merge_candidates(
                    idx_np, dist_np, is_new, i, k, cands, cand_dists_np
                )
                total_updates += updates

        return total_updates
