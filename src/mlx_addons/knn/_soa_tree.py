"""Structure-of-Arrays tree layout for GPU-native traversal.

Everything is a flat numpy array (built on CPU) then converted to MLX
for GPU consumption.  No Python objects per node.

Layout
------
For a tree with L leaves and N total nodes (leaves + internal):

  node_center   : (N, dim) float32  — geometric center
  node_bmin     : (N, dim) float32  — bounding-box min
  node_bmax     : (N, dim) float32  — bounding-box max
  node_is_leaf  : (N,) bool         — True for leaf nodes
  node_child_start : (N,) int32     — first child index (internal) or first particle index (leaf)
  node_child_end   : (N,) int32     — one-past-last child / particle index
  node_npart    : (N,) int32        — total particle count in subtree

Nodes are laid out level-by-level (BFS order): leaves first, then
coarser levels.  Children of node i are contiguous in
[child_start[i], child_end[i]).
"""

import numpy as np
import mlx.core as mx

from ._morton import zsort, morton_encode
from ._config import TreeConfig
from ._data import get_pos


# ---------------------------------------------------------------------------
#  Morton-aware leaf splitting (same as tree.py)
# ---------------------------------------------------------------------------

def _detect_leaves(codes: np.ndarray, npart: int, max_leaf: int) -> np.ndarray:
    if npart <= 0:
        return np.array([0], dtype=np.int32)
    splits = [0]
    i = 0
    while i < npart:
        target = min(i + max_leaf, npart)
        if target >= npart:
            splits.append(npart); break
        half = i + max(max_leaf // 2, 1)
        win_end = target
        best_pos, best_level = target, -1
        for j in range(half, win_end):
            diff = int(codes[j - 1] ^ codes[j])
            if diff > 0:
                lev = diff.bit_length()
                if lev > best_level:
                    best_level = lev; best_pos = j
        splits.append(best_pos); i = best_pos
    return np.array(splits, dtype=np.int32)


# ---------------------------------------------------------------------------
#  Build SoA tree
# ---------------------------------------------------------------------------

class SoATree:
    """Flat Structure-of-Arrays tree for GPU traversal."""

    __slots__ = (
        "num_particles", "dim", "n_nodes", "n_leaves", "n_levels",
        "leaf_splits",
        "center_np", "bmin_np", "bmax_np",
        "is_leaf_np", "child_start_np", "child_end_np", "npart_np",
        "level_offsets_np",
        "_mx_cache",
    )

    def __init__(self, pos_np: np.ndarray, codes: np.ndarray,
                 cfg: TreeConfig = TreeConfig()):
        self.num_particles = pos_np.shape[0]
        self.dim = pos_np.shape[1]
        self._mx_cache = {}

        # --- Leaves ---
        leaf_max = cfg.max_leaf_size
        self.leaf_splits = _detect_leaves(codes, self.num_particles, leaf_max)
        n_leaves = len(self.leaf_splits) - 1
        self.n_leaves = n_leaves

        # --- Leaf-level node properties ---
        ls = self.leaf_splits
        leaf_center = np.zeros((n_leaves, self.dim), np.float32)
        leaf_bmin = np.full((n_leaves, self.dim), np.inf, np.float32)
        leaf_bmax = np.full((n_leaves, self.dim), -np.inf, np.float32)
        leaf_npart = np.zeros(n_leaves, np.int32)

        for i in range(n_leaves):
            s, e = int(ls[i]), int(ls[i + 1])
            if e > s:
                pts = pos_np[s:e]
                leaf_center[i] = np.mean(pts, axis=0)
                leaf_bmin[i] = np.min(pts, axis=0)
                leaf_bmax[i] = np.max(pts, axis=0)
                leaf_npart[i] = e - s

        # --- Build hierarchy bottom-up ---
        # Level 0 = leaves.  Level 1+ = coarser nodes.
        # Each level groups consecutive children.
        levels_center = [leaf_center]
        levels_bmin = [leaf_bmin]
        levels_bmax = [leaf_bmax]
        levels_npart = [leaf_npart]
        levels_child_start = [ls[:-1].copy()]  # leaf -> particle start
        levels_child_end = [ls[1:].copy()]     # leaf -> particle end
        levels_is_leaf = [np.ones(n_leaves, dtype=bool)]

        coarse_fac = cfg.coarse_fac
        current_n = n_leaves
        current_codes = codes[ls[:-1]]  # representative code per leaf

        while current_n > 1:
            max_size = int(current_n / coarse_fac)
            max_size = max(max_size, 1)
            # Group by cumulative count
            group_size = max(int(coarse_fac), 2)
            n_parent = int(np.ceil(current_n / group_size))

            parent_cs = np.minimum(np.arange(n_parent, dtype=np.int32) * group_size, current_n)
            parent_ce = np.minimum(parent_cs + group_size, current_n)

            # Ensure we actually reduce the node count
            if n_parent >= current_n:
                # Force coarsening: group into pairs at minimum
                n_parent = max(int(np.ceil(current_n / 2)), 1)
                parent_cs = np.minimum(np.arange(n_parent, dtype=np.int32) * 2, current_n)
                parent_ce = np.minimum(parent_cs + 2, current_n)

            prev_center = levels_center[-1]
            prev_bmin = levels_bmin[-1]
            prev_bmax = levels_bmax[-1]
            prev_npart = levels_npart[-1]

            p_center = np.zeros((n_parent, self.dim), np.float32)
            p_bmin = np.full((n_parent, self.dim), np.inf, np.float32)
            p_bmax = np.full((n_parent, self.dim), -np.inf, np.float32)
            p_npart = np.zeros(n_parent, np.int32)

            for i in range(n_parent):
                cs, ce = int(parent_cs[i]), int(parent_ce[i])
                if ce > cs:
                    weights = prev_npart[cs:ce].astype(np.float64)
                    total_w = np.sum(weights)
                    if total_w > 0:
                        p_center[i] = np.sum(
                            prev_center[cs:ce] * weights[:, None], axis=0
                        ) / total_w
                    p_bmin[i] = np.min(prev_bmin[cs:ce], axis=0)
                    p_bmax[i] = np.max(prev_bmax[cs:ce], axis=0)
                    p_npart[i] = int(np.sum(prev_npart[cs:ce]))

            # child_start/end for internal nodes point into PREVIOUS level
            # We need global node indices.  Previous level starts at
            # sum of all earlier levels' sizes.
            prev_offset = sum(len(c) for c in levels_center[:-1])
            p_child_start = parent_cs + prev_offset
            p_child_end = parent_ce + prev_offset

            levels_center.append(p_center)
            levels_bmin.append(p_bmin)
            levels_bmax.append(p_bmax)
            levels_npart.append(p_npart)
            levels_child_start.append(p_child_start)
            levels_child_end.append(p_child_end)
            levels_is_leaf.append(np.zeros(n_parent, dtype=bool))

            current_n = n_parent
            current_codes = current_codes[parent_cs]

        self.n_levels = len(levels_center)
        self.n_nodes = sum(len(c) for c in levels_center)

        # --- Concatenate all levels into flat arrays ---
        self.center_np = np.concatenate(levels_center).astype(np.float32)
        self.bmin_np = np.concatenate(levels_bmin).astype(np.float32)
        self.bmax_np = np.concatenate(levels_bmax).astype(np.float32)
        self.is_leaf_np = np.concatenate(levels_is_leaf)
        self.child_start_np = np.concatenate(levels_child_start).astype(np.int32)
        self.child_end_np = np.concatenate(levels_child_end).astype(np.int32)
        self.npart_np = np.concatenate(levels_npart).astype(np.int32)

        # Level boundaries
        sizes = np.array([len(c) for c in levels_center], dtype=np.int32)
        self.level_offsets_np = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)

    # --- MLX accessors (lazy) ---
    def _mx(self, name):
        if name not in self._mx_cache:
            self._mx_cache[name] = mx.array(getattr(self, name + "_np"))
        return self._mx_cache[name]

    @property
    def center(self): return self._mx("center")
    @property
    def bmin(self): return self._mx("bmin")
    @property
    def bmax(self): return self._mx("bmax")
    @property
    def child_start(self): return self._mx("child_start")
    @property
    def child_end(self): return self._mx("child_end")
    @property
    def npart_mx(self): return self._mx("npart")

    def root_range(self):
        """Return (start, end) indices of the top-level (root) nodes."""
        lo = int(self.level_offsets_np[-2])
        hi = int(self.level_offsets_np[-1])
        return lo, hi


# ---------------------------------------------------------------------------
#  High-level: sort + build SoA tree
# ---------------------------------------------------------------------------

def zsort_and_soa_tree(pos, cfg: TreeConfig = TreeConfig()):
    """Sort positions + build SoA tree.

    Returns (pos_sorted_mx, sort_indices_mx, codes_np, tree).
    """
    p = get_pos(pos)
    pos_sorted, indices, codes_np = zsort(p)
    mx.eval(pos_sorted, indices)
    pos_np = np.array(pos_sorted)

    tree = SoATree(pos_np, codes_np, cfg)
    return pos_sorted, indices, codes_np, tree
