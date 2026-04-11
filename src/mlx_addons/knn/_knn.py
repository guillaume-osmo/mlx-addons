"""KNN v6 — v5 plus GPU segmented top-k kernel.

Execution model:
  CPU builds tree (SoA).
  GPU does:
    1. Frontier-based tree walk (bulk bbox distance per level).
    2. Flatten leaf pairs -> per-particle candidate segments.
    3. Fused distance + register-heap top-k (one thread per query particle).
  CPU only launches passes, never touches individual nodes or particles.
"""

import mlx.core as mx
import numpy as np

from ._config import KNNConfig
from ._data import get_pos
from ._soa_tree import zsort_and_soa_tree, SoATree
from ._gpu_kernels import bbox_dist_gpu
from ._metal_kernels import batched_l2_gpu, segmented_topk_gpu



def _expand_pair_children(csa: np.ndarray, cea: np.ndarray,
                          csb: np.ndarray, ceb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized cartesian expansion of child ranges for paired nodes.

    For each parent pair i, emit children(a_i) x children(b_i) without a Python
    per-pair loop. Returns two int32 arrays of equal length.
    """
    la = (cea - csa).astype(np.int64, copy=False)
    lb = (ceb - csb).astype(np.int64, copy=False)
    pair_counts = la * lb
    total = int(pair_counts.sum())
    if total == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)

    grp = np.repeat(np.arange(len(csa), dtype=np.int32), pair_counts.astype(np.int32, copy=False))
    starts = np.cumsum(pair_counts, dtype=np.int64) - pair_counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, pair_counts.astype(np.int32, copy=False))

    lb_grp = lb[grp]
    a = csa[grp].astype(np.int64, copy=False) + (within // lb_grp)
    b = csb[grp].astype(np.int64, copy=False) + (within % lb_grp)
    return a.astype(np.int32, copy=False), b.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
#  Frontier-based tree walk (GPU bbox distance, CPU filtering)
# ---------------------------------------------------------------------------

def _frontier_walk(tree: SoATree, boxsize: float = 0.0) -> np.ndarray:
    """Descend tree via frontier of node pairs, return leaf-level pairs.

    Returns (L, 2) int32 array of (leaf_i, leaf_j) pairs.
    """
    dim = tree.dim
    root_lo, root_hi = tree.root_range()
    n_roots = root_hi - root_lo

    # Initial frontier: all-to-all among root nodes
    ra = np.repeat(np.arange(root_lo, root_hi, dtype=np.int32), n_roots)
    rb = np.tile(np.arange(root_lo, root_hi, dtype=np.int32), n_roots)
    frontier_a = ra
    frontier_b = rb

    leaf_pairs_a = []
    leaf_pairs_b = []

    is_leaf = tree.is_leaf_np
    cs_np = tree.child_start_np
    ce_np = tree.child_end_np
    npart_np = tree.npart_np
    bmin_np = tree.bmin_np
    bmax_np = tree.bmax_np

    while len(frontier_a) > 0:
        P = len(frontier_a)

        # GPU: compute bbox min-distance² for all pairs
        d2 = bbox_dist_gpu(
            mx.array(frontier_a), mx.array(frontier_b),
            mx.array(bmin_np.reshape(-1)), mx.array(bmax_np.reshape(-1)),
            dim)
        mx.eval(d2)
        d2_np = np.array(d2)

        # Filter: non-zero npart + distance budget pruning
        valid = (npart_np[frontier_a] > 0) & (npart_np[frontier_b] > 0)
        frontier_a = frontier_a[valid]
        frontier_b = frontier_b[valid]
        d2_np = d2_np[valid]

        # Budget pruning: for each receiving node, keep only the
        # N_BUDGET closest source nodes (by bbox distance).
        # This is what makes the tree walk O(N log N) instead of O(N²).
        BUDGET = 64
        if len(frontier_a) > 0:
            # Sort by (recv_node, distance), then keep only the first BUDGET
            # entries inside each recv-node group without a Python group loop.
            order = np.lexsort((d2_np, frontier_a))
            fa_s = frontier_a[order]
            fb_s = frontier_b[order]

            _, first_idx, counts = np.unique(fa_s, return_index=True, return_counts=True)
            grp_starts = np.repeat(first_idx, counts)
            rank_in_grp = np.arange(len(fa_s), dtype=np.int32) - grp_starts
            keep_mask = rank_in_grp < BUDGET

            frontier_a = fa_s[keep_mask]
            frontier_b = fb_s[keep_mask]

        if len(frontier_a) == 0:
            break

        # Classify: both-leaf, one-leaf, neither-leaf
        a_leaf = is_leaf[frontier_a]
        b_leaf = is_leaf[frontier_b]

        both_leaf = a_leaf & b_leaf
        neither_leaf = ~a_leaf & ~b_leaf
        a_only_leaf = a_leaf & ~b_leaf
        b_only_leaf = ~a_leaf & b_leaf

        # Emit leaf pairs
        if np.any(both_leaf):
            leaf_pairs_a.append(frontier_a[both_leaf])
            leaf_pairs_b.append(frontier_b[both_leaf])

        # Expand: for non-leaf pairs, expand into children
        # Use vectorised numpy — no Python per-pair loops.
        next_a = []
        next_b = []

        def _expand_one_side(fa, fb, expand_fa):
            """Expand fa -> children, keep fb fixed (or vice versa)."""
            if len(fa) == 0: return
            if expand_fa:
                src, fixed = fa, fb
            else:
                src, fixed = fb, fa
            cstarts = cs_np[src]
            cends = ce_np[src]
            clens = cends - cstarts
            total = int(np.sum(clens))
            if total == 0: return
            # Vectorised range expansion
            offsets = np.repeat(cstarts - np.concatenate([[0], np.cumsum(clens)[:-1]]), clens)
            children = (np.arange(total, dtype=np.int32) + offsets)
            fixed_expanded = np.repeat(fixed, clens)
            if expand_fa:
                next_a.append(children); next_b.append(fixed_expanded)
            else:
                next_a.append(fixed_expanded); next_b.append(children)

        def _expand_both(fa, fb):
            """Expand both sides: each (a,b) -> children(a) x children(b)."""
            if len(fa) == 0:
                return
            ea, eb = _expand_pair_children(cs_np[fa], ce_np[fa], cs_np[fb], ce_np[fb])
            if len(ea) > 0:
                next_a.append(ea)
                next_b.append(eb)

        _expand_both(frontier_a[neither_leaf], frontier_b[neither_leaf])
        _expand_one_side(frontier_a[a_only_leaf], frontier_b[a_only_leaf], False)   # expand b
        _expand_one_side(frontier_a[b_only_leaf], frontier_b[b_only_leaf], True)    # expand a

        if next_a:
            frontier_a = np.concatenate(next_a)
            frontier_b = np.concatenate(next_b)
        else:
            break

    if leaf_pairs_a:
        lp_a = np.concatenate(leaf_pairs_a)
        lp_b = np.concatenate(leaf_pairs_b)
        return np.stack([lp_a, lp_b], axis=1)
    else:
        return np.zeros((0, 2), dtype=np.int32)


# ---------------------------------------------------------------------------
#  Flatten leaf pairs -> per-particle candidate segments
# ---------------------------------------------------------------------------

def _build_segments(leaf_pairs: np.ndarray, leaf_splits: np.ndarray,
                    npart: int) -> tuple:
    """Convert leaf pairs to CSR-style candidate segments.

    v5 uses scatter-like range accumulation for segment sizing:
      diff[q_start] += target_leaf_size
      diff[q_end]   -= target_leaf_size
    followed by a cumsum. That removes the per-query slice assignment from the
    sizing pass. Materialization still writes one contiguous block per unique
    query leaf.
    """
    if len(leaf_pairs) == 0:
        return np.array([], np.int32), np.zeros(npart + 1, np.int32)

    lp_a = leaf_pairs[:, 0].astype(np.int32, copy=False)
    lp_b = leaf_pairs[:, 1].astype(np.int32, copy=False)

    q_starts = leaf_splits[lp_a].astype(np.int32, copy=False)
    q_ends = leaf_splits[lp_a + 1].astype(np.int32, copy=False)
    t_starts = leaf_splits[lp_b].astype(np.int32, copy=False)
    t_counts = (leaf_splits[lp_b + 1] - t_starts).astype(np.int32, copy=False)

    diff = np.zeros(npart + 1, np.int64)
    np.add.at(diff, q_starts, t_counts.astype(np.int64, copy=False))
    np.add.at(diff, q_ends, -t_counts.astype(np.int64, copy=False))
    seg_sizes = np.cumsum(diff[:-1], dtype=np.int64).astype(np.int32, copy=False)

    seg_offsets = np.empty(npart + 1, np.int32)
    seg_offsets[0] = 0
    np.cumsum(seg_sizes, out=seg_offsets[1:])
    total_targets = int(seg_offsets[-1])
    if total_targets == 0:
        return np.array([], np.int32), seg_offsets

    order = np.argsort(lp_a, kind="stable")
    lp_a = lp_a[order]
    lp_b = lp_b[order]
    unique_leaves, first_idx, counts = np.unique(lp_a, return_index=True, return_counts=True)

    target_ids = np.empty(total_targets, np.int32)
    write_pos = 0
    for g, qleaf in enumerate(unique_leaves.tolist()):
        s = int(first_idx[g]); e = s + int(counts[g])
        t_leaves = lp_b[s:e]
        starts = leaf_splits[t_leaves].astype(np.int32, copy=False)
        lens = (leaf_splits[t_leaves + 1] - starts).astype(np.int32, copy=False)
        target_size = int(lens.sum())
        if target_size == 0:
            continue

        grp_starts = np.cumsum(lens, dtype=np.int64) - lens
        offsets = np.repeat(starts.astype(np.int64, copy=False) - grp_starts, lens)
        leaf_targets = (np.arange(target_size, dtype=np.int64) + offsets).astype(np.int32, copy=False)

        qs = int(leaf_splits[qleaf]); qe = int(leaf_splits[qleaf + 1])
        qcount = qe - qs
        if qcount <= 0:
            continue

        block = qcount * target_size
        target_ids[write_pos:write_pos + block] = np.tile(leaf_targets, qcount)
        write_pos += block

    return target_ids, seg_offsets


# ---------------------------------------------------------------------------
#  Fused KNN evaluation (GPU)
# ---------------------------------------------------------------------------

def _knn_fused(pos_mx: mx.array, target_ids_np: np.ndarray,
               seg_offsets_np: np.ndarray, npart: int, k: int) -> tuple:
    """Run segmented KNN in bulk with GPU top-k selection.

    v6 keeps v5's flat candidate construction, computes candidate distances in
    one bulk pass, then runs a segmented Metal kernel with one thread per query
    row to maintain the best k neighbors in registers.
    """
    counts = np.diff(seg_offsets_np).astype(np.int32, copy=False)
    total = int(counts.sum())
    if total == 0:
        return (mx.full((npart, k), float("inf"), dtype=mx.float32),
                mx.full((npart, k), -1, dtype=mx.int32))

    qids_np = np.repeat(np.arange(npart, dtype=np.int32), counts)
    tids_np = target_ids_np.astype(np.int32, copy=False)

    if total >= 100_000:
        d2 = batched_l2_gpu(pos_mx, mx.array(qids_np), mx.array(tids_np))
    else:
        pos_np = np.array(pos_mx)
        dx = pos_np[qids_np] - pos_np[tids_np]
        d2 = mx.array(np.sum(dx * dx, axis=-1).astype(np.float32))

    cur_dist = mx.full((npart, k), float("inf"), dtype=mx.float32)
    cur_ids = mx.full((npart, k), -1, dtype=mx.int32)
    out_d2, out_ids = segmented_topk_gpu(
        d2,
        mx.array(tids_np),
        mx.array(seg_offsets_np.astype(np.int32, copy=False)),
        cur_dist,
        cur_ids,
        k,
    )
    return mx.sqrt(out_d2), out_ids


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def knn_v6(pos, k: int, boxsize: float = 0.0, cfg: KNNConfig = KNNConfig(),
           result: str = "rad_idx"):
    """KNN v6 — v2 KNN core + v3 frontier expansion helper.

    Args:
        pos: (N, dim) positions.
        k: number of neighbors (max 64).
        boxsize: periodic (0 = non-periodic, periodic not yet GPU-accelerated).
        result: ``"rad"``, ``"idx"``, or ``"rad_idx"``.
    """
    p = get_pos(pos)
    npart = p.shape[0]

    pos_sorted, sort_indices, _, tree = zsort_and_soa_tree(p, cfg.tree)
    mx.eval(pos_sorted, sort_indices)

    # Step 1: frontier-based tree walk -> leaf pairs
    leaf_pairs = _frontier_walk(tree, boxsize)

    # Step 2: flatten -> per-particle candidate segments
    target_ids, seg_offsets = _build_segments(leaf_pairs, tree.leaf_splits, npart)

    # Step 3: fused KNN evaluation
    rnn_z, inn_z = _knn_fused(pos_sorted, target_ids, seg_offsets, npart, k)

    # Map back to input order
    sort_np = np.array(sort_indices)
    inn_z_np = np.array(inn_z)
    inn_orig = np.where(inn_z_np >= 0, sort_np[inn_z_np], -1)
    inv = np.empty(npart, np.int32)
    inv[sort_np] = np.arange(npart, dtype=np.int32)

    rnn = mx.array(np.array(rnn_z)[inv])
    inn = mx.array(inn_orig[inv])

    res = []
    for key in result.split("_"):
        if key == "rad": res.append(rnn)
        elif key == "idx": res.append(inn)
        else: raise ValueError(f"Unknown result key: {key}")
    return tuple(res) if len(res) > 1 else res[0]
