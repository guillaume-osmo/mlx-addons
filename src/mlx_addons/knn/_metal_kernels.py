"""Custom Metal kernels for KNN hot loops.

Provides GPU-accelerated versions of:
  - Batched pairwise squared L2 distance
  - Per-row top-k merge (for KNN)
  - Threshold mask (for FoF linking)
  - Fused Morton encoding (float-bit + interleave)
"""

from typing import Dict
import mlx.core as mx

# ---------------------------------------------------------------------------
#  Kernel source strings
# ---------------------------------------------------------------------------

_PAIRWISE_L2_SRC = """
    uint i = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (i >= L || j >= M) return;

    float d = 0.0f;
    for (uint d_idx = 0; d_idx < DIM; d_idx++) {
        float diff = qpos[i * DIM + d_idx] - tpos[j * DIM + d_idx];
        d += diff * diff;
    }
    dist2[i * M + j] = d;
"""

_TOPK_MERGE_SRC = """
    uint row = thread_position_in_grid.x;
    if (row >= L) return;

    float r[64];
    int   idx[64];
    for (uint i = 0; i < K; i++) {
        r[i]   = cur_dist[row * K + i];
        idx[i] = cur_ids[row * K + i];
    }

    for (uint j = 0; j < M; j++) {
        float d = new_dist[row * M + j];
        int   id = new_ids[j];
        if (d >= r[K - 1]) continue;
        int pos = K - 1;
        while (pos > 0 && d < r[pos - 1]) {
            r[pos] = r[pos - 1];
            idx[pos] = idx[pos - 1];
            pos--;
        }
        r[pos] = d;
        idx[pos] = id;
    }

    for (uint i = 0; i < K; i++) {
        out_dist[row * K + i] = r[i];
        out_ids[row * K + i]  = idx[i];
    }
"""

_THRESHOLD_MASK_SRC = """
    uint flat_idx = thread_position_in_grid.x;
    if (flat_idx >= TOTAL) return;
    float rlink2_val = as_type<float>(params[0]);
    mask[flat_idx] = (dist2[flat_idx] < rlink2_val) ? (uint8_t)1 : (uint8_t)0;
"""

_MORTON_ENCODE_3D_SRC = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N) return;

    uint ux = as_type<uint>(pos[idx * 3 + 0]);
    uint uy = as_type<uint>(pos[idx * 3 + 1]);
    uint uz = as_type<uint>(pos[idx * 3 + 2]);

    ux = (ux & 0x80000000u) ? ~ux : (ux ^ 0x80000000u);
    uy = (uy & 0x80000000u) ? ~uy : (uy ^ 0x80000000u);
    uz = (uz & 0x80000000u) ? ~uz : (uz ^ 0x80000000u);

    ux >>= 11u; uy >>= 11u; uz >>= 11u;

    auto spread = [](ulong v) -> ulong {
        v = (v | (v << 32)) & 0x001F00000000FFFFull;
        v = (v | (v << 16)) & 0x001F0000FF0000FFull;
        v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
        v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
        v = (v | (v <<  2)) & 0x1249249249249249ull;
        return v;
    };

    codes[idx] = (spread((ulong)ux) << 2) | (spread((ulong)uy) << 1) | spread((ulong)uz);
"""

# Batched version: compute distances for ALL leaf pairs at once.
# Each particle belongs to a query leaf and has an interaction list of target particles.
# We flatten everything into one big distance computation.
_BATCHED_L2_SRC = """
    // Grid: (total_pairs, 1, 1)
    // For pair at index flat_idx:
    //   query particle = query_ids[flat_idx]
    //   target particle = target_ids[flat_idx]
    //   output[flat_idx] = ||pos[q] - pos[t]||^2
    uint flat_idx = thread_position_in_grid.x;
    if (flat_idx >= TOTAL_PAIRS) return;

    int qi = query_ids[flat_idx];
    int ti = target_ids[flat_idx];

    float d = 0.0f;
    for (uint d_idx = 0; d_idx < DIM; d_idx++) {
        float diff = pos[qi * DIM + d_idx] - pos[ti * DIM + d_idx];
        d += diff * diff;
    }
    dist2[flat_idx] = d;
"""



_SEGMENTED_TOPK_SRC = """
    uint row = thread_position_in_grid.x;
    if (row >= Q) return;

    float r[64];
    int idx[64];
    for (uint i = 0; i < K; i++) {
        r[i] = cur_dist[row * K + i];
        idx[i] = cur_ids[row * K + i];
    }

    int start = seg_offsets[row];
    int end = seg_offsets[row + 1];
    for (int p = start; p < end; p++) {
        float d = flat_dist[p];
        int id = target_ids[p];
        if (id == (int)row) continue;
        if (!(d < r[K - 1])) continue;
        int pos = (int)K - 1;
        while (pos > 0 && d < r[pos - 1]) {
            r[pos] = r[pos - 1];
            idx[pos] = idx[pos - 1];
            pos--;
        }
        r[pos] = d;
        idx[pos] = id;
    }

    for (uint i = 0; i < K; i++) {
        out_dist[row * K + i] = r[i];
        out_ids[row * K + i] = idx[i];
    }
"""

# ---------------------------------------------------------------------------
#  Kernel cache
# ---------------------------------------------------------------------------

_KERNEL_CACHE: Dict[str, object] = {}


def _get_kernel(name: str):
    if name not in _KERNEL_CACHE:
        if name == "pairwise_l2":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="pairwise_l2", input_names=["qpos", "tpos"],
                output_names=["dist2"], source=_PAIRWISE_L2_SRC)
        elif name == "topk_merge":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="topk_merge",
                input_names=["new_dist", "new_ids", "cur_dist", "cur_ids"],
                output_names=["out_dist", "out_ids"], source=_TOPK_MERGE_SRC)
        elif name == "threshold_mask":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="threshold_mask", input_names=["dist2", "params"],
                output_names=["mask"], source=_THRESHOLD_MASK_SRC)
        elif name == "morton_encode_3d":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="morton_encode_3d", input_names=["pos"],
                output_names=["codes"], source=_MORTON_ENCODE_3D_SRC)
        elif name == "batched_l2":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="batched_l2",
                input_names=["pos", "query_ids", "target_ids"],
                output_names=["dist2"], source=_BATCHED_L2_SRC)
        elif name == "segmented_topk":
            _KERNEL_CACHE[name] = mx.fast.metal_kernel(
                name="segmented_topk",
                input_names=["flat_dist", "target_ids", "seg_offsets", "cur_dist", "cur_ids"],
                output_names=["out_dist", "out_ids"], source=_SEGMENTED_TOPK_SRC)
        else:
            raise ValueError(f"Unknown kernel: {name}")
    return _KERNEL_CACHE[name]


# ---------------------------------------------------------------------------
#  Python wrappers
# ---------------------------------------------------------------------------

def pairwise_l2_gpu(qpos: mx.array, tpos: mx.array) -> mx.array:
    """(L, M) squared L2 distances between qpos (L, dim) and tpos (M, dim)."""
    L, dim = qpos.shape
    M = tpos.shape[0]
    kernel = _get_kernel("pairwise_l2")
    (dist2,) = kernel(
        inputs=[mx.reshape(qpos, (-1,)), mx.reshape(tpos, (-1,))],
        template=[("L", L), ("M", M), ("DIM", dim)],
        grid=(L, M, 1), threadgroup=(min(16, L), min(16, M), 1),
        output_shapes=[(L * M,)], output_dtypes=[mx.float32])
    return mx.reshape(dist2, (L, M))


def batched_l2_gpu(pos: mx.array, query_ids: mx.array, target_ids: mx.array) -> mx.array:
    """Compute squared L2 distances for arbitrary (qi, ti) pairs.

    Args:
        pos: (N, dim) all positions.
        query_ids: (P,) int32 query particle indices.
        target_ids: (P,) int32 target particle indices.

    Returns:
        (P,) float32 squared distances.
    """
    P = query_ids.shape[0]
    dim = pos.shape[1]
    kernel = _get_kernel("batched_l2")
    (dist2,) = kernel(
        inputs=[mx.reshape(pos, (-1,)), query_ids, target_ids],
        template=[("TOTAL_PAIRS", P), ("DIM", dim)],
        grid=(P, 1, 1), threadgroup=(min(256, P), 1, 1),
        output_shapes=[(P,)], output_dtypes=[mx.float32])
    return dist2


def topk_merge_gpu(new_dist, new_ids, cur_dist, cur_ids, k):
    """Merge new candidates into existing top-k on GPU."""
    L, M = new_dist.shape
    assert k <= 64
    kernel = _get_kernel("topk_merge")
    out_dist, out_ids = kernel(
        inputs=[mx.reshape(new_dist, (-1,)), new_ids,
                mx.reshape(cur_dist, (-1,)), mx.reshape(cur_ids, (-1,))],
        template=[("L", L), ("M", M), ("K", k)],
        grid=(L, 1, 1), threadgroup=(min(256, L), 1, 1),
        output_shapes=[(L * k,), (L * k,)], output_dtypes=[mx.float32, mx.int32])
    return mx.reshape(out_dist, (L, k)), mx.reshape(out_ids, (L, k))


def threshold_mask_gpu(dist2: mx.array, rlink2: float) -> mx.array:
    """GPU threshold: returns uint8 mask where dist2 < rlink2."""
    import struct
    L, M = dist2.shape
    total = L * M
    kernel = _get_kernel("threshold_mask")
    rlink2_bits = struct.unpack("I", struct.pack("f", float(rlink2)))[0]
    params = mx.array([rlink2_bits], dtype=mx.uint32)
    (mask,) = kernel(
        inputs=[mx.reshape(dist2, (-1,)), params],
        template=[("TOTAL", total)],
        grid=(total, 1, 1), threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)], output_dtypes=[mx.uint8])
    return mx.reshape(mask, (L, M))


def morton_encode_gpu(pos: mx.array) -> mx.array:
    """Fused Morton encoding on GPU (3D float32)."""
    N = pos.shape[0]
    assert pos.shape[1] == 3
    kernel = _get_kernel("morton_encode_3d")
    (codes,) = kernel(
        inputs=[mx.reshape(pos, (-1,))],
        template=[("N", N)],
        grid=(N, 1, 1), threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)], output_dtypes=[mx.uint64])
    return codes



def segmented_topk_gpu(flat_dist: mx.array, target_ids: mx.array, seg_offsets: mx.array,
                       cur_dist: mx.array, cur_ids: mx.array, k: int):
    """Segmented per-query top-k on GPU."""
    Q = cur_dist.shape[0]
    assert k <= 64
    kernel = _get_kernel("segmented_topk")
    out_dist, out_ids = kernel(
        inputs=[flat_dist, target_ids, seg_offsets,
                mx.reshape(cur_dist, (-1,)), mx.reshape(cur_ids, (-1,))],
        template=[("Q", Q), ("K", k)],
        grid=(Q, 1, 1), threadgroup=(min(256, Q), 1, 1),
        output_shapes=[(Q * k,), (Q * k,)], output_dtypes=[mx.float32, mx.int32])
    return mx.reshape(out_dist, (Q, k)), mx.reshape(out_ids, (Q, k))
