# Copyright (c) 2024 Guillaume
# SPDX-License-Identifier: MIT

"""
GPU Metal kernels for batched linear algebra on Apple Silicon.

Implements Cholesky factorization and Cholesky-based linear solve for
batched SPD matrices (k <= 128). Three kernel tiers:

  1. Single-thread (k <= 32): One thread per batch element, device memory.
  2. Threadgroup-cooperative (32 < k <= 80): Shared memory, parallel cols.
  3. Large single-thread (80 < k <= 128): Device memory, y[K] arrays.

Performance: 25-40x faster than CPU LAPACK for batch >= 1000 (tier 1).
"""

import mlx.core as mx

MAX_GPU_K = 128
SHARED_K_MAX = 80  # Max K for threadgroup memory (K*K*4 <= ~25.6 KB)

# ---------------------------------------------------------------------------
# Tier 1: Single-thread kernels (K <= 32)
# ---------------------------------------------------------------------------

_CHOLESKY_SOLVE_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;

uint base = bid * K * K;

for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        L[base + i * K + j] = A[base + i * K + j];

for (uint j = 0; j < K; j++) {
    float s = L[base + j * K + j];
    for (uint p = 0; p < j; p++) {
        float v = L[base + j * K + p];
        s -= v * v;
    }
    L[base + j * K + j] = sqrt(max(s, 1e-12f));
    float inv_ljj = 1.0f / L[base + j * K + j];
    for (uint i = j + 1; i < K; i++) {
        float t = L[base + i * K + j];
        for (uint p = 0; p < j; p++)
            t -= L[base + i * K + p] * L[base + j * K + p];
        L[base + i * K + j] = t * inv_ljj;
    }
}

uint bbase = bid * K * M;
for (uint col = 0; col < M; col++) {
    float y[32];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (uint i = 0; i < K; i++) {
        float t = y[i];
        for (uint p = 0; p < i; p++)
            t -= L[base + i * K + p] * y[p];
        y[i] = t / L[base + i * K + i];
    }
    for (int i = (int)K - 1; i >= 0; i--) {
        float t = y[i];
        for (uint p = (uint)i + 1; p < K; p++)
            t -= L[base + p * K + (uint)i] * y[p];
        y[(uint)i] = t / L[base + (uint)i * K + (uint)i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

_CHOLESKY_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;

uint base = bid * K * K;

for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        out[base + i * K + j] = A[base + i * K + j];

for (uint j = 0; j < K; j++) {
    float s = out[base + j * K + j];
    for (uint p = 0; p < j; p++) {
        float v = out[base + j * K + p];
        s -= v * v;
    }
    out[base + j * K + j] = sqrt(max(s, 1e-12f));
    float inv_ljj = 1.0f / out[base + j * K + j];
    for (uint i = j + 1; i < K; i++) {
        float t = out[base + i * K + j];
        for (uint p = 0; p < j; p++)
            t -= out[base + i * K + p] * out[base + j * K + p];
        out[base + i * K + j] = t * inv_ljj;
    }
    for (uint i = 0; i < j; i++)
        out[base + i * K + j] = 0.0f;
}
"""

_TRIL_SOLVE_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;
uint lbase = bid * K * K;
uint bbase = bid * K * M;
for (uint col = 0; col < M; col++) {
    float y[32];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (uint i = 0; i < K; i++) {
        float t = y[i];
        for (uint p = 0; p < i; p++)
            t -= L[lbase + i * K + p] * y[p];
        y[i] = t / L[lbase + i * K + i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

_TRIU_SOLVE_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;
uint lbase = bid * K * K;
uint bbase = bid * K * M;
for (uint col = 0; col < M; col++) {
    float y[32];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (int i = (int)K - 1; i >= 0; i--) {
        float t = y[(uint)i];
        for (uint p = (uint)i + 1; p < K; p++)
            t -= L[lbase + p * K + (uint)i] * y[p];
        y[(uint)i] = t / L[lbase + (uint)i * K + (uint)i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

# ---------------------------------------------------------------------------
# Tier 2: Threadgroup-cooperative kernels (32 < K <= 80)
# ---------------------------------------------------------------------------

_CHOLESKY_SOLVE_TG_SRC = """
uint bid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint T = threads_per_threadgroup.x;
if (bid >= BATCH) return;

uint abase = bid * K * K;
uint bbase = bid * K * M;

threadgroup float sL[K * K];
for (uint i = tid; i < K * K; i += T)
    sL[i] = A[abase + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint j = 0; j < K; j++) {
    if (tid == 0) {
        float s = sL[j * K + j];
        for (uint p = 0; p < j; p++) {
            float v = sL[j * K + p];
            s -= v * v;
        }
        sL[j * K + j] = sqrt(max(s, 1e-12f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.0f / sL[j * K + j];
    for (uint i = j + 1 + tid; i < K; i += T) {
        float t = sL[i * K + j];
        for (uint p = 0; p < j; p++)
            t -= sL[i * K + p] * sL[j * K + p];
        sL[i * K + j] = t * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint i = tid; i < K * K; i += T) {
    uint row = i / K, col = i % K;
    L[abase + i] = (col <= row) ? sL[i] : 0.0f;
}

for (uint col = tid; col < M; col += T) {
    float y[K];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (uint i = 0; i < K; i++) {
        float t = y[i];
        for (uint p = 0; p < i; p++)
            t -= sL[i * K + p] * y[p];
        y[i] = t / sL[i * K + i];
    }
    for (int i = (int)K - 1; i >= 0; i--) {
        float t = y[(uint)i];
        for (uint p = (uint)i + 1; p < K; p++)
            t -= sL[p * K + (uint)i] * y[p];
        y[(uint)i] = t / sL[(uint)i * K + (uint)i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

_CHOLESKY_TG_SRC = """
uint bid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint T = threads_per_threadgroup.x;
if (bid >= BATCH) return;

uint base = bid * K * K;

threadgroup float sL[K * K];
for (uint i = tid; i < K * K; i += T)
    sL[i] = A[base + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint j = 0; j < K; j++) {
    if (tid == 0) {
        float s = sL[j * K + j];
        for (uint p = 0; p < j; p++) {
            float v = sL[j * K + p];
            s -= v * v;
        }
        sL[j * K + j] = sqrt(max(s, 1e-12f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.0f / sL[j * K + j];
    for (uint i = j + 1 + tid; i < K; i += T) {
        float t = sL[i * K + j];
        for (uint p = 0; p < j; p++)
            t -= sL[i * K + p] * sL[j * K + p];
        sL[i * K + j] = t * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint i = tid; i < K * K; i += T) {
    uint row = i / K, col = i % K;
    out[base + i] = (col <= row) ? sL[i] : 0.0f;
}
"""

_TRIL_SOLVE_TG_SRC = """
uint bid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint T = threads_per_threadgroup.x;
if (bid >= BATCH) return;

uint lbase = bid * K * K;
uint bbase = bid * K * M;

threadgroup float sL[K * K];
for (uint i = tid; i < K * K; i += T)
    sL[i] = L[lbase + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint col = tid; col < M; col += T) {
    float y[K];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (uint i = 0; i < K; i++) {
        float t = y[i];
        for (uint p = 0; p < i; p++)
            t -= sL[i * K + p] * y[p];
        y[i] = t / sL[i * K + i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

_TRIU_SOLVE_TG_SRC = """
uint bid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint T = threads_per_threadgroup.x;
if (bid >= BATCH) return;

uint lbase = bid * K * K;
uint bbase = bid * K * M;

threadgroup float sL[K * K];
for (uint i = tid; i < K * K; i += T)
    sL[i] = L[lbase + i];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint col = tid; col < M; col += T) {
    float y[K];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (int i = (int)K - 1; i >= 0; i--) {
        float t = y[(uint)i];
        for (uint p = (uint)i + 1; p < K; p++)
            t -= sL[p * K + (uint)i] * y[p];
        y[(uint)i] = t / sL[(uint)i * K + (uint)i];
    }
    for (uint i = 0; i < K; i++)
        x[bbase + i * M + col] = y[i];
}
"""

# ---------------------------------------------------------------------------
# Tier 3: Large single-thread kernels (80 < K <= 128)
# Same as tier 1 but float y[K] instead of float y[32].
# ---------------------------------------------------------------------------

_CHOLESKY_SOLVE_LG_SRC = _CHOLESKY_SOLVE_SRC.replace("float y[32]", "float y[K]")
_TRIL_SOLVE_LG_SRC = _TRIL_SOLVE_SRC.replace("float y[32]", "float y[K]")
_TRIU_SOLVE_LG_SRC = _TRIU_SOLVE_SRC.replace("float y[32]", "float y[K]")

# ---------------------------------------------------------------------------
# Householder QR (single-thread, all K)
# ---------------------------------------------------------------------------

_QR_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;

uint base = bid * K * K;
uint tau_base = bid * K;

for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        R[base + i * K + j] = A[base + i * K + j];

for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        Q[base + i * K + j] = (i == j) ? 1.0f : 0.0f;

for (uint j = 0; j < K; j++) {
    float sigma = 0.0f;
    for (uint i = j; i < K; i++) {
        float v = R[base + i * K + j];
        sigma += v * v;
    }
    float norm_x = sqrt(max(sigma, 1e-30f));

    float x_jj = R[base + j * K + j];
    float sign_xjj = (x_jj >= 0.0f) ? 1.0f : -1.0f;
    float v_head = x_jj + sign_xjj * norm_x;

    float vtv = v_head * v_head;
    for (uint i = j + 1; i < K; i++) {
        float vi = R[base + i * K + j];
        vtv += vi * vi;
    }
    float tau = (vtv > 1e-30f) ? (2.0f / vtv) : 0.0f;
    tau_buf[tau_base + j] = tau;

    for (uint row = 0; row < K; row++) {
        float dot = Q[base + row * K + j] * v_head;
        for (uint i = j + 1; i < K; i++)
            dot += Q[base + row * K + i] * R[base + i * K + j];
        Q[base + row * K + j] -= tau * dot * v_head;
        for (uint i = j + 1; i < K; i++)
            Q[base + row * K + i] -= tau * dot * R[base + i * K + j];
    }

    for (uint col = j + 1; col < K; col++) {
        float dot = v_head * R[base + j * K + col];
        for (uint i = j + 1; i < K; i++)
            dot += R[base + i * K + j] * R[base + i * K + col];
        R[base + j * K + col] -= tau * v_head * dot;
        for (uint i = j + 1; i < K; i++)
            R[base + i * K + col] -= tau * R[base + i * K + j] * dot;
    }

    R[base + j * K + j] = -sign_xjj * norm_x;
    for (uint i = j + 1; i < K; i++)
        R[base + i * K + j] = 0.0f;
}
"""

# ---------------------------------------------------------------------------
# Threadgroup-cooperative QR (K >= 12, 2*K*K*4 < 32 KB → K <= 56)
# Parallelises reflector application across rows (Q) and columns (R).
# ---------------------------------------------------------------------------

QR_SHARED_K_MAX = 63  # 2*63*63*4 + 16 ≈ 31.8 KB < 32 KB

_QR_TG_SRC = """
uint bid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint T = threads_per_threadgroup.x;
if (bid >= BATCH) return;

uint base = bid * K * K;

threadgroup float sR[K * K];
threadgroup float sQ[K * K];
threadgroup float sInfo[4];

for (uint i = tid; i < K * K; i += T)
    sR[i] = A[base + i];
for (uint i = tid; i < K * K; i += T) {
    uint row = i / K, col = i % K;
    sQ[i] = (row == col) ? 1.0f : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint j = 0; j < K; j++) {
    // 1. Householder vector (thread 0)
    if (tid == 0) {
        float sigma = 0.0f;
        for (uint i = j; i < K; i++) {
            float v = sR[i * K + j];
            sigma += v * v;
        }
        float nx = sqrt(max(sigma, 1e-30f));
        float x_jj = sR[j * K + j];
        float sgn = (x_jj >= 0.0f) ? 1.0f : -1.0f;
        float vh = x_jj + sgn * nx;
        float vtv = vh * vh;
        for (uint i = j + 1; i < K; i++) {
            float vi = sR[i * K + j];
            vtv += vi * vi;
        }
        float t = (vtv > 1e-30f) ? (2.0f / vtv) : 0.0f;
        sInfo[0] = t;   sInfo[1] = vh;
        sInfo[2] = sgn;  sInfo[3] = nx;
        tau_buf[bid * K + j] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tau = sInfo[0];
    float v_head = sInfo[1];
    float sign_xjj = sInfo[2];
    float norm_x = sInfo[3];

    // 2. Apply reflector to Q — parallelize across rows
    for (uint row = tid; row < K; row += T) {
        float dot = sQ[row * K + j] * v_head;
        for (uint i = j + 1; i < K; i++)
            dot += sQ[row * K + i] * sR[i * K + j];
        sQ[row * K + j] -= tau * dot * v_head;
        for (uint i = j + 1; i < K; i++)
            sQ[row * K + i] -= tau * dot * sR[i * K + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Apply reflector to R cols j+1..K-1 — parallelize across cols
    for (uint col = j + 1 + tid; col < K; col += T) {
        float dot = v_head * sR[j * K + col];
        for (uint i = j + 1; i < K; i++)
            dot += sR[i * K + j] * sR[i * K + col];
        sR[j * K + col] -= tau * v_head * dot;
        for (uint i = j + 1; i < K; i++)
            sR[i * K + col] -= tau * sR[i * K + j] * dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Set R column j
    if (tid == 0) {
        sR[j * K + j] = -sign_xjj * norm_x;
        for (uint i = j + 1; i < K; i++)
            sR[i * K + j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint i = tid; i < K * K; i += T)
    Q[base + i] = sQ[i];
for (uint i = tid; i < K * K; i += T)
    R[base + i] = sR[i];
"""

# ---------------------------------------------------------------------------
# Kernel cache & dispatch
# ---------------------------------------------------------------------------

_KERNEL_CACHE = {}


def _get_kernel(name, input_names, output_names, source):
    if name not in _KERNEL_CACHE:
        _KERNEL_CACHE[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )
    return _KERNEL_CACHE[name]


def _prep_b(b):
    if b.ndim == 2:
        b_3d = mx.expand_dims(b, -1)
        was_2d = True
    else:
        b_3d = b
        was_2d = False
    return mx.reshape(b_3d.astype(mx.float32), (-1,)), b_3d.shape[-1], was_2d


def _unpack(x_flat, batch, k, m, was_2d):
    x = mx.reshape(x_flat, (batch, k, m))
    return mx.squeeze(x, axis=-1) if was_2d else x


def _grid_tg(k, batch):
    """Return (grid, threadgroup) for the appropriate tier."""
    if k <= 32:
        return (batch, 1, 1), (min(256, batch), 1, 1)
    elif k <= SHARED_K_MAX:
        tg_size = min(k, 64)
        return (batch * tg_size, 1, 1), (tg_size, 1, 1)
    else:
        return (batch, 1, 1), (min(256, batch), 1, 1)


def _pick_src(k, src_st, src_tg, src_lg):
    if k <= 32:
        return src_st
    elif k <= SHARED_K_MAX:
        return src_tg
    return src_lg


def _tier_tag(k):
    if k <= 32:
        return ""
    elif k <= SHARED_K_MAX:
        return "_tg"
    return "_lg"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(A: mx.array, b: mx.array) -> mx.array:
    """Solve A @ x = b for batched SPD matrices (k <= 128) on GPU."""
    batch, k = A.shape[0], A.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)

    tag = _tier_tag(k)
    src = _pick_src(k, _CHOLESKY_SOLVE_SRC, _CHOLESKY_SOLVE_TG_SRC,
                    _CHOLESKY_SOLVE_LG_SRC)
    kernel = _get_kernel(
        f"cholesky_solve{tag}_{k}_{m}", ["A", "b"], ["L", "x"], src,
    )
    grid, tg = _grid_tg(k, batch)
    _, x_flat = kernel(
        inputs=[A_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(batch * k * k,), (batch * k * m,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def cholesky(A: mx.array) -> mx.array:
    """Cholesky factorization for batched SPD matrices (k <= 128) on GPU."""
    batch, k = A.shape[0], A.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))

    if k <= 32 or k > SHARED_K_MAX:
        src = _CHOLESKY_SRC
        tag = "" if k <= 32 else "_lg"
        grid = (batch, 1, 1)
        threadgroup = (min(256, batch), 1, 1)
    else:
        src = _CHOLESKY_TG_SRC
        tag = "_tg"
        tg_size = min(k, 64)
        grid = (batch * tg_size, 1, 1)
        threadgroup = (tg_size, 1, 1)

    kernel = _get_kernel(f"cholesky{tag}_{k}", ["A"], ["out"], src)
    (L_flat,) = kernel(
        inputs=[A_flat],
        template=[("K", k), ("BATCH", batch)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(batch * k * k,)],
        output_dtypes=[mx.float32],
    )
    return mx.reshape(L_flat, (batch, k, k))


def tril_solve(L: mx.array, b: mx.array) -> mx.array:
    """Solve L @ x = b where L is lower triangular (k <= 128) on GPU."""
    batch, k = L.shape[0], L.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"
    L_flat = mx.reshape(L.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)

    tag = _tier_tag(k)
    src = _pick_src(k, _TRIL_SOLVE_SRC, _TRIL_SOLVE_TG_SRC, _TRIL_SOLVE_LG_SRC)
    kernel = _get_kernel(
        f"tril_solve{tag}_{k}_{m}", ["L", "b"], ["x"], src,
    )
    grid, tg = _grid_tg(k, batch)
    (x_flat,) = kernel(
        inputs=[L_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(batch * k * m,)],
        output_dtypes=[mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def triu_solve(L: mx.array, b: mx.array) -> mx.array:
    """Solve L^T @ x = b where L is lower triangular (k <= 128) on GPU."""
    batch, k = L.shape[0], L.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"
    L_flat = mx.reshape(L.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)

    tag = _tier_tag(k)
    src = _pick_src(k, _TRIU_SOLVE_SRC, _TRIU_SOLVE_TG_SRC, _TRIU_SOLVE_LG_SRC)
    kernel = _get_kernel(
        f"triu_solve{tag}_{k}_{m}", ["L", "b"], ["x"], src,
    )
    grid, tg = _grid_tg(k, batch)
    (x_flat,) = kernel(
        inputs=[L_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(batch * k * m,)],
        output_dtypes=[mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def qr(A: mx.array) -> tuple:
    """QR factorization for batched square matrices on GPU via Householder."""
    squeeze = False
    if A.ndim == 2:
        A = mx.expand_dims(A, 0)
        squeeze = True

    batch, k = A.shape[0], A.shape[1]
    assert A.shape[2] == k, f"qr requires square matrices, got {A.shape}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))

    if k >= 12 and k <= QR_SHARED_K_MAX:
        # Threadgroup cooperative — fast shared memory + parallel reflectors
        kernel = _get_kernel(
            f"qr_tg_{k}", ["A"], ["Q", "R", "tau_buf"], _QR_TG_SRC,
        )
        tg_size = min(k, 64)
        Q_flat, R_flat, _ = kernel(
            inputs=[A_flat],
            template=[("K", k), ("BATCH", batch)],
            grid=(batch * tg_size, 1, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(batch * k * k,), (batch * k * k,), (batch * k,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        Q = mx.reshape(Q_flat, (batch, k, k))
        R = mx.reshape(R_flat, (batch, k, k))
    elif k < 12:
        # Single-thread Metal kernel — good for tiny K
        kernel = _get_kernel(
            f"qr_{k}", ["A"], ["Q", "R", "tau_buf"], _QR_SRC,
        )
        tg = min(256, batch)
        Q_flat, R_flat, _ = kernel(
            inputs=[A_flat],
            template=[("K", k), ("BATCH", batch)],
            grid=(batch, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(batch * k * k,), (batch * k * k,), (batch * k,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        Q = mx.reshape(Q_flat, (batch, k, k))
        R = mx.reshape(R_flat, (batch, k, k))
    else:
        # k > QR_SHARED_K_MAX: fall back to CPU LAPACK (faster than
        # column-by-column MLX ops for large k)
        Q, R = mx.linalg.qr(A.astype(mx.float32), stream=mx.cpu)

    if squeeze:
        Q = mx.squeeze(Q, axis=0)
        R = mx.squeeze(R, axis=0)

    return Q, R
