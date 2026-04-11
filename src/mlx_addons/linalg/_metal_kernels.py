# Copyright (c) 2024 Guillaume
# SPDX-License-Identifier: MIT

"""
GPU Metal kernels for batched linear algebra on Apple Silicon.

Implements Cholesky factorization and Cholesky-based linear solve for
batched small SPD matrices (k <= 32). One thread per batch element.

Uses device-memory scratch buffers instead of thread-local arrays to
work around a Metal compiler bug with certain array sizes (K=6-9).

Performance: 25-40x faster than CPU LAPACK for batch >= 1000.
"""

import mlx.core as mx

MAX_GPU_K = 32

# ---------------------------------------------------------------------------
# Metal shader sources — device-memory variants
# ---------------------------------------------------------------------------

# Fused Cholesky + solve. Uses 'L' output as scratch for factorization.
_CHOLESKY_SOLVE_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;

uint base = bid * K * K;

// Copy A into scratch (L buffer)
for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        L[base + i * K + j] = A[base + i * K + j];

// Cholesky factorization in-place on L
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

// Solve for each RHS column using L
uint bbase = bid * K * M;
for (uint col = 0; col < M; col++) {
    // Forward substitution: L @ y = b
    float y[32];
    for (uint i = 0; i < K; i++)
        y[i] = b[bbase + i * M + col];
    for (uint i = 0; i < K; i++) {
        float t = y[i];
        for (uint p = 0; p < i; p++)
            t -= L[base + i * K + p] * y[p];
        y[i] = t / L[base + i * K + i];
    }
    // Back substitution: L^T @ x = y
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
    // Zero upper triangle
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

# Householder QR factorization.  Produces Q (orthogonal) and R (upper tri).
# Applies reflectors to Q *before* zeroing R column — avoids clobbering
# the Householder vector stored in R's subdiagonal.
_QR_SRC = """
uint bid = thread_position_in_grid.x;
if (bid >= BATCH) return;

uint base = bid * K * K;
uint tau_base = bid * K;

// Copy A into R
for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        R[base + i * K + j] = A[base + i * K + j];

// Q = I
for (uint i = 0; i < K; i++)
    for (uint j = 0; j < K; j++)
        Q[base + i * K + j] = (i == j) ? 1.0f : 0.0f;

for (uint j = 0; j < K; j++) {
    // 1. Column norm from diagonal down
    float sigma = 0.0f;
    for (uint i = j; i < K; i++) {
        float v = R[base + i * K + j];
        sigma += v * v;
    }
    float norm_x = sqrt(max(sigma, 1e-30f));

    // 2. Householder vector: v_head = x[j,j] + sign*norm, v[i>j] = R[i,j]
    float x_jj = R[base + j * K + j];
    float sign_xjj = (x_jj >= 0.0f) ? 1.0f : -1.0f;
    float v_head = x_jj + sign_xjj * norm_x;

    // 3. tau = 2 / (v^T v)
    float vtv = v_head * v_head;
    for (uint i = j + 1; i < K; i++) {
        float vi = R[base + i * K + j];
        vtv += vi * vi;
    }
    float tau = (vtv > 1e-30f) ? (2.0f / vtv) : 0.0f;
    tau_buf[tau_base + j] = tau;

    // 4. Apply to Q FIRST (R column j subdiag still has v)
    for (uint row = 0; row < K; row++) {
        float dot = Q[base + row * K + j] * v_head;
        for (uint i = j + 1; i < K; i++)
            dot += Q[base + row * K + i] * R[base + i * K + j];
        Q[base + row * K + j] -= tau * dot * v_head;
        for (uint i = j + 1; i < K; i++)
            Q[base + row * K + i] -= tau * dot * R[base + i * K + j];
    }

    // 5. Apply to R columns j+1..K-1
    for (uint col = j + 1; col < K; col++) {
        float dot = v_head * R[base + j * K + col];
        for (uint i = j + 1; i < K; i++)
            dot += R[base + i * K + j] * R[base + i * K + col];
        R[base + j * K + col] -= tau * v_head * dot;
        for (uint i = j + 1; i < K; i++)
            R[base + i * K + col] -= tau * R[base + i * K + j] * dot;
    }

    // 6. Set R column j directly
    R[base + j * K + j] = -sign_xjj * norm_x;
    for (uint i = j + 1; i < K; i++)
        R[base + i * K + j] = 0.0f;
}
"""


# ---------------------------------------------------------------------------
# Kernel cache
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(A: mx.array, b: mx.array) -> mx.array:
    """
    Solve A @ x = b for batched SPD matrices using Cholesky on GPU.

    Args:
        A: (batch, k, k) symmetric positive definite matrices
        b: (batch, k, m) or (batch, k) right-hand sides

    Returns:
        x: same shape as b
    """
    batch, k = A.shape[0], A.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)

    kernel = _get_kernel(
        f"cholesky_solve_{k}_{m}", ["A", "b"], ["L", "x"], _CHOLESKY_SOLVE_SRC,
    )
    tg = min(256, batch)
    _, x_flat = kernel(
        inputs=[A_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=(batch, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(batch * k * k,), (batch * k * m,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def cholesky(A: mx.array) -> mx.array:
    """
    Cholesky factorization for batched SPD matrices on GPU.

    Args:
        A: (batch, k, k) symmetric positive definite matrices

    Returns:
        L: (batch, k, k) lower triangular Cholesky factors
    """
    batch, k = A.shape[0], A.shape[1]
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))
    kernel = _get_kernel(f"cholesky_{k}", ["A"], ["out"], _CHOLESKY_SRC)
    tg = min(256, batch)
    (L_flat,) = kernel(
        inputs=[A_flat],
        template=[("K", k), ("BATCH", batch)],
        grid=(batch, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(batch * k * k,)],
        output_dtypes=[mx.float32],
    )
    return mx.reshape(L_flat, (batch, k, k))


def tril_solve(L: mx.array, b: mx.array) -> mx.array:
    """
    Solve L @ x = b where L is lower triangular, on GPU.

    Args:
        L: (batch, k, k) lower triangular matrices
        b: (batch, k, m) or (batch, k) right-hand sides

    Returns:
        x: same shape as b
    """
    batch, k = L.shape[0], L.shape[1]
    assert k <= MAX_GPU_K
    L_flat = mx.reshape(L.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)
    kernel = _get_kernel(
        f"tril_solve_{k}_{m}", ["L", "b"], ["x"], _TRIL_SOLVE_SRC,
    )
    tg = min(256, batch)
    (x_flat,) = kernel(
        inputs=[L_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=(batch, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(batch * k * m,)],
        output_dtypes=[mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def triu_solve(L: mx.array, b: mx.array) -> mx.array:
    """
    Solve L^T @ x = b where L is lower triangular, on GPU.

    Args:
        L: (batch, k, k) lower triangular matrices
        b: (batch, k, m) or (batch, k) right-hand sides

    Returns:
        x: same shape as b
    """
    batch, k = L.shape[0], L.shape[1]
    assert k <= MAX_GPU_K
    L_flat = mx.reshape(L.astype(mx.float32), (-1,))
    b_flat, m, was_2d = _prep_b(b)
    kernel = _get_kernel(
        f"triu_solve_{k}_{m}", ["L", "b"], ["x"], _TRIU_SOLVE_SRC,
    )
    tg = min(256, batch)
    (x_flat,) = kernel(
        inputs=[L_flat, b_flat],
        template=[("K", k), ("M", m), ("BATCH", batch)],
        grid=(batch, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(batch * k * m,)],
        output_dtypes=[mx.float32],
    )
    return _unpack(x_flat, batch, k, m, was_2d)


def qr(A: mx.array) -> tuple:
    """
    QR factorization for batched square matrices on GPU via Householder.

    25-50x faster than CPU ``mx.linalg.qr`` for batched small matrices.

    Args:
        A: (batch, k, k) or (k, k) square matrices.  k <= MAX_GPU_K (32).

    Returns:
        (Q, R): Q orthogonal, R upper triangular, same leading shape as A.
    """
    squeeze = False
    if A.ndim == 2:
        A = mx.expand_dims(A, 0)
        squeeze = True

    batch, k = A.shape[0], A.shape[1]
    assert A.shape[2] == k, f"qr requires square matrices, got {A.shape}"
    assert k <= MAX_GPU_K, f"Matrix size {k} exceeds GPU limit {MAX_GPU_K}"

    A_flat = mx.reshape(A.astype(mx.float32), (-1,))

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

    if squeeze:
        Q = mx.squeeze(Q, axis=0)
        R = mx.squeeze(R, axis=0)

    return Q, R
