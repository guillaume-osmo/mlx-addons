"""
GPU-accelerated batched linear algebra for MLX.

Metal GPU kernels for Cholesky, solve, QR, and determinant on batched matrices
of **any size**. Small matrices (k <= 128) use direct Metal kernels with three tiers:
per-thread (k<=32), threadgroup-cooperative (33-80), large per-thread (81-128).
Larger matrices use a blocked algorithm with GPU matmul for SYRK updates.

Usage::

    import mlx.core as mx
    from mlx_addons.linalg import solve, cholesky, qr, det, slogdet

    A = mx.array(...)  # (batch, n, n) SPD matrices — any n
    b = mx.array(...)  # (batch, n, m) right-hand sides
    x = solve(A, b)    # GPU-accelerated solve

    Q, R = qr(A)       # Householder QR on GPU (k <= 32)
    d = det(A)          # determinant via LU
    s, logd = slogdet(A)  # signed log-determinant

Functions:
    solve       - Batched A @ x = b for SPD matrices (Cholesky-based, any size)
    cholesky    - Batched Cholesky factorization (any size)
    qr          - Batched Householder QR factorization (k <= 128)
    tril_solve  - Batched forward substitution L @ x = b (k <= 128)
    triu_solve  - Batched back substitution L^T @ x = b (k <= 128)
    det         - Batched determinant via LU factorization
    slogdet     - Batched signed log-determinant via LU
    logdet_spd  - Batched log-determinant for SPD matrices via GPU Cholesky
"""

from ._blocked import blocked_cholesky, blocked_solve
from ._metal_kernels import (
    solve_cholesky,
    solve_lu,
    tril_solve,
    triu_solve,
    qr,
    MAX_GPU_K,
    SHARED_K_MAX,
)
from ._det import det, slogdet, logdet_spd

# Public API: solve and cholesky handle any matrix size
solve = blocked_solve
cholesky = blocked_cholesky

__all__ = [
    "solve",
    "cholesky",
    "solve_cholesky",
    "solve_lu",
    "qr",
    "tril_solve",
    "triu_solve",
    "det",
    "slogdet",
    "logdet_spd",
    "MAX_GPU_K",
    "SHARED_K_MAX",
]
