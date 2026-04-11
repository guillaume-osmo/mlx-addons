"""
GPU-accelerated batched linear algebra for MLX.

Metal GPU kernels for Cholesky, solve, and determinant on batched matrices
of **any size**. Small matrices (k <= 32) use per-thread Metal kernels.
Larger matrices use a blocked algorithm: small Metal kernels for diagonal
blocks + MLX matmul for the bulk of the work (SYRK updates).

Usage::

    import mlx.core as mx
    from mlx_addons.linalg import solve, cholesky, det, slogdet

    A = mx.array(...)  # (batch, n, n) SPD matrices — any n
    b = mx.array(...)  # (batch, n, m) right-hand sides
    x = solve(A, b)    # GPU-accelerated solve

    d = det(A)          # determinant via LU
    s, logd = slogdet(A)  # signed log-determinant

Functions:
    solve       - Batched A @ x = b for SPD matrices (Cholesky-based, any size)
    cholesky    - Batched Cholesky factorization (any size)
    tril_solve  - Batched forward substitution L @ x = b (k <= 32)
    triu_solve  - Batched back substitution L^T @ x = b (k <= 32)
    det         - Batched determinant via LU factorization
    slogdet     - Batched signed log-determinant via LU
    logdet_spd  - Batched log-determinant for SPD matrices via GPU Cholesky
"""

from ._blocked import blocked_cholesky, blocked_solve
from ._metal_kernels import (
    tril_solve,
    triu_solve,
    MAX_GPU_K,
)
from ._det import det, slogdet, logdet_spd

# Public API: solve and cholesky handle any matrix size
solve = blocked_solve
cholesky = blocked_cholesky

__all__ = [
    "solve",
    "cholesky",
    "tril_solve",
    "triu_solve",
    "det",
    "slogdet",
    "logdet_spd",
    "MAX_GPU_K",
]
