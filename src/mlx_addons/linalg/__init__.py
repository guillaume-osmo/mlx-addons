"""
GPU-accelerated batched linear algebra for MLX.

Provides Metal GPU kernels for Cholesky factorization and linear solve
on batched small SPD matrices (k <= 32). Each thread handles one matrix
in the batch, achieving 25-40x speedup over CPU LAPACK.

Usage::

    import mlx.core as mx
    from mlx_addons.linalg import solve, cholesky

    A = mx.array(...)  # (batch, k, k) SPD matrices
    b = mx.array(...)  # (batch, k, m) right-hand sides
    x = solve(A, b)    # (batch, k, m) solutions — runs on GPU

Functions:
    solve           - Batched A @ x = b for SPD matrices (Cholesky-based)
    cholesky        - Batched Cholesky factorization
    tril_solve      - Batched forward substitution (L @ x = b)
    triu_solve      - Batched back substitution (L^T @ x = b)
"""

from ._metal_kernels import (
    solve,
    cholesky,
    tril_solve,
    triu_solve,
    MAX_GPU_K,
)

__all__ = ["solve", "cholesky", "tril_solve", "triu_solve", "MAX_GPU_K"]
