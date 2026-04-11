# Copyright (c) 2024 Guillaume
# SPDX-License-Identifier: MIT

"""
Blocked (tiled) Cholesky factorization and solve for arbitrarily large
matrices, fully on GPU.

Strategy: decompose the N×N matrix into B×B blocks. The algorithm becomes:
  1. Small diagonal block Cholesky (B×B) — our Metal kernel
  2. Triangular solve (TRSM) for below-diagonal blocks — our Metal kernel
  3. Symmetric rank-k update (SYRK) for the trailing submatrix — MLX matmul

Step 3 is ~90% of the FLOPS for large N and is pure matmul — fully
GPU-accelerated by MLX's optimized Metal matmul.

This removes the 32×32 limit entirely. Any matrix size works.
"""

import mlx.core as mx
from ._metal_kernels import (
    solve as _small_solve,
    cholesky as _small_cholesky,
    tril_solve as _small_tril_solve,
    MAX_GPU_K,
)

# Block size for tiled decomposition. Must be <= MAX_GPU_K.
# 32 is optimal: large enough for good matmul efficiency,
# small enough for our per-thread Metal kernel.
BLOCK_SIZE = MAX_GPU_K


def blocked_cholesky(A: mx.array) -> mx.array:
    """
    Batched Cholesky factorization for SPD matrices of ANY size on GPU.

    For k <= 32: dispatches directly to Metal kernel (one thread per matrix).
    For k > 32: uses blocked algorithm with matmul-heavy SYRK updates.

    Args:
        A: (batch, n, n) symmetric positive definite matrices

    Returns:
        L: (batch, n, n) lower triangular Cholesky factors
    """
    batch, n, _ = A.shape

    if n <= MAX_GPU_K:
        return _small_cholesky(A)

    B = BLOCK_SIZE
    # Work on a mutable copy (we'll build L in-place conceptually)
    # MLX is functional, so we use slicing + concatenation
    L = mx.zeros_like(A)

    # We need to iterate over block columns
    # For each block column j:
    #   1. Cholesky on diagonal block A[j,j]
    #   2. TRSM: solve L[i,j] for i > j
    #   3. SYRK: update A[i,k] for i,k > j

    # Since MLX arrays are immutable, we accumulate updates on A
    # and extract L blocks as we go. Use numpy-style but via mx ops.

    # Convert to list-of-blocks approach
    nblocks = (n + B - 1) // B
    block_sizes = [min(B, n - i * B) for i in range(nblocks)]

    # Work with the full matrix, updating the trailing submatrix each step
    work = A.astype(mx.float32)

    # Accumulate L blocks
    L_blocks = [[None] * nblocks for _ in range(nblocks)]

    for j in range(nblocks):
        j0 = j * B
        j1 = j0 + block_sizes[j]
        bj = block_sizes[j]

        # 1. Cholesky on diagonal block
        diag_block = work[:, j0:j1, j0:j1]  # (batch, bj, bj)
        L_jj = _small_cholesky(diag_block)    # GPU Metal kernel
        L_blocks[j][j] = L_jj

        # 2. TRSM: for each block row i > j, solve L_jj @ L_ij^T = A_ij^T
        #    i.e. L_ij = A_ij @ inv(L_jj^T) = solve(L_jj, A_ij^T)^T
        for i in range(j + 1, nblocks):
            i0 = i * B
            i1 = i0 + block_sizes[i]

            A_ij = work[:, i0:i1, j0:j1]  # (batch, bi, bj)
            # L_ij @ L_jj^T = A_ij  =>  L_jj @ L_ij^T = A_ij^T
            # L_ij^T = solve(L_jj, A_ij^T)  =>  L_ij = solve(L_jj, A_ij^T)^T
            L_ij_T = _small_tril_solve(L_jj, A_ij.swapaxes(-2, -1))
            L_blocks[i][j] = L_ij_T.swapaxes(-2, -1)

        # 3. SYRK update: A[i,k] -= L[i,j] @ L[k,j]^T for i,k > j
        # This is the matmul-heavy step — runs on GPU via MLX matmul
        if j + 1 < nblocks:
            # Gather all below-diagonal L blocks for column j
            remaining_start = (j + 1) * B
            # Concatenate L blocks for rows > j in column j
            L_col_blocks = []
            for i in range(j + 1, nblocks):
                L_col_blocks.append(L_blocks[i][j])
            L_col = mx.concatenate(L_col_blocks, axis=1)  # (batch, n-remaining_start, bj)

            # Compute the full SYRK update: L_col @ L_col^T
            update = L_col @ L_col.swapaxes(-2, -1)  # (batch, n-r, n-r) — THIS IS MATMUL

            # Subtract from trailing submatrix
            trailing = work[:, remaining_start:, remaining_start:]
            work_list = []
            # Reconstruct work matrix with updated trailing block
            # Top-left stays the same, trailing gets updated
            top = work[:, :remaining_start, :]
            bottom_left = work[:, remaining_start:, :remaining_start]
            new_trailing = trailing - update
            bottom = mx.concatenate([bottom_left, new_trailing], axis=2)
            work = mx.concatenate([top, bottom], axis=1)

    # Assemble L from blocks
    rows = []
    for i in range(nblocks):
        row_blocks = []
        for j in range(nblocks):
            if L_blocks[i][j] is not None:
                row_blocks.append(L_blocks[i][j])
            else:
                bi = block_sizes[i]
                bj = block_sizes[j]
                row_blocks.append(mx.zeros((batch, bi, bj), dtype=mx.float32))
        rows.append(mx.concatenate(row_blocks, axis=2))
    L = mx.concatenate(rows, axis=1)

    return L


def blocked_solve(A: mx.array, b: mx.array) -> mx.array:
    """
    Solve A @ x = b for batched SPD matrices of ANY size on GPU.

    Uses blocked Cholesky: L L^T = A, then forward + back substitution.
    All steps run on GPU (Metal kernels for small blocks, MLX matmul for updates).

    Args:
        A: (batch, n, n) symmetric positive definite matrices
        b: (batch, n, m) or (batch, n) right-hand sides

    Returns:
        x: same shape as b
    """
    batch, n, _ = A.shape

    if n <= MAX_GPU_K:
        return _small_solve(A, b)

    was_2d = b.ndim == 2
    if was_2d:
        b = mx.expand_dims(b, -1)
    m = b.shape[-1]

    B = BLOCK_SIZE
    nblocks = (n + B - 1) // B
    block_sizes = [min(B, n - i * B) for i in range(nblocks)]

    # Step 1: Cholesky
    L = blocked_cholesky(A)

    # Step 2: Forward substitution L @ y = b (block-wise)
    y = mx.zeros((batch, n, m), dtype=mx.float32)
    b = b.astype(mx.float32)
    for i in range(nblocks):
        i0 = i * B
        i1 = i0 + block_sizes[i]

        # rhs = b[i] - sum(L[i,j] @ y[j] for j < i)
        rhs = b[:, i0:i1, :]
        for j in range(i):
            j0 = j * B
            j1 = j0 + block_sizes[j]
            L_ij = L[:, i0:i1, j0:j1]
            rhs = rhs - L_ij @ y[:, j0:j1, :]  # matmul on GPU

        # Solve L[i,i] @ y[i] = rhs
        L_ii = L[:, i0:i1, i0:i1]
        y_i = _small_tril_solve(L_ii, rhs)  # Metal kernel

        # Insert into y
        y = mx.concatenate([y[:, :i0, :], y_i, y[:, i1:, :]], axis=1)

    # Step 3: Back substitution L^T @ x = y (block-wise, reverse order)
    x = mx.zeros((batch, n, m), dtype=mx.float32)
    LT = L.swapaxes(-2, -1)
    for i in range(nblocks - 1, -1, -1):
        i0 = i * B
        i1 = i0 + block_sizes[i]

        rhs = y[:, i0:i1, :]
        for j in range(i + 1, nblocks):
            j0 = j * B
            j1 = j0 + block_sizes[j]
            LT_ij = LT[:, i0:i1, j0:j1]
            rhs = rhs - LT_ij @ x[:, j0:j1, :]  # matmul on GPU

        # Solve L^T[i,i] @ x[i] = rhs  =>  L[i,i]^T @ x[i] = rhs
        L_ii = L[:, i0:i1, i0:i1]
        # triu_solve: L^T @ x = rhs
        from ._metal_kernels import triu_solve as _small_triu_solve
        x_i = _small_triu_solve(L_ii, rhs)

        x = mx.concatenate([x[:, :i0, :], x_i, x[:, i1:, :]], axis=1)

    if was_2d:
        x = mx.squeeze(x, axis=-1)
    return x
