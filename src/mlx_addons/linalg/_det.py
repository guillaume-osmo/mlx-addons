# Copyright (c) 2024 Guillaume
# SPDX-License-Identifier: MIT

"""
Determinant and log-determinant via LU factorization on GPU.

MLX lacks native det/slogdet. We compute them via:
    LU factorization → diagonal of U → product (with sign from pivots)

For SPD matrices, an optimized path uses Cholesky:
    det(A) = det(L)^2 = (prod(diag(L)))^2
"""

import mlx.core as mx


def _lu_det_parts(A: mx.array):
    """
    Compute (sign, log|det|) via LU factorization.

    Args:
        A: (..., n, n) square matrices

    Returns:
        sign: (...,) sign of determinant (+1 or -1)
        logabsdet: (...,) log of absolute value of determinant
    """
    # MLX lu_factor is CPU-only, but the factorization is a small part
    # of the overall pipeline and the result is used by GPU ops downstream.
    lu, pivots = mx.linalg.lu_factor(A, stream=mx.cpu)
    diag_u = mx.diagonal(lu, axis1=-2, axis2=-1)

    # Count pivot swaps: pivot[i] != i means a row swap happened
    n = pivots.shape[-1]
    identity = mx.arange(n, dtype=pivots.dtype)
    n_swaps = mx.sum(pivots != identity, axis=-1)
    pivot_sign = 1 - 2 * (n_swaps % 2)  # (-1)^n_swaps

    # Sign from diagonal elements of U
    diag_sign = mx.prod(mx.sign(diag_u), axis=-1)

    # Total sign
    sign = pivot_sign * diag_sign

    # Log absolute determinant (numerically stable)
    logabsdet = mx.sum(mx.log(mx.abs(diag_u)), axis=-1)

    return sign, logabsdet


def _cholesky_logdet(A: mx.array):
    """
    Compute log-determinant for SPD matrices via Cholesky (faster, GPU).

    det(A) = det(L L^T) = det(L)^2 = (prod diag(L))^2
    log|det(A)| = 2 * sum(log(diag(L)))

    Args:
        A: (..., n, n) SPD matrices

    Returns:
        logdet: (...,) log-determinant (always positive for SPD)
    """
    from . import cholesky as _cholesky
    L = _cholesky(A)
    diag_L = mx.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * mx.sum(mx.log(diag_L), axis=-1)


def det(A: mx.array) -> mx.array:
    """
    Determinant of batched matrices.

    Args:
        A: (..., n, n) square matrices

    Returns:
        det: (...,) determinants
    """
    sign, logabsdet = _lu_det_parts(A)
    return sign * mx.exp(logabsdet)


def slogdet(A: mx.array):
    """
    Signed log-determinant of batched matrices.

    More numerically stable than det() for large matrices.

    Args:
        A: (..., n, n) square matrices

    Returns:
        sign: (...,) signs (+1 or -1)
        logabsdet: (...,) log of absolute determinant
    """
    return _lu_det_parts(A)


def logdet_spd(A: mx.array) -> mx.array:
    """
    Log-determinant of batched SPD matrices via Cholesky (GPU-accelerated).

    Faster than slogdet for symmetric positive definite matrices because
    it uses our GPU Cholesky kernel instead of CPU LU.

    Args:
        A: (..., n, n) symmetric positive definite matrices

    Returns:
        logdet: (...,) log-determinants
    """
    return _cholesky_logdet(A)
