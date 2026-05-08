# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Pulay's DIIS extrapolator and the standard SCF commutator residual.

DIIS solves the augmented Pulay system

    [B  -1] [c]   [0 ]
    [-1  0] [λ] = [-1]

with ``B[i,j] = <e_i, e_j>`` (Frobenius inner products of error matrices),
returning coefficients ``c`` such that the extrapolated iterate
``F_extrap = sum_i c_i F_i`` minimizes the norm of the linear combination of
residuals subject to ``sum_i c_i = 1``.

Inputs are lists of ``(..., N, N)`` arrays (Fock and error per history step).
Leading dims are treated as independent batch elements; each gets its own
extrapolation. The augmented system is small (``nd + 1 <= 7`` typically) and
indefinite, so it's solved with :func:`mlx_addons.linalg.solve_lu` (general
LU, Metal-accelerated).

References:

- P. Pulay, *Chem. Phys. Lett.* 73, 393 (1980).
- P. Pulay, *J. Comp. Chem.* 3, 556 (1982).
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from ..linalg import solve_lu


def commutator_error(F: mx.array, P: mx.array) -> mx.array:
    """Standard SCF DIIS error vector ``e = F P - P F``.

    Vanishes exactly when ``[F, P] = 0`` (SCF stationary point in an
    orthogonal basis). Batched over leading dims.

    Args:
        F: (..., N, N) Fock matrix.
        P: (..., N, N) density matrix.

    Returns:
        e: (..., N, N) commutator residual.
    """
    return F @ P - P @ F


def pulay_diis(
    history_F: Sequence[mx.array],
    history_e: Sequence[mx.array],
    *,
    max_history: int = 6,
) -> mx.array:
    """Pulay DIIS extrapolation of a Fock history.

    Args:
        history_F: list of ``(..., N, N)`` Fock matrices, oldest first.
        history_e: list of ``(..., N, N)`` error matrices in the same order
            and length as ``history_F``.
        max_history: keep at most this many most-recent vectors.

    Returns:
        ``F_extrap`` of shape ``(..., N, N)`` — the DIIS-extrapolated Fock.
        If fewer than 2 history entries are present, the latest ``F`` is
        returned unchanged.

    Notes:
        Solves the augmented system via :func:`mlx_addons.linalg.solve_lu`
        (Metal-accelerated, batched, ``k <= 128``). Leading batch dims of the
        input arrays are flattened into the LU solver's single batch axis and
        unflattened on return.

        DIIS extrapolates ``F`` only — never the density matrix ``rho``. A
        linear combination of idempotent ``rho`` is not idempotent, so feeding
        purification with a DIIS-mixed ``rho`` would break it. Apply DIIS to
        ``F``, then re-purify (or diagonalize) to get the next ``rho``.
    """
    F_hist = list(history_F)[-max_history:]
    e_hist = list(history_e)[-max_history:]
    nd = len(F_hist)
    if nd < 2:
        return F_hist[-1]

    dtype = F_hist[0].dtype
    leading_shape = F_hist[0].shape[:-2]
    N = F_hist[0].shape[-1]

    # Stack history along a new leading axis: (nd, ..., N, N)
    F_stack = mx.stack(F_hist, axis=0)
    e_stack = mx.stack(e_hist, axis=0)

    # B[..., i, j] = sum_{n,m} e_i[..., n, m] * e_j[..., n, m]
    B = mx.einsum("i...nm,j...nm->...ij", e_stack, e_stack)

    # Build the augmented system per-batch:
    #   A = [[B, -1], [-1^T, 0]]  shape (..., nd+1, nd+1)
    #   b = [0, ..., 0, -1]^T     shape (..., nd+1, 1)
    minus_ones_col = -mx.ones((*leading_shape, nd, 1), dtype=dtype)
    minus_ones_row = -mx.ones((*leading_shape, 1, nd), dtype=dtype)
    zero_corner = mx.zeros((*leading_shape, 1, 1), dtype=dtype)

    top = mx.concatenate([B, minus_ones_col], axis=-1)
    bot = mx.concatenate([minus_ones_row, zero_corner], axis=-1)
    A = mx.concatenate([top, bot], axis=-2)              # (..., nd+1, nd+1)

    rhs_zeros = mx.zeros((*leading_shape, nd, 1), dtype=dtype)
    rhs_one = -mx.ones((*leading_shape, 1, 1), dtype=dtype)
    rhs = mx.concatenate([rhs_zeros, rhs_one], axis=-2)  # (..., nd+1, 1)

    # solve_lu wants a single (batch, k, k) leading dim. Flatten leading dims.
    n_batch = 1
    for d in leading_shape:
        n_batch *= d
    A_flat = mx.reshape(A, (n_batch, nd + 1, nd + 1))
    rhs_flat = mx.reshape(rhs, (n_batch, nd + 1, 1))
    coeffs_full = solve_lu(A_flat, rhs_flat)             # (n_batch, nd+1, 1)
    coeffs_full = mx.reshape(coeffs_full, (*leading_shape, nd + 1, 1))
    coeffs = coeffs_full[..., :nd, 0]                    # (..., nd)

    # F_extrap[..., n, m] = sum_i coeffs[..., i] * F_stack[i, ..., n, m]
    F_extrap = mx.einsum("...i,i...nm->...nm", coeffs, F_stack)
    return F_extrap
