# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Symmetric-eigensolver helpers and spectral bounds for batched matrices.

Two primitives:

- :func:`gershgorin_bounds` — cheap (lo, hi) bracket for the spectrum of a
  symmetric matrix via Gershgorin disks. Used as the initial-guess input to
  density-matrix purification (no diagonalization needed).

- :func:`batched_eigh` — thin wrapper over :func:`mx.linalg.eigh`. Exists as a
  single indirection point so that, if MLX's eigh turns out to fall back to
  CPU on Metal, a custom GPU kernel can be plugged in here without touching
  any consumer.

Both batched over arbitrary leading dims; inputs assumed symmetric (only the
lower triangle is read by ``eigh``; ``gershgorin_bounds`` reads the whole
matrix and is exact regardless of symmetry).
"""

from __future__ import annotations

import mlx.core as mx


def gershgorin_bounds(A: mx.array) -> tuple[mx.array, mx.array]:
    """Cheap spectral bracket via Gershgorin disks.

    Every eigenvalue of ``A`` lies in the union of disks
    ``[A_ii - R_i, A_ii + R_i]`` where ``R_i = sum_{j != i} |A_ij|``.

    For a symmetric ``A`` this gives a sound (lo, hi) bracket for the entire
    spectrum without any factorization. Used as the initial-guess scaling
    input for density-matrix purification.

    Args:
        A: (..., N, N) matrix. Batched over arbitrary leading dims.

    Returns:
        ``(lo, hi)`` — each of shape ``(...,)`` with the same dtype as ``A``.
    """
    abs_A = mx.abs(A)
    diag = mx.diagonal(A, axis1=-2, axis2=-1)            # (..., N)
    abs_diag = mx.diagonal(abs_A, axis1=-2, axis2=-1)    # (..., N)
    row_abs_sum = mx.sum(abs_A, axis=-1)                 # (..., N)
    R = row_abs_sum - abs_diag                           # off-diagonal abs sum
    lo = mx.min(diag - R, axis=-1)
    hi = mx.max(diag + R, axis=-1)
    return lo, hi


def batched_eigh(A: mx.array, *, stream=None) -> tuple[mx.array, mx.array]:
    """Batched symmetric eigendecomposition — indirection over MLX's eigh.

    Equivalent to ``mx.linalg.eigh(A)``: returns ``(w, v)`` where ``w`` is
    ascending per matrix and the columns of ``v`` are the eigenvectors.
    Batched on the last two dims.

    Defaults to the CPU stream because MLX 0.31.x raises
    ``"linalg::eigh not yet supported on the GPU"`` for the default
    Metal/GPU stream. The wrapper exists so a future custom Metal eigh can
    replace this body in one place once MLX (or this package) ships one.

    Args:
        A: (..., N, N) symmetric matrix.
        stream: optional MLX stream override. Defaults to ``mx.cpu``.

    Returns:
        ``(w, v)`` — eigenvalues ``(..., N)`` ascending, eigenvectors
        ``(..., N, N)``.
    """
    if stream is None:
        stream = mx.cpu
    return mx.linalg.eigh(A, stream=stream)
