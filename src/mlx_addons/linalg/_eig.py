# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Symmetric-eigensolver helpers and spectral bounds for batched matrices.

Four primitives:

- :func:`gershgorin_bounds` — cheap (lo, hi) bracket for the spectrum of a
  symmetric matrix via Gershgorin disks. Used as the initial-guess input to
  density-matrix purification (no diagonalization needed).

- :func:`jacobi_eigh` — batched symmetric eigh on the **Metal GPU** via
  cyclic Jacobi rotations. Returns ``(w, V)`` for ``N <= 32`` symmetric
  matrices. Closes the "MLX 0.31.x has no GPU eigh" gap for the small-N
  regime that semiempirical SCF lives in (NDDO valence-sp k = 5..50).

- :func:`batched_eigh` — public eigh entry point. Dispatches to
  :func:`jacobi_eigh` for ``N <= 32`` (Metal GPU), and falls back to
  :func:`mx.linalg.eigh` on the CPU stream otherwise.

- :func:`gen_eigh` — generalized symmetric eigenproblem
  ``F C = S C diag(w)`` with SPD ``S``, via Cholesky reduction. Reuses
  :func:`mlx_addons.linalg.cholesky` and the Metal triangular solves.

All batched over arbitrary leading dims; inputs assumed symmetric (only the
lower triangle is read by ``eigh``; ``gershgorin_bounds`` reads the whole
matrix and is exact regardless of symmetry).
"""

from __future__ import annotations

import mlx.core as mx


# Maximum N supported by the Jacobi kernels. Both variants store M[N*N] +
# Q[N*N] = 2*32^2*4 = 8 KB; thread-local for the per-thread variant,
# threadgroup-shared for the cooperative variant.
JACOBI_MAX_N = 32
# Threads per threadgroup for the cooperative kernel.
JACOBI_TG_THREADS = 64
# Heuristic crossover: at small batch the threadgroup kernel amortizes its
# launch overhead and beats the thread kernel; at large batch the thread
# kernel wins because each per-matrix Jacobi sweep is tiny and TG
# coordination is wasted work. Verified by bench on M-series.
JACOBI_TG_BATCH_THRESHOLD = 256

_jacobi_eigh_kernel_thread = None
_jacobi_eigh_kernel_tg = None


def _get_jacobi_eigh_kernel_thread():
    """Build (once) the thread-local cyclic-Jacobi kernel.

    One thread = one matrix. M[N*N] + Q[N*N] live in thread-local storage.
    Eigenvalues are returned unsorted; the public dispatcher sorts via
    ``mx.argsort`` on the GPU side. Convergence is per-rotation via a
    relative-threshold test against the diagonal magnitudes.
    """
    global _jacobi_eigh_kernel_thread
    if _jacobi_eigh_kernel_thread is not None:
        return _jacobi_eigh_kernel_thread
    _jacobi_eigh_kernel_thread = mx.fast.metal_kernel(
        name="batched_jacobi_eigh_thread_f32",
        input_names=["A"],
        output_names=["W", "V"],
        source=r"""
        constexpr int NMAX = 32;
        constexpr int SWEEPS = 24;
        uint b = thread_position_in_grid.x;
        int N = int(A_shape[1]);
        thread float M[NMAX * NMAX];
        thread float Q[NMAX * NMAX];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                M[i*N + j] = A[b*N*N + i*N + j];
                Q[i*N + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (int sweep = 0; sweep < SWEEPS; ++sweep) {
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) {
                    float app = M[p*N + p];
                    float aqq = M[q*N + q];
                    float apq = M[p*N + q];
                    if (fabs(apq) > 1e-8f * (fabs(app) + fabs(aqq) + 1.0f)) {
                        float tau = (aqq - app) / (2.0f * apq);
                        float t = copysign(1.0f, tau)
                                  / (fabs(tau) + sqrt(1.0f + tau*tau));
                        float c = rsqrt(1.0f + t*t);
                        float s = t * c;
                        for (int k = 0; k < N; ++k) {
                            if (k != p && k != q) {
                                float mkp = M[k*N + p];
                                float mkq = M[k*N + q];
                                float new_kp = c*mkp - s*mkq;
                                float new_kq = s*mkp + c*mkq;
                                M[k*N + p] = new_kp;
                                M[p*N + k] = new_kp;
                                M[k*N + q] = new_kq;
                                M[q*N + k] = new_kq;
                            }
                        }
                        M[p*N + p] = c*c*app - 2.0f*s*c*apq + s*s*aqq;
                        M[q*N + q] = s*s*app + 2.0f*s*c*apq + c*c*aqq;
                        M[p*N + q] = 0.0f;
                        M[q*N + p] = 0.0f;
                        for (int k = 0; k < N; ++k) {
                            float qkp = Q[k*N + p];
                            float qkq = Q[k*N + q];
                            Q[k*N + p] = c*qkp - s*qkq;
                            Q[k*N + q] = s*qkp + c*qkq;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < N; ++i) {
            W[b*N + i] = M[i*N + i];
            for (int j = 0; j < N; ++j) {
                V[b*N*N + i*N + j] = Q[i*N + j];
            }
        }
        """,
    )
    return _jacobi_eigh_kernel_thread


def _get_jacobi_eigh_kernel_tg():
    """Build (once) the threadgroup-cooperative cyclic-Jacobi kernel.

    One threadgroup = one matrix; ``JACOBI_TG_THREADS`` (64) threads
    cooperate on each rotation. M and Q live in threadgroup-shared memory.
    Each (p, q) rotation parallelizes the row/column update and the
    eigenvector accumulation across the threads. Wins over the thread
    kernel for ``N >= JACOBI_TG_THRESHOLD_N``.
    """
    global _jacobi_eigh_kernel_tg
    if _jacobi_eigh_kernel_tg is not None:
        return _jacobi_eigh_kernel_tg
    _jacobi_eigh_kernel_tg = mx.fast.metal_kernel(
        name="batched_jacobi_eigh_tg_f32",
        input_names=["A"],
        output_names=["W", "V"],
        source=r"""
        constexpr int NMAX = 32;
        constexpr int THREADS = 64;
        constexpr int SWEEPS = 20;
        uint b = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        int N = int(A_shape[1]);
        threadgroup float M[NMAX * NMAX];
        threadgroup float Q[NMAX * NMAX];
        // Cooperative load + identity init for Q.
        for (uint idx = tid; idx < uint(N * N); idx += THREADS) {
            uint i = idx / uint(N);
            uint j = idx - i * uint(N);
            M[idx] = A[b * uint(N * N) + idx];
            Q[idx] = (i == j) ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int sweep = 0; sweep < SWEEPS; ++sweep) {
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) {
                    float app = M[p*N + p];
                    float aqq = M[q*N + q];
                    float apq = M[p*N + q];
                    float c = 1.0f;
                    float s = 0.0f;
                    if (fabs(apq) > 1e-8f * (fabs(app) + fabs(aqq) + 1.0f)) {
                        float tau = (aqq - app) / (2.0f * apq);
                        float t = copysign(1.0f, tau)
                                  / (fabs(tau) + sqrt(1.0f + tau*tau));
                        c = rsqrt(1.0f + t*t);
                        s = t * c;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    // Parallel row/col update for k != p, k != q.
                    for (uint k = tid; k < uint(N); k += THREADS) {
                        if (int(k) != p && int(k) != q) {
                            float mkp = M[k*N + p];
                            float mkq = M[k*N + q];
                            float mkp2 = c*mkp - s*mkq;
                            float mkq2 = s*mkp + c*mkq;
                            M[k*N + p] = mkp2;
                            M[p*N + k] = mkp2;
                            M[k*N + q] = mkq2;
                            M[q*N + k] = mkq2;
                        }
                    }
                    // Parallel eigenvector update (all rows of Q's cols p, q).
                    for (uint k = tid; k < uint(N); k += THREADS) {
                        float qkp = Q[k*N + p];
                        float qkq = Q[k*N + q];
                        Q[k*N + p] = c*qkp - s*qkq;
                        Q[k*N + q] = s*qkp + c*qkq;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    // One thread updates the (p,p), (q,q), (p,q), (q,p) entries.
                    if (tid == 0) {
                        M[p*N + p] = c*c*app - 2.0f*s*c*apq + s*s*aqq;
                        M[q*N + q] = s*s*app + 2.0f*s*c*apq + c*c*aqq;
                        M[p*N + q] = 0.0f;
                        M[q*N + p] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        // Cooperative write-out.
        for (uint i = tid; i < uint(N); i += THREADS) {
            W[b * uint(N) + i] = M[i*N + i];
        }
        for (uint idx = tid; idx < uint(N * N); idx += THREADS) {
            V[b * uint(N * N) + idx] = Q[idx];
        }
        """,
    )
    return _jacobi_eigh_kernel_tg


def _sort_eig_pairs(W: mx.array, V: mx.array) -> tuple[mx.array, mx.array]:
    """Sort eigenpairs ascending by eigenvalue using GPU-side argsort."""
    idx = mx.argsort(W, axis=-1)                           # (B, N)
    W_sorted = mx.take_along_axis(W, idx, axis=-1)         # (B, N)
    # V columns are eigenvectors (axis=-1 = column index): permute columns.
    V_sorted = mx.take_along_axis(V, idx[:, None, :], axis=-1)
    return W_sorted, V_sorted


def jacobi_eigh(A: mx.array, *, kernel: str = "auto") -> tuple[mx.array, mx.array]:
    """Batched symmetric eigh on Metal GPU via cyclic Jacobi rotations.

    Two kernels available, auto-dispatched by batch size:

    - ``"thread"`` — 1 GPU thread = 1 matrix; thread-local M and Q. Wins
      at large batch sizes where launch overhead is amortized across many
      independent threads (mlxmolkit's NDDO regime: B in the hundreds-to-
      thousands, k = 5..32).
    - ``"tg"`` — 1 threadgroup = 1 matrix; ``JACOBI_TG_THREADS`` (64)
      threads cooperate on row/col and eigenvector updates per rotation;
      M and Q in threadgroup-shared memory. Wins at small batch sizes
      where the per-rotation parallelization beats single-threaded inner
      work.

    Crossover is around ``B = JACOBI_TG_BATCH_THRESHOLD`` (256) on
    M-series silicon, roughly independent of ``N`` in the supported range.

    Args:
        A: ``(B, N, N)`` symmetric, float32, ``N <= JACOBI_MAX_N`` (32).
        kernel: ``"auto"`` (default — picks based on ``B``), ``"thread"``,
            or ``"tg"``.

    Returns:
        ``(W, V)`` — ``W`` shape ``(B, N)`` ascending; ``V`` shape
        ``(B, N, N)`` with eigenvectors as columns; ``A V = V diag(W)``.

    Raises:
        ValueError: if ``A.ndim != 3`` or ``A`` is not square or
            ``N > JACOBI_MAX_N`` or ``kernel`` is unknown.
    """
    if A.ndim != 3:
        raise ValueError(f"jacobi_eigh expects (B, N, N); got shape {A.shape}")
    B, N, MM = A.shape
    if N != MM:
        raise ValueError(f"jacobi_eigh expects square matrices; got ({N}, {MM})")
    if N > JACOBI_MAX_N:
        raise ValueError(
            f"jacobi_eigh supports N <= {JACOBI_MAX_N}; got N={N}"
        )
    if kernel not in ("auto", "thread", "tg"):
        raise ValueError(f"jacobi_eigh kernel must be 'auto'|'thread'|'tg'; got {kernel!r}")
    A_f32 = A.astype(mx.float32) if A.dtype != mx.float32 else A
    use_tg = kernel == "tg" or (kernel == "auto" and B < JACOBI_TG_BATCH_THRESHOLD)
    if use_tg:
        kfn = _get_jacobi_eigh_kernel_tg()
        # MLX grid is total threads, NOT number of threadgroups: B groups
        # of JACOBI_TG_THREADS threads each → grid_x = B * JACOBI_TG_THREADS.
        W_unsorted, V_unsorted = kfn(
            inputs=[A_f32],
            grid=(B * JACOBI_TG_THREADS, 1, 1),
            threadgroup=(JACOBI_TG_THREADS, 1, 1),
            output_shapes=[(B, N), (B, N, N)],
            output_dtypes=[mx.float32, mx.float32],
        )
    else:
        kfn = _get_jacobi_eigh_kernel_thread()
        W_unsorted, V_unsorted = kfn(
            inputs=[A_f32],
            grid=(B, 1, 1),
            threadgroup=(min(256, B), 1, 1),
            output_shapes=[(B, N), (B, N, N)],
            output_dtypes=[mx.float32, mx.float32],
        )
    return _sort_eig_pairs(W_unsorted, V_unsorted)


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
    """Batched symmetric eigendecomposition — dispatches GPU-vs-CPU.

    For ``N <= JACOBI_MAX_N`` (32), uses :func:`jacobi_eigh` (Metal GPU,
    cyclic Jacobi). For larger ``N``, falls back to :func:`mx.linalg.eigh`
    on the CPU stream (MLX 0.31.x has no GPU LAPACK eigh).

    Args:
        A: (..., N, N) symmetric matrix.
        stream: optional MLX stream override for the CPU fallback path.
            Ignored when the Jacobi GPU path is taken. Defaults to ``mx.cpu``.

    Returns:
        ``(w, v)`` — eigenvalues ``(..., N)`` ascending, eigenvectors
        ``(..., N, N)``.
    """
    n = A.shape[-1]
    if n <= JACOBI_MAX_N:
        # Promote to (B, N, N) for the Jacobi kernel.
        was_2d = A.ndim == 2
        if was_2d:
            A_b = A[None, :, :]
        else:
            # Flatten any leading dims into a single batch.
            leading = A.shape[:-2]
            B = 1
            for d in leading:
                B *= d
            A_b = mx.reshape(A, (B, n, n))
        w, v = jacobi_eigh(A_b)
        if was_2d:
            return w[0], v[0]
        if A.ndim > 3:
            w = mx.reshape(w, (*A.shape[:-2], n))
            v = mx.reshape(v, (*A.shape[:-2], n, n))
        return w, v
    if stream is None:
        stream = mx.cpu
    return mx.linalg.eigh(A, stream=stream)


def gen_eigh(
    F: mx.array,
    S: mx.array,
    *,
    stream=None,
) -> tuple[mx.array, mx.array]:
    """Generalized symmetric eigenproblem ``F C = S C diag(w)``.

    Solves via Cholesky reduction:

    1. ``L L^T = S``  (Cholesky of the SPD overlap)
    2. ``A = L^-1 F L^-T``  (symmetric)
    3. ``A V = V diag(w)``  (standard symmetric eigh)
    4. ``C = L^-T V``  (back-transform)

    Eigenvalues ``w`` are ascending; eigenvectors ``C`` are S-orthonormal:
    ``C^T S C = I`` and ``F C = S C diag(w)``.

    Used by tight-binding electronic structure codes (xTB, DFTB, EHT) where
    the AO basis is non-orthogonal so the secular equation is generalized
    rather than standard.

    Args:
        F: (..., N, N) symmetric Fock-like matrix.
        S: (..., N, N) symmetric positive-definite overlap matrix.
        stream: stream for the inner :func:`mx.linalg.eigh` call. Defaults
            to ``mx.cpu`` since MLX 0.31.x raises on the GPU stream.

    Returns:
        ``(w, C)`` — eigenvalues ``(..., N)``, eigenvectors ``(..., N, N)``.

    Note:
        The reduction uses :func:`tril_solve` / :func:`triu_solve` which
        currently support ``N <= MAX_GPU_K`` (128) in mlx-addons. For
        larger ``N`` a blocked back-transform path is needed (TBD).
    """
    from ._metal_kernels import tril_solve, triu_solve, MAX_GPU_K
    from ._blocked import blocked_cholesky as cholesky

    n = F.shape[-1]
    if n > MAX_GPU_K:
        raise NotImplementedError(
            f"gen_eigh currently supports n <= {MAX_GPU_K}; got n={n}. "
            "Blocked back-transform path is TBD."
        )

    # Promote 2-D inputs to (1, N, N) for the kernels (which require a
    # leading batch axis); we squeeze it back on return.
    was_2d = F.ndim == 2
    if was_2d:
        F = F[None, :, :]
        S = S[None, :, :]

    # Cholesky: L L^T = S  (lower triangular)
    L = cholesky(S)

    # Reduce: A = L^-1 F L^-T
    # Step 1: M = L^-1 F  (solve L M = F column-by-column)
    M = tril_solve(L, F)
    # Step 2: A^T = L^-1 M^T  →  A = (L^-1 M^T)^T = M L^-T
    A_T = tril_solve(L, mx.swapaxes(M, -2, -1))
    A = mx.swapaxes(A_T, -2, -1)
    # Symmetrize against floating-point asymmetry from the two-step solve.
    A = 0.5 * (A + mx.swapaxes(A, -2, -1))

    # Standard symmetric eigh on the reduced matrix.
    if stream is None:
        stream = mx.cpu
    w, V_tilde = mx.linalg.eigh(A, stream=stream)

    # Back-transform: C = L^-T V_tilde  (solve L^T C = V_tilde)
    C = triu_solve(L, V_tilde)
    if was_2d:
        w = w[0]
        C = C[0]
    return w, C
