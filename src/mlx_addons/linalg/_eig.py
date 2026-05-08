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


# Maximum N supported by the thread-local Jacobi kernel. Each thread holds
# both M[N*N] and V[N*N] in registers — 2 * 32^2 * 4 bytes = 8 KB per thread.
JACOBI_MAX_N = 32

_jacobi_eigh_kernel = None


def _get_jacobi_eigh_kernel():
    """Build (once) the batched cyclic-Jacobi eigh Metal kernel.

    Body computes both eigenvalues (M's diagonal after convergence) and
    eigenvectors (an identity matrix rotated in lockstep with M). Cyclic
    sweeps over all (p<q) pairs; each rotation zeros M[p,q] = M[q,p] and
    mixes columns p and q of the running V.
    """
    global _jacobi_eigh_kernel
    if _jacobi_eigh_kernel is not None:
        return _jacobi_eigh_kernel
    _jacobi_eigh_kernel = mx.fast.metal_kernel(
        name="batched_jacobi_eigh_f32",
        input_names=["A"],
        output_names=["W", "V"],
        source=r"""
        constexpr int MAX_N = 32;
        constexpr int SWEEPS = 32;
        uint b = thread_position_in_grid.x;
        int N = int(A_shape[1]);
        if (N > MAX_N) return;

        // Thread-local copies of M (working matrix) and V (running
        // eigenvector matrix initialized to identity).
        thread float M[MAX_N * MAX_N];
        thread float Vm[MAX_N * MAX_N];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                M[i * N + j] = A[b * N * N + i * N + j];
                Vm[i * N + j] = (i == j) ? 1.0f : 0.0f;
            }
        }

        // Cyclic Jacobi sweeps over all p < q.
        for (int sweep = 0; sweep < SWEEPS; ++sweep) {
            float off = 0.0f;
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) {
                    float app = M[p * N + p];
                    float aqq = M[q * N + q];
                    float apq = M[p * N + q];
                    if (fabs(apq) <= 1e-7f) continue;
                    off += apq * apq;
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = copysign(1.0f, tau)
                              / (fabs(tau) + sqrt(1.0f + tau * tau));
                    float c = rsqrt(1.0f + t * t);
                    float s = t * c;
                    // Apply rotation to M (rows/cols p, q except diagonals).
                    for (int k = 0; k < N; ++k) {
                        if (k == p || k == q) continue;
                        float mkp = M[k * N + p];
                        float mkq = M[k * N + q];
                        float new_kp = c * mkp - s * mkq;
                        float new_kq = s * mkp + c * mkq;
                        M[k * N + p] = new_kp;
                        M[p * N + k] = new_kp;
                        M[k * N + q] = new_kq;
                        M[q * N + k] = new_kq;
                    }
                    M[p * N + p] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                    M[q * N + q] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                    M[p * N + q] = 0.0f;
                    M[q * N + p] = 0.0f;
                    // Apply same rotation to columns p and q of V.
                    for (int k = 0; k < N; ++k) {
                        float vkp = Vm[k * N + p];
                        float vkq = Vm[k * N + q];
                        Vm[k * N + p] = c * vkp - s * vkq;
                        Vm[k * N + q] = s * vkp + c * vkq;
                    }
                }
            }
            if (off < 1e-14f) break;  // converged
        }

        // Extract eigenvalues, then sort ascending (insertion sort, N<=32).
        thread float w_local[MAX_N];
        thread int   idx[MAX_N];
        for (int i = 0; i < N; ++i) { w_local[i] = M[i * N + i]; idx[i] = i; }
        for (int i = 1; i < N; ++i) {
            float key_w = w_local[i]; int key_i = idx[i];
            int j = i - 1;
            while (j >= 0 && w_local[j] > key_w) {
                w_local[j + 1] = w_local[j];
                idx[j + 1] = idx[j];
                j--;
            }
            w_local[j + 1] = key_w;
            idx[j + 1] = key_i;
        }
        // Write sorted eigenvalues + correspondingly permuted eigenvectors.
        for (int i = 0; i < N; ++i) W[b * N + i] = w_local[i];
        for (int col = 0; col < N; ++col) {
            int src_col = idx[col];
            for (int row = 0; row < N; ++row) {
                V[b * N * N + row * N + col] = Vm[row * N + src_col];
            }
        }
        """,
    )
    return _jacobi_eigh_kernel


def jacobi_eigh(A: mx.array) -> tuple[mx.array, mx.array]:
    """Batched symmetric eigh on Metal GPU via cyclic Jacobi rotations.

    Closes the "MLX has no GPU eigh" gap for the small-N regime where
    semiempirical SCF lives. Each thread handles one matrix with
    thread-local storage; outputs ``(w, V)`` with ``w`` ascending and ``V``
    columns the corresponding orthonormal eigenvectors.

    Args:
        A: ``(B, N, N)`` symmetric, float32, ``N <= JACOBI_MAX_N`` (32).

    Returns:
        ``(W, V)`` — ``W`` shape ``(B, N)`` ascending, ``V`` shape
        ``(B, N, N)`` with eigenvectors as columns.

    Raises:
        ValueError: if ``A.ndim != 3`` or ``A`` is not square or
            ``N > JACOBI_MAX_N``.
    """
    if A.ndim != 3:
        raise ValueError(f"jacobi_eigh expects (B, N, N); got shape {A.shape}")
    B, N, M = A.shape
    if N != M:
        raise ValueError(f"jacobi_eigh expects square matrices; got ({N}, {M})")
    if N > JACOBI_MAX_N:
        raise ValueError(
            f"jacobi_eigh supports N <= {JACOBI_MAX_N}; got N={N}"
        )
    A_f32 = A.astype(mx.float32) if A.dtype != mx.float32 else A
    kernel = _get_jacobi_eigh_kernel()
    W, V = kernel(
        inputs=[A_f32],
        grid=(B, 1, 1),
        threadgroup=(min(256, B), 1, 1),
        output_shapes=[(B, N), (B, N, N)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return W, V


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
