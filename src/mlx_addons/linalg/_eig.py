# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Symmetric-eigensolver helpers and spectral bounds for batched matrices.

Four primitives:

- :func:`gershgorin_bounds` — cheap (lo, hi) bracket for the spectrum of a
  symmetric matrix via Gershgorin disks. Used as the initial-guess input to
  density-matrix purification (no diagonalization needed).

- :func:`jacobi_eigh` — batched symmetric eigh on the **Metal GPU** via
  cyclic Jacobi rotations. Returns ``(w, V)`` for ``N <= 64`` symmetric
  matrices. Closes the "MLX 0.31.x has no GPU eigh" gap for the small-N
  regime that semiempirical SCF lives in (NDDO valence-sp k = 5..50).

- :func:`batched_eigh` — public eigh entry point. Dispatches to
  :func:`jacobi_eigh` for ``N <= 64`` (Metal GPU), and falls back to
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


# Maximum public N supported by the Jacobi GPU path. For N <= 32, A and V fit
# together in fast local storage. For 32 < N <= 64, A stays in threadgroup
# memory and V is accumulated in the output/device buffer.
JACOBI_MAX_N = 64
JACOBI_RESIDENT_MAX_N = 32
# Threadgroup-specialized buckets for the cooperative kernel. Keeping the
# threadgroup footprint at 8x8 / 16x16 / 32x32 improves occupancy versus using a
# single 32x32 kernel for every size.
JACOBI_TG_BUCKETS = (8, 16, 32)
JACOBI_TG_THREADS_BY_BUCKET = {
    8: 8,
    16: 16,
    32: 32,
}
# Size-aware crossover for the cooperative kernel. Each tuple is
# (max_n, max_batch_for_tg); None means always use TG in that size band.
JACOBI_TG_AUTO_LIMITS = (
    (8, 2048),
    (16, None),
    (20, 4096),
    (32, None),
)

_jacobi_eigh_kernel_thread = None
_jacobi_eigh_kernel_tg = {}
_jacobi_eigh_kernel_vg = {}


def _tg_bucket(N: int) -> int:
    for bucket in JACOBI_TG_BUCKETS:
        if N <= bucket:
            return bucket
    return JACOBI_TG_BUCKETS[-1]


def _tg_config(N: int) -> tuple[int, int]:
    bucket = _tg_bucket(N)
    return bucket, JACOBI_TG_THREADS_BY_BUCKET[bucket]


def _auto_uses_tg(B: int, N: int) -> bool:
    if N < 8:
        return False
    for max_n, limit in JACOBI_TG_AUTO_LIMITS:
        if N <= max_n:
            return limit is None or B <= limit
    return False


def _get_jacobi_eigh_kernel_thread():
    """Build (once) the thread-local cyclic-Jacobi kernel.

    One thread = one matrix. M[N*N] + Q[N*N] live in thread-local storage.
    Eigenpairs are sorted before write-out so the hot path does not need an
    extra ``mx.argsort`` / gather dispatch. Convergence is per-rotation via a
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
            bool any_rot = false;
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) {
                    float app = M[p*N + p];
                    float aqq = M[q*N + q];
                    float apq = M[p*N + q];
                    if (fabs(apq) > 1e-8f * (fabs(app) + fabs(aqq) + 1.0f)) {
                        any_rot = true;
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
            if (!any_rot) {
                break;
            }
        }
        int order[NMAX];
        for (int i = 0; i < N; ++i) {
            order[i] = i;
        }
        for (int i = 0; i < N - 1; ++i) {
            int best = i;
            float best_val = M[order[i] * N + order[i]];
            for (int j = i + 1; j < N; ++j) {
                float val = M[order[j] * N + order[j]];
                if (val < best_val) {
                    best = j;
                    best_val = val;
                }
            }
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
        for (int out_col = 0; out_col < N; ++out_col) {
            int src_col = order[out_col];
            W[b*N + out_col] = M[src_col*N + src_col];
            for (int row = 0; row < N; ++row) {
                V[b*N*N + row*N + out_col] = Q[row*N + src_col];
            }
        }
        """,
    )
    return _jacobi_eigh_kernel_thread


def _get_jacobi_eigh_kernel_tg(nmax: int, threads: int):
    """Build (once) the threadgroup-cooperative cyclic-Jacobi kernel.

    One threadgroup = one matrix. ``threads`` lanes cooperate on each rotation;
    M and Q live in threadgroup-shared memory sized to ``nmax``. Each (p, q)
    rotation parallelizes the row/column update and eigenvector accumulation
    across the threadgroup.
    """
    global _jacobi_eigh_kernel_tg
    key = (nmax, threads)
    if key in _jacobi_eigh_kernel_tg:
        return _jacobi_eigh_kernel_tg[key]
    source = r"""
        constexpr int NMAX = __NMAX__;
        constexpr int THREADS = __THREADS__;
        constexpr int SWEEPS = 20;
        uint b = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        int N = int(A_shape[1]);
        threadgroup float M[NMAX * NMAX];
        threadgroup float Q[NMAX * NMAX];
        threadgroup int order[NMAX];
        threadgroup int did_rotate;
        // Cooperative load + identity init for Q.
        for (uint idx = tid; idx < uint(N * N); idx += THREADS) {
            uint i = idx / uint(N);
            uint j = idx - i * uint(N);
            M[idx] = A[b * uint(N * N) + idx];
            Q[idx] = (i == j) ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int sweep = 0; sweep < SWEEPS; ++sweep) {
            if (tid == 0) {
                did_rotate = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
                    if (s != 0.0f) {
                        if (tid == 0) {
                            did_rotate = 1;
                        }
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (did_rotate == 0) {
                break;
            }
        }
        if (tid == 0) {
            for (int i = 0; i < N; ++i) {
                order[i] = i;
            }
            for (int i = 0; i < N - 1; ++i) {
                int best = i;
                float best_val = M[order[i] * N + order[i]];
                for (int j = i + 1; j < N; ++j) {
                    float val = M[order[j] * N + order[j]];
                    if (val < best_val) {
                        best = j;
                        best_val = val;
                    }
                }
                int tmp = order[i];
                order[i] = order[best];
                order[best] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Cooperative sorted write-out.
        for (uint out_col = tid; out_col < uint(N); out_col += THREADS) {
            int src_col = order[out_col];
            W[b * uint(N) + out_col] = M[src_col*N + src_col];
        }
        for (uint idx = tid; idx < uint(N * N); idx += THREADS) {
            uint row = idx / uint(N);
            uint out_col = idx - row * uint(N);
            int src_col = order[out_col];
            V[b * uint(N * N) + idx] = Q[row*N + src_col];
        }
        """
    source = source.replace("__NMAX__", str(nmax)).replace("__THREADS__", str(threads))
    _jacobi_eigh_kernel_tg[key] = mx.fast.metal_kernel(
        name=f"batched_jacobi_eigh_tg_f32_n{nmax}_t{threads}",
        input_names=["A"],
        output_names=["W", "V"],
        source=source,
    )
    return _jacobi_eigh_kernel_tg[key]


def _get_jacobi_eigh_kernel_vg(nmax: int, threads: int):
    """Build (once) the larger-N Jacobi kernel with V in device memory."""
    global _jacobi_eigh_kernel_vg
    key = (nmax, threads)
    if key in _jacobi_eigh_kernel_vg:
        return _jacobi_eigh_kernel_vg[key]
    source = r"""
        constexpr int NMAX = __NMAX__;
        constexpr int THREADS = __THREADS__;
        constexpr int SWEEPS = 30;
        uint b = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        int N = int(A_shape[1]);
        uint nn = uint(N * N);
        threadgroup float M[NMAX * NMAX];
        threadgroup float rowbuf[NMAX];
        threadgroup int order[NMAX];
        threadgroup int did_rotate;

        for (uint idx = tid; idx < nn; idx += THREADS) {
            uint i = idx / uint(N);
            uint j = idx - i * uint(N);
            M[idx] = A[b * nn + idx];
            V[b * nn + idx] = (i == j) ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        for (int sweep = 0; sweep < SWEEPS; ++sweep) {
            if (tid == 0) {
                did_rotate = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
                    if (s != 0.0f) {
                        if (tid == 0) {
                            did_rotate = 1;
                        }
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
                        for (uint k = tid; k < uint(N); k += THREADS) {
                            uint base = b * nn + k * uint(N);
                            float vkp = V[base + uint(p)];
                            float vkq = V[base + uint(q)];
                            V[base + uint(p)] = c*vkp - s*vkq;
                            V[base + uint(q)] = s*vkp + c*vkq;
                        }
                        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
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
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (did_rotate == 0) {
                break;
            }
        }

        if (tid == 0) {
            for (int i = 0; i < N; ++i) {
                order[i] = i;
            }
            for (int i = 0; i < N - 1; ++i) {
                int best = i;
                float best_val = M[order[i] * N + order[i]];
                for (int j = i + 1; j < N; ++j) {
                    float val = M[order[j] * N + order[j]];
                    if (val < best_val) {
                        best = j;
                        best_val = val;
                    }
                }
                int tmp = order[i];
                order[i] = order[best];
                order[best] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint out_col = tid; out_col < uint(N); out_col += THREADS) {
            int src_col = order[out_col];
            W[b * uint(N) + out_col] = M[src_col*N + src_col];
        }

        // Sort V columns in-place one row at a time. The row scratch avoids
        // read-after-write hazards without allocating a second V output.
        for (uint row = 0; row < uint(N); ++row) {
            uint base = b * nn + row * uint(N);
            for (uint col = tid; col < uint(N); col += THREADS) {
                rowbuf[col] = V[base + col];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint out_col = tid; out_col < uint(N); out_col += THREADS) {
                int src_col = order[out_col];
                V[base + out_col] = rowbuf[src_col];
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        """
    source = source.replace("__NMAX__", str(nmax)).replace("__THREADS__", str(threads))
    _jacobi_eigh_kernel_vg[key] = mx.fast.metal_kernel(
        name=f"batched_jacobi_eigh_vg_f32_n{nmax}_t{threads}",
        input_names=["A"],
        output_names=["W", "V"],
        source=source,
    )
    return _jacobi_eigh_kernel_vg[key]


def jacobi_eigh(A: mx.array, *, kernel: str = "auto") -> tuple[mx.array, mx.array]:
    """Batched symmetric eigh on Metal GPU via cyclic Jacobi rotations.

    Three kernels are available:

    - ``"thread"`` — 1 GPU thread = 1 matrix; thread-local M and Q. Wins
      at large batch sizes where launch overhead is amortized across many
      independent threads (mlxmolkit's NDDO regime: B in the hundreds-to-
      thousands, k = 5..32).
    - ``"tg"`` — 1 threadgroup = 1 matrix; 8/16/32 threads cooperate on
      row/col and eigenvector updates per rotation, with M and Q in a
      size-specialized threadgroup-shared footprint.
    - ``"vg"`` — 1 threadgroup = 1 matrix for 32 < N <= 64; A is kept in
      threadgroup memory and V is accumulated in device/output memory.

    The auto crossover is size-aware: tiny matrices use the thread kernel for
    high-throughput batches, medium matrices use the cooperative kernel until
    batches are large enough to saturate the thread kernel, and N near 32 uses
    the cooperative kernel by default.

    Args:
        A: ``(B, N, N)`` symmetric, float32, ``N <= JACOBI_MAX_N`` (64).
        kernel: ``"auto"`` (default — picks based on ``B``), ``"thread"``,
            ``"tg"``, or ``"vg"``.

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
    if kernel not in ("auto", "thread", "tg", "vg"):
        raise ValueError(f"jacobi_eigh kernel must be 'auto'|'thread'|'tg'|'vg'; got {kernel!r}")
    A_f32 = A.astype(mx.float32) if A.dtype != mx.float32 else A
    use_vg = kernel == "vg" or (kernel == "auto" and N > JACOBI_RESIDENT_MAX_N)
    if use_vg:
        nmax = 48 if N <= 48 else 64
        threads = 64
        kfn = _get_jacobi_eigh_kernel_vg(nmax, threads)
        W_unsorted, V_unsorted = kfn(
            inputs=[A_f32],
            grid=(B * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(B, N), (B, N, N)],
            output_dtypes=[mx.float32, mx.float32],
        )
    elif kernel == "thread" and N > JACOBI_RESIDENT_MAX_N:
        raise ValueError(f"kernel='thread' supports N <= {JACOBI_RESIDENT_MAX_N}; got N={N}")
    elif kernel == "tg" and N > JACOBI_RESIDENT_MAX_N:
        raise ValueError(f"kernel='tg' supports N <= {JACOBI_RESIDENT_MAX_N}; got N={N}")
    elif kernel == "tg" or (kernel == "auto" and _auto_uses_tg(B, N)):
        nmax, threads = _tg_config(N)
        kfn = _get_jacobi_eigh_kernel_tg(nmax, threads)
        # MLX grid is total threads, NOT number of threadgroups: B groups
        # of `threads` threads each -> grid_x = B * threads.
        W_unsorted, V_unsorted = kfn(
            inputs=[A_f32],
            grid=(B * threads, 1, 1),
            threadgroup=(threads, 1, 1),
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
    return W_unsorted, V_unsorted


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

    For ``N <= JACOBI_MAX_N`` (64), uses :func:`jacobi_eigh` (Metal GPU,
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


def eigh_small_batch(
    A: mx.array,
    *,
    symmetrize: bool = True,
    stream=None,
) -> tuple[mx.array, mx.array]:
    """MLX-style entry point for many small symmetric eigendecompositions.

    This is a named convenience wrapper around :func:`batched_eigh`: for
    ``N <= JACOBI_MAX_N`` it uses the Metal cyclic-Jacobi kernel, and for larger
    matrices it preserves the existing CPU-stream fallback. The explicit name is
    useful for downstream scientific packages that want to signal they are using
    the batched-small Apple-GPU path intentionally.

    Args:
        A: ``(..., N, N)`` real symmetric matrices.
        symmetrize: If true, decompose ``0.5 * (A + A.T)``. This protects callers
            from tiny floating-point asymmetry in covariance/Fock-like matrices.
        stream: optional stream for the CPU fallback path.

    Returns:
        ``(w, V)`` with eigenvalues ascending and eigenvectors in columns.
    """
    A_in = A.astype(mx.float32) if A.dtype != mx.float32 else A
    if symmetrize:
        A_in = 0.5 * (A_in + mx.swapaxes(A_in, -2, -1))
    return batched_eigh(A_in, stream=stream)


def eigh_symmetric_3x3(
    A: mx.array,
    *,
    symmetrize: bool = True,
) -> tuple[mx.array, mx.array]:
    """Convenience eigensolver for batched 3x3 covariance/inertia matrices."""
    if A.shape[-2:] != (3, 3):
        raise ValueError(f"eigh_symmetric_3x3 expects shape (..., 3, 3); got {A.shape}")
    return eigh_small_batch(A, symmetrize=symmetrize)


def principal_axes_3x3(A: mx.array) -> tuple[mx.array, mx.array]:
    """Return descending eigenvalues and right-handed axes for 3x3 matrices.

    The returned ``axes`` matrix stores principal axes as columns. If the
    eigensolver returns a left-handed frame, the last axis is flipped so
    ``det(axes)`` is positive.
    """
    w, V = eigh_symmetric_3x3(A)
    order = mx.argsort(w, axis=-1)[..., ::-1]
    w_desc = mx.take_along_axis(w, order, axis=-1)
    col_order = mx.broadcast_to(mx.expand_dims(order, axis=-2), V.shape)
    axes = mx.take_along_axis(V, col_order, axis=-1)
    det = _det3(axes)
    flip = mx.where(det < 0.0, -1.0, 1.0).astype(axes.dtype)
    last = axes[..., :, 2] * flip[..., None]
    axes = mx.concatenate([axes[..., :, :2], last[..., None]], axis=-1)
    return w_desc, axes


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
        currently support ``N <= MAX_GPU_K`` (128) in mlx-addons. Sizes up
        to 80 use Metal kernels; 81..128 use CPU-stream triangular solves for
        correctness until a blocked GPU back-transform path lands.
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


def _det3(A: mx.array) -> mx.array:
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 0, 2]
    d = A[..., 1, 0]
    e = A[..., 1, 1]
    f = A[..., 1, 2]
    g = A[..., 2, 0]
    h = A[..., 2, 1]
    i = A[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
