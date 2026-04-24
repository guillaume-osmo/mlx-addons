"""Sparse × dense matrix multiplication on Metal.

Currently exposes :func:`csr_matmul` — a minimalist CSR × dense SpMM
kernel. One Metal thread per output element: ``C[i, j] = sum_l A[i, l] * B[l, j]``.
Loads the row slice ``indptr[i]:indptr[i+1]`` of ``indices``/``values`` and
accumulates dot products against ``B[:, j]``.

For highly sparse matrices (density ≲ 5%, any shape) and for very wide
feature spaces (n_features ≫ n_components) this beats a densified matmul
both in compute and in memory. For denser or very small problems, a plain
``A @ B`` with a dense ``A`` wins — kernel-launch overhead + one-thread-per-
output-element parallelism is no match for Apple's tensor-core GEMM path.

See :func:`csr_from_dense` for a convenience builder from a dense numpy
array.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import numpy as np

from ._metal_kernels import _get_kernel


_CSR_SPMM_SRC = """
// Grid: (M * N), thread_position_in_grid.x encodes (i, j).
uint gid = thread_position_in_grid.x;
if (gid >= M * N) return;
uint i = gid / N;
uint j = gid % N;

uint row_start = indptr[i];
uint row_end   = indptr[i + 1];

float acc = 0.0f;
for (uint p = row_start; p < row_end; p++) {
    uint col = indices[p];
    acc += values[p] * B[col * N + j];
}
C[gid] = acc;
"""


def csr_matmul(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    B: np.ndarray | mx.array,
    M: int,
) -> mx.array:
    """CSR-formatted sparse × dense matrix multiply on Metal.

    Parameters
    ----------
    indptr : (M + 1,) uint32 np.ndarray
        CSR row pointer.
    indices : (nnz,) uint32 np.ndarray
        Column indices of non-zeros.
    values : (nnz,) float32 np.ndarray
        Non-zero values.
    B : (K, N) array, dense. ``K`` must equal ``indices.max() + 1``.
    M : int
        Number of rows of the sparse matrix (unambiguous — ``indptr`` has ``M + 1`` entries).

    Returns
    -------
    C : (M, N) mlx.array on the active stream (Metal by default).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import random as sprandom
    >>> from mlx_addons.linalg import csr_matmul
    >>> A = sprandom(500, 1000, density=0.02, format="csr", dtype=np.float32)
    >>> B = np.random.randn(1000, 64).astype(np.float32)
    >>> C = csr_matmul(A.indptr.astype(np.uint32), A.indices.astype(np.uint32), A.data, B, M=500)
    >>> C.shape
    (500, 64)
    """
    if np.asarray(indptr).shape != (M + 1,):
        raise ValueError(f"indptr must have shape ({M + 1},), got {np.asarray(indptr).shape}")
    B_mx = mx.array(np.ascontiguousarray(np.asarray(B, dtype=np.float32)))
    N = B_mx.shape[-1]

    # Keep all CSR arrays on MLX
    indptr_mx = mx.array(np.ascontiguousarray(indptr, dtype=np.uint32))
    indices_mx = mx.array(np.ascontiguousarray(indices, dtype=np.uint32))
    values_mx = mx.array(np.ascontiguousarray(values, dtype=np.float32))

    kernel = _get_kernel(
        "csr_spmm", ["indptr", "indices", "values", "B"], ["C"], _CSR_SPMM_SRC,
    )
    total = M * N
    tg = min(256, total)
    (C_flat,) = kernel(
        inputs=[indptr_mx, indices_mx, values_mx, B_mx],
        template=[("M", M), ("N", N)],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )
    return mx.reshape(C_flat, (M, N))


def csr_from_dense(A: np.ndarray, tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dense ``(M, K)`` array to CSR ``(indptr, indices, values)``.

    Entries with ``|A[i, j]| <= tol`` are treated as zero. Useful for
    interop with ``csr_matmul`` without needing scipy.sparse.
    """
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float32))
    mask = np.abs(A) > tol
    indptr = np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.uint32)
    rows, cols = np.nonzero(mask)
    indices = cols.astype(np.uint32)
    values = A[rows, cols].astype(np.float32)
    return indptr, indices, values
