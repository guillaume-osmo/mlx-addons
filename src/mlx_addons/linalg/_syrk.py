"""Symmetric rank-k update (BLAS ``SYRK``) for MLX: ``X @ X.T`` at ~half the flops.

MLX dispatches ``X @ X.T`` to the same general GEMM as ``X @ Y`` and pays full
price, even though the result is symmetric (measured on M4 Pro, bf16, 8192x8192:
151.2 ms for ``X @ X.T`` vs 149.4 ms for ``X @ Y``).

This module recovers the symmetry saving **without a custom Metal kernel**. Split
``X`` into ``k`` row-blocks and compute only the ``k(k+1)/2`` upper-triangular
output blocks, mirroring the rest with a transpose. The flop ratio versus a full
GEMM is ``(1 + 1/k) / 2`` — 0.75 at ``k=2``, 0.5625 at ``k=8``, approaching 0.5.
MLX's ``steel_gemm`` stays near peak on 1024-row output tiles, so most of the
theoretical saving survives (measured on M4 Pro, float32):

    shape=(2048, 4096)   5.57 ms ->   4.48 ms (1.24x, 2 blocks)
    shape=(4096, 4096)  21.51 ms ->  14.93 ms (1.44x, 4 blocks)
    shape=(8192, 4096)  86.03 ms ->  54.67 ms (1.57x, 8 blocks)
    shape=(8192, 8192) 208.31 ms -> 123.79 ms (1.68x, 8 blocks)

In bfloat16 the gains are slightly larger (1.72x at 8192x8192). Applied to both
symmetric matmuls of the Newton-Schulz orthogonalization in
``mlx.optimizers.Muon``, this is 1.14x end-to-end at 2048x2048 and 1.35x at
8192x8192 — only 2 of the 3 matmuls per step have symmetric outputs, so the
per-matmul gain is diluted. See ``benchmarks/bench_syrk.py``.

The trick is the CPU-free equivalent of the CUDA kernel in `flash-muon
<https://github.com/nil0x9/flash-muon>`_, which skips lower-triangular GEMM tiles
inside a fused kernel; the idea is due to Laker Newhouse et al. A hand-written
``mx.fast.metal_kernel`` SYRK could add at most ~1.15x over this, since it would
have to out-perform ``steel_gemm`` (already at ~78% of the M4 Pro's fp32 peak).

**Blocking is not free.** Each block is a separate dispatch, and the mirrored
halves cost ``O(n^2)`` extra writes. When the contraction dimension is short the
blocks become thin-K GEMMs and the whole thing loses (measured 0.65-0.80x for
``K=256`` at every ``M``). :func:`syrk` therefore falls back to a plain matmul
below the calibrated thresholds, so it is always safe to call.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

#: Target rows per output block. Smaller tiles cut flops further but drop GEMM
#: efficiency; 1024 was the measured optimum on M4 Pro (512-row tiles regressed).
BLOCK_ROWS = 1024

#: Never split into more than this many blocks (dispatch count grows as k^2/2).
MAX_BLOCKS = 16

#: Below this output dimension the saved flops do not pay for the extra
#: dispatches and the mirror writes.
MIN_DIM = 2048

#: Below this contraction length the per-block GEMMs are too thin to stay
#: efficient, and blocking is a net loss regardless of the output size.
MIN_CONTRACT = 1024


def _num_blocks(
    dim: int,
    contract: int,
    blocks: Optional[int],
    block_rows: int,
    min_dim: int,
    min_contract: int,
) -> int:
    """Resolve the block count, or 1 to signal "use a plain matmul"."""
    if blocks is not None:
        return max(1, min(int(blocks), dim))
    if dim < min_dim or contract < min_contract:
        return 1
    return max(1, min(dim // block_rows, MAX_BLOCKS, dim))


def syrk(
    X: mx.array,
    *,
    trans: bool = False,
    blocks: Optional[int] = None,
    block_rows: int = BLOCK_ROWS,
    min_dim: int = MIN_DIM,
    min_contract: int = MIN_CONTRACT,
) -> mx.array:
    """Symmetric product ``X @ X.T`` (or ``X.T @ X``) skipping redundant blocks.

    Numerically equivalent to the plain matmul up to summation order: the
    diagonal blocks accumulate identically, and off-diagonal blocks are computed
    once and mirrored, so the result is *exactly* symmetric — unlike
    ``X @ X.T``, whose upper and lower halves can differ in the last ulp.

    Parameters
    ----------
    X : (..., M, K) mlx.array
        Dense matrix, optionally batched. Any float dtype; leading batch
        dimensions are broadcast by ``mx.matmul`` as usual.
    trans : bool, default False
        If True compute ``X.T @ X`` (shape ``(..., K, K)``) instead of
        ``X @ X.T`` (shape ``(..., M, M)``), blocking over columns.
    blocks : int, optional
        Force the number of blocks. ``1`` disables blocking entirely.
        By default the count is chosen as ``dim // block_rows``, capped at
        ``MAX_BLOCKS``, and disabled below the thresholds below.
    block_rows : int, default 1024
        Target rows per output block for the automatic block count.
    min_dim : int, default 2048
        Output dimension below which blocking is skipped.
    min_contract : int, default 1024
        Contraction length below which blocking is skipped.

    Returns
    -------
    (..., M, M) or (..., K, K) mlx.array
        The symmetric product, same dtype as ``X``.

    Notes
    -----
    Thresholds were calibrated on an M4 Pro (20-core GPU). They are advisory,
    not correctness-critical — pass ``blocks=`` to override on other hardware.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx_addons.linalg import syrk
    >>> X = mx.random.normal((4096, 4096))
    >>> A = syrk(X)                    # ~1.44x faster than X @ X.T
    >>> C = syrk(X, trans=True)        # Gram matrix X.T @ X
    """
    if X.ndim < 2:
        raise ValueError(f"syrk expects at least a 2D array, got shape {X.shape}")

    axis = -1 if trans else -2
    dim = X.shape[axis]
    contract = X.shape[-2 if trans else -1]

    k = _num_blocks(dim, contract, blocks, block_rows, min_dim, min_contract)
    if k < 2:
        return mx.matmul(mx.swapaxes(X, -1, -2), X) if trans else mx.matmul(
            X, mx.swapaxes(X, -1, -2)
        )

    # Split into k near-equal slices along the output axis. Sizes need not
    # divide evenly; boundaries are chosen so slices differ by at most 1.
    bounds = [(dim * i // k, dim * (i + 1) // k) for i in range(k)]
    if trans:
        parts = [mx.swapaxes(X[..., :, s:e], -1, -2) for s, e in bounds]
    else:
        parts = [X[..., s:e, :] for s, e in bounds]

    # Upper triangle only — this is where the flops are saved.
    upper = {
        (i, j): mx.matmul(parts[i], mx.swapaxes(parts[j], -1, -2))
        for i in range(k)
        for j in range(i, k)
    }

    rows = [
        mx.concatenate(
            [
                upper[(i, j)] if j >= i else mx.swapaxes(upper[(j, i)], -1, -2)
                for j in range(k)
            ],
            axis=-1,
        )
        for i in range(k)
    ]
    return mx.concatenate(rows, axis=-2)


def gram(X: mx.array, **kwargs) -> mx.array:
    """Gram matrix ``X.T @ X`` — :func:`syrk` with ``trans=True``.

    Convenience alias for the covariance / kernel-method call pattern, where
    ``X`` is ``(n_samples, n_features)`` and the wanted product contracts over
    samples.
    """
    return syrk(X, trans=True, **kwargs)
