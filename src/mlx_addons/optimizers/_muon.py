"""Muon with a symmetry-aware Newton-Schulz orthogonalization.

``mlx.optimizers.Muon`` replaces each 2D parameter update with the nearest
orthogonal matrix, computed by a fixed-point Newton-Schulz iteration::

    A = X @ X.T
    B = b * A + c * (A @ A)
    X = a * X + B @ X

Two of those three matmuls have symmetric outputs — ``X @ X.T`` trivially, and
``A @ A`` because ``A`` is symmetric, so ``A @ A = A @ A.T``. MLX computes both
with a general GEMM. Routing them through :func:`mlx_addons.linalg.syrk` skips
the redundant lower-triangular blocks (measured on M4 Pro, bfloat16, 5 steps):

    shape=(2048, 2048)    39.2 ms ->   34.4 ms (1.14x)
    shape=(4096, 4096)   294.1 ms ->  231.6 ms (1.27x)
    shape=(8192, 8192)  2373.1 ms -> 1755.8 ms (1.35x)

The per-matmul gain (up to 1.72x) is diluted because the third matmul, ``B @ X``,
is not symmetric.

This is the MLX equivalent of `flash-muon <https://github.com/nil0x9/flash-muon>`_,
which fuses the same idea into a CUDA/Triton kernel. Unlike flash-muon's
optimizer, nothing here requires ``torch.distributed``.

**When this is worth it.** Only for full-rank 2D parameters of 2048 or more on
the shorter side — below that :func:`~mlx_addons.linalg.syrk` correctly falls
back and this class is exactly ``mlx.optimizers.Muon``, bit for bit. LoRA/DoRA
adapters (rank <= 64) and small MLP heads see nothing. And a faster Muon is
still Muon: benchmark it against AdamW on your own task before adopting it.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers

from ..linalg._syrk import MIN_CONTRACT, MIN_DIM, _num_blocks, syrk


def zeropower_via_newtonschulz5(X: mx.array, steps: int = 5) -> mx.array:
    """Orthogonalize ``X`` by 5th-order Newton-Schulz, using SYRK where possible.

    Numerically equivalent to ``mlx.optimizers.Muon._zeropower_via_newtonschulz5``
    up to summation order. The quintic coefficients are Keller Jordan's, tuned to
    push the singular values of ``X`` toward 1 rather than to converge to the
    exact polar factor — so the output is only approximately orthogonal, by
    design.

    Parameters
    ----------
    X : (M, N) mlx.array
        Matrix to orthogonalize, typically a momentum-smoothed gradient.
    steps : int, default 5
        Number of Newton-Schulz iterations.

    Returns
    -------
    (M, N) mlx.array
        Approximately orthogonal matrix with the same shape and dtype as ``X``.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array for Newton-Schulz, got shape {X.shape}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = X.shape[-2] > X.shape[-1]
    if transpose_needed:
        X = X.T

    # Blocking is decided once, on the post-transpose shape: every iterate has
    # the same dimensions, so the answer cannot change between steps.
    dim, contract = X.shape[-2], X.shape[-1]
    use_syrk = _num_blocks(dim, contract, None, 1024, MIN_DIM, MIN_CONTRACT) > 1

    X = X / (mx.linalg.norm(X, keepdims=True) + 1e-7)
    for _ in range(steps):
        if use_syrk:
            A = syrk(X)
            B = b * A + c * syrk(A)
        else:
            # Below the SYRK thresholds, keep MLX's fused addmm path — it is
            # faster than an unfused `b * A + c * (A @ A)`.
            A = X @ X.T
            B = mx.addmm(b * A, A, A, beta=1.0, alpha=c)
        X = mx.addmm(a * X, B, X, beta=1.0, alpha=1.0)

    return X.T if transpose_needed else X


class Muon(mlx.optimizers.Muon):
    """Drop-in :class:`mlx.optimizers.Muon` with SYRK-accelerated Newton-Schulz.

    Identical arguments, state, and update rule — only the orthogonalization
    inner loop differs. See the module docstring for measured speedups and for
    when they materialize.

    Examples
    --------
    >>> from mlx_addons.optimizers import Muon
    >>> opt = Muon(learning_rate=0.02, momentum=0.95)
    >>> opt.update(model, grads)  # doctest: +SKIP
    """

    def _zeropower_via_newtonschulz5(self, X: mx.array, steps: int) -> mx.array:
        return zeropower_via_newtonschulz5(X, steps=steps)
