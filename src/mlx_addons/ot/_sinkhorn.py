"""Batched entropic optimal transport (Sinkhorn) on Apple Silicon via MLX.

Solves many small OT problems at once on the GPU. Unlike the textbook ``1/a`` scaling,
the iteration here divides defensively, so distributions with **empty bins** (zero mass)
are handled without NaNs — common for sparse histograms / profiles.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def sinkhorn2_batch(a, b, C, reg: float, *, num_iters: int = 100) -> np.ndarray:
    """Entropic-regularized OT transport cost ``<T, C>`` for a batch of problems.

    Parameters
    ----------
    a : (P, B) array
        Source masses for ``P`` problems (each row a distribution; rows need not be
        strictly positive — zero-mass bins are fine).
    b : (P, B) array
        Target masses.
    C : (P, B, B) or (B, B) array
        Ground cost. A single ``(B, B)`` matrix is broadcast across the batch.
    reg : float
        Entropic regularization strength (``epsilon``). Smaller = closer to exact OT,
        but needs more iterations / can underflow.
    num_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    (P,) np.ndarray of transport costs ``<T, C>``.
    """
    a = mx.array(np.ascontiguousarray(np.asarray(a, dtype=np.float32)))
    b = mx.array(np.ascontiguousarray(np.asarray(b, dtype=np.float32)))
    C = mx.array(np.ascontiguousarray(np.asarray(C, dtype=np.float32)))
    if C.ndim == 2:
        C = mx.broadcast_to(C[None], (a.shape[0],) + tuple(C.shape))

    K = mx.exp(-C / float(reg))                       # (P, B, B)
    u = mx.ones_like(a)
    for _ in range(num_iters):
        KTu = mx.matmul(mx.swapaxes(K, 1, 2), u[..., None])[..., 0]
        v = b / (KTu + 1e-30)
        Kv = mx.matmul(K, v[..., None])[..., 0]
        u = a / (Kv + 1e-30)
    T = u[..., None] * K * v[:, None, :]              # transport plan
    cost = (T * C).sum(axis=(1, 2))
    mx.eval(cost)
    return np.array(cost)
