"""Kernel helpers for kernel PCA / Nyström on Apple Silicon.

Pairwise kernels computed via Metal matmul:

    RBF(x, y)  = exp(-γ ||x - y||²)  — expand as  ||x||² + ||y||² - 2 x·y
    poly(x, y) = (γ x·y + coef0)^degree
    linear     = x·y
    sigmoid    = tanh(γ x·y + coef0)

The squared-Euclidean-distance matrix uses one ``X @ Y.T`` (Metal) plus an
elementwise add, so an RBF kernel on (N, D) × (M, D) costs essentially one
matmul regardless of D.
"""

from __future__ import annotations

from typing import Literal, Optional

import mlx.core as mx
import numpy as np


KernelName = Literal["linear", "rbf", "poly", "polynomial", "sigmoid"]


def _sq_dists_mx(X: mx.array, Y: Optional[mx.array] = None) -> mx.array:
    """Pairwise squared Euclidean distances (no sqrt). Returns shape (N, M).

    ``X : (N, D)``, ``Y : (M, D)``. If ``Y is None`` uses ``Y = X``.
    Computed as ``||x||² + ||y||² - 2 X @ Y.T`` — one Metal matmul.
    """
    if Y is None:
        Y = X
    X_sq = mx.sum(X * X, axis=1, keepdims=True)           # (N, 1)
    Y_sq = mx.sum(Y * Y, axis=1, keepdims=True).T         # (1, M)
    cross = X @ Y.T                                        # Metal matmul
    D = X_sq + Y_sq - 2.0 * cross
    # Clip tiny negatives from round-off
    return mx.maximum(D, mx.zeros_like(D))


def pairwise_kernel(
    X: np.ndarray | mx.array,
    Y: Optional[np.ndarray | mx.array] = None,
    *,
    kernel: KernelName = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute a pairwise kernel matrix on Metal. Returns ``np.ndarray`` of shape (N, M).

    Parameters
    ----------
    X : (N, D) array
    Y : (M, D) array or None
        If ``None``, computes the symmetric ``K(X, X)``.
    kernel : one of "linear", "rbf", "poly"/"polynomial", "sigmoid".
    gamma : kernel scale. For RBF and sigmoid, defaults to ``1 / n_features``.
        For poly, defaults to ``1 / n_features``.
    degree, coef0 : polynomial / sigmoid parameters (sklearn defaults).
    """
    X_mx = mx.array(np.ascontiguousarray(np.asarray(X, dtype=np.float32)))
    if Y is None:
        Y_mx = X_mx
    else:
        Y_mx = mx.array(np.ascontiguousarray(np.asarray(Y, dtype=np.float32)))
    n_features = X_mx.shape[1]

    if gamma is None:
        gamma = 1.0 / float(n_features)

    if kernel == "linear":
        K = X_mx @ Y_mx.T
    elif kernel == "rbf":
        D = _sq_dists_mx(X_mx, Y_mx)
        K = mx.exp(-gamma * D)
    elif kernel in ("poly", "polynomial"):
        K = mx.power(gamma * (X_mx @ Y_mx.T) + coef0, degree)
    elif kernel == "sigmoid":
        K = mx.tanh(gamma * (X_mx @ Y_mx.T) + coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    mx.eval(K)
    return np.array(K, dtype=np.float32)
