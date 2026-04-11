"""
GPU-accelerated K-nearest neighbors for MLX.

Uses Z-order tree construction + Metal GPU kernels for batched
distance computation and segmented top-k selection.

Usage::

    import mlx.core as mx
    from mlx_addons.knn import knn

    pos = mx.random.normal((10000, 3))
    distances, indices = knn(pos, k=16)

Functions:
    knn         - Find k nearest neighbors using Z-order tree + GPU
"""

from ._knn import knn_v6 as knn
from ._config import KNNConfig, TreeConfig

__all__ = ["knn", "KNNConfig", "TreeConfig"]
