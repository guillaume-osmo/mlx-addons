"""Clustering algorithms on Apple Silicon via MLX.

Currently exposes:

- :class:`KMeans` — Lloyd's algorithm with k-means++ init, Metal matmul
  for the assignment and update steps. sklearn-compatible.
"""

from ._kmeans import KMeans

__all__ = ["KMeans"]
