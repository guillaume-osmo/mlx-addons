"""Vector similarity primitives + online clustering for novelty-aware sampling.

Exports:
- cosine_sim_batched / max_cosine_to_set       — dense float vectors
- tanimoto_binary_batched / max_tanimoto_to_set — binary fingerprints
- StreamingFingerprintBank                       — append-only MLX-resident matrix
- OnlineSingleLinkCluster                        — vector-agnostic online clustering

Designed for streaming novelty filters in generative-sampling loops
(e.g. molecule generators with scaffold-aware accept/reject).
"""
from ._core import (
    OnlineSingleLinkCluster,
    StreamingFingerprintBank,
    cosine_sim_batched,
    l2_normalize,
    max_cosine_to_set,
    max_tanimoto_to_set,
    tanimoto_binary_batched,
)

__all__ = [
    "cosine_sim_batched",
    "max_cosine_to_set",
    "l2_normalize",
    "tanimoto_binary_batched",
    "max_tanimoto_to_set",
    "StreamingFingerprintBank",
    "OnlineSingleLinkCluster",
]
