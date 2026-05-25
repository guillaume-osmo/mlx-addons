"""Fast recurrent cells with custom Metal kernels.

Currently exports:
- MetalLSTM:        drop-in for mlx.nn.LSTM (single direction, single layer)
                    using a fused per-timestep gate-and-state Metal kernel.
                    1.3-2.4× faster than mlx.nn.LSTM on Apple Silicon.
- GroupedMetalLSTM: G parallel LSTMs sharing the same input but each with
                    its own weights — runs in one cell kernel launch per
                    timestep instead of G launches. Useful when an architecture
                    has multiple parallel LSTM branches.

Kernel design ported from PR ml-explore/mlx#3089 (Godin, 2024) and adapted to
the user-land `mx.fast.metal_kernel` API.
"""
from ._metal_lstm import (
    GroupedMetalLSTM,
    MetalLSTM,
    metal_grouped_lstm_scan,
    metal_lstm_scan,
)

__all__ = [
    "MetalLSTM",
    "GroupedMetalLSTM",
    "metal_lstm_scan",
    "metal_grouped_lstm_scan",
]
