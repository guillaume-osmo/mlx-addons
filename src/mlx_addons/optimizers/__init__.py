"""Optimizers for MLX that exploit structure the stock implementations miss.

Usage::

    from mlx_addons.optimizers import Muon

    opt = Muon(learning_rate=0.02, momentum=0.95)   # drop-in for mlx.optimizers.Muon
    opt.update(model, grads)

Functions / classes:
    Muon                          - Muon with SYRK-accelerated Newton-Schulz
    zeropower_via_newtonschulz5   - The orthogonalization step, standalone
"""

from ._muon import Muon, zeropower_via_newtonschulz5

__all__ = ["Muon", "zeropower_via_newtonschulz5"]
