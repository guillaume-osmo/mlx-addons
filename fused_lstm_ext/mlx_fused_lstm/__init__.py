import mlx.core as mx
from ._ext import fused_lstm_fwd as _fwd, fused_lstm_bwd  # noqa: F401

def fused_lstm(x, Wx, Wh, bias, h0, c0):
    """Trainable input-fused LSTM (native primitive: fwd kernel + fused BPTT
    backward via the C++ vjp, no recompute). x:[B,T,IN] Wx:[4H,IN] Wh:[4H,H]
    bias:[4H] h0,c0:[B,H] -> h:[B,T,H]."""
    return _fwd(x, Wx, Wh, bias, h0, c0)[0]
