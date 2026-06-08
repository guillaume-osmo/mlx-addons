"""Correctness tests for the input-fused LSTM and GRU JIT kernels.

Forward + all gradients are checked against an eager MLX reference (PyTorch
conventions). GPU/Metal only (float32).
"""
import mlx.core as mx

from mlx_addons import fused_lstm, fused_gru


def _rel(a, b):
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-8))


def _randn(*s, k=1.0):
    return mx.random.normal(s) * k


# ----------------------------------------------------------------- LSTM
def _eager_lstm(x, Wx, Wh, bias, h0, c0):
    B, T, IN = x.shape
    H = h0.shape[1]
    ip = x @ Wx.T + bias
    h, c = h0, c0
    hs = []
    for t in range(T):
        z = ip[:, t, :] + h @ Wh.T
        i = mx.sigmoid(z[:, :H]); f = mx.sigmoid(z[:, H:2 * H])
        g = mx.tanh(z[:, 2 * H:3 * H]); o = mx.sigmoid(z[:, 3 * H:])
        c = f * c + i * g
        h = o * mx.tanh(c)
        hs.append(h)
    return mx.stack(hs, axis=1)


def test_fused_lstm_forward_and_grads():
    mx.random.seed(0)
    B, T, IN, H = 24, 20, 64, 32
    x = _randn(B, T, IN, k=0.5)
    Wx = _randn(4 * H, IN, k=0.1); Wh = _randn(4 * H, H, k=0.1)
    bias = _randn(4 * H, k=0.1); h0 = _randn(B, H, k=0.1); c0 = _randn(B, H, k=0.1)

    hf = fused_lstm(x, Wx, Wh, bias, h0, c0)
    he = _eager_lstm(x, Wx, Wh, bias, h0, c0)
    mx.eval(hf, he)
    assert _rel(hf, he) < 1e-5

    cot = _randn(B, T, H)
    gf = mx.grad(lambda *p: mx.sum(fused_lstm(*p, c0) * cot), argnums=(0, 1, 2, 3, 4))(x, Wx, Wh, bias, h0)
    ge = mx.grad(lambda *p: mx.sum(_eager_lstm(*p, c0) * cot), argnums=(0, 1, 2, 3, 4))(x, Wx, Wh, bias, h0)
    mx.eval(gf, ge)
    for a, b in zip(gf, ge):
        assert _rel(a, b) < 1e-5


# ----------------------------------------------------------------- GRU
def _eager_gru(x, Wx, Wh, bih, bhh, h0):
    B, T, IN = x.shape
    H = h0.shape[1]
    xall = x @ Wx.T
    h = h0
    outs = []
    for t in range(T):
        xt = xall[:, t, :]; hp = h @ Wh.T
        r = mx.sigmoid(xt[:, :H] + bih[:H] + hp[:, :H] + bhh[:H])
        z = mx.sigmoid(xt[:, H:2 * H] + bih[H:2 * H] + hp[:, H:2 * H] + bhh[H:2 * H])
        n = mx.tanh(xt[:, 2 * H:] + bih[2 * H:] + r * (hp[:, 2 * H:] + bhh[2 * H:]))
        h = (1 - z) * n + z * h
        outs.append(h)
    return mx.stack(outs, axis=1)


def test_fused_gru_forward_and_grads():
    mx.random.seed(1)
    B, T, IN, H = 24, 20, 64, 32
    x = _randn(B, T, IN, k=0.5)
    Wx = _randn(3 * H, IN, k=0.1); Wh = _randn(3 * H, H, k=0.1)
    bih = _randn(3 * H, k=0.1); bhh = _randn(3 * H, k=0.1); h0 = _randn(B, H, k=0.1)

    hf = fused_gru(x, Wx, Wh, bih, bhh, h0)
    he = _eager_gru(x, Wx, Wh, bih, bhh, h0)
    mx.eval(hf, he)
    assert _rel(hf, he) < 1e-5

    cot = _randn(B, T, H)
    gf = mx.grad(lambda *p: mx.sum(fused_gru(*p) * cot), argnums=(0, 1, 2, 3, 4, 5))(x, Wx, Wh, bih, bhh, h0)
    ge = mx.grad(lambda *p: mx.sum(_eager_gru(*p) * cot), argnums=(0, 1, 2, 3, 4, 5))(x, Wx, Wh, bih, bhh, h0)
    mx.eval(gf, ge)
    for a, b in zip(gf, ge):
        assert _rel(a, b) < 1e-5


if __name__ == "__main__":
    test_fused_lstm_forward_and_grads(); print("LSTM forward+grads: PASS")
    test_fused_gru_forward_and_grads(); print("GRU  forward+grads: PASS")
