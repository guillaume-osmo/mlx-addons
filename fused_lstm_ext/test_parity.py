"""Parity tests for the native fused-LSTM extension.

Checks the compiled primitive against an eager MLX LSTM (the oracle) for both
the forward pass and every gradient, and against the JIT `mx.fast.metal_kernel`
implementation in `mlx_addons.fused_rnn` where that is importable.

    python test_parity.py
"""
import mlx.core as mx

from mlx_fused_lstm import fused_lstm as native_lstm

B, T, IN, H = 12, 16, 32, 16
TOL_FWD = 1e-4
TOL_GRAD = 1e-3


def eager(x, Wx, Wh, bias, h0, c0):
    ip = x @ Wx.T + bias
    h, c, outs = h0, c0, []
    for t in range(x.shape[1]):
        i, f, g, o = mx.split(ip[:, t, :] + h @ Wh.T, 4, axis=-1)
        c = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(g)
        h = mx.sigmoid(o) * mx.tanh(c)
        outs.append(h)
    return mx.stack(outs, axis=1)


def inputs():
    mx.random.seed(0)
    return (
        mx.random.normal((B, T, IN)) * 0.3,
        mx.random.normal((4 * H, IN)) * 0.1,
        mx.random.normal((4 * H, H)) * 0.1,
        mx.random.normal((4 * H,)) * 0.1,
        mx.zeros((B, H)),
        mx.zeros((B, H)),
    )


def maxdiff(a, b):
    return float(mx.max(mx.abs(a - b)))


def main():
    args = inputs()

    d = maxdiff(native_lstm(*args), eager(*args))
    assert d < TOL_FWD, f"forward vs eager: {d:.2e}"
    print(f"forward   vs eager            max|delta| = {d:.2e}  OK")

    try:
        from mlx_addons.fused_rnn import fused_lstm as jit_lstm
    except ImportError:
        print("forward   vs JIT kernel      SKIPPED (mlx_addons not importable)")
    else:
        d = maxdiff(native_lstm(*args), jit_lstm(*args))
        assert d < TOL_FWD, f"forward vs JIT: {d:.2e}"
        print(f"forward   vs JIT kernel      max|delta| = {d:.2e}  OK")

    cot = mx.random.normal((B, T, H))
    argnums = (0, 1, 2, 3, 4)  # x, Wx, Wh, bias, h0 (dc0 is defined as zero)
    names = ("dx", "dWx", "dWh", "dbias", "dh0")

    def loss(fn):
        return lambda *p: mx.sum(fn(*p) * cot)

    g_native = mx.grad(loss(native_lstm), argnums=argnums)(*args)
    g_eager = mx.grad(loss(eager), argnums=argnums)(*args)
    for name, gn, ge in zip(names, g_native, g_eager):
        d = maxdiff(gn, ge)
        scale = max(float(mx.max(mx.abs(ge))), 1e-8)
        assert d / scale < TOL_GRAD, f"{name}: {d:.2e} (rel {d / scale:.2e})"
        print(f"grad {name:<6} vs eager        max|delta| = {d:.2e}  rel = {d / scale:.2e}  OK")

    print("\nall parity checks passed")


if __name__ == "__main__":
    main()
