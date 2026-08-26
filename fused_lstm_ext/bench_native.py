"""Time the native fused-LSTM primitive against the JIT kernel and eager MLX.

Reports forward and forward+backward wall time per call.

    python bench_native.py
"""
import time

import mlx.core as mx

from mlx_fused_lstm import fused_lstm as native_lstm

SHAPES = [(32, 64, 128, 64), (128, 64, 128, 64), (512, 64, 128, 64)]
WARMUP, ITERS = 3, 20


def eager(x, Wx, Wh, bias, h0, c0):
    ip = x @ Wx.T + bias
    h, c, outs = h0, c0, []
    for t in range(x.shape[1]):
        i, f, g, o = mx.split(ip[:, t, :] + h @ Wh.T, 4, axis=-1)
        c = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(g)
        h = mx.sigmoid(o) * mx.tanh(c)
        outs.append(h)
    return mx.stack(outs, axis=1)


def timeit(fn, *args):
    for _ in range(WARMUP):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


def main():
    try:
        from mlx_addons.fused_rnn import fused_lstm as jit_lstm
    except ImportError:
        jit_lstm = None

    print(f"{'B':>5} {'T':>4} {'IN':>4} {'H':>4} | {'eager':>9} {'JIT':>9} {'native':>9} | speedup")
    print("-" * 68)
    for B, T, IN, H in SHAPES:
        mx.random.seed(0)
        args = (
            mx.random.normal((B, T, IN)) * 0.3,
            mx.random.normal((4 * H, IN)) * 0.1,
            mx.random.normal((4 * H, H)) * 0.1,
            mx.random.normal((4 * H,)) * 0.1,
            mx.zeros((B, H)),
            mx.zeros((B, H)),
        )
        cot = mx.random.normal((B, T, H))

        def grad_of(fn):
            return mx.grad(lambda *p: mx.sum(fn(*p) * cot), argnums=(0, 1, 2, 3))

        for label, wrap in (("fwd", lambda f: f), ("fwd+bwd", grad_of)):
            t_e = timeit(wrap(eager), *args)
            t_j = timeit(wrap(jit_lstm), *args) if jit_lstm else float("nan")
            t_n = timeit(wrap(native_lstm), *args)
            tag = f"{B:>5} {T:>4} {IN:>4} {H:>4}"
            print(f"{tag} | {t_e:8.2f}m {t_j:8.2f}m {t_n:8.2f}m | "
                  f"{t_e / t_n:5.1f}x vs eager  ({label})")


if __name__ == "__main__":
    main()
