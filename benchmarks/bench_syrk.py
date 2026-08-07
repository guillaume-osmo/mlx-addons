#!/usr/bin/env python3
"""Benchmark syrk (blocked symmetric product) against MLX's plain matmul.

Also reports the Newton-Schulz orthogonalization used by the Muon optimizer,
where two of the three matmuls per step have symmetric outputs.

Run:  PYTHONPATH=src python benchmarks/bench_syrk.py
"""

import time

import mlx.core as mx

from mlx_addons.linalg import syrk


def bench(fn, *args, warmup=6, repeat=25):
    # Sub-millisecond shapes need real repeat counts: at repeat=10 the small
    # cases swing by +-15% and invent regressions that are not there.
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e3


def bench_syrk(dtype=mx.float32):
    print(f"\n=== X @ X.T   ({dtype}) ===")
    print(f"{'M':>6} {'K':>6} {'matmul ms':>10} {'syrk ms':>9} {'speedup':>8} {'blocks':>7} {'TFLOP/s':>9}")
    for M, K in (
        (1024, 1024), (2048, 512), (2048, 1024), (2048, 4096),
        (4096, 1024), (4096, 4096), (8192, 1024), (8192, 4096), (8192, 8192),
    ):
        X = mx.random.normal((M, K)).astype(dtype)
        mx.eval(X)
        reps = 6 if M >= 4096 else 25
        t_ref = bench(lambda a: a @ a.T, X, repeat=reps)
        t_syr = bench(syrk, X, repeat=reps)
        blocks = max(1, min(M // 1024, 16)) if (M >= 2048 and K >= 1024) else 1
        tflops = 2 * M * M * K / (t_syr * 1e-3) / 1e12
        print(
            f"{M:>6} {K:>6} {t_ref:>10.2f} {t_syr:>9.2f} {t_ref/t_syr:>7.2f}x "
            f"{blocks:>7} {tflops:>9.2f}"
        )


def bench_newton_schulz(dtype=mx.bfloat16, steps=5):
    """NS5 as in mlx.optimizers.Muon, with and without syrk."""
    a, b, c = (3.4445, -4.7750, 2.0315)

    def ns5(X, sym):
        tr = X.shape[-2] > X.shape[-1]
        if tr:
            X = X.T
        X = X / (mx.linalg.norm(X, keepdims=True) + 1e-7)
        for _ in range(steps):
            A = syrk(X) if sym else X @ X.T
            AA = syrk(A) if sym else A @ A
            X = mx.addmm(a * X, b * A + c * AA, X, beta=1.0, alpha=1.0)
        return X.T if tr else X

    print(f"\n=== Newton-Schulz ({steps} steps, {dtype}) ===")
    print(f"{'shape':>14} {'baseline ms':>12} {'syrk ms':>9} {'speedup':>8}")
    for shape in ((1024, 1024), (2048, 2048), (4096, 1024), (4096, 4096), (8192, 8192)):
        X = mx.random.normal(shape).astype(dtype)
        mx.eval(X)
        reps = 4 if max(shape) >= 4096 else 15
        t_ref = bench(lambda z: ns5(z, False), X, repeat=reps)
        t_sym = bench(lambda z: ns5(z, True), X, repeat=reps)
        print(f"{str(shape):>14} {t_ref:>12.2f} {t_sym:>9.2f} {t_ref/t_sym:>7.2f}x")


if __name__ == "__main__":
    info = mx.device_info()
    print(f"device: {info['device_name']}  ({info['architecture']})")
    bench_syrk(mx.float32)
    bench_syrk(mx.bfloat16)
    bench_newton_schulz()
