#!/usr/bin/env python3
"""Benchmark linalg kernels across all three tiers."""

import time
import numpy as np
import mlx.core as mx
from mlx_addons.linalg import solve, cholesky
from mlx_addons.linalg._metal_kernels import (
    solve as metal_solve,
    cholesky as metal_cholesky,
    SHARED_K_MAX,
    MAX_GPU_K,
)


def make_spd(batch, k, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((batch, k, k)).astype(np.float32)
    A = A @ A.transpose(0, 2, 1) + np.eye(k, dtype=np.float32) * float(k)
    return mx.array(A)


def make_rhs(batch, k, m=1, seed=43):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((batch, k, m)).astype(np.float32))


def bench(fn, *args, warmup=5, repeat=20):
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def tier_label(k):
    if k <= 32:
        return "tier1-ST"
    elif k <= SHARED_K_MAX:
        return "tier2-TG"
    else:
        return "tier3-LG"


print(f"{'op':>10} {'batch':>6} {'k':>4} {'m':>2} {'tier':>10} {'ms':>8}")
print("-" * 50)

# Benchmark metal_solve (direct kernel, no blocked layer)
for k in [5, 10, 15, 20, 30, 32, 48, 64, 80, 96, 128]:
    for batch in [1000, 10000]:
        if k > MAX_GPU_K:
            continue
        A = make_spd(batch, k)
        b = make_rhs(batch, k)
        t = bench(metal_solve, A, b)
        print(f"{'solve':>10} {batch:>6} {k:>4} {1:>2} {tier_label(k):>10} {t*1000:>8.2f}")

print()

# Benchmark metal_cholesky
for k in [5, 10, 20, 30, 32, 48, 64, 80, 96, 128]:
    for batch in [1000, 10000]:
        if k > MAX_GPU_K:
            continue
        A = make_spd(batch, k)
        t = bench(metal_cholesky, A)
        print(f"{'cholesky':>10} {batch:>6} {k:>4} {'':>2} {tier_label(k):>10} {t*1000:>8.2f}")

print()

# Compare CPU vs GPU for solve
print("\nCPU vs GPU comparison (solve):")
print(f"{'k':>4} {'batch':>6} {'CPU ms':>8} {'GPU ms':>8} {'speedup':>8}")
print("-" * 40)
for k in [10, 20, 30, 48, 64, 80]:
    batch = 5000
    A = make_spd(batch, k)
    b = make_rhs(batch, k)

    # CPU
    def cpu_solve(A, b):
        return mx.linalg.solve(A, b, stream=mx.cpu)
    t_cpu = bench(cpu_solve, A, b)

    # GPU
    t_gpu = bench(metal_solve, A, b)

    print(f"{k:>4} {batch:>6} {t_cpu*1000:>8.2f} {t_gpu*1000:>8.2f} {t_cpu/t_gpu:>7.1f}x")
