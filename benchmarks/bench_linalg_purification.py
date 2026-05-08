#!/usr/bin/env python3
"""Benchmark eigh-based vs SP2-purification density-matrix construction.

Sweeps matrix dimension N and compares wall-clock for:
- ``mlx_addons.linalg.batched_eigh`` (uses MLX eigh on CPU stream)
- ``mlx_addons.linalg.sp2_purify``   (Niklasson SP2 / TC2 on Metal GPU)

Output: a CSV with columns ``N, eigh_ms, sp2_ms, sp2_speedup``.

Background: the published SP2-vs-eigh crossover for dense matrices is at
N ≈ 1000-2000 (Cawkwell 2012 JCTC, Bock 2025 JCTC). For NDDO valence-sp
basis on small organics the basis sizes are 5-50 — eigh is expected to win
decisively. This bench quantifies the actual crossover on Apple silicon.
"""

import csv
import sys
import time

import mlx.core as mx
import numpy as np

from mlx_addons.linalg import batched_eigh, sp2_purify


def make_fock_like(N, seed=0):
    """Symmetric matrix with structured spectrum (occupied + virtual + gap)."""
    rng = np.random.default_rng(seed)
    n_occ = N // 4
    eigvals = np.concatenate([
        np.linspace(-3.0, -1.0, n_occ),
        np.linspace(1.0, 5.0, N - n_occ),
    ]).astype(np.float32)
    Q = np.linalg.qr(rng.standard_normal((N, N)).astype(np.float32))[0]
    A = Q @ np.diag(eigvals) @ Q.T
    return (A + A.T) / 2, n_occ


def bench_fn(fn, *args, warmup=3, repeat=10):
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    mx.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def bench_eigh(F):
    return batched_eigh(F)


def bench_sp2(F, n_occ):
    return sp2_purify(F, n_occ)


def main():
    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        Ns = [10, 50, 200, 1000]

    print(f"{'N':>6} {'n_occ':>6} {'eigh_ms':>10} {'sp2_ms':>10} {'sp2_speedup':>14}")
    print("-" * 50)
    rows = []
    for N in Ns:
        F_np, n_occ = make_fock_like(N, seed=N)
        F = mx.array(F_np)

        t_eigh = bench_fn(bench_eigh, F) * 1000
        t_sp2 = bench_fn(bench_sp2, F, n_occ) * 1000
        speedup = t_eigh / t_sp2 if t_sp2 > 0 else float("inf")
        rows.append((N, n_occ, t_eigh, t_sp2, speedup))
        print(f"{N:>6} {n_occ:>6} {t_eigh:>10.2f} {t_sp2:>10.2f} {speedup:>13.2f}x")

    out_path = "bench_purification.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "n_occ", "eigh_ms", "sp2_ms", "sp2_speedup"])
        w.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
