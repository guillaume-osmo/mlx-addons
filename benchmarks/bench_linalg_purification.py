#!/usr/bin/env python3
"""Benchmark eigh-based vs SP2-purification density-matrix construction.

Two sweep modes:

- ``--mode N``     — single-matrix sweep over matrix size N (per-call cost)
- ``--mode batch`` — batched sweep over batch size B at fixed k (the realistic
                     mlxmolkit workload: many molecules, small Fock per mol)

Compares:
- ``numpy_loop`` (CPU)             — current per-molecule LAPACK loop
- ``mlx_addons.linalg.batched_eigh`` (CPU stream, MLX-batched LAPACK)
- ``mlx_addons.linalg.sp2_purify``   (Niklasson SP2 / TC2 on Metal GPU)

Background: the published SP2-vs-eigh crossover for dense matrices is at
N ≈ 1000-2000 (Cawkwell 2012 JCTC, Bock 2025 JCTC) — that's the per-call
crossover. For *batched* small-N workloads (mlxmolkit's regime: hundreds
of molecules with NDDO valence-sp basis ~k=30), SP2's launch-overhead is
amortized across the batch and the crossover happens around B ≈ 500.
"""

import argparse
import csv
import time

import mlx.core as mx
import numpy as np

from mlx_addons.linalg import batched_eigh, sp2_purify


def make_fock_like_single(N, seed=0):
    rng = np.random.default_rng(seed)
    n_occ = N // 4
    eigvals = np.concatenate([
        np.linspace(-3.0, -1.0, n_occ),
        np.linspace(1.0, 5.0, N - n_occ),
    ]).astype(np.float32)
    Q = np.linalg.qr(rng.standard_normal((N, N)).astype(np.float32))[0]
    A = Q @ np.diag(eigvals) @ Q.T
    return (A + A.T) / 2, n_occ


def make_fock_like_batch(B, k, seed=0):
    rng = np.random.default_rng(seed)
    n_occ = k // 4
    eigvals = np.concatenate([
        np.linspace(-3.0, -1.0, n_occ),
        np.linspace(1.0, 5.0, k - n_occ),
    ]).astype(np.float32)
    Qs = np.linalg.qr(rng.standard_normal((B, k, k)).astype(np.float32))[0]
    A = np.stack([Qs[i] @ np.diag(eigvals) @ Qs[i].T for i in range(B)])
    return ((A + A.transpose(0, 2, 1)) / 2).astype(np.float32), n_occ


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


def numpy_loop_density(A_np, n_occ):
    out = np.empty_like(A_np)
    if A_np.ndim == 2:
        w, v = np.linalg.eigh(A_np)
        return 2 * v[:, :n_occ] @ v[:, :n_occ].T
    for i in range(A_np.shape[0]):
        w, v = np.linalg.eigh(A_np[i])
        out[i] = 2 * v[:, :n_occ] @ v[:, :n_occ].T
    return out


def bench_numpy_loop(A_np, n_occ, warmup=3, repeat=5):
    for _ in range(warmup):
        _ = numpy_loop_density(A_np, n_occ)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = numpy_loop_density(A_np, n_occ)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def sweep_N(quick=False):
    Ns = [10, 50, 200, 1000] if quick else [10, 20, 50, 100, 200, 500, 1000, 2000]
    print(f"\n=== Single-matrix sweep (varying N) ===")
    print(f"{'N':>6} {'n_occ':>6} {'np_ms':>10} {'mxh_ms':>10} {'sp2_ms':>10} {'sp2/np':>10}")
    print("-" * 60)
    rows = []
    for N in Ns:
        F_np, n_occ = make_fock_like_single(N, seed=N)
        F = mx.array(F_np)
        t_np = bench_numpy_loop(F_np, n_occ) * 1000
        t_eigh = bench_fn(batched_eigh, F) * 1000
        t_sp2 = bench_fn(sp2_purify, F, n_occ) * 1000
        rows.append((N, n_occ, t_np, t_eigh, t_sp2, t_np / t_sp2 if t_sp2 > 0 else 0))
        print(f"{N:>6} {n_occ:>6} {t_np:>10.2f} {t_eigh:>10.2f} {t_sp2:>10.2f} {t_np / t_sp2:>9.2f}x")
    return rows, ["N", "n_occ", "numpy_ms", "mx_eigh_cpu_ms", "sp2_metal_ms", "sp2_vs_numpy"]


def sweep_batch(k=30, quick=False):
    Bs = [1, 10, 100, 1000] if quick else [1, 10, 50, 100, 250, 500, 1000, 2500]
    print(f"\n=== Batch sweep (k={k}, varying B = number of molecules) ===")
    print(f"{'B':>6} {'k':>4} {'np_loop_ms':>12} {'mxh_cpu_ms':>12} {'sp2_metal_ms':>14} {'sp2/np':>10}")
    print("-" * 70)
    rows = []
    for B in Bs:
        A_np, n_occ = make_fock_like_batch(B, k, seed=B)
        F = mx.array(A_np)
        t_np = bench_numpy_loop(A_np, n_occ) * 1000
        t_eigh = bench_fn(batched_eigh, F) * 1000
        t_sp2 = bench_fn(sp2_purify, F, n_occ) * 1000
        rows.append((B, k, n_occ, t_np, t_eigh, t_sp2, t_np / t_sp2 if t_sp2 > 0 else 0))
        print(f"{B:>6} {k:>4} {t_np:>12.2f} {t_eigh:>12.2f} {t_sp2:>14.2f} {t_np / t_sp2:>9.2f}x")
    return rows, ["B", "k", "n_occ", "numpy_loop_ms", "mx_eigh_cpu_ms", "sp2_metal_ms", "sp2_vs_numpy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["N", "batch", "both"], default="both")
    p.add_argument("--k", type=int, default=30, help="basis size for batch sweep")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.mode in ("N", "both"):
        rows, header = sweep_N(quick=args.quick)
        with open("bench_purification_N.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("Wrote bench_purification_N.csv")

    if args.mode in ("batch", "both"):
        rows, header = sweep_batch(k=args.k, quick=args.quick)
        with open("bench_purification_batch.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("Wrote bench_purification_batch.csv")


if __name__ == "__main__":
    main()
