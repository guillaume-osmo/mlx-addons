#!/usr/bin/env python3
"""Benchmark Pulay DIIS extrapolation cost vs plain Picard on a synthetic
fixed-point problem.

Setup: solve ``x = M x + b`` for various matrix sizes. ``M`` constructed
with controlled spectral radius < 1 so plain Picard converges. Reports:

- iterations to ``||x - x*|| < 1e-6`` for plain Picard
- iterations to same tolerance with DIIS extrapolation
- wall-clock for one DIIS solve (the per-iteration overhead)
"""

import time

import mlx.core as mx
import numpy as np

from mlx_addons.solvers import pulay_diis


def synthetic_problem(N, spectral_radius=0.9, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    eigvals = rng.uniform(-spectral_radius, spectral_radius, size=N)
    eigvals[0] = spectral_radius
    M = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(N)
    return M.astype(np.float32), b.astype(np.float32)


def picard(M, b, tol=1e-6, max_iter=2000):
    x = np.zeros_like(b)
    for it in range(max_iter):
        x_new = M @ x + b
        if np.linalg.norm(x_new - x) < tol:
            return it + 1, x_new
        x = x_new
    return max_iter, x


def picard_diis(M, b, history=6, tol=1e-6, max_iter=2000):
    N = b.shape[0]
    x = np.zeros_like(b)
    F_hist, e_hist = [], []
    # Build initial history with plain Picard.
    for _ in range(2):
        x_new = M @ x + b
        F_hist.append(mx.array(x_new.reshape(N, 1)))
        e_hist.append(mx.array((x_new - x).reshape(N, 1)))
        x = x_new
    n_iter = 2
    for _ in range(max_iter):
        F_extrap = pulay_diis(F_hist, e_hist, max_history=history)
        mx.eval(F_extrap)
        x = np.array(F_extrap).reshape(N)
        x_new = M @ x + b
        F_hist.append(mx.array(x_new.reshape(N, 1)))
        e_hist.append(mx.array((x_new - x).reshape(N, 1)))
        n_iter += 1
        if np.linalg.norm(x_new - x) < tol:
            return n_iter, x_new
        x = x_new
    return n_iter, x


def main():
    print(f"{'N':>5} {'rho':>6} {'picard_it':>11} {'diis_it':>9} {'speedup':>9}")
    print("-" * 50)
    for N in [16, 32, 64, 128, 256]:
        for rho in [0.7, 0.9, 0.95]:
            M, b = synthetic_problem(N, spectral_radius=rho, seed=N + int(rho * 100))
            it_p, _ = picard(M, b)
            it_d, _ = picard_diis(M, b)
            speedup = it_p / max(it_d, 1)
            print(f"{N:>5} {rho:>6.2f} {it_p:>11d} {it_d:>9d} {speedup:>8.2f}x")

    # Per-call cost of one DIIS extrapolation (the overhead).
    print("\nPer-call cost of pulay_diis:")
    for N in [16, 64, 256, 1024]:
        F_hist = [mx.array(np.random.randn(N, 1).astype(np.float32)) for _ in range(6)]
        e_hist = [mx.array(np.random.randn(N, 1).astype(np.float32)) for _ in range(6)]
        # warmup
        for _ in range(3):
            mx.eval(pulay_diis(F_hist, e_hist))
        mx.synchronize()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = pulay_diis(F_hist, e_hist)
            mx.eval(out)
            mx.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"  N={N:>5}: {sorted(times)[10] * 1000:>6.3f} ms")


if __name__ == "__main__":
    main()
