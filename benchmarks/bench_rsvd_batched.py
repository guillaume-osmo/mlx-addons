"""Benchmark batched randomized_svd vs a serial Python loop.

Batched path puts B independent SVDs through a single Metal dispatch for
the matmuls (``X @ Ω``, ``X.mT @ Y``, ``Q.mT @ X``, ``Q @ Û``) and uses
MLX's CPU batched QR/SVD. For small inputs where Metal matmul already
dominates, the speedup grows with batch size — 4–6× typical at B≥8.
"""

import time
import numpy as np

from mlx_addons.linalg import randomized_svd


def _time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench(batch, n, m, k, n_iter=4):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((batch, n, m)).astype(np.float32)

    t_batched = _time(
        lambda: randomized_svd(X, n_components=k, n_iter=n_iter, random_state=42)
    )

    def serial():
        for i in range(batch):
            randomized_svd(X[i], n_components=k, n_iter=n_iter, random_state=42)
    t_serial = _time(serial)

    print(
        f"batch={batch:>3}  n={n:>5}  m={m:>5}  k={k:>3}: "
        f"serial={t_serial*1000:8.1f} ms  batched={t_batched*1000:7.1f} ms  "
        f"speedup={t_serial/t_batched:.2f}×"
    )


if __name__ == "__main__":
    print("=" * 78)
    print("  batched randomized_svd vs serial Python loop  (M3 Max, float32)")
    print("=" * 78)
    print()

    # Hold (n, m, k) fixed and sweep batch — shows scaling with batch size.
    print("[small matrices — Metal matmul dominates]")
    for B in [1, 4, 8, 16, 32]:
        bench(batch=B, n=500, m=128, k=16)
    print()

    print("[medium matrices — QR overhead starts to matter]")
    for B in [1, 4, 8, 16]:
        bench(batch=B, n=2000, m=512, k=32)
    print()

    print("[TabPFN-scale: 8 estimator ensemble per molecular dataset]")
    bench(batch=8, n=633, m=128, k=32)
    bench(batch=8, n=1500, m=128, k=32)
    bench(batch=8, n=2000, m=192, k=32)
