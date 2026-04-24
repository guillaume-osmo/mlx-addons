"""Benchmark mlx_addons.cluster.KMeans vs sklearn."""

import time
import numpy as np
from sklearn.cluster import KMeans as SkKMeans

from mlx_addons.cluster import KMeans


def _time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench(n, d, k):
    rng = np.random.default_rng(0)
    # Structured blobs (not pure noise) — more realistic workload
    centers = rng.uniform(-10, 10, size=(k, d)).astype(np.float32)
    labels = rng.integers(0, k, size=n)
    X = centers[labels] + rng.standard_normal((n, d)).astype(np.float32) * 0.5

    t_sk = _time(lambda: SkKMeans(n_clusters=k, n_init=3, random_state=0).fit(X), warmup=1, repeat=2)
    t_ours = _time(lambda: KMeans(n_clusters=k, n_init=3, random_state=0).fit(X), warmup=1, repeat=2)
    print(
        f"  n={n:>6}  d={d:>4}  k={k:>3}: sklearn {t_sk*1000:8.1f} ms   "
        f"ours {t_ours*1000:7.1f} ms   {t_sk/t_ours:.1f}×"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  KMeans fit (Lloyd's + k-means++, Metal matmul, M3 Max, float32)")
    print("=" * 70)
    for n, d, k in [
        (1000, 16, 8),
        (5000, 32, 16),
        (10000, 64, 32),
        (50000, 64, 32),
        (100000, 128, 64),
    ]:
        bench(n, d, k)
