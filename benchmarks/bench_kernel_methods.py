"""Benchmark Nystroem + KernelPCA vs sklearn."""

import time
import numpy as np
from sklearn.decomposition import KernelPCA as SkKPCA
from sklearn.kernel_approximation import Nystroem as SkNystroem

from mlx_addons.decomposition import Nystroem, KernelPCA, pairwise_kernel


def _time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_nystroem(n, d, m):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)

    t_sk = _time(lambda: SkNystroem(n_components=m, kernel="rbf", gamma=0.1, random_state=0).fit(X))
    t_ours = _time(lambda: Nystroem(n_components=m, kernel="rbf", gamma=0.1, random_state=0).fit(X))
    print(f"  Nystroem   (n={n}, d={d}, m={m}): sklearn {t_sk*1000:7.1f} ms  ours {t_ours*1000:6.1f} ms  {t_sk/t_ours:.1f}×")


def bench_kpca(n, d, k):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)

    t_sk = _time(lambda: SkKPCA(n_components=k, kernel="rbf", gamma=0.1).fit(X), warmup=1, repeat=2)
    t_ours = _time(lambda: KernelPCA(n_components=k, kernel="rbf", gamma=0.1).fit(X))
    print(f"  KernelPCA  (n={n}, d={d}, k={k}): sklearn {t_sk*1000:7.1f} ms  ours {t_ours*1000:6.1f} ms  {t_sk/t_ours:.1f}×")


if __name__ == "__main__":
    print("=" * 78)
    print("  Kernel methods: Nyström / KernelPCA  (RBF, M3 Max, float32)")
    print("=" * 78)
    print("\n[Nystroem fit]")
    for n, d, m in [(1000, 20, 100), (5000, 50, 300), (10000, 100, 500)]:
        bench_nystroem(n, d, m)

    print("\n[KernelPCA fit]  (O(n²) kernel matrix — best for n ≲ 3000)")
    for n, d, k in [(500, 10, 20), (1500, 30, 40), (3000, 50, 60)]:
        bench_kpca(n, d, k)
