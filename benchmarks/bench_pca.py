"""Benchmark mlx_addons.decomposition.PCA vs sklearn's PCA."""

import time
import numpy as np
from sklearn.decomposition import PCA as SkPCA

from mlx_addons.decomposition import PCA


def _time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench(shape, k):
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape).astype(np.float32)

    t_sk_full = _time(lambda: SkPCA(n_components=k, svd_solver="full", random_state=42).fit(X))
    t_sk_rand = _time(lambda: SkPCA(n_components=k, svd_solver="randomized", random_state=42).fit(X))
    t_ours = _time(lambda: PCA(n_components=k, random_state=42).fit(X))

    print(f"shape={shape}  k={k}")
    print(f"  sklearn full PCA      : {t_sk_full*1000:8.1f} ms")
    print(f"  sklearn randomized PCA: {t_sk_rand*1000:8.1f} ms")
    print(f"  mlx_addons PCA (GPU)  : {t_ours*1000:8.1f} ms   — {t_sk_rand/t_ours:.0f}× vs sklearn rand")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("  PCA: mean-center + randomized SVD (Metal matmul + CPU QR/tiny-SVD)")
    print("=" * 70 + "\n")
    for shape, k in [
        ((633, 128), 32),
        ((2000, 512), 32),
        ((5000, 1024), 64),
        ((10000, 2048), 32),
    ]:
        bench(shape, k)
