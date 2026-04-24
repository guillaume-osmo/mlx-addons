"""Benchmark randomized SVD vs scipy ARPACK / sklearn / MLX full CPU SVD."""

import time
import numpy as np
import mlx.core as mx
from sklearn.decomposition import TruncatedSVD as SkTSVD

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


def bench(shape, k, n_iter=4, skip_arpack=False):
    """Run all SVD variants on one size.

    ``skip_arpack=True`` omits scipy ARPACK — at shapes beyond ~(5000, 1024)
    ARPACK takes 30+ s, which dominates the bench for no useful comparison
    (sklearn randomized is a fairer baseline).
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape).astype(np.float32)

    t_arpack = None
    if not skip_arpack:
        t_arpack = _time(lambda: SkTSVD(n_components=k, algorithm="arpack", random_state=42).fit(X))
    t_sk_rand = _time(lambda: SkTSVD(n_components=k, algorithm="randomized", random_state=42, n_iter=n_iter).fit(X))

    def mlx_full():
        with mx.stream(mx.cpu):
            U, S, Vt = mx.linalg.svd(mx.array(X))
            mx.eval(U, S, Vt)
    t_mlx_full = _time(mlx_full)

    t_ours = _time(lambda: randomized_svd(X, n_components=k, n_iter=n_iter, random_state=42))

    print(f"shape={shape}  k={k}")
    if t_arpack is not None:
        print(f"  scipy ARPACK          : {t_arpack*1000:8.1f} ms")
    print(f"  sklearn randomized    : {t_sk_rand*1000:8.1f} ms")
    print(f"  MLX full SVD (CPU)    : {t_mlx_full*1000:8.1f} ms")
    speedup_label = (
        f"{t_arpack/t_ours:.0f}× vs ARPACK" if t_arpack is not None
        else f"{t_sk_rand/t_ours:.0f}× vs sklearn randomized"
    )
    print(f"  mlx_addons rSVD (GPU) : {t_ours*1000:8.1f} ms   — {speedup_label}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("  randomized_svd: Metal matmul + CPU QR/tiny-SVD")
    print("=" * 70 + "\n")
    # ARPACK is prohibitively slow above ~5000×1024 (~30 s per call), so we
    # skip it for the large shapes and report vs sklearn randomized instead.
    for shape, k, skip_arpack in [
        ((633, 128), 32, False),
        ((2000, 512), 32, False),
        ((5000, 1024), 64, False),
        ((10000, 2048), 32, True),
    ]:
        bench(shape, k, skip_arpack=skip_arpack)
