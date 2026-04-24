"""Benchmark mlx_addons.decomposition.{Gaussian,Sparse}RandomProjection + csr_matmul."""

import time
import numpy as np
from sklearn.random_projection import GaussianRandomProjection as SkGRP
from sklearn.random_projection import SparseRandomProjection as SkSRP

from mlx_addons.decomposition import GaussianRandomProjection, SparseRandomProjection
from mlx_addons.linalg import csr_matmul, csr_from_dense


def _time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_grp(n, d, k):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    t_sk = _time(lambda: SkGRP(n_components=k, random_state=0).fit(X).transform(X))
    t_ours = _time(lambda: GaussianRandomProjection(n_components=k, random_state=0).fit(X).transform(X))
    print(f"  Gaussian  n={n:>5}  d={d:>5}  k={k:>3}: sklearn {t_sk*1000:7.1f} ms  ours {t_ours*1000:6.1f} ms   {t_sk/t_ours:.1f}×")


def bench_srp(n, d, k):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    t_sk = _time(lambda: SkSRP(n_components=k, random_state=0).fit(X).transform(X))
    t_dense = _time(
        lambda: SparseRandomProjection(
            n_components=k, random_state=0, store_sparse=False,
        ).fit(X).transform(X)
    )
    t_sparse = _time(
        lambda: SparseRandomProjection(
            n_components=k, random_state=0, store_sparse=True,
        ).fit(X).transform(X)
    )
    print(
        f"  Sparse    n={n:>5}  d={d:>5}  k={k:>3}: sklearn {t_sk*1000:7.1f} ms  "
        f"ours-dense {t_dense*1000:6.1f} ms  ours-csr {t_sparse*1000:6.1f} ms"
    )


def bench_spmm(M, K, N, density):
    """CSR × dense matmul vs dense A @ B."""
    rng = np.random.default_rng(0)
    A_dense = rng.standard_normal((M, K)).astype(np.float32)
    A_dense *= (rng.random(A_dense.shape) < density)
    B = rng.standard_normal((K, N)).astype(np.float32)

    t_dense = _time(lambda: A_dense @ B)

    indptr, indices, values = csr_from_dense(A_dense)
    nnz = int(values.shape[0])
    # Warmup csr kernel
    _ = csr_matmul(indptr, indices, values, B, M=M)
    t_sparse = _time(lambda: csr_matmul(indptr, indices, values, B, M=M))

    print(
        f"  SpMM     M={M:>5}  K={K:>5}  N={N:>3}  density={density:5.2%}  nnz={nnz:>8}: "
        f"dense {t_dense*1000:6.1f} ms  csr {t_sparse*1000:6.1f} ms  {t_dense/t_sparse:.2f}×"
    )


if __name__ == "__main__":
    print("=" * 100)
    print("  GaussianRandomProjection  (fit + transform)")
    print("=" * 100)
    for n, d, k in [(1000, 500, 64), (5000, 2048, 128), (10000, 4096, 128), (1000, 16384, 256)]:
        bench_grp(n, d, k)

    print("\n" + "=" * 100)
    print("  SparseRandomProjection  (fit + transform, density='auto' = 1/sqrt(d))")
    print("=" * 100)
    for n, d, k in [(1000, 2048, 128), (5000, 4096, 128), (1000, 16384, 256), (1000, 65536, 256)]:
        bench_srp(n, d, k)

    print("\n" + "=" * 100)
    print("  csr_matmul (CSR × dense) vs dense matmul")
    print("=" * 100)
    for M, K, N, density in [
        (1000, 2048, 64, 0.02),
        (1000, 8192, 64, 0.01),
        (5000, 16384, 64, 0.005),
        (10000, 32768, 128, 0.002),
    ]:
        bench_spmm(M, K, N, density)
