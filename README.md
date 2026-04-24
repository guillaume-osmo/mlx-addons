# mlx-addons

GPU-accelerated operations for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Custom Metal compute kernels for operations that are missing or CPU-only in core MLX.

## Features

### `mlx_addons.linalg` — Batched Linear Algebra on GPU

Metal GPU kernels for Cholesky, solve, QR, determinant, and triangular solve on batched matrices of **any size**. Three kernel tiers for maximum performance:

1. **Per-thread** (k <= 32): one thread per matrix, device memory. Best for small k with large batches.
2. **Threadgroup-cooperative** (k 33-80): threadgroup shared memory, parallel column updates. Best throughput for medium k.
3. **Blocked** (k > 80): tiled algorithm with GPU matmul for SYRK updates. Scales to any size.

```python
from mlx_addons.linalg import solve, cholesky, qr

# Solve A @ x = b for 10,000 matrices at once — on GPU
A = mx.array(...)  # (10000, 64, 64) SPD matrices
b = mx.array(...)  # (10000, 64, 1)
x = solve(A, b)    # 30ms on GPU vs 311ms on CPU

Q, R = qr(A_small) # Householder QR, up to 10-29x faster than CPU
```

#### Cholesky Solve (batch = 10K, M3 Max)

| k  | CPU (ms) | GPU (ms) | Speedup |
|---:|---------:|---------:|--------:|
|  5 |      7.7 |     0.63 | **12x** |
| 15 |     24.8 |     0.97 | **26x** |
| 30 |     72.0 |     2.20 | **33x** |
| 48 |    176.4 |    10.51 | **17x** |
| 64 |    310.7 |    29.62 | **10x** |
| 80 |    569.4 |    66.64 |  **9x** |

Scales to any matrix size via blocked algorithm (k > 80 uses 80-block Cholesky + GPU matmul):

| k (batch=1K) | CPU (ms) | GPU (ms) | Speedup |
|--------------:|---------:|---------:|--------:|
| 128           |    102.2 |     14.4 |  **7x** |
| 256           |    436.5 |     59.9 |  **7x** |
| 512           |   1691.1 |    222.3 |  **8x** |

#### QR Factorization (batch = 10K, M3 Max)

Uses threadgroup-cooperative Metal kernel with shared memory for both Q and R matrices. Parallelises Householder reflector application across rows and columns.

| k  | CPU (ms) | GPU (ms) | Speedup |
|---:|---------:|---------:|--------:|
| 16 |     26.4 |     0.85 | **31x** |
| 20 |     35.6 |     1.43 | **25x** |
| 32 |     75.1 |     7.55 | **10x** |
| 48 |    155.4 |    32.94 |  **5x** |
| 63 |    294.4 |    93.20 |  **3x** |

For k > 63, falls back to CPU LAPACK (the threadgroup memory limit of 32 KB fits two k*k matrices up to k=63).

**Functions:** `solve`, `cholesky`, `qr`, `tril_solve`, `triu_solve`, `det`, `slogdet`, `logdet_spd`

#### Randomized Truncated SVD (Halko-Martinsson-Tropp)

Metal GPU matmul for the range-finding, projection and lift steps; CPU MLX stream for the QR on the ``(n, k+p)`` basis and the small ``(k+p, m)`` final SVD. Subspace iteration with re-orthogonalization (Halko-Martinsson-Tropp Algorithm 4.4). Dramatically faster than scipy ARPACK for low-rank truncation:

| Matrix shape      | k  | scipy ARPACK | sklearn randomized | MLX full CPU SVD | **mlx_addons rSVD** |
|:------------------|---:|-------------:|-------------------:|-----------------:|--------------------:|
| (633, 128)        | 32 |      394 ms  |            501 ms  |           12 ms  |           **8 ms**  |
| (2000, 512)       | 32 |      13.1 s  |             1.4 s  |           92 ms  |          **12 ms**  |
| (5000, 1024)      | 64 |      21.6 s  |             3.9 s  |           928 ms |          **46 ms**  |
| (10000, 2048)     | 32 |  ~30 s†      |             3.4 s  |           4.4 s  |          **59 ms**  |

† ARPACK skipped above 5000×1024 in the default benchmark (takes 30+ s per call). All measurements on M3 Max with ``mx.clear_cache()`` between runs.

```python
from mlx_addons.linalg import randomized_svd, TruncatedSVD

U, S, Vt = randomized_svd(X, n_components=32, n_iter=4)

# sklearn-compatible class
svd = TruncatedSVD(n_components=32).fit(X)
Z = svd.transform(X)         # (n_samples, 32) low-rank projection
```

**Functions:** `randomized_svd`, `TruncatedSVD`

Batched input is supported: pass an array of shape ``(batch, n, m)`` and all matmuls (``X @ Ω``, ``X.mT @ Y``, ``Q.mT @ X``, ``Q @ Û``) go through a single Metal dispatch. The QR and final SVD run on MLX CPU stream but use batched LAPACK, so B independent low-rank truncations cost much less than B calls. Typical speedups vs a serial Python loop at (n=500, m=128, k=16):

| batch | serial loop | **batched** | speedup |
|------:|------------:|------------:|--------:|
|   1   |     3.8 ms  |    5.7 ms   |  0.66×  |
|   4   |    19.9 ms  |    6.7 ms   |  **3.0×**  |
|   8   |    34.7 ms  |    9.5 ms   |  **3.6×**  |
|  16   |    66.3 ms  |   12.2 ms   |  **5.4×**  |
|  32   |   128.3 ms  |   23.0 ms   |  **5.6×**  |

```python
# 8 independent truncations in one call
U, S, Vt = randomized_svd(X_batch, n_components=32)   # X_batch: (8, n, m)
```

### `mlx_addons.decomposition` — sklearn-style decompositions on GPU

#### Randomized PCA

Drop-in replacement for ``sklearn.decomposition.PCA`` backed by `randomized_svd`: mean-centering + Metal-accelerated SVD. Supports `transform` / `inverse_transform` / `whiten` / `explained_variance_ratio_`.

| Matrix shape      | k  | sklearn full | sklearn randomized | **mlx_addons PCA** |
|:------------------|---:|-------------:|-------------------:|-------------------:|
| (633, 128)        | 32 |     13 ms    |            371 ms  |          **8 ms**  |
| (2000, 512)       | 32 |    137 ms    |           3.7 s    |         **18 ms**  |
| (5000, 1024)      | 64 |    664 ms    |           7.4 s    |         **48 ms**  |
| (10000, 2048)     | 32 |    4.0 s     |           6.8 s    |         **85 ms**  |

```python
from mlx_addons.decomposition import PCA

pca = PCA(n_components=32, whiten=False, random_state=0).fit(X)
Z = pca.transform(X_new)                 # (n_samples, 32)
X_hat = pca.inverse_transform(Z)         # lossy reconstruction
print(pca.explained_variance_ratio_)     # descending
```

#### Nyström kernel approximation + KernelPCA

Drop-in replacements for ``sklearn.kernel_approximation.Nystroem`` and ``sklearn.decomposition.KernelPCA``. Kernel matrix construction (RBF / polynomial / linear / sigmoid) runs via Metal matmul: one ``X @ Y.T`` for the pairwise inner products plus elementwise ops for the kernel function. Eigendecomposition on the small (m × m) or (n × n) kernel matrix runs on MLX CPU stream via ``mx.linalg.eigh``.

| Method | shape | sklearn | **mlx_addons** | speedup |
|:-------|:------|--------:|---------------:|:-------:|
| Nystroem   | n=1000, d=20,  m=100  | 233 ms | **2 ms**  | **109×** |
| Nystroem   | n=5000, d=50,  m=300  | 332 ms | **6 ms**  | **54×**  |
| Nystroem   | n=10000, d=100, m=500 | 388 ms | **14 ms** | **27×**  |
| KernelPCA  | n=500,  d=10, k=20    | 310 ms | **20 ms** | **16×**  |
| KernelPCA  | n=1500, d=30, k=40    | 505 ms | **134 ms**| **3.8×** |
| KernelPCA  | n=3000, d=50, k=60    | 1.4 s  | **693 ms**| **2.0×** |

```python
from mlx_addons.decomposition import Nystroem, KernelPCA, pairwise_kernel

# Nyström: φ(x) Metal-accelerated feature map, Z Z.T ≈ K_rbf(X, X)
ny = Nystroem(n_components=100, kernel="rbf", gamma=0.1).fit(X_train)
Z_train = ny.transform(X_train)
Z_test  = ny.transform(X_test)

# Kernel PCA
kpca = KernelPCA(n_components=16, kernel="rbf", gamma=0.1).fit(X_train)
Z = kpca.transform(X_test)

# Raw kernel matrix if you need it
K = pairwise_kernel(X, Y, kernel="rbf", gamma=0.1)   # (N, M)
```

### `mlx_addons.knn` — K-Nearest Neighbors on GPU

Z-order tree construction + Metal GPU kernels for batched distance computation and segmented top-k selection. Supports up to **256 neighbors**.

```python
from mlx_addons.knn import knn

pos = mx.random.normal((100000, 3))
distances, indices = knn(pos, k=16)
```

**Pipeline:** Morton encoding -> Z-order sort -> SoA tree build -> GPU frontier walk -> Metal segmented top-k

## Install

```bash
pip install mlx-addons
```

Or from source:

```bash
git clone https://github.com/guillaume-osmo/mlx-addons.git
cd mlx-addons
pip install -e .
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- MLX >= 0.20.0

## License

MIT
