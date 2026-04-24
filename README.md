# mlx-addons

GPU-accelerated operations for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Custom Metal compute kernels for operations that are missing or CPU-only in core MLX.

## Patterns & recipes

Three recipes that come up repeatedly — each documented with a measured improvement on real data.

### 1. Swap `sklearn.decomposition.PCA` → `mlx_addons.decomposition.PCA`

Drop-in replacement backed by Metal randomized SVD.

```python
# Before:
from sklearn.decomposition import PCA
# After:
from mlx_addons.decomposition import PCA
```

**Measured on ChemeleonSMD fingerprints (35633 × 2048 float32):**

| PCA dim | sklearn PCA | mlx_addons PCA | speedup |
|--------:|------------:|---------------:|:-------:|
|      64 |      9.4 s  |        464 ms  | **20×** |
|     128 |      9.7 s  |        651 ms  | **15×** |
|     192 |     10.1 s  |        893 ms  | **11×** |

### 2. Ensemble random projection for robustness + accuracy

Single PCA is sensitive to noisy top eigenvectors on ill-conditioned data; single random projection has high variance; **averaging a few of each hits a lower RMSE than either alone**.

```python
from mlx_addons.decomposition import EnsembleRandomProjection, ensemble_mean_predict

ens = EnsembleRandomProjection(
    n_components=128, n_pca=1, n_sparse=2, n_gaussian=2, random_state=42,
).fit(X_all_molecules)     # global, unsupervised

def fit_predict(Ztr, ytr, Zte):
    return TabICLRegressorMLX(n_estimators=8, random_state=42).fit(Ztr, ytr).predict(Zte).flatten()

y_pred = ensemble_mean_predict(ens, fit_predict, X_train, y_train, X_test)
```

**MoleculeACE ChemeleonSMD v5 × PCA=128 × TabICL-MLX (first 10 targets, mean RMSE):**

| Feature transform               | Mean RMSE | Δ vs PCA |
|---------------------------------|----------:|---------:|
| PCA (single)                    | 0.6281    | —        |
| SparseRP × 5                    | 0.6239    | −0.0042  |
| GaussianRP × 5                  | 0.6241    | −0.0040  |
| **PCA + 2 SRP + 2 GRP (mix)**   | **0.6215**| **−0.0066** |

Why it works — PCA captures high-variance directions; RPs preserve distances uniformly (JL lemma); averaging cancels per-basis variance. See [`benchmarks/bench_ensemble_rp.py`](benchmarks/bench_ensemble_rp.py).

### 3. Use `csr_matmul` when your left operand is sparse + wide

Writing `A @ B` as dense in MLX gives Metal's optimized GEMM but does all the zero multiplies. For genuine sparse workloads (graph Laplacians, message passing, one-hot features), the CSR path skips them.

```python
from mlx_addons.linalg import csr_matmul, csr_from_dense

indptr, indices, values = csr_from_dense(A)
C = csr_matmul(indptr, indices, values, B, M=A.shape[0])
```

Measured up to **622× faster than dense matmul** at 0.2% density on (10000, 32768) × (32768, 128). Scales with ``nnz``, not ``M × K``.

---

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

#### Random projection (Johnson-Lindenstrauss)

sklearn-compatible ``GaussianRandomProjection`` and ``SparseRandomProjection`` (Achlioptas / Li). Projection is one Metal matmul; ``SparseRandomProjection`` can optionally keep its matrix in CSR format (``store_sparse=True``) for downstream GNN-shape SpMM, but the dense path is faster at RP-typical shapes (small ``k``, large ``n_samples``).

| shape                    | k   | sklearn Gaussian | **ours** | speedup |
|:-------------------------|----:|-----------------:|---------:|--------:|
| n=5000,  d=2048          | 128 |            17 ms |   **8 ms** |  **2.2×** |
| n=10000, d=4096          | 128 |            56 ms |  **12 ms** |  **4.6×** |
| n=1000,  d=16384         | 256 |            98 ms |  **32 ms** |  **3.1×** |

```python
from mlx_addons.decomposition import GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim

# k chosen automatically to preserve pairwise distances to (1 ± eps)
rp = GaussianRandomProjection(n_components="auto", eps=0.2, random_state=0).fit(X)
Z = rp.transform(X)

# Ternary {-s, 0, +s} sparse matrix (Achlioptas density = 1/sqrt(d) by default)
rp = SparseRandomProjection(n_components=128, random_state=0).fit(X)
```

#### EnsembleRandomProjection — mixed PCA + RP ensemble

```python
from mlx_addons.decomposition import EnsembleRandomProjection, ensemble_mean_predict

ens = EnsembleRandomProjection(
    n_components=128, n_pca=1, n_sparse=2, n_gaussian=2, random_state=42,
).fit(X_all)                                    # fits 5 feature maps at once
Z = ens.transform(X_all)                        # (5, n_samples, 128) stacked views

# Typical pattern: fit your regressor on each member, average predictions.
def fit_predict(Ztr, ytr, Zte):
    return YourRegressor(...).fit(Ztr, ytr).predict(Zte)

y_pred = ensemble_mean_predict(ens, fit_predict, X_train, y_train, X_test)
```

**Measured improvement on MoleculeACE** (ChemeleonSMD v5 fingerprints → 128-d → TabICL-MLX, first 10 targets):

| Feature transform | Mean RMSE | Δ vs PCA |
|:------------------|----------:|---------:|
| PCA (single)                    | 0.6281 | — |
| SparseRP × 5 (seeds averaged)   | 0.6239 | −0.0042 |
| GaussianRP × 5                  | 0.6241 | −0.0040 |
| **PCA + 2 SRP + 2 GRP (default recipe)** | **0.6215** | **−0.0066** ✅ |

PCA wins individually on large, well-conditioned datasets (CHEMBL204, 214); RP ensembles win on small / ill-conditioned ones (CHEMBL1871 −0.028, CHEMBL1862 −0.034). The mix captures both sides.

#### Sparse matmul (CSR × dense) on Metal

A general-purpose SpMM primitive, not tied to random projection. One Metal thread per output element; scales with ``nnz``, not ``M × K``. Dramatic wins on highly-sparse wide inputs (think graph Laplacians, GNN message passing, one-hot feature matrices):

| (M, K, N) | density | nnz | dense matmul | **csr_matmul** | speedup |
|:----------|:-------:|----:|-------------:|---------------:|--------:|
| (1000, 2048, 64)    | 2.0% |   41k |   1.5 ms |  **0.1 ms** |   **27×** |
| (1000, 8192, 64)    | 1.0% |   82k |   8.8 ms |  **0.1 ms** |  **167×** |
| (5000, 16384, 64)   | 0.5% |  409k |    65 ms |  **0.1 ms** |  **535×** |
| (10000, 32768, 128) | 0.2% |  655k |   219 ms |  **0.4 ms** |  **622×** |

```python
from mlx_addons.linalg import csr_matmul, csr_from_dense
indptr, indices, values = csr_from_dense(A)       # (M, K) dense → CSR
C = csr_matmul(indptr, indices, values, B, M=A.shape[0])   # (M, K) @ (K, N) → (M, N)
```

### `mlx_addons.cluster` — Clustering on GPU

#### KMeans (Lloyd's algorithm, k-means++ init)

Assignment = one Metal matmul + ``argmin``; update = one-hot ``.T @ X`` matmul. No custom kernels — the whole Lloyd loop is expressed in MLX ops.

| Shape                    | sklearn | **mlx_addons** | speedup |
|:-------------------------|--------:|---------------:|:-------:|
| n=1000,   d=16,  k=8     |   15 ms |    25 ms       | 0.6×    |
| n=5000,   d=32,  k=16    |   42 ms |    36 ms       | 1.2×    |
| n=10000,  d=64,  k=32    |  351 ms |    79 ms       | **4.4×** |
| n=50000,  d=64,  k=32    |  1.2 s  |   133 ms       | **8.8×** |
| n=100000, d=128, k=64    |  7.4 s  |   465 ms       | **16×**  |

```python
from mlx_addons.cluster import KMeans

km = KMeans(n_clusters=32, n_init=3, random_state=0).fit(X)
labels = km.labels_                 # (n_samples,)
centers = km.cluster_centers_       # (n_clusters, n_features)
new_labels = km.predict(X_new)
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
