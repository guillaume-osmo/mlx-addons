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
