# mlx-addons

GPU-accelerated operations for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Custom Metal compute kernels for operations that are missing or CPU-only in core MLX.

## Features

### `mlx_addons.linalg` — Batched Linear Algebra on GPU

Metal GPU kernels for Cholesky factorization and linear solve on batched small SPD matrices. **25-40x faster** than CPU LAPACK for batch sizes >= 1000.

```python
from mlx_addons.linalg import solve, cholesky

# Solve A @ x = b for 10,000 matrices at once — on GPU
A = mx.array(...)  # (10000, 15, 15) SPD matrices
b = mx.array(...)  # (10000, 15, 1)
x = solve(A, b)    # 0.9ms on GPU vs 22ms on CPU
```

| Batch | k  | CPU (ms) | GPU (ms) | Speedup |
|------:|---:|---------:|---------:|--------:|
| 10K   |  5 |      6.5 |     0.17 | **39x** |
| 10K   | 15 |     22.2 |     0.90 | **25x** |
| 10K   | 30 |     64.1 |     2.03 | **32x** |

**Functions:** `solve`, `cholesky`, `tril_solve`, `triu_solve`

Supports matrices up to 32x32 (stored in thread-local Metal registers). Larger matrices fall back to CPU.

### `mlx_addons.knn` — K-Nearest Neighbors on GPU

Z-order tree construction + Metal GPU kernels for batched distance computation and segmented top-k selection.

```python
from mlx_addons.knn import knn

pos = mx.random.normal((100000, 3))
distances, indices = knn(pos, k=16)
```

**Pipeline:** Morton encoding → Z-order sort → SoA tree build → GPU frontier walk → Metal segmented top-k

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
