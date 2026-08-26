# Benchmarks

Every measured table for [mlx-addons](../README.md), with its hardware and
methodology. The [README's crossover table](../README.md#read-this-before-you-benchmark-it)
summarises where each path stops winning; this file is the evidence.

**Methodology.** Unless a table says otherwise: Apple M3 Max, `mx.clear_cache()`
between runs, float32. The SYRK numbers are M4 Pro. Below 4096 the
sub-millisecond shapes swing ±15% at `repeat=10` — use `repeat ≥ 25`, or you
will measure noise and report it as a regression.

---

## linalg

### Cholesky solve — batch = 10,000

| k | CPU (ms) | GPU (ms) | Speedup |
|---:|---:|---:|---:|
| 5 | 7.7 | 0.63 | **12×** |
| 15 | 24.8 | 0.97 | **26×** |
| 30 | 72.0 | 2.20 | **33×** |
| 48 | 176.4 | 10.51 | **17×** |
| 64 | 310.7 | 29.62 | **10×** |
| 80 | 569.4 | 66.64 | **9×** |

Above k = 80 the blocked algorithm takes over (80-block Cholesky + GPU matmul),
so there is no size ceiling — batch = 1,000:

| k | CPU (ms) | GPU (ms) | Speedup |
|---:|---:|---:|---:|
| 128 | 102.2 | 14.4 | **7×** |
| 256 | 436.5 | 59.9 | **7×** |
| 512 | 1691.1 | 222.3 | **8×** |

### QR factorization — batch = 10,000

Threadgroup-cooperative Metal kernel, shared memory for both Q and R,
Householder reflectors applied in parallel across rows and columns.

| k | CPU (ms) | GPU (ms) | Speedup |
|---:|---:|---:|---:|
| 16 | 26.4 | 0.85 | **31×** |
| 20 | 35.6 | 1.43 | **25×** |
| 32 | 75.1 | 7.55 | **10×** |
| 48 | 155.4 | 32.94 | **5×** |
| 63 | 294.4 | 93.20 | **3×** |

**k > 63 falls back to CPU LAPACK.** The 32 KB threadgroup limit fits two k×k
matrices up to k = 63.

### Randomized truncated SVD (Halko–Martinsson–Tropp)

Metal matmul for range-finding, projection and lift; MLX CPU stream for the QR
on the `(n, k+p)` basis and the small `(k+p, m)` final SVD. Subspace iteration
with re-orthogonalization (HMT Algorithm 4.4).

| Matrix shape | k | scipy ARPACK | sklearn randomized | MLX full CPU SVD | **rSVD** |
|:---|---:|---:|---:|---:|---:|
| (633, 128) | 32 | 394 ms | 501 ms | 12 ms | **8 ms** |
| (2000, 512) | 32 | 13.1 s | 1.4 s | 92 ms | **12 ms** |
| (5000, 1024) | 64 | 21.6 s | 3.9 s | 928 ms | **46 ms** |
| (10000, 2048) | 32 | ~30 s † | 3.4 s | 4.4 s | **59 ms** |

† ARPACK is skipped above 5000×1024 in the default benchmark — 30+ s per call.

**Batched**, at (n = 500, m = 128, k = 16). B independent truncations cost far
less than B calls, because all four matmuls go through one Metal dispatch and
the QR/SVD use batched LAPACK:

| batch | serial loop | **batched** | speedup |
|---:|---:|---:|---:|
| 1 | 3.8 ms | 5.7 ms | 0.66× |
| 4 | 19.9 ms | 6.7 ms | **3.0×** |
| 8 | 34.7 ms | 9.5 ms | **3.6×** |
| 16 | 66.3 ms | 12.2 ms | **5.4×** |
| 32 | 128.3 ms | 23.0 ms | **5.6×** |

**Batch 1 loses.** Use a serial loop below batch 4.

### `jacobi_eigh` vs MLX's CPU-stream `eigh`

Best of the two kernels, by batch size B:

| B | k=8 | k=16 | k=24 | k=32 |
|---:|---:|---:|---:|---:|
| 100 | 0.44× | 0.81× | 0.83× | 0.80× |
| 500 | **3.4×** | **1.4×** | **1.5×** | **1.3×** |
| 1000 | **3.3×** | **2.9×** | **1.7×** | **1.5×** |
| 2000 | **6.8×** | **5.5×** | **2.7×** | **1.6×** |

Below B ≈ 100 launch overhead dominates and the CPU stream wins at every k.
Above ≈ 500 GPU Jacobi pulls ahead and the gap widens with B. The sweet spot is
batched semiempirical SCF (mlxmolkit's RM1/AM1, xTB GFN0/1/2 — k = 5..32,
B = hundreds to thousands).

### SP2 purification vs dense `eigh`

The published crossover for GPU SP2 vs LAPACK is N ≈ 1000–2000. Measured here:

| N | n_occ | `eigh` (ms) | `sp2_purify` (ms) | speedup |
|---:|---:|---:|---:|---:|
| 10 | 2 | 0.21 | 7.04 | 0.03× |
| 50 | 12 | 0.16 | 6.10 | 0.03× |
| 200 | 50 | 2.12 | 17.23 | 0.12× |
| 1000 | 250 | 83.69 | 33.73 | **2.48×** |

Below ~500 basis functions `eigh` wins decisively — per-launch overhead
dominates the tiny matmuls. Above ~1000, SP2's matmul-only inner loop pulls
ahead. Use SP2 when N is large, or when an eigendecomposition is otherwise
unavailable.

### `syrk` — symmetric rank-k update

**M4 Pro, float32.** MLX dispatches `X @ X.T` to the same general GEMM as
`X @ Y` and takes no symmetry saving (bf16, 8192²: 151.2 ms vs 149.4 ms).
Blocking into `k` row-blocks costs `(1 + 1/k)/2` of the flops:

| shape | dense GEMM | **syrk** | speedup | blocks |
|:---|---:|---:|---:|---:|
| (2048, 4096) | 5.57 ms | 4.48 ms | 1.24× | 2 |
| (4096, 4096) | 21.51 ms | 14.93 ms | 1.44× | 4 |
| (8192, 4096) | 86.03 ms | 54.67 ms | 1.57× | 8 |
| (8192, 8192) | 208.31 ms | 123.79 ms | **1.68×** | 8 |

bfloat16 is slightly better (1.72× at 8192²). Gains need `M ≥ MIN_DIM` (2048)
**and** contraction `≥ MIN_CONTRACT` (1024); thin-K blocks *lose* — 0.65–0.80×
at K = 256.

Applied to both symmetric matmuls of Newton–Schulz in `optimizers.Muon`, this is
**1.14× end-to-end at 2048² and 1.35× at 8192²** — only 2 of the 3 matmuls per
step have symmetric outputs, so the per-matmul gain is diluted.

This is the CPU-free equivalent of the CUDA kernel in
[flash-muon](https://github.com/nil0x9/flash-muon), which skips lower-triangular
GEMM tiles inside a fused kernel; the idea is due to Laker Newhouse et al. A
hand-written `mx.fast.metal_kernel` SYRK could add at most ~1.15× on top, since
it would have to out-perform `steel_gemm` — already ~78% of the M4 Pro's fp32
peak. That is why there is no custom kernel here.

### Sparse matmul (CSR × dense)

One Metal thread per output element; scales with `nnz`, not `M × K`.

| (M, K, N) | density | nnz | dense matmul | **csr_matmul** | speedup |
|:---|:---:|---:|---:|---:|---:|
| (1000, 2048, 64) | 2.0% | 41k | 1.5 ms | **0.1 ms** | **27×** |
| (1000, 8192, 64) | 1.0% | 82k | 8.8 ms | **0.1 ms** | **167×** |
| (5000, 16384, 64) | 0.5% | 409k | 65 ms | **0.1 ms** | **535×** |
| (10000, 32768, 128) | 0.2% | 655k | 219 ms | **0.4 ms** | **622×** |

Density is the whole story. Writing `A @ B` dense gets Metal's optimized GEMM
but does every zero multiply; the CSR path skips them. Worth it for graph
Laplacians, GNN message passing and one-hot feature matrices — not for
dense-ish data.

---

## decomposition

### PCA vs sklearn

| Matrix shape | k | sklearn full | sklearn randomized | **mlx_addons PCA** |
|:---|---:|---:|---:|---:|
| (633, 128) | 32 | 13 ms | 371 ms | **8 ms** |
| (2000, 512) | 32 | 137 ms | 3.7 s | **18 ms** |
| (5000, 1024) | 64 | 664 ms | 7.4 s | **48 ms** |
| (10000, 2048) | 32 | 4.0 s | 6.8 s | **85 ms** |

On real fingerprint data — **ChemeleonSMD, 35633 × 2048 float32**:

| PCA dim | sklearn PCA | **mlx_addons PCA** | speedup |
|---:|---:|---:|:---:|
| 64 | 9.4 s | **464 ms** | **20×** |
| 128 | 9.7 s | **651 ms** | **15×** |
| 192 | 10.1 s | **893 ms** | **11×** |

### Nyström and KernelPCA

Kernel matrix construction (RBF / poly / linear / sigmoid) is one `X @ Y.T` plus
elementwise ops on Metal; the eigendecomposition of the small (m × m) or (n × n)
kernel matrix runs on the MLX CPU stream via `mx.linalg.eigh`.

| Method | shape | sklearn | **mlx_addons** | speedup |
|:---|:---|---:|---:|:---:|
| Nystroem | n=1000, d=20, m=100 | 233 ms | **2 ms** | **109×** |
| Nystroem | n=5000, d=50, m=300 | 332 ms | **6 ms** | **54×** |
| Nystroem | n=10000, d=100, m=500 | 388 ms | **14 ms** | **27×** |
| KernelPCA | n=500, d=10, k=20 | 310 ms | **20 ms** | **16×** |
| KernelPCA | n=1500, d=30, k=40 | 505 ms | **134 ms** | **3.8×** |
| KernelPCA | n=3000, d=50, k=60 | 1.4 s | **693 ms** | **2.0×** |

### Random projection (Johnson–Lindenstrauss)

One Metal matmul. `SparseRandomProjection` can keep its matrix in CSR
(`store_sparse=True`) for downstream GNN-shape SpMM, but the dense path is
faster at RP-typical shapes (small k, large n).

| shape | k | sklearn Gaussian | **ours** | speedup |
|:---|---:|---:|---:|---:|
| n=5000, d=2048 | 128 | 17 ms | **8 ms** | **2.2×** |
| n=10000, d=4096 | 128 | 56 ms | **12 ms** | **4.6×** |
| n=1000, d=16384 | 256 | 98 ms | **32 ms** | **3.1×** |

### EnsembleRandomProjection — accuracy, not speed

The only component here that changes *accuracy*. MoleculeACE, ChemeleonSMD v5
fingerprints → 128-d → TabICL-MLX, first 10 targets, mean RMSE:

| Feature transform | Mean RMSE | Δ vs PCA |
|:---|---:|---:|
| PCA (single) | 0.6281 | — |
| SparseRP × 5 (seeds averaged) | 0.6239 | −0.0042 |
| GaussianRP × 5 | 0.6241 | −0.0040 |
| **PCA + 2 SRP + 2 GRP (default recipe)** | **0.6215** | **−0.0066** |

Why it works: PCA captures high-variance directions; random projections preserve
distances uniformly (JL lemma); averaging cancels per-basis variance. PCA wins
alone on large, well-conditioned targets (CHEMBL204, 214); RP ensembles win on
small, ill-conditioned ones (CHEMBL1871 −0.028, CHEMBL1862 −0.034). The mix
captures both sides. Reproduce with
[`benchmarks/bench_ensemble_rp.py`](../benchmarks/bench_ensemble_rp.py).

---

## cluster

### KMeans vs sklearn

| Shape | sklearn | **mlx_addons** | speedup |
|:---|---:|---:|:---:|
| n=1000, d=16, k=8 | 15 ms | 25 ms | 0.6× |
| n=5000, d=32, k=16 | 42 ms | 36 ms | 1.2× |
| n=10000, d=64, k=32 | 351 ms | 79 ms | **4.4×** |
| n=50000, d=64, k=32 | 1.2 s | 133 ms | **8.8×** |
| n=100000, d=128, k=64 | 7.4 s | 465 ms | **16×** |

**Below n ≈ 5000 sklearn wins.** The GPU path only pays off once the matmul is
large enough to hide the launch.
