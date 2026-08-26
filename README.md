# mlx-addons

**GPU-accelerated operations for [MLX](https://github.com/ml-explore/mlx) on
Apple Silicon** — Metal compute kernels for things that are missing from core
MLX, or that only run on its CPU stream.

Most of it is a drop-in replacement for something you already use: swap the
import, keep the code.

```python
from sklearn.decomposition import PCA      # 9.4 s
from mlx_addons.decomposition import PCA   # 464 ms
```

## What's in the box

| Module | What it gives you | Headline |
|---|---|---|
| [`linalg`](#linalg) | Batched Cholesky / solve / QR / det / triangular solve on **any** matrix size; randomized SVD; symmetric eigensolvers; density-matrix purification; CSR sparse matmul; SYRK | `solve` **33×** CPU at k=30; `csr_matmul` **622×** dense at 0.2% density |
| [`decomposition`](#decomposition) | sklearn-style `PCA`, `TruncatedSVD`, `Nystroem`, `KernelPCA`, Gaussian/Sparse random projection, and a mixed PCA+RP ensemble | `PCA` **11–20×** sklearn; `Nystroem` **109×** |
| [`cluster`](#cluster) | `KMeans` (Lloyd + k-means++), whole loop in MLX ops | **16×** sklearn at n=100k |
| [`knn`](#knn) | Exact k-NN via Z-order tree + Metal kernels, up to 256 neighbours | 100k points, k=16 |
| [`nndescent`](#knn) | Approximate k-NN *graph* construction (NNDescent), pure MLX | — |
| [`ensemble`](#ensemble) | `ExtraTreesRegressorMLXCSR` — CSR/segment-scatter ExtraTrees, no index matrix, zero padding waste | **2.7–4.5×** sklearn above n=20k |
| [`neighbors`](#neighbors) | `KernelDensity` (all six sklearn kernels) + `bootstrap_kde`, tiled so peak memory is flat in sample count | **21–35×** sklearn; 269 MB at any N |
| [`solvers`](#solvers) | Pulay DIIS extrapolation + commutator residual, batched | — |
| [`optimizers`](#optimizers) | `Muon` with SYRK-accelerated Newton–Schulz | 1.14–1.35× end-to-end |
| [`recurrent`](#recurrent) | `MetalLSTM` / `GroupedMetalLSTM` — fused Metal cell kernels | — |
| [`fused_rnn`, `fused_gru`](#recurrent) | Whole-sequence LSTM/GRU in **one** JIT Metal kernel, forward + trainable fused BPTT | GRU ~4–7× eager inference, ~17× training |

Full measurements, hardware and methodology: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Install

```bash
pip install mlx-addons
```

From source:

```bash
git clone https://github.com/guillaume-osmo/mlx-addons.git
cd mlx-addons && pip install -e .
```

**Requirements:** macOS on Apple Silicon (M1–M4), Python ≥ 3.10, MLX ≥ 0.20.

---

## Read this before you benchmark it

A GPU kernel is not uniformly faster. Every path here has a size below which
launch overhead dominates and the CPU stream wins — and the useful thing this
README can tell you is *where that line is*, so you do not discover it as a
disappointing measurement.

| Path | Use it when | It **loses** when |
|---|---|---|
| `solve`, `cholesky` | any k, large batch | batch is small (per-launch overhead) |
| `qr` | k ≤ 63 | k > 63 — **falls back to CPU LAPACK** (32 KB threadgroup limit fits two k×k up to 63) |
| `jacobi_eigh` | B ≳ 500 | **B ≲ 100** — CPU stream is faster at every k |
| `sp2_purify` | N ≳ 1000 | **N ≲ 500** — `eigh` wins decisively (0.03× at N=50) |
| `csr_matmul` | genuinely sparse, wide left operand | dense-ish input — it scales with `nnz`, so density is the whole story |
| `syrk` / `gram` | M ≥ `MIN_DIM` (2048) **and** contraction ≥ `MIN_CONTRACT` (1024) | thin-K: 0.65–0.80× at K=256 |
| `randomized_svd` batched | batch ≥ 4 | **batch = 1** (0.66× — a serial loop is faster) |
| `ExtraTreesRegressorMLXCSR` | n ≳ 20,000 | **n ≈ 3,000 — 0.99×**, sklearn is level. And it is *not bit-reproducible* — see below |
| `KernelDensity` | large sample counts, or when the naive path OOMs | tiny problems, where sklearn is already milliseconds |
| `KMeans` | n ≳ 5000 | n = 1000 (0.6× sklearn) |
| `PCA`, `Nystroem`, `KernelPCA` | almost always | very small n, where sklearn's full path is already milliseconds |

Two more traps worth stating plainly:

* **Benchmark noise.** Below 4096 the sub-millisecond shapes swing ±15% at
  `repeat=10` and will invent regressions that are not there. Use `repeat ≥ 25`.
* **Cache.** Call `mx.clear_cache()` between runs, or you will measure the
  allocator rather than the kernel.
* **`ensemble` is statistically reproducible, not bit-reproducible.** Scatter-add
  with duplicate indices has no fixed accumulation order on Metal. Over five
  identical runs the median per-row difference is 6e-8, but the max is 3.6e-2 on
  6–7 rows in 1000 — rounding occasionally flips a near-tie split and those rows
  land in a different leaf. R² was stable to six decimals. If you need bit-exact
  reruns, the reductions have to be ordered deterministically.

---

## API reference

### `linalg`

Batched linear algebra via Metal. Three kernel tiers are dispatched by matrix
size: **per-thread** (k ≤ 32, one thread per matrix), **threadgroup-cooperative**
(k 33–80, shared memory, parallel column updates), and **blocked** (k > 80,
tiled with GPU matmul for the SYRK updates) — so there is no size ceiling.

| Group | Functions |
|---|---|
| Solve & factor | `solve`, `solve_cholesky`, `solve_lu`, `cholesky`, `qr`, `tril_solve`, `triu_solve` |
| Determinants | `det`, `slogdet`, `logdet_spd` |
| Low-rank | `randomized_svd`, `TruncatedSVD` |
| Sparse | `csr_matmul`, `csr_from_dense` |
| Symmetric eigen | `jacobi_eigh`, `batched_eigh`, `eigh_small_batch`, `gen_eigh`, `gershgorin_bounds` |
| 3×3 fast paths | `eigh_symmetric_3x3`, `principal_axes_3x3` |
| Geometry | `kabsch_rmsd`, `KabschRMSDResult` |
| Purification | `sp2_purify`, `mcweeny_purify` |
| Symmetric matmul | `syrk`, `gram` |

```python
from mlx_addons.linalg import solve, qr, randomized_svd, csr_matmul, syrk

A = mx.array(...)          # (10000, 64, 64) SPD
x = solve(A, mx.array(...))          # 30 ms GPU vs 311 ms CPU
Q, R = qr(A_small)                   # Householder QR
U, S, Vt = randomized_svd(X, n_components=32, n_iter=4)
G = syrk(X)                          # X @ X.T at ~half the flops
```

`randomized_svd` accepts batched input `(batch, n, m)`: all four matmuls go
through a single Metal dispatch, and the QR and final small SVD use batched
LAPACK on the CPU stream. See the crossover table above — batch 1 loses.

**Symmetric eigensolvers.** `jacobi_eigh` closes the "MLX has no GPU `eigh`" gap
for the small-N regime that semiempirical SCF lives in. It supports
`N ≤ JACOBI_MAX_N` (**96**) and auto-dispatches between three kernels — a
thread-local one (1 thread = 1 matrix, for large batches), a
threadgroup-cooperative one (1 threadgroup = 1 matrix, parallel row/col updates
per rotation, for small batches), and a vector-group variant above
`JACOBI_RESIDENT_MAX_N` (32). The cooperative crossover is size-aware rather
than a single number — `JACOBI_TG_AUTO_LIMITS` maps a size band to the largest
batch that still prefers it. Force one with `kernel="thread" | "tg" | "vg"`.

**`syrk` takes the symmetry saving MLX does not.** MLX dispatches `X @ X.T` to
the same general GEMM as `X @ Y` and pays full price (M4 Pro, bf16, 8192²:
151.2 vs 149.4 ms). Splitting `X` into `k` row-blocks and computing only the
`k(k+1)/2` upper-triangular output blocks costs `(1 + 1/k)/2` of the flops —
1.68× at 8192². No custom Metal kernel: one would have to beat `steel_gemm`
(already ~78% of the M4 Pro's fp32 peak) for at most ~1.15× more.

### `decomposition`

sklearn-compatible, backed by `randomized_svd` and Metal matmul.

| Class / function | sklearn equivalent |
|---|---|
| `PCA` | `sklearn.decomposition.PCA` — with `transform`, `inverse_transform`, `whiten`, `explained_variance_ratio_` |
| `TruncatedSVD` | `sklearn.decomposition.TruncatedSVD` |
| `Nystroem` | `sklearn.kernel_approximation.Nystroem` (RBF / poly / linear / sigmoid) |
| `KernelPCA` | `sklearn.decomposition.KernelPCA` |
| `GaussianRandomProjection`, `SparseRandomProjection` | `sklearn.random_projection.*` (Achlioptas / Li) |
| `johnson_lindenstrauss_min_dim` | `sklearn.random_projection.johnson_lindenstrauss_min_dim` |
| `pairwise_kernel` | raw kernel matrix, if you need it |
| `EnsembleRandomProjection`, `ensemble_mean_predict` | *no equivalent* — see below |

**`EnsembleRandomProjection` is the one thing here with no sklearn counterpart,
and the only one that changes accuracy rather than speed.** A single PCA is
sensitive to noisy top eigenvectors on ill-conditioned data; a single random
projection has high variance; averaging a few of each beats either alone.

```python
from mlx_addons.decomposition import EnsembleRandomProjection, ensemble_mean_predict

ens = EnsembleRandomProjection(
    n_components=128, n_pca=1, n_sparse=2, n_gaussian=2, random_state=42,
).fit(X_all)                       # global, unsupervised — fits 5 feature maps

def fit_predict(Ztr, ytr, Zte):
    return YourRegressor(...).fit(Ztr, ytr).predict(Zte)

y_pred = ensemble_mean_predict(ens, fit_predict, X_train, y_train, X_test)
```

On MoleculeACE (ChemeleonSMD v5 → 128-d → TabICL-MLX, first 10 targets), the
default 1 PCA + 2 sparse + 2 Gaussian recipe gives mean RMSE **0.6215** against
**0.6281** for a single PCA. PCA wins alone on large, well-conditioned targets;
RP ensembles win on small, ill-conditioned ones (CHEMBL1871 −0.028,
CHEMBL1862 −0.034). The mix captures both sides.

### `cluster`

```python
from mlx_addons.cluster import KMeans
km = KMeans(n_clusters=32, n_init=3, random_state=0).fit(X)
km.labels_, km.cluster_centers_, km.predict(X_new)
```

Assignment is one Metal matmul + `argmin`; update is a one-hot `.T @ X` matmul.
No custom kernels — the whole Lloyd loop is MLX ops.

### `ensemble`

```python
from mlx_addons.ensemble import ExtraTreesRegressorMLXCSR

et = ExtraTreesRegressorMLXCSR(n_estimators=100, max_depth=10,
                               max_features=1.0, random_state=0).fit(X, y)
y_pred = et.predict(X_test)
```

**No index matrix, and no padding.** The obvious way to build trees on a GPU is a
dense `(m, L)` index matrix per level, padded to the largest node — and it cannot
win. Measured padding waste at depth 10 is **12–13× serial and 31–41× once the
forest is batched**, because `L` becomes the forest-wide max child size. Batching
makes it *worse*.

This module keeps no index matrix at all. It holds one node-assignment vector and
routes every row elementwise, per level:

```
node <- 2*node + 1 + (x[row, feat[node]] >= thr[node])
```

Every per-node statistic is then a segment reduction over that vector, and MLX has
all three natively as single scatters (`.at[gid].add` / `.minimum` / `.maximum`).
Memory per level is `O(T·N)` — independent of depth *and* of node-size skew.
Padding waste: **0×**, and 30× faster than the padded level-wise version.

Two things that were not obvious. Removing the per-level host syncs (~1000 → 10)
made it **2–3× slower** — the syncs were never the bottleneck, the dense gather
was. And node layout is the heap (children of `i` at `2i+1`, `2i+2`): packing a
level as `concatenate([all_left, all_right])` agrees with that only at depth ≤ 1
and silently corrupts everything below, holding training R² at 0.157 while looking
structurally plausible. `tests/test_csr_trees.py` guards it via monotonicity of
training R² in depth.

### `knn`

```python
from mlx_addons.knn import knn, KNNConfig, TreeConfig
distances, indices = knn(mx.random.normal((100000, 3)), k=16)   # k up to 256
```

Pipeline: Morton encoding → Z-order sort → SoA tree build → GPU frontier walk →
Metal segmented top-k. For *approximate* k-NN **graphs** rather than exact
queries, `mlx_addons.nndescent` implements NNDescent in pure MLX.

### `neighbors`

Drop-in for `sklearn.neighbors.KernelDensity` — all six sklearn kernels
(`gaussian`, `tophat`, `epanechnikov`, `exponential`, `linear`, `cosine`),
matching sklearn's density to ~1e-5 in float32.

```python
from mlx_addons.neighbors import KernelDensity, bootstrap_kde, VALID_KERNELS

kde = KernelDensity(bandwidth=0.2).fit(X)     # X: (n_samples, n_features)
logp = kde.score_samples(grid)                # matches sklearn
p = kde.eval_density(grid)                    # == exp(score_samples), cheaper

band = bootstrap_kde(returns, grid, n_samples=50_000, n_boot=1000, bandwidth=2e-4)
```

**Peak memory is flat in the sample count.** A grid×samples KDE is normally one
huge kernel matrix reduced along an axis; this walks both axes in tiles and
accumulates, so the peak is bounded by `query_tile × sample_tile` rather than
`grid × samples` — **269 MB whether N is 10K or 250K**, where the materialised
path needs 5000 MB at 250K and OOMs not far past it.

### `solvers`

```python
from mlx_addons.solvers import pulay_diis, commutator_error
e = commutator_error(F, P)                                  # F @ P - P @ F
F_extrap = pulay_diis(F_history, e_history, max_history=6)
```

The augmented `(nd+1) × (nd+1)` Pulay system is solved via `linalg.solve_lu`.
Batched over leading dims, so one call extrapolates every molecule.

### `optimizers`

```python
from mlx_addons.optimizers import Muon, zeropower_via_newtonschulz5
```

`Muon` with SYRK-accelerated Newton–Schulz orthogonalization. The end-to-end
gain is **1.14× at 2048² and 1.35× at 8192²**, not the full 1.68× `syrk` gives
in isolation — only 2 of the 3 matmuls per step have symmetric outputs.

### `recurrent`

```python
from mlx_addons.recurrent import MetalLSTM, GroupedMetalLSTM
from mlx_addons import fused_lstm, fused_lstm_sequence, fused_gru, fused_gru_sequence
```

`recurrent` fuses the LSTM **cell**. `fused_rnn` / `fused_gru` fuse the **whole
sequence** into a single JIT Metal kernel with the input projection folded in,
including a trainable fused-BPTT backward — `fused_lstm` matches or beats
MPSGraph, and the GRU path is roughly 4–7× eager inference and ~17× training.

---

## Benchmarks

Every measured table — Cholesky, QR, rSVD, PCA, Nyström, KernelPCA, random
projection, CSR matmul, KMeans, Jacobi eigh, SP2 purification, SYRK, CSR
ExtraTrees, KDE — lives in
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**, with the hardware and methodology
for each.

## License

MIT
