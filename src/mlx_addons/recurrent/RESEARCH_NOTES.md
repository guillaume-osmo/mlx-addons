# LSTM Metal kernel optimization — research notes

## Status
- **Shipped (in `_metal_lstm.py`):**
  - per-cell kernel (`metal_lstm_scan`) — wins at B ≥ 128
  - full-scan kernel `metal_lstm_full_scan` (1 thread per gate column) — wins at B ≤ 64
  - `metal_lstm_scan_auto` — picks the right variant
  - Forward + VJP for single and grouped variants → training works

## Not shipped — explored and ruled out

### 1. simdgroup_matrix full-scan (single-simdgroup tg)
- 1 simdgroup (32 threads) per threadgroup, processes 8 batch elements
- Output tiled (8, 4H=256) into 32 col_tiles × 8 K-steps = 256 sequential simdgroup MMAs per timestep
- **Result**: correct, but 3-5× slower than the naive full-scan across all batches
- **Reason**: single simdgroup tg under-utilizes the GPU; col_tile loop is sequential

### 2. Batch-shared full-scan (4 batches per tg, 256 threads)
- Each tg processes BATCH_PER_TG=4 batch elements, sharing Wh reads via L1
- Per thread: 4 gate cols × 64 K-muls = 256 muls per timestep
- **Race-condition bug**: thread X's write to `h_shared` was visible to thread Y's matmul read within the same timestep. Fix: explicit barrier between read-phase and write-phase, storing gates in registers.
- **Result (post-fix)**: correct, but doesn't beat the auto-select wrapper anywhere
- **Reason**: Apple Silicon L2 cache already absorbs the Wh reads at B=64 in the naive variant; the added coordination cost matches the bandwidth savings

## What would actually win (deferred)

To beat the existing variants in the large-batch regime, the full GEMM treatment is needed:

1. **Multi-batch-tile per threadgroup** (32-64 batches, 128-256 threads = 4-8 simdgroups)
2. **Tiled, threadgroup-cached `Wh`** (chunks that fit in TG memory; double-buffer next chunk)
3. **Register-tiled accumulators** spanning the K-axis
4. **Software-pipelined simdgroup_load** to hide latency
5. **Proper col_tile parallelisation** across simdgroups within the tg

Reference implementations of this pattern are in MLX's own
`mlx/backend/metal/kernels/steel/gemm/` (~1500 lines of MSL + C++).

## Bench takeaways
For our shape (T=42, H=64, vocab=27, GENEVA²S model), Apple's MPS LSTM kernel
in `mlx.nn.LSTM` is the slowest variant in our zoo despite being native; our
per-cell kernel with the C++ fused gate update beats it by 1.4-2.4× at every
batch size. The full-scan variant adds another 2× at small batches by
eliminating the 84 sequential kernel launches.

For LSTM with much larger hidden sizes (≥256) the calculus would shift —
the matmul work would dominate launch overhead, and a proper GEMM-tuned
full-scan kernel would matter more. For maxlen=42 / hidden=64 we're already
near the limit of what a naive Metal kernel can deliver.
