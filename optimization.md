# kbit GEMM Kernel: Optimization Guide

This document catalogs the remaining performance optimizations for the
production kbit GEMM kernel (`kbit_gemm_prod`). Each optimization is
described with its expected impact, implementation approach, and testing
strategy.

The kernel is functionally complete (fp16 + bf16, split-K, ldmatrix with
swizzle, cp.async double-buffered pipeline, 139 tests passing). The
remaining work is purely about throughput.

---

## Current State (Baseline)

**Kernel configuration:**
- TILE_M = 16 (one MMA M-block per warp)
- TILE_N = 128 (N_BLOCKS = 2, each warp covers 16 columns)
- TILE_K = 64 (4 MMA k-sub-tiles of 16)
- 256 threads = 8 warps, each warp handles the same M rows and a slice of N
- Double-buffered cp.async pipeline
- ldmatrix.x4 with XOR bank-conflict swizzle for A tile

**RTX 4090 benchmark (K=4, fp16, k_chunks=1):**

| M | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|---:|------:|------:|----------:|------------:|--------:|
| 1 | 4096 | 4096 | 109 | 43 | 0.39x |
| 1 | 4096 | 11008 | 82 | 128 | **1.56x** |
| 4 | 4096 | 4096 | 92 | 22 | 0.24x |
| 4 | 4096 | 11008 | 100 | 121 | **1.21x** |
| 16 | 4096 | 4096 | 149 | 28 | 0.19x |

**Why it's slow for square matrices:** Each thread block computes a
16x128 output tile. With M=16, only 1 M-tile exists, meaning only
(N/128) blocks launch. For N=4096, that's 32 blocks on a 128-SM GPU --
25% utilization. And each block does very little compute per shared
memory load because TILE_M=16 means only one MMA row-block per warp.

**Why it wins for M=1 large-N:** The GEMM is memory-bandwidth-bound.
The kernel reads 4-bit compressed weights (4x less data than fp16
cuBLAS), which directly translates to speedup.

---

## Optimization 1: Multi-M-Block Tiling

**Priority: HIGHEST. This is the single biggest performance lever.**

### The Problem

Currently TILE_M=16. Each warp executes 2 MMA operations per k-sub-tile
(N_BLOCKS=2). The A fragment is loaded once and used for only 2 MMAs.
The compute-to-load ratio is low.

### The Fix

Template the kernel on `M_BLOCKS` (1, 2, 3, 4). TILE_M becomes
`M_BLOCKS * 16`. Each warp handles multiple M-blocks, reusing the same
B fragment across all of them:

```
Current (M_BLOCKS=1):
  Each warp: 1 M-block x 2 N-blocks = 2 MMAs per k-sub-tile

Target (M_BLOCKS=4):
  Each warp: 4 M-blocks x 2 N-blocks = 8 MMAs per k-sub-tile
```

The B fragment (dequantized from bit-planes) is the expensive part --
codebook lookup via shuffle, absmax multiply. With M_BLOCKS=4, this cost
is amortized over 4x more MMA operations.

### Implementation

1. Add `M_BLOCKS` template parameter to `kbit_gemm_prod`
2. FragC accumulator becomes `float frag_c[M_BLOCKS][N_BLOCKS][4]`
3. A fragment loading: load `M_BLOCKS` fragments per k-sub-tile (ldmatrix
   for each M-block's 16 rows)
4. Inner loop: for each B fragment, iterate over M_BLOCKS and issue MMA
5. A tile in shared memory grows: `M_BLOCKS * 16 * TILE_K * sizeof(scalar_t)`
6. Output write: iterate over M_BLOCKS for the C tile write
7. Host-side dispatch selects M_BLOCKS based on M:
   - M <= 16: M_BLOCKS=1
   - M <= 32: M_BLOCKS=2
   - M <= 48: M_BLOCKS=3
   - M >= 49: M_BLOCKS=4

### Shared Memory Impact

| M_BLOCKS | TILE_M | A tile (bytes) | B tile K=4 | Absmax | Per stage | 2 stages |
|---------:|-------:|---------------:|-----------:|-------:|----------:|---------:|
| 1 | 16 | 2,048 | 4,096 | 256 | 6,400 | 12,800 |
| 2 | 32 | 4,096 | 4,096 | 256 | 8,448 | 16,896 |
| 4 | 64 | 8,192 | 4,096 | 256 | 12,544 | 25,088 |

All fit within RTX 4090's 100 KB limit. Even M_BLOCKS=4 with 4 pipeline
stages would use ~50 KB.

### Register Impact

FragC grows from 2*4 = 8 floats to M_BLOCKS*2*4 = 32 floats for M_BLOCKS=4.
FragA grows from 4 uint32 to M_BLOCKS*4 = 16 uint32. Total registers ~50-60,
well within the 255 limit.

### Expected Speedup

For M=4, K_dim=4096, N=4096 with M_BLOCKS=4: each block does 4x more compute
per B tile load. Since the kernel is currently B-load-limited for these sizes,
expect roughly **2-3x improvement** (not full 4x due to diminishing returns
from A tile growth).

### Test Strategy

- M_BLOCKS=1 must produce identical output to the current kernel (bit-exact)
- M_BLOCKS=2,3,4 must match Python reference within existing tolerance
- Test partial M-tiles: M=5 with M_BLOCKS=4 (TILE_M=64, only 5 rows valid)

---

## Optimization 2: Larger N_BLOCKS per Warp

**Priority: HIGH. Complements multi-M-block.**

### The Problem

Currently N_BLOCKS=2, so each warp covers 16 of the 128 tile columns.
With 8 warps, that's 8*16 = 128 columns (full tile). But each warp
only issues 2 MMA ops per k-sub-tile per M-block.

### The Fix

Increase N_BLOCKS to 4 (each warp covers 32 columns). Then 4 warps
cover the full TILE_N=128. The remaining 4 warps cover additional M
rows (for the 2-warps-along-M x 4-warps-along-N layout from the
design doc).

### Warp Layout

The design doc specifies for TILE_M=64, TILE_N=128:

```
2 warps along M (each handles 32 rows = 2 M-blocks)
x 4 warps along N (each handles 32 cols = 4 N-blocks)
= 8 warps total

Each warp: 2 M-blocks x 4 N-blocks = 8 MMAs per k-sub-tile
With TILE_K=64 (4 k-sub-tiles): 32 MMAs per warp per K-tile
```

This is the target configuration. Combined with multi-M-block, it gives
each warp 4x more compute than the current kernel.

### Implementation

1. Change N_BLOCKS to 4
2. Change warp-to-tile mapping: `warp_m = warp_id / 4`, `warp_n = warp_id % 4`
3. Each warp handles M-blocks `[warp_m * M_BLOCKS_PER_WARP ... (warp_m+1) * M_BLOCKS_PER_WARP - 1]`
   and N-blocks `[warp_n * 4 ... warp_n * 4 + 3]`
4. Fragment accumulators: `frag_c[M_BLOCKS_PER_WARP][4][4]`

### Expected Speedup

Combined with multi-M-block: each thread block does **8x** more compute
per B tile load compared to current (4x from M, 2x from N). For M>=4
square matrices, expect the kernel to **match or beat cuBLAS**.

---

## Optimization 3: C Output Staging Through Shared Memory

**Priority: MEDIUM. Improves memory write efficiency.**

### The Problem

Currently, each thread writes its FragC values directly to global memory.
The MMA fragment layout means threads in a warp write to scattered row
positions:
- Thread with gid=0 writes rows 0, 8
- Thread with gid=1 writes rows 1, 9
- etc.

These writes hit different cache lines (each row is N*2 bytes apart),
causing uncoalesced writes.

### The Fix

After the K-tile loop, stage the output through shared memory:

1. Each warp writes its FragC values to shared memory in the natural
   fragment order (scattered rows, but shmem is fast)
2. `__syncthreads()`
3. All threads cooperatively read from shared memory in row-major order
   and write to global memory with coalesced access (consecutive threads
   write consecutive addresses within the same row)

### Shared Memory Reuse

The pipeline's shared memory is no longer needed during the output phase
(the K-tile loop is done). The C staging area can reuse the pipeline
buffers. For TILE_M=64, TILE_N=128, the C tile is 64*128*2 = 16 KB in
fp16, which fits easily in one pipeline stage's allocation.

### Expected Speedup

Moderate. The output write is not on the critical path for large K_dim
(the K-tile loop dominates). For small K_dim or when the kernel is
already close to bandwidth-optimal, this can give **5-15% improvement**.

---

## Optimization 4: Persistent Kernel

**Priority: MEDIUM. Helps SM utilization for small tile counts.**

### The Problem

The current 2D/3D grid launch creates one block per output tile (or per
split-K chunk). When the number of tiles is less than the GPU's SM count,
SMs sit idle.

### The Fix

Launch exactly `num_SMs` blocks. Each block loops over assigned work items
(linearized (m_tile, n_tile, k_chunk) triples). Benefits:

1. **Better utilization:** All SMs are always active
2. **Accumulator persistence:** When consecutive work items share the same
   output tile, the accumulators stay in registers (no atomicAdd needed)
3. **First-contributor optimization:** The first block to write a tile does
   a plain store to the fp32 workspace (no need to zero it first). Only
   subsequent contributors use atomicAdd.

### Implementation

See design doc Section 6 for the full design. The key structure:

```cpp
int total_work = m_tiles * n_tiles * k_chunks;
int work_per_block = div_ceil(total_work, gridDim.x);
int my_start = blockIdx.x * work_per_block;
int my_end = min(my_start + work_per_block, total_work);

int prev_mn = -1;
for (int work_id = my_start; work_id < my_end; work_id++) {
    int mn_id = work_id / k_chunks;
    int k_chunk_id = work_id % k_chunks;
    if (mn_id != prev_mn) {
        if (prev_mn >= 0) write_output(...);
        zero_accumulators();
        prev_mn = mn_id;
    }
    process_k_range(k_chunk_id, ...);
}
if (prev_mn >= 0) write_output(...);
```

### Expected Speedup

Depends on the shape. For shapes where `m_tiles * n_tiles < num_SMs`
(e.g., M=16, N=4096 on a 128-SM GPU: 1*32=32 tiles), the persistent
kernel can **2-3x** improve throughput by enabling split-K without the
atomicAdd overhead. For shapes with many tiles, the benefit is marginal.

---

## Optimization 5: cp.async for A Tile

**Priority: LOW. Minor improvement.**

### The Problem

Currently A is loaded synchronously (element-by-element) while B and
absmax use cp.async. A could also use cp.async for better latency hiding.

### The Complication

A needs bounds checking (`gr < M && gc < K_dim`) and XOR swizzle on the
destination address. cp.async copies from a source address to a destination
address, so the swizzle can be applied to the destination. But bounds
checking is harder -- cp.async doesn't support conditional copies.

### Possible Approach

Use `cp.async.cg.shared.global` for the interior of the A tile (rows that
are guaranteed in-bounds), and synchronous loads only for boundary rows.
For TILE_M=64 and M=4096, almost all rows are in-bounds. Only the last
M-tile may have boundary rows.

### Expected Speedup

Small (2-5%). A tile is only 2-8 KB per stage, much smaller than B tile.
The synchronous load latency is already partially hidden by the pipeline.

---

## Recommended Implementation Order

1. **Multi-M-block tiling** (Optimization 1) -- biggest impact, enables the
   target warp layout
2. **Larger N_BLOCKS** (Optimization 2) -- natural companion to multi-M-block,
   together they achieve the design doc's target of 32 MMAs per warp per K-tile
3. **C output staging** (Optimization 3) -- polish for write efficiency
4. **Persistent kernel** (Optimization 4) -- improves edge cases
5. **cp.async for A** (Optimization 5) -- diminishing returns

After optimizations 1+2, re-benchmark. If the kernel matches cuBLAS for
M=1-32 with large N, the remaining optimizations can be deprioritized in
favor of integration work (wiring into Linear4bit, auto-tuning k_chunks).

---

## Integration Work (Not Performance, But Required)

These are not performance optimizations but are needed to ship:

- **Wire into LinearNbit module:** Replace the dequant+cuBLAS path with a
  call to `kbit_gemm_prod` when conditions are met (CUDA, fp16/bf16,
  N % 128 == 0, K_dim % 64 == 0)
- **Auto-select k_chunks:** Based on M, N, K_dim, and SM count. Formula
  from design doc Section 6.2.
- **Remove staging kernels:** Clean up Stages 3-5 kernels, keeping only
  the production kernel and the debug MMA test
- **Lint + PR:** Run ruff/clang-format, merge to main
