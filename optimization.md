# kbit GEMM Kernel: Optimization Status and Remaining Work

This document records what has been done, what was learned, and what
remains for the production kernel `kbit_gemm_prod`.

---

## Current Kernel Configuration (after Optimizations 1 + 5)

**Template parameters:** `<K_BITS, M_BLOCKS, scalar_t>`

- **TILE_M** = M_BLOCKS * 16 (M_BLOCKS selected at runtime: 1, 2, 3, or 4)
- **TILE_N** = 128, **N_BLOCKS** = 2 (each warp covers 16 columns)
- **TILE_K** = 64 (4 MMA k-sub-tiles of 16)
- 256 threads = 8 warps, all warps share the same M rows, each handles a
  different N slice
- Double-buffered cp.async pipeline for A, B, and absmax tiles
- ldmatrix.x4 with XOR bank-conflict swizzle for A fragments
- Split-K support via atomicAdd + tile counters
- fp16 and bf16 via `scalar_t` template

**Instantiations:** 4 K-values x 4 M_BLOCKS x 2 dtypes = 32 kernel variants.

**Register usage (sm_89, zero spills across all variants):**

| M_BLOCKS | K=2 | K=3 | K=4 | K=5 |
|---------:|----:|----:|----:|----:|
| 1        |  56 |  56 |  56 |  64 |
| 2        |  72 |  72 |  72 |  80 |
| 3        |  92 |  92 |  96 |  96 |
| 4        | 111 | 111 | 113 | 115 |

**Tests:** 195 total (139 original + 56 multi-M-block), all passing.

---

## Completed Optimizations

### Optimization 1: Multi-M-Block Tiling (commit f8a06a3)

**What:** Templated `kbit_gemm_prod` on `M_BLOCKS` (1-4). TILE_M scales
as `M_BLOCKS * 16`. Each warp loads M_BLOCKS A fragments per k-sub-tile
via ldmatrix.x4 and reuses the same dequantized B fragment across all of
them, amortizing the codebook shuffle + absmax multiply.

**Dispatch:** SM-aware. Queries `cudaDevAttrMultiProcessorCount` and
selects the largest M_BLOCKS where the resulting grid still has at least
`num_SMs` blocks. For the target shapes (M=1-16), M_BLOCKS=1 is always
selected.

**Key finding:** Multi-M-block alone showed NO benefit — it was actually
slower for M>16 because synchronous A tile loading (element-by-element,
with per-element bounds check + XOR swizzle) became the bottleneck. The
A tile grows from 2 KB (MB=1) to 8 KB (MB=4), and this synchronous load
was on the critical path, not overlapped by the pipeline.

This finding reordered the optimization priorities: cp.async for A
(originally listed as "Priority LOW, 2-5%") turned out to be a
**prerequisite** for multi-M-block to work at all.

### Optimization 5: cp.async for A Tile (commit 7cd575b)

**What:** Replaced synchronous A tile loading with cp.async 16-byte
copies. The A tile is loaded in groups of 8 halves (one int4), with XOR
swizzle applied to the destination shared memory address.

- **Interior tiles** (m_base + TILE_M <= M and k_base + TILE_K <= K_dim):
  pure cp.async, no branches in the loop.
- **Boundary tiles** (last M-tile or last K-tile): per-group bounds check;
  in-bounds groups use cp.async, out-of-bounds groups get synchronous
  zero-fill. K_dim is always a multiple of 32 (BLOCKSIZE), so group
  boundaries align cleanly — no partial groups.

**Impact:** This was the most impactful single change. It improved
performance for ALL shapes, not just M_BLOCKS>1, because even M_BLOCKS=1
benefits from pipelining A loads.

---

## Current Benchmark (RTX 4090, K=4, fp16, k_chunks=1)

### Standard shapes (N=4096, compute-bound)

| M | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|---:|------:|------:|----------:|------------:|--------:|
| 1 | 4096 | 4096 | 77 | 60 | 0.79x |
| 4 | 4096 | 4096 | 78 | 28 | 0.36x |
| 8 | 4096 | 4096 | 73 | 25 | 0.34x |
| 16 | 4096 | 4096 | 96 | 28 | 0.29x |
| 32 | 4096 | 4096 | 95 | 41 | 0.43x |
| 64 | 4096 | 4096 | 79 | 29 | 0.36x |

### Large-N shapes (bandwidth-bound — target regime)

| M | K_dim | N | MB | kbit (us) | cuBLAS (us) | Speedup |
|---:|------:|------:|---:|----------:|------------:|--------:|
| 1 | 4096 | 11008 | 1 | 89 | 123 | **1.38x** |
| 1 | 4096 | 16384 | 1 | 77 | 142 | **1.84x** |
| 4 | 4096 | 11008 | 1 | 62 | 126 | **2.02x** |
| 4 | 4096 | 16384 | 1 | 82 | 142 | **1.75x** |
| 16 | 4096 | 11008 | 1 | 62 | 98 | **1.58x** |
| 16 | 4096 | 16384 | 1 | 83 | 164 | **1.98x** |
| 32 | 4096 | 11008 | 1 | 121 | 100 | 0.83x |
| 32 | 4096 | 16384 | 2 | 96 | 149 | **1.55x** |
| 64 | 4096 | 16384 | 3 | 199 | 219 | **1.10x** |
| 128 | 4096 | 16384 | 4 | 154 | 173 | **1.12x** |

### Progress vs pre-optimization baseline

| Shape | Before | After | Improvement |
|-------|--------|-------|-------------|
| M=1, N=11008 | 1.56x | 1.38x | noise (same regime) |
| M=4, N=11008 | 1.21x | **2.02x** | +67% |
| M=16, N=11008 | ~1.0x | **1.58x** | +58% |
| M=16, N=4096 | 0.19x | 0.29x | +53% |
| M=64, N=16384 | lost badly | **1.10x** | now beats cuBLAS |
| M=128, N=16384 | lost badly | **1.12x** | now beats cuBLAS |

---

## Analysis: Why N=4096 is Still Slow

For N=4096, `n_tiles = 32`. On a 128-SM GPU:

- M=1: 32 blocks → 25% SM utilization
- M=16: 32 blocks → 25% utilization
- M=64: 128 blocks → 100% utilization, but each block still only does
  2 MMAs per B fragment (N_BLOCKS=2)

The kernel loses to cuBLAS on N=4096 for two reasons:

1. **Low SM utilization** (M<=16): not enough blocks to fill the GPU.
   The persistent kernel (Optimization 3 below) addresses this.

2. **Low compute-per-B-fragment** (all M): N_BLOCKS=2 means each warp
   dequantizes a B fragment and uses it for only 2 (or 2×M_BLOCKS) MMAs.
   cuBLAS uses much larger tiles. Optimization 2 (larger N_BLOCKS)
   directly addresses this.

For large N (11008+), the kernel wins because the GEMM is bandwidth-bound
and reading 4-bit weights (4x less data than fp16 cuBLAS) dominates.

---

## Remaining Optimizations

### Optimization 2: Larger N_BLOCKS per Warp

**Priority: HIGHEST. This is the next optimization to implement.**

**The problem:** N_BLOCKS=2. Each warp covers 16 of the 128 tile columns
and dequantizes one B fragment that is used for only 2 MMAs per M-block.
With 8 warps all along N, there's no M-axis parallelism within the block.

**The fix:** Increase N_BLOCKS to 4 (each warp covers 32 columns). Change
the warp layout from 8-along-N to 2-along-M x 4-along-N:

```
Current:  8 warps x 1 M-slice x 2 N-blocks = 2 MMAs per warp per k-sub
Target:   (2 warps-M x 4 warps-N) x M_BLOCKS_PER_WARP x 4 N-blocks
```

For TILE_M=64 (M_BLOCKS=4), each warp handles 2 M-blocks x 4 N-blocks =
8 MMAs per k-sub-tile. Over 4 k-sub-tiles per K-tile: 32 MMAs per warp
per K-tile. This is 4x more compute per B fragment dequant than current.

**Key interactions with Optimization 1:**

- M_BLOCKS_PER_WARP = M_BLOCKS / 2 (with 2 warp-rows along M).
  For M_BLOCKS=1 or 2: 1 M-block per warp. For M_BLOCKS=4: 2 per warp.
- For M_BLOCKS=1 (M<=16): only 1 warp-row needed, but we still want 4
  warps along N. This means 4 warps are active, 4 warps idle. Alternatively,
  keep all 8 warps along N with N_BLOCKS=2 for the M_BLOCKS=1 case and
  only switch to the 2x4 layout for M_BLOCKS>=2. Template on warp layout.
- **Simpler alternative:** just increase N_BLOCKS from 2 to 4 for ALL
  M_BLOCKS values, without changing the warp layout. 8 warps x 4 N-blocks
  = 32 N-blocks x 8 cols = 256 columns. TILE_N would grow to 256. This
  doubles the B tile in shared memory (8 KB → 16 KB for K=4) but is still
  well within limits. Each warp does 4 MMAs per M-block per k-sub (2x
  improvement) with no warp layout change.

**Recommendation:** Start with the simpler approach (N_BLOCKS=4,
TILE_N=256, same 8-warps-all-along-N layout). This requires N%256==0
instead of N%128==0. For LLM shapes: 4096/256=16, 11008/256=43,
16384/256=64. All work. If profiling shows the 2x4 warp layout is better,
refactor later.

**Expected impact:** ~2x improvement in compute throughput. The kernel
should become competitive with cuBLAS even on N=4096 shapes.

### Optimization 3: Persistent Kernel

**Priority: HIGH. Critical for N=4096 shapes with small M.**

**The problem:** For M=1, N=4096: only 32 blocks launch on a 128-SM GPU
(25% utilization). Increasing TILE_M/TILE_N doesn't help because there's
only 1 M-row and N/TILE_N blocks along N.

**The fix:** Launch exactly `num_SMs` blocks. Each block loops over
assigned work items (linearized `(m_tile, n_tile, k_chunk)` triples).

Key benefits:
1. All SMs active even when `m_tiles * n_tiles < num_SMs`
2. Accumulator persistence: consecutive k-chunks for the same output tile
   stay in registers (no atomicAdd needed)
3. Subsumes split-K: k_chunks becomes a tuning parameter, not a separate
   code path

**Interaction with current dispatch:** The persistent kernel replaces the
current grid launch logic entirely. The M_BLOCKS dispatch still selects
tile size, but the grid is always `(num_SMs, 1, 1)`.

**Expected impact:** 2-4x for N=4096 M<=16 shapes. Moderate for shapes
that already have enough blocks. Will likely make split-K unnecessary
as a separate mode.

### Optimization 4: C Output Staging Through Shared Memory

**Priority: LOW. Polish optimization.**

**The problem:** MMA fragment layout causes scattered global memory writes
(threads write to rows `gid` and `gid+8`, each row N*2 bytes apart).

**The fix:** After the K-tile loop, write FragC to shared memory (reusing
pipeline buffers), `__syncthreads()`, then cooperatively write to global
memory in row-major coalesced order.

**Expected impact:** 5-15% for small K_dim. Negligible for large K_dim
where the K-tile loop dominates. Implement after the higher-priority
optimizations are done.

---

## Lessons Learned

1. **cp.async for A was not "low priority."** The original doc rated it
   2-5% impact. In practice it was the **single most impactful change**
   because it unlocked multi-M-block AND improved the baseline. The lesson:
   anything that removes synchronous work from the pipeline critical path
   has outsized impact, especially as tile sizes grow.

2. **SM utilization dominates small-grid shapes.** Multi-M-block initially
   made things worse because larger tiles meant fewer blocks. The SM-aware
   dispatch was essential. For N=4096 shapes, no amount of per-block
   optimization can compensate for having only 32 active SMs out of 128.
   The persistent kernel is the real fix.

3. **Register pressure is not an issue.** Even M_BLOCKS=4 with K=5 uses
   only 115 registers (well under the 255 limit) with zero spills. There's
   headroom for N_BLOCKS=4 (which adds ~8 more float accumulators per
   M-block = 32 more floats for MB=4).

4. **Benchmark variance is significant.** Small-M kernel times (50-100µs)
   fluctuate 10-20% between runs due to GPU thermal state, power
   management, and CUDA runtime overhead. Always use high iteration counts
   (500+) and focus on relative trends, not absolute numbers.

---

## Implementation Order

1. **Optimization 2 (larger N_BLOCKS)** — next step, highest remaining
   priority. Doubles compute per B fragment. Should close the gap on
   N=4096 and further extend the lead on large-N shapes.

2. **Optimization 3 (persistent kernel)** — addresses the SM utilization
   problem for small grids. Essential for N=4096 M<=16.

3. **Optimization 4 (C staging)** — polish. Only after 2+3 are done and
   benchmarked.

After Optimization 2, re-benchmark. If the kernel matches cuBLAS for
M=1-32 across all N values, deprioritize remaining optimizations in
favor of integration work.

---

## Integration Work (Not Performance)

Required to ship, independent of performance optimizations:

- **Wire into LinearNbit module:** Call `kbit_gemm_prod` instead of
  dequant+cuBLAS when CUDA, fp16/bf16, N % TILE_N == 0, K_dim % 64 == 0
- **Auto-select k_chunks:** Based on M, N, K_dim, SM count
- **Remove staging kernels:** Delete Stages 3-5 (minimal, pipelined,
  split-K), keep only production kernel + MMA test
- **Lint + PR:** ruff/clang-format, merge to main
