# kbit GEMM Kernel: Optimization Status and Remaining Work

This document records what has been done, what was learned, and what
remains for the production kernel `kbit_gemm_prod`.

---

## Current Kernel Configuration (after Optimizations 1 + 3 + 5)

**Template parameters:** `<K_BITS, M_BLOCKS, scalar_t>`

- **TILE_M** = M_BLOCKS * 16 (M_BLOCKS selected at runtime: 1, 2, 3, or 4)
- **TILE_N** = 128, **N_BLOCKS** = 2 (each warp covers 16 columns)
- **TILE_K** = 64 (4 MMA k-sub-tiles of 16)
- 256 threads = 8 warps, all warps share the same M rows, each handles a
  different N slice
- **Persistent kernel**: launches `min(num_SMs, total_work)` blocks that
  loop over work items instead of one block per tile
- **Auto k_splits**: when mn_tiles < num_SMs, automatically splits K
  dimension to create enough work items to fill SMs
- Double-buffered cp.async pipeline for A, B, and absmax tiles
- ldmatrix.x4 with XOR bank-conflict swizzle for A fragments
- fp16 and bf16 via `scalar_t` template

**Instantiations:** 4 K-values x 4 M_BLOCKS x 2 dtypes = 32 kernel variants.

**Register usage (sm_89, zero spills across all variants):**

| M_BLOCKS | K=2 | K=3 | K=4 | K=5 |
|---------:|----:|----:|----:|----:|
| 1        |  56 |  56 |  56 |  64 |
| 2        |  72 |  72 |  72 |  80 |
| 3        |  92 |  92 |  96 |  96 |
| 4        | 111 | 111 | 113 | 115 |

**Tests:** 85 production tests, all passing.

---

## Completed Optimizations

### Optimization 1: Multi-M-Block Tiling (commit f8a06a3)

**What:** Templated `kbit_gemm_prod` on `M_BLOCKS` (1-4). TILE_M scales
as `M_BLOCKS * 16`. Each warp loads M_BLOCKS A fragments per k-sub-tile
via ldmatrix.x4 and reuses the same dequantized B fragment across all of
them, amortizing the codebook shuffle + absmax multiply.

**Key finding:** Multi-M-block alone showed NO benefit — it was actually
slower for M>16 because synchronous A tile loading became the bottleneck.
cp.async for A (Optimization 5) was required first.

### Optimization 5: cp.async for A Tile (commit 7cd575b)

**What:** Replaced synchronous A tile loading with cp.async 16-byte
copies. XOR swizzle applied to destination shmem address.

- **Interior tiles**: pure cp.async, no branches.
- **Boundary tiles**: per-group bounds check.

**Impact:** Single most impactful change. Improved ALL shapes because
even M_BLOCKS=1 benefits from pipelining A loads.

### Optimization 3: Persistent Kernel (commit 78fb6bb)

**What:** Converted the kernel from one-block-per-tile to a persistent
work loop. Each block processes multiple (m_tile, n_tile, k_split) work
items in round-robin. The launcher auto-selects k_splits when
mn_tiles < num_SMs to create enough work to fill all SMs.

**M_BLOCKS dispatch:** With the persistent kernel, the SM utilization
concern for M_BLOCKS selection is removed. The dispatcher now simply
picks the largest M_BLOCKS that fits M (>48→4, >32→3, >16→2, else 1).

**Impact — mixed results, needs tuning:**

The persistent kernel improved some shapes (especially M=32-64 at
N=16384) but regressed others. The auto k_splits introduces atomicAdd
overhead that hurts shapes where k_splits=1 was previously sufficient.

### Optimization 2: Larger N_BLOCKS (TILE_N=256) — ATTEMPTED, REVERTED

**What was tried:** Increased TILE_N from 128 to 256 and N_BLOCKS from
2 to 4, keeping 8 warps all along N. Each warp covers 32 columns (4
N-blocks) instead of 16 (2 N-blocks). The repack format stayed at
KBIT_TILE_N=128; the kernel loaded two adjacent repack tiles per GEMM
tile (contiguous in memory, so the addressing worked naturally).

**Result: massive regression.** The grid size halved (n_tiles = N/256
instead of N/128), cutting SM utilization in half. For bandwidth-bound
shapes, fewer active SMs means less aggregate memory bandwidth:

| Shape | Before (TILE_N=128) | After (TILE_N=256) |
|-------|:---:|:---:|
| M=4, N=11008 | **2.02x** | 0.92x |
| M=1, N=16384 | **1.84x** | 1.00x |
| M=16, N=16384 | **1.98x** | 1.30x |

**Root cause:** For bandwidth-bound shapes, SM utilization matters more
than per-tile compute efficiency. Doubling TILE_N gives 2x more compute
per tile but halves the number of tiles, reducing total memory bandwidth
the GPU can deliver.

**This approach will work after the persistent kernel is properly tuned**
(since persistent always launches num_SMs blocks regardless of tile count).
But the persistent kernel itself needs the k_splits overhead fixed first.

---

## Current Benchmark (RTX 4090, K=4, fp16, persistent kernel)

### Standard shapes (N=4096)

| M | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|---:|------:|------:|----------:|------------:|--------:|
| 1 | 4096 | 4096 | 72 | 43 | 0.59x |
| 4 | 4096 | 4096 | 70 | 26 | 0.37x |
| 8 | 4096 | 4096 | 73 | 22 | 0.30x |
| 16 | 4096 | 4096 | 82 | 23 | 0.28x |
| 32 | 4096 | 4096 | 70 | 27 | 0.38x |
| 64 | 4096 | 4096 | 70 | 25 | 0.36x |
| 128 | 4096 | 4096 | 71 | 32 | 0.46x |

### Large-N shapes (bandwidth-bound — target regime)

| M | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|---:|------:|------:|----------:|------------:|--------:|
| 1 | 4096 | 11008 | 83 | 100 | **1.22x** |
| 1 | 4096 | 16384 | 90 | 175 | **1.95x** |
| 4 | 4096 | 11008 | 83 | 125 | **1.50x** |
| 4 | 4096 | 16384 | 81 | 165 | **2.04x** |
| 16 | 4096 | 11008 | 103 | 120 | **1.17x** |
| 16 | 4096 | 16384 | 82 | 145 | **1.76x** |
| 32 | 4096 | 16384 | 97 | 172 | **1.77x** |
| 64 | 4096 | 16384 | 92 | 157 | **1.71x** |
| 128 | 4096 | 16384 | 219 | 200 | 0.91x |

### Comparison: persistent kernel vs pre-persistent baseline

| Shape | Pre-persistent | Persistent | Change |
|-------|:---:|:---:|:---:|
| M=1, N=4096 | 0.79x | 0.59x | -25% (k_splits overhead) |
| M=1, N=11008 | 1.38x | 1.22x | -12% (k_splits overhead) |
| M=1, N=16384 | 1.84x | **1.95x** | +6% |
| M=4, N=11008 | 2.02x | 1.50x | -26% (k_splits overhead) |
| M=4, N=16384 | 1.75x | **2.04x** | +17% |
| M=32, N=16384 | 1.55x | **1.77x** | +14% |
| M=64, N=16384 | 1.10x | **1.71x** | +55% |
| M=128, N=16384 | 1.12x | 0.91x | -19% (M_BLOCKS dispatch) |

---

## Critical Analysis: What Needs Fixing

### Problem 1: Auto k_splits is too aggressive

The persistent kernel auto-splits K whenever `mn_tiles < num_SMs`. For
shapes like M=4, N=11008 (mn_tiles=86, num_SMs=128), it uses k_splits=2.
This introduces atomicAdd + tile_counters + fp32 workspace + final
conversion overhead, which outweighs the benefit of filling 128 SMs
instead of 86.

**Fix options (in order of preference):**

1. **Higher threshold:** Only auto-split when mn_tiles < num_SMs / 4
   (severe underutilization). For most shapes, k_splits stays at 1.

2. **Conditional workspace:** Pass a flag from Python indicating whether
   the workspace is zeroed. Only use k_splits > 1 when the workspace is
   available and zeroed.

3. **Remove auto k_splits entirely:** Let the Python side control it
   (restore the k_chunks parameter behavior). The persistent loop still
   benefits from load balancing across waves even with k_splits=1.

**Recommendation:** Option 1. Change the threshold from `mn_tiles < num_sms`
to `mn_tiles < num_sms / 4` in `kbitGemmProdLaunch`. This means k_splits > 1
only activates for truly small grids (< 32 tiles on 128 SMs).

### Problem 2: M=128, N=16384 regression

With M=128, the dispatcher selects M_BLOCKS=4 (TILE_M=64). This gives
m_tiles=2, n_tiles=128, mn_tiles=256, k_tiles=64. With k_splits=1 and
256 work items on 128 SMs, each SM handles 2 tiles. The persistent loop
overhead (zeroing accumulators, re-initializing pipeline per tile) may
explain the 0.91x vs previous 1.12x.

**Fix:** For shapes where mn_tiles >= num_SMs (full utilization without
k_splits), the persistent loop overhead hurts. Consider a fast path that
skips the loop when total_work == gridDim.x (each block handles exactly
one work item, equivalent to non-persistent behavior).

### Problem 3: N=4096 is still 0.28-0.59x vs cuBLAS

Even with the persistent kernel filling SMs via k_splits, N=4096 shapes
are far behind cuBLAS. The issue is fundamental: each warp only does
2 MMAs per B-fragment dequant (N_BLOCKS=2). cuBLAS uses much larger
tiles and achieves higher compute-per-load ratios.

**Fix:** This is where larger N_BLOCKS will help, but it requires
TILE_N=256 (grid halving), which only works with a properly-tuned
persistent kernel that doesn't suffer from the k_splits overhead.

---

## Remaining Optimizations (Revised Priority Order)

### 1. Tune Persistent Kernel k_splits Threshold (HIGHEST, quick fix)

Raise the auto k_splits threshold to avoid the atomicAdd overhead for
shapes that already have reasonable SM utilization. Add a fast path for
work_items == gridDim.x to eliminate loop overhead when all SMs are busy.

**Expected impact:** Restore the pre-persistent performance for large-N
shapes (M=4 N=11008 back to ~2.0x) while keeping the persistent benefit
for shapes that need it (M=32-64 N=16384).

### 2. Larger N_BLOCKS (TILE_N=256) — RE-ATTEMPT after k_splits fix

With the persistent kernel properly tuned, TILE_N=256 should work
because the grid size reduction is irrelevant (persistent always uses
num_SMs blocks). The implementation from the reverted attempt is known
to be correct (85 tests passed). Key details:

- No repack changes needed: the kernel loads two adjacent 128-wide
  repack tiles per 256-wide GEMM tile (contiguous in memory)
- Shmem budget OK: worst case K=5 MB=4 is ~38 KB per block (2 stages)
- Register headroom: ~32 extra float accumulators, estimated ~147 regs
- Requires N % 256 == 0 (all LLM shapes satisfy this)

### 3. C Output Staging (LOW, polish)

Coalesced global writes instead of scattered fragment writes.
5-15% for small K_dim. Implement after 1+2 are done and benchmarked.

---

## Lessons Learned

1. **cp.async for A was not "low priority."** Originally rated 2-5%
   impact, it was the single most impactful change because it removed
   synchronous work from the pipeline critical path.

2. **Tile size increases halve the grid.** Both multi-M-block and
   TILE_N=256 initially caused regressions because the grid shrank,
   reducing SM utilization. Any tile size increase needs either an
   SM-aware dispatch that avoids it when the grid is small, or a
   persistent kernel that decouples grid size from SM utilization.

3. **Auto k_splits has high overhead.** The atomicAdd + fp32 workspace +
   tile_counters + final conversion path is significantly more expensive
   than direct writes. Only use it when the SM utilization gain clearly
   outweighs the overhead (mn_tiles << num_SMs).

4. **The persistent loop itself has overhead.** Zeroing accumulators and
   re-initializing the cp.async pipeline per work item adds cycles. When
   each SM only handles one tile (total_work <= gridDim.x), the loop
   overhead is pure waste. Add a fast path.

5. **Register pressure is not an issue.** Even M_BLOCKS=4 with K=5 uses
   only 115 registers with zero spills. There's headroom for N_BLOCKS=4.

6. **Benchmark variance is significant.** Small-M kernel times fluctuate
   10-20% between runs. Use high iteration counts (500+) and focus on
   relative trends.

---

## Implementation Order

1. **Tune k_splits threshold + fast path** — highest priority, should be
   a small change to the launcher. Re-benchmark to confirm regressions
   are fixed.

2. **Re-attempt TILE_N=256** — once the persistent kernel is tuned, the
   grid size halving is no longer a concern. The implementation is
   already validated (tests passed in the reverted attempt).

3. **C staging** — polish optimization, low priority.

After steps 1+2, re-benchmark. The target is ≥1.5x vs cuBLAS for all
LLM shapes (M=1-64, N=4096-16384). If N=4096 shapes are still slow,
consider whether they matter for the target use case (they may not — LLM
inference typically has N ≥ 11008 for the large linear layers).

---

## Integration Work (Not Performance)

Required to ship, independent of performance optimizations:

- **Wire into LinearNbit module:** Call `kbit_gemm_prod` instead of
  dequant+cuBLAS when CUDA, fp16/bf16, N % TILE_N == 0, K_dim % 64 == 0
- **Remove staging kernels:** Delete Stages 3-5 (minimal, pipelined,
  split-K), keep only production kernel + MMA test
- **Lint + PR:** ruff/clang-format, merge to main
