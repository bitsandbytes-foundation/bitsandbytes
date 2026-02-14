# kbit GEMM Kernel: Optimization Guide

This document records the current state of the production kernel
`kbit_gemm_prod`, comprehensive performance data on real model shapes,
analysis of bottlenecks, and a detailed roadmap for further optimization.

All benchmarks: RTX 4090 (128 SMs, sm_89), GPU clocks locked at 2520 MHz,
500 iterations, K=4, fp16 unless stated otherwise.

---

## 1. Current Kernel Configuration

**Template parameters:** `<K_BITS, M_BLOCKS, scalar_t>`

- **TILE_M** = M_BLOCKS * 16 (M_BLOCKS selected at runtime: 1, 2, 3, or 4)
- **TILE_N** = 128, **N_BLOCKS** = 2 (each warp covers 16 columns)
- **TILE_K** = 64 (4 MMA k-sub-tiles of 16)
- 256 threads = 8 warps, all warps share the same M rows, each handles a
  different N slice
- Persistent work loop with tuned k_splits (only for mn_tiles < num_SMs/4)
- Non-persistent grid (grid_size = total_work) when k_splits = 1
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

## 2. Performance on Real Model Shapes

### Target batch size: M >= 32

The kernel targets LLM inference with batch sizes M=32-64. These are the
GEMM shapes from the FFN and attention layers of real models.

### 2.1 K=4, fp16 — All M values

| Layer | K_dim | N | M=1 | M=4 | M=16 | M=32 | M=64 | M=128 |
|-------|------:|-----:|:---:|:---:|:----:|:----:|:----:|:-----:|
| Llama3-70B gate/up | 8192 | 28672 | 2.81x | 2.52x | 2.37x | **2.84x** | **1.93x** | 0.91x |
| Llama3-8B gate/up | 4096 | 14336 | 1.47x | 1.94x | 1.77x | **1.65x** | **1.41x** | 0.91x |
| Llama2-7B gate/up | 4096 | 11008 | 1.00x | 1.65x | 1.43x | **1.38x** | **1.05x** | 0.76x |
| Llama3-70B down | 28672 | 8192 | 1.00x | 1.16x | 1.08x | 0.66x | 0.93x | 0.74x |
| Llama3-8B down | 14336 | 4096 | 0.51x | 0.52x | 0.56x | 0.45x | 0.43x | 0.48x |
| Llama2-7B down | 11008 | 4096 | 0.59x | 0.41x | 0.56x | 0.80x | 0.44x | 0.48x |
| Llama3-8B QKV | 4096 | 4096 | 0.48x | 0.25x | 0.27x | 0.31x | 0.21x | 0.41x |
| GLM4.7 shared gate/up | 2048 | 10240 | 0.70x | 0.30x | 0.33x | 0.36x | 0.40x | 0.51x |
| GLM4.7 routed expert | 2048 | 1536 | 0.30x | 0.30x | 0.30x | - | - | - |

### 2.2 Absolute timings (M=32, M=64, K=4, fp16)

| Layer | K_dim | N | M=32 kbit | M=32 cuBLAS | M=64 kbit | M=64 cuBLAS |
|-------|------:|-----:|----------:|------------:|----------:|------------:|
| Llama3-70B gate/up | 8192 | 28672 | 232 us | 674 us | 312 us | 652 us |
| Llama3-8B gate/up | 4096 | 14336 | 101 us | 173 us | 101 us | 142 us |
| Llama2-7B gate/up | 4096 | 11008 | 121 us | 138 us | 118 us | 104 us |
| Llama3-8B down | 14336 | 4096 | 288 us | 141 us | 319 us | 159 us |
| Llama3-70B down | 28672 | 8192 | 537 us | 503 us | 677 us | 534 us |

### 2.3 K-value comparison (M=32)

| Layer | K=2 | K=3 | K=4 | K=5 |
|-------|:---:|:---:|:---:|:---:|
| Llama3-8B gate/up | 1.92x | 1.86x | 1.65x | 1.55x |
| Llama2-7B gate/up | 1.24x | 1.36x | 1.18x | 0.93x |
| Llama3-70B gate/up | 2.62x | 2.62x | 2.10x | 1.93x |
| Llama3-8B down | 0.70x | 0.59x | 0.55x | 0.52x |
| Llama3-70B down | 1.12x | 0.90x | 0.95x | 0.97x |

All K values follow the same pattern. K=2-3 are slightly faster (fewer
bit-planes to load and extract). The kernel generalizes well across K.

### 2.4 Summary

**Where the kernel wins (gate/up projections, N >= 11008):**
- Llama3-70B: 1.9-2.8x at M=32-64
- Llama3-8B: 1.4-1.7x at M=32-64
- Llama2-7B: 1.1-1.4x at M=32-64

**Where the kernel loses:**
- Down projections (N=4096-8192): 0.4-0.9x — small N means low SM utilization
- Attention QKV (N=4096): 0.2-0.5x — same problem
- GLM-4.7-Flash (K_dim=2048): 0.3-0.7x — K_dim too small for pipeline
- M=128: performance degrades across all shapes

---

## 3. Performance Analysis

### 3.1 Bandwidth utilization (M=32, K=4)

| Layer | Data read | Kernel time | Achieved BW | % of 900 GB/s |
|-------|----------:|------------:|------------:|---------------:|
| Llama3-8B gate/up | 32 MB | 97 us | 333 GB/s | 37% |
| Llama2-7B gate/up | 25 MB | 83 us | 300 GB/s | 33% |
| Llama3-70B gate/up | 127 MB | 248 us | 513 GB/s | 57% |
| Llama3-8B down | 32 MB | 257 us | 126 GB/s | 14% |
| Llama3-70B down | 127 MB | 511 us | 249 GB/s | 28% |

The kernel achieves 33-57% of peak bandwidth for gate/up shapes and only
14-28% for down shapes. For K=4 quantized weights, the theoretical maximum
speedup over cuBLAS fp16 GEMM is approximately 4x (reading 4x less data).
We are at 1.2-2.8x, meaning significant headroom remains.

### 3.2 Why gate/up wins but down loses

The key variable is N (output columns), not K_dim (reduction dimension).

**Gate/up (large N):** N=11008-28672 gives n_tiles=86-224 with TILE_N=128.
Most or all SMs are occupied, achieving good aggregate bandwidth.

**Down (small N):** N=4096-8192 gives n_tiles=32-64 with TILE_N=128. At
M=32 with M_BLOCKS=2, m_tiles=1, so mn_tiles=32-64. On 128 SMs, that is
25-50% utilization. Fewer active SMs means less aggregate memory bandwidth
and less concurrent compute.

### 3.3 Compute bottleneck: the dequant inner loop

Per TILE_K iteration, each warp executes:

1. **Load A fragments** via ldmatrix: M_BLOCKS ldmatrix.x4 per k-sub-tile
2. **For each N-block (2 iterations):**
   - 4 shmem loads (B bit-planes, K=4)
   - 1 absmax decode (shmem load + 5 ALU ops)
   - 4 elements x (4 shifts + 4 ANDs + 3 ORs) = 44 ALU ops for bit extraction
   - 4 `__shfl_sync` for codebook lookup
   - 4 multiplies for absmax scaling
   - 2 `pack_two` for fragment assembly
   - M_BLOCKS MMA instructions

Per TILE_K iteration (4 k-sub-tiles x 2 N-blocks = 8 inner iterations):
~472 ALU + 32 shuffles + 32 shmem loads + 8*M_BLOCKS MMAs.

The ALU work (bit extraction + codebook lookup + scaling) is dense and
partially serialized with the MMA operations because they share the same
warp's instruction stream. The MMA runs on the tensor core (independent
functional unit) but the dequant ALU work must complete before the MMA
can issue, creating a dependency chain.

With N_BLOCKS=2, each B-fragment dequant feeds only 2 MMAs (per M_BLOCKS).
Doubling to N_BLOCKS=4 would amortize the dequant cost over 4 MMAs,
halving the effective compute overhead per useful FLOP.

### 3.4 SM utilization analysis (TILE_N=128 vs 256)

For M=32, M_BLOCKS=2, TILE_M=32, m_tiles=1:

| Shape | TILE_N=128 n_tiles | SM util | TILE_N=256 n_tiles | SM util |
|-------|-------------------:|--------:|-------------------:|--------:|
| Llama3-70B gate/up (N=28672) | 224 | 100% | 112 | 87% |
| Llama3-8B gate/up (N=14336) | 112 | 87% | 56 | 44% |
| Llama2-7B gate/up (N=11008) | 86 | 67% | 43 | 34% |
| Llama3-70B down (N=8192) | 64 | 50% | 32 | 25% |
| Llama3-8B down (N=4096) | 32 | 25% | 16 | 12% |

TILE_N=256 halves the SM utilization. For 70B gate/up (87% → 87%), this
is fine. For 8B gate/up (87% → 44%), it is a concern. For down
projections, it would be catastrophic.

**Implication:** TILE_N=256 should only be used when N is large enough
that the SM utilization loss is acceptable, or when the persistent
kernel with k_splits compensates. Shape-adaptive dispatch is needed.

---

## 4. Completed Optimizations

### 4.1 Multi-M-Block Tiling (commit f8a06a3)

Templated `kbit_gemm_prod` on `M_BLOCKS` (1-4). Each warp loads
M_BLOCKS A fragments per k-sub-tile and reuses the same dequantized B
fragment across all of them, amortizing dequant cost per M row.

### 4.2 cp.async for A Tile (commit 7cd575b)

Replaced synchronous A tile loading with cp.async 16-byte copies with XOR
swizzle. Single most impactful change: improved ALL shapes by pipelining
A loads alongside compute.

### 4.3 Persistent Kernel (commit 78fb6bb)

Converted from one-block-per-tile to a persistent work loop. Each block
processes multiple (m_tile, n_tile, k_split) work items in round-robin.
Auto-selects k_splits when mn_tiles < num_SMs/4.

### 4.4 k_splits Threshold Tuning (commit 6e18c03)

Raised auto k_splits threshold from `mn_tiles < num_sms` to
`mn_tiles < num_sms / 4`. Uses non-persistent grid (grid_size = total_work)
when k_splits = 1 to avoid loop overhead. Key result: M=128 N=16384
improved from 0.86x to 0.98x (+13%).

### 4.5 TILE_N=256 — Previously Attempted and Reverted

Increased TILE_N to 256 and N_BLOCKS to 4. Implementation was correct (85
tests passed), but halving the grid caused massive regression because SM
utilization dropped. This approach requires shape-adaptive dispatch.

---

## 5. Optimization Roadmap

### Step 1: TILE_N=256 + TILE_K=128 (HIGHEST PRIORITY)

**What:** Increase both tile dimensions simultaneously:
- TILE_N: 128 → 256, N_BLOCKS: 2 → 4
- TILE_K: 64 → 128, k-sub-tiles per iteration: 4 → 8

**Why both at once:** They address different bottlenecks and interact:
- TILE_N=256 halves dequant-per-MMA (4 MMAs per B dequant instead of 2),
  directly reducing the compute bottleneck identified in section 3.3
- TILE_K=128 doubles compute per pipeline iteration, halving the number
  of `__syncthreads()` barriers and improving pipeline amortization

**Shared memory budget (2 stages):**

| M_BLOCKS | K | A stage | B stage | Absmax | Total/stage | 2 stages |
|---------:|--:|--------:|--------:|-------:|------------:|---------:|
| 1 | 4 | 4 KB | 16 KB | 512 B | 20.5 KB | 41 KB |
| 2 | 4 | 8 KB | 16 KB | 512 B | 24.5 KB | 49 KB |
| 4 | 4 | 16 KB | 16 KB | 512 B | 32.5 KB | 65 KB |
| 4 | 5 | 16 KB | 20 KB | 512 B | 36.5 KB | 73 KB |

Computation for TILE_N=256, TILE_K=128:
- A stage: M_BLOCKS * 16 * 128 * 2 bytes
- B stage: 256 * (128/32) * K * 4 bytes = 256 * 4 * K * 4
- Absmax: 256 * (128/32) = 1024 bytes, aligned to 16 → 1024 bytes

RTX 4090 supports up to 100 KB dynamic shmem per block. M_BLOCKS=4 K=5
at 73 KB fits. M_BLOCKS=2 (the M=32 case) at 49 KB fits easily.

**Register estimate:** N_BLOCKS=4 adds 2 extra float accumulators per
M_BLOCK (frag_c grows from [MB][2][4] to [MB][4][4]). For M_BLOCKS=2
K=4, estimated ~104 regs (from 72 + 32). Still zero spills.

**Repack compatibility:** The repack kernel uses KBIT_TILE_N=128. The
GEMM kernel would load two adjacent repack tiles per 256-wide GEMM
tile. They are contiguous in memory (tile layout is kt * n_tiles + nt),
so this works naturally with the existing repack format.

**Shape-adaptive dispatch:** TILE_N=256 only when N >= 10240. For
smaller N, keep TILE_N=128 to preserve SM utilization. This requires
adding TILE_N as a template parameter (doubles instantiation count to
64 variants, acceptable for compilation time).

**TILE_K=128 constraint:** Requires K_dim >= 128 (satisfied by all real
model shapes) and K_dim % 128 == 0 or a boundary check in the pipeline.
Most LLM shapes have K_dim divisible by 128. For shapes where K_dim is
only divisible by 64, fall back to TILE_K=64.

**Expected impact:**
- Gate/up projections: the dequant-per-MMA ratio halves. If dequant is
  ~50% of kernel time (section 3.3), expect ~25% speedup. Combined with
  TILE_K=128 pipeline improvements: **30-50% speedup**.
- Llama3-8B gate/up M=32: 1.65x → ~2.0-2.5x
- Llama3-70B gate/up M=32: 2.10x → ~2.5-3.0x
- Down projections with TILE_N=128 fallback: unchanged

### Step 2: B Fragment Register Double-Buffering (HIGH)

**What:** Preload the next N-block's B bit-planes from shmem while the
current N-block's MMA executes on the tensor core.

Current inner loop (per k-sub-tile):
```
load A fragment
for nb = 0..N_BLOCKS-1:
    load B planes from shmem     ← stalls until data arrives
    dequant (bit extract + shuffle + scale)
    MMA                          ← tensor core, independent unit
```

Optimized (software-pipelined):
```
load A fragment
preload B planes for nb=0
for nb = 0..N_BLOCKS-1:
    dequant current B planes     ← uses already-loaded data
    preload B planes for nb+1    ← overlaps with dequant ALU
    MMA                          ← overlaps with next preload
```

**Why it helps:** The shmem loads for B planes have ~20-30 cycle latency.
By issuing the loads for the next iteration before the current dequant,
the warp scheduler can interleave these instructions, hiding the latency.
Marlin uses this pattern (frag_b_quant[k%2]).

**Expected impact:** 10-20% improvement from hiding shmem load latency.

### Step 3: C Output Staging via Shared Memory (MEDIUM)

**What:** Instead of scattered fragment writes directly to global memory,
stage the output tile in shared memory first, then write coalesced.

Current write pattern: each thread writes 2 elements at
`C[m_row, c_col]` and `C[m_row, c_col+1]`. Within a warp, 8 different
rows are written (gid=0..7), each ~22 KB apart for N=11008. This is
non-coalesced: 8 separate cache lines per warp write.

Staged: after compute, store fragments to shmem (bank-conflict-free layout),
then __syncthreads, then coalesced 16-byte writes to global memory.

**Expected impact:** 5-15% for shapes with small K_dim (faster output
relative to total kernel time). Less impact for large K_dim.

### Step 4: Deeper cp.async Pipeline (MEDIUM)

**What:** Increase pipeline depth from 2 stages to 3 or 4 stages.

With 2 stages, `cp_async_wait<1>()` blocks until the previous group
completes. With 3 stages, `cp_async_wait<2>()` allows 2 groups to be
outstanding, providing more slack for variable memory latency.

**Trade-off:** Each extra stage costs one STAGE_BYTES of shmem. For
TILE_N=256 TILE_K=128 M_BLOCKS=2 K=4: 24.5 KB per stage. 3 stages =
73.5 KB (fits), 4 stages = 98 KB (tight, might not fit with M_BLOCKS=4).

**Expected impact:** 5-10% improvement from better load-compute overlap,
especially for shapes with variable memory access latency.

### Step 5: Revisit k_splits for Down Projections (LOW)

**What:** For down projections (N=4096, mn_tiles=32), the kernel has
only 25% SM utilization. k_splits could fill more SMs at the cost of
atomicAdd overhead.

**Analysis needed:** Profile whether the bandwidth gain from filling
SMs outweighs the atomicAdd + workspace + conversion overhead for these
specific shapes. The current threshold (mn_tiles < num_sms/4 = 32) means
N=4096 is right at the boundary.

**Alternative:** Accept that down projections (N <= 4096) are not the
kernel's target regime. In real inference, the gate/up projection
dominates runtime (larger matrix), so optimizing gate/up is more
impactful for end-to-end latency.

### Step 6: Warp Specialization (FUTURE, if needed)

**What:** Dedicated producer warps (issue cp.async loads) and consumer
warps (dequant + MMA), communicating via shmem barriers.

**When:** Only if Steps 1-4 do not reach the target speedup. This is
complex (barrier management, warp role assignment, reduced consumer
parallelism) and should only be attempted after simpler optimizations
are exhausted.

**Expected impact:** Could push bandwidth utilization from 33-57% toward
70-80%, yielding an additional 30-50% speedup on top of Steps 1-4.

---

## 6. Implementation Order

### Phase 1: TILE_N=256 + TILE_K=128 with shape-adaptive dispatch

1. Add TILE_N as a template parameter alongside M_BLOCKS
2. Implement TILE_K=128 inner loop (8 k-sub-tiles per iteration)
3. Shape-adaptive dispatch: TILE_N=256 for N >= 10240, TILE_N=128 otherwise
4. TILE_K=128 when K_dim % 128 == 0, TILE_K=64 fallback otherwise
5. Update Python side: N % 256 check for large-N path, workspace sizing
6. Update tests: add N=256 minimum for TILE_N=256 test shapes
7. Benchmark all real model shapes at M=32, M=64 across K=2-5
8. Compare gate/up speedups to current baseline

### Phase 2: Inner loop optimization

9. B fragment register double-buffering
10. C output staging via shmem
11. Benchmark and compare

### Phase 3: Pipeline tuning

12. 3-stage pipeline (if shmem permits)
13. Re-evaluate k_splits for down projections
14. Final benchmark sweep

### Phase 4: Integration

15. Wire into LinearNbit module
16. Remove staging kernels (keep only production + MMA test)
17. Lint (ruff, clang-format) and PR to main

---

## 7. Target Performance

For M=32, K=4:

| Layer | Current | After Phase 1 (est.) | After Phase 2 (est.) | Theoretical max |
|-------|:-------:|:--------------------:|:--------------------:|:---------------:|
| Llama3-70B gate/up | 2.10x | 2.5-3.0x | 2.8-3.5x | ~4x |
| Llama3-8B gate/up | 1.65x | 2.0-2.5x | 2.3-2.8x | ~4x |
| Llama2-7B gate/up | 1.18x | 1.5-2.0x | 1.7-2.2x | ~4x |
| Llama3-70B down | 0.95x | ~1.0x | ~1.0-1.2x | ~4x |
| Llama3-8B down | 0.55x | ~0.55x | ~0.6-0.7x | ~4x |

Down projections (small N) are unlikely to be competitive. The target is
to make gate/up projections fast enough that the overall FFN inference
time (gate/up + down combined) is faster than cuBLAS fp16 for both
projections combined. At M=32:

- Llama3-8B FFN: gate/up (101 us kbit) + down (288 us kbit) = 389 us
  vs gate/up (173 us cuBLAS) + down (141 us cuBLAS) = 314 us.
  Currently 0.81x overall. Need gate/up to drop to ~50 us to break even.

- Llama3-70B FFN: gate/up (232 us kbit) + down (537 us kbit) = 769 us
  vs gate/up (674 us cuBLAS) + down (503 us cuBLAS) = 1177 us.
  Currently **1.53x overall**. Phase 1 target: ~2x overall.

The 70B model is where the kernel has real end-to-end impact today.
For 7-8B models, the gate/up wins are offset by down projection losses.

---

## 8. Model Shape Reference

### GLM-4.7-Flash (MoE, hidden=2048)

Not a good target for this kernel. K_dim=2048 is too small for the
pipeline (only 32 TILE_K iterations with TILE_K=64, 16 with TILE_K=128).
All layers lose to cuBLAS.

- Shared expert: K_dim=2048, N=10240
- Routed expert: K_dim=2048, N=1536 (64 experts, top-4)
- Attention: MLA with q_lora_rank=768, kv_lora_rank=512

### Llama-style models (good targets)

| Model | hidden | intermediate | QKV | gate/up (N) | down (N) |
|-------|-------:|-------------:|----:|------------:|---------:|
| Llama 2 7B | 4096 | 11008 | 4096 | 11008 | 4096 |
| Llama 3 8B | 4096 | 14336 | 4096 | 14336 | 4096 |
| Llama 2 13B | 5120 | 13824 | 5120 | 13824 | 5120 |
| Llama 3 70B | 8192 | 28672 | 8192 | 28672 | 8192 |
| Mistral 7B | 4096 | 14336 | 4096 | 14336 | 4096 |
| Qwen2.5 7B | 3584 | 18944 | 3584 | 18944 | 3584 |

All gate/up N values are divisible by 256. All K_dim values are
divisible by 128 (required for TILE_K=128).

---

## 9. Lessons Learned

1. **Benchmark on real model shapes, not synthetic grids.** Synthetic
   shapes (M=4, K=4096, N=16384) showed 2.0x+ but the actual model
   shapes tell a different story. Down projections are a problem.

2. **SM utilization dominates for bandwidth-bound shapes.** Any tile
   size increase that reduces the grid size reduces aggregate bandwidth.
   Shape-adaptive dispatch is essential.

3. **The kernel is compute-bound at 33-57% bandwidth utilization.**
   The dequant inner loop (bit extraction + codebook shuffle + absmax
   scaling) is the primary bottleneck, not memory bandwidth.

4. **N_BLOCKS is the most impactful knob.** Going from N_BLOCKS=2 to 4
   doubles the compute amortization per dequant. This directly attacks
   the compute bottleneck.

5. **K_dim must be large (>= 4096) for the pipeline to be effective.**
   With K_dim=2048 (GLM-4.7-Flash), the pipeline has too few iterations
   to amortize startup/drain overhead.

6. **Larger K benefits from this kernel more.** 70B models (K_dim=8192,
   N=28672) show 2-3x speedup. The larger both dimensions are, the
   better the kernel performs relative to cuBLAS.

7. **End-to-end FFN analysis matters.** The kernel must be fast enough
   on gate/up to compensate for the down projection loss, or the down
   projection must also be competitive. For 70B models, the gate/up win
   is large enough to dominate. For 7-8B models, it is marginal.
