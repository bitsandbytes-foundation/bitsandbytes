# kbit GEMM Kernel: Optimization Guide

This document records the current state of the production kernel
`kbit_gemm_prod`, comprehensive performance data on real model shapes,
analysis of bottlenecks, and a detailed roadmap for further optimization.

All benchmarks: RTX 4090 (128 SMs, sm_89), GPU clocks locked at 2520 MHz,
300 iterations, K=4, fp16 unless stated otherwise.

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

The kernel targets LLM inference with batch sizes M=32-64.

### 2.1 Primary target: MoE models (Qwen3-Coder-Next, GLM-4.7-Flash)

These MoE models have small hidden_size (2048) and small per-expert
dimensions. They represent the most important and most challenging target.

**Qwen3-Coder-Next** (70B+ MoE, hidden=2048, 512 experts, 10 per token):

| Layer | K_dim | N | M=1 | M=4 | M=16 | M=32 | M=64 |
|-------|------:|-----:|:---:|:---:|:----:|:----:|:----:|
| Dense gate/up | 2048 | 5120 | 0.32x | 0.31x | 0.32x | 0.28x | 0.46x |
| Dense down | 5120 | 2048 | 0.31x | 0.31x | 0.31x | 0.37x | 0.37x |
| Q proj | 2048 | 4096 | 0.29x | 0.22x | 0.42x | 0.39x | 0.37x |
| O proj | 4096 | 2048 | 0.29x | 0.30x | 0.31x | 0.37x | 0.31x |
| MoE gate/up | 2048 | 512 | 0.28x | 0.38x | 0.37x | 0.37x | 0.37x |
| MoE down | 512 | 2048 | 0.25x | 0.35x | 0.36x | 0.31x | 0.33x |
| Shared expert | 2048 | 512 | 0.28x | 0.35x | 0.39x | 0.42x | 0.37x |

**GLM-4.7-Flash** (MoE, hidden=2048, 64 experts, top-4):

| Layer | K_dim | N | M=1 | M=4 | M=16 | M=32 | M=64 |
|-------|------:|-----:|:---:|:---:|:----:|:----:|:----:|
| Shared gate/up | 2048 | 10240 | 0.73x | 0.27x | 0.35x | 0.34x | 0.40x |
| Shared down | 10240 | 2048 | 0.68x | 0.39x | 0.43x | 0.45x | 0.36x |
| Routed gate/up | 2048 | 1536 | 0.28x | 0.32x | 0.28x | 0.47x | 0.37x |
| Routed down | 1536 | 2048 | 0.31x | 0.33x | 0.31x | 0.35x | 0.32x |

**Every single Qwen3 and GLM-4.7 shape loses to cuBLAS.** The kernel is
2-3.5x slower across the board for these models.

### 2.2 Secondary target: Dense Llama-style models

These dense models have large K_dim (4096-8192) and large N for gate/up
projections. The kernel performs well on gate/up but loses on down
projections.

| Layer | K_dim | N | M=1 | M=4 | M=16 | M=32 | M=64 |
|-------|------:|-----:|:---:|:---:|:----:|:----:|:----:|
| Llama3-70B gate/up | 8192 | 28672 | 2.51x | 1.82x | 3.06x | **2.21x** | **1.84x** |
| Llama3-8B gate/up | 4096 | 14336 | 1.67x | 1.67x | 2.10x | **1.64x** | **1.42x** |
| Llama3-70B down | 28672 | 8192 | 1.36x | 1.00x | 0.94x | 0.99x | 0.88x |
| Llama3-8B down | 14336 | 4096 | 0.53x | 0.54x | 0.44x | 0.61x | 0.46x |

### 2.3 Absolute timings (M=32, K=4, fp16)

| Layer | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|-------|------:|-----:|----------:|------------:|--------:|
| Qwen3 dense gate/up | 2048 | 5120 | 90.6 | 37.6 | 0.41x |
| Qwen3 dense down | 5120 | 2048 | 81.9 | 37.9 | 0.46x |
| Qwen3 Q proj | 2048 | 4096 | 84.3 | 29.7 | 0.35x |
| Qwen3 O proj | 4096 | 2048 | 96.1 | 43.4 | 0.45x |
| Qwen3 MoE gate/up | 2048 | 512 | 76.2 | 30.8 | 0.40x |
| Qwen3 MoE down | 512 | 2048 | 78.9 | 22.6 | 0.29x |
| Qwen3 shared expert | 2048 | 512 | 70.6 | 27.4 | 0.39x |
| GLM4.7 shared gate/up | 2048 | 10240 | 72.7 | 25.9 | 0.36x |
| GLM4.7 shared down | 10240 | 2048 | 88.6 | 27.1 | 0.31x |
| GLM4.7 routed gate/up | 2048 | 1536 | 108.4 | 42.2 | 0.39x |
| GLM4.7 routed down | 1536 | 2048 | 107.7 | 36.5 | 0.34x |
| Llama3-8B gate/up | 4096 | 14336 | 82.5 | 138.8 | 1.68x |
| Llama3-70B gate/up | 8192 | 28672 | 230.4 | 511.1 | 2.22x |
| Llama3-8B down | 14336 | 4096 | 335.0 | 184.9 | 0.55x |
| Llama3-70B down | 28672 | 8192 | 524.2 | 520.3 | 0.99x |

### 2.4 K-value comparison (M=32)

| Layer | K=2 | K=3 | K=4 | K=5 |
|-------|:---:|:---:|:---:|:---:|
| Llama3-8B gate/up | 1.92x | 1.86x | 1.65x | 1.55x |
| Llama3-70B gate/up | 2.62x | 2.62x | 2.10x | 1.93x |
| Llama3-8B down | 0.70x | 0.59x | 0.55x | 0.52x |
| Llama3-70B down | 1.12x | 0.90x | 0.95x | 0.97x |

---

## 3. Performance Analysis

### 3.1 The overhead problem: bandwidth utilization and overhead multiplier

The most revealing metric is the **overhead multiplier**: how many times
slower the kernel is compared to the pure bandwidth floor (data size /
peak bandwidth).

| Layer | K_dim | N | Data (MB) | n_tiles | SM% | k_tiles | BW% | kbit (us) | Overhead |
|-------|------:|-----:|----------:|--------:|----:|--------:|----:|----------:|---------:|
| Qwen3 dense gate/up | 2048 | 5120 | 5.7 | 40 | 31% | 32 | 7% | 90.6 | **14.3x** |
| Qwen3 dense down | 5120 | 2048 | 5.9 | 16 | 12% | 80 | 8% | 81.9 | **12.5x** |
| Qwen3 Q proj | 2048 | 4096 | 4.6 | 32 | 25% | 32 | 6% | 84.3 | **16.5x** |
| Qwen3 O proj | 4096 | 2048 | 4.7 | 16 | 12% | 64 | 5% | 96.1 | **18.3x** |
| Qwen3 MoE gate/up | 2048 | 512 | 0.7 | 4 | 3% | 32 | 1% | 76.2 | **99.7x** |
| Qwen3 MoE down | 512 | 2048 | 0.6 | 16 | 12% | 8 | 1% | 78.9 | **120.4x** |
| Qwen3 shared expert | 2048 | 512 | 0.7 | 4 | 3% | 32 | 1% | 70.6 | **92.3x** |
| GLM4.7 shared gate/up | 2048 | 10240 | 11.3 | 80 | 62% | 32 | 17% | 72.7 | **5.8x** |
| GLM4.7 shared down | 10240 | 2048 | 11.8 | 16 | 12% | 160 | 15% | 88.6 | **6.8x** |
| GLM4.7 routed gate/up | 2048 | 1536 | 1.8 | 12 | 9% | 32 | 2% | 108.4 | **54.1x** |
| GLM4.7 routed down | 1536 | 2048 | 1.8 | 16 | 12% | 24 | 2% | 107.7 | **54.8x** |
| Llama3-8B gate/up | 4096 | 14336 | 31.5 | 112 | 88% | 64 | 42% | 82.5 | 2.4x |
| Llama3-70B gate/up | 8192 | 28672 | 125.3 | 224 | 100% | 128 | 60% | 230.4 | 1.7x |

**Key finding: the kernel has a ~70-90us fixed floor regardless of problem
size.** MoE expert shapes (0.6-0.7 MB of data) take 70-80us when the
bandwidth floor is < 1us. This means 99% of kernel time is overhead for
these shapes.

### 3.2 Overhead breakdown

The overhead comes from three sources, in order of impact:

**1. SM underutilization (dominant for small N)**

For Qwen3 MoE gate/up (N=512): only 4 out of 128 SMs are active (3%).
The aggregate memory bandwidth is proportionally reduced: ~3% of 900 GB/s
= 28 GB/s effective. Even with perfect compute efficiency, the kernel
cannot run fast when 97% of the GPU is idle.

For Qwen3 dense gate/up (N=5120): 40 tiles = 31% SM utilization.
For GLM4.7 shared gate/up (N=10240): 80 tiles = 62% utilization.

**2. Per-tile pipeline overhead (dominant for small K_dim)**

Each k_tile iteration incurs:
- 2x `__syncthreads()` barriers (~25-50 cycles each)
- cp_async_wait stall
- Pipeline loop control (branch, address calculation)

With K_dim=2048 (32 k_tiles), these overheads repeat 32 times. With
K_dim=8192 (128 k_tiles), they repeat 128 times but the per-tile compute
also increases. The ratio of overhead to useful work is worse for small
K_dim because there are fewer FLOPs per tile to amortize the fixed barrier
costs.

**3. Dequant compute density (always present)**

Per TILE_K iteration: ~472 ALU + 32 shuffles + 32 shmem loads +
8*M_BLOCKS MMAs. The dequant ALU is serialized with MMA, creating a
dependency chain that limits instruction-level parallelism.

### 3.3 Why MoE shapes are fundamentally harder

MoE models have a structural mismatch with fused GEMM kernels:

1. **Small hidden_size (2048)**: K_dim=2048 gives only 32 k_tile iterations.
   The cp.async pipeline has ~2 tiles of fill/drain overhead = 6% wasted
   iterations. More importantly, there are not enough iterations to
   achieve steady-state pipeline throughput.

2. **Small N per expert (512-1536)**: With TILE_N=128, N=512 gives only
   4 tiles. On 128 SMs, 97% of the GPU sits idle.

3. **Small M per expert**: With 512 experts and 10 per token for a batch
   of 32 tokens, each expert sees ~0.6 tokens on average. Most experts
   see 0-2 tokens. This means M=1-4 per expert GEMM.

4. **cuBLAS is well-optimized for small GEMMs**: cuBLAS handles these
   shapes in 22-43us. It uses fundamentally different strategies for
   small problems (warp-level GEMMs, different tile sizes, no pipeline).

### 3.4 The ~70us kernel floor

The kernel takes 70-90us for ALL small shapes, even when the actual
computation is trivial. This floor comes from:

- Kernel launch overhead: ~5-10us
- Shared memory allocation + pipeline initialization: ~5us
- First cp.async group issue + wait: ~10-15us (global memory latency)
- 32 k_tile iterations x barrier overhead: ~10-20us
- Dequant compute: ~10-20us (even at 3% SM utilization)
- Output write: ~5us

For shapes where cuBLAS completes in 22-37us, our 70us floor makes us
2-3x slower regardless of any compute optimization.

### 3.5 Comparison: where the kernel architecture works vs. fails

| Regime | Example | SM% | k_tiles | Overhead | Verdict |
|--------|---------|----:|--------:|---------:|---------|
| Large K_dim, large N | Llama3-70B gate/up | 100% | 128 | 1.7x | **Wins 2.2x** |
| Large K_dim, moderate N | Llama3-8B gate/up | 88% | 64 | 2.4x | **Wins 1.7x** |
| Small K_dim, large N | GLM4.7 shared gate/up | 62% | 32 | 5.8x | Loses 0.36x |
| Small K_dim, moderate N | Qwen3 dense gate/up | 31% | 32 | 14.3x | Loses 0.41x |
| Small K_dim, small N | Qwen3 MoE expert | 3% | 32 | 99.7x | Loses 0.40x |
| Tiny K_dim, moderate N | Qwen3 MoE down | 12% | 8 | 120.4x | Loses 0.29x |

**The kernel needs BOTH high SM utilization (n_tiles >= 64) AND enough
k_tiles (>= 64) to be competitive.** This means K_dim >= 4096 and
N >= 8192 in the current configuration.

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

### 5.1 Priority reassessment

The previous roadmap focused on TILE_N=256 + TILE_K=128 to improve Llama
gate/up projections. However, the primary targets are now MoE models
(Qwen3-Coder-Next, GLM-4.7-Flash), where the kernel loses across ALL
shapes.

**The fundamental problem for MoE shapes is not dequant compute efficiency
(which TILE_N=256 addresses) — it is the per-tile overhead and SM
underutilization.** The planned TILE_N=256 + TILE_K=128 optimization would
make MoE shapes WORSE:

- TILE_N=256 halves n_tiles (already critically low for MoE)
- TILE_K=128 halves k_tiles (already critically low at K_dim=2048)

Both changes increase per-tile overhead relative to useful work, which is
the exact opposite of what MoE shapes need.

**TILE_N=256 + TILE_K=128 is still valuable for Llama-scale dense models**
but should NOT be the highest priority.

### Step 1: Reduce the fixed overhead floor (HIGHEST PRIORITY)

**Problem:** The kernel takes 70-90us regardless of problem size. For MoE
shapes where cuBLAS completes in 22-37us, no amount of compute
optimization can overcome a 70us floor.

**Approach: lightweight kernel variant for small problems.**

Design a second kernel path (not a replacement — an additional dispatch
option) optimized for low latency rather than high throughput:

- **No cp.async pipeline**: Use synchronous global loads directly to
  registers, then store to shared memory. Eliminates pipeline fill/drain
  overhead and the cp_async_fence/wait machinery. For small K_dim (16-32
  k_tiles), the pipeline's latency-hiding benefit is minimal because there
  are not enough iterations to reach steady state.

- **Smaller thread block (128 threads = 4 warps)**: Reduces per-barrier
  synchronization cost. With 4 warps, `__syncthreads()` is faster (fewer
  warps to synchronize). Also reduces shared memory pressure.

- **Single-stage shared memory**: No double buffering. Load a tile, sync,
  compute, repeat. Simpler control flow = less overhead per iteration.

- **Tuned tile sizes for small shapes**: TILE_N=64 (to increase n_tiles
  for small N), TILE_K=32 (to reduce per-tile data and allow more
  k_tiles for small K_dim).

**Dispatch logic:** Use the lightweight kernel when `K_dim * N < threshold`
(e.g., when the problem is small enough that the overhead dominates).
Use the full production kernel for large problems.

**Target:** Reduce the small-shape floor from 70-90us to 20-30us. If
achieved, MoE shapes would go from 0.3-0.4x to 0.8-1.2x.

**SM utilization concern:** Even with TILE_N=64, N=512 gives only 8 tiles
= 6% SM utilization. For the Qwen3 MoE expert case (N=512, K_dim=2048),
getting below cuBLAS's 30us is extremely challenging with a single-expert
kernel. This may ultimately require grouped/batched expert execution at
the framework level (step 4).

### Step 2: k_splits tuning for moderate shapes (HIGH)

**Problem:** Shapes like GLM4.7 shared gate/up (K=2048, N=10240,
80 tiles, 62% SM util) have decent N but still lose at 0.36x. The overhead
multiplier is 5.8x — better than the tiny shapes but still poor.

**Approach:** For shapes where mn_tiles < num_sms but k_tiles is large
enough to split, enable k_splits to fill more SMs. The current threshold
(`mn_tiles < num_sms / 4` = 32) is too conservative for this regime.

Specifically, for GLM4.7 shared gate/up: 80 mn_tiles, 32 k_tiles. With
k_splits=2, total_work=160, filling all 128 SMs. Each split handles 16
k_tiles. The atomicAdd overhead may be worth the SM fill for this shape.

**Tuning needed:** Benchmark k_splits=2 for shapes in the 32-128 mn_tiles
range with K_dim=2048-5120. Determine the crossover point where k_splits
helps vs. hurts.

**Expected impact:** GLM4.7 shared gate/up: 0.36x → possibly 0.5-0.7x.
Still won't beat cuBLAS but narrows the gap.

### Step 3: TILE_N=256 + TILE_K=128 for large shapes (HIGH)

**This is the original Phase 1 plan, preserved for Llama-scale models.**

Implement with shape-adaptive dispatch:
- TILE_N=256 only when N >= 10240 AND K_dim >= 4096
- TILE_K=128 only when K_dim >= 4096 AND K_dim % 128 == 0
- Keep TILE_N=128 / TILE_K=64 for all other shapes

**Expected impact for Llama:**
- Llama3-70B gate/up M=32: 2.22x → 2.5-3.0x
- Llama3-8B gate/up M=32: 1.68x → 2.0-2.5x

**No impact on MoE shapes** (they use the TILE_N=128/TILE_K=64 path or
the lightweight kernel).

**Shared memory budget (2 stages, TILE_N=256, TILE_K=128):**

| M_BLOCKS | K | A stage | B stage | Absmax | Total/stage | 2 stages |
|---------:|--:|--------:|--------:|-------:|------------:|---------:|
| 1 | 4 | 4 KB | 16 KB | 1 KB | 21 KB | 42 KB |
| 2 | 4 | 8 KB | 16 KB | 1 KB | 25 KB | 50 KB |
| 4 | 4 | 16 KB | 16 KB | 1 KB | 33 KB | 66 KB |
| 4 | 5 | 16 KB | 20 KB | 1 KB | 37 KB | 74 KB |

All fit within RTX 4090's 100 KB dynamic shmem limit.

### Step 4: Grouped expert GEMM for MoE (MEDIUM-HIGH)

**Problem:** Even with the lightweight kernel, individual MoE expert GEMMs
(N=512, M=1-4) cannot efficiently use the GPU. Only 4-8 tiles on 128 SMs.

**Approach:** Instead of dispatching one kernel per expert, batch all
active experts into a single kernel launch:

- All experts share the same K_dim and N dimensions
- The kernel processes multiple experts in one launch, with each
  thread block handling a different (expert_id, tile) combination
- Input: gathered activation matrix A_gathered[total_tokens, K_dim] +
  expert_ids[total_tokens] + all expert weights
- The grid is total_active_experts * tiles_per_expert

With 32 tokens x 10 experts = 320 expert-invocations, and 4 tiles per
expert (N=512), that is 1280 tiles — filling all 128 SMs 10x over.

**This is an API-level change** (new op signature, new repack format for
batched weights) but reuses the same inner loop. The key insight is that
the dequant + MMA core is already efficient — the problem is launch
overhead and SM underutilization, both of which batching solves.

**Expected impact:** MoE expert shapes could go from 0.3-0.4x (per expert)
to 1.5-2.5x (batched), because the total data read is still K_BITS/16
of cuBLAS and the overhead is amortized over hundreds of tiles.

### Step 5: Inner loop optimization (MEDIUM)

**B fragment register double-buffering:** Preload next N-block's B planes
while current MMA executes. Hides 20-30 cycle shmem load latency.
Expected: 10-20% improvement on all shapes.

**C output staging via shmem:** Coalesced output writes instead of
scattered fragment writes. Expected: 5-15% improvement.

These apply to both the production kernel and the lightweight kernel.

### Step 6: Warp specialization (FUTURE)

Dedicated producer/consumer warps. Only if Steps 1-5 are insufficient.

---

## 6. Implementation Order

### Phase 1: Lightweight kernel for small shapes (target: MoE models)

1. Design lightweight kernel variant with synchronous loads, smaller
   thread block, single-stage shmem
2. Implement with TILE_N=64, TILE_K=32, 128 threads
3. Dispatch: use lightweight kernel when K_dim <= 2048 OR N <= 2048
4. Benchmark Qwen3 and GLM4.7 shapes
5. Tune k_splits threshold for moderate shapes (mn_tiles 32-128)
6. Benchmark GLM4.7 shared gate/up with k_splits=2

### Phase 2: TILE_N=256 + TILE_K=128 (target: Llama-scale models)

7. Add TILE_N/TILE_K as template parameters
8. Shape-adaptive dispatch: large tiles only for K_dim >= 4096 AND N >= 10240
9. Benchmark Llama shapes
10. B fragment register double-buffering
11. C output staging

### Phase 3: Grouped expert GEMM (target: MoE per-expert layers)

12. Design grouped expert API and repack format
13. Implement grouped kernel launch
14. Benchmark Qwen3 MoE and GLM4.7 routed expert shapes
15. Compare against per-expert cuBLAS

### Phase 4: Integration

16. Wire into LinearNbit module
17. Remove staging kernels (keep production + lightweight + grouped)
18. Lint and PR to main

---

## 7. Target Performance

For M=32, K=4:

### MoE models (after Phase 1 lightweight kernel):

| Layer | Current | Phase 1 target | Theoretical max |
|-------|:-------:|:--------------:|:---------------:|
| Qwen3 dense gate/up (K=2048, N=5120) | 0.41x | 0.7-1.0x | ~4x |
| Qwen3 O proj (K=4096, N=2048) | 0.45x | 0.6-0.9x | ~4x |
| GLM4.7 shared gate/up (K=2048, N=10240) | 0.36x | 0.6-0.9x | ~4x |
| GLM4.7 routed gate/up (K=2048, N=1536) | 0.39x | 0.5-0.7x | ~4x |
| Qwen3 MoE gate/up (K=2048, N=512) | 0.40x | 0.4-0.6x | ~4x |

### MoE routed experts (after Phase 3 grouped GEMM):

| Layer | Current | Phase 3 target | Theoretical max |
|-------|:-------:|:--------------:|:---------------:|
| Qwen3 MoE gate/up (K=2048, N=512) | 0.40x | 1.5-2.5x | ~4x |
| GLM4.7 routed gate/up (K=2048, N=1536) | 0.39x | 1.5-2.5x | ~4x |

### Dense Llama-style models (after Phase 2):

| Layer | Current | Phase 2 target | Theoretical max |
|-------|:-------:|:--------------:|:---------------:|
| Llama3-70B gate/up | 2.22x | 2.5-3.0x | ~4x |
| Llama3-8B gate/up | 1.68x | 2.0-2.5x | ~4x |
| Llama3-70B down | 0.99x | ~1.0x | ~4x |
| Llama3-8B down | 0.55x | ~0.55x | ~4x |

### Honest assessment

- **Phase 1 (lightweight kernel) is unlikely to fully close the gap for
  MoE shapes.** Even with 2x overhead reduction, going from 0.3-0.4x to
  0.6-0.8x still loses to cuBLAS. The SM utilization problem is structural
  for small N.

- **Phase 3 (grouped expert GEMM) is where the real MoE win is.** Batching
  hundreds of expert invocations into one kernel eliminates both the launch
  overhead and SM underutilization problems. This is how production MoE
  inference frameworks (vLLM, SGLang) handle expert execution.

- **Phase 2 (TILE_N=256) is high confidence for Llama models.** The
  analysis is well understood and the implementation was previously validated.

---

## 8. Model Shape Reference

### Qwen3-Coder-Next (MoE, 70B+, hidden=2048)

Primary optimization target. Key dimensions:

- hidden_size: 2048
- intermediate_size: 5120 (dense FFN)
- moe_intermediate_size: 512 (per-expert)
- shared_expert_intermediate_size: 512
- num_experts: 512, num_experts_per_tok: 10
- num_attention_heads: 16, num_key_value_heads: 2, head_dim: 256
- 48 layers

GEMM shapes:
- Dense gate/up: K=2048, N=5120
- Dense down: K=5120, N=2048
- Q proj: K=2048, N=4096 (16 heads x 256)
- KV proj: K=2048, N=512 (2 heads x 256)
- O proj: K=4096, N=2048
- MoE gate/up: K=2048, N=512
- MoE down: K=512, N=2048

### GLM-4.7-Flash (MoE, hidden=2048)

- Shared expert: K=2048, N=10240
- Routed expert: K=2048, N=1536 (64 experts, top-4)
- Attention: MLA with q_lora_rank=768, kv_lora_rank=512

### Llama-style models

| Model | hidden | gate/up (N) | down (N) |
|-------|-------:|------------:|---------:|
| Llama 2 7B | 4096 | 11008 | 4096 |
| Llama 3 8B | 4096 | 14336 | 4096 |
| Llama 3 70B | 8192 | 28672 | 8192 |
| Mistral 7B | 4096 | 14336 | 4096 |
| Qwen2.5 7B | 3584 | 18944 | 3584 |

---

## 9. Lessons Learned

1. **Benchmark on the actual target models.** The kernel was designed and
   optimized for Llama-scale dense shapes. MoE models have fundamentally
   different GEMM dimensions that expose the kernel's weaknesses.

2. **Fixed overhead dominates for small problems.** The kernel has a ~70us
   floor from launch + pipeline + barriers. For MoE expert shapes where
   cuBLAS takes 22-37us, no amount of compute optimization can compensate.

3. **SM utilization is the primary bottleneck for small N.** With N=512
   and TILE_N=128, only 4 out of 128 SMs are active. The GPU is 97% idle.

4. **K_dim must be large (>= 4096) for the pipeline to be effective.**
   With K_dim=2048, there are only 32 k_tile iterations — not enough to
   amortize pipeline overhead.

5. **MoE expert GEMMs need batching, not per-expert optimization.** A
   single expert's GEMM is too small to efficiently utilize the GPU.
   Grouped execution is the correct architectural approach.

6. **TILE_N=256 and TILE_K=128 help the wrong shapes.** They improve
   large-K_dim, large-N shapes (Llama gate/up) but make small shapes
   worse by reducing tiles. Shape-adaptive dispatch is essential, and
   the MoE shapes need the opposite optimization direction (smaller tiles,
   less overhead).

7. **cuBLAS is well-optimized for small GEMMs.** It uses fundamentally
   different strategies for small problems. Beating cuBLAS at its own
   game (small GEMMs) is much harder than beating it at large
   bandwidth-bound GEMMs.

8. **The kernel's value proposition is different per model class:**
   - Dense 70B+ models: significant win on gate/up (2.2x), marginal overall
   - Dense 7-8B models: modest win on gate/up (1.7x), break-even overall
   - MoE models: no win without grouped expert execution
