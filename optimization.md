# kbit GEMM Kernel: Optimization Guide

RTX 4090 (128 SMs, sm_89), clocks locked at 2520 MHz, 300 iters, K=4,
fp16, M=32 unless stated otherwise.

---

## 1. The Fundamental Opportunity

We read **3.6x less data** than cuBLAS. If our per-byte execution overhead
matched cuBLAS, we would achieve **3.5-3.7x speedup on every shape**:

| Layer | kbit data | cuBLAS data | cuBLAS ovhd | If kbit same ovhd | vs cuBLAS |
|-------|----------:|------------:|------------:|---------:|------:|
| Qwen3 dense gate/up (2048x5120) | 5.7 MB | 21.1 MB | 1.6x | 10.2 us | **3.7x** |
| Qwen3 dense down (5120x2048) | 5.9 MB | 21.3 MB | 1.6x | 10.5 us | **3.6x** |
| GLM4.7 shared gate/up (2048x10240) | 11.3 MB | 42.1 MB | 0.6x | 6.9 us | **3.7x** |
| GLM4.7 shared down (10240x2048) | 11.8 MB | 42.6 MB | 0.6x | 7.5 us | **3.6x** |
| GLM4.7 routed gate/up (2048x1536) | 1.8 MB | 6.4 MB | 5.9x | 11.8 us | **3.6x** |
| Llama3-8B gate/up (4096x14336) | 31.5 MB | 117.7 MB | 1.1x | 37.1 us | **3.7x** |
| Llama3-70B gate/up (8192x28672) | 125.3 MB | 470.3 MB | 1.0x | 136.2 us | **3.8x** |

**We are not data-limited. We are overhead-limited.** The compression
advantage is real and consistent. The entire optimization problem is
reducing per-byte overhead to match cuBLAS.

---

## 2. Current Performance and the Overhead Gap

| Layer | kbit (us) | cuBLAS (us) | Speedup | kbit ovhd | cuBLAS ovhd | Gap |
|-------|----------:|------------:|--------:|----------:|------------:|----:|
| Qwen3 dense gate/up | 90.6 | 37.6 | 0.41x | 14.3x | 1.6x | 8.9x |
| Qwen3 dense down | 81.9 | 37.9 | 0.46x | 12.5x | 1.6x | 7.8x |
| GLM4.7 shared gate/up | 72.7 | 25.9 | 0.36x | 5.8x | 0.6x | 10.4x |
| GLM4.7 shared down | 88.6 | 27.1 | 0.31x | 6.8x | 0.6x | 12.2x |
| GLM4.7 routed gate/up | 108.4 | 42.2 | 0.39x | 54.1x | 5.9x | 9.2x |
| Qwen3 MoE gate/up | 76.2 | 30.8 | 0.40x | 99.7x | ~40x | 2.5x |
| Llama3-8B gate/up | 82.5 | 138.8 | **1.68x** | 2.4x | 1.1x | 2.2x |
| Llama3-70B gate/up | 230.4 | 511.1 | **2.22x** | 1.7x | 1.0x | 1.7x |

"Overhead" = actual time / (data_read / 900 GB/s). "Gap" = our overhead /
cuBLAS overhead. The gap shows how many x we need to improve.

For Llama 70B, the gap is only 1.7x — our overhead is close to cuBLAS.
For Qwen3/GLM4.7 shapes, the gap is 8-12x — we have massive overhead.

**Note:** cuBLAS achieves <1x overhead on some MoE shapes because the
weight data fits in L2 cache (72 MB on RTX 4090). All Qwen3 and GLM4.7
weights fit in L2. Our compressed data also fits in L2, so we have the
same caching advantage — we just aren't exploiting it due to overhead.

---

## 3. Where the Overhead Comes From

### 3.1 SM underutilization (biggest factor for medium N)

| Shape | n_tiles (TILE_N=128) | SM utilization |
|-------|---------------------:|---------------:|
| Qwen3 MoE gate/up (N=512) | 4 | 3% |
| GLM4.7 routed gate/up (N=1536) | 12 | 9% |
| Qwen3 dense (N=2048) | 16 | 12% |
| Qwen3 Q proj (N=4096) | 32 | 25% |
| Qwen3 dense gate/up (N=5120) | 40 | 31% |
| GLM4.7 shared gate/up (N=10240) | 80 | 62% |
| Llama3-8B gate/up (N=14336) | 112 | 88% |
| Llama3-70B gate/up (N=28672) | 224 | 100% |

With M=32 (m_tiles=1), the grid size equals n_tiles. On 128 SMs,
anything below 128 tiles means idle SMs. Idle SMs = wasted memory
bandwidth capacity.

**k_splits can fix this.** With k_splits=2, GLM4.7 shared gate/up goes
from 80 tiles to 160 total work items, filling all 128 SMs. The
atomicAdd overhead is small (~0.5 us) compared to the bandwidth gain
from activating 48 more SMs.

The current threshold (`mn_tiles < num_sms / 4 = 32`) is too
conservative — it never activates k_splits for these shapes.

### 3.2 Short pipeline (K_dim = 2048)

With K_dim=2048 and TILE_K=64: only 32 k_tile iterations. The 2-stage
pipeline has 1 tile of fill/drain overhead = 3% waste. But worse, with
only 2 stages in flight, there is minimal slack for variable memory
latency. If one load takes longer than expected, the pipeline stalls.

With k_splits=2 and 16 k_tiles per split, the pipeline is even shorter.
A 3-stage pipeline (instead of 2) provides 2x more latency slack at
the cost of 1 more prefill iteration.

### 3.3 Dequant compute cost

Per weight element: ~13 ALU ops (bit extract + shuffle codebook + scale).
cuBLAS does 0 ops per weight element (just feeds fp16 to MMA). This is
inherent and cannot be eliminated — it is the price of compression.

But the dequant runs on INT32/FP16 ALU while MMA runs on tensor cores.
They are different functional units. With proper scheduling (B fragment
double-buffering, deeper pipeline), the dequant can overlap with MMA
and memory loads. Currently the dequant is on the critical path because
the inner loop is sequential: load B → dequant → MMA → next N-block.

---

## 4. Optimization Plan

### Step 1: Aggressive k_splits for K_dim <= 4096 shapes (HIGHEST PRIORITY)

**What:** Lower the k_splits threshold so that shapes with moderate SM
utilization (31-88%) get k_splits to fill all SMs.

**Code change:** In `kbitGemmProdLaunch` (ops.cu line 2067):
```cpp
// OLD: only split when severely underutilized (< 25%)
if (mn_tiles < num_sms / 4 && k_tiles > 1)

// NEW: split when any SM would be idle, but cap conservatively
if (mn_tiles < num_sms && k_tiles > 1) {
    k_splits = min(k_tiles, (num_sms + mn_tiles - 1) / mn_tiles);
    // But cap at a reasonable value to limit atomicAdd overhead
    k_splits = min(k_splits, 4);
}
```

**Expected SM utilization change:**

| Shape | mn_tiles | Current k_splits | New k_splits | New total | New SM% |
|-------|--------:|--------:|--------:|--------:|--------:|
| Qwen3 dense gate/up (N=5120) | 40 | 1 | 4 | 160 | 100% |
| GLM4.7 shared gate/up (N=10240) | 80 | 1 | 2 | 160 | 100% |
| GLM4.7 shared down (N=2048) | 16 | 1 | 4 | 64 | 50% |
| Qwen3 Q proj (N=4096) | 32 | 1 | 4 | 128 | 100% |
| Qwen3 O proj (N=2048) | 16 | 1 | 4 | 64 | 50% |
| Llama3-8B gate/up (N=14336) | 112 | 1 | 1 | 112 | 88% |

**Expected impact:** For shapes currently at 31-62% SM utilization,
k_splits brings them to 100%. This should roughly double effective
bandwidth, cutting kernel time in half. Combined with the 3.6x data
compression advantage:

- GLM4.7 shared gate/up: 72.7us → ~35us → **0.74x** (from 0.36x)
- Qwen3 dense gate/up: 90.6us → ~45us → **0.83x** (from 0.41x)

These are conservative estimates. If the bandwidth gain from filling
all SMs is superlinear (L2 cache becomes more effective with more SMs
issuing requests), the improvement could be larger.

**Risk:** k_splits adds atomicAdd + workspace overhead. Per split:
each thread does ~16 atomicAdd fp32 operations at ~50 cycles each =
0.3us per work item. Plus threadfence (~0.1us) and tile_counter
increment. Total: ~0.5us per k_split contribution. For k_splits=4,
that's ~2us total overhead. Small relative to the 30-60us gain from
better SM utilization.

**Must benchmark:** The crossover point where k_splits overhead exceeds
the SM fill benefit. Start with k_splits capped at 4 and tune down if
atomicAdd contention is worse than expected.

### Step 2: 3-stage pipeline (HIGH)

**What:** Increase pipeline depth from 2 to 3 stages.

**Why:** With k_splits=2-4 and K_dim=2048, each split processes only
8-16 k_tiles. The 2-stage pipeline has minimal latency slack — if one
global load takes longer than the compute for one tile, the pipeline
stalls. 3 stages provide 2x more slack.

**Code change:** In `kbit_gemm_prod`:
```cpp
// 3-stage pipeline
// Shmem: 3 * STAGE_BYTES instead of 2 * STAGE_BYTES
// Prefill 2 stages, then enter loop with cp_async_wait<1>()
fetch_tile(0, kt_start); cp_async_fence();
if (kt_start + 1 < kt_end) {
    fetch_tile(1, kt_start + 1); cp_async_fence();
}

for (int kt = kt_start; kt < kt_end; kt++) {
    int cur = (kt - kt_start) % 3;
    cp_async_wait<1>();
    __syncthreads();
    if (kt + 2 < kt_end) {
        fetch_tile((kt + 2 - kt_start) % 3, kt + 2);
        cp_async_fence();
    }
    compute_tile(cur);
    __syncthreads();
}
cp_async_wait<0>();
```

**Shmem budget (3 stages):**

| M_BLOCKS | K | Per stage | 3 stages | Fits 100 KB? |
|---------:|--:|----------:|---------:|:-------------|
| 1 | 4 | 4.3 KB | 12.9 KB | YES |
| 2 | 4 | 8.4 KB | 25.3 KB | YES |
| 4 | 4 | 16.5 KB | 49.6 KB | YES |
| 4 | 5 | 20.6 KB | 61.9 KB | YES |

All variants fit with headroom.

**Expected impact:** 5-15% improvement on K_dim=2048 shapes by reducing
pipeline stalls. Larger impact when combined with k_splits (shorter
per-split pipeline benefits more from extra stage).

### Step 3: Profile the kernel (HIGH)

**What:** Run `ncu` (Nsight Compute) profiling on key shapes to identify
exactly where execution time is spent.

```bash
ncu --set full -o profile_qwen3 python bench_single_shape.py --K 2048 --N 5120 --M 32
```

**Key metrics to check:**
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — global load sectors
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` — tensor core utilization
- `sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active` — ALU utilization
- `sm__warps_issue_stalled_*` — stall reasons breakdown

**Why:** The performance model predicts ~20-25us for GLM4.7 shared
gate/up, but we measure 72.7us. There is a 3x unexplained gap.
Profiling will reveal whether the bottleneck is memory stalls,
compute stalls, barrier stalls, or something else entirely.

This informs whether further optimization should focus on memory access
patterns, compute scheduling, or pipeline structure.

### Step 4: B fragment register double-buffering (HIGH)

**What:** Overlap shmem B loads with MMA execution in the inner loop.

Current inner loop (per k_sub_tile, per N_block):
```
load B planes from shmem  →  dequant  →  MMA  →  next N_block
     [stall]                  [ALU]      [TC]
```

With double-buffering:
```
preload B[nb+1] from shmem  →  dequant B[nb]  →  MMA  →  next
     [shmem load]                [ALU, overlap]    [TC]
```

The shmem loads for the next N_block's B planes (4 uint32 reads,
~20-30 cycle latency each) overlap with the current N_block's dequant
ALU work. This removes the shmem load stall from the critical path.

**Expected impact:** 10-20% improvement on all shapes. The dequant ALU
work (~50 cycles per N_block iteration) provides enough instructions to
hide the shmem load latency.

### Step 5: TILE_N=256 + TILE_K=128 for large shapes (HIGH)

For Llama-scale shapes (K_dim >= 4096, N >= 10240):

- TILE_N 128→256, N_BLOCKS 2→4: halves dequant-per-MMA ratio
- TILE_K 64→128: halves pipeline iterations and barrier count

Shape-adaptive dispatch: only use large tiles when K_dim >= 4096 AND
N >= 10240. MoE shapes continue using TILE_N=128 / TILE_K=64.

**Expected impact:** Llama3-70B gate/up: 2.22x → 2.5-3.0x. Llama3-8B
gate/up: 1.68x → 2.0-2.5x.

**Shmem budget (2 stages, TILE_N=256, TILE_K=128):**

| M_BLOCKS | K | Per stage | 2 stages | Fits? |
|---------:|--:|----------:|---------:|:------|
| 2 | 4 | 25 KB | 50 KB | YES |
| 4 | 5 | 37 KB | 74 KB | YES |

### Step 6: Grouped expert GEMM for MoE routed experts (MEDIUM-HIGH)

Individual MoE expert GEMMs (N=512, M=1-4) have only 4-8 tiles on
128 SMs. No kernel optimization can fix 3% SM utilization.

**Solution:** Batch all active experts into a single kernel launch.

With 32 tokens x 10 experts = 320 expert-invocations, 4 tiles per
expert: 1280 tiles → all 128 SMs fully utilized, 10x over.

**Design:**
- Input: A_gathered[total_tokens, K_dim] + expert_offsets + all expert
  weight pointers (or a single stacked weight tensor)
- Each thread block handles one (expert_id, n_tile) combination
- Inner loop is identical to production kernel
- Grid: num_active_experts * (N / TILE_N)

This reuses the entire existing inner loop. The change is in the
launcher and work distribution, not the MMA/dequant code.

**Expected impact:** MoE expert shapes: 0.3-0.4x → 1.5-2.5x (batched).

### Step 7: C output staging via shmem (MEDIUM)

Stage output through shmem for coalesced global writes instead of
scattered per-fragment writes. 5-15% improvement.

---

## 5. Implementation Order

### Phase 1: Quick wins (k_splits + pipeline)

1. Lower k_splits threshold: `mn_tiles < num_sms`, cap at 4
2. Benchmark Qwen3 + GLM4.7 shapes with new k_splits
3. Implement 3-stage pipeline
4. Benchmark again — measure combined impact
5. Profile with ncu to find remaining bottlenecks
6. Tune k_splits cap based on atomicAdd contention data

### Phase 2: Inner loop + large tiles

7. B fragment register double-buffering
8. TILE_N=256 + TILE_K=128 with shape-adaptive dispatch
9. C output staging
10. Benchmark all shapes across K=2-5

### Phase 3: Grouped expert GEMM

11. Design grouped expert kernel API
12. Implement batched work distribution
13. Benchmark Qwen3 MoE and GLM4.7 routed expert shapes

### Phase 4: Integration

14. Wire into LinearNbit module
15. Lint and PR

---

## 6. Performance Targets

For M=32, K=4:

### After Phase 1 (k_splits + 3-stage pipeline):

| Layer | Current | Target | How |
|-------|:-------:|:------:|-----|
| GLM4.7 shared gate/up (K=2048, N=10240) | 0.36x | **0.7-1.0x** | k_splits=2, all SMs active |
| Qwen3 dense gate/up (K=2048, N=5120) | 0.41x | **0.7-1.0x** | k_splits=4, all SMs active |
| Qwen3 Q proj (K=2048, N=4096) | 0.35x | **0.5-0.8x** | k_splits=4, all SMs active |
| GLM4.7 shared down (K=10240, N=2048) | 0.31x | **0.5-0.7x** | k_splits=4, 50% → SM |
| Llama3-8B gate/up (K=4096, N=14336) | 1.68x | **1.7x** | No change (already 88% SM) |
| Llama3-70B gate/up (K=8192, N=28672) | 2.22x | **2.2x** | No change (already 100% SM) |

### After Phase 2 (inner loop + large tiles):

| Layer | Phase 1 | Target | How |
|-------|:-------:|:------:|-----|
| GLM4.7 shared gate/up | 0.7-1.0x | **1.0-1.5x** | +B double-buf, +3-stage |
| Qwen3 dense gate/up | 0.7-1.0x | **0.9-1.3x** | +B double-buf |
| Llama3-70B gate/up | 2.2x | **2.5-3.0x** | TILE_N=256, TILE_K=128 |
| Llama3-8B gate/up | 1.7x | **2.0-2.5x** | TILE_N=256, TILE_K=128 |

### After Phase 3 (grouped expert GEMM):

| Layer | Current | Target |
|-------|:-------:|:------:|
| Qwen3 MoE gate/up (N=512, batched) | 0.40x | **1.5-2.5x** |
| GLM4.7 routed gate/up (N=1536, batched) | 0.39x | **1.5-2.5x** |

### Theoretical ceiling

3.5-3.7x on all shapes (set by data compression ratio). Achieving this
requires matching cuBLAS's per-byte overhead, which may not be fully
possible due to the inherent dequant compute cost. Realistic ceiling:
**2.5-3.0x** on shapes with good SM utilization.

---

## 7. Model Shape Reference

### Qwen3-Coder-Next (MoE, 70B+, hidden=2048)

Primary target. 512 experts, 10 per token, 48 layers.

| Layer type | K_dim | N | Weight (kbit) | Fits L2? |
|------------|------:|-----:|---------:|:---------|
| Dense gate/up | 2048 | 5120 | 5.2 MB | YES |
| Dense down | 5120 | 2048 | 5.2 MB | YES |
| Q proj | 2048 | 4096 | 4.2 MB | YES |
| KV proj | 2048 | 512 | 0.5 MB | YES |
| O proj | 4096 | 2048 | 4.2 MB | YES |
| MoE gate/up (per expert) | 2048 | 512 | 0.5 MB | YES |
| MoE down (per expert) | 512 | 2048 | 0.5 MB | YES |

### GLM-4.7-Flash (MoE, hidden=2048)

| Layer type | K_dim | N | Weight (kbit) | Fits L2? |
|------------|------:|-----:|---------:|:---------|
| Shared gate/up | 2048 | 10240 | 10.5 MB | YES |
| Shared down | 10240 | 2048 | 10.5 MB | YES |
| Routed gate/up | 2048 | 1536 | 1.6 MB | YES |
| Routed down | 1536 | 2048 | 1.6 MB | YES |

### Llama-style models

| Model | hidden | gate/up (N) | Weight (kbit) | Fits L2? |
|-------|-------:|------------:|----------:|:---------|
| Llama 3 8B | 4096 | 14336 | 29.4 MB | YES |
| Llama 3 70B | 8192 | 28672 | 117.4 MB | NO |
| Qwen2.5 7B | 3584 | 18944 | 34.0 MB | YES |

Note: for Llama3-8B, kbit data (29 MB) fits in L2 but cuBLAS data
(117 MB) does not. This is a structural advantage for kbit — we
get L2 bandwidth (~2 TB/s) while cuBLAS must use DRAM (~900 GB/s).

---

## 8. Key Insights

1. **The kernel's advantage (3.6x less data) is real and consistent.**
   If overhead matched cuBLAS, we'd win 3.5-3.7x on every shape.
   The problem is purely execution overhead.

2. **SM utilization is the single biggest overhead source for MoE
   shapes.** GLM4.7 shared gate/up has 62% SM util; Qwen3 dense
   gate/up has 31%. k_splits can fix this immediately.

3. **The current k_splits threshold is too conservative.** It was set
   to `num_sms / 4` to avoid atomicAdd overhead, but the SM
   utilization gain far outweighs the atomicAdd cost for shapes in
   the 31-88% utilization range.

4. **All MoE weight data fits in L2 cache (72 MB).** This means
   effective bandwidth is potentially 2+ TB/s, not 900 GB/s. The
   kernel should benefit from this, but only if enough SMs are active
   to generate sufficient L2 requests.

5. **Individual MoE expert GEMMs (N=512) need batching.** No
   per-kernel optimization can fix 3% SM utilization. Grouped
   execution is the architectural solution.

6. **TILE_N=256 is still important for Llama shapes** but should not
   be the first priority. k_splits tuning has higher expected impact
   on MoE shapes and requires minimal code change.

7. **Profile before over-engineering.** The 3x unexplained gap
   between theoretical estimates and measured time suggests there
   may be a simple bottleneck (L2 thrashing, bank conflicts, stall
   pattern) that profiling would reveal immediately.
