# kbit GEMM Kernel: Optimization Phase 2

RTX 4090 (128 SMs, sm_89), K=4, fp16, M=32 unless stated otherwise.

**Target models:** Qwen3-Coder-Next (MoE, 70B+, hidden=2048) and
GLM-4.7-Flash (MoE, hidden=2048). These are MoE models where
individual expert GEMMs have small N (512-1536), producing few tiles
on 128 SMs. Llama-scale dense shapes already achieve ~2x over cuBLAS
and are not a priority.

---

## 1. Phase 1 Summary

Three changes were made to the production kernel (`kbit_gemm_prod`):

1. **Two-tier k_splits heuristic.** Tier 1 (unchanged): aggressive
   split-K for severe SM underutilization (< 25%). Tier 2 (new):
   conservative split-K (cap 2) when data exceeds L2 cache (> 24 MB)
   and SM utilization is moderate. Impact: Llama3-8B improved ~25%
   (115us to 87us). MoE shapes unaffected.

2. **Branchless absmax decode.** New `decode_e4m4_absmax_branchless()`
   eliminates two conditional branches that generate BSSY/BSYNC
   divergence-handling pairs in SASS. Subnormals (absmax < 2^-10)
   treated as normal path.

3. **Interleaved bit extraction.** All 4 fragment elements' bit
   extractions interleaved in a single loop over K_BITS, giving the
   compiler more ILP across elements and bit-planes.

All 195 tests pass. Correctness verified up to Llama3-70B shape
(8192x28672), max relative error < 0.08%.

### Phase 1 performance (M=32, K=4)

| Layer | kbit (us) | cuBLAS (us) | Speedup |
|-------|----------:|------------:|--------:|
| Qwen3 dense gate/up (2048x5120) | 68 | 22 | 0.32x |
| Qwen3 dense down (5120x2048) | 71 | 26 | 0.37x |
| GLM4.7 shared gate/up (2048x10240) | 73 | 27 | 0.37x |
| GLM4.7 shared down (10240x2048) | 74 | 29 | 0.39x |
| GLM4.7 routed gate/up (2048x1536) | 78 | 28 | 0.36x |
| Llama3-8B gate/up (4096x14336) | 87 | 135 | 1.54x |
| Llama3-70B gate/up (8192x28672) | 230 | 596 | 2.59x |

**Phase 1 conclusion:** marginal changes to the inner loop cannot fix
the MoE shapes. The problem is structural.

---

## 2. Root Cause: The Kernel Is Instruction-Limited

### 2.1 The numbers

The kernel reads **3.6x less data** than cuBLAS. If per-byte overhead
matched cuBLAS, every shape would achieve 3.5-3.7x speedup. Instead
MoE shapes run at 0.3-0.4x. The overhead is not bandwidth — it is
instruction count.

For Qwen3 gate/up (K=2048, N=5120):
- kbit data: 5.6 MB. L2 transfer at 2 TB/s: **2.8 us**
- Measured kernel time: **68 us**
- Overhead ratio: **24x**

The kernel spends 24x longer than it would take to simply read the
data from L2. For GLM4.7 shapes the ratio is 13-24x. For Llama3-70B
(DRAM-bound, fully SM-utilized) the ratio is 1.6x — close to
cuBLAS.

### 2.2 SASS instruction breakdown

The compiled kernel has ~1264 SASS instructions per k_tile iteration
(M_BLOCKS=2, K=4, fp16). Per k_tile the inner loop is fully unrolled
across 4 k_sub * 2 N_BLOCKS = 8 pairs:

| Category | Count | % | What |
|----------|------:|---:|------|
| Bit extraction (SHF+LOP3+IMAD) | ~512 | 40% | 4 elements * 4 bits * 4 ops * 8 pairs |
| A fragment load (addr+ldmatrix) | ~160 | 13% | Swizzle address math + 2 ldmatrix, x8 |
| Fetch + barriers + loop | ~160 | 13% | cp.async issue, __syncthreads, kt loop |
| Absmax decode + convert | ~64 | 5% | shmem load + decode + f2h, x8 |
| B plane shmem load | ~56 | 4% | 4 loads + addr, x8 |
| Codebook shuffle (SHFL) | ~32 | 3% | 4 shuffles, x8 |
| Scale multiply (HMUL) | ~32 | 3% | 4 hmul, x8 |
| Pack + MMA | ~48 | 4% | 2 pack + 2 MMA, x8 |
| Other (misc addr, control) | ~200 | 16% | |
| **Total** | **~1264** | | |

**Tensor core MMA: 16 instructions = 1.3%.** The tensor cores are
idle 98.7% of the time. The kernel is an ALU program that
occasionally does a matrix multiply.

### 2.3 Cycle budget

At 32 k_tiles per block:
- Dynamic instruction count: ~40,000 per thread
- With 2 warps per scheduler (occupancy = 8/48 = 16.7%): ~80,000
  cycles of execution per scheduler
- At 2.52 GHz: ~32 us of pure instruction execution
- Add memory stalls (cp.async wait, shmem latency) and barrier
  stalls (__syncthreads with 8 warps): ~35 us
- Total: ~67 us. Matches measurement of 68-78 us.

### 2.4 Why k_splits cannot help MoE shapes

All Qwen3 and GLM4.7 weight data fits in L2 cache (72 MB on 4090).
Effective bandwidth is ~2 TB/s from L2, not ~900 GB/s from DRAM. With
data already in L2, adding more SMs via k_splits does not increase
bandwidth — it only adds atomicAdd overhead.

Benchmarking confirmed this: enabling k_splits=4 for Qwen3 gate/up
(31% SM util to 100% SM util) changed kernel time from 72 us to 71 us
(within noise).

### 2.5 Why inner loop tweaks have diminishing returns

The interleaved bit extraction and branchless absmax reduced
instruction count by an estimated 5-10%. But 5-10% of 1264 is ~60-120
fewer instructions per k_tile. At 32 k_tiles: ~2000-4000 fewer
dynamic instructions. Time saved: ~2-4 us out of 68 us. Below the
5-10% benchmark noise.

To get a meaningful speedup, we need to remove **hundreds** of
instructions per k_tile, not tens.

### 2.6 Additional finding: B-tile bank conflicts for K=4

The B-tile shared memory layout uses stride = 2*K = 8 words per
column. For K=4: gcd(8, 32) = 8, so only 4 unique banks for 8
column groups. This is a **2-way bank conflict** on every B-tile
read in the inner loop.

The design doc (kbit_gemm_context.md Section 5) identified this and
proposed +1 padding (stride=9, all 8 banks unique), but the fix was
never implemented in the production kernel. Fixing this eliminates
4 wasted cycles per (ks, nb) pair = 32 cycles per k_tile.

This should be fixed regardless of other changes.

---

## 3. Attempted: Dequant-During-Fetch Restructuring (v2)

### 3.1 What we tried

Moved all dequantization from the compute phase to the fetch phase.
The compute_tile became a pure ldmatrix+MMA loop (~200 instructions
per k_tile, down from ~1000). B tile stored as dequantized fp16 in
shmem with XOR swizzle for bank-conflict-free ldmatrix.x2.trans
loading.

The v2 kernel compiled, passed all 85 production tests, and produced
correct results (error within fp16 accumulation tolerance).

### 3.2 Why it didn't help

Benchmark results (v2 vs v1, M=32, K=4):

| Layer | v1 (us) | v2 (us) | Change |
|-------|--------:|--------:|-------:|
| Qwen3 MoE gate/up (2048x512) | 75 | 70 | -7% |
| Qwen3 dense gate/up (2048x5120) | 72 | 70 | -3% |
| GLM4.7 shared gate/up (2048x10240) | 73 | 130 | **+78%** |
| GLM4.7 shared down (10240x2048) | 80 | 71 | -11% |

Moving dequant from compute to fetch just moved the bottleneck.
The pipeline cannot overlap them because with double-buffered
stages, the fetch for tile N+1 must complete before compute can
start on it. The total work per k_tile is unchanged — ~700 ALU
instructions for dequant + ~200 for MMA, regardless of which
phase they run in.

Worse, v2 added overhead:
- B shmem grew from 4 KB to 16 KB per stage (dequantized fp16
  vs packed bit-planes), increasing shmem pressure
- Lost cp.async for B (replaced with regular global loads +
  shmem stores for the dequantized data)
- 32 scalar stores per thread per quantization block to shmem

### 3.3 Why overlap strategies fail on Ada (sm_89)

Three overlap approaches were considered:

**Option A (multi-stage pipeline):** More stages let fetch and
compute overlap across different tiles. But fetch is 3.5x longer
than compute, so even with 4 stages the fetch is the critical path.

**Option B (dequant during MMA in same warp):** Issue MMA, then do
ALU dequant while tensor cores execute. **Does not work on Ada.**
`mma.sync` is synchronous — the warp stalls until MMA completes
(~16-32 cycles). The dequant needs ~300+ cycles. The warp cannot
do ALU work while stalled on `mma.sync`.

**Option C (warp specialization):** Split 8 warps into MMA warps
and dequant warps. When an MMA warp stalls on `mma.sync` (~30
cycles), the scheduler switches to a dequant warp. Problem: the
dequant is 10-40x more work than MMA. The MMA warps would be idle
most of the time. Overlap recovers at most ~10% of the dequant cost.

### 3.4 The fundamental constraint

On Ada/Ampere/consumer-Blackwell GPUs using `mma.sync`, the ALU
dequant work cannot be hidden behind tensor core execution. The two
are serialized within each warp, and warp-level interleaving provides
negligible overlap due to the extreme ALU:MMA ratio (39:1).

This constraint does NOT apply to:
- **Hopper (sm_90a):** `wgmma.mma_async` is truly asynchronous —
  the warp continues executing ALU after issuing MMA.
- **Blackwell datacenter (sm_100a):** `tcgen05.mma` is single-thread
  asynchronous with dedicated Tensor Memory (TMEM).

Consumer Blackwell (sm_120, RTX 5090, RTX PRO 6000) uses `mma.sync`,
same as Ada. Confirmed: `wgmma` instructions produce compiler errors
on sm_120 targets.

### 3.5 Decision

**V2 kernel reverted.** The v1 inner loop is retained as-is. For
MoE shapes, the performance bottleneck is not the inner loop — it
is the low SM utilization from launching individual expert GEMMs.

---

## 4. The Path Forward: Grouped Expert GEMM

### 4.1 Why this is the right approach

Individual MoE expert GEMMs on Qwen3-Coder-Next:
- Expert gate/up: K=2048, N=512 → 4 tiles on 128 SMs (3% util)
- Expert down: K=512, N=2048 → 16 tiles on 128 SMs (12% util)
- Kernel time: ~70-75 us (instruction-limited, L2-resident)
- cuBLAS: ~22-27 us (also underutilized, but lower overhead)

The v1 kernel already achieves ~2x over cuBLAS on large shapes where
SMs are fully utilized (Llama3-8B: 1.5x, Llama3-70B: 2.6x). The
compression advantage (3.6x less data) is real — it just can't be
realized when 97% of SMs are idle.

A grouped expert GEMM batches all active experts into one kernel
launch:
- Qwen3-Next inference, batch=32, top-8 routing:
  256 expert invocations × 4 tiles = 1024 total tiles
- All 128 SMs active, ~8 tiles per SM
- Total weight data: ~32-64 MB across unique experts → DRAM-bound
- Compression advantage applies → expected ~2x over cuBLAS

### 4.2 API design

New op: `kbit_grouped_gemm(A_list, B_packed_list, absmax_list,
codebook, K_dim, N, k)` where the lists contain per-expert tensors
(or a single concatenated tensor with offset arrays).

The kernel reuses the v1 inner loop. The persistent work distribution
changes: instead of iterating over (m_tile, n_tile, k_split) for one
matrix, it iterates over (expert_id, m_tile, n_tile, k_split) across
all experts.

### 4.3 Implementation sketch

```cpp
// Grouped GEMM: each work item is (expert, mn_tile, k_split)
// Expert metadata passed via constant memory or kernel args.
struct ExpertDesc {
    const scalar_t* A;     // [M_expert, K_dim]
    int M;                 // tokens routed to this expert
    int b_offset;          // offset into packed B / absmax arrays
};

// Persistent kernel distributes work across all experts
for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
    // Decode: which expert, which (m,n) tile, which k_split
    auto [expert_id, mn_id, ks_id] = decode_work_id(work_id);
    const auto& desc = experts[expert_id];
    // ... same inner loop as v1 ...
}
```

### 4.4 Performance estimate

With 1024 tiles on 128 SMs and DRAM-bound data:
- Weight read: ~40 MB compressed at 900 GB/s = 44 us
- cuBLAS equivalent: ~40 MB × 3.6 = 144 MB at 900 GB/s = 160 us
- Expected speedup: ~2-3x vs fp16 cuBLAS grouped GEMM
- Per-expert amortized time: ~0.2 us (vs 70 us individually)

---

## 5. Implementation Order

### Step 1: Grouped expert GEMM kernel
The primary deliverable. Extend the v1 persistent kernel to handle
multiple experts in one launch. Metadata (per-expert A pointer, M,
B offset) passed via kernel args or constant memory.

### Step 2: Python API and expert batching
New `kbit_grouped_gemm` op. Python-side logic to:
- Collect active experts and their routed tokens
- Build the expert descriptor array
- Launch the grouped kernel
- Scatter results back to per-token outputs

### Step 3: Integration with LinearNbit / MoE module
Wire the grouped GEMM into the MoE forward pass. This requires
coordination with the router/gating logic.

### Step 4 (future): Hopper/Blackwell datacenter codepath
For sm_90a+ GPUs, a separate kernel using `wgmma.mma_async` (Hopper)
or `tcgen05.mma` (Blackwell DC) where dequant-during-MMA overlap is
viable. This would also benefit per-expert shapes without grouping.

---

## 6. GPU Architecture Reference

| GPU | SM | MMA instruction | Async? | Our approach |
|-----|-----|-----------------|--------|-------------|
| RTX 4090 | sm_89 | `mma.sync` | No | Grouped GEMM |
| RTX 5090 | sm_120 | `mma.sync` (ext) | No | Grouped GEMM |
| RTX PRO 6000 | sm_120 | `mma.sync` (ext) | No | Grouped GEMM |
| H100/H200 | sm_90a | `wgmma.mma_async` | Yes | Future: dequant overlap |
| B200/GB200 | sm_100a | `tcgen05.mma` | Yes | Future: dequant overlap |

sm_120 (consumer Blackwell) gains FP4/FP6 tensor core data types and
more SMs (up to 192 on GB202) but retains the synchronous `mma.sync`
model. The grouped GEMM approach works on all of these GPUs.

---

## 7. Model Shape Reference

### Qwen3-Coder-Next (primary target)

| Layer type | K_dim | N | kbit data | Tiles | SM util |
|------------|------:|-----:|----------:|------:|--------:|
| MoE gate/up (per expert) | 2048 | 512 | 0.5 MB | 4 | 3% |
| MoE down (per expert) | 512 | 2048 | 0.5 MB | 16 | 12% |
| Dense gate/up | 2048 | 5120 | 5.2 MB | 40 | 31% |
| Dense down | 5120 | 2048 | 5.2 MB | 16 | 12% |
| Q proj | 2048 | 4096 | 4.2 MB | 32 | 25% |
| KV proj | 2048 | 512 | 0.5 MB | 4 | 3% |
| O proj | 4096 | 2048 | 4.2 MB | 16 | 12% |

MoE expert shapes are the priority. With grouped GEMM (256+
invocations batched), effective tile count reaches 1000+ and SM
utilization hits 100%.

### GLM-4.7-Flash (secondary target)

| Layer type | K_dim | N | kbit data | Tiles | SM util |
|------------|------:|-----:|----------:|------:|--------:|
| Routed gate/up | 2048 | 1536 | 1.6 MB | 12 | 9% |
| Routed down | 1536 | 2048 | 1.6 MB | 16 | 12% |
| Shared gate/up | 2048 | 10240 | 10.5 MB | 80 | 62% |
| Shared down | 10240 | 2048 | 10.5 MB | 16 | 12% |

All shapes fit in L2 cache (72 MB on 4090) when launched
individually. With grouped GEMM, total data across experts exceeds
L2, making the kernel DRAM-bound — exactly where the 3.6x
compression advantage pays off.
