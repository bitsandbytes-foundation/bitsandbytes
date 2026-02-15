# MMA Kernel Optimization Spec

## Current State

The scalar GEMV kernel (v8) handles M=1-4 efficiently, achieving 3-5x speedup
over cuBLAS fp16 at M=1. However, at M>=2, cuBLAS switches to tensor core GEMM
and is 1.2-1.6x faster than our scalar kernel. The existing MMA kernel
(`kbit_gemm_prod`) is too slow at small M to fill this gap.

### Scalar GEMV v8 (k=4, shape 0: K=2048 N=5120)

| M | us   | GB/s | vs cuBLAS fp16 |
|---|------|------|----------------|
| 1 | 13.1 | 512  | 3.9x faster    |
| 2 | 14.8 | 450  | 1.2x slower    |
| 3 | 16.6 | 401  | 1.3x slower    |
| 4 | 19.8 | 337  | 1.6x slower    |

cuBLAS fp16: ~12.3 us for M=2-4 (tensor cores, flat scaling).

### Target

An MMA-based dequant kernel that beats cuBLAS fp16 for M=2-16 by leveraging
the 3.2x data compression from k-bit quantization while using tensor cores for
the multiply-accumulate. Target: **8-10 us for M=2-4** (matching the theoretical
DRAM minimum of 8.7 us at 75% bandwidth).

---

## Why the Current MMA Kernel is Slow

Three compounding problems at small M, analyzed for k=4, K=2048, N=5120:

### 1. SM Utilization: 31%

With TILE_N=128, there are only `N/128 = 40` n-tiles. At M<=16, `m_tiles=1`,
so `total_work = 40`. On 128 SMs (RTX 4090), 88 SMs sit completely idle.

The k_splits heuristic doesn't trigger because B data (5.6 MB) is under the
24 MB DRAM threshold. Even with aggressive k_splits:

| k_splits | total_work | grid  | SM util |
|----------|-----------|-------|---------|
| 1        | 40        | 40    | 31%     |
| 2        | 80        | 80    | 62%     |
| 4        | 160       | 128   | 100%    |

But k_splits > 1 adds atomicAdd overhead and a __threadfence + tile_counter
synchronization per work item.

### 2. MMA Compute Waste: 75-94%

`mma.sync.aligned.m16n8k16` is the smallest MMA tile on sm_89. It computes
16 M-rows regardless of actual M. At M=1, 15/16 rows are zero-padded:

| M  | Useful outputs | Total MMA outputs | Utilization |
|----|---------------|-------------------|-------------|
| 1  | 128           | 2048              | 6.2%        |
| 2  | 256           | 2048              | 12.5%       |
| 4  | 512           | 2048              | 25.0%       |
| 8  | 1024          | 2048              | 50.0%       |
| 16 | 2048          | 2048              | 100.0%      |

This is an inherent hardware limitation — there is no m4n8k16 or m8n8k16 on
Ada Lovelace. M < 16 always wastes MMA compute.

### 3. A Tile DRAM Waste

Loading TILE_M * TILE_K * 2 = 2048 bytes per A stage, but at M=1 only
128 bytes are useful (6%). At M=4: 512 bytes useful (25%). This wastes
DRAM bandwidth and cp.async slots.

### 4. Dequant is the Bottleneck, Not MMA

Per B element, dequant requires:
- k bit extractions (shift + AND + shift + OR each): ~3k instructions
- 1 `__shfl_sync` (codebook lookup): 1 instruction
- 1 scale multiply: 1 instruction
- Total: ~3k + 2 instructions per element (14 for k=4)

Per TILE_N x TILE_K tile: 128 * 64 = 8192 elements to dequant.
Each thread dequants 4 elements per iteration (idx0-idx3), so
8192 / 4 / 32 lanes = 64 iterations per warp.

The MMA instruction (m16n8k16) takes ~8 cycles on tensor cores.
The dequant to prepare one B fragment takes ~64 scalar instructions.
**MMA is not the bottleneck — dequant is.**

---

## Optimization Strategy

### Dispatch Policy

Use the right kernel for each M range:

| M range | Kernel          | Rationale                                    |
|---------|-----------------|----------------------------------------------|
| 1       | Scalar GEMV v8  | 3-5x faster than cuBLAS, MMA wastes 94%      |
| 2-4     | MMA dequant v2  | Tensor cores amortize dequant, data savings   |
| 5-16    | MMA dequant v2  | Increasing MMA utilization, still data-bound  |
| 17+     | MMA prod (existing) | Full MMA utilization, existing kernel works |

### Architecture: MMA Dequant v2

Key changes from `kbit_gemm_prod`:

#### A. Reduce TILE_N from 128 to 64

This is the single most impactful change for SM utilization:

| TILE_N | n_tiles (N=5120) | shmem/stage | Max blocks/SM | Notes           |
|--------|-----------------|-------------|---------------|-----------------|
| 128    | 40              | 6400 B      | 8             | Current, 31% SM |
| 64     | 80              | 4224 B      | 12            | 62% SM at k=1   |
| 32     | 160             | 3136 B      | 16            | 100%+ SM        |

TILE_N=64 with k_splits=2 gives 160 work items = 100% SM utilization.
TILE_N=32 gives 160 tiles without needing k_splits, avoiding atomicAdd overhead.

Recommendation: **TILE_N=64 with k_splits=2** for best balance of SM util
vs. per-block work granularity. Consider TILE_N=32 as a fallback for
shapes where N is small.

Block structure at TILE_N=64:
- 128 threads (4 warps), each warp handles 16 columns (2 MMA N-blocks of 8)
- Or 256 threads (8 warps), each warp handles 8 columns (1 MMA N-block)
- Prefer 128 threads: fewer warps = more blocks/SM, better for small M

#### B. Decouple Dequant from MMA via Shared Memory

Current flow (per warp, per k-step):
```
load planes from shmem → bit extract → shuffle → scale → pack frag_b → MMA
```
This serializes dequant and MMA. The tensor cores idle during dequant.

Proposed flow — **dequant-to-shmem**:
```
Phase 1: All threads cooperatively dequant B tile → fp16 values in shmem
Phase 2: ldmatrix loads dequanted B from shmem → MMA
```

Benefits:
- `ldmatrix` is a single instruction to load a full MMA fragment from shmem
- MMA pipeline stays full — no scalar dequant in the critical path
- All threads participate in dequant (better parallelism)
- Clean double-buffering: dequant tile K+1 while MMA processes tile K

Shmem cost at TILE_N=64:
- B dequanted: 64 * 64 * 2 = 8192 bytes per stage
- A: 16 * 64 * 2 = 2048 bytes per stage
- Total: 10240 bytes/stage, 20480 bytes double-buffered
- Max 5 blocks/SM (100 KB limit) → 10 warps (128-thread blocks) or
  20 warps (if 4 warps/block with 5 blocks). Occupancy: 20-42%.

At TILE_N=32:
- B dequanted: 32 * 64 * 2 = 4096 bytes
- Total: 6144 bytes/stage, 12288 bytes double-buffered
- Max 8 blocks/SM → 32 warps = 67% occupancy. Better.

Trade-off: TILE_N=32 has better occupancy but 2x more tiles to process
and less N-parallelism per block.

#### C. Cooperative Dequant

In the dequant-to-shmem approach, all threads participate in dequanting:

```
Elements per tile: TILE_N * TILE_K = 64 * 64 = 4096 (at TILE_N=64)
Threads per block: 128
Elements per thread: 32
```

Each thread:
1. Loads K_BITS packed uint32 planes from B shmem (already fetched via cp.async)
2. Extracts bit indices for its assigned elements
3. Does __shfl_sync for codebook lookup
4. Multiplies by scale (absmax)
5. Writes fp16 result to B_dequant shmem

This is essentially the scalar GEMV's inner loop, but writing to shmem
instead of accumulating. The `__shfl_sync` requires all lanes to participate,
so threads within a warp must process elements from the same quantization
block (same codebook lookup pattern).

Thread mapping for dequant:
- 128 threads process 4096 elements = 128 quant blocks of 32 elements each
- Thread t handles quant block t (for TILE_K=64, KB_PER_TILE=2: 128 cols * 2 blocks)
- Each thread dequants 32 elements, writes 32 fp16 values to shmem

After `__syncthreads()`, all threads switch to MMA consumer role.

#### D. Smarter k_splits Heuristic

The current heuristic is too conservative. Replace with:

```
mn_tiles = m_tiles * n_tiles
target_blocks = num_sms  // fill all SMs

if mn_tiles >= target_blocks:
    k_splits = 1  // enough parallelism from M*N tiles
else:
    k_splits = min(k_tiles, ceil(target_blocks / mn_tiles))
    k_splits = min(k_splits, 4)  // cap to limit atomicAdd overhead
```

For M=2, N=5120, TILE_N=64: mn_tiles=80, target=128, k_splits=2,
total_work=160. All SMs active.

#### E. Avoid A Waste at Small M

At M < TILE_M (=16), most of the A tile is zero-padded. Two approaches:

**Option 1: Guard the cp.async** (current approach, already implemented).
Only fetch rows 0..M-1. Remaining shmem rows are zeroed cheaply.
This already works but wastes shmem space.

**Option 2: Dynamic TILE_M.** Use M_BLOCKS=1 (TILE_M=16) always for M<=16,
and accept the A waste. The A tile is small (2 KB) relative to B (4-8 KB),
so the waste is tolerable. Not worth the complexity of variable TILE_M.

Recommendation: Keep current approach. A waste is minor.

---

## Implementation Plan

### Phase 1: TILE_N=64 + Aggressive k_splits

Minimal changes to `kbit_gemm_prod`:
1. Add a TILE_N=64 variant (template parameter or separate kernel)
2. Reduce block to 128 threads (4 warps)
3. Update k_splits heuristic to always fill SMs
4. Update dispatcher to use TILE_N=64 for M <= 16

Expected impact: SM utilization 31% → 100%. Estimated 2-3x speedup for
small M, bringing the MMA kernel to ~15-20 us range.

### Phase 2: Dequant-to-Shmem

Major restructure of the compute loop:
1. Add B_dequant shmem buffer (TILE_N * TILE_K * 2 bytes per stage)
2. Split compute_tile into dequant_phase + mma_phase with __syncthreads between
3. Dequant phase: all threads extract bits, shuffle codebook, write fp16 to shmem
4. MMA phase: ldmatrix loads B fragments from shmem, runs MMA
5. Double-buffer: overlap dequant of tile K+1 with MMA of tile K

Expected impact: removes dequant from MMA critical path. Combined with
Phase 1, estimated 10-14 us for M=2-4 (competitive with cuBLAS 12.3 us).

### Phase 3: Tuning

1. Profile with ncu, identify remaining bottlenecks
2. Tune TILE_N (32 vs 64) per shape
3. Tune k_splits cap (2 vs 4)
4. Consider warp specialization (dedicated dequant vs MMA warps)
5. Consider persistent kernel for Phase 2 (reuse shmem across tiles)

---

## Expected Results

| M | Current MMA (est) | Phase 1 (est) | Phase 2 (est) | cuBLAS fp16 | Scalar GEMV v8 |
|---|-------------------|---------------|---------------|-------------|---------------|
| 1 | ~40 us            | ~20 us        | ~15 us        | 51.1 us     | **13.1 us**   |
| 2 | ~42 us            | ~18 us        | ~12 us        | 12.3 us     | 14.8 us       |
| 4 | ~44 us            | ~16 us        | ~10 us        | 12.5 us     | 19.8 us       |
| 8 | ~46 us            | ~14 us        | ~9 us         | ~12.5 us    | N/A           |
| 16| ~20 us            | ~12 us        | ~8 us         | ~12.5 us    | N/A           |

At M=1, scalar GEMV v8 remains the best choice. At M>=2, the optimized MMA
kernel should match or beat cuBLAS while reading 3.2x less data. The crossover
between scalar GEMV and MMA shifts from M~2 (vs cuBLAS) to M~2 (our own
kernels), giving the best of both worlds.

## Theoretical Limits

DRAM payload for k=4, K=2048, N=5120 (independent of M for M<=16):
- B_packed: 5.24 MB, B_absmax: 1.31 MB, A: negligible
- Total: ~6.6 MB
- At 100% DRAM peak (1008 GB/s): 6.5 us
- At 75%: 8.7 us
- At 50%: 13.0 us

cuBLAS fp16 reads 21.0 MB (3.2x more). Even at 100% DRAM utilization,
cuBLAS cannot go below 20.8 us for a pure memory-bound GEMV. The reason
cuBLAS achieves 12.3 us at M=2 is that it switches to a compute-bound
tensor core GEMM that reuses data in registers/shmem.

Our MMA kernel's advantage: read 6.6 MB instead of 21.0 MB. If we can
keep the tensor core pipeline fed, the 3.2x data reduction translates
directly to a 3.2x speed advantage at the DRAM-bound limit.
