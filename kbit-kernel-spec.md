# kbit inference kernels for Qwen3-Coder-Next 70B

RTX 4090 (128 SMs, sm_89), k=2..5, fp16/bf16.

## Workflow

The default workflow is benchmark-first. Tests are only run right
before a commit, not during development iterations.

1. **Edit kernel code.**
2. **Benchmark.** Always benchmark after changes. Do not run tests
   at this stage.
   ```bash
   bash benchmarks/bench_ncu.sh
   ```
   This runs the full grid: 5 shapes × 4 k-values × M=1,2,3,4,8
   for MMA, scalar GEMV, and cuBLAS fp16 baselines. Takes ~30-60s.
   Override M values with `M_VALS=3,4 bash benchmarks/bench_ncu.sh`.

   The script uses ncu (single-process, time-only metric) for MMA and
   scalar kernels, and CUDA events for cuBLAS fp16. Output is three
   tables of `shape k M avg_us`. Compare the "after" numbers against
   the "before" numbers to confirm improvement or regression.
3. **Repeat 1-2** until performance is satisfactory.
4. **Run tests (pre-commit only).** Before committing, run the
   kbit matmul tests:
   ```bash
   pytest tests/test_kbit_gemm.py tests/test_scalar_gemv.py -v --tb=short -n 4
   ```
   Do not run the full test suite. Only these two test files cover the
   kernels in this document.
5. **Commit and push.**

---

## Target model

Qwen3-Coder-Next 70B is a Mixture-of-Experts model with hidden_dim=2048.
The inference workload spans four layer types with distinct shapes:

| Layer | K | N | Data (k=4) | Notes |
|-------|----:|-----:|----------:|-------|
| MoE gate/up (per expert) | 2048 | 512 | 0.5 MB | 512 experts, top-8 routing |
| MoE down (per expert) | 512 | 2048 | 0.5 MB | |
| Dense gate/up | 2048 | 5120 | 5.2 MB | Shared across all tokens |
| Dense down | 5120 | 2048 | 5.2 MB | |
| Q proj | 2048 | 4096 | 4.2 MB | |
| KV proj | 2048 | 512 | 0.5 MB | |
| O proj | 4096 | 2048 | 4.2 MB | |

At inference batch size 32 with top-8 routing, a single forward pass
invokes ~256 expert GEMMs plus the dense/attention layers. Individual
expert shapes produce only 4-16 tiles on 128 SMs (3-12% utilization).
Dense shapes produce 16-80 tiles (12-62%).

The batch size M seen by each kernel varies:
- **M=1**: autoregressive token generation (dominant use case)
- **M=1-4**: MoE experts after routing (few tokens per expert)
- **M=1-32+**: dense layers (full batch)
- **M=32-512+**: prefill / prompt processing

---

## Four-kernel strategy

Each kernel covers a range of M where it has a structural advantage.
The dispatch logic selects the best kernel per (layer_type, M) pair.

| Kernel | M range | Layer types | Data format |
|--------|---------|-------------|-------------|
| 1. Scalar GEMV | 1-4 | Dense, attention | Flat (quantize_kbit) |
| 2. MMA dequant | 5-16 | Dense, attention | Tiled (repack_kbit) |
| 3. Dequant + cuBLAS | 17+ | Dense, attention | Flat -> fp16 |
| 4. Grouped expert GEMV | 1-4 | MoE experts | Tiled (repack_kbit) |

Why four kernels instead of one:
- At M=1, tensor cores waste 94% of their compute (m16n8k16 pads 15
  zero rows). A scalar kernel that avoids MMA entirely wins by 3-5x.
- At M=5-16, MMA utilization rises to 31-100%. The 3.2x data
  compression from k-bit quantization beats cuBLAS, which must read
  the full fp16 weight matrix.
- At M>16, cuBLAS tensor core GEMM is highly optimized and pipeline-
  efficient. Dequantizing to fp16 and calling cuBLAS is simpler and
  competitive, because cuBLAS hides the extra data movement behind
  its compute pipeline.
- MoE experts launched individually waste 88-97% of SMs. Grouping
  all active experts into one kernel launch solves this.

---

## 1. Scalar GEMV (`kbit_scalar_gemv`)

**Location:** `ops.cu:2571`

**Operation:** C[M,N] = A[M,K] * W_kbit^T, M=1..4.

**Architecture:**
- 64 threads (2 warps), one output column per block
- Grid = N (direct mapping, no persistent loop)
- `__launch_bounds__(64, 24)` for M<=2, `__launch_bounds__(64, 16)` for M>2
- No shared memory for B data, no cp.async, no split-K

**Data format:**
- B_packed: flat from `quantize_kbit` — `[N * num_k_blocks * k]` uint32
- B_absmax: flat float32 — `[N * num_k_blocks]`
- No repack step needed

**Inner loop (V8):**

Each thread strides through quantization blocks along K:
```
for each quant block (stride 64):
    load k bit-plane words (vectorized: int2 for k=2, int4 for k=4)
    load float32 absmax

    for sub = 0..3:            // 4 groups of 8 elements
        load A[m, k_base + sub*8 .. +7] via int4 (8 fp16 values)
        for j = 0..7:
            extract k-bit index from bit-planes
            w = __shfl_sync(cb, idx) * absmax
            for m = 0..M_VAL-1:
                acc[m] += w * A_vec[m][j]
```

The key optimization: dequantize each weight once, then FMA across all
M rows. The int4 vector load for A amortizes address computation and
gives the compiler 8 independent FMA chains for ILP.

**Reduction:**
- Intra-warp: shuffle reduction (5 steps)
- Inter-warp: 2-phase shared memory (32 bytes), single `__syncthreads`
- Thread 0 writes M output values to C

**Performance (Qwen3 dense gate/up, K=2048 N=5120, k=4):**

| M | Time (us) | BW (GB/s) | vs cuBLAS fp16 |
|---|-----------|-----------|----------------|
| 1 | 13.1 | 512 | 3.9x faster |
| 2 | 14.8 | 450 | 1.2x slower |
| 3 | 16.6 | 401 | 1.3x slower |
| 4 | 19.8 | 337 | 1.6x slower |

The kernel is purely DRAM-bound (arithmetic intensity = 3.2 FLOP/byte
for k=4, far below the 82 FLOP/byte compute-to-memory ratio of
RTX 4090).

**Design decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Columns per block | 1 | Grid=N gives full SM occupancy for N>=1536 |
| Thread count | 64 (2 warps) | Fewer threads = more blocks/SM = better occupancy |
| A storage | Global/L1 | No A reuse with C=1; A fits in L1 (~4-10 KB) |
| B absmax | float32 | Uses quantize_kbit output directly, no repack |
| Inner loop | Vectorized 4x8 | int4 A loads + sub-loop gives ILP without blowing registers |

---

## 2. MMA dequant kernel (`kbit_gemm_prod`)

**Location:** `ops.cu:1784`

**Operation:** C[M,N] = A[M,K] * W_kbit^T, M=1..64+.

**Architecture:**
- TILE_N=64 for M<=16 (128 threads, 4 warps, `__launch_bounds__(128, 12)`)
- TILE_N=128 for M>16 (256 threads, 8 warps)
- TILE_K=64, TILE_M=16*M_BLOCKS (M_BLOCKS=1..4)
- Double-buffered cp.async pipeline for A, B, and absmax tiles
- Persistent kernel with split-K when tiles < target SM occupancy

**Data format:**
- B_packed: tiled from `repack_kbit` — `[k_tiles * n_tiles * TILE_N * B_COL_WORDS]`
- B_absmax: E4M4 uint8 tiled — `[k_tiles * n_tiles * TILE_N * KB_PER_TILE]`

**Dequant + MMA flow (per k-tile, per warp):**
```
load B bit-planes from shmem (4 uint32 for k=4)
load absmax from shmem, decode E4M4 -> fp16
for each (k_sub, n_block) pair:
    extract 4 k-bit indices from bit-planes
    __shfl_sync codebook lookup for each
    multiply by absmax, pack into MMA B-fragment
    ldmatrix for A-fragment from shmem (XOR swizzled)
    mma.sync.aligned.m16n8k16
```

**k_splits heuristic (TILE_N=64):**
```
target_blocks = 128 SMs * 4 blocks/SM = 512
if mn_tiles < 512:
    k_splits = min(k_tiles, ceil(512 / mn_tiles))
grid = min(512, mn_tiles * k_splits)
```

Split-K uses atomicAdd + tile_counters for the last-arriving split to
do the final reduction.

**Performance characteristics:**
- Wins 31/48 benchmark configs vs scalar GEMV (dominates at large K)
- Dense_down (5120x2048): 1.72x over scalar GEMV at M=4
- KV_proj (2048x512): loses to scalar GEMV (too few N-tiles)
- At M>=4 for most shapes, MMA amortizes the dequant cost

**The fundamental constraint on Ada:**
`mma.sync` is synchronous — the warp stalls until the MMA completes
(~16-32 cycles). Dequant requires ~300+ ALU cycles per MMA. The two
are serialized within each warp. Warp-level interleaving provides
negligible overlap due to the extreme ALU:MMA ratio (39:1 measured
from SASS). This means dequant is always on the critical path.

This does NOT apply to Hopper (`wgmma.mma_async`) or datacenter
Blackwell (`tcgen05.mma`), where MMA is truly asynchronous. Consumer
Blackwell (sm_120, RTX 5090) uses `mma.sync`, same as Ada.

**Occupancy analysis (TILE_N=64, k=4):**

| Resource | Per block | Per SM (9 blocks) | SM limit | Limiter? |
|----------|-----------|-------------------|----------|----------|
| Registers | 55/thread * 128 = 7040 | 63360 | 65536 | Yes (9 blocks max) |
| Shmem | 8.2 KB | 73.8 KB | 100 KB | No |
| Warps | 4 | 36 | 48 | No |

Theoretical max occupancy: 75% (9 blocks/SM, 36 warps).
Current heuristic caps at 4 blocks/SM -> ~28% achieved.
The gap between 28% and 75% is the main optimization opportunity.

Three directions to close this gap:
1. Increase TARGET_BLOCKS_PER_SM from 4 to 8-9 (more k_splits, more
   atomic reduction overhead, but better latency hiding)
2. Reduce register pressure from 55 to ~45 (move codebook to shmem,
   frees 8 registers, enables 11 blocks/SM)
3. Warp specialization (1 producer + 3 consumer warps; does not
   improve occupancy numerically but decouples load/compute pipelines)

---

## 3. Dequant + cuBLAS (large M fallback)

**Operation:** dequantize W to fp16, then call cuBLAS GEMM.

**Flow:**
1. `dequantize_kbit(B_packed, codebook, B_absmax, k, n_elements, fp16)` -> W_fp16
2. `torch.mm(A, W_fp16.T)` or `torch.bmm` for batched

**When this wins:**
At M>16, cuBLAS tensor core GEMM achieves near-peak throughput.
cuBLAS reads 3.2x more data (full fp16 weights vs k-bit compressed),
but it hides this behind a deeply pipelined compute schedule that our
MMA dequant kernel cannot match (due to the synchronous dequant
bottleneck on Ada).

For Qwen3 dense_gateup at M=32, k=4: cuBLAS achieves ~22 us, while
the MMA dequant kernel takes ~68 us (instruction-limited, only 1.3%
of execution is MMA). A fused dequant kernel would take ~5 us for
this shape, so dequant + cuBLAS ~27 us would beat 68 us.

**Current dequant implementation is not fused.** `dequantize_kbit`
dispatches ~15 PyTorch elementwise kernels per call, giving a constant
~800 us overhead regardless of shape. This makes dequant + cuBLAS
non-competitive at M<64. A fused dequant CUDA kernel is needed for
strategy 3 to be viable.

The crossover point depends on shape. For DRAM-bound shapes (Llama3-8B
gate/up at 4096x14336), the MMA dequant kernel wins at 1.5x over
cuBLAS because the 3.2x bandwidth savings dominate. For L2-resident
shapes (MoE experts, small dense layers), cuBLAS wins because the
kernel is instruction-limited, not bandwidth-limited.

**Data format:** Uses flat layout (same as scalar GEMV). The
`dequantize_kbit` launcher handles both uint8 E4M4 and float32 absmax.

---

## 4. Grouped expert GEMV (`kbit_grouped_scalar_gemv`)

**Location:** `ops.cu:2736`

**Operation:** For each expert e: C_e[M_e, N] = A_e[M_e, K] * W_e^T,
all experts in one kernel launch.

**Current architecture (needs V8 optimizations):**
- 128 threads (4 warps), COLS_PER_BLOCK=4 (each warp handles 1 column)
- Grid = (ceil(N/4), num_experts) — Y-dimension indexes experts
- Uses tiled layout with E4M4 absmax (from `repack_kbit`)
- Hard-coded M_VAL=4 template (no M-dispatch)
- Element-at-a-time A loads (old V1 inner loop)

**What needs to change:**

The grouped kernel inner loop is the pre-V8 design. It is missing:
- int4 vectorized A loads (sub-loop of 4 groups of 8 elements)
- 64-thread / 2-warp configuration with `__launch_bounds__` tuning
- M_VAL dispatch (1/2/3/4 templates instead of always 4)

The decision on data format is open: the scalar GEMV uses flat layout
with float32 absmax (no repack), while the grouped kernel currently
uses tiled layout with E4M4. The flat format avoids the repack step
but uses 4x more bandwidth for absmax. For MoE shapes where expert
weights are L2-resident, the extra absmax bandwidth may not matter.

**Why grouping is necessary:**

Individual expert launches for Qwen3 MoE:
- gate/up (2048x512): 4 tiles on 128 SMs = 3% utilization
- Kernel time: ~70 us (instruction-limited, L2-resident)
- cuBLAS: ~22 us (also underutilized)

Grouped launch with 256 expert invocations (batch=32, top-8):
- 256 * 4 tiles = 1024 tiles across 128 SMs = full utilization
- Total weight data: ~32-64 MB across unique experts -> DRAM-bound
- The 3.2x compression advantage now applies

**There is also a grouped MMA variant** (`kbit_grouped_gemm_prod` at
`ops.cu:2182`) that uses the MMA kernel inner loop with a persistent
work distribution across experts. This handles M>4 per expert. It uses
binary search on work_offsets to find the expert for each work item.

---

## Data formats

Two formats exist, and which kernel uses which matters:

**Flat (from `quantize_kbit`):**
- B_packed: `[N * num_k_blocks * k]` uint32, row-major per column
- B_absmax: `[N * num_k_blocks]` float32
- No preprocessing. Used by: scalar GEMV, dequant kernel.

**Tiled (from `repack_kbit`):**
- B_packed: reorganized into `[k_tiles * n_tiles * TILE_N * B_COL_WORDS]`
  for coalesced cp.async loads per tile
- B_absmax: E4M4-encoded uint8, same tiled layout
- Requires a one-time repack pass. Used by: MMA kernel, grouped kernels.

E4M4 encodes each float32 absmax as a single byte (4-bit exponent +
4-bit mantissa). Decode is branchless: `ldexp(mantissa, exponent-bias)`.
This saves 4x bandwidth for absmax reads but adds a decode step in
the inner loop.

---

## Per-bit-width considerations (k=2..5)

| k | Codebook entries | B load per block | Absmax fraction | Notes |
|---|-----------------|------------------|-----------------|-------|
| 2 | 4 | 8 bytes (uint2) | 33% of data | Highest absmax overhead |
| 3 | 8 | 12 bytes (3x uint32) | 25% | Non-power-of-2 stride, still coalesced |
| 4 | 16 | 16 bytes (int4) | 18% | Best vectorized load alignment |
| 5 | 32 | 20 bytes (int4 + uint32) | 15% | Codebook needs full warp (32 entries) |

The codebook is loaded into a register and accessed via `__shfl_sync`.
For k<=4, only lanes 0..(2^k-1) hold meaningful values. For k=5, all
32 lanes are used.

The inner loop scales linearly with k: k bit-extractions per weight
element (shift + AND + shift + OR each). For k=4, that is ~14 ALU ops
per element; for k=2, ~8 ops.

---

## GPU architecture reference

| GPU | SM | MMA instruction | Async MMA? | Kernel strategy |
|-----|-----|-----------------|------------|----------------|
| RTX 4090 | sm_89 | mma.sync | No | All 4 kernels as described |
| RTX 5090 | sm_120 | mma.sync (ext) | No | Same strategy, more SMs (192) |
| H100/H200 | sm_90a | wgmma.mma_async | Yes | Could overlap dequant + MMA |
| B200/GB200 | sm_100a | tcgen05.mma | Yes | Could overlap dequant + MMA |

On Hopper/datacenter-Blackwell, the MMA dequant kernel could be
restructured to issue MMA asynchronously while doing ALU dequant in
parallel. This would eliminate the 39:1 instruction overhead that
limits the current kernel on Ada. That is a separate future effort.
