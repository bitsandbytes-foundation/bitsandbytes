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
   Override M values: `M_VALS=1,2 bash benchmarks/bench_ncu.sh`.
   Override expert count: `NUM_EXPERTS=16 bash benchmarks/bench_ncu.sh`.

   Default runs M=1..8 (scalar/grouped limited to M<=4 automatically).

   The script first prints raw per-kernel tables (MMA, Scalar, Grouped,
   Grouped MMA, cuBLAS), then a model-level summary: **one table per M
   value** with all kernels as columns, all (shape, k) combinations as
   rows:

   ```
   M=1:
   +========+=====+=======+========+=========+=========+=======+========+=========+
   | shape  |   k |   MMA | Scalar | Grouped | Grp MMA |  fp16 |   Best | vs fp16 |
   +--------+-----+-------+--------+---------+---------+-------+--------+---------+
   | gateup |   2 |  15.3 |    9.6 |     -   |     -   |  18.5 | Scalar |   1.93x |
   | gateup |   3 |  16.8 |   10.8 |     -   |     -   |  18.5 | Scalar |   1.72x |
   ...
   | moe_gu |   4 |   -   |    -   |    11.5 |    11.7 |  10.8 | Grouped |   0.94x |
   | moe_dn |   4 |   -   |    -   |    24.8 |    12.0 |  12.3 | Grp MMA |   1.03x |
   ...
   | TOTAL  |     |       |        |         |         |       |        |         |
   |  k=2   |   2 |       |        |         |         |       |   57.5 |  1.73x  |
   |  k=3   |   3 |       |        |         |         |       |   65.6 |  1.51x  |
   |  k=4   |   4 |       |        |         |         |       |   74.3 |  1.34x  |
   |  k=5   |   5 |       |        |         |         |       |   82.7 |  1.20x  |
   +========+=====+=======+========+=========+=========+=======+========+=========+
   ```

   Dense shapes (gateup, down, Q, O, KV) show MMA, Scalar, and fp16.
   MoE shapes (moe_gu, moe_dn) show Grouped, Grp MMA, and fp16 (bmm).
   "Best" picks the fastest kbit kernel (not fp16).
   "vs fp16" is fp16 / Best — values >1.00x mean kbit wins, <1.00x mean
   fp16 is faster. A dash "-" means no kbit kernel exists for that config.
   TOTAL has one row per k-value: it sums the best kernel time across all
   7 shapes for that k, giving the total weight matmul time per transformer
   block at that quantization level. Each shape appears once per block.

   Compare the "after" tables against the "before" tables to confirm
   improvement or regression.
3. **Repeat 1-2** until performance is satisfactory.
4. **Run tests (pre-commit only).** Before committing, run the
   kbit matmul tests:
   ```bash
   pytest tests/test_kbit_gemm.py tests/test_scalar_gemv.py -v --tb=short -n 4
   ```
   Do not run the full test suite. Only these two test files cover the
   kernels in this document.
5. **Commit and push.**
6. **Report results.** After benchmarking, print every per-M summary
   table (M=1 through M=8) verbatim from the benchmark output. Write
   the tables directly in your response text — do not summarize or
   abbreviate. The user will inspect the tables themselves.

### Dequant overhead benchmark (run only when requested)

A separate benchmark measures the cost of dequantizing k-bit weights
to fp16 before calling cuBLAS. This does not need to be run during
normal kernel development — only run it if the user explicitly asks,
or if the dequantize_kbit kernel or dispatch path changes.

```bash
bash benchmarks/bench_dequant.sh
```

This uses ncu to measure the actual `kDequantizeBlockwise_kbit_vec`
kernel time (no Python dispatch overhead), then CUDA events for fp16
matmul. Output is one table per k value showing fp16 time, dequant
time, total, and a speed ratio (fp16 / total) where 1.00 = full fp16
speed. Dequant time scales linearly with element count and k.

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

See `token_analysis.md` for a detailed workload analysis using real
token distributions from 397 Claude Code sessions. The analysis shows
that in single-user inference, M=1 decode accounts for 80-84% of total
GEMM time. In multi-user vLLM serving, the M distribution is bimodal
(M=num_users for decode-only iterations, M=num_users+chunk for prefill
iterations), and the crossover where quantized kernels become slower
than fp16 is at ~16 concurrent users.

---

## Five-kernel strategy

Each kernel covers a range of M where it has a structural advantage.
The dispatch logic selects the best kernel per (layer_type, M) pair.

| Kernel | M range | Layer types | Data format |
|--------|---------|-------------|-------------|
| 1. Scalar GEMV | 1-4 | Dense, attention | Flat (quantize_kbit), float32 absmax |
| 2. MMA dequant | 5-16 | Dense, attention | Tiled (repack_kbit), E4M4 absmax |
| 3. Dequant + cuBLAS | 17+ | Dense, attention | Flat -> fp16 |
| 4. Grouped scalar GEMV | 1-4 | MoE experts | Flat (quantize_kbit), float32 absmax |
| 5. Grouped MMA | 1+ | MoE experts | Tiled (repack_kbit), E4M4 absmax |

Why five kernels instead of one:
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
- The grouped scalar GEMV and grouped MMA serve complementary roles:
  scalar wins at M=1-4 for moe_gu (K=2048, N=512) where its C=1
  grid gives better parallelism; grouped MMA wins at all M for
  moe_dn (K=512, N=2048) and at M>4 for moe_gu.

**Practical importance (from workload analysis in `token_analysis.md`):**

In real deployments, the M distribution is bimodal — not uniform. With
vLLM continuous batching, iterations are either pure-decode (M=num_users)
or decode+prefill (M=num_users+chunk_size). The MMA kernel's M=5-16
range falls in the gap between these modes.

| Scenario | Scalar share | MMA share | dq+cuBLAS share |
|----------|-------------|-----------|-----------------|
| 1 user | 87% | 0% | 13% |
| 4 users | 59% | 0% | 41% |
| 8 users | 0% | 45% | 55% |
| 16 users | 0% | 24% | 76% |
| 32+ users | 0% | 6% | 94% |

**Current optimization priority:**

1. **MoE grouped kernel at large M (prefill)** — the remaining
   bottleneck. At M=544 (32-user prefill), the grouped MMA kernel is
   ~1.7x slower than raw fp16 BMM. MoE layers account for 22-30% of
   per-block time, making this the dominant source of regression at
   scale. Potential fix: hybrid dispatch that switches to dq+cuBLAS
   BMM for MoE layers when M exceeds a threshold.
2. **Scalar GEMV at M=1-4** — highest absolute time contributor in
   1-4 user decode (30-37% of total). Already well-optimized (V8).
3. **Dense dequant overhead** — already well-optimized, barely matters.

---

## 1. Scalar GEMV (`kbit_scalar_gemv`)

**Location:** `ops.cu` (search for `kbit_scalar_gemv`)

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

**Location:** `ops.cu` (search for `kbit_gemm_prod`)

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

**Dequant kernel** (`kDequantizeBlockwise_kbit_vec`): a single CUDA
kernel that reads k-bit packed data + absmax and writes fp16 output.
Templated on absmax type: float32 (from `quantize_kbit` directly),
uint8 E4M4, or fp16. The float32 absmax path was added to eliminate
a previous Python-side E4M4 conversion that launched ~15 PyTorch
elementwise kernels (~800 us). Now it is a single kernel launch.

Dequant GPU kernel times (ncu-measured, k=4):

| Shape | Elements | Kernel time |
|-------|----------|-------------|
| gateup/down | 10.5M | ~30 us |
| Q/O | 8.4M | ~25 us |
| KV | 1.0M | ~5 us |

Times scale linearly with element count and k.

**Crossover vs MMA:** At M<=16, MMA beats dequant+cuBLAS on most
shapes because the fixed dequant cost (~25-30 us) is large relative
to the matmul. At M>=64, dequant+cuBLAS wins because cuBLAS scales
efficiently while MMA is instruction-limited. The crossover is
M=32-64 depending on shape.

**Data format:** Uses flat layout (same as scalar GEMV). The
`dequantize_kbit` launcher handles float32, uint8 E4M4, and fp16
absmax via the `_KBIT_ABSMAX_SUFFIX` dispatch map.

---

## 4. Grouped scalar GEMV (`kbit_grouped_scalar_gemv`)

**Location:** `ops.cu` (search for `kbit_grouped_scalar_gemv`)

**Operation:** For each expert e: C_e[M_e, N] = A_e[M_e, K] * W_e^T,
all experts in one kernel launch.

**Architecture (V8):**
- 64 threads (2 warps), one output column per block (C=1)
- Grid = (N, num_experts) — Y-dimension indexes experts
- `__launch_bounds__(64, 24)` for M<=2, `__launch_bounds__(64, 16)` for M>2
- M_VAL dispatch (1/2/3/4 templates)

**Data format:**
- B_packed_all: flat from `quantize_kbit` — concatenated per-expert,
  each `[N * num_k_blocks * k]` uint32 (truncated to exact size)
- B_absmax_all: flat float32 — concatenated per-expert,
  each `[N * num_k_blocks]` float32 (truncated to exact size)
- No repack step needed. Uses same flat layout as the dense scalar GEMV.

**Inner loop:** Identical to the dense scalar GEMV (V8): vectorized
int4 A loads, 4-group sub-loop of 8 elements, shuffle codebook lookup.
The only difference is per-expert pointer arithmetic using
`expert_offsets[expert_id]` to find each expert's A, B, and C regions.

**Why grouped scalar wins for moe_gu (K=2048, N=512) at M<=4:**
With C=1, the grid is N × num_experts = 512 × 8 = 4096 blocks. This
gives full SM utilization (32 blocks/SM). The grouped MMA at this shape
has far fewer blocks due to tiling overhead.

**Quantize_kbit padding:** `quantize_kbit` appends a small padding
(4 packed words + 1 absmax) to each expert's output. The test and
benchmark helpers truncate each expert's data to the exact expected
size before concatenation, so the kernel's arithmetic indexing
(`expert_id * N * num_k_blocks * K_BITS`) works correctly.

---

## 5. Grouped MMA (`kbit_grouped_gemm_prod`)

**Location:** `ops.cu` (search for `kbit_grouped_gemm_prod`)

**Operation:** For each expert e: C_e[M_e, N] = A_e[M_e, K] * W_e^T,
all experts in one kernel launch. Handles all M values (no M<=4 limit).

**Architecture:**
- TILE_N=64 for M<=16 (128 threads, 4 warps) — doubles N-tiles for
  small-N shapes like moe_gu (N=512)
- TILE_N=128 for M>16 (256 threads, 8 warps)
- TILE_K=64, TILE_M=16*M_BLOCKS (M_BLOCKS=1..4)
- Double-buffered cp.async pipeline (same as dense MMA kernel)
- Persistent kernel with auto k_splits
- Caller passes `max_M` to select M_BLOCKS template and compute
  total_work on the host (no device-to-host sync needed)

**Data format:**
- B_packed_all: tiled from `repack_kbit` — concatenated per-expert
- B_absmax_all: E4M4 uint8 tiled — concatenated per-expert
- Uses same tiled layout as the dense MMA kernel

**Work distribution (inline linear scan):**

Each block gets a flat `work_id` and maps it to (expert, m_tile,
n_tile, k_split) via a linear scan over `expert_offsets`:
```
tiles_so_far = 0
for e = 0..num_experts-1:
    M_e = expert_offsets[e+1] - expert_offsets[e]
    m_tiles_e = ceil(M_e / TILE_M)
    mn_tiles_e = m_tiles_e * n_tiles
    expert_total = mn_tiles_e * k_splits
    if work_id < tiles_so_far + expert_total:
        expert_id = e
        break
    tiles_so_far += expert_total
```

With 8 active experts, this is 8 iterations of integer math — faster
than binary search with unpredictable branches, and eliminates the
previous `cudaMemcpy` + `cudaMalloc` + `cudaFree` that computed
work_offsets on the host.

**k_splits heuristic:**
Same as dense MMA: targets 4 blocks/SM for TILE_N=64, 1 block/SM for
TILE_N=128. For moe_gu (K=2048, N=512) at M=1 with TILE_N=64:
8 N-tiles × 8 experts = 64 mn_tiles, target = 512, so k_splits = 8
(K=2048 has 32 k-tiles). Total work = 512 blocks, 4 per SM.

**Split-K write-back:**
When k_splits > 1, partial results are atomicAdd'd to a float32
workspace. The last block to arrive (tracked by `tile_counters`)
converts the workspace to the output dtype. The workspace and
tile_counters are allocated and zeroed per-call in the Python backend.

**Performance (k=4, 8 experts):**

| Shape | M | Grp MMA (us) | fp16 BMM (us) | vs fp16 |
|-------|---|-------------|---------------|---------|
| moe_gu (2048×512) | 1 | 11.7 | 10.8 | 0.94x |
| moe_gu | 4 | 11.8 | 12.3 | 1.04x |
| moe_gu | 8 | 12.3 | 12.5 | 1.02x |
| moe_gu | 32 | 19.3 | 12.5 | 0.65x |
| moe_gu | 64 | 28.3 | 17.0 | 0.60x |
| moe_dn (512×2048) | 1 | 12.0 | 12.3 | 1.03x |
| moe_dn | 4 | 12.2 | 12.3 | 1.01x |
| moe_dn | 8 | 13.1 | 12.0 | 0.91x |
| moe_dn | 32 | 15.4 | 24.2 | 1.57x |
| moe_dn | 64 | 22.3 | 12.6 | 0.57x |

The kernel wins or matches at M=1-8, is competitive at M=16, and
loses at M=32+ where cuBLAS BMM becomes compute-bound. At large M
(prefill), the MoE grouped kernel is ~1.7x slower than fp16 BMM,
which is the dominant source of regression at 16-32 concurrent users
(MoE layers account for 22-30% of per-block time).

**Known issue: large-M MoE regression.**
At prefill M=8/expert (512 experts), moe_gu takes 33.6 us vs 12.2 us
fp16 — a 2.75x gap. The theoretical limit with perfect dequant/MMA
overlap is 4.5 us (2.7x *faster* than fp16). The kernel is 7.4x off
this limit due to serialized dequant. See `moe-kernel-spec.md` for the
optimization plan: Phase 1 (tile tuning, hours) targets 22 us, Phase 2
(warp-specialized producer/consumer pipeline, same idea as Marlin)
targets < 10 us.

---

## Data formats

Two formats exist, and which kernel uses which matters:

**Flat (from `quantize_kbit`):**
- B_packed: `[N * num_k_blocks * k]` uint32, row-major per column
- B_absmax: `[N * num_k_blocks]` float32
- No preprocessing. Used by: scalar GEMV, grouped scalar GEMV,
  dequant kernel.

**Tiled (from `repack_kbit`):**
- B_packed: reorganized into `[k_tiles * n_tiles * TILE_N * B_COL_WORDS]`
  for coalesced cp.async loads per tile
- B_absmax: E4M4-encoded uint8, same tiled layout
- Requires a one-time repack pass. Used by: MMA kernel, grouped MMA
  kernel.

E4M4 encodes each float32 absmax as a single byte (4-bit exponent +
4-bit mantissa). Decode is branchless: `ldexp(mantissa, exponent-bias)`.
This saves 4x bandwidth for absmax reads but adds a decode step in
the inner loop.

**Note:** The grouped scalar GEMV and grouped MMA use different data
formats. The grouped scalar GEMV uses flat layout with float32 absmax
(same as the dense scalar GEMV), while the grouped MMA uses tiled
layout with E4M4 absmax (same as the dense MMA). This means MoE
expert weights must be stored in both formats if both kernels are used
in the dispatch, or a runtime conversion must happen. Currently the
benchmark prepares each format separately.

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
| RTX 4090 | sm_89 | mma.sync | No | All 5 kernels as described |
| RTX 5090 | sm_120 | mma.sync (ext) | No | Same strategy, more SMs (192) |
| H100/H200 | sm_90a | wgmma.mma_async | Yes | Could overlap dequant + MMA |
| B200/GB200 | sm_100a | tcgen05.mma | Yes | Could overlap dequant + MMA |

On Hopper/datacenter-Blackwell, the MMA dequant kernel could be
restructured to issue MMA asynchronously while doing ALU dequant in
parallel. This would eliminate the 39:1 instruction overhead that
limits the current kernel on Ada. That is a separate future effort.
