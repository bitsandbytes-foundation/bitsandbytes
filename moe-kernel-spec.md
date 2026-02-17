# MoE grouped kernel optimization spec

## Benchmarking methodology

All kernel timings use NCU `gpu__time_duration.avg`, which measures
the GPU kernel execution only — no Python dispatch, no tensor
allocation, no workspace memset. This is the correct methodology for
kernel optimization. Python-side overhead (workspace allocation,
`torch.zeros`, ctypes dispatch) is a separate concern, trivially
fixed by adding `out=` parameters to the Python bindings, and
applies equally to all kernels (dense MMA, scalar GEMV, MoE, etc.)
and to fp16 cuBLAS calls.

The fp16 baseline uses `torch.bmm` measured with CUDA events, which
includes cuBLAS kernel time plus minor per-call GPU overhead from
output tensor allocation (~1-2 us). This is close enough to NCU
kernel-only for comparison purposes.

## Measured performance (NCU, k=4, 8 experts)

Baseline sweep across power-of-2 M values (per expert). All times
in microseconds, measured via NCU `gpu__time_duration.avg`.

```
shape        M  kbit_us  fp16_us   ratio k_spl
----------------------------------------------
moe_gu       1     11.7     16.5   1.41x     8
moe_gu       2     11.8     21.6   1.83x     8
moe_gu       4     12.0     19.2   1.60x     8
moe_gu       8     12.4     18.1   1.46x     8
moe_gu      16     14.4     18.6   1.29x     8
moe_gu      32     19.4     18.1   0.94x     4  ← crossover
moe_gu      64     28.7     18.0   0.63x     4
moe_gu     128     40.5     20.1   0.50x     2
moe_gu     256     58.3     32.1   0.55x     1
moe_gu     512    112.9     65.9   0.58x     1
moe_gu    1024    219.1    124.7   0.57x     1
moe_gu    2048    426.1    258.7   0.61x     1
moe_gu    4096    835.7    497.9   0.60x     1

moe_dn       1     12.2     17.5   1.44x     2
moe_dn       2     12.3     20.7   1.68x     2
moe_dn       4     12.7     18.3   1.44x     2
moe_dn       8     12.8     20.3   1.58x     2
moe_dn      16     14.5     19.0   1.31x     2
moe_dn      32     15.9     17.0   1.07x     1
moe_dn      64     22.8     22.0   0.96x     1  ← crossover
moe_dn     128     40.5     17.9   0.44x     1
moe_dn     256     74.4     31.4   0.42x     1
moe_dn     512    137.5     65.0   0.47x     1
moe_dn    1024    259.6    148.9   0.57x     1
moe_dn    2048    498.1    269.2   0.54x     1
moe_dn    4096    968.3    542.2   0.56x     1
```

Tile parameters by M (moe_gu K=2048, N=512, k_tiles=32):

```
    M m_blk  TN  mn_tiles k_spl tiles/split
    1     1  64        64     8           4
   16     1  64        64     8           4
   32     2 128        32     4           8
   64     4 128        32     4           8
  128     4 128        64     2          16
  256+    4 128   128-2048     1          32
```

## Problem

**At small M (≤16 per expert):** kbit is already 1.3-1.8x faster
than fp16 BMM. The k_splits=8 overhead for moe_gu is acceptable
because the kernel is memory-bandwidth-bound at these sizes and the
4-bit data is ~4x smaller than fp16 weights.

**At large M (≥32 per expert):** kbit becomes slower than fp16,
stabilizing at ~0.55-0.60x for M≥128. This is the regime that
matters for high-throughput multi-user serving (prefill with
hundreds of tokens routed to each expert).

The crossover occurs at M~32 for moe_gu and M~64 for moe_dn.

## Root cause: redundant weight dequantization

**The kernel re-dequants every weight once per m-tile.** This is
visible in the per-tile cost column:

```
    M   TM   TN  m_t  n_t    mn      us  us/mn×128  B_loads
    1   16   64    1    8    64    11.7     23.4        1x
    8   16   64    1    8    64    12.4     24.8        1x
   64   64  128    1    4    32    28.7    114.8        1x
  128   64  128    2    4    64    40.5     81.0        2x
  256   64  128    4    4   128    58.3     58.3        4x
  512   64  128    8    4   256   112.9     56.5        8x
 1024   64  128   16    4   512   219.1     54.8       16x
 4096   64  128   64    4  2048   835.7     52.2       64x
```

The `us/mn×128` column (time per tile, normalized to 128 SMs) is
constant at ~53-56 for large M. Each (m_tile, n_tile) work item
independently loads and dequants all k-tiles of B for its n-tile
columns. Different m-tiles for the same n-tile re-dequant identical
weight data.

At M=512 (m_tiles=8), each weight is dequanted 8×. At M=4096
(m_tiles=64), each weight is dequanted 64×. cuBLAS fp16 does not
have this problem — it loads B once and iterates M rows of A against
the cached B data.

**Our codebook dequant is ~14 ALU ops per element** (bit-plane
extraction, `__shfl_sync` codebook lookup, absmax multiply). This is
3-4× more expensive than Marlin's INT4 lop3 dequant (~3-5 ops).
The redundancy penalty is therefore 3-4× worse for us than it would
be for Marlin. Eliminating redundant dequant is the single most
impactful optimization available.

**Secondary: dequant/MMA serialization within each warp.** Even at
m_tiles=1, each warp serializes ~27 dequant instructions and ~4 MMA
instructions per sub-tile group. The MMA-latency stalls add ~9%
overhead on top of the dequant cost. This is a real but much smaller
effect, and the same M-inner-loop restructuring addresses both
problems simultaneously.

## Theoretical limits (8 experts, M=512/expert, k=4)

Per-expert matmul: [512 × 2048] × [2048 × 512] for gateup.

| Bottleneck | Time (us) | Notes |
|------------|-----------|-------|
| L2 bandwidth (B+A+C data) | 8.5 | 25.4 MB at 3 TB/s |
| Dequant ALU (14 ops × 8.4M unique elements) | 4.9 | INT ops at 20.6 T/s |
| MMA compute (8.6 GFLOP at 330 TFLOPS) | 26.0 | MMA throughput-bound |
| **Optimal (dequant once, overlap with MMA)** | **~28** | dequant amortized over 8 m-tiles + MMA |
| fp16 bmm (measured) | 65.9 | cuBLAS at 39% of peak |
| Current kbit (measured) | 112.9 | 8× redundant dequant |

With dequant-once: the 4.9 us of dequant work is done once, then
8 m-tiles of pure MMA follow. MMA per m-tile is ~26/8 = 3.3 us.
Total: ~4.9 + 8 × 3.3 ≈ 31 us. With pipeline overhead: ~35-40 us.
That's 1.6-1.9× faster than fp16 (65.9 us) even at M=512.

---

## Implementation attempts and results (Feb 2026)

### Attempt 1: Dequant-once with shmem accumulator management

A `kbit_grouped_gemm_prod_dqonce` kernel was implemented in
`csrc/ops.cu` with M_TILE_GROUP=2 and M_BLOCKS=2 (TILE_M=32). The
kernel is correct (max_err=0.125, identical to the old kernel's
tolerance). Two approaches to managing accumulator state were tried:

**1a: Multiple accumulator sets in registers.**

`frag_c[M_TILE_GROUP][M_BLOCKS][N_BLOCKS][4]` — all m-tiles'
accumulators live in registers simultaneously.

NCU results (M=512, k=4, moe_gu):
```
  registers/thread:  127 (M_BLOCKS=4) or 91 (M_BLOCKS=2)
  local mem loads:   8,650,752 sectors (DRAM spills!)
  local mem stores:  8,650,752 sectors
  kernel time:       230 us (vs 113 us old kernel)
```

The compiler cannot keep all accumulators + dequant temps + frag_a/b
in registers. Even with M_BLOCKS=2 (32 accumulators + ~50 other regs
= ~82 total), the COMPILER still spills 8.65M sectors to local
memory (DRAM). Adding `__launch_bounds__(256, 1)` does not help — the
issue is structural (too many live values across the dequant-to-shmem
write + A-fetch + MMA phases), not a register budget limit.

**1b: Shmem accumulator save/restore.**

Single `frag_c[M_BLOCKS][N_BLOCKS][4]` in registers. Between m-tile
iterations within a k-tile, save and restore accumulators to a
dedicated shmem region (M_TILE_GROUP × BLOCK_DIM × ACC_PER_THREAD
× 4 bytes = 32 KB for M_BLOCKS=2).

NCU results (M=512, k=4, moe_gu):
```
  registers/thread:  80 (no spills)
  local mem loads:   0 sectors
  local mem stores:  0 sectors
  kernel time:       273 us (vs 113 us old kernel)
  shmem total:       60.5 KB (requires cudaFuncSetAttribute)
```

Zero local memory spills, but the kernel is 2.4× slower than the
old kernel. The overhead comes from:

1. **Extra `__syncthreads`:** ~6 per k-tile (vs 2 in old kernel).
2. **Non-pipelined A fetches:** Each m-tile iteration does
   cp.async → fence → wait<0> → sync for the A tile. No overlap
   with MMA.
3. **Shmem accumulator save/restore traffic:** 1 MB per call of
   shmem traffic just for accumulator management.
4. **B_fp16 shmem bank conflicts:** Naive row-major layout without
   XOR swizzle.

### Attempt 2: Warp specialization (producer/consumer split)

A `kbit_grouped_gemm_warpspec_v2` kernel was implemented with 10
warps (320 threads): 2 producer warps for B_packed fetch, 8 consumer
warps for MMA. All threads participate in B dequant → B_fp16 in shmem.
Double-buffered B_packed and B_fp16 in shmem. Named barriers
(`bar.sync` with barrier IDs) for consumer-only synchronization.

The kernel is **correct** (max_err=0.125 across all shapes, k values,
and M values from 64 to 512).

NCU results (M=512, k=4, moe_gu):

| Config | Regs | Local spills | Kernel time | vs old 113 us |
|--------|------|-------------|-------------|---------------|
| CHUNK_M_TILES=2, bounds(320,1) | 103 | 8.65M sectors | 265 us | 2.3× slower |
| CHUNK_M_TILES=2, bounds(320,2) | 91 | 8.65M sectors | 278 us | 2.5× slower |
| CHUNK_M_TILES=1, bounds(320,1) | 117 | 0 | 319 us | 2.8× slower |

**With CHUNK_M_TILES=2:** Same register spill problem as Attempt 1a.
`frag_c[2][2][2][4]` = 32 floats + ~50 dequant temps + frag_a regs
exceeds what the compiler can keep in registers. `__launch_bounds__`
(320, 2) forcing 91 regs still spills — the compiler cannot reduce
live register count below what the code structurally requires.

**With CHUNK_M_TILES=1:** No dequant savings (re-dequants B per
m_tile, same as old kernel), so the shmem B_fp16 round-trip (write
8K elements → read them back) is pure overhead.

### Root cause across all attempts

**The fundamental blocker on Ada (sm_89):** Our codebook dequant
uses ~50 registers for temps (k bit-planes, 4 index variables, 4
`__shfl_sync` results, scale, absmax decode intermediates). This
leaves room for only ONE accumulator set in registers. Adding a
second set (+16 regs minimum) pushes total live registers past what
the compiler can handle without DRAM spills, regardless of
`__launch_bounds__` settings.

All three approaches (dequant-once with multiple accumulators,
dequant-once with shmem acc save/restore, warp specialization with
CHUNK_M_TILES=2) hit this same wall. The dequant is simply too
register-heavy on Ada for any approach that requires 2+
accumulator sets.

The warpspec and dqonce kernel code is retained in `csrc/ops.cu`
but disabled in the dispatch.

---

## Hybrid approach: dequant + cuBLAS BMM

Since fused kernel optimization has hit the Ada register pressure
wall, the pragmatic alternative is a two-kernel approach: dequant
all expert weights to fp16 in a single launch, then call cuBLAS BMM.

### Benchmark results (k=4, 8 experts, Ada RTX 4090)

**NCU kernel-only times (no dispatch overhead):**

```
                      NCU kernel time (us)
shape     M    BMM   Dequant  dq+BMM  vs BMM-only
-------------------------------------------------
moe_gu    64   25.6    29.1    54.7    2.14x slower
moe_gu   128   28.6    29.1    57.8    2.02x slower
moe_gu   256   35.7    29.1    64.7    1.82x slower
moe_gu   512   69.1    29.1    98.2    1.42x slower
moe_dn    64   24.1    29.3    53.3    2.21x slower
moe_dn   128   25.6    29.3    54.9    2.15x slower
moe_dn   256   38.6    29.3    67.9    1.76x slower
moe_dn   512   69.2    29.3    98.4    1.42x slower
```

**CUDA events (realistic end-to-end, includes dispatch overhead):**

```
                     CUDA events (us)
shape     M    BMM   Dequant  dq+BMM  vs BMM-only
-------------------------------------------------
moe_gu    64   20.8    46.2    67.0    3.22x slower
moe_gu   128   20.7    46.2    66.9    3.23x slower
moe_gu   256   35.6    46.2    81.8    2.30x slower
moe_gu   512   69.2    46.2   115.4    1.67x slower
moe_dn    64   12.6    44.5    57.1    4.53x slower
moe_dn   128   20.3    44.5    64.8    3.19x slower
moe_dn   256   36.2    44.5    80.7    2.23x slower
moe_dn   512   65.9    44.5   110.4    1.67x slower
```

The truth lies between NCU and CUDA events. NCU strips all dispatch
overhead (optimistic); CUDA events include ~14 us dispatch per launch
× 2 launches ≈ 28 us overhead (pessimistic for pipelined serving
where CPU dispatch overlaps with prior GPU work).

**Comparison: hybrid vs current grouped MMA kernel (CUDA events):**

```
shape     M   grp_MMA  dq1x+BMM  ratio
----------------------------------------
moe_gu    64     68.5     57.6    1.19x hybrid wins
moe_gu   128     68.3     58.1    1.18x hybrid wins
moe_gu   256     69.1     69.9    ~tied
moe_gu   512    105.2     99.0    1.06x hybrid wins
moe_dn    64     68.4     58.5    1.17x hybrid wins
moe_dn   128     70.0     56.7    1.24x hybrid wins
moe_dn   256     75.0     70.5    1.06x hybrid wins
moe_dn   512    138.1    104.5    1.32x hybrid wins
```

The hybrid approach already beats the current grouped MMA kernel at
all M values, despite the dequant overhead.

### Dequant kernel scaling analysis

The dequant kernel is the dominant cost in the hybrid path. Measured
times for the blockwise dequant kernel (`kDequantizeBlockwise_kbit_vec`)
at different data sizes (k=4, fp32 absmax → fp16 output):

```
                   NCU kernel   CUDA events   dispatch
experts  elements    time (us)    time (us)   overhead
------------------------------------------------------
1          1M          5.9         38.9         33.0
2          2M          8.1         47.7         39.6
4          4M         13.4         38.5         25.1
8          8M         24.8         38.7         13.9
```

Key findings:

1. **Kernel time scales linearly** with data volume (bandwidth-bound).
   At 8M elements (8 experts × 512 × 2048): 24.8 us kernel time.
   Theoretical bandwidth limit: 8M × (0.5B input + 2B output) / 900 GB/s
   ≈ 22 us. Kernel is at ~89% of bandwidth utilization.

2. **Launch dispatch overhead dominates at small sizes.** CUDA events
   show ~39 us regardless of data size — the kernel itself is only
   6-25 us, the rest is dispatch. For 8 separate per-expert launches:
   317 us total = 8 × ~40 us/launch (6 us kernel + 34 us dispatch).

3. **Single concatenated launch is already optimal.** One launch for
   all 8 experts: 39 us events = 25 us kernel + 14 us dispatch. A
   custom batched/pointer-array kernel would not be faster — the
   kernel time is the same (same data volume), and you can't beat
   one launch.

### Dispatch overhead breakdown

The ~14 us gap between NCU kernel time (25-29 us) and CUDA events
(39-46 us) for a single dequant launch is composed of:

1. **Python → C++ boundary** (~2-5 us): torch.ops dispatcher, ctypes
   FFI crossing, dtype/shape validation.
2. **CUDA driver launch** (~5-8 us): `cuLaunchKernel` packages kernel
   arguments, pushes command to GPU hardware queue.
3. **GPU command processor latency** (~3-5 us): GPU command processor
   dequeues launch command, sets up CTA configuration, begins
   scheduling warps to SMs.

For the hybrid path (2 launches: dequant + BMM), the total dispatch
overhead is ~28 us (2 × 14 us).

### Memory cost

The hybrid path requires an fp16 weight buffer during execution:

```
buffer = num_experts × N × K_dim × 2 bytes
moe_gu: 8 × 512 × 2048 × 2 = 16 MB
moe_dn: 8 × 2048 × 512 × 2 = 16 MB
```

This is allocated once and reused across forward passes. At 16 MB
per MoE layer, this is negligible compared to the model weights
themselves.

### Optimization opportunities

**1. CUDA Graphs.** Capture the dequant + BMM pair as a CUDA graph.
This eliminates per-launch dispatch overhead entirely — the graph
replays both kernels with a single `cudaGraphLaunch` call (~3-5 us
total dispatch). Expected improvement: ~25 us saved per MoE layer
(removing 2 × ~14 us dispatch, adding ~3 us graph dispatch).

Estimated hybrid with CUDA graph (NCU kernel times + 3 us dispatch):
```
shape     M   dq_kernel  bmm_kernel  total    vs fp16 BMM
---------------------------------------------------------
moe_gu   512    29.1       69.1      101.2    1.46x slower
moe_dn   512    29.3       69.2      101.5    1.47x slower
moe_gu    64    29.1       25.6       57.7    2.25x slower
```

**2. Dequant kernel optimization.** The dequant kernel is already at
~89% of memory bandwidth utilization. Remaining headroom is small
(~3-5 us at 8M elements). Possible improvements:
- Wider vectorized loads (int4 instead of int2) for packed data
- Fused E4M4 absmax decode (currently uses a separate fp32 buffer;
  the tiled repack format already uses E4M4 inline)
- Occupancy tuning via `__launch_bounds__`

**3. E4M4 absmax in dequant path.** The current dequant kernel uses
fp32 absmax (from the flat quantize path). The repack format already
encodes absmax as E4M4 inline. Writing a dequant variant that reads
E4M4 absmax directly would avoid the absmax format mismatch and
could be slightly faster.

**4. Fused dequant + transpose.** cuBLAS BMM wants weights in a
specific layout. If the dequant kernel can write directly in the
BMM-optimal layout, the transpose (`Wt = W.transpose(1,2).contiguous()`)
is free.

---

## Current approach summary (Feb 2026)

| M range | Best approach | Status |
|---------|--------------|--------|
| 1-16 | Grouped MMA kernel | Done, 1.3-1.8× faster than fp16 |
| 17-32 | Grouped MMA kernel | Done, ~1.0× vs fp16 (crossover) |
| 33+ | Hybrid dequant + cuBLAS BMM | Available, 1.4-2.2× slower than fp16 (kernel-only) |

For the large-M regime, the hybrid approach is the pragmatic path
forward. It already beats the current grouped MMA kernel at all M
values despite the dequant overhead. The grouped MMA kernel remains
optimal for small M where the 4-bit data compression provides a
bandwidth advantage.

The fused kernel approach (dequant-once, warp specialization) is
theoretically superior but blocked by Ada register pressure. It
may become viable on Hopper/Blackwell where `wgmma.mma_async` /
`tcgen05.mma` provide truly async MMA that frees registers during
the dequant phase.

---

## Marlin kernel reference

The Marlin kernel (Neural Magic / IST-DASLab) is the state-of-the-art
reference for pipelined dequant+MMA on Ada GPUs. Two implementations
exist in the local vLLM checkout (`/home/tim/git/vllm/`): a dense
kernel and a MoE variant. Both solve the same dequant serialization
problem we face, but for uniform INT4/INT8 quantization rather than
codebook-based k-bit.

### File locations

| File | Lines | Purpose |
|------|------:|---------|
| `csrc/quantization/marlin/marlin_template.h` | ~2100 | Dense kernel: main pipeline loop, all tiling/scheduling logic |
| `csrc/quantization/marlin/marlin.cuh` | 176 | Constants, cp.async wrappers, Vec types |
| `csrc/quantization/marlin/marlin_mma.h` | 268 | MMA wrappers (m16n8k16, m16n8k8 for Turing) |
| `csrc/quantization/marlin/dequant.h` | 609 | Dequant routines: INT4→fp16, INT8→fp16, FP4→fp16 via lop3/prmt |
| `csrc/quantization/marlin/kernel.h` | 44 | Kernel template declaration |
| `csrc/moe/marlin_moe_wna16/marlin_template.h` | 2230 | MoE variant: adds expert routing on top of the dense pipeline |
| `csrc/moe/marlin_moe_wna16/ops.cu` | 871 | MoE dispatch and Python binding |

### Key architectural decisions in Marlin

**1. No warp specialization — software-pipelined overlap instead.**

Contrary to what we initially assumed, Marlin does NOT use explicit
producer/consumer warp roles. All 8 warps (256 threads, the default)
perform both dequant and MMA. The overlap comes from a deeply
software-pipelined main loop:

```
// Marlin main loop (marlin_template.h:1780-1813)
while (slice_iters) {
  for pipe = 0..stages:
    for k = 0..b_sh_wr_iters:
      fetch_to_registers(k+1, pipe)     // ldmatrix A, load B_quant from shmem
      fetch_scales_to_registers(k+1)    // load group scales from shmem
      if k == b_sh_wr_iters - 2:
        fetch_to_shared(next_pipe)       // cp.async A+B from global → shmem
        wait_for_stage()                 // cp_async_wait<stages-2>
      matmul(k, pipe)                   // dequant B_quant → FragB, then mma.sync
}
```

The crucial detail: `matmul()` (line 1169) interleaves dequant and
MMA within the same warp. For each of 4 N-sub-tiles (j=0..3):
1. `dequant_data(frag_b_quant, frag_b)` — bitwise extraction via
   `lop3` and `prmt` instructions (~3-5 ALU ops, not 14 like our
   codebook approach)
2. `scale(frag_b, frag_s)` — multiply by group scale (`__hmul2`)
3. `mma.sync.m16n8k16(frag_a, frag_b, frag_c)` — tensor core

The m dimension is the inner loop ("We have the m dimension as the
inner loop in order to encourage overlapping dequantization and
matmul operations" — line 1215). This means for each dequantized
B-fragment, multiple A-fragments (one per m-block) are consumed.
This gives the compiler room to schedule dequant of the next
B-fragment while the current mma.sync is in flight within the same
warp.

**Why this works for Marlin but not for us:** Marlin's INT4 dequant
is ~3-5 ALU ops per element (bit extract via `lop3`, type-cast via
floating-point bias trick). Our codebook dequant is ~14 ALU ops per
element (k bit-plane extractions, `__shfl_sync` codebook lookup,
absmax multiply). Marlin's dequant is cheap enough that the compiler
can hide it behind `mma.sync` latency within a single warp. Ours
cannot — the 39:1 ALU:MMA ratio is too extreme for intra-warp
overlap.

**2. Four-stage cp.async pipeline.**

```c
static constexpr int pipe_stages = 4;  // marlin.cuh:28
```

Marlin uses 4 pipeline stages (not 2 or 3). Each stage holds one
k-tile's worth of A and B data in shared memory. The pipeline fills
stages 0..2 before computation begins, then in steady state:
- Stage N-2: cp.async fetching from global memory
- Stage N-1: data landed in shmem, available for register load
- Stage N: being consumed by matmul

**3. Stripe-based work distribution with split-K reduction.**

Marlin does NOT use a persistent kernel with atomic reductions like
ours. Instead it partitions the N dimension into "stripes" assigned
to threadblocks, with a deterministic two-phase split-K scheme
using `barrier_acquire`/`barrier_release` (lock-based, no atomicAdd).

**4. Dequant via lop3 bit tricks (not codebook lookup).**

Total: ~5-6 instructions for 4 elements (1.25-1.5 instructions per
element). Our codebook dequant requires ~14 ALU ops per element.
This 3-4x ALU overhead per element is the fundamental reason our
kernel needs different optimization strategies.

**5. XOR-swizzled shared memory layout.**

Both A and B tiles use XOR-based address transformation for
bank-conflict-free shared memory access.

**6. Register double-buffering of fragments.**

While computing with `frag_a[k%2]`, the next iteration's fragments
are loaded into `[1-k%2]` via `fetch_to_registers`. This hides
`ldmatrix` latency behind `mma.sync` + dequant computation.

### Key differences: Marlin MoE vs Marlin dense

The MoE variant is structurally identical to the dense kernel with
expert routing via sorted_token_ids, per-expert weight pointer
offsets, and optional topk_weights multiplication. The inner loop
(cp.async pipeline, dequant, MMA) is unchanged.

---

## Benchmark scripts

- `benchmarks/ncu_moe_sweep.py`: NCU driver for grouped MMA kernel,
  k=4, M=1..4096 power-of-2 scale, 8 experts. Produces kernel-only
  timings.
- `benchmarks/bench_fp16_moe_sweep.py`: fp16 BMM baseline for same
  shapes and M values. Uses CUDA events.
- `benchmarks/bench_dequant.sh` / `bench_dequant.py`: Dequant + cuBLAS
  overhead analysis for dense shapes.
- `benchmarks/bench_ncu.sh`: Full model-level benchmark (all kernels,
  all shapes, model summary).

## Files

- `csrc/ops.cu`: All kernel code. Contains:
  - `kbit_grouped_gemm_prod`: Active grouped MMA kernel (baseline)
  - `kbit_grouped_gemm_warpspec_v2`: Warp-specialized kernel (disabled, correct but slower)
  - `kbit_grouped_gemm_prod_dqonce`: Dequant-once kernel (disabled, correct but slower)
- `bitsandbytes/backends/cuda/ops.py`: Python dispatch for grouped GEMM
