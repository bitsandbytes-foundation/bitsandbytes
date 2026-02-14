# kbit GEMM Kernel: Optimization Phase 2

RTX 4090 (128 SMs, sm_89), K=4, fp16, M=32 unless stated otherwise.

**Target models:** Qwen3-Coder-Next (MoE, 70B+, hidden=2048) and
GLM-4.7-Flash (MoE, hidden=2048). Llama-scale shapes are secondary.

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

## 3. The Restructuring: Dequantize During Fetch

### 3.1 Core idea

**Move all dequantization from the compute phase to the fetch phase.**

Current architecture:
```
fetch_tile:   load raw bit-planes to shmem (cp.async)
compute_tile: read bit-planes from shmem → extract bits → codebook
              lookup → scale → pack → MMA
              ~1000 instructions per k_tile
```

Proposed architecture:
```
fetch_tile:   load bit-planes from global → registers (regular loads)
              dequantize in registers: extract bits, codebook lookup, scale
              store dequantized fp16 values to shmem (in ldmatrix layout)
              load A tile to shmem (cp.async, same as before)
compute_tile: ldmatrix A from shmem, ldmatrix B from shmem → MMA
              ~40 instructions per k_tile
```

The compute phase becomes a pure MMA loop — structurally identical to
cuBLAS. All dequantization work moves to the fetch phase where it
overlaps with the async A-tile pipeline and with tensor core execution.

### 3.2 Why this works

The dequant uses INT32 ALU (bit extraction) and FP16 ALU (shuffle,
multiply). The MMA uses tensor cores. These are **different functional
units** that execute concurrently. By separating dequant (fetch phase)
from MMA (compute phase), we let them overlap:

```
Pipeline timeline:
  fetch(tile N+1):  [---dequant B---][cp.async A]
  compute(tile N):  [---MMA loop---]
                    ↑ overlaps ↑
```

### 3.3 Instruction count comparison

Per k_tile, compute phase:

| Operation | Current | Proposed |
|-----------|--------:|---------:|
| B plane load from shmem | 56 | 0 |
| Absmax decode | 64 | 0 |
| Bit extraction | 512 | 0 |
| Codebook shuffle | 32 | 0 |
| Scale multiply | 32 | 0 |
| Pack to half2 | 16 | 0 |
| ldmatrix B | 0 | 16 |
| A load (ldmatrix) | 16 | 16 |
| A addr compute | 144 | 144 |
| MMA | 16 | 16 |
| **Total compute** | **~888** | **~192** |

The fetch phase gains ~700 instructions (dequant work), but this
overlaps with compute via the pipeline. Net effect: the critical
path shortens from ~1000 to ~200 instructions per k_tile.

**4.6x reduction in critical-path instruction count.**

### 3.4 Shared memory layout change

Current: B tile stores raw bit-plane words.
```
B_shmem: TILE_N * (TILE_K/32) * K * 4 bytes
  K=4: 128 * 2 * 4 * 4 = 4 KB per stage
```

Proposed: B tile stores dequantized fp16 values.
```
B_shmem: TILE_N * TILE_K * 2 bytes
  128 * 64 * 2 = 16 KB per stage
```

Impact on total shmem per stage:

| M_BLOCKS | Current (A+B+abs) | Proposed (A+B_deq) | Delta |
|---------:|------------------:|-------------------:|------:|
| 1 | 6.3 KB | 18.0 KB | +11.7 KB |
| 2 | 8.3 KB | 20.0 KB | +11.7 KB |
| 3 | 10.3 KB | 22.0 KB | +11.7 KB |
| 4 | 12.3 KB | 24.0 KB | +11.7 KB |

With 2 stages: max 48 KB (M_BLOCKS=4). With 3 stages: max 72 KB.
All fit within the 4090's 100 KB shared memory limit.

### 3.5 Dequant-during-fetch implementation sketch

```cpp
auto fetch_tile = [&](int stage, int kt) {
    // --- A tile: cp.async as before ---
    // (swizzled layout, 256 threads, 1 int4 per thread)
    for (int i = threadIdx.x; i < A_GROUPS; i += blockDim.x) {
        // ... swizzle address computation ...
        cp_async_cg_16(sh_a_dst, A_global_src);
    }

    // --- B tile: load, dequant, store fp16 ---
    // Total elements: TILE_N * TILE_K = 128 * 64 = 8192
    // 256 threads → 32 elements per thread
    //
    // Each thread processes a contiguous run of 32 elements
    // within one or more columns. For each quantization block
    // of 32 elements:
    //   1. Load K bit-plane words from global memory
    //   2. Load E4M4 absmax byte from global memory
    //   3. Decode absmax, extract indices, codebook lookup, scale
    //   4. Store 32 dequantized fp16 values to shmem
    //
    // The shmem layout must be ldmatrix-compatible (col-major
    // within each m8n8 sub-tile, with XOR swizzle for bank
    // conflict avoidance).
    const int tile_idx = kt * n_tiles + n_tile;
    const int b_global_base = tile_idx * B_STAGE_WORDS;
    const int abs_global_base = tile_idx * ABS_STAGE_BYTES;

    // Each thread handles 32 elements = 1 quantization block
    // Thread i handles block (threadIdx.x) within the tile
    // Block layout in the tile: col * KB_PER_TILE + k_block
    constexpr int BLOCKS_PER_TILE = TILE_N * KB_PER_TILE;  // 256
    for (int blk = threadIdx.x; blk < BLOCKS_PER_TILE; blk += blockDim.x) {
        int col = blk / KB_PER_TILE;
        int k_block = blk % KB_PER_TILE;

        // Load K bit-plane words
        unsigned int planes[K_BITS];
        int bp_base = b_global_base + col * B_COL_WORDS + k_block * K_BITS;
        for (int b = 0; b < K_BITS; b++)
            planes[b] = B_packed[bp_base + b];

        // Decode absmax
        unsigned char raw_abs = B_absmax[abs_global_base + col * KB_PER_TILE + k_block];
        scalar_t scale = Ops::from_float(decode_e4m4_absmax_branchless(raw_abs));

        // Dequantize 32 elements
        int k_base_in_tile = k_block * 32;
        for (int elem = 0; elem < 32; elem++) {
            int idx = 0;
            for (int b = 0; b < K_BITS; b++)
                idx |= ((planes[b] >> elem) & 1) << b;
            scalar_t val = Ops::mul(
                __shfl_sync(0xFFFFFFFF, cb_val, idx), scale);
            // Store to shmem in ldmatrix-compatible layout
            // (details of swizzle pattern depend on the B
            //  fragment mapping for m16n8k16)
            sh_b_deq[shmem_index(col, k_base_in_tile + elem)] = val;
        }
    }
    cp_async_fence();
};
```

The `shmem_index()` function maps (col, k) to the shared memory
address that ldmatrix expects. This requires understanding the
m16n8k16 B fragment register mapping:

- For B[k, n] in the MMA, thread t holds:
  - `frag_b[0]` = B[2*(t%4), t/4] and B[2*(t%4)+1, t/4]
  - `frag_b[1]` = B[2*(t%4)+8, t/4] and B[2*(t%4)+9, t/4]
- ldmatrix loads from a column-major layout with XOR swizzle

The compute phase becomes:

```cpp
auto compute_tile = [&](int stage) {
    scalar_t* a_ptr = sh_a(stage);
    scalar_t* b_deq_ptr = sh_b_deq(stage);

    for (int ks = 0; ks < 4; ks++) {
        // Load A fragments via ldmatrix (unchanged)
        uint32_t frag_a[M_BLOCKS][4];
        for (int mb = 0; mb < M_BLOCKS; mb++) {
            // ... same swizzle + ldmatrix as current ...
        }

        // Load B fragments via ldmatrix (NEW — no dequant)
        for (int nb = 0; nb < N_BLOCKS; nb++) {
            uint32_t frag_b[2];
            // ldmatrix from dequantized B in shmem
            // (address computation + ldmatrix.sync.aligned.m8n8.x2)

            for (int mb = 0; mb < M_BLOCKS; mb++)
                mma_m16n8k16<scalar_t>(frag_a[mb], frag_b, frag_c[mb][nb]);
        }
    }
};
```

### 3.6 The pipeline overlap question

The dequant-during-fetch adds ~700 instructions to the fetch phase.
With compute_tile at ~200 instructions, the fetch is 3.5x longer
than compute. A simple 2-stage pipeline would stall at cp_async_wait
because the next tile's fetch hasn't finished.

Solutions (choose one):

**Option A: 3-4 stage pipeline.** More stages give the fetch more
time to complete before compute needs the data. With 3 stages, the
fetch for tile N+2 overlaps with compute for tiles N and N+1. Cost:
3 * 24 KB = 72 KB shmem (fits).

**Option B: Overlap dequant with MMA in the same phase.** After
issuing MMA instructions (which queue on the tensor core pipeline),
use the ALU to dequantize the NEXT tile's B data:

```
for each k_tile:
    __syncthreads()
    // Phase 1: MMA on current tile (tensor cores)
    // Phase 2: dequant next tile's B (ALU, overlaps with MMA)
    compute_tile(cur_stage);          // issues MMA to tensor cores
    dequant_b_to_shmem(next_stage);   // ALU runs while MMA executes
    __syncthreads()
```

This is more complex but uses only 2 stages of shmem. The MMA
instructions take ~64-128 cycles to fully retire on the tensor core
pipeline. The dequant takes ~280 cycles on ALU. With both running
concurrently, the critical path is max(128, 280) = 280 cycles per
k_tile instead of 128 + 700 = 828 cycles sequentially.

**Option C: Warp specialization (Hopper-style on Ampere/Ada).** Split
the 8 warps into 2 producer warps (dequant + load) and 6 consumer
warps (MMA). Producers continuously dequantize B tiles and write to
shmem. Consumers continuously read from shmem and execute MMA. A
shared flag or barrier coordinates between them.

This provides the cleanest overlap but is the most complex to
implement. It also changes the occupancy profile: 6 MMA warps have
better compute density, while 2 producer warps handle all the ALU-
heavy dequant work. This is the pattern used by Hopper's TMA-based
kernels (where the TMA unit replaces the producer warps for data
loading, and producers only do dequant).

**Recommendation:** Start with Option A (3-stage pipeline). It is the
simplest and provides adequate overlap for the MoE shapes. If
profiling shows the fetch phase is still the bottleneck, move to
Option B. Option C is future work for maximum performance.

### 3.7 Expected performance

With the restructured kernel, compute_tile drops from ~1000 to ~200
instructions per k_tile. For 32 k_tiles:

- Dynamic instruction count: ~6,400 per thread (was ~40,000)
- With 2 warps per scheduler: ~12,800 cycles
- At 2.52 GHz: ~5 us per block
- Plus fetch overlap + barriers: ~3-5 us
- **Estimated total: 8-13 us** (was 68-78 us)

For Qwen3 gate/up (data = 5.6 MB, L2 transfer = 2.8 us):
- At 10 us: speedup = 22 us / 10 us = **2.2x vs cuBLAS**
- Overhead ratio drops from 24x to ~3.6x

For GLM4.7 shared gate/up (data = 11.1 MB, L2 transfer = 5.5 us):
- At 12 us: speedup = 27 us / 12 us = **2.3x vs cuBLAS**

These estimates assume the dequant fully overlaps with compute via
the pipeline. If overlap is only partial (e.g., 70%), times would be
~15-20 us, still 1.1-1.5x vs cuBLAS. Either way, a dramatic
improvement over the current 0.3-0.4x.

---

## 4. Additional Optimizations

These can be done before, during, or after the restructuring.

### 4.1 Fix B-tile bank conflicts (standalone fix, do first)

Add +1 padding to B-tile stride in shared memory:

```cpp
// Current: stride = KB_PER_TILE * K_BITS (= 8 for K=4)
// Fixed:   stride = KB_PER_TILE * K_BITS + 1 (= 9 for K=4)
constexpr int B_COL_STRIDE = B_COL_WORDS + 1;  // +1 padding
```

Update all shmem B addressing to use `B_COL_STRIDE` instead of
`B_COL_WORDS`. Update shmem size calculation accordingly.

After restructuring: if B stores dequantized fp16 instead of bit-
planes, the bank conflict pattern changes. The new layout needs its
own bank conflict analysis (likely requires XOR swizzle matching the
ldmatrix pattern, same as the A tile).

### 4.2 3-stage pipeline (do with or before restructuring)

Change pipeline depth from 2 to 3 stages. This improves latency
hiding for all shapes and is essential for the restructured kernel
where the fetch phase is heavier.

Shmem budget (3 stages, restructured kernel):

| M_BLOCKS | Per stage | 3 stages | Fits 100 KB? |
|---------:|----------:|---------:|:-------------|
| 1 | 18.0 KB | 54.0 KB | YES |
| 2 | 20.0 KB | 60.0 KB | YES |
| 4 | 24.0 KB | 72.0 KB | YES |

### 4.3 TILE_N=64 for small N (after restructuring)

For shapes where N/128 < num_sms (e.g., Qwen3 gate/up with 40
tiles on 128 SMs), use TILE_N=64 to double the tile count. This
improves SM utilization from 31% to 62%.

After restructuring, the compute phase is pure MMA and runs fast
regardless of tile size. The dequant in the fetch phase is
proportional to tile volume, so TILE_N=64 halves the per-tile
dequant work (good for pipeline balance).

Tradeoff: N_BLOCKS drops from 2 to 1, halving the MMA reuse of
each B dequant. But if the kernel is memory-latency-limited (not
compute-limited), this is acceptable.

### 4.4 Grouped expert GEMM for MoE routed experts

Individual MoE expert GEMMs (N=512, M=1-4) produce only 4-8 tiles
on 128 SMs. No per-kernel optimization can fix 3% SM utilization.

Solution: batch all active experts into one kernel launch. With 32
tokens * 10 experts = 320 invocations, 4 tiles each: 1280 total
tiles. All SMs fully utilized.

This is an API-level change (new `kbit_grouped_gemm` op) that reuses
the same inner loop. Do this after the single-expert kernel is fast.

---

## 5. Implementation Order

### Step 1: Fix B-tile bank conflicts
Standalone 10-line fix. Fixes 2-way bank conflict for K=4. No
restructuring needed. Benchmark to measure impact (expected ~2-5%
improvement, worth doing for correctness of the shmem layout).

### Step 2: Restructure fetch phase (dequant during fetch)
The main event. Estimated 200-300 lines of kernel code changes:
- New shmem layout for dequantized B (ldmatrix-compatible, swizzled)
- Rewrite fetch_tile to load+dequant+store instead of cp.async for B
- Rewrite compute_tile as pure ldmatrix+MMA loop
- Update shmem size calculations
- 3-stage pipeline from the start

Test plan: verify correctness on all existing test shapes, then
benchmark. Expected 5-8x improvement on MoE shapes.

### Step 3: Tune pipeline depth and tile sizes
Based on profiling the restructured kernel:
- If fetch is still the bottleneck: try 4-stage pipeline or Option B
  (overlap dequant with MMA in same phase)
- If SM utilization limits small-N shapes: add TILE_N=64 dispatch
- Profile with ncu to identify remaining bottlenecks

### Step 4: Grouped expert GEMM
Batch multiple expert GEMMs into one kernel launch. Reuses the
restructured inner loop. API: new `kbit_grouped_gemm` op.

### Step 5: Integration
Wire into LinearNbit module. Lint and PR.

---

## 6. Risk Assessment

**Shmem capacity.** The restructured kernel uses ~4x more B shmem
(fp16 vs packed). With 3 stages at M_BLOCKS=4: 72 KB. The 4090 has
100 KB. Margin is tight but sufficient. On GPUs with less shmem
(e.g., older cards with 48 KB), M_BLOCKS=4 with 3 stages would not
fit. Fallback: 2 stages (48 KB) or M_BLOCKS=2 (60 KB).

**ldmatrix for B.** The B fragment in m16n8k16 has a specific
register layout. ldmatrix.sync.aligned.m8n8.x2 can load it, but the
shmem layout must match exactly. This requires getting the swizzle
pattern right. Getting it wrong produces incorrect results that are
hard to debug. Recommendation: write a standalone test kernel that
verifies ldmatrix B loading against manual register packing before
integrating into the GEMM kernel.

**Fetch/compute balance.** If the dequant during fetch takes longer
than expected (e.g., due to global memory latency for B loads, which
are no longer cp.async), the pipeline stalls. Mitigation: the B data
fits in L2 for all target shapes, so global loads complete in ~100
cycles. The dequant ALU work (~280 cycles) dominates, and this
overlaps with the tensor core pipeline.

**Register pressure.** The fetch phase needs K temporary registers for
bit-plane words, plus the codebook register, plus the absmax. The
compute phase needs M_BLOCKS*N_BLOCKS*4 accumulator registers plus
fragment registers. Since fetch and compute alternate (not
simultaneous), the compiler can reuse registers. Expected: no
increase in register pressure vs current kernel.

---

## 7. Model Shape Reference (Target Shapes)

### Qwen3-Coder-Next (primary target)

| Layer type | K_dim | N | kbit data | Fits L2? |
|------------|------:|-----:|----------:|:---------|
| Dense gate/up | 2048 | 5120 | 5.2 MB | YES |
| Dense down | 5120 | 2048 | 5.2 MB | YES |
| Q proj | 2048 | 4096 | 4.2 MB | YES |
| KV proj | 2048 | 512 | 0.5 MB | YES |
| O proj | 4096 | 2048 | 4.2 MB | YES |
| MoE gate/up (per expert) | 2048 | 512 | 0.5 MB | YES |
| MoE down (per expert) | 512 | 2048 | 0.5 MB | YES |

### GLM-4.7-Flash (secondary target)

| Layer type | K_dim | N | kbit data | Fits L2? |
|------------|------:|-----:|----------:|:---------|
| Shared gate/up | 2048 | 10240 | 10.5 MB | YES |
| Shared down | 10240 | 2048 | 10.5 MB | YES |
| Routed gate/up | 2048 | 1536 | 1.6 MB | YES |
| Routed down | 1536 | 2048 | 1.6 MB | YES |

All target shapes fit in L2 cache (72 MB on 4090). The kernel must
be optimized for L2-resident data, not DRAM bandwidth.
