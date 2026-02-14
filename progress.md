# kbit GEMM Kernel: Progress Report and Design Decision Record

This document is an exhaustive record of all design discussions, decisions,
technical analysis, and implementation progress for the fused kbit
dequantization + GEMM kernel in bitsandbytes. It is written to be
self-contained: a developer reading this document should understand every
decision that was made, why it was made, what alternatives were considered,
and what the implications are for implementation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Source Materials Studied](#2-source-materials-studied)
3. [Interview Process and Structure](#3-interview-process-and-structure)
4. [Design Decision: Bit-Plane Format](#4-design-decision-bit-plane-format)
5. [Design Decision: Shared Memory Bank Conflicts](#5-design-decision-shared-memory-bank-conflicts)
6. [Design Decision: Atomic Ordering in Split-K](#6-design-decision-atomic-ordering-in-split-k)
7. [Design Decision: fp32 vs fp16 Accumulation](#7-design-decision-fp32-vs-fp16-accumulation)
8. [Design Decision: Pipeline Depth](#8-design-decision-pipeline-depth)
9. [Design Decision: Warp Layout and M_BLOCKS Dispatch](#9-design-decision-warp-layout-and-m_blocks-dispatch)
10. [Design Decision: Weight Layout and Repack Convention](#10-design-decision-weight-layout-and-repack-convention)
11. [Design Decision: N and K Alignment](#11-design-decision-n-and-k-alignment)
12. [Design Decision: Partial M-tile Handling](#12-design-decision-partial-m-tile-handling)
13. [Design Decision: A-tile Swizzle](#13-design-decision-a-tile-swizzle)
14. [Design Decision: C Output Write Strategy](#14-design-decision-c-output-write-strategy)
15. [Design Decision: Grid Sizing](#15-design-decision-grid-sizing)
16. [Design Decision: B-tile Load Coalescing](#16-design-decision-b-tile-load-coalescing)
17. [Design Decision: Register Pressure and Occupancy](#17-design-decision-register-pressure-and-occupancy)
18. [Design Decision: bf16 Support](#18-design-decision-bf16-support)
19. [Design Decision: Template Instantiations](#19-design-decision-template-instantiations)
20. [Design Decision: Target Architecture](#20-design-decision-target-architecture)
21. [Design Decision: Minimum Problem Size](#21-design-decision-minimum-problem-size)
22. [Design Decision: Workspace Allocation](#22-design-decision-workspace-allocation)
23. [K-Value Analysis: Why K=3 and K=5 Are Not Special](#23-k-value-analysis-why-k3-and-k5-are-not-special)
24. [Tensor Core Fragment Layout Deep Dive](#24-tensor-core-fragment-layout-deep-dive)
25. [Performance Model and Targets](#25-performance-model-and-targets)
26. [Correctness Verification Strategy](#26-correctness-verification-strategy)
27. [Implementation Pipeline: The 6-Stage Approach](#27-implementation-pipeline-the-6-stage-approach)
28. [Implementation Progress: Stage 1 Complete](#28-implementation-progress-stage-1-complete)
29. [Shared Memory Budget Analysis](#29-shared-memory-budget-analysis)
30. [Risk Register](#30-risk-register)
31. [File Locations and Worktree Setup](#31-file-locations-and-worktree-setup)
32. [How to Read the Spec (cuda-spec.md)](#32-how-to-read-the-spec)
33. [Next Steps](#33-next-steps)

---

## 1. Project Overview

### 1.1 What We Are Building

A fused CUDA kernel that combines weight dequantization and matrix multiplication
(GEMM) into a single operation. The kernel computes:

```
C[M, N] = A[M, K_dim] * W_kbit[K_dim, N]^T
```

Where:
- A is the activation matrix (fp16 or bf16), typically M=1-32 tokens
- W is the weight matrix, stored in kbit-quantized format (K=2,3,4,5 bits)
- C is the output matrix (fp16 or bf16)

### 1.2 Why This Matters

Currently, bitsandbytes has standalone quantize and dequantize kernels for kbit
quantization, but no fused GEMM. To do inference with quantized weights, you must:

1. Dequantize the entire weight matrix back to fp16
2. Call cuBLAS GEMM on the fp16 weights

This is wasteful because:
- Step 1 writes a full fp16 weight matrix to global memory
- Step 2 reads it back from global memory
- The weight data moves through memory twice

A fused kernel dequantizes weights on-the-fly in registers/shared memory and
feeds them directly to tensor core MMA instructions. The weight data moves
through memory only once, in its compressed form. For K=4 (4-bit weights),
this means reading 4x less data from global memory.

### 1.3 Target Use Case

LLM inference with small batch sizes (M=1-32). The weight matrices are large
(K_dim=4096-16384, N=4096-16384). At these batch sizes, the GEMM is
memory-bandwidth-bound, so reading 4x less weight data translates directly
to ~4x speedup.

### 1.4 Relationship to Existing Code

The kbit quantization system lives on the `feature/kbit-quantization` branch.
It implements:
- `quantize_kbit()`: quantizes a tensor using K-bit blockwise quantization
- `dequantize_kbit()`: reconstructs the tensor from packed format
- Codebook generation, E4M4 absmax encoding, bit-plane packing

The GEMM kernel builds on top of this quantization system. It uses the same
packed data format, the same codebook, and the same absmax encoding. The new
branch `feature/kbit-gemm` is based on `feature/kbit-quantization`.

---

## 2. Source Materials Studied

Before the interview, the following source files were read in full:

### 2.1 Design Document

`agents/kbit_gemm_context.md` -- the complete design context document (~1400
lines). This covers:
- Existing kbit implementation (quantize, dequantize, E4M4, bit-plane packing)
- Marlin kernel architecture as reference
- GEMM kernel design (tile sizes, thread config, register allocation)
- Weight storage format and repacking
- Inner loop: dequantization + MMA
- Persistent kernel and work distribution
- Pipeline and shared memory
- Codebook and absmax handling
- Performance analysis
- Kernel dispatch and Python integration
- File organization and build
- Error budget
- Template instantiations

### 2.2 Existing kbit CUDA Kernels

From `feature/kbit-quantization` branch, `csrc/ops.cu` lines 670-870:

**`kQuantizeBlockwise_kbit<T, K>`**: The quantize kernel. Each warp processes
one block of 32 elements. Algorithm:
1. Each lane loads one element
2. Warp-reduce absmax via `__shfl_down_sync` butterfly reduction
3. Normalize by absmax
4. Brute-force nearest-neighbor codebook search (broadcast each codebook entry
   via `__shfl_sync`, compare distances)
5. Pack via `__ballot_sync`: K bit-plane words per block

**`kDequantizeBlockwise_kbit_vec<T, K, BLOCKS_PER_WARP, ABSMAX_T>`**: The
dequantize kernel. Each warp processes 4 blocks (BLOCKS_PER_WARP=4). Algorithm:
1. Load codebook into lane registers
2. For each block: load K bit-plane words via shuffle broadcast (only lane
   `bit` does the global load, broadcasts to all), unpack index, codebook
   lookup via `__shfl_sync`, scale by absmax

**`decode_e4m4_absmax`**: Decodes E4M4 uint8 to float32 via IEEE 754 bit
manipulation. ~5 integer ALU ops. Handles normal and subnormal cases.

### 2.3 Marlin Kernel (vllm)

From `~/git/vllm/csrc/quantization/marlin/`:

**`marlin_template.h`** (~2070 lines): The main kernel template. Key sections:
- Line 271-281: Stripe partitioning explanation
- Line 916-923: Pipeline wait/fence (`cp_async_wait<stages-2>()`)
- Line 927-939: Register fetch from shared memory (double-buffered `frag_b_quant[k%2]`)
- Line 1167-1285: `matmul()` inner loop with dequant + scale + MMA
- Line 1780-1813: Main K-loop with pipeline interleaving
- Line 1839-2068: Output reduction and slice management

**`dequant.h`** (~610 lines): Dequantization functions using `lop3` (3-input
logical operation) and `prmt` (byte permutation) PTX instructions. These are
purely bitwise operations that reinterpret INT4/INT8/FP4/FP8 packed values
as FP16/BF16 by manipulating the IEEE 754 bit representation directly.

Key insight from dequant.h: Marlin's dequant is a **linear mapping** from
integer indices to floating-point values. For INT4, the 4-bit value is placed
into the mantissa/exponent fields of an FP16 number, then a bias is subtracted.
This is fundamentally different from our codebook-based approach, where the
mapping is **arbitrary** (defined by the codebook lookup table).

**`marlin_mma.h`** (~270 lines): MMA instruction wrappers. Inline PTX assembly
for `m16n8k16` instructions:
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (fp16 in, fp32 accum)
- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (bf16 in, fp32 accum)
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` (fp16 in, fp16 accum)

**`marlin.cu`** (~530 lines): Host dispatch. Priority-ordered thread configs
for small batch (m_blocks=1) and large batch (m_blocks>1). Config validation
against shared memory limits.

### 2.4 Python Functional API

From `feature/kbit-quantization` branch, `bitsandbytes/functional.py`:
- `create_normal_float_codebook(k)`: Creates 2^K reconstruction levels at
  expected values of N(0,1) within equiprobable bins, normalized to [-1,1]
- `encode_absmax_e4m4()`: float32 -> uint8 E4M4 encoding
- `decode_absmax_e4m4()`: uint8 E4M4 -> float32 decoding
- `quantize_kbit()`: High-level quantize API
- `dequantize_kbit()`: High-level dequantize API

### 2.5 Existing Test Suite

`tests/test_kbit_quantization.py` (~1400 lines): Comprehensive tests covering
all stages of the quantization implementation. This established the testing
patterns we follow for the GEMM kernel.

---

## 3. Interview Process and Structure

The design was hardened through a structured CUDA-specific technical interview
covering ~20 questions across these areas:

- Memory access patterns (bank conflicts, coalescing, cache behavior)
- Warp execution model (fragment mapping, divergence, shuffle usage)
- Synchronization and correctness (atomics, fences, race conditions)
- Precision and numerical behavior (accumulation, type conversions)
- Resource pressure (registers, shared memory, occupancy)
- Edge cases (alignment, partial tiles, min/max sizes)
- Integration (data layout, Python bindings, workspace management)
- Performance model (targets, bottlenecks, degradation modes)

Each decision below captures the question asked, the options considered,
the choice made, and the reasoning.

---

## 4. Design Decision: Bit-Plane Format

### The Question

Should the GEMM kernel use the existing bit-plane format (K uint32 words per
block of 32 elements, where word j contains bit j of all elements), or convert
to contiguous K-bit packing (where each element's K bits are adjacent)?

### The Decision

Keep bit-plane format. Do not convert to contiguous packing.

### Why This Matters

The packing format determines:
1. How data is stored in global and shared memory
2. How threads extract indices in the inner loop
3. Whether the format works uniformly across all K values

### Detailed Analysis

**Bit-plane format (chosen):** For each block of 32 elements, store K uint32
words. Word j contains bit j of all 32 elements. To reconstruct the K-bit
index for element i, extract bit i from each of the K words and OR them
together:

```
index = 0;
for (bit = 0; bit < K; bit++)
    index |= ((plane_word[bit] >> element_position) & 1) << bit;
```

This requires K shift+mask+OR operations per element, running on INT32 ALU.

**Contiguous packing (rejected):** Pack K-bit indices contiguously into uint32
words. For K=4: 8 elements per word (clean). For K=3: 10.67 elements per word
(element straddles word boundaries). For K=5: 6.4 elements per word (also
straddles).

The problem with contiguous packing for K=3 and K=5:
```
K=4: 32/4 = 8 elements per word  --> clean, no straddling
K=3: 32/3 = 10.67               --> element 10 crosses word boundary
K=5: 32/5 = 6.4                 --> element 6 crosses word boundary
```

Extracting an element that straddles a word boundary requires reading two
adjacent uint32 words, masking bits from both, and shifting/ORing them together.
The extraction code becomes K-dependent and complex.

**Why bit-planes win:**
1. **Uniform across all K**: K=2,3,4,5 all work identically. No special cases.
2. **Same memory footprint**: Both formats use K*4 bytes per 32 elements.
3. **ALU cost is hidden**: The K shift+mask+OR ops run on INT32 ALU, which is
   a different functional unit from the tensor cores. In the steady state, the
   tensor cores are executing MMA while the INT32 unit extracts indices for the
   next iteration. The cost is effectively zero.
4. **No format conversion needed**: The quantize kernel already produces
   bit-planes via `__ballot_sync`. The repack only changes tile layout.
5. **Already proven**: The standalone dequant kernel uses this format.

### Performance Impact

None measurable. The INT32 ALU operations for bit-plane extraction overlap
with tensor core MMA execution. Both formats have the same memory footprint.
The bit-plane format is strictly simpler without being slower.

---

## 5. Design Decision: Shared Memory Bank Conflicts

### The Problem

Shared memory has 32 banks, each 4 bytes wide. When two threads in the same
warp access different addresses that map to the same bank, a bank conflict
occurs and the accesses serialize (taking 2 cycles instead of 1 for a 2-way
conflict, 4 cycles for 4-way, etc.).

In the GEMM kernel's inner loop, each thread loads K bit-plane words from
shared memory for its assigned column in the B tile. The 32 threads in a warp
are organized into 8 groups of 4 threads (matching the m16n8k16 MMA fragment
layout where column = lane_id/4). The 4 threads in each group access the SAME
shared memory address (broadcast, no conflict). But the 8 groups access
DIFFERENT addresses, and these addresses must not alias to the same bank.

### The Analysis

The B-tile data in shared memory is laid out as:
```
sh_b[col * stride + k_block * K + bit_plane]
```

Where `stride = (TILE_K / 32) * K = 2 * K` words per column.

For 8 columns with stride S, bank conflict occurs when two columns i and j
satisfy `(i * S) % 32 == (j * S) % 32`, which happens when `gcd(S, 32) > 4`.

Analysis per K value (without padding):

**K=2, stride=4:** `gcd(4, 32) = 4`. Banks: {0, 4, 8, 12, 16, 20, 24, 28}.
All 8 unique. No conflict.

**K=3, stride=6:** `gcd(6, 32) = 2`. Banks: {0, 6, 12, 18, 24, 30, 4, 10}.
All 8 unique. No conflict.

**K=4, stride=8:** `gcd(8, 32) = 8`. Banks: {0, 8, 16, 24, 0, 8, 16, 24}.
Only 4 unique banks. **2-way bank conflict!** Columns 0 and 4 hit the same
bank. Columns 1 and 5 hit the same bank. Etc.

**K=5, stride=10:** `gcd(10, 32) = 2`. Banks: {0, 10, 20, 30, 8, 18, 28, 6}.
All 8 unique. No conflict.

### Why K=4 Is the Critical Case

K=4 is the most important bit-width because:
- NF4 (bitsandbytes' flagship quantization format used in QLoRA) is 4-bit
- GPTQ, AWQ, and most production quantized inference uses 4-bit
- K=2,3 degrade model quality too much for most applications
- K=5 doesn't compress enough to justify itself over FP8

So the one K value with bank conflicts is the one that matters most.

### The Fix

Add 1 word of padding per column, making `stride = 2 * K + 1`.

An odd number always has `gcd(odd, 32) = 1`, so the bank pattern never
repeats within 8 columns. Verification:

**K=2, stride=5:** Banks: {0, 5, 10, 15, 20, 25, 30, 3}. All unique.
**K=3, stride=7:** Banks: {0, 7, 14, 21, 28, 3, 10, 17}. All unique.
**K=4, stride=9:** Banks: {0, 9, 18, 27, 4, 13, 22, 31}. All unique.
**K=5, stride=11:** Banks: {0, 11, 22, 1, 12, 23, 2, 13}. All unique.

### Memory Cost

The padding adds 1 uint32 per column per K-tile in shared memory.
For TILE_N=128 columns with 4 pipeline stages: 128 * 1 * 4 = 512 words
= 2 KB extra. The GPU has 100-228 KB of shared memory. Negligible.

### Why Not Swizzle Instead

Marlin uses an XOR-based swizzle for its B-tile shared memory layout.
However, Marlin's B-tile read pattern is fundamentally different from ours.
Marlin reads packed INT4 values according to the MMA fragment layout, which
requires a specific permutation. Our read pattern is per-column (4 threads
broadcast the same address), which is inherently simpler. The +1 padding
eliminates all conflicts without the complexity of a swizzle function.

---

## 6. Design Decision: Atomic Ordering in Split-K

### The Problem

When split-K is active (multiple CUDA thread blocks contribute partial sums
to the same output tile), the partial results must be combined correctly.
The design uses:

1. First contributor: plain store to fp32 workspace in global memory
2. Subsequent contributors: `atomicAdd` to the workspace
3. Last contributor: reads workspace, converts fp32 -> fp16, writes to output C

The "last contributor" is detected via an atomic counter:
```cpp
int count = atomicAdd(&tile_counter[mn_id], 1);
if (count == num_contributors - 1) {
    // I'm the last one: convert and write output
}
```

### The Ordering Bug

Without a memory fence, the following race condition exists:

```
Block A:                              Block B:
  store partial to workspace            atomicAdd partial to workspace
  atomicAdd(&counter, 1) -> 0          atomicAdd(&counter, 1) -> 1
                                        // B sees count == 1 (last!)
                                        // B reads workspace
                                        // BUT: Block A's store may not
                                        // be visible to Block B yet!
```

`atomicAdd` guarantees atomicity of the individual operation (the counter
increment is correct), but it does NOT guarantee that other writes to
different addresses are visible. Block B could see the incremented counter
but read stale (zero or partial) workspace values.

### The Fix

Insert `__threadfence()` between the workspace write and the counter increment:

```cpp
// Write partial results (store or atomicAdd)
write_to_workspace(frag_c, workspace, ...);
__threadfence();  // ensures all prior writes are globally visible
int count = atomicAdd(&tile_counter[mn_id], 1);
```

`__threadfence()` guarantees that all writes from this thread block that
occurred before the fence are visible to all other thread blocks. This
means when Block B reads the counter and decides it's the last contributor,
it is guaranteed to see Block A's workspace writes.

### Why Plain Store Is Safe for the First Contributor

The first contributor uses a plain store (not `atomicAdd`) to write its
partial result. This works because:

1. The first contributor is the only writer to that workspace location at
   that time. There's no concurrent writer to race with.
2. The `__threadfence()` after the store ensures the store is globally
   visible before the counter is incremented.
3. Subsequent contributors see counter >= 1, so they know the workspace
   has been initialized and use `atomicAdd` to add their contribution.

Using `atomicExch` instead of a plain store would also work but adds
unnecessary overhead. The plain store is correct given the fence.

### Performance Cost

`__threadfence()` costs ~50-100 cycles. It executes once per output tile
per thread block. A thread block processes many K-tiles (hundreds to thousands
of cycles of MMA work) before writing output. The fence cost is negligible --
less than 0.1% of total kernel time.

---

## 7. Design Decision: fp32 vs fp16 Accumulation

### The Question

The `m16n8k16` MMA instruction has two variants:
- fp32 accumulation: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- fp16 accumulation: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`

Should we use fp32 or fp16 accumulation?

### The Decision

Use fp32 accumulation exclusively. Convert to fp16/bf16 only at the final
output stage.

### Throughput Analysis by Architecture

**Ampere (A100, sm_80):** Both variants have the SAME throughput -- 256 FMA
ops per warp per cycle. The tensor cores do not run faster with fp16
accumulation. The only difference is accumulator register size (fp16 uses
half the registers).

**Ada Lovelace (4090, sm_89):** Same as Ampere. No throughput difference.

**Hopper (H100, sm_90):** fp16 accumulation can achieve up to 2x throughput
in some configurations due to different datapath handling.

### Why fp32 Matters Even for Quantized Weights

One might think: "The weights are already K=4 quantized with ~6% error per
element. Why bother with fp32 accumulation when the input is already lossy?"

The answer is that quantization error and accumulation error are fundamentally
different:

**Quantization error** is per-element and bounded. Each weight has at most
~6% error from its true value. This error is random-like and partially
cancels across the reduction dimension.

**Accumulation error** is systematic and grows with the reduction length.
When adding thousands of fp16 products (K_dim=4096+):
- fp16 has ~10-bit mantissa (1024 representable values per exponent range)
- After ~1000 additions, small products are rounded away entirely because
  they fall below the ULP of the running sum
- This creates a systematic bias that does NOT cancel

With fp32 accumulation:
- 23-bit mantissa (8 million representable values per exponent range)
- Can sum millions of terms without significant precision loss
- The final fp32->fp16 conversion loses precision only once

DeepSeek demonstrated this effect in production: switching from fp16 to fp32
accumulation in their MoE models improved quality measurably, even with
already-quantized weights.

### For Our Target Use Case

| Batch size | Bottleneck     | fp16 accum benefit | fp32 accum cost    |
|-----------|----------------|-------------------|--------------------|
| M <= 32   | Memory-bound   | None (MMA isn't    | Free (not the      |
|           |                | the bottleneck)    | bottleneck)        |
| M >= 128  | Compute-bound  | Up to 2x on Hopper | Half peak FLOPS    |
|           |                |                    | on Hopper          |

For M <= 32 (the primary use case): fp32 accumulation is completely free
because the kernel is waiting on memory bandwidth, not tensor core throughput.

For M >= 128 (rare for inference): the quality tradeoff is unacceptable.
Users running quantized models are already precision-sensitive; compounding
quantization error with accumulation error is a bad tradeoff.

---

## 8. Design Decision: Pipeline Depth

### The Question

How many pipeline stages should the kernel use for the cp.async global-to-
shared-memory pipeline?

### The Decision

4 stages.

### How the Pipeline Works

The `cp.async` instruction initiates an asynchronous copy from global memory
to shared memory. The GPU hardware copies data in the background while the
SM executes other instructions. Multiple copies can be in-flight
simultaneously (pipelined).

A "stage" is one slot in a circular buffer in shared memory. With N stages,
you can have N-1 copies in-flight while processing the Nth:

```
4-stage pipeline (stages 0,1,2,3):
  Cycle 0: Start copy for tiles 0,1,2
  Cycle T: Process tile 0, start copy for tile 3
  Cycle 2T: Process tile 1, start copy for tile 4
  ...
```

The `cp_async_wait<N>()` instruction stalls until at most N async copies
remain outstanding. With `cp_async_wait<stages-2>()`, we wait until only
`stages-2` copies are in-flight, meaning the current stage's data is ready.

### Why 4 Stages

On Ampere/Ada, `cp.async` latency is approximately 200-400 cycles for a
global memory load (depends on cache hit, memory controller load, etc.).

A single K-tile of compute (dequant + 4 MMA sub-tiles) takes roughly
100-200 cycles.

With 2-stage double buffering: the pipeline hides 1 K-tile of latency
(~100-200 cycles). If the global load takes 300+ cycles, the pipeline stalls
waiting for data.

With 4-stage buffering: the pipeline hides 3 K-tiles of latency
(~300-600 cycles). This comfortably covers global memory latency even in
worst-case scenarios (cache miss, memory contention).

### Shared Memory Cost

Per stage (TILE_M=64, TILE_N=128, K=5 worst case):
- A tile: 64 * 64 * 2 = 8,192 bytes
- B tile: 128 * 2 * 5 * 4 + padding = ~5,632 bytes
- Absmax: 128 * 2 = 256 bytes
- Total: ~14,080 bytes

4 stages: ~56 KB. Available: 100 KB (4090), 164 KB (A100), 228 KB (H100).
Fits comfortably on all target GPUs.

### Pipeline Management (No Warp Specialization)

On Ampere/Ada, warp specialization is not used. All 8 warps cooperate on
both loading and computing:

```cpp
// Pre-fill 3 stages ahead
for (int s = 0; s < 3; s++)
    fetch_tile(stage=s, k_tile=s);
cp_async_fence();

for (int kt = 0; kt < num_k_tiles; kt++) {
    cp_async_wait<2>();      // wait for current stage
    __syncthreads();

    if (kt + 3 < num_k_tiles)
        fetch_tile(stage=(kt+3)%4, k_tile=kt+3);  // prefetch
    cp_async_fence();

    process_k_tile(stage=kt%4, frag_c, cb_h);  // dequant + MMA
}
cp_async_wait<0>();  // drain
```

In `fetch_tile`, each of the 256 threads loads a fraction of the A and B
tiles. For A: each thread loads ~32 bytes (8 KB / 256 threads). For B:
each thread loads ~16-20 bytes. The loads are distributed via a strided loop.

Warp specialization (dedicated producer/consumer warps) is a Hopper-specific
optimization using TMA. It is listed as a future consideration, not part of
the initial implementation.

---

## 9. Design Decision: Warp Layout and M_BLOCKS Dispatch

### Thread Block Structure

256 threads = 8 warps. The warps are arranged in a 2D grid to partition the
output tile:

- `warps_m` warps along the M dimension
- `warps_n` warps along the N dimension
- `warps_m * warps_n = 8`

Each warp handles a sub-tile of size `(M_BLOCKS_per_warp * 16) x (N_BLOCKS_per_warp * 8)`.

### Adaptive Layout Based on M_BLOCKS

The warp layout adapts to the M dimension:

**M_BLOCKS=1 (TILE_M=16, M=1-16):** Layout is 1x8. All 8 warps along N.
Each warp handles 16 rows x 16 columns. This is the primary use case for
LLM inference with small batch sizes.

**M_BLOCKS=2 (TILE_M=32, M=17-32):** Layout is 2x4. 2 warps along M,
4 along N. Each warp handles 16 rows x 32 columns.

**M_BLOCKS=3 (TILE_M=48, M=33-48):** Layout is 2x4 (with 3 M-blocks split
as 2+1 or handled via different warp-to-M-block mapping). Edge case, rarely
used.

**M_BLOCKS=4 (TILE_M=64, M=49+):** Layout is 2x4. Each warp handles
32 rows x 32 columns. This is the Marlin-standard layout.

### Dispatch Logic

The host-side dispatch function selects M_BLOCKS before launching the kernel:

```cpp
int m_blocks;
if (M <= 16)      m_blocks = 1;
else if (M <= 32) m_blocks = 2;
else if (M <= 48) m_blocks = 3;
else              m_blocks = 4;
```

This is a compile-time constant within each kernel instantiation (it's a
template parameter), so the warp layout is fixed for the duration of the
kernel execution. No runtime branches in the inner loop.

### Why This Is Not a Fundamental Architecture Decision

The warp layout is a small configuration choice that affects two things:
1. The mapping of `warp_id` to `(warp_m, warp_n)` coordinates
2. The M_BLOCKS and N_BLOCKS counts per warp

Changing the layout means changing a few lines of index math, not the kernel
structure. The inner loop (dequant + MMA) is identical regardless of layout.

### Data Reuse Implications

With 2x4 layout (M_BLOCKS >= 2): each dequantized B fragment (FragB) is
reused across 2 M-blocks. The codebook lookup + scale multiply cost is
amortized. This favors larger M.

With 1x8 layout (M_BLOCKS = 1): each FragB is used only once. But there are
twice as many N-blocks per warp, so fewer warps compete for the same B data
in shared memory. This favors small M / large N.

For the target use case (M <= 32), both layouts work well. The difference
is small enough that profiling on real workloads should guide the final choice.

---

## 10. Design Decision: Weight Layout and Repack Convention

### The Problem

PyTorch Linear layers store weights as `[out_features, in_features] = [N, K_dim]`.
The GEMM computes `C[M, N] = A[M, K_dim] * W[N, K_dim]^T`.

The quantize kernel flattens the weight to 1D and quantizes sequentially.
The flat index for element (n, k) in a [N, K_dim] matrix is `n * K_dim + k`.

The GEMM kernel tiles along K_dim and N. To make tiles contiguous in memory,
the repack kernel must understand the weight layout.

### The Decision

The repack kernel accepts PyTorch's native `[N, K_dim]` layout. The transpose
is handled internally via index math. Users do not need to call `.t().contiguous()`.

### How It Works

In the repack kernel, when mapping element (n, k) to its flat block:
```python
flat_index = n * K_dim + k       # [N, K_dim] row-major
block_id = flat_index // 32
```

This is different from `[K_dim, N]` row-major where it would be `k * N + n`.
The repack kernel reads from the flat layout using the [N, K_dim] indexing
and writes to the tiled layout organized by (k_tile, n_tile) positions.

### Why This Matters

Getting the index math wrong silently produces a transposed GEMM -- the output
has the right shape but wrong values. This is one of the highest-risk bugs in
the implementation (see Risk Register, Section 30).

### User-Facing API

```python
# User quantizes their weight (PyTorch native layout)
packed, absmax, codebook = quantize_kbit(W)  # W is [N, K_dim]

# User repacks for GEMM (no transpose needed)
packed_tiled, absmax_tiled = repack_for_gemm(packed, absmax, K_dim, N, k)

# User runs GEMM
C = kbit_gemm(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)
```

---

## 11. Design Decision: N and K Alignment

### N Alignment

**Decision:** Require N to be divisible by TILE_N (128).

**Rationale:** All common LLM weight matrices have N dimensions that are
multiples of 128 (e.g., 4096, 8192, 11008, 14336). Supporting arbitrary N
would require:
- Partial N-tile masking in the kernel
- Branch divergence at tile boundaries
- Padding logic in shared memory loads
- More complex output write masking

None of this complexity is needed for real workloads.

**If N is not a multiple of 128:** Pad the weight matrix at the Python level
before quantization. The padded columns have zero weights and contribute
nothing to the output. The Python API trims the output to the original N.

### K_dim Alignment

**K_dim must be divisible by 32** (the quantization blocksize). This is
inherent to the quantization system and not a new constraint.

**K_dim % TILE_K (64):** When K_dim is not divisible by 64, the final K-tile
is partial (only 32 elements instead of 64). This is handled by a separate
code path that does bounds checking on the last K-tile.

The separate code path is a runtime branch: `if (kt == last_k_tile && is_partial)`.
Branch prediction almost always predicts "not partial" (correct for all but the
last iteration). The misprediction penalty is negligible -- one pipeline stall
per K dimension traversal per block.

For typical LLM dimensions (K_dim = 4096, 8192, 11008, etc.), K_dim is always
a multiple of 64 and this code path is never executed.

---

## 12. Design Decision: Partial M-tile Handling

### The Problem

When M is not divisible by TILE_M (e.g., M=100, TILE_M=64), the last M-tile
has fewer valid rows than TILE_M. Loading out-of-bounds rows from A reads
garbage or segfaults. Writing out-of-bounds rows to C corrupts memory.

### The Decision

Use predicated `cp.async` for A loads and masked writes for C output.

### How It Works

**A loads:** The `cp.async` instruction supports a predicate. When the
predicate is false, the copy writes zeros to shared memory instead of reading
from global memory. Threads compute `row < M` and use this as the predicate.
Out-of-bounds rows get zero-filled in shared memory.

```cpp
bool pred = (my_row < M);
if (pred)
    cp_async4(&sh_a[offset], &A_global[a_offset]);
else
    // Zero-fill the shared memory slot
    sh_a[offset] = 0;
```

**MMA execution:** The tensor core MMA operates on whatever data is in the
fragments. For zero-filled rows, it computes `0 * B = 0`. These zero outputs
are in the right positions and simply need to be discarded.

**C writes:** Threads check `row < M` before writing output. Invalid rows
are skipped. This is a simple predicated store.

### Why Not Pad at the Python Level

Padding M at the Python level would also work (allocate A with padded rows,
allocate C with padded rows, trim after). But this adds memory overhead and
API complexity. The kernel-side handling is straightforward and the predicate
evaluation is in the epilogue, not the inner loop.

---

## 13. Design Decision: A-tile Swizzle

### The Problem

The A tile is loaded into shared memory and then read via `ldmatrix`
instructions to fill MMA A-fragments. The `ldmatrix` instruction reads from
shared memory using a specific thread-to-address mapping that, with a naive
row-major layout, causes severe bank conflicts (up to 8-way).

### The Decision

Use an XOR-based swizzle, preferably adopting Marlin's pattern if it's not
too bloated. If Marlin's pattern is overly complex, implement a standard
`addr ^= (addr >> 2) & 0x7` swizzle.

### How Swizzling Works

When storing data to shared memory, the write address is XORed with a
function of the row index:

```cpp
// Write A[row][col] to shared memory
int swizzled_col = col ^ ((row % 8) * some_pattern);
sh_a[row * stride + swizzled_col] = A_global[row * K_dim + col];
```

When reading via `ldmatrix`, the same swizzle is applied to the read address.
The swizzle ensures that threads in a warp, which follow the `ldmatrix` access
pattern, hit different banks.

### Why A-Swizzle Is Needed but B-Swizzle Is Not

**A tile:** Read via `ldmatrix`, which has a specific thread-to-address mapping
dictated by the hardware. This mapping creates bank conflicts with naive layout.
Swizzle is required.

**B tile:** Read with a per-column broadcast pattern (4 threads read the same
address for their column). The +1 padding eliminates bank conflicts for all K
values. No swizzle needed.

---

## 14. Design Decision: C Output Write Strategy

### The Problem

When the kernel finishes accumulating a tile of C (in fp32 FragC registers),
it must write the results to global memory (as fp16/bf16). The FragC layout
follows the MMA fragment mapping, where each thread holds results for
scattered positions (2 rows, 1 column per MMA sub-tile). Direct register-to-
global-memory writes would be uncoalesced -- threads in a warp would write to
different rows, hitting different cache lines.

### The Decision

Stage output through shared memory for coalesced writes.

### How It Works

1. Each warp writes its FragC values to shared memory in row-major order.
   The shared memory is reused from the pipeline (which is no longer needed
   during the output phase).
2. A `__syncthreads()` ensures all writes complete.
3. Threads then read from shared memory in a pattern that gives coalesced
   global writes (consecutive threads read consecutive addresses, then write
   to consecutive global addresses in the same row of C).

### For Split-K

When split-K is active and the block writes to the fp32 workspace (not the
final fp16 output), the writes can be direct (no staging) because:
1. The workspace is temporary and fp32
2. The write pattern doesn't need to be perfectly coalesced for a one-time
   write that's not on the critical path
3. The final fp32->fp16 conversion (done by the last contributor) goes
   through the staging path

---

## 15. Design Decision: Grid Sizing

### The Decision

Grid = `min(num_SMs, total_work_items)`.

### Why

The persistent kernel launches a fixed number of blocks that loop over work
items. Launching exactly `num_SMs` blocks is the standard approach, but when
`total_work < num_SMs`, excess blocks enter the loop, find no work, and exit
immediately. This wastes a few microseconds of launch overhead but is not
measurable in practice.

Using `min(num_SMs, total_work)` avoids launching blocks that will immediately
exit. It's slightly cleaner but functionally equivalent.

### Why Not Occupancy-Aware Launch

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` could be used to determine
how many blocks actually fit per SM (given register and shared memory usage).
For our kernel, this returns 1 block per SM (due to high register usage).
So the occupancy-aware grid size equals `num_SMs`, which is what we already
use. No benefit from the extra API call.

---

## 16. Design Decision: B-tile Load Coalescing

### The Decision

Simple linear thread-to-word mapping with a strided loop for `cp.async` loads.

### How It Works

```cpp
int total_int4s = TILE_N * (TILE_K / 32) * K_BITS / 4;  // compile-time
for (int i = threadIdx.x; i < total_int4s; i += blockDim.x)
    cp_async4(&sh_b_int4[i], &B_global[b_offset + i]);
```

Each thread loads one or more 16-byte chunks. Consecutive threads load
consecutive chunks -> coalesced access.

### Why Different K Values Don't Cause Problems

The B tile size varies with K:
```
K=2:  512 words = 2 KB  -> 128 int4 loads -> 128/256 threads = 0.5 per thread
K=3:  768 words = 3 KB  -> 192 int4 loads -> 0.75 per thread
K=4: 1024 words = 4 KB  -> 256 int4 loads -> exactly 1 per thread
K=5: 1280 words = 5 KB  -> 320 int4 loads -> 1.25 per thread
```

The strided loop handles all cases naturally:
- K=2: 128 threads active (first 128), 128 idle. Still coalesced.
- K=3: 192 threads active, 64 idle.
- K=4: All 256 threads load exactly once. Perfect 1:1.
- K=5: All 256 threads load once, then 64 threads load a second time.

The B tile is small relative to the A tile (2-5 KB vs 8 KB), so even partial
utilization on the B load doesn't affect overall performance -- A loading
dominates bandwidth.

### Alignment

All tile sizes are multiples of 16 bytes:
- K=2: 512 * 4 = 2048 bytes. 2048/16 = 128. OK.
- K=3: 768 * 4 = 3072 bytes. 3072/16 = 192. OK.
- K=4: 1024 * 4 = 4096 bytes. 4096/16 = 256. OK.
- K=5: 1280 * 4 = 5120 bytes. 5120/16 = 320. OK.

So `cp_async4` (16-byte copy) alignment is never an issue.

### Why No B-tile Swizzle

Marlin swizzles its B-tile shared memory writes to align with its fragment
read pattern. Our B-tile read pattern is fundamentally different (per-column
broadcast), and the +1 padding already eliminates bank conflicts. No swizzle
needed.

---

## 17. Design Decision: Register Pressure and Occupancy

### Register Count Estimate

Per thread (K=4, M_BLOCKS=4, worst case):

**FragC accumulators:** This is the largest consumer. Each MMA position
produces 4 float values per thread. With M_BLOCKS=4 and N_BLOCKS=4:
- 4 M-blocks * 4 N-blocks * 2 sub-tiles per N-block = 32 MMA positions
- 32 * 4 floats = 128 floats = 128 registers

**FragA (double-buffered):** For the A-side of MMA, each thread holds
4 registers per M-block per pipeline buffer:
- 4 M-blocks * 2 buffers * 4 regs = 32 registers

**Other:** Bit-plane temporaries (K=4 uint32), codebook (1 half), absmax
(2 values), loop variables, address calculations: ~20 registers.

**Total:** ~180 registers per thread.

### Occupancy

With 180 registers per thread and 256 threads per block:
- Registers per block: 180 * 256 = 46,080
- A100 has 65,536 registers per SM
- 65,536 / 46,080 = 1.42 -> 1 block per SM

This gives occupancy = 256 threads / 2048 max threads per SM = 12.5%.

### Why 1 Block/SM Is Fine

This seems low, but it's standard for high-performance GEMM kernels. Marlin
also runs at 1 block per SM. The reason low occupancy works:

1. **The kernel is compute-bound for large M.** Tensor core MMA keeps the
   functional units busy. Occupancy matters more for memory-bound kernels
   where you need thread-level parallelism to hide memory latency.

2. **The pipeline hides latency.** The 4-stage cp.async pipeline provides
   instruction-level parallelism that substitutes for thread-level parallelism.
   While one stage is being processed, the next is being loaded.

3. **For small M (memory-bound regime):** Occupancy doesn't help because the
   bottleneck is memory bandwidth, not thread scheduling. More threads would
   just increase contention on the memory bus.

### Spill Risk

If the compiler requires more than 255 registers per thread, it spills to
local memory (off-chip DRAM). This is catastrophic for performance. The
estimated 180 registers is below the limit, but compiler optimizations
(or failure to optimize) can change this.

**Mitigation:** Check register usage with `--ptxas-options=-v` during
compilation. If spilling occurs, consider:
- Capping M_BLOCKS at 3 (reduces FragC from 128 to 96 registers)
- Reducing N_BLOCKS by using a different warp layout
- Using `__launch_bounds__` to hint the compiler

---

## 18. Design Decision: bf16 Support

### The Decision

Support both fp16 and bf16 from day one, templated on `scalar_t`.

### What Changes for bf16

1. **MMA instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
   instead of the fp16 variant. Different PTX assembly, same performance.

2. **Codebook storage:** The codebook is loaded as float32 and converted to
   `scalar_t` (half or nv_bfloat16) at kernel start. The conversion function
   changes: `__float2half()` vs `__float2bfloat16()`.

3. **Scale multiply:** `__hmul()` works for both half and nv_bfloat16 (both
   implement the `*` operator). No code change needed.

4. **Output conversion:** The fp32 FragC accumulator is converted to `scalar_t`
   at the output stage. `__float2half()` vs `__float2bfloat16()`.

5. **ldmatrix:** Works the same for fp16 and bf16 (both are 16-bit types,
   same memory layout).

### Why bf16 Matters

Most modern LLMs (LLaMA, Mistral, Qwen, etc.) use bf16 for training and
inference. The activations (A matrix) are in bf16. If the kernel only supports
fp16, users must convert A to fp16 before calling the GEMM, which adds
overhead and loses the dynamic range advantage of bf16.

### Template Impact

Adding bf16 doubles the template instantiations:
- Before: 4 K values * 4 M_BLOCKS = 16 variants
- After: 4 K values * 4 M_BLOCKS * 2 dtypes = 32 variants

This is still manageable (Marlin has 100+ variants).

---

## 19. Design Decision: Template Instantiations

### Template Parameters

```cpp
template <int K_BITS, int M_BLOCKS, typename scalar_t>
__global__ void kbit_gemm_kernel(...);
```

### Total Count

- K_BITS: 2, 3, 4, 5 (4 values)
- M_BLOCKS: 1, 2, 3, 4 (4 values)
- scalar_t: half, nv_bfloat16 (2 values)

GEMM kernel: 4 * 4 * 2 = 32 variants
Repack kernel: 4 * 2 = 8 variants (templated on K_BITS and tile sizes)
Total: 40 variants

### Source Code vs Binary Code

The kernel is written ONCE as a templated function (~500-1000 lines). The
compiler generates 40 specialized versions of machine code. The source code
is not duplicated.

### Compile Time

Each variant takes NVCC roughly 10-30 seconds to compile and optimize.
Total: ~5-15 minutes for a full build. This is acceptable for a CUDA library.

For faster iteration during development, you can instantiate only the variants
you're testing (e.g., just K=4, M_BLOCKS=1, half) and add the rest later.

### Dispatch

The host-side dispatch function selects the right variant based on runtime
parameters:

```cpp
void kbit_gemm_dispatch(int K_bits, int m_blocks, bool is_bf16, ...) {
    if (is_bf16) {
        switch (K_bits) {
            case 2: switch (m_blocks) { case 1: launch<2,1,nv_bfloat16>(...); break; ... }
            ...
        }
    } else {
        switch (K_bits) {
            case 2: switch (m_blocks) { case 1: launch<2,1,half>(...); break; ... }
            ...
        }
    }
}
```

This dispatch adds zero overhead to the kernel itself -- it's a host-side
decision made before the kernel launch.

---

## 20. Design Decision: Target Architecture

### The Decision

sm_80+ (Ampere and newer). No Volta (sm_70) or Turing (sm_75) support.

### Why

The kernel relies on `cp.async` (async global-to-shared memory copy), which
requires sm_80+. Without `cp.async`, the kernel would need a completely
different loading strategy (synchronous loads with explicit double-buffering
via `__syncthreads`), which is significantly less efficient.

The target hardware includes:
- **A100** (sm_80): Datacenter Ampere. 164 KB shared memory, 108 SMs.
- **4090** (sm_89): Consumer Ada Lovelace. 100 KB shared memory, 128 SMs.
  This is the developer's actual hardware.
- **H100** (sm_90): Datacenter Hopper. 228 KB shared memory, 132 SMs.

### Future Hopper Optimizations

Hopper (sm_90) supports TMA (Tensor Memory Accelerator) and warp
specialization. These could provide significant speedups:
- TMA: hardware-managed tile loading, freeing warps for compute
- Warp specialization: dedicated producer warps for loading, consumer warps
  for compute, with explicit producer-consumer synchronization

These are listed as future optimizations, not part of the initial implementation.

---

## 21. Design Decision: Minimum Problem Size

### The Decision

Always use the fused kernel. No fallback to dequant + cuBLAS for small problems.

### Why

For tiny problems (e.g., M=1, N=128, K=64), the fused kernel has overhead:
- Kernel launch latency (~5 us)
- Pipeline fill/drain (~3 K-tiles worth)
- Persistent loop setup

But the actual computation also completes in microseconds. Optimizing the
fallback threshold adds code complexity for a case that doesn't matter in
practice. Real LLM inference uses K_dim >= 4096, where the fused kernel
always has enough work.

The kernel is never WRONG for small problems, just potentially slightly
slower than cuBLAS. Since the absolute time is microseconds either way,
the simplicity of "always fused" outweighs the micro-optimization.

---

## 22. Design Decision: Workspace Allocation

### What Needs Allocating

When split-K is active:
1. **fp32 workspace:** `[M, N]` float32 tensor for partial sum accumulation
2. **Tile counters:** `[m_tiles * n_tiles]` int32 tensor for last-contributor detection

### Allocation Strategy

Use PyTorch's caching allocator. Allocate via `torch.empty()` in the Python
CUDA backend each GEMM call. PyTorch's allocator caches freed blocks and
reuses them for subsequent allocations of the same size, so the actual
`cudaMalloc` only happens once. Subsequent calls reuse cached memory.

### Per-Call Requirements

The tile counters must be zeroed before each GEMM call with split-K:
```python
tile_counters.zero_()  # or cudaMemsetAsync on the C side
```

This is an async memset (~1 us for a few KB) that overlaps with kernel launch
overhead. Negligible.

### When Split-K Is Not Needed

For `m_tiles * n_tiles >= num_SMs`, no split-K is needed. Each block owns
complete output tiles and writes fp16 directly to C. No workspace, no
atomics, no counters. This is the common case for large M.

---

## 23. K-Value Analysis: Why K=3 and K=5 Are Not Special

### The Concern

K=3 and K=5 are odd numbers that don't divide 32 evenly. The concern was
whether they require special handling anywhere in the kernel.

### The Analysis

**Bit-plane packing:** K uint32 words per block of 32 elements, regardless
of whether K is even or odd. The `__ballot_sync` operation produces one word
per bit. K=3 -> 3 words. K=5 -> 5 words. No boundary crossing, no special
cases.

**Index extraction:** K shift+mask+OR operations per element. The `#pragma
unroll` loop unrolls to 2, 3, 4, or 5 operations respectively. All run on
INT32 ALU. No special cases.

**Codebook lookup:** `__shfl_sync` with 2^K entries. For K=2: 4 entries
(lanes 0-3 hold values, lanes 4-31 hold 0). For K=5: 32 entries (all lanes
hold values). The shuffle instruction handles all cases -- it reads from
lane `idx % 32`, which is correct for all K <= 5.

**B-tile size:** Varies with K (2-5 KB per stage). The strided loop for
cp.async handles all sizes. No special cases.

**Bank conflicts:** Only K=4 has conflicts (with the unpadded layout). With
the +1 padding fix, all K values are conflict-free. The padding fix works
because it makes the stride odd, which is coprime with 32 for ANY K.

**Absmax:** Independent of K. Always 1 byte (E4M4) per block of 32 elements.

### What Actually Varies

| Aspect | K=2 | K=3 | K=4 | K=5 |
|--------|-----|-----|-----|-----|
| B-tile size/stage | 2 KB | 3 KB | 4 KB | 5 KB |
| Dequant ALU ops/elem | 2 | 3 | 4 | 5 |
| Codebook entries | 4 | 8 | 16 | 32 |
| Compression ratio | 7.1x | 4.9x | 3.8x | 3.0x |

The only K-specific code is the template parameter `K_BITS` that controls
the `#pragma unroll` count. Everything else is K-agnostic.

### Why Contiguous Packing WOULD Break for K=3, K=5

If we had chosen contiguous packing instead of bit-planes:
```
K=4: 32/4 = 8 elements per uint32   -> clean
K=3: 32/3 = 10.67 per uint32        -> element straddles word boundary!
K=5: 32/5 = 6.4 per uint32          -> element straddles word boundary!
```

Contiguous packing requires different extraction code for each K value,
with K=3 and K=5 needing cross-word masking. Bit-plane format avoids this.

---

## 24. Tensor Core Fragment Layout Deep Dive

### The m16n8k16 MMA Instruction

This is the fundamental compute primitive. It computes a 16x8 output tile
from 16x16 (A) and 16x8 (B) input tiles, accumulating into fp32.

### B-Fragment Thread Mapping

For the B matrix (k=16 rows, n=8 columns), each thread (lane 0-31) owns
4 elements organized as 2 half2 values:

```
b[0] (half2): rows {2*(lane%4), 2*(lane%4)+1}, column = lane/4
b[1] (half2): rows {2*(lane%4)+8, 2*(lane%4)+9}, column = lane/4
```

The critical property: **all 4 elements a thread needs are in the SAME column.**
Threads 0-3 all access column 0. Threads 4-7 all access column 1. Etc.

This means:
- The column index is `lane_id / 4` (integer division)
- 4 threads share each column -> 4-way broadcast on shared memory reads
- 8 distinct columns per warp -> 8 different shared memory addresses

### How N-Blocks Extend This

Each MMA covers 8 columns. To cover a 32-column warp sub-tile, the warp
iterates over 4 N-blocks. For N-block `nb`:

```
tile_column = warp_n_offset + nb * 8 + lane_id / 4
```

The `lane_id / 4` value is fixed for a given thread. Only the base offset
(`warp_n_offset + nb * 8`) changes per N-block. The shared memory address
shifts by 8 columns worth of data each iteration.

### Row Mapping for Dequantization

Within a column, the 4 elements a thread needs are at rows:
```
row_base = 2 * (lane_id % 4)
rows = {row_base, row_base+1, row_base+8, row_base+9}
```

For lane 0: rows {0, 1, 8, 9}
For lane 1: rows {2, 3, 10, 11}
For lane 2: rows {4, 5, 12, 13}
For lane 3: rows {6, 7, 14, 15}

These rows are positions within a block of 32 elements (one bit-plane word).
To extract the index for row `r`, the thread reads bit `r` from each of the
K bit-plane words.

### Putting It All Together

For one N-block, one k-sub-tile:
1. Compute column index: `col = warp_n_offset + nb * 8 + lane_id / 4`
2. Determine k-block: `kb = k_sub / 2` (sub-tiles 0,1 -> block 0; 2,3 -> block 1)
3. Load K bit-plane words from shared memory at `sh_b[col * stride + kb * K + bit]`
4. For each of 4 rows: extract K-bit index from the bit-plane words
5. Codebook lookup: `val = __shfl_sync(mask, cb_h, idx)`
6. Scale: `val *= absmax`
7. Pack into half2: `frag_b[0] = make_half2(val[0], val[1])`, etc.
8. Feed to MMA instruction

---

## 25. Performance Model and Targets

### Arithmetic Intensity

Per thread block per K-tile (TILE_M=64, TILE_N=128, TILE_K=64, K=4):
- Compute: 8 warps * 32 MMA ops * 256 FMA ops = 65,536 FMAs * 2 (K-sub-tiles have 2 blocks) = 262,144 FLOPs
- Memory loads:
  - A: 64 * 64 * 2 = 8,192 bytes
  - B: 128 * 2 * 4 * 4 = 4,096 bytes
  - Absmax: 128 * 2 = 256 bytes
  - Total: 12,544 bytes
- Intensity: 262,144 / 12,544 = **20.9 FLOP/byte**

Compare fp16 GEMM (same tiles, B in fp16):
- B: 128 * 64 * 2 = 16,384 bytes
- Total: 24,832 bytes
- Intensity: 262,144 / 24,832 = 10.6 FLOP/byte

The kbit kernel has **~2x higher arithmetic intensity** due to compressed weights.

### Roofline Analysis

**4090 (Ada Lovelace):**
- Peak fp16 tensor: 83 TFLOPS
- Peak bandwidth: 1 TB/s
- Ridge point: 83,000 / 1,000 = 83 FLOP/byte

For a 4096x4096 weight with K=4:

| M | Intensity | Regime | Expected speedup vs fp16 |
|---|-----------|--------|--------------------------|
| 1 | ~3 | Memory-bound | ~3.8x (weight data 3.8x smaller) |
| 8 | ~24 | Memory-bound | ~2.5x |
| 32 | ~93 | Near ridge | ~1.5x |
| 128 | ~296 | Compute-bound | ~1x (limited by tensor core throughput) |

### Performance Targets

**M=1 (batch=1):** Target ~4x faster than cuBLAS fp16 GEMM. This is the
theoretical maximum from 4x less weight data. Achieving >50% of this
(~2x speedup) would be a good initial result.

**M=32:** Target near-theoretical bandwidth utilization (>50%).

**M >= 128:** No hard targets. The codebook lookup (shuffle-based) is
inherently more expensive than Marlin's linear dequant (bitwise ops), so
we expect lower peak FLOPS utilization than Marlin.

**All M:** Must be faster than standalone `dequant_kbit()` + cuBLAS. If
the fused kernel is slower, there's no point in it.

---

## 26. Correctness Verification Strategy

### Two-Pronged Approach

**1. Reference match (torch.allclose):**
Compare fused GEMM output against `torch.matmul(A, dequant_kbit(W).T)`.
Tolerance: `rtol=0.1, atol=0.1 * output_mean` to account for E4M4 absmax
error propagation.

**2. SQNR-based:**
Measure Signal-to-Quantization-Noise Ratio between fused GEMM and unquantized
fp16 GEMM. Target: SQNR > 10 dB for K=4 (the quantization noise dominates;
the fused kernel should not add measurable additional noise).

### Why Both Are Needed

The reference match catches bugs in the dequantization + MMA logic (wrong
indices, wrong scales, wrong accumulation). It compares against a known-good
dequant path.

The SQNR test catches cases where the output is technically correct but
numerically degraded beyond what quantization should introduce (e.g., from
missing fp32 accumulation, or from precision loss in the codebook conversion).

---

## 27. Implementation Pipeline: The 6-Stage Approach

### Why Staged

CUDA kernel development is notoriously hard to debug. A single wrong index
or missing synchronization can produce silently wrong results. By building
incrementally, each stage adds exactly one source of complexity. If a stage
breaks, you know where to look.

### Stage 1: Python Reference (COMPLETE)

Write Python implementations of:
- `repack_kbit_ref()`: transforms flat packed data to GEMM-tiled layout
- `unrepack_kbit_ref()`: inverse of repack (for round-trip testing)
- `kbit_gemm_ref()`: dequant (via unrepack) then matmul
- `kbit_gemm_ref_direct()`: direct quantize -> dequant -> matmul

These are ground truth for all later stages. They run on CPU (no GPU needed),
making them easy to debug with print statements and Python debuggers.

### Stage 2: CUDA Repack Kernel

Implement the CUDA repack kernel. Test: bit-exact uint32 match with Python
reference. This is a simple gather/scatter kernel (no tensor cores, no
pipeline, no shared memory complexity). If this is wrong, all subsequent
stages produce garbage.

### Stage 3: Minimal CUDA GEMM

The simplest possible GEMM:
- Synchronous global memory loads (no cp.async)
- 1 block per output tile (no persistent kernel)
- Process all K-tiles sequentially in a simple loop
- Single pipeline stage (load -> process -> load -> process)

This validates:
- Tiled layout addressing (does the kernel read the right data?)
- Bit-plane extraction from shared memory
- Codebook lookup via `__shfl_sync`
- MMA fragment assembly and execution
- Output write

Test: match Python reference within tolerance.

### Stage 4: cp.async Pipeline

Replace synchronous loads with 4-stage cp.async pipeline. No other changes.
The math should be identical -- we're just changing WHEN data is loaded, not
WHAT data is loaded.

Test: must match Stage 3 output exactly (bitwise). If there's any difference,
the pipeline has a synchronization bug.

### Stage 5: Persistent Kernel + Split-K

Add:
- Work distribution across `min(num_SMs, total_work)` blocks
- Accumulator management (persist across consecutive k_chunks for same output tile)
- Split-K via atomicAdd + __threadfence() + tile counters
- First-contributor plain store + subsequent atomicAdd
- Last-contributor fp32->fp16 conversion

Test: match Stage 4 for non-split-K cases. Match Python reference for
forced split-K cases (with slightly relaxed tolerance for fp32 accumulation
order differences).

### Stage 6: Optimization + bf16 + Benchmarks

Add:
- A-tile XOR swizzle for bank-conflict-free ldmatrix
- C output staging through shared memory for coalesced writes
- bf16 support (template on scalar_t)
- Performance benchmarking across M, N, K_dim, K values
- Comparison against cuBLAS and standalone dequant + cuBLAS

---

## 28. Implementation Progress: Stage 1 Complete

### What Was Built

File: `tests/test_kbit_gemm.py` in the `feature/kbit-gemm` worktree.

Contains:
- Helper functions (codebook generation, quantize/dequant/pack/unpack refs,
  E4M4 encode/decode)
- `repack_kbit_ref()`: Python reference repack (flat -> tiled)
- `unrepack_kbit_ref()`: Python reference unrepack (tiled -> flat)
- `kbit_gemm_ref()`: Reference fused GEMM (via unrepack + dequant + matmul)
- `kbit_gemm_ref_direct()`: Direct reference (quantize -> dequant -> matmul)

### Test Results

38 tests, all passing:

**TestRepackRef (24 tests):**
- `test_repack_round_trip` [K=2,3,4,5]: repack -> unrepack recovers original
  data bit-exactly.
- `test_repack_tile_contiguity` [K=2,3,4,5]: output size matches expected
  tile count.
- `test_repack_various_sizes` [4 sizes x 4 K values]: works for different
  aligned matrix dimensions.

**TestFusedGemmRef (14 tests):**
- `test_gemm_matches_direct` [K=2,3,4,5]: fused GEMM matches direct reference
  within E4M4 tolerance.
- `test_gemm_m1` [K=2,3,4,5]: works for M=1.
- `test_gemm_various_batch_sizes` [M=1,4,16,32]: works across batch sizes.
- `test_gemm_fp16_output_quality`: SQNR > 10 dB vs unquantized fp16.
- `test_gemm_nonstandard_codebook`: works with asymmetric codebook.

### Tolerance Calibration

The fused GEMM reference goes through E4M4 absmax encode/decode, while the
direct reference uses float32 absmax. This introduces per-block error of up
to ~6.25% (E4M4 mantissa precision). For near-zero output values, relative
error becomes huge even with tiny absolute error.

The tests use `torch.allclose` with:
- `rtol=0.1` (10% relative tolerance for E4M4 error propagation)
- `atol=0.05-0.1 * C_direct.abs().mean()` (absolute tolerance scaled to
  output magnitude, handling near-zero values)

---

## 29. Shared Memory Budget Analysis

### Per-Stage Breakdown

For TILE_M=64, TILE_N=128:

| Component | K=2 | K=3 | K=4 | K=5 |
|-----------|-----|-----|-----|-----|
| A tile (fp16) | 8,192 B | 8,192 B | 8,192 B | 8,192 B |
| B tile (packed) | 2,048 B | 3,072 B | 4,096 B | 5,120 B |
| B padding (+1/col) | 512 B | 512 B | 512 B | 512 B |
| Absmax (E4M4) | 256 B | 256 B | 256 B | 256 B |
| **Per stage** | **11,008 B** | **12,032 B** | **13,056 B** | **14,080 B** |

4 stages:

| K | Total shmem | 4090 (100 KB) | A100 (164 KB) | H100 (228 KB) |
|---|-------------|---------------|---------------|----------------|
| 2 | 44 KB | 56% | 27% | 19% |
| 3 | 48 KB | 48% | 29% | 21% |
| 4 | 52 KB | 52% | 32% | 23% |
| 5 | 56 KB | 56% | 34% | 25% |

All fit comfortably. The C output staging area (reusing pipeline shmem) needs
TILE_M * TILE_N * 2 = 64 * 128 * 2 = 16 KB, which fits in one pipeline stage.

### For Smaller M_BLOCKS

When TILE_M = 16 (M_BLOCKS=1), the A tile shrinks to 16 * 64 * 2 = 2 KB.
Per stage drops to ~7-10 KB. 4 stages: ~28-40 KB. Even more headroom.

---

## 30. Risk Register

### Risk 1: A-tile Swizzle Correctness (HIGH)

**Problem:** Getting the XOR swizzle wrong causes silent bank conflicts on
`ldmatrix` reads. The kernel produces correct results but at ~50% shared
memory throughput.

**Detection:** Only visible via nsight compute profiling (bank conflict
metrics). Not detectable from output correctness.

**Mitigation:** Implement Stage 3 (minimal GEMM) first WITHOUT the swizzle.
This establishes a correctness baseline. Add the swizzle in Stage 6 and
verify it doesn't change output while improving profiled performance.

### Risk 2: Repack Index Math (HIGH)

**Problem:** A single index error in the flat-to-tiled permutation silently
corrupts all GEMM results. The kernel runs, the output has the right shape,
but the values are wrong.

**Detection:** The Python reference repack enables bit-exact validation. If
the CUDA repack matches the Python repack element-by-element, the index math
is correct.

**Mitigation:** Stage 2 exists specifically to validate the repack in isolation,
before the GEMM kernel is built. The round-trip test (repack -> unrepack ->
verify) provides a second layer of validation.

### Risk 3: Inter-Block Synchronization in Split-K (HIGH)

**Problem:** Missing `__threadfence()` or incorrect counter logic causes rare,
non-deterministic wrong results. May only manifest under specific timing
conditions (high GPU load, specific work distributions).

**Detection:** Difficult. Wrong results may appear correct most of the time
and only fail under specific conditions.

**Mitigation:**
1. Code review focusing on the `__threadfence()` placement
2. Test with forced split-K on small problems (e.g., 2 blocks sharing a
   single output tile) where the output is easily hand-verified
3. Run tests many times with different random seeds to catch intermittent
   failures

### Risk 4: Register Spilling (MEDIUM)

**Problem:** Compiler uses more registers than estimated, causing spills to
local memory. Performance drops significantly.

**Detection:** Check `--ptxas-options=-v` output during compilation. Look
for "spill stores" and "spill loads" in the per-kernel statistics.

**Mitigation:** If spilling occurs:
- Cap M_BLOCKS at 3 (saves 32 registers per thread)
- Use `__launch_bounds__(256, 1)` to hint the compiler
- Manually reduce live register ranges (e.g., don't double-buffer FragA)

### Risk 5: Pipeline Underutilization for Small K_dim (MEDIUM)

**Problem:** If K_dim/64 < 4 (fewer K-tiles than pipeline stages), the
pipeline never reaches steady state. Most time is spent in fill/drain phases.

**Mitigation:** Not a concern for the target use case (K_dim >= 4096 = 64
K-tiles). For very small K_dim, the computation is so fast that the overhead
doesn't matter in absolute terms.

### Risk 6: K=5 Codebook Using All 32 Lanes (LOW)

**Problem:** For K=5, all 32 warp lanes hold codebook entries. There are no
"unused" lanes as a safety margin. If an index extraction bug produces an
out-of-range value, it would read from an unexpected lane.

**Mitigation:** The index extraction from 5 bit-planes can only produce values
0-31 by construction (5 bits can represent 0-31). The `__shfl_sync` instruction
wraps indices modulo 32, providing additional safety. Explicit K=5 tests in the
test suite verify correctness.

---

## 31. File Locations and Worktree Setup

### Worktree

```
~/git/bnb-kbit-gemm/     Branch: feature/kbit-gemm
                          Based on: feature/kbit-quantization
```

Created from the main bitsandbytes checkout:
```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-kbit-gemm -b feature/kbit-gemm feature/kbit-quantization
```

### Key Files

| File | Purpose |
|------|---------|
| `agents/kbit_gemm_context.md` | Complete design context document |
| `cuda-spec.md` | Distilled spec from interview (gitignored) |
| `progress.md` | This document (progress report) |
| `tests/test_kbit_gemm.py` | Stage 1 Python reference + tests |
| `tests/test_kbit_quantization.py` | Existing kbit quant tests |
| `csrc/ops.cu` | Existing kbit CUDA kernels (quant/dequant) |
| `bitsandbytes/functional.py` | Python kbit API |

### Files to Be Created (Future Stages)

| File | Stage | Purpose |
|------|-------|---------|
| `csrc/kernels.cu` | 2-5 | GEMM + repack CUDA kernels |
| `csrc/kernels.cuh` | 2-5 | Kernel declarations |
| `csrc/pythonInterface.cpp` | 2-5 | C wrappers (append) |
| `bitsandbytes/_ops.py` | 2-5 | torch.library op defs (append) |
| `bitsandbytes/backends/cuda/ops.py` | 2-5 | CUDA backend dispatch (append) |

---

## 32. How to Read the Spec

The `cuda-spec.md` file is structured for implementation reference, not for
understanding the design decisions. Here's how to use it:

### Section 1 (Kernel Design Summary)

Start here. This tells you what you're building: the function signature,
template parameters, launch configuration, tile sizes. The M_BLOCKS dispatch
table shows how the kernel adapts to different batch sizes.

### Section 2 (Memory Access Plan)

The detailed data flow. Read this before writing any load/store code. The
bank conflict fix (B-tile +1 padding) is critical -- implement it from the
start, not as an afterthought. The shared memory layout table gives exact
byte counts per component.

### Section 3 (Warp Execution Plan)

The thread-to-data mapping. This is the hardest part to get right. The
B-fragment layout (Section 24 of this document) explains how threads map to
columns and rows within the MMA instruction. Understanding this mapping is
essential for writing the dequantization inner loop.

### Section 4 (Data Layout)

The tiled memory format. Use the Python reference (`repack_kbit_ref`) as the
authoritative specification. The CUDA repack kernel must produce bit-exact
matching output.

### Section 5 (Correctness Constraints)

The synchronization requirements. The `__threadfence()` placement (Section 6
of this document) is a correctness requirement, not an optimization.

### Section 7 (Key Decisions)

Quick reference table of all decisions with one-line reasoning. Useful for
"why did we choose X?" questions. This document (progress.md) has the full
reasoning for each decision.

### Section 8 (Risks)

Must-read before starting implementation. Each risk has a specific mitigation
strategy.

---

## 33. Next Steps

### Immediate: Stage 2 (CUDA Repack Kernel)

1. Implement `kbit_repack_kernel<K_BITS, TILE_K, TILE_N>` in `csrc/kernels.cu`
2. The kernel is a simple gather/scatter: read from flat layout, write to
   tiled layout. No tensor cores, no shared memory pipeline.
3. Test: bit-exact uint32 match against `repack_kbit_ref()` from Stage 1
4. Also test: round-trip (repack -> unrepack on CPU side) preserves data

### After Stage 2: Stage 3 (Minimal GEMM)

This is the hardest stage. It validates all the core math:
- Reading bit-plane words from the tiled layout in shared memory
- Extracting K-bit indices using the fragment row mapping
- Codebook lookup via `__shfl_sync`
- E4M4 absmax decode and scale application
- MMA fragment assembly and execution
- Output write (initially direct, no staging)

Start with K=4, M_BLOCKS=1, half only. Get one configuration working before
templating on K, M_BLOCKS, and scalar_t.

### Stages 4-6

These are incremental improvements to the Stage 3 kernel. Each stage has a
clear test criterion (match previous stage's output). The implementation risk
decreases with each stage because the core math is already validated.

---

## Appendix: Interview Question Log

For reference, here is every question asked during the interview and the
decision reached:

1. **FragB column mapping across N-blocks** -> Need to work out (led to
   detailed analysis in Section 24)
2. **Atomic ordering in split-K** -> `__threadfence()` needed (Section 6)
3. **K_dim alignment with TILE_K** -> Partial K-tile handling with separate
   code path (Section 11)
4. **Minimum compute capability** -> sm_80+ only (Section 20)
5. **B-tile bank conflicts** -> +1 padding per column (Section 5)
6. **First contributor store pattern** -> Plain store + fence (Section 6)
7. **Partial K-tile implementation** -> Runtime branch, rarely taken (Section 11)
8. **A-tile swizzle** -> Marlin's or custom XOR (Section 13)
9. **C output write coalescing** -> Stage through shared memory (Section 14)
10. **N alignment** -> Require N % 128 == 0 (Section 11)
11. **Pipeline depth** -> 4 stages (Section 8)
12. **bf16 support** -> From day one (Section 18)
13. **Accuracy bar** -> Both allclose and SQNR tests (Section 26)
14. **Repack testing** -> Python reference + CUDA validation (Section 27)
15. **Workspace allocation** -> PyTorch caching allocator (Section 22)
16. **Performance targets** -> ~4x at M=1, measure and iterate (Section 25)
17. **K=5 codebook** -> Test explicitly, no correctness concern (Section 23)
18. **Grid sizing** -> min(SMs, total_work) (Section 15)
19. **B-load coalescing** -> Linear mapping, strided loop (Section 16)
20. **Shared memory budget** -> Fits, no concern (Section 29)
21. **Weight layout** -> Accept [N, K_dim], transpose in repack (Section 10)
22. **Minimum problem size** -> Always use fused kernel (Section 21)
23. **Register pressure** -> 1 block/SM is fine (Section 17)
24. **Partial M-tiles** -> Predicated cp.async + masked write (Section 12)
25. **Warp layout** -> Adapts to M_BLOCKS (Section 9)
26. **Template instantiations** -> 40 variants, manageable (Section 19)
27. **fp32 vs fp16 accumulation** -> fp32 always (Section 7)
28. **K=3, K=5 handling** -> Bit-plane format handles uniformly (Section 23)
29. **Non-standard codebook** -> Test with one case (Section 26)

---

## 34. Implementation Progress: Stages 2–3 Complete

### Stage 2: CUDA Repack Kernel

**File:** `csrc/ops.cu` (appended to existing kbit code)

The repack kernel transforms flat bit-plane packed data into the GEMM-tiled
layout. Each CUDA thread block handles one output tile. The kernel is a simple
gather/scatter — no tensor cores, no shared memory pipeline.

**Key design:** The output layout is organized so that one output tile
(TILE_K × TILE_N = 64 × 128) contains all the packed bit-plane words and
E4M4 absmax values needed for one iteration of the GEMM inner loop. This
enables the GEMM kernel to load contiguous chunks of global memory into
shared memory.

**Tests (25 PASSING):**
- `TestRepackCUDA::test_repack_matches_reference` [K=2,3,4,5]: bit-exact
  uint32 match against Python reference.
- `TestRepackCUDA::test_repack_output_sizes`: output buffer sizes are correct.
- `TestRepackCUDA::test_repack_round_trip_with_gemm` [K=2,3,4,5]: repacked
  data fed through CUDA GEMM produces correct output.
- `TestRepackCUDA::test_repack_various_sizes` [4 sizes × 4 K values]: works
  for 128×128, 128×256, 256×128, 256×256.

No issues encountered during Stage 2.

### Stage 3: Minimal Fused Dequant + GEMM Kernel

**File:** `csrc/ops.cu` (function `kbit_gemm_minimal<K_BITS>`)

The minimal GEMM validates all the core math without the async pipeline. It
uses synchronous shared memory loads, one warp per 16-column output slice,
and m16n8k16 tensor core MMA instructions with fp32 accumulation.

**Design:**
- Grid: (n_tiles, m_tiles), 256 threads (8 warps) per block
- TILE_M=16, TILE_K=64, TILE_N=128
- Each warp handles 16 columns (2 MMA N-blocks of 8 columns each)
- 4 k-sub-tiles per TILE_K (each 16 elements = one MMA k-dimension)
- Codebook stored in registers via `__shfl_sync` lookup
- E4M4 absmax decoded on the fly from uint8

**Tests (13 PASSING):**
- `TestGemmCUDA::test_gemm_matches_reference` [K=2,3,4,5]: matches Python
  reference within E4M4 + fp16 accumulation tolerance.
- `TestGemmCUDA::test_gemm_various_sizes` [4 sizes × K=4]: works for
  128×128, 128×256, 256×128, 256×256.
- `TestGemmCUDA::test_gemm_various_M` [M=1,4,8,16 × K=4]: works across
  batch sizes.
- `TestGemmCUDA::test_gemm_sqnr`: SQNR > 20 dB for K=4 and K=5.

### Bug: MMA A-Fragment Register Ordering (Stage 3)

**Symptom:** The MMA m16n8k16 instruction produced results that only
accumulated k=0..7 instead of k=0..15. C[0,0] was 36 (sum of 1..8) instead
of 136 (sum of 1..16). Identity matrix tests passed by coincidence since
B's identity values are only in the first 8 rows.

**Root cause:** The A-fragment register array was ordered as:
```
frag_a[0] = {A[gid, tid*2..tid*2+1]}       (row_lo, k_lo)  ← correct
frag_a[1] = {A[gid, tid*2+8..tid*2+9]}     (row_lo, k_hi)  ← WRONG
frag_a[2] = {A[gid+8, tid*2..tid*2+1]}     (row_hi, k_lo)  ← WRONG
frag_a[3] = {A[gid+8, tid*2+8..tid*2+9]}   (row_hi, k_hi)  ← correct
```

The hardware expects registers ordered for two consecutive m16n8k8 operations
(the Turing decomposition): a[0],a[1] handle k_lo, a[2],a[3] handle k_hi.
Within each pair, a[even]=row_lo and a[odd]=row_hi. So the correct order is:
```
a[0] = row_lo, k_lo
a[1] = row_hi, k_lo    ← rows interleaved BEFORE k-halves
a[2] = row_lo, k_hi
a[3] = row_hi, k_hi
```

**How it was found:** Fragment data was confirmed correct by writing a
dump-fragments kernel that outputs each thread's register values. The data
was perfect — the bug was purely in which register position each value was
assigned to. The fix was discovered by examining Marlin's `mma_trans()`
function in `marlin_mma.h`, which decomposes m16n8k16 into two m16n8k8 calls
on Turing (sm_75). The first call uses a[0],a[1] with b[0], the second uses
a[2],a[3] with b[1]. This reveals the interleaved ordering.

**Fix:** Swap frag_a[1] and frag_a[2] in both the test kernel and the GEMM
kernel. The same fix was applied in the GEMM kernel's A-fragment loading from
shared memory.

**Lesson for future stages:** The PTX ISA documentation describes fragment
coordinates but does NOT clearly specify register ordering for m16n8k16. The
Turing m16n8k8 decomposition is the authoritative reference for register
assignment. Always verify MMA fragment ordering against Marlin's implementation.

### Updated File Map

| File | Purpose |
|------|---------|
| `tests/test_kbit_gemm.py` | All stage tests (76 total) |
| `csrc/ops.cu` | Repack kernel, GEMM kernel, MMA test kernel |
| `csrc/pythonInterface.cpp` | C wrappers for repack/GEMM/MMA |
| `bitsandbytes/_ops.py` | torch.library op definitions |
| `bitsandbytes/backends/cuda/ops.py` | CUDA backend dispatch |

### Commit History

```
bff83e6 Add Stage 2 repack kernel, Stage 3 minimal GEMM kernel (76 tests pass)
f95a7f2 Fix analytical error bound for K=5 with E4M4 absmax
8a2817e Template dequant kernel on output type, add bf16/fp32 native output
03415e1 Remove scalar dequant kernel, fp32 absmax, and Stage 1-3 scaffolding
2973bf5 Add vectorized dequant kernel and E4M4 uint8 absmax support
2825890 Complete k-bit quantization: Stages 6-8, Python API, 218 tests pass
```

---

## 35. Next Steps: Stage 4 (cp.async Pipeline)

### Goal

Replace synchronous global→shared memory loads with a double-buffered
cp.async pipeline. Math remains identical — this is a pure performance change.
Test criterion: output matches Stage 3 bit-for-bit.

### Changes

1. Double the shared memory allocation (2 stages × per-stage size)
2. Replace the cooperative thread loads with `cp.async` copies
3. Add pipeline fence/wait logic around the k-tile loop
4. Prefetch the next tile while computing the current one

### Key Design Points

- 2-stage double buffer (not 4-stage; simpler, sufficient for this tile size)
- `cp_async_wait<1>()` inside the loop waits for the computing stage
- The first tile is prefetched before the loop starts
- `cp_async_wait<0>()` after the loop drains the pipeline

---

## 36. Implementation Progress: Stage 4-6 Complete

### Stage 4: cp.async Double-Buffered Pipeline (commit 9b155d3)

Replaces synchronous global→shared memory loads with `cp.async` double buffering.
B tile and absmax loaded via `cp.async.cg.shared.global` (16-byte copies, L2 only).
A tile loaded synchronously (needs M/K_dim bounds checking).
Output is bit-exact identical to Stage 3 for all K values.

**Tests:** 13 new tests → 89 total (all pass).

### Stage 5: Split-K GEMM (commit fdcec9c)

Adds split-K support: multiple blocks share an output tile, each handling a
subset of k-tiles. Partial sums accumulated via atomicAdd in fp32 workspace.
Grid is 2D for k_chunks=1, 3D for k_chunks>1. Last contributor (detected via
atomic tile counter) converts fp32→fp16 output.

**Tests:** 21 new tests → 110 total (all pass).

### Stage 6: Production Kernel with bf16, ldmatrix, Swizzle, Benchmarks

#### bf16 Support (commit 24406d2)

New production kernel `kbit_gemm_prod` templates on `scalar_t` (half or
__nv_bfloat16). Uses `if constexpr` to select the right MMA PTX instruction:
- fp16: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- bf16: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

Helper structs `ScalarOps<T>`, `pack_two<T>`, and `mma_m16n8k16<T>` abstract
type-specific operations. 8 kernel variants instantiated (4 K × 2 dtypes).

fp16 path matches Stage 5 split-K output bit-for-bit.
bf16 path matches Python reference within tolerance for all K values.

**Tests:** 29 new tests → 139 total (all pass).

#### ldmatrix + XOR Swizzle (commit b64bb91)

Replaced 8 element-by-element shared memory reads per A fragment with a single
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` instruction.

**The bank conflict problem:** Without swizzle, the A tile stored in shared
memory with stride TILE_K=64 halves (128 bytes) causes every row to start at
the same bank (stride is a multiple of 128 bytes = the bank repeat distance).
This gives 8-way bank conflicts during ldmatrix.

**The fix:** XOR-based swizzle at 8-half (16-byte) granularity:
```
col_group = col / 8
swizzled_group = col_group ^ (row % 8)
swizzled_col = swizzled_group * 8 + (col % 8)
```

Applied during A tile write to shared memory AND in the ldmatrix address
calculation. The XOR distributes 8 threads in an ldmatrix group across 8
different banks (zero conflicts).

Output is mathematically identical (verified by tests).

#### Benchmark Results (commit 27cf6a2)

RTX 4090, K=4 (4-bit), fp16, k_chunks=1:

| M | K_dim | N | kbit (µs) | kbit TFLOPS | cuBLAS (µs) | Speedup |
|---:|------:|------:|----------:|------------:|------------:|--------:|
| 1 | 4096 | 4096 | 109 | 0.31 | 43 | 0.39x |
| 1 | 4096 | 11008 | 82 | 1.10 | 128 | **1.56x** |
| 4 | 4096 | 11008 | 100 | 3.61 | 121 | **1.21x** |
| 4 | 4096 | 4096 | 92 | 1.46 | 22 | 0.24x |

**Analysis:** The kernel wins in the memory-bandwidth-bound regime (M=1, large
N) where reading 4x less weight data matters. It loses in compute-bound cases
because the current tile is small (TILE_M=16, only 2 N-blocks per warp).

### Optimization Opportunities for Further Work

1. **Multi-M-block tiling:** Template on M_BLOCKS (1-4) so TILE_M scales to
   32/48/64. This is the biggest performance lever for M>1.
2. **Larger N_BLOCKS:** Use more of the warp's N-dimension capacity.
3. **C output staging through shared memory:** Coalesce the scattered fragment
   writes to global memory (currently each thread writes to non-contiguous rows).
4. **Persistent kernel:** Replace the 3D grid with a persistent kernel that
   loops over work items, reducing launch overhead and enabling better SM
   utilization for small tile counts.

### Commit History (Stages 4-6)

```
27cf6a2 Add kbit GEMM benchmark script
b64bb91 Add ldmatrix + XOR swizzle for A-fragment loading in production kernel
24406d2 Add Stage 6 production kernel with bf16 support (139 tests pass)
fdcec9c Add Stage 5 split-K GEMM kernel (110 tests pass)
9b155d3 Add Stage 4 pipelined GEMM kernel with cp.async double-buffering (89 tests pass)
```
