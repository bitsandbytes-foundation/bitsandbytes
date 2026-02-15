# kbit GEMM Kernel: Complete Development Record

This document is an exhaustive record of every design decision, implementation
stage, optimization attempt, benchmark result, and architectural constraint
encountered during the development of the fused kbit dequantization + GEMM
kernel in bitsandbytes. It is written to be fully self-contained: a developer
reading this document should understand the entire project state, why every
decision was made, what was tried and what failed, and what the path forward is.

**Companion document:** [`optimization.md`](optimization.md) contains the
current kernel strategy and optimization plan, including the three-kernel
dispatch (scalar GEMV, grouped GEMM, dequant+cuBLAS) with benchmark data.

---

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Target Models and Shapes](#2-target-models-and-shapes)
3.  [Quantization Format: Bit-Plane Packing](#3-quantization-format-bit-plane-packing)
4.  [Codebook and Absmax Encoding](#4-codebook-and-absmax-encoding)
5.  [Source Materials Studied](#5-source-materials-studied)
6.  [Design Interview and Hardening](#6-design-interview-and-hardening)
7.  [Design Decision Record](#7-design-decision-record)
    - 7.1  Bit-Plane Format
    - 7.2  Shared Memory Bank Conflicts (B-tile +1 Padding)
    - 7.3  Atomic Ordering in Split-K
    - 7.4  fp32 vs fp16 Accumulation
    - 7.5  Pipeline Depth
    - 7.6  Warp Layout and M_BLOCKS Dispatch
    - 7.7  Weight Layout and Repack Convention
    - 7.8  N and K Alignment
    - 7.9  Partial M-tile Handling
    - 7.10 A-tile XOR Swizzle
    - 7.11 C Output Write Strategy
    - 7.12 Grid Sizing
    - 7.13 B-tile Load Coalescing
    - 7.14 Register Pressure and Occupancy
    - 7.15 bf16 Support
    - 7.16 Template Instantiations
    - 7.17 Target Architecture
    - 7.18 Minimum Problem Size
    - 7.19 Workspace Allocation
8.  [Tensor Core Fragment Layout](#8-tensor-core-fragment-layout)
9.  [K-Value Analysis: Why K=3 and K=5 Are Not Special](#9-k-value-analysis)
10. [Shared Memory Budget Analysis](#10-shared-memory-budget-analysis)
11. [Performance Model and Roofline](#11-performance-model-and-roofline)
12. [Correctness Verification Strategy](#12-correctness-verification-strategy)
13. [Implementation Stage 1: Python Reference](#13-implementation-stage-1-python-reference)
14. [Implementation Stage 2: CUDA Repack Kernel](#14-implementation-stage-2-cuda-repack-kernel)
15. [Implementation Stage 3: Minimal CUDA GEMM](#15-implementation-stage-3-minimal-cuda-gemm)
16. [Implementation Stage 4: cp.async Pipeline](#16-implementation-stage-4-cpasync-pipeline)
17. [Implementation Stage 5: Split-K](#17-implementation-stage-5-split-k)
18. [Implementation Stage 6: Production Kernel](#18-implementation-stage-6-production-kernel)
19. [Optimization Phase 1: Inner Loop Tweaks](#19-optimization-phase-1-inner-loop-tweaks)
20. [Optimization: B-tile Bank Conflict Fix Attempt](#20-optimization-b-tile-bank-conflict-fix-attempt)
21. [Optimization Phase 2: V2 Kernel (Dequant-During-Fetch)](#21-optimization-phase-2-v2-kernel)
22. [Root Cause Analysis: Why MoE Shapes Are Slow](#22-root-cause-analysis)
23. [GPU Architecture Constraints: mma.sync vs wgmma](#23-gpu-architecture-constraints)
24. [The Path Forward: Grouped Expert GEMM](#24-the-path-forward-grouped-expert-gemm)
25. [Risk Register](#25-risk-register)
26. [File Locations and Worktree Setup](#26-file-locations-and-worktree-setup)
27. [Full Commit History](#27-full-commit-history)
28. [Current Status](#28-current-status)

---

## 1. Project Overview

### 1.1 What We Are Building

A fused CUDA kernel that combines weight dequantization and matrix multiplication
(GEMM) into a single operation:

```
C[M, N] = A[M, K_dim] * W_kbit[K_dim, N]^T
```

Where:
- A is the activation matrix (fp16 or bf16), typically M=1-32 tokens
- W is the weight matrix, stored in kbit-quantized format (K=2,3,4,5 bits per weight)
- C is the output matrix (fp16 or bf16)

### 1.2 Why This Matters

Currently, bitsandbytes has standalone quantize and dequantize kernels for kbit
quantization, but no fused GEMM. To do inference with quantized weights, you must:

1. Dequantize the entire weight matrix back to fp16 (writes full fp16 matrix to GMEM)
2. Call cuBLAS GEMM on the fp16 weights (reads it back from GMEM)

This is wasteful because the weight data moves through memory twice. A fused
kernel dequantizes weights on-the-fly in registers/shared memory and feeds them
directly to tensor core MMA instructions. For K=4 (4-bit weights), this means
reading **3.6x less data** from global memory compared to cuBLAS.

### 1.3 Target Use Case

LLM inference with small batch sizes (M=1-32). Weight matrices are large
(K_dim=2048-28672, N=512-28672). At these batch sizes, the GEMM is
memory-bandwidth-bound, so reading 3.6x less weight data can translate to
significant speedups.

### 1.4 Relationship to Existing Code

The kbit quantization system lives on the `feature/kbit-quantization` branch.
It implements:
- `quantize_kbit()`: quantizes a tensor using K-bit blockwise quantization
- `dequantize_kbit()`: reconstructs the tensor from packed format
- Codebook generation, E4M4 absmax encoding, bit-plane packing

The GEMM kernel builds on top of this quantization system using the same
packed data format, codebook, and absmax encoding. The GEMM branch
`feature/kbit-gemm` is based on `feature/kbit-quantization`.

### 1.5 Current Hardware

Development and benchmarking on **RTX 4090** (Ada Lovelace):
- SM count: 128
- Architecture: sm_89
- Shared memory: 100 KB per SM
- L2 cache: 72 MB
- Memory bandwidth: ~1 TB/s (GDDR6X)
- L2 bandwidth: ~2 TB/s (measured effective)
- MMA instruction: `mma.sync` (synchronous, warp stalls until complete)
- Clocks locked at 2520 MHz for benchmarking

---

## 2. Target Models and Shapes

### 2.1 Primary Target: Qwen3-Coder-Next (MoE, 70B+, hidden=2048)

This is a Mixture-of-Experts model with 512 experts, 10 per token, 48 layers.
The MoE expert shapes are the most important optimization target because they
have extremely low SM utilization when launched individually.

| Layer type | K_dim | N | kbit data | Tiles (TILE_N=128) | SM util |
|------------|------:|-----:|----------:|------:|--------:|
| MoE gate/up (per expert) | 2048 | 512 | 0.5 MB | 4 | 3% |
| MoE down (per expert) | 512 | 2048 | 0.5 MB | 16 | 12% |
| Dense gate/up | 2048 | 5120 | 5.2 MB | 40 | 31% |
| Dense down | 5120 | 2048 | 5.2 MB | 16 | 12% |
| Q proj | 2048 | 4096 | 4.2 MB | 32 | 25% |
| KV proj | 2048 | 512 | 0.5 MB | 4 | 3% |
| O proj | 4096 | 2048 | 4.2 MB | 16 | 12% |

**Key insight:** MoE expert shapes produce only 4-16 tiles on 128 SMs, meaning
3-12% SM utilization. No inner-loop optimization can fix this. Grouped expert
GEMM (batching all active expert invocations into one kernel launch) is the
architectural solution.

### 2.2 Secondary Target: GLM-4.7-Flash (MoE, hidden=2048)

| Layer type | K_dim | N | kbit data | Tiles | SM util |
|------------|------:|-----:|----------:|------:|--------:|
| Routed gate/up | 2048 | 1536 | 1.6 MB | 12 | 9% |
| Routed down | 1536 | 2048 | 1.6 MB | 16 | 12% |
| Shared gate/up | 2048 | 10240 | 10.5 MB | 80 | 62% |
| Shared down | 10240 | 2048 | 10.5 MB | 16 | 12% |

All shapes fit in L2 cache (72 MB on RTX 4090) when launched individually.

### 2.3 Llama-style Dense Models (Not Priority)

| Model | hidden | gate/up (N) | kbit data | Fits L2? |
|-------|-------:|------------:|----------:|:---------|
| Llama 3 8B | 4096 | 14336 | 29.4 MB | YES |
| Llama 3 70B | 8192 | 28672 | 117.4 MB | NO |

The kernel already achieves ~1.5-2.6x over cuBLAS on these shapes. They are
**not** a priority because they already work well. The focus is on MoE shapes.

### 2.4 Importance Note

All MoE weight data for both Qwen3-Next and GLM-4.7-Flash fits in L2 cache.
This means effective memory bandwidth is ~2 TB/s from L2, not ~1 TB/s from
DRAM. When data is L2-resident, the kernel is instruction-limited, not
bandwidth-limited. This is the core challenge for MoE shapes.

---

## 3. Quantization Format: Bit-Plane Packing

### 3.1 Format Description

Each quantization block contains 32 elements (blocksize=32, one warp). For K-bit
quantization, the block is represented as:

- **K uint32 words** ("bit-planes"): word j contains bit j of all 32 elements'
  indices. Extracted via `__ballot_sync` during quantization.
- **1 E4M4 uint8** absmax: the maximum absolute value of the block, encoded in
  a compact 4-bit exponent + 4-bit mantissa format.

To reconstruct the K-bit index for element i within a block:
```cpp
int idx = 0;
for (int b = 0; b < K_BITS; b++)
    idx |= ((plane_word[b] >> i) & 1) << b;
```

Then the dequantized value is: `codebook[idx] * absmax`

### 3.2 Why Bit-Planes (Not Contiguous Packing)

**Contiguous packing** would pack K-bit indices sequentially into uint32 words.
For K=4: 8 elements per word (clean). For K=3: 10.67 elements per word
(elements straddle word boundaries). For K=5: 6.4 elements per word (also
straddles).

Bit-plane format was chosen because:

1. **Uniform across all K**: K=2,3,4,5 all work identically. No special cases
   for cross-word boundary extraction.
2. **Same memory footprint**: Both formats use K*4 bytes per 32 elements.
3. **Already proven**: The quantize kernel produces bit-planes via `__ballot_sync`.
   The dequant kernel reads them. No format conversion needed.
4. **Produced naturally by warp primitives**: `__ballot_sync` produces one bit-plane
   word per call. This is the idiomatic CUDA way to pack warp-level boolean results.

**Disadvantage**: Extracting one element's index requires K shift+mask+OR operations
(one per bit-plane), creating a serial dependency chain. This is the main source
of ALU overhead in the inner loop. See Section 22 for the full analysis of why
this matters and why it cannot be fixed by inner-loop tweaks alone.

### 3.3 Memory Layout: Flat vs Tiled

The quantize kernel produces a **flat** layout: block 0's K words, then block 1's
K words, etc. The GEMM kernel needs a **tiled** layout organized by
(k_tile, n_tile) for efficient loading into shared memory.

A **repack kernel** transforms flat → tiled. The tiled layout places all data for
one GEMM tile (TILE_K=64 × TILE_N=128) contiguously in memory, enabling bulk
`cp.async` copies from global to shared memory.

---

## 4. Codebook and Absmax Encoding

### 4.1 Codebook

Generated by `create_normal_float_codebook(k)` in `bitsandbytes/functional.py`.
It places 2^K reconstruction levels at the expected values of N(0,1) within 2^K
equiprobable bins, then normalizes to [-1, 1].

Properties:
- Sorted ascending
- Roughly symmetric around 0
- Normalized so `abs(max) == 1.0`
- Cached per (k, device) pair
- Stored as float32, converted to half/bf16 at kernel startup

For K=4, this is conceptually similar to NF4 (bitsandbytes' flagship 4-bit format
used in QLoRA), with minor numerical differences.

### 4.2 Codebook in the GEMM Kernel

The codebook has at most 2^5 = 32 entries (for K=5). The kernel stores the
codebook in **warp registers**: each lane holds one codebook entry. Lookup is
via `__shfl_sync(mask, cb_h, idx)` — a warp shuffle that broadcasts lane `idx`'s
value to the requesting thread.

This is fundamentally different from Marlin's approach, where dequantization is
a linear bit manipulation (shift + subtract). Our codebook lookup is arbitrary
(any mapping from index to value), which makes it more flexible but also means
we can't use the same bitwise tricks Marlin uses.

### 4.3 E4M4 Absmax Format

The absmax (maximum absolute value per block of 32 elements) is encoded as a
uint8 in E4M4 format: 4-bit exponent, 4-bit mantissa. This provides a dynamic
range of ~2^-10 to ~240 with 6.25% relative precision per block.

Decode function: `decode_e4m4_absmax(uint8_t raw) -> float32`
- Extracts exponent and mantissa fields
- Constructs IEEE 754 float via bit manipulation
- Handles normal and subnormal (exponent=0) cases

In the production kernel, a **branchless** variant is used that eliminates
conditional branches for raw==0 and subnormal cases. This removes BSSY/BSYNC
divergence-handling pairs from the SASS output (see Section 19.2).

---

## 5. Source Materials Studied

### 5.1 Design Document

`agents/kbit_gemm_context.md` (in the main bitsandbytes repo): ~1400 lines
covering the complete design context. Sections include existing kbit
implementation, Marlin kernel architecture, GEMM kernel design, weight storage
format, inner loop design, persistent kernel, pipeline, codebook handling,
performance analysis, dispatch, and file organization.

### 5.2 Marlin Kernel (vLLM Reference)

From `~/git/vllm/csrc/quantization/marlin/`:

- **`marlin_template.h`** (~2070 lines): Main kernel template. Key sections:
  stripe partitioning (line 271-281), pipeline wait/fence (line 916-923),
  register fetch from shmem (line 927-939), `matmul()` inner loop with
  dequant + scale + MMA (line 1167-1285), main K-loop (line 1780-1813),
  output reduction (line 1839-2068).

- **`dequant.h`** (~610 lines): Dequantization using `lop3` (3-input logical
  op) and `prmt` (byte permutation) PTX. These are bitwise operations that
  reinterpret INT4/INT8/FP4/FP8 as FP16/BF16 by manipulating IEEE 754 bits.
  **Key insight**: Marlin's dequant is a linear mapping; ours is an arbitrary
  codebook lookup. This is the fundamental difference.

- **`marlin_mma.h`** (~270 lines): MMA instruction wrappers. Inline PTX for
  `m16n8k16` instructions with fp32 accumulators. Also contains the Turing
  `mma_trans()` decomposition (m16n8k16 → two m16n8k8) which was critical for
  understanding the A-fragment register ordering (see Section 15, Stage 3 bug).

- **`marlin.cu`** (~530 lines): Host dispatch with priority-ordered thread configs.

### 5.3 Existing kbit CUDA Kernels

From `feature/kbit-quantization` branch, `csrc/ops.cu`:

- **`kQuantizeBlockwise_kbit<T, K>`** (line 682): Quantize kernel. Per warp:
  load element → reduce absmax → normalize → brute-force codebook search →
  pack via `__ballot_sync`.

- **`kDequantizeBlockwise_kbit_vec<T, K, ...>`**: Vectorized dequant kernel.
  Each warp processes 4 blocks. Loads codebook into lane registers, broadcasts
  bit-planes via shuffle, unpacks indices, looks up codebook, scales by absmax.

- **`decode_e4m4_absmax`**: E4M4 uint8 → float32 via IEEE 754 bit manipulation.

---

## 6. Design Interview and Hardening

The kernel design was hardened through a structured CUDA-specific technical
interview covering ~29 questions across:

- Memory access patterns (bank conflicts, coalescing, cache behavior)
- Warp execution model (fragment mapping, divergence, shuffle usage)
- Synchronization and correctness (atomics, fences, race conditions)
- Precision and numerical behavior (accumulation, type conversions)
- Resource pressure (registers, shared memory, occupancy)
- Edge cases (alignment, partial tiles, min/max sizes)
- Integration (data layout, Python bindings, workspace management)
- Performance model (targets, bottlenecks, degradation modes)

Each design decision below captures the question asked, the options considered,
the choice made, and the reasoning. See the Appendix at the end of this section
for the complete interview question log.

### Interview Question Log

1. FragB column mapping across N-blocks → detailed analysis in Section 8
2. Atomic ordering in split-K → `__threadfence()` needed (Section 7.3)
3. K_dim alignment with TILE_K → partial K-tile handling (Section 7.8)
4. Minimum compute capability → sm_80+ only (Section 7.17)
5. B-tile bank conflicts → +1 padding per column (Section 7.2)
6. First contributor store pattern → plain store + fence (Section 7.3)
7. Partial K-tile implementation → runtime branch, rarely taken (Section 7.8)
8. A-tile swizzle → XOR-based (Section 7.10)
9. C output write coalescing → stage through shared memory (Section 7.11)
10. N alignment → require N % 128 == 0 (Section 7.8)
11. Pipeline depth → 4 stages originally, settled on 2 in production (Section 7.5)
12. bf16 support → from day one (Section 7.15)
13. Accuracy bar → both allclose and SQNR tests (Section 12)
14. Repack testing → Python reference + CUDA validation (Section 14)
15. Workspace allocation → PyTorch caching allocator (Section 7.19)
16. Performance targets → ~4x at M=1, measure and iterate (Section 11)
17. K=5 codebook using all 32 lanes → test explicitly (Section 9)
18. Grid sizing → min(SMs, total_work) (Section 7.12)
19. B-load coalescing → linear mapping, strided loop (Section 7.13)
20. Shared memory budget → fits, no concern (Section 10)
21. Weight layout → accept [N, K_dim], transpose in repack (Section 7.7)
22. Minimum problem size → always use fused kernel (Section 7.18)
23. Register pressure → 1 block/SM is fine (Section 7.14)
24. Partial M-tiles → predicated cp.async + masked write (Section 7.9)
25. Warp layout → adapts to M_BLOCKS (Section 7.6)
26. Template instantiations → 40 variants, manageable (Section 7.16)
27. fp32 vs fp16 accumulation → fp32 always (Section 7.4)
28. K=3, K=5 handling → bit-plane format handles uniformly (Section 9)
29. Non-standard codebook → test with one case (Section 12)

---

## 7. Design Decision Record

### 7.1 Bit-Plane Format

**Decision:** Keep bit-plane format. Do not convert to contiguous packing.

**Why:** Bit-plane format works uniformly for K=2,3,4,5 without cross-word
boundary handling. Same memory footprint. Produced naturally by `__ballot_sync`
during quantization. The ALU cost of K shift+mask+OR operations per element
runs on INT32 units, which was originally expected to overlap with tensor core
MMA execution. (In practice, `mma.sync` prevents this overlap on Ada — see
Section 22 for the full analysis.)

**Disadvantage (discovered later):** The bit extraction creates a serial
dependency chain of ~12 dependent operations for K=4, contributing to the
instruction-limited bottleneck on L2-resident MoE shapes. This was identified
as unfixable via inner-loop tweaks alone (Section 22.5).

### 7.2 Shared Memory Bank Conflicts (B-tile)

**Problem:** Shared memory has 32 banks, 4 bytes each. The B-tile stride is
`2 * K` words per column. For K=4: stride=8, `gcd(8, 32) = 8`, meaning only 4
unique banks for 8 column groups → **2-way bank conflict** on every B-tile read.

Bank conflict analysis per K:
```
K=2, stride=4:  gcd(4, 32) = 4   → 8 unique banks → no conflict
K=3, stride=6:  gcd(6, 32) = 2   → 8 unique banks → no conflict
K=4, stride=8:  gcd(8, 32) = 8   → 4 unique banks → 2-way conflict!
K=5, stride=10: gcd(10, 32) = 2  → 8 unique banks → no conflict
```

K=4 is the most important bit-width (NF4, GPTQ, AWQ all use 4-bit).

**Design fix:** +1 padding per column, making `stride = 2 * K + 1`. An odd
stride is always coprime with 32 (gcd(odd, 32) = 1), eliminating all conflicts.
Memory cost: 128 * 1 * 4 bytes * stages = 2 KB extra. Negligible.

**Implementation status:** The +1 padding fix was designed but NOT implemented
in the production kernel. An attempt to implement it (Section 20) showed that
replacing cp.async with per-column copies (needed to handle padding gaps) added
more overhead than the bank conflict savings. The production kernel retains the
2-way bank conflict for K=4. This is acceptable because the bank conflicts are
not the dominant bottleneck.

### 7.3 Atomic Ordering in Split-K

**Problem:** When multiple blocks contribute partial sums to the same output
tile, a race condition exists: Block B could see the incremented counter but
read stale workspace values if Block A's store hasn't become globally visible.

**Fix:** `__threadfence()` between workspace write and counter increment:
```cpp
write_to_workspace(frag_c, workspace, ...);
__threadfence();  // ensures all prior writes are globally visible
int count = atomicAdd(&tile_counter[mn_id], 1);
```

The first contributor uses a plain store (not atomicAdd) to write its partial
result. This is safe because the first contributor is the only writer at that
time, and `__threadfence()` ensures visibility before the counter increment.

Cost: ~50-100 cycles per output tile per block. Negligible (<0.1% of total time).

### 7.4 fp32 vs fp16 Accumulation

**Decision:** fp32 accumulation exclusively. Convert to fp16/bf16 only at output.

**Why:** Quantization error (~6% per element for K=4) is per-element, bounded,
and partially cancels across the reduction dimension. Accumulation error is
systematic and grows with reduction length — after ~1000 fp16 additions, small
products are rounded away entirely. fp32 accumulation (23-bit mantissa) prevents
this. DeepSeek demonstrated the quality impact in production MoE models.

For M<=32 (our target): fp32 accumulation is free — the kernel is waiting on
memory bandwidth, not tensor core throughput. No tradeoff.

### 7.5 Pipeline Depth

**Original design:** 4 stages. Hides 3 K-tiles of latency (~300-600 cycles),
covering worst-case global memory latency.

**Production kernel:** Uses 2-stage double buffering. The production kernel's
inner loop has ~1264 SASS instructions per k_tile (Section 22.2), which provides
plenty of latency hiding even with just 2 stages.

Shared memory cost per stage (TILE_M=64, TILE_N=128, K=5 worst case): ~14 KB.
2 stages: ~28 KB. 4 stages: ~56 KB. All fit on the RTX 4090's 100 KB.

### 7.6 Warp Layout and M_BLOCKS Dispatch

256 threads = 8 warps, arranged in a 2D grid:

| M_BLOCKS | TILE_M | Layout | Per-warp sub-tile |
|:--------:|:------:|:------:|:------------------|
| 1 | 16 | 1×8 | 16 rows × 16 cols |
| 2 | 32 | 2×4 | 16 rows × 32 cols |
| 3 | 48 | 2×4 | variable |
| 4 | 64 | 2×4 | 32 rows × 32 cols |

Host-side dispatch selects M_BLOCKS as a template parameter:
```cpp
if (M <= 16)      m_blocks = 1;
else if (M <= 32) m_blocks = 2;
else if (M <= 48) m_blocks = 3;
else              m_blocks = 4;
```

No runtime branches in the inner loop — warp layout is compile-time.

### 7.7 Weight Layout and Repack Convention

The repack kernel accepts PyTorch's native `[N, K_dim]` layout. Transpose is
handled internally via index math. Users do not need `.t().contiguous()`.

User-facing API:
```python
packed, absmax, codebook = quantize_kbit(W)      # W is [N, K_dim]
packed_tiled, absmax_tiled = repack_for_gemm(packed, absmax, K_dim, N, k)
C = kbit_gemm(A, packed_tiled, absmax_tiled, codebook, K_dim, N, k)
```

### 7.8 N and K Alignment

- **N:** Must be divisible by TILE_N (128). All common LLM weight matrices
  satisfy this. If not, pad at the Python level and trim output.
- **K_dim:** Must be divisible by 32 (the quantization blocksize). When
  K_dim % TILE_K (64) != 0, the final K-tile is partial, handled by a runtime
  branch that is rarely taken and has negligible misprediction cost.

### 7.9 Partial M-tile Handling

When M is not divisible by TILE_M, the last M-tile has fewer valid rows.
- **A loads:** `cp.async` with predicate `row < M`. Out-of-bounds rows get
  zero-filled in shared memory.
- **MMA:** Operates on whatever data is in fragments. Zero rows → zero output.
- **C writes:** Predicated `row < M` check before writing. Invalid rows skipped.

### 7.10 A-tile XOR Swizzle

Without swizzle, the A tile stored with stride TILE_K=64 halves (128 bytes)
causes every row to start at the same bank → 8-way bank conflicts during
`ldmatrix`.

Fix: XOR-based swizzle at 8-half (16-byte) granularity:
```cpp
col_group = col / 8;
swizzled_group = col_group ^ (row % 8);
swizzled_col = swizzled_group * 8 + (col % 8);
```

Applied during A tile write to shmem AND in the ldmatrix address calculation.
Distributes 8 threads across 8 different banks (zero conflicts).

### 7.11 C Output Write Strategy

Stage output through shared memory for coalesced writes:
1. Each warp writes FragC to shmem in row-major order (reusing pipeline shmem)
2. `__syncthreads()` ensures all writes complete
3. Threads read from shmem in a coalesced pattern, write to global C

For split-K workspace writes (fp32, temporary), direct writes are used without
staging since they're not on the critical path.

### 7.12 Grid Sizing

Grid = `min(num_SMs, total_work_items)`. Standard persistent kernel approach.
The kernel launches a fixed number of blocks that loop over work items. With
high register usage, only 1 block fits per SM, so grid = num_SMs effectively.

### 7.13 B-tile Load Coalescing

Simple linear thread-to-word mapping with strided loop for `cp.async`:
```cpp
int total_int4s = TILE_N * (TILE_K / 32) * K_BITS / 4;  // compile-time
for (int i = threadIdx.x; i < total_int4s; i += blockDim.x)
    cp_async4(&sh_b_int4[i], &B_global[b_offset + i]);
```

Works for all K values. Alignment is always satisfied (tile sizes are multiples
of 16 bytes). The B tile is small relative to A (2-5 KB vs 8 KB), so even
partial thread utilization doesn't affect performance.

### 7.14 Register Pressure and Occupancy

Per thread (K=4, M_BLOCKS=4, worst case):
- FragC accumulators: 32 MMA positions × 4 floats = 128 registers
- FragA (double-buffered): 4 M_BLOCKS × 2 buffers × 4 regs = 32 registers
- Other (bit-planes, codebook, absmax, loop vars): ~20 registers
- **Total: ~180 registers per thread**

With 256 threads: 46,080 registers per block. A100 has 65,536 → 1 block per SM.
Occupancy: 256/2048 = 12.5%.

**Why 1 block/SM is fine:** Standard for high-performance GEMM. Marlin also
runs at 1 block/SM. The cp.async pipeline provides instruction-level parallelism
that substitutes for thread-level parallelism.

### 7.15 bf16 Support

Supported from day one, templated on `scalar_t`. Changes for bf16:
- MMA PTX instruction (different opcode, same performance on Ada)
- Codebook conversion: `__float2bfloat16()` instead of `__float2half()`
- Output conversion: same
- `ldmatrix`: unchanged (both are 16-bit types)

Doubles template instantiations from 16 to 32 variants. Manageable.

### 7.16 Template Instantiations

```cpp
template <int K_BITS, int M_BLOCKS, typename scalar_t>
__global__ void kbit_gemm_prod(...);
```

- K_BITS: 2, 3, 4, 5 (4 values)
- M_BLOCKS: 1, 2, 3, 4 (4 values)
- scalar_t: half, nv_bfloat16 (2 values)
- GEMM kernel: 32 variants
- Repack kernel: 8 variants
- Total: 40 variants, ~5-15 minutes full build

### 7.17 Target Architecture

sm_80+ (Ampere and newer). No Volta (sm_70) or Turing (sm_75). Required for
`cp.async` (async global-to-shared memory copy).

Tested on:
- RTX 4090 (sm_89, primary development hardware)
- Targets: A100 (sm_80), H100 (sm_90)

### 7.18 Minimum Problem Size

Always use the fused kernel. No fallback to dequant + cuBLAS. The kernel is
never wrong for small problems, just potentially microseconds slower. Simplicity
of "always fused" outweighs micro-optimization for edge cases.

### 7.19 Workspace Allocation

When split-K is active:
- **fp32 workspace:** `[M, N]` float32 for partial sum accumulation
- **Tile counters:** `[m_tiles * n_tiles]` int32 for last-contributor detection

Allocated via PyTorch's caching allocator (`torch.empty()`). Tile counters
zeroed via `zero_()` before each GEMM call (~1 us async memset). When split-K
is not needed (common case for large M), no workspace is allocated.

---

## 8. Tensor Core Fragment Layout

### 8.1 The m16n8k16 MMA Instruction

The fundamental compute primitive:
```
D[16,8] = A[16,16] * B[16,8] + C[16,8]
```
with A in row-major, B in column-major, fp16/bf16 inputs, fp32 accumulators.

### 8.2 B-Fragment Thread Mapping

For B (k=16 rows, n=8 columns), each thread (lane 0-31) owns 4 elements as
2 half2 values:
```
b[0] (half2): rows {2*(lane%4), 2*(lane%4)+1}, column = lane/4
b[1] (half2): rows {2*(lane%4)+8, 2*(lane%4)+9}, column = lane/4
```

Critical property: **all 4 elements a thread needs are in the SAME column.**
Threads 0-3 access column 0, threads 4-7 access column 1, etc.
- Column index: `lane_id / 4` (integer division)
- 4 threads share each column → 4-way broadcast on shmem reads
- 8 distinct columns per warp → 8 different shmem addresses

### 8.3 N-Block Extension

Each MMA covers 8 columns. To cover a larger warp sub-tile, iterate over
N-blocks:
```
tile_column = warp_n_offset + nb * 8 + lane_id / 4
```

### 8.4 Row Mapping for Dequantization

Within a column, the 4 rows a thread needs:
```
row_base = 2 * (lane_id % 4)
rows = {row_base, row_base+1, row_base+8, row_base+9}
```

For lane 0: rows {0, 1, 8, 9}. For lane 1: rows {2, 3, 10, 11}. Etc.

These rows are positions within a block of 32 elements (one bit-plane word).
To extract the index for row `r`, extract bit `r` from each K bit-plane word.

### 8.5 A-Fragment Register Ordering Bug (Stage 3)

**Critical finding:** The PTX ISA documentation describes fragment coordinates
but does NOT clearly specify register ordering for m16n8k16. The correct
ordering was discovered by examining Marlin's `mma_trans()` function, which
decomposes m16n8k16 into two m16n8k8 calls.

**Wrong ordering (caused half the k-accumulation to be lost):**
```
frag_a[0] = (row_lo, k_lo)   ← correct
frag_a[1] = (row_lo, k_hi)   ← WRONG position
frag_a[2] = (row_hi, k_lo)   ← WRONG position
frag_a[3] = (row_hi, k_hi)   ← correct
```

**Correct ordering (Turing decomposition):**
```
frag_a[0] = (row_lo, k_lo)   ← for first m16n8k8
frag_a[1] = (row_hi, k_lo)   ← rows interleaved BEFORE k-halves
frag_a[2] = (row_lo, k_hi)   ← for second m16n8k8
frag_a[3] = (row_hi, k_hi)
```

**Lesson:** Always verify MMA fragment ordering against Marlin's implementation,
not just the PTX ISA documentation.

---

## 9. K-Value Analysis: Why K=3 and K=5 Are Not Special

K=3 and K=5 are odd numbers that don't divide 32. The concern was whether they
need special handling.

**Analysis:** Nothing varies except:

| Aspect | K=2 | K=3 | K=4 | K=5 |
|--------|-----|-----|-----|-----|
| B-tile size/stage | 2 KB | 3 KB | 4 KB | 5 KB |
| Dequant ALU ops/elem | 2 | 3 | 4 | 5 |
| Codebook entries | 4 | 8 | 16 | 32 |
| Compression ratio | 7.1x | 4.9x | 3.8x | 3.0x |
| Bank conflicts (unpadded) | None | None | **2-way** | None |

The `#pragma unroll` loop unrolls to the appropriate count. `__ballot_sync`
produces K words regardless of K being odd. `__shfl_sync` handles all codebook
sizes (reads from lane `idx % 32`). The strided cp.async loop handles all B-tile
sizes.

**If contiguous packing had been chosen instead:** K=3 (10.67 elements/word)
and K=5 (6.4 elements/word) would require cross-word boundary extraction code.
Bit-plane format avoids this entirely.

---

## 10. Shared Memory Budget Analysis

### Per-Stage Breakdown (TILE_M=64, TILE_N=128)

| Component | K=2 | K=3 | K=4 | K=5 |
|-----------|----:|----:|----:|----:|
| A tile (fp16) | 8,192 B | 8,192 B | 8,192 B | 8,192 B |
| B tile (packed) | 2,048 B | 3,072 B | 4,096 B | 5,120 B |
| B padding (+1/col) | 512 B | 512 B | 512 B | 512 B |
| Absmax (E4M4) | 256 B | 256 B | 256 B | 256 B |
| **Per stage** | **11,008** | **12,032** | **13,056** | **14,080** |

**2 stages (production):**

| K | Total shmem | 4090 (100 KB) | A100 (164 KB) | H100 (228 KB) |
|---|-------------|:-------------:|:-------------:|:--------------:|
| 2 | 22 KB | 22% | 13% | 10% |
| 3 | 24 KB | 24% | 15% | 11% |
| 4 | 26 KB | 26% | 16% | 11% |
| 5 | 28 KB | 28% | 17% | 12% |

**4 stages:**

| K | Total shmem | 4090 (100 KB) |
|---|-------------|:-------------:|
| 2 | 44 KB | 44% |
| 5 | 56 KB | 56% |

All configurations fit with substantial headroom. The C output staging area
(reusing pipeline shmem) needs TILE_M × TILE_N × 2 = 16 KB max.

For smaller M_BLOCKS (M_BLOCKS=1, TILE_M=16): A tile shrinks to 2 KB per stage.
Per stage drops to ~7-10 KB.

---

## 11. Performance Model and Roofline

### 11.1 Arithmetic Intensity

Per thread block per K-tile (TILE_M=64, TILE_N=128, TILE_K=64, K=4):
- Compute: 262,144 FLOPs
- Memory: 12,544 bytes (A: 8,192 + B: 4,096 + absmax: 256)
- Intensity: **20.9 FLOP/byte**

Compare fp16 GEMM (same tiles, B in fp16):
- Memory: 24,832 bytes
- Intensity: 10.6 FLOP/byte

The kbit kernel has ~2x higher arithmetic intensity due to compressed weights.

### 11.2 RTX 4090 Roofline

- Peak fp16 tensor: 83 TFLOPS
- Peak bandwidth: ~1 TB/s
- Ridge point: 83 FLOP/byte

| M | Intensity | Regime | Expected vs fp16 |
|---|-----------|--------|:----------------:|
| 1 | ~3 | Memory-bound | ~3.8x |
| 8 | ~24 | Memory-bound | ~2.5x |
| 32 | ~93 | Near ridge | ~1.5x |
| 128 | ~296 | Compute-bound | ~1x |

### 11.3 The Data Advantage (Fundamental)

The kernel reads **3.6x less data** than cuBLAS for K=4. This is a real,
consistent advantage. If per-byte execution overhead matched cuBLAS, every
shape would achieve 3.5-3.7x speedup:

| Layer | kbit data | cuBLAS data | If overhead matched |
|-------|----------:|------------:|:-------------------:|
| Qwen3 gate/up (2048×5120) | 5.7 MB | 21.1 MB | **3.7x** |
| GLM4.7 shared gate/up (2048×10240) | 11.3 MB | 42.1 MB | **3.7x** |
| Llama3-8B gate/up (4096×14336) | 31.5 MB | 117.7 MB | **3.7x** |
| Llama3-70B gate/up (8192×28672) | 125.3 MB | 470.3 MB | **3.8x** |

The entire optimization problem is reducing per-byte overhead to match cuBLAS.

---

## 12. Correctness Verification Strategy

### Two-Pronged Approach

1. **Reference match (`torch.allclose`):** Compare fused GEMM against
   `torch.matmul(A, dequant_kbit(W).T)`. Tolerance: `rtol=0.1, atol=0.1 *
   output_mean` to account for E4M4 absmax error propagation.

2. **SQNR-based:** Signal-to-Quantization-Noise Ratio between fused GEMM and
   unquantized fp16 GEMM. Target: SQNR > 10 dB for K=4 (quantization noise
   dominates; fused kernel should not add measurable additional noise).

Both are needed: reference match catches logic bugs (wrong indices, scales,
accumulation). SQNR catches precision degradation beyond what quantization
should introduce.

### Tolerance Calibration

The fused GEMM goes through E4M4 absmax encode/decode (6.25% precision), while
direct reference uses float32 absmax. For near-zero output values, relative
error becomes huge even with tiny absolute error. Tests use:
- `rtol=0.1` (10% relative)
- `atol=0.05-0.1 * C_direct.abs().mean()` (absolute, scaled to output magnitude)

---

## 13. Implementation Stage 1: Python Reference

### What Was Built

File: `tests/test_kbit_gemm.py`

Contains:
- Helper functions (codebook generation, quantize/dequant/pack/unpack refs,
  E4M4 encode/decode)
- `repack_kbit_ref()`: Python reference repack (flat → tiled)
- `unrepack_kbit_ref()`: Python reference unrepack (tiled → flat)
- `kbit_gemm_ref()`: Reference fused GEMM (via unrepack + dequant + matmul)
- `kbit_gemm_ref_direct()`: Direct reference (quantize → dequant → matmul)

### Test Results: 38 tests passing

**TestRepackRef (24 tests):**
- `test_repack_round_trip` [K=2,3,4,5]: bit-exact round-trip
- `test_repack_tile_contiguity` [K=2,3,4,5]: correct output sizes
- `test_repack_various_sizes` [4 sizes × 4 K]: works for aligned dims

**TestFusedGemmRef (14 tests):**
- `test_gemm_matches_direct` [K=2,3,4,5]: matches direct reference
- `test_gemm_m1` [K=2,3,4,5]: works for M=1
- `test_gemm_various_batch_sizes` [M=1,4,16,32]: works across batch sizes
- `test_gemm_fp16_output_quality`: SQNR > 10 dB vs unquantized fp16
- `test_gemm_nonstandard_codebook`: works with asymmetric codebook

---

## 14. Implementation Stage 2: CUDA Repack Kernel

### What Was Built

File: `csrc/ops.cu` (appended to existing kbit code)

The repack kernel transforms flat bit-plane packed data into the GEMM-tiled
layout. Each CUDA thread block handles one output tile. Simple gather/scatter —
no tensor cores, no shared memory pipeline.

Output layout: one tile (TILE_K=64 × TILE_N=128) contains all packed bit-plane
words and E4M4 absmax values for one GEMM inner loop iteration, enabling
contiguous `cp.async` copies.

### Test Results: 25 passing (89 total cumulative)

- `test_repack_matches_reference` [K=2,3,4,5]: bit-exact uint32 match
- `test_repack_output_sizes`: correct buffer sizes
- `test_repack_round_trip_with_gemm` [K=2,3,4,5]: repacked data → correct GEMM
- `test_repack_various_sizes` [4 sizes × 4 K]: works for 128-256 dims

No issues encountered.

### Commit: bff83e6

---

## 15. Implementation Stage 3: Minimal CUDA GEMM

### What Was Built

Function `kbit_gemm_minimal<K_BITS>` in `csrc/ops.cu`.

The minimal GEMM validates all core math without async pipeline:
- Synchronous shared memory loads
- Grid: (n_tiles, m_tiles), 256 threads (8 warps) per block
- TILE_M=16, TILE_K=64, TILE_N=128
- Each warp: 16 columns (2 MMA N-blocks of 8 columns each)
- 4 k-sub-tiles per TILE_K
- Codebook via `__shfl_sync` lookup
- E4M4 absmax decoded on the fly

### The MMA A-Fragment Register Ordering Bug

**Symptom:** MMA only accumulated k=0..7 instead of k=0..15. C[0,0] was 36
(sum of 1..8) instead of 136 (sum of 1..16). Identity matrix tests passed by
coincidence.

**Root cause:** Fragment registers frag_a[1] and frag_a[2] were swapped. The
hardware expects registers ordered for the Turing m16n8k8 decomposition: rows
interleaved before k-halves.

**How found:** A dump-fragments test kernel showed the data was correct but in
wrong register positions. Comparing against Marlin's `mma_trans()` revealed the
correct interleaved ordering.

**Fix:** Swap frag_a[1] and frag_a[2].

**Lesson:** PTX ISA docs are ambiguous on m16n8k16 register ordering. The Turing
decomposition (two m16n8k8) is the authoritative reference. Always verify
against Marlin.

### Test Results: 13 passing (76 total cumulative)

- `test_gemm_matches_reference` [K=2,3,4,5]: matches Python ref
- `test_gemm_various_sizes` [4 sizes × K=4]: multiple dimensions
- `test_gemm_various_M` [M=1,4,8,16 × K=4]: batch sizes
- `test_gemm_sqnr`: SQNR > 20 dB for K=4 and K=5

### Commit: bff83e6

---

## 16. Implementation Stage 4: cp.async Pipeline

### What Was Built

Replaced synchronous global→shared memory loads with `cp.async` double buffering.

- B tile and absmax via `cp.async.cg.shared.global` (16-byte copies, L2 only)
- A tile loaded synchronously (needs M/K_dim bounds checking)
- 2-stage double buffer
- `cp_async_wait<1>()` inside loop, `cp_async_wait<0>()` to drain

Output is **bit-exact identical** to Stage 3 for all K values. This is a pure
performance change — math is unchanged.

### Test Results: 13 new → 89 total (all pass)

### Commit: 9b155d3

---

## 17. Implementation Stage 5: Split-K

### What Was Built

Split-K support for low-tile-count shapes:
- Multiple blocks share an output tile, each handling a subset of k-tiles
- Partial sums accumulated via `atomicAdd` in fp32 workspace
- Grid: 2D for k_chunks=1, 3D for k_chunks>1
- Last contributor detected via atomic tile counter
- Last contributor converts fp32→fp16 output

### Test Results: 21 new → 110 total (all pass)

### Commit: fdcec9c

---

## 18. Implementation Stage 6: Production Kernel

### 18.1 bf16 Support (commit 24406d2)

New production kernel `kbit_gemm_prod` templates on `scalar_t`. Uses
`if constexpr` to select MMA PTX:
- fp16: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- bf16: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

Helper structs `ScalarOps<T>`, `pack_two<T>`, `mma_m16n8k16<T>` abstract
type-specific operations. 8 kernel variants (4 K × 2 dtypes).

fp16 matches Stage 5 bit-for-bit. bf16 matches Python reference.

**Tests:** 29 new → 139 total (all pass).

### 18.2 ldmatrix + XOR Swizzle (commit b64bb91)

Replaced 8 element-by-element shmem reads per A fragment with a single
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` instruction.

XOR swizzle at 8-half granularity eliminates 8-way bank conflicts. Output is
mathematically identical.

### 18.3 Multi-M-Block Tiling (commit f8a06a3)

Extended production kernel to support M_BLOCKS=1,2,3,4 (TILE_M up to 64).
Template parameter controls warp layout. 195 tests passing after this change.

### 18.4 A-tile cp.async (commit 7cd575b)

Converted A tile loading from synchronous to `cp.async`. Both A and B now use
the async pipeline.

### 18.5 Persistent Kernel (commit 78fb6bb)

Converted to persistent kernel with work distribution across `min(num_SMs,
total_work)` blocks. Auto k_splits heuristic for shapes with low SM utilization.

### 18.6 Initial Benchmark Results (commit 27cf6a2)

RTX 4090, K=4, fp16:

| M | K_dim | N | kbit (us) | cuBLAS (us) | Speedup |
|--:|------:|------:|----------:|------------:|--------:|
| 1 | 4096 | 4096 | 109 | 43 | 0.39x |
| 1 | 4096 | 11008 | 82 | 128 | **1.56x** |
| 4 | 4096 | 11008 | 100 | 121 | **1.21x** |
| 4 | 4096 | 4096 | 92 | 22 | 0.24x |

Wins in memory-bandwidth-bound regime (M=1, large N). Loses in compute-bound
cases due to dequant overhead.

### 18.7 Commit History (Stages 4-6)

```
27cf6a2 Add kbit GEMM benchmark script
b64bb91 Add ldmatrix + XOR swizzle for A-fragment loading in production kernel
24406d2 Add Stage 6 production kernel with bf16 support (139 tests pass)
fdcec9c Add Stage 5 split-K GEMM kernel (110 tests pass)
9b155d3 Add Stage 4 pipelined GEMM kernel with cp.async double-buffering (89 tests pass)
```

---

## 19. Optimization Phase 1: Inner Loop Tweaks

After the production kernel was functionally complete with 195 tests passing,
three Phase 1 optimizations were applied to the inner loop.

### 19.1 Two-Tier k_splits Heuristic (commit dc4343b)

**Tier 1 (unchanged):** Aggressive split-K for severe SM underutilization
(< 25% = mn_tiles < num_sms / 4).

**Tier 2 (new):** Conservative split-K (cap 2) when data exceeds L2 cache
(> 24 MB) and SM utilization is moderate. This helps Llama3-8B shapes where
weight data is too large for L2.

**Impact:** Llama3-8B improved ~25% (115us → 87us). MoE shapes unaffected
(their data fits in L2, so adding SMs via k_splits doesn't help — see
Section 22.4 for the full explanation).

### 19.2 Branchless Absmax Decode (commit dc4343b)

New `decode_e4m4_absmax_branchless()` eliminates two conditional branches
(`if raw == 0`, `if e == 0`) that generate BSSY/BSYNC divergence-handling pairs
in SASS. Subnormals (absmax < 2^-10) treated as normal path since no real
weight block has absmax this small.

```cpp
// Old: 2 branches → 16 BSSY/BSYNC pairs per TILE_K iteration
if (raw == 0) return 0.0f;
int e = raw >> 4;
int m = raw & 0xF;
if (e == 0) return ldexpf(...);

// New: branchless via predicated select
int e = raw >> 4;
int m = raw & 0xF;
unsigned int ieee = (unsigned int)(e - E4M4_BIAS + 127) << 23 | (unsigned int)m << 19;
float result = __uint_as_float(ieee);
result = (raw == 0) ? 0.0f : result;
```

**Impact:** Eliminates ~512 BSSY/BSYNC convergence points per block. Estimated
2-3us savings, but below 5-10% benchmark noise.

### 19.3 Interleaved Bit Extraction (commit dc4343b)

Interleaved all 4 fragment elements' bit extractions in a single loop over
K_BITS, giving the compiler more ILP across elements and bit-planes:

```cpp
// All 4 elements extracted in parallel per bit-plane iteration
for (int b = 0; b < K_BITS; b++) {
    idx0 |= ((planes[b] >> bit0) & 1) << b;
    idx1 |= ((planes[b] >> bit1) & 1) << b;
    idx2 |= ((planes[b] >> bit2) & 1) << b;
    idx3 |= ((planes[b] >> bit3) & 1) << b;
}
```

**Impact:** Modest ILP improvement. Below benchmark noise for MoE shapes.

### 19.4 Phase 1 Benchmark Results

RTX 4090, M=32, K=4, fp16, after all Phase 1 changes:

| Layer | kbit (us) | cuBLAS (us) | Speedup |
|-------|----------:|------------:|--------:|
| Qwen3 dense gate/up (2048×5120) | 68 | 22 | 0.32x |
| Qwen3 dense down (5120×2048) | 71 | 26 | 0.37x |
| GLM4.7 shared gate/up (2048×10240) | 73 | 27 | 0.37x |
| GLM4.7 shared down (10240×2048) | 74 | 29 | 0.39x |
| GLM4.7 routed gate/up (2048×1536) | 78 | 28 | 0.36x |
| Llama3-8B gate/up (4096×14336) | 87 | 135 | **1.54x** |
| Llama3-70B gate/up (8192×28672) | 230 | 596 | **2.59x** |

**Phase 1 conclusion:** Marginal inner-loop changes cannot fix MoE shapes.
The problem is structural. See Section 22 for the root cause analysis.

---

## 20. Optimization: B-tile Bank Conflict Fix Attempt

### 20.1 What Was Tried

The design doc specified +1 padding per B-tile column to fix K=4 2-way bank
conflicts. Implementation:

1. Changed B shmem stride from `B_COL_WORDS` (8) to `B_COL_STRIDE` (9)
2. Replaced bulk `cp.async` copy with per-column copies (because padding gaps
   make contiguous copy impossible)
3. Updated all shmem read addresses to use padded stride

### 20.2 Result

**Mixed.** Some shapes got slower (Qwen3 down: 72→90us, GLM4.7 routed: 72→87us).

The bank conflict fix itself should help, but replacing `cp.async` with regular
per-column loads hurt more than the bank conflict savings. The per-column copy
loop adds instruction overhead and loses the async nature of `cp.async`.

### 20.3 Alternative Attempt: XOR Swizzle for B

Considered XOR swizzle instead of padding. But the B-tile read pattern is
per-column broadcast (4 threads share each address), which is fundamentally
simple. The +1 padding is the right fix; the problem is the fetch mechanism.

### 20.4 Decision

**Reverted.** The bank conflict remains for K=4 (2-way, ~4 wasted cycles per
(ks, nb) pair = 32 cycles per k_tile). Not worth the complexity of changing
the fetch mechanism. The bank conflicts are not the dominant bottleneck.

---

## 21. Optimization Phase 2: V2 Kernel (Dequant-During-Fetch)

### 21.1 The Hypothesis

Move all dequantization from the compute phase to the fetch phase. The
`compute_tile` becomes a pure `ldmatrix A` + `ldmatrix B` + MMA loop (~200
instructions per k_tile, down from ~1000). B tile stored as dequantized fp16
in shmem with XOR swizzle for bank-conflict-free `ldmatrix.x2.trans` loading.

**Expected outcome:** Fetch and compute would overlap in the pipeline, reducing
effective per-tile time from max(fetch, compute) to something less than the sum.

### 21.2 Implementation Details

The v2 kernel was written as `kbit_gemm_prod_v2`, compiled successfully, and
passed all 85 production tests with correct results (error within fp16
accumulation tolerance).

Key changes:
- B shmem layout: dequantized fp16, stored as `b_deq[n * TILE_K + k]`
  (n-major, k-minor) with XOR swizzle
- Fetch phase: load raw kbit data → dequantize in registers → store fp16 to shmem
- Compute phase: `ldmatrix.x2.trans` for B + `ldmatrix.x4` for A + MMA
- Used `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` for B fragments

### 21.3 ldmatrix.x2.trans Layout Details

For B stored column-major as `B_shmem[n][k]`, with two 8×8 sub-tiles (k0-7 and
k8-15):

- Thread t provides address for column `t % 8` of sub-matrix `(t / 8) % 2`
- Address: `&b_deq[(n_base + (t%8)) * TILE_K + k_base + (t/8)%2 * 8]`
- 8 elements at that address are contiguous (k varies) → works

XOR swizzle for bank conflicts:
```
swizzled_k_group = (k / 8) ^ (n % 8)
shmem_idx = n * TILE_K + swizzled_k_group * 8 + k % 8
```

### 21.4 Benchmark Results: V2 Did Not Help

| Layer | v1 (us) | v2 (us) | Change |
|-------|--------:|--------:|-------:|
| Qwen3 MoE gate/up (2048×512) | 75 | 70 | -7% |
| Qwen3 dense gate/up (2048×5120) | 72 | 70 | -3% |
| GLM4.7 shared gate/up (2048×10240) | 73 | 130 | **+78%** |
| GLM4.7 shared down (10240×2048) | 80 | 71 | -11% |

### 21.5 Why V2 Failed

Moving dequant from compute to fetch just moved the bottleneck. The pipeline
cannot overlap them because with double-buffered stages, the fetch for tile N+1
must complete before compute can start on it. The total work per k_tile is
unchanged — ~700 ALU instructions for dequant + ~200 for MMA, regardless of
which phase they run in.

V2 also added overhead:
- B shmem grew from 4 KB to 16 KB per stage (dequantized fp16 vs packed
  bit-planes), increasing shmem pressure
- Lost `cp.async` for B (replaced with regular global loads + shmem stores)
- 32 scalar stores per thread per quantization block to shmem

### 21.6 Why Overlap Strategies Fail on Ada (sm_89)

Three overlap approaches were analyzed:

**Option A (multi-stage pipeline):** More stages let fetch and compute overlap
across different tiles. But fetch is 3.5x longer than compute, so even 4 stages
can't hide it. Critical path is always the fetch (dequant).

**Option B (dequant during MMA in same warp):** Issue MMA, then do ALU dequant
while tensor cores execute. **Does not work on Ada.** `mma.sync` is synchronous
— the warp stalls until MMA completes (~16-32 cycles). The dequant needs ~300+
cycles. The warp cannot do ALU work while stalled on `mma.sync`.

**Option C (warp specialization):** Split warps into MMA warps and dequant
warps. When an MMA warp stalls on `mma.sync` (~30 cycles), the scheduler
switches to a dequant warp. Problem: dequant is 10-40x more work than MMA.
MMA warps idle most of the time. Overlap recovers at most ~10% of dequant cost.

### 21.7 Decision

**V2 kernel reverted.** The v1 inner loop is retained as-is. For MoE shapes,
the performance bottleneck is not the inner loop — it is the low SM utilization
from launching individual expert GEMMs. The grouped expert GEMM is the fix.

---

## 22. Root Cause Analysis: Why MoE Shapes Are Slow

### 22.1 The Numbers

The kernel reads 3.6x less data than cuBLAS. If per-byte overhead matched
cuBLAS, every shape would achieve 3.5-3.7x speedup. Instead MoE shapes run
at 0.3-0.4x. The overhead is not bandwidth — it is instruction count.

For Qwen3 gate/up (K=2048, N=5120):
- kbit data: 5.6 MB. L2 transfer at 2 TB/s: **2.8 us**
- Measured kernel time: **68 us**
- Overhead ratio: **24x**

The kernel spends 24x longer than it would take to simply read the data from L2.

### 22.2 SASS Instruction Breakdown

The compiled kernel has ~1264 SASS instructions per k_tile (M_BLOCKS=2, K=4,
fp16). Per k_tile the inner loop is fully unrolled across 4 k_sub × 2 N_BLOCKS
= 8 pairs:

| Category | Count | % | What |
|----------|------:|---:|------|
| Bit extraction (SHF+LOP3+IMAD) | ~512 | 40% | 4 elements × 4 bits × 4 ops × 8 pairs |
| A fragment load (addr+ldmatrix) | ~160 | 13% | Swizzle address math + ldmatrix, ×8 |
| Fetch + barriers + loop | ~160 | 13% | cp.async issue, __syncthreads, kt loop |
| Absmax decode + convert | ~64 | 5% | shmem load + decode + f2h, ×8 |
| B plane shmem load | ~56 | 4% | 4 loads + addr, ×8 |
| Codebook shuffle (SHFL) | ~32 | 3% | 4 shuffles, ×8 |
| Scale multiply (HMUL) | ~32 | 3% | 4 hmul, ×8 |
| Pack + MMA | ~48 | 4% | 2 pack + 2 MMA, ×8 |
| Other (misc addr, control) | ~200 | 16% | |
| **Total** | **~1264** | | |

**Tensor core MMA: 16 instructions = 1.3%.** The tensor cores are idle 98.7%
of the time. The kernel is an ALU program that occasionally does a matrix
multiply.

### 22.3 Cycle Budget

At 32 k_tiles per block:
- Dynamic instruction count: ~40,000 per thread
- With 2 warps per scheduler (occupancy = 16.7%): ~80,000 cycles per scheduler
- At 2.52 GHz: ~32 us of pure instruction execution
- Add memory stalls + barrier stalls: ~35 us
- Total: ~67 us. **Matches measured 68-78 us.**

### 22.4 Why k_splits Cannot Help MoE Shapes

All Qwen3 and GLM4.7 weight data fits in L2 cache (72 MB on RTX 4090).
Effective bandwidth is ~2 TB/s from L2, not ~1 TB/s from DRAM. With data
already in L2, adding more SMs via k_splits does not increase bandwidth — it
only adds atomicAdd overhead.

Benchmarking confirmed: k_splits=4 for Qwen3 gate/up (31% → 100% SM util)
changed kernel time from 72us to 71us (within noise).

### 22.5 Why Inner Loop Tweaks Have Diminishing Returns

The interleaved bit extraction and branchless absmax reduced instruction count
by ~5-10% = ~60-120 fewer instructions per k_tile. At 32 k_tiles: ~2000-4000
fewer dynamic instructions → ~2-4 us saved out of 68 us. Below benchmark noise.

To get meaningful speedup, we need to remove **hundreds** of instructions per
k_tile, not tens. This is impossible without changing the fundamental approach.

### 22.6 The Fundamental Constraint

On Ada/Ampere/consumer-Blackwell GPUs using `mma.sync`, the ALU dequant work
cannot be hidden behind tensor core execution. The two are serialized within
each warp, and warp-level interleaving provides negligible overlap due to the
extreme ALU:MMA ratio (39:1).

This constraint does NOT apply to Hopper (sm_90a) with `wgmma.mma_async` or
Blackwell datacenter (sm_100a) with `tcgen05.mma`, where MMA is truly
asynchronous.

---

## 23. GPU Architecture Constraints: mma.sync vs wgmma

### 23.1 The Architectural Divide

| GPU | Arch | SM | MMA instruction | Async? | Our approach |
|-----|------|----|-----------------|:------:|:-------------|
| RTX 4090 | Ada | sm_89 | `mma.sync` | No | Grouped GEMM |
| RTX 5090 | Blackwell consumer | sm_120 | `mma.sync` (ext) | No | Grouped GEMM |
| RTX PRO 6000 | Blackwell workstation | sm_120 | `mma.sync` (ext) | No | Grouped GEMM |
| H100/H200 | Hopper | sm_90a | `wgmma.mma_async` | Yes | Dequant-during-MMA viable |
| B200/GB200 | Blackwell DC | sm_100a | `tcgen05.mma` | Yes | Dequant-during-MMA viable |

### 23.2 Verification: sm_120 Uses mma.sync

Confirmed via multiple sources:
- SageAttention issue #291 shows `wgmma.mma_async` produces compiler errors on
  sm_120 targets
- CUDA Toolkit 12.8 forum discussions confirm sm_120 does not support wgmma
- Microbenchmarking papers confirm sm_120 retains synchronous MMA model

Consumer Blackwell (RTX 5090, RTX PRO 6000) gains FP4/FP6 tensor core data
types and more SMs (192 on full GB202 die vs 128 on AD102), but the MMA model
stays synchronous. NVIDIA reserves async MMA for datacenter parts.

### 23.3 Implications

For ALL consumer GPUs (RTX 4090, 5090, PRO 6000):
- The 39:1 ALU:MMA ratio means dequant dominates regardless of scheduling
- The inner loop cannot be made significantly faster
- Grouped expert GEMM (Section 24) is the correct strategy

For datacenter GPUs (H100, B200):
- `wgmma.mma_async` allows the warp to continue ALU work after issuing MMA
- Dequant-during-MMA overlap becomes viable
- A separate codepath using wgmma would benefit even individual expert shapes
- This is a future optimization, not the immediate priority

---

## 24. The Path Forward: Grouped Expert GEMM

### 24.1 Why This Is the Right Approach

Individual MoE expert GEMMs on Qwen3-Coder-Next:
- Expert gate/up: K=2048, N=512 → 4 tiles on 128 SMs (3% utilization)
- Expert down: K=512, N=2048 → 16 tiles (12% utilization)
- Kernel time: ~70-75 us (instruction-limited, L2-resident)
- cuBLAS: ~22-27 us (also underutilized, but lower instruction overhead)

The v1 kernel already achieves ~2x over cuBLAS on large shapes where SMs are
fully utilized (Llama3-8B: 1.5x, Llama3-70B: 2.6x). The compression advantage
is real — it just can't be realized when 97% of SMs are idle.

A grouped expert GEMM batches all active experts into one kernel launch:
- Qwen3-Next inference, batch=32, top-8 routing: 256 expert invocations
  × 4 tiles = 1024 total tiles
- All 128 SMs active, ~8 tiles per SM
- Total weight data: ~32-64 MB across unique experts → DRAM-bound
- Compression advantage applies → expected **~2x over cuBLAS**

### 24.2 API Design

New op: `kbit_grouped_gemm(A_list, B_packed_list, absmax_list, codebook,
K_dim, N, k)` where the lists contain per-expert tensors (or a single
concatenated tensor with offset arrays).

The kernel reuses the v1 inner loop. The persistent work distribution changes:
instead of iterating over (m_tile, n_tile, k_split) for one matrix, it iterates
over (expert_id, m_tile, n_tile, k_split) across all experts.

### 24.3 Implementation Sketch

```cpp
struct ExpertDesc {
    const scalar_t* A;     // [M_expert, K_dim]
    int M;                 // tokens routed to this expert
    int b_offset;          // offset into packed B / absmax arrays
};

// Persistent kernel distributes work across all experts
for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
    auto [expert_id, mn_id, ks_id] = decode_work_id(work_id);
    const auto& desc = experts[expert_id];
    // ... same inner loop as v1 ...
}
```

Expert metadata passed via kernel args or constant memory.

### 24.4 Performance Estimate

With 1024 tiles on 128 SMs and DRAM-bound data:
- Weight read: ~40 MB compressed at 900 GB/s = 44 us
- cuBLAS equivalent: ~40 MB × 3.6 = 144 MB at 900 GB/s = 160 us
- Expected speedup: ~2-3x vs fp16 cuBLAS grouped GEMM
- Per-expert amortized time: ~0.2 us (vs 70 us individually)

### 24.5 Why the V1 Inner Loop Is Good Enough

The inner loop at 1264 instructions per k_tile is instruction-limited when data
is L2-resident (MoE shapes). But when the grouped GEMM makes the kernel
DRAM-bound (total data across experts exceeds L2), the instruction execution
overlaps with the longer DRAM latency. The 3.6x compression advantage then
translates directly to bandwidth savings.

This is exactly what we observe for Llama-scale shapes: Llama3-70B (117 MB,
DRAM-bound) achieves 2.6x. The grouped expert GEMM should behave similarly.

### 24.6 Implementation Plan

**Step 1:** Grouped expert GEMM kernel. Extend the v1 persistent kernel to
handle multiple experts in one launch.

**Step 2:** Python API and expert batching. New `kbit_grouped_gemm` op.
Python-side logic to collect active experts, build descriptor array, launch
kernel, scatter results.

**Step 3:** Integration with LinearNbit / MoE module. Wire into the MoE
forward pass.

**Step 4 (future):** Hopper/Blackwell datacenter codepath using `wgmma.mma_async`
where dequant-during-MMA overlap is viable.

---

## 25. Risk Register

### Risk 1: A-tile Swizzle Correctness (HIGH) — RESOLVED

Getting XOR swizzle wrong causes silent bank conflicts on `ldmatrix` reads.
Correct results but ~50% shmem throughput.

**Mitigation:** Implemented without swizzle first (Stage 3), then added swizzle
in Stage 6 and verified output unchanged while profiled performance improved.

**Status:** Resolved. Swizzle implemented and tested.

### Risk 2: Repack Index Math (HIGH) — RESOLVED

Single index error silently corrupts all GEMM results. The kernel runs, the
output has the right shape, but values are wrong.

**Mitigation:** Python reference repack enables bit-exact validation. CUDA
repack matches Python element-by-element. Round-trip test provides second layer.

**Status:** Resolved. Bit-exact match confirmed in Stage 2.

### Risk 3: Inter-Block Synchronization in Split-K (HIGH) — RESOLVED

Missing `__threadfence()` or incorrect counter logic causes rare, non-deterministic
wrong results.

**Mitigation:** Code review + `__threadfence()` placement verified + tested with
forced split-K on small problems + many random seeds.

**Status:** Resolved. Split-K passes all tests reliably.

### Risk 4: Register Spilling (MEDIUM) — MONITORED

Compiler uses more registers than estimated, causing spills to local memory.

**Mitigation:** Checked `--ptxas-options=-v` output. No spilling observed. If
it occurs: cap M_BLOCKS at 3, use `__launch_bounds__`.

**Status:** No spilling observed. Monitoring.

### Risk 5: Pipeline Underutilization for Small K_dim (MEDIUM) — ACCEPTED

K_dim/64 < pipeline stages → pipeline never reaches steady state.

**Status:** Not a concern for target use case (K_dim >= 2048).

### Risk 6: MMA Fragment Ordering (HIGH) — RESOLVED

PTX ISA docs ambiguous on m16n8k16 register ordering.

**Mitigation:** Discovered and fixed in Stage 3 (Section 15). Now verified
against Marlin's Turing decomposition.

**Status:** Resolved.

---

## 26. File Locations and Worktree Setup

### Worktree

```
~/git/bnb-kbit-gemm/     Branch: feature/kbit-gemm
                          Based on: feature/kbit-quantization
```

Created from main bitsandbytes checkout:
```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-kbit-gemm -b feature/kbit-gemm feature/kbit-quantization
```

### Key Files

| File | Lines | Purpose |
|------|------:|---------|
| `csrc/ops.cu` | 2311 | All CUDA kernels: quantize, dequant, repack, GEMM |
| `tests/test_kbit_gemm.py` | ~1400 | All stage tests (195 total) |
| `benchmarks/bench_kbit_gemm.py` | ~200 | Benchmark script |
| `progress.md` | — | This document |
| `optimization2.md` | ~360 | Phase 2 optimization analysis |
| `bitsandbytes/functional.py` | — | Python kbit API (quantize, dequant, codebook) |
| `bitsandbytes/_ops.py` | — | torch.library op definitions |
| `bitsandbytes/backends/cuda/ops.py` | — | CUDA backend dispatch |
| `csrc/pythonInterface.cpp` | — | C wrappers for repack/GEMM |

### Kernel Source Structure (csrc/ops.cu)

The production kernel `kbit_gemm_prod<K, MB, scalar_t>` is at approximately
line 1782. Key sections within the file:

- Lines ~670-870: Quantize kernel (`kQuantizeBlockwise_kbit`)
- Lines ~870-1100: Dequantize kernel (`kDequantizeBlockwise_kbit_vec`)
- Lines ~1100-1400: Repack kernel
- Lines ~1400-1500: Helper structs (ScalarOps, pack_two, mma_m16n8k16)
- Lines ~1500-1780: Stage 3/4/5 kernels (retained for reference/testing)
- Lines ~1782-2070: **Production GEMM kernel** (`kbit_gemm_prod`)
- Lines ~2070-2311: Launcher and dispatch (`kbitGemmProdLaunch`, etc.)

---

## 27. Full Commit History

```
0d77a61 docs: Rewrite optimization plan — revert v2, focus on grouped expert GEMM
dc4343b Phase 1 inner loop opts: branchless absmax, interleaved extraction, two-tier k_splits
90cd7cf docs: Add SASS analysis and inner loop optimization steps
f301ba1 docs: Rewrite optimization guide around overhead gap analysis
d736ba0 docs: Add MoE model benchmarks and revise optimization roadmap
fc1d1a1 docs: Rewrite optimization guide with real model benchmarks
6e18c03 Tune persistent kernel k_splits threshold and grid sizing
f480540 docs: Update optimization guide with persistent kernel findings
78fb6bb Convert production GEMM to persistent kernel with auto k_splits
6fb6823 docs: Rewrite optimization guide with completed work and updated priorities
7cd575b Convert A tile loading to cp.async and tune M_BLOCKS dispatch
f8a06a3 Add multi-M-block tiling to production GEMM kernel (195 tests pass)
4d51152 docs: Add optimization guide and update progress report
a91c313 docs: Update progress report with Stages 4-6 completion
27cf6a2 Add kbit GEMM benchmark script
b64bb91 Add ldmatrix + XOR swizzle for A-fragment loading in production kernel
24406d2 Add Stage 6 production kernel with bf16 support (139 tests pass)
fdcec9c Add Stage 5 split-K GEMM kernel (110 tests pass)
9b155d3 Add Stage 4 pipelined GEMM kernel with cp.async double-buffering (89 tests pass)
ad64c98 docs: Update progress report with Stages 2-3 completion and MMA bug analysis
bff83e6 Add Stage 2 repack kernel, Stage 3 minimal GEMM kernel (76 tests pass)
f95a7f2 Fix analytical error bound for K=5 with E4M4 absmax
f52b572 Fix lint and formatting issues from CI pre-commit checks
8a2817e Template dequant kernel on output type, add bf16/fp32 native output
03415e1 Remove scalar dequant kernel, fp32 absmax, and Stage 1-3 scaffolding
2973bf5 Add vectorized dequant kernel and E4M4 uint8 absmax support
4b17a2f Remove implementation progress report
2825890 Complete k-bit quantization: Stages 6-8, Python API, 218 tests pass
fb649f1 Fix RDC device linking: move kernels to ops.cu, all 157 tests pass
c39f791 Add k-bit quantization kernels (K=2-5, blocksize=32) -- WIP
```

---

## 28. Current Status

### What's Done

- **Production kernel** (`kbit_gemm_prod`): functionally complete, 195 tests pass
- **Supported configs:** K=2,3,4,5 × M_BLOCKS=1,2,3,4 × fp16/bf16
- **Features:** split-K, persistent kernel, ldmatrix with XOR swizzle, cp.async
  double-buffered pipeline, auto k_splits heuristic
- **Benchmark infrastructure:** bench_kbit_gemm.py

### Performance Summary

| Shape class | Example | vs cuBLAS | Status |
|-------------|---------|:---------:|:-------|
| Large dense (DRAM-bound) | Llama3-70B 8192×28672 | **2.6x** | Good |
| Medium dense (DRAM-bound) | Llama3-8B 4096×14336 | **1.5x** | Good |
| MoE/small dense (L2-resident) | Qwen3 2048×5120 | 0.3x | Blocked: instruction-limited |
| Individual MoE expert | Qwen3 2048×512 | 0.3x | Blocked: 3% SM utilization |

### What Was Tried and Failed

1. **Phase 1 inner loop tweaks** (branchless absmax, interleaved extraction,
   k_splits): marginal improvement on large shapes, no effect on MoE shapes.
2. **B-tile +1 padding for bank conflicts**: replacing cp.async with per-column
   copies added more overhead than it saved. Reverted.
3. **V2 kernel (dequant-during-fetch)**: moved bottleneck but didn't reduce it.
   mma.sync prevents ALU/MMA overlap on Ada. Reverted.

### What's Next

1. **Grouped expert GEMM kernel** — the primary deliverable. Batch all MoE
   expert invocations into one kernel launch, achieving 100% SM utilization
   and DRAM-bound behavior where the 3.6x compression advantage pays off.
2. **Python API and expert batching** — collect active experts, build descriptor
   array, launch kernel, scatter results.
3. **Integration with LinearNbit / MoE module** — wire into MoE forward pass.
4. **Future: Hopper/Blackwell DC codepath** — wgmma-based kernel where
   dequant-during-MMA overlap is viable.

### Key Insight for New Developers

The kernel works. It produces correct results for all K values and both dtypes.
It achieves >2x speedup over cuBLAS for DRAM-bound shapes. The challenge is
purely at the workload distribution level: individual MoE expert GEMMs don't
generate enough tiles to utilize the GPU. The inner loop does not need further
optimization — the grouped expert GEMM is the fix.
