# FLUTE Kernel: Comprehensive Technical Guide

This document provides a thorough analysis of the FLUTE (Flexible Lookup Table Engine)
kernel for lookup-table-quantized LLM inference. It covers the kernel architecture,
implementation details, performance characteristics, and relevance to the bitsandbytes
kbit GEMM kernel design.

---

## Executive Summary: FLUTE vs. Bitsandbytes kbit

FLUTE and the bitsandbytes kbit GEMM kernel are two different approaches to the same
problem — fused dequantization + matrix multiplication for lookup-table-quantized LLM
weights — with comparable instruction-level efficiency.

**They are similar in:**
- Core operation: load compressed weights, dequant via codebook, tensor core MMA
- Instruction count per element: roughly comparable (~3-6 ops depending on bit width)
- Performance regime: both achieve 2-4x over FP16 at small batch, converging to dense
  throughput at large batch (fundamental property of weight-only quantization)
- Both require offline weight repacking for GEMM-friendly tile layout

**FLUTE trades flexibility for per-shape optimization:**
- Built on CUTLASS 3 / CuTe — gets multi-stage pipelining and Stream-K for free
- Requires per-(shape, bits, group_size, GPU) compilation and auto-tuning
- Shape-specialized binaries limit deployment flexibility
- CUTLASS dependency (pinned to v3.4.1)
- 3-bit uses bit-slice decomposition (1+2 split) — different code path, ~33% more
  instructions than 4-bit
- No 5-bit support
- Focused on A100/A6000; RTX 4090 supported but less tuned

**kbit trades CUTLASS infrastructure for simplicity and breadth:**
- Self-contained hand-written CUDA, no external dependencies
- Uniform code path for K=2,3,4,5 via bit-plane format — no special cases
- No per-shape recompilation or tuning needed
- Register-based codebook lookup via `__shfl_sync` (zero memory, 1 cycle)
- E4M4 absmax (1 byte per block of 32) — finer granularity than FLUTE's FP16 scales
- Developed and tested on RTX 4090; not yet tuned for data center GPUs

**Bottom line:** FLUTE does not have a fundamental architectural advantage over the kbit
design. The two kernels have similar instruction-level efficiency with different
engineering trade-offs. FLUTE's head start is that it exists as a working fused GEMM
today and has been benchmarked on data center GPUs. Once the kbit GEMM is implemented
and tuned for A100/H100, there is no reason to expect FLUTE would be meaningfully
faster. The bitsandbytes ecosystem integration (Transformers, PEFT, Accelerate) and
broader bit-width support (K=2-5 uniform) are practical advantages that matter more
than marginal kernel-level performance differences.

FLUTE has limited real-world adoption despite its EMNLP 2024 publication — it is not
a default in any major inference framework and has known issues (shape specialization,
numerical instability at some configurations, bfloat16 underperformance). It is best
understood as an academic contribution that validates the LUT-quantized GEMM approach,
not as a production system to compete against.

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [The Core Problem: LUT-Quantized GEMM on GPUs](#2-the-core-problem-lut-quantized-gemm-on-gpus)
3. [Three-Part Solution Architecture](#3-three-part-solution-architecture)
4. [Offline Weight Restructuring (Section 3.1)](#4-offline-weight-restructuring)
5. [Vectorized Lookup Table with Duplication (Section 3.2)](#5-vectorized-lookup-table-with-duplication)
6. [Stream-K Workload Partitioning (Section 3.3)](#6-stream-k-workload-partitioning)
7. [CUTLASS 3 / CuTe Implementation](#7-cutlass-3--cute-implementation)
8. [Source Code Structure](#8-source-code-structure)
9. [Kernel Configuration and Tuning](#9-kernel-configuration-and-tuning)
10. [NormalFloat and NFL (Learned NormalFloat)](#10-normalfloat-and-nfl-learned-normalfloat)
11. [Performance Analysis](#11-performance-analysis)
12. [Comparison with Other Kernels](#12-comparison-with-other-kernels)
13. [Relevance to Bitsandbytes kbit GEMM](#13-relevance-to-bitsandbytes-kbit-gemm)
14. [Limitations and Known Issues](#14-limitations-and-known-issues)
15. [Links and References](#15-links-and-references)

---

## 1. Overview and Motivation

**Paper**: "Fast Matrix Multiplications for Lookup Table-Quantized LLMs"
**Authors**: Han Guo, William Brandon, Radostin Cholakov, Jonathan Ragan-Kelley,
Eric P. Xing, Yoon Kim
**Published**: EMNLP 2024 (Findings)
**ArXiv**: 2407.10960 (v4, January 17, 2025)

FLUTE is a CUDA kernel engine for efficient inference of weight-quantized LLMs where
the quantization is based on **lookup tables** (LUT) rather than uniform (linear)
integer quantization. This distinction is critical:

- **Uniform quantization** (e.g., standard INT4): `dequant(q) = q * scale + zero`
  Simple arithmetic, easily fused with GEMM.

- **LUT quantization** (e.g., NF4, custom codebooks): `dequant(q) = table[q] * scale`
  Requires a table lookup per element, which is fundamentally different from arithmetic
  dequantization and presents unique GPU optimization challenges.

FLUTE supports arbitrary lookup tables, making it compatible with:
- Integer quantization: int4, int3, int2
- Floating-point: fp4, fp3, fp2
- Normal float variants: nf4, nf3, nf2
- Learned Normal Float (NFL): A learnable extension to QLoRA's nf4
- Custom arbitrary tables (any 2^K values)

At batch sizes < 32 with group size 128 (typical LLM inference), FLUTE achieves
**2-4x speedup** over existing GEMM kernels and **1.5-2x end-to-end throughput
improvement** on LLaMA-3 models.

---

## 2. The Core Problem: LUT-Quantized GEMM on GPUs

The paper identifies three fundamental challenges for building a high-performance
LUT-quantized matmul kernel on GPUs:

### Challenge 1: Tensor Core Data Layout Requirements

Tensor Cores have strict requirements on data types, shapes, and layouts. Quantized
weights at non-standard bit widths (especially 3-bit) cannot be packed evenly into
the 128-bit vectorized memory accesses that feed the tensor core pipeline. For
example:

- 4-bit: 32 values per 128-bit word (clean)
- 3-bit: 42.67 values per 128-bit word (does not divide evenly)
- 2-bit: 64 values per 128-bit word (clean)

The 3-bit case is problematic: you cannot load a clean set of 3-bit values with a
single 128-bit async copy instruction.

### Challenge 2: Dynamic Indexing Limitations

LUT-based dequantization requires dynamic indexing into a table. GPUs do not natively
support efficient dynamic indexing of data in their fastest on-chip storage (registers).
The alternatives are:

- **Registers**: No dynamic indexing. Would need a switch/case statement.
- **Shared memory**: Supports dynamic indexing but has limited bandwidth (32 banks,
  32-bit each) and potential bank conflicts.
- **Constant memory**: Broadcasts to all threads if they access the same address, but
  serializes if they access different addresses.

Since each thread typically looks up a different index, shared memory is the natural
choice, but naive implementations suffer from bank conflicts.

### Challenge 3: Wave Quantization at Small Problem Sizes

With low-bit quantization and small batch sizes, the weight matrix is small, producing
fewer output tiles. If the number of tiles doesn't fill all SMs evenly, some SMs sit
idle in the last "wave" (wave quantization). This is a significant efficiency loss
for the small-matrix regime that LLM inference typically operates in.

---

## 3. Three-Part Solution Architecture

FLUTE addresses these challenges with three complementary techniques:

1. **Offline weight restructuring** (Section 3.1): Reorder quantized weights at
   model-load time so that after dequantization, the data is already in the layout
   that tensor cores expect. This moves bit-manipulation overhead from runtime to
   load time.

2. **Vectorized and duplicated lookup table** (Section 3.2): Store the LUT in shared
   memory, but access two values simultaneously (vectorization) and duplicate the
   table across banks (duplication) to eliminate bank conflicts.

3. **Stream-K workload partitioning** (Section 3.3): Use fine-grained work distribution
   across SMs to minimize wave quantization effects.

---

## 4. Offline Weight Restructuring

### The Problem

Consider 3-bit quantization. Each weight is a 3-bit index into a lookup table.
Packing these into 128-bit words for async copy:

- 128 / 3 = 42.67 — doesn't divide evenly
- You can't load exactly N complete 3-bit values with a single vector load

Standard approaches pad to 4 bits (wasting 25% of storage) or use complex runtime
bit manipulation to extract 3-bit fields from packed words.

### FLUTE's Approach: Bit-Slice Decomposition

FLUTE splits the 3-bit representation into two "bit-slices":
- A **1-bit partition** (the most significant bit)
- A **2-bit partition** (the two least significant bits)

Each partition is stored separately and can be loaded with standard 128-bit async
copy instructions:
- The 1-bit partition: 128 values per 128-bit word
- The 2-bit partition: 64 values per 128-bit word

After loading both slices into registers, they are combined via bit manipulation:

```
combined_index = (bit_slice_1 << 2) | bit_slice_2
```

This avoids any runtime overhead from non-aligned bit extraction.

### Offline Reordering

The quantized weight matrix is permuted offline (at model load time) so that after
the bit-slices are loaded and dequantized, the resulting values are already in the
exact register layout that the `m16n8k16` tensor core instruction expects.

This is possible because the quantized weights are **static** during inference — they
never change. So the permutation is computed once and applied once. At runtime, the
kernel simply loads pre-permuted data and feeds it to tensor cores without any
reordering overhead.

The permutation accounts for:
- The thread-to-element mapping of the MMA instruction
- The shared-memory-to-register copy layout (ldmatrix)
- The bit-slice separation

### For 4-bit Quantization

4-bit is simpler: 32 values per 128-bit word, clean division. No bit-slice
decomposition needed. The offline restructuring still applies — weights are permuted
so that the dequantized layout matches tensor core expectations.

### For 2-bit Quantization

2-bit is also clean: 64 values per 128-bit word. Same approach as 4-bit.

---

## 5. Vectorized Lookup Table with Duplication

### The Problem: Shared Memory Bank Conflicts

The lookup table for dequantization is stored in shared memory. For K-bit
quantization, the table has 2^K entries. When 32 threads in a warp each look up
a different index, the access pattern can cause bank conflicts.

Shared memory has 32 banks, each 4 bytes wide. If two threads access different
4-byte words in the same bank, the accesses are serialized.

For a 4-bit LUT with 16 entries of 2 bytes (half precision) each:
- Total LUT size: 32 bytes
- The 16 half values occupy banks 0-7 (2 half values per 4-byte bank)
- Threads accessing different indices in the same bank conflict

### Vectorized Lookup

FLUTE creates an **expanded lookup table** containing every possible pair of
consecutive indices. Instead of looking up one value at a time, it looks up two
values simultaneously.

For 4-bit quantization:
- Original table: 2^4 = 16 entries of `half` (2 bytes each) = 32 bytes
- Vectorized table: 2^8 = 256 entries of `half2` (4 bytes each) = 1024 bytes

The kernel extracts pairs of 4-bit indices from packed data, forms an 8-bit index,
and uses it to load a `half2` containing both dequantized values in a single shared
memory transaction. This halves the number of shared memory accesses.

For 3-bit quantization:
- Original: 2^3 = 8 entries
- Vectorized: 2^6 = 64 entries of `half2` = 256 bytes

### LUT Duplication

Even with vectorization, bank conflicts can still occur. For the 4-bit vectorized
table (256 × 4 bytes = 1024 bytes), the entries map across 256 banks positions,
cycling through all 32 banks 8 times. If 8 threads in a warp happen to access
entries that map to the same bank, you get an 8-way conflict.

FLUTE mitigates this by **duplicating** the entire vectorized table multiple times
in shared memory, placing each copy at a different base address that shifts the
bank alignment. When a thread would conflict on one copy, it can access a
different copy that maps to a different bank.

The number of duplicates is a tuning parameter. For 4-bit with 256 entries:
- 1 copy: up to 8-way conflicts
- 2 copies: up to 4-way conflicts
- 4 copies: up to 2-way conflicts
- 8 copies: conflict-free (8 KB total — still small vs. 48-164 KB shared memory)

For 3-bit with 64 entries:
- Vectorized table is only 256 bytes
- 2-way conflicts max, so fewer duplicates needed

The duplication count is selected during auto-tuning (see Section 9).

### Implementation Detail

The dequantization in the kernel (`packbits_utils.hpp`) supports multiple modes:

```cpp
enum QuantMapModeEnum {
    Basic,           // Standard per-element LUT lookup
    Vectorized,      // Vectorized half2 lookup (default)
    Vectorized_32,   // Vectorized with 32-entry table
    Vectorized_16,   // Vectorized with 16-entry table
    Vectorized_8,    // Vectorized with 8-entry table
    WarpShuffle,     // __shfl_sync-based lookup (registers)
    Marlin           // Marlin-style arithmetic dequant
};
```

The `Vectorized` mode is the default and primary mode. The `WarpShuffle` mode uses
`__shfl_sync()` for in-register lookups (similar to bitsandbytes' approach). The
`Marlin` mode delegates to Marlin's `lop3`-based arithmetic dequantization for
uniform INT4.

---

## 6. Stream-K Workload Partitioning

### The Problem: Wave Quantization

Standard GEMM kernels partition the output matrix into tiles and launch one
threadblock per tile. If the number of tiles doesn't divide evenly by the number
of SMs, the last wave has idle SMs.

Example: 32 output tiles on 132 SMs (H100). Only 32/132 = 24% utilization.
Even with split-K to create more blocks, the granularity is coarse.

### Stream-K Solution

Stream-K (introduced by CUTLASS) partitions work at a finer granularity than
output tiles. Instead of assigning one complete output tile to each threadblock,
it distributes individual K-tiles across threadblocks.

The work is linearized: all (M-tile, N-tile, K-tile) combinations are laid out
in a 1D sequence and distributed evenly across a fixed number of threadblocks
(typically = num_SMs).

When multiple threadblocks contribute to the same output tile (because they
process different K-ranges), they synchronize via a semaphore-based fixup:

1. Non-finishing blocks store partial accumulator values in a global workspace
2. Synchronization via `cutlass::Barrier` primitives (`wait_lt`, `wait_eq`,
   `arrive_inc`)
3. The finishing block reads, reduces, and writes the final result

### FLUTE's Stream-K Implementation

FLUTE's `TileScheduler` (`tile_scheduler_utils.hpp`) implements both Split-K
and Stream-K modes:

```cpp
enum DecompositionModeEnum {
    SplitK,    // Fixed K-split across slices
    StreamK    // Fine-grained K-tile distribution
};
```

In Stream-K mode:
- Total tiles = `tiles_M × tiles_N × tiles_K`
- `tiles_per_block = total_tiles / num_blocks`
- `blocks_special = total_tiles % num_blocks` (these get one extra tile)

The `FixupHelper` handles the inter-block reduction:
- `BACKWARDS` flag reverses logical block ordering so the last block coordinates
- Partial sums accumulated in FP32 for numerical stability
- Global reduction done in FP16 to minimize memory traffic

---

## 7. CUTLASS 3 / CuTe Implementation

FLUTE is built entirely on **CUTLASS 3.x** (specifically v3.4.1) using the
**CuTe** (CUDA Templates) abstraction layer. This is a significant architectural
choice that differs from hand-written CUDA kernels like Marlin.

### CUTLASS 3.x Architecture Layers

CUTLASS 3.x decomposes GEMM into composable layers:

1. **Device layer**: Top-level API, manages grid launch
2. **Kernel layer**: Thread block-level orchestration
3. **Collective layer**: Multi-thread cooperation patterns (sync, pipelining)
4. **Tiled MMA/Copy**: Spatial micro-kernels for tiling
5. **Atom layer**: Hardware-specific instructions (MMA, ldmatrix, cp.async)

FLUTE customizes the **Collective** and **Tiled Copy** layers to inject LUT
dequantization into the standard GEMM pipeline.

### CuTe Abstractions Used

- **Layouts**: `SmemLayoutA`, `SmemLayoutQ`, `SmemLayoutS`, etc. with 3x3x3
  swizzle patterns for bank-conflict-free shared memory access
- **TiledCopy**: Separate copy operations for A matrix (activations), Q matrix
  (packed quantized weights), Q2 (second bit-slice for 3-bit), and S (scales)
- **TiledMma**: SM80_16x8x16 MMA operations for half/bfloat16
- **Async copy**: `cp.async` for global → shared memory transfers with predication
- **Register fragments**: `FragA`, `FragB`, `FragC`, `FragS` for tensor core inputs

### The GEMM Pipeline

The kernel's main loop (from `qgemm_kernel.hpp`) follows this pattern:

```
1. PREFETCH: Load lookup table from global → shared memory (once)

2. TILE LOOP: For each K-tile:
   a. Async copy: input tile (X) from global → shared
   b. Async copy: quantized weight slices (Q1, Q2, S) from global → shared
   c. Wait for copies to complete

3. FRAGMENT LOOP: For each register-backed fragment within the tile:
   a. Copy fragment data from shared → registers (ldmatrix for A)
   b. Load packed weight data from shared → registers
   c. For 3-bit: Combine bit-slices in registers
      Q_combined = combine(Q1_reg, Q2_reg)
   d. Vectorized dequantization:
      W_dequant = vec_dequantize(Q_combined, scale_reg, LUT_shared)
   e. Tensor core MMA:
      Y_reg = tensor_core_mma(Y_reg, X_reg, W_dequant)

4. EPILOGUE: Convert FP32 accumulators → FP16, write to global memory
   (with Stream-K fixup if needed)
```

### Multi-Stage Pipeline

The kernel uses circular shared memory buffers with configurable pipeline depth
(`Stages` template parameter, typically 2-4). This overlaps global→shared copies
with shared→register copies and computation:

- Stage N: Computing MMA on fragments from shared memory
- Stage N+1: Loading next tile from global to shared memory

The number of stages is a tuning parameter (see Section 9).

---

## 8. Source Code Structure

Repository: https://github.com/HanGuo97/flute

### CUDA/C++ Sources (`flute/csrc/`)

| File | Purpose |
|---|---|
| `qgemm_kernel.hpp` | **Main kernel**: Template device function `qgemm_device` and host launcher `qgemm_host`. Contains the full GEMM pipeline with dequantization. |
| `config.hpp` | **Configuration**: `GemmConfig` template with all tile sizes, thread counts, shared memory layouts, MMA configurations, copy operations. |
| `packbits_utils.hpp` | **Dequantization**: `DequantizationTraits` template with specializations for 2/3/4-bit, vectorized/shuffle/Marlin modes. Core dequant logic. |
| `tile_scheduler_utils.hpp` | **Work distribution**: `TileScheduler` with Split-K and Stream-K modes. `FixupHelper` for inter-block reduction. |
| `conversion_utils.hpp` | **Type conversion**: Register-level tensor type conversion using CUTLASS converters. |
| `marlin_utils.hpp` | **Marlin compatibility**: Marlin-style `lop3`-based INT4 dequantization for uniform quantization mode. |
| `qgemm_kernel_raw_generated.cu` | **Generated instantiations**: Pre-compiled kernel variants for supported shapes/configs. |
| `qgemm_kernel_example.cu` | **Example**: Template instantiation example showing how to configure a kernel. |
| `qgemm.cpp` | **PyTorch binding**: C++ entry point that dispatches to the appropriate kernel template. |
| `hadamard_transform_cuda.cu` | **Hadamard transform**: CUDA kernel for the HadaCore integration. |
| `cutlass_extensions_bf16.h` | **BF16 extensions**: Additional bfloat16 support utilities. |

### Python Sources (`flute/`)

| File | Purpose |
|---|---|
| `ops.py` | PyTorch custom op registration with fake tensor implementations for torch.compile. |
| `tune.py` | Auto-tuning: benchmarks multiple kernel configurations and selects the fastest. |
| `packbits_utils.py` | Weight packing: `to_binary`, `from_binary`, `pack_bools_into_integers`, `pack_integer_tensors`. |
| `nf_utils.py` | NormalFloat codebook generation via inverse Gaussian CDF. Quantization/dequantization. |
| `utils.py` | General utilities. |
| `codegen_utils.py` | Code generation helpers for kernel instantiation. |

### Key Configuration Parameters (`config.hpp`)

The `GemmConfig` template is parameterized by:

```
Data types:
  T    — compute type (half, bfloat16)
  TQ   — quantized weight type (int16)
  TC   — accumulation type (float)
  TR   — reduction type

Threading:
  Warps   — number of warps per block
  Threads — total threads (must be multiple of 128)

Quantization:
  NumBits    — 2, 3, or 4
  GroupSize  — 32, 64, 128, or 256
  NumPacked  — number of packed elements per int16

Tiling:
  TileM, TileK, TileP — tile dimensions for M, K, packed-weight axes
  Stages              — pipeline depth (2-4)
  StagesG             — pipeline stages for scale loading

Copy operations:
  G2SCopySizeA, G2SCopySizeQ, etc. — transfer granularity

MMA configuration:
  MmaTheM, MmaTheN, MmaTheK — thread layout within MMA
  MmaPrmM, MmaPrmN, MmaPrmK — permutation within MMA
```

---

## 9. Kernel Configuration and Tuning

FLUTE is **shape-specialized** — for each combination of (M, N, K, num_bits,
group_size, dtype, GPU), a specific kernel configuration is selected via benchmarking.

### What Gets Tuned

The `template_id` parameter encodes a specific combination of:
- Tile sizes (TileM, TileN, TileK)
- Pipeline stages
- Number of LUT duplicates (for bank conflict mitigation)
- Thread block configuration
- MMA layout

### Tuning Process

From `tune.py`:

1. For a given matrix shape and quantization config, enumerate candidate
   `template_id` values
2. For each candidate, run the kernel at least 100 times
3. Measure average execution time
4. Select the fastest `template_id`
5. Cache the result for future use

The tuned `template_id` is stored in the model's metadata and passed to `qgemm()`
at inference time.

### Correctness Verification

After tuning, the framework runs correctness checks:
- Generates test cases with known-good outputs
- Compares against thresholds: FP16 ≤ 2.0e-3, BF16 ≤ 1.1e-2

### Limitations

- Each new model shape requires re-tuning
- Different tensor parallel configurations create different shapes
- The team is working on JIT tuning to reduce this constraint
- As of January 2025, experimental auto-tune support removes some shape/GPU
  specialization

---

## 10. NormalFloat and NFL (Learned NormalFloat)

### NormalFloat (NF) Codebook

The standard NF codebook (same concept as QLoRA's NF4) generates quantization
levels from the inverse Gaussian CDF:

1. Generate 2^(b-1) evenly-spaced probability values in [δ, 1/2] and [1/2, 1-δ]
   where δ = 1/2 × (1/30 + 1/32)
2. Convert to quantiles via inverse CDF: q_i = Φ^(-1)(p_i)
3. Normalize: q̃_i = q_i / q_{2^b - 1}

The result is a symmetric codebook in [-1, 1] optimized for normally-distributed
weights.

### Group-Level Scaling

For a weight group u with absmax s = max(|u|):
- Quantize: c_j = argmin_i |q̃_i - u_j/s|
- Dequantize: T[Q_{ij}] × s_{(i×j) mod B}

### NFL (Learned NormalFloat)

NFL extends NF by learning the scale parameter σ̃:

1. Reformulate quantization: c_j = argmin_i |sσ̃q_i - u_j|
2. Initialize σ̃ from the standard NF normalization constant: σ̃ = 1/Φ^(-1)(1-δ)
3. Optimize σ̃ via gradient descent on negative log-likelihood
4. Use calibration data: 128 examples × 2048 tokens from WikiText-2
5. Apply straight-through estimator for the argmin gradient
6. Save the learned scale as sσ̃/σ (preserves dequantization format)

This adds minimal overhead (learning one scalar per group) but measurably improves
quantization quality.

### Results

LLaMA-3.1 8B with NFL W4G64:
- WikiText-2 perplexity: 6.24 (vs 6.31 unquantized — actually better due to
  the calibration fitting)

LLaMA-3.1 70B with NFL W4G64:
- WikiText-2 perplexity: 3.09 (vs 2.82 unquantized)

---

## 11. Performance Analysis

### Kernel-Level Benchmarks

**4-bit quantization, group size 128:**
- 2-4× speedup over FP16 `torch.mm` at batch < 32
- Outperforms bitsandbytes and BitBLAS-NF4 LUT kernels
- Competitive with uniform-quantization kernels (Marlin, BitBLAS-INT4)
- At batch sizes > 32, advantage diminishes (GEMM becomes compute-bound)

**3-bit quantization:**
- Supported where most other LUT kernels don't support it at all
- Consistent speedups across group sizes 32, 64, 128, 256

### End-to-End LLM Throughput

**LLaMA-3 8B** (batch=1, single GPU):
- 4-bit, group=128: ~2.2× tokens/s improvement, perplexity 6.2
- 3-bit, group=128: ~2.4× tokens/s improvement, perplexity 4.6

**LLaMA-3 70B** (tensor parallelism):
- 4-bit, group=256: ~1.9-2.0× improvement (4×A6000, 2×A100)
- 3-bit, group=256: ~1.7-2.0× improvement (4×A6000, 2×A100)

**LLaMA-3.1 405B**: Enables single-node inference (impossible without
quantization)

### Hardware-Specific Performance

Optimized for **Ampere GPUs** (A100, A6000, RTX 4090). Not yet optimized for
Hopper (H100), though it runs. bfloat16 is slower than float16, likely due to
lack of Ampere hardware-accelerated bfloat16 atomic-add.

---

## 12. Comparison with Other Kernels

### FLUTE vs. Marlin

| Aspect | FLUTE | Marlin |
|---|---|---|
| **Quantization type** | LUT-based (arbitrary codebooks) | Uniform (INT4/INT8 linear) |
| **Bit widths** | 2, 3, 4 | 4, 8 |
| **Dequant method** | Shared memory LUT lookup | `lop3` bit manipulation in registers |
| **Work distribution** | Stream-K (CUTLASS) | Custom stripe partitioning |
| **Implementation** | CUTLASS 3 / CuTe templates | Hand-written CUDA |
| **Weight format** | Offline-restructured, bit-sliced | Custom tiled INT4 packing |
| **Bank conflict handling** | LUT duplication + vectorization | N/A (arithmetic dequant) |
| **Target GPU** | Ampere (SM80) | Ampere + Hopper |
| **Performance (4-bit)** | Competitive at batch < 32 | Slightly faster at small batch |
| **3-bit support** | Yes | No |
| **Codebook flexibility** | Arbitrary | Linear only |

Key insight: Marlin uses register-level arithmetic for dequantization (no memory
access), while FLUTE uses shared memory lookup. For uniform quantization, Marlin's
approach is faster. For non-uniform/codebook quantization, FLUTE's approach is
necessary.

FLUTE also includes a `Marlin` mode in its `QuantMapModeEnum` that delegates to
Marlin-style `lop3` dequantization for the uniform INT4 case.

### FLUTE vs. bitsandbytes (Current)

| Aspect | FLUTE | bitsandbytes |
|---|---|---|
| **Approach** | Fused dequant+GEMM | Separate dequant, then cuBLAS |
| **Tensor cores** | Yes (via CUTLASS MMA) | No (dequant only, cuBLAS for GEMM) |
| **LUT mechanism** | Vectorized shared memory | `__shfl_sync` in registers |
| **Bit widths** | 2, 3, 4 | 2, 3, 4, 5 (kbit branch) |
| **Performance** | 2-4× over dequant+cuBLAS | Baseline (dequant+cuBLAS) |

### FLUTE vs. Proposed kbit GEMM (from kbit_gemm_context.md)

| Aspect | FLUTE | Proposed kbit GEMM |
|---|---|---|
| **Framework** | CUTLASS 3 / CuTe | Hand-written CUDA |
| **LUT storage** | Shared memory (vectorized+duplicated) | Registers (`__shfl_sync`) |
| **Work distribution** | Stream-K (CUTLASS built-in) | Persistent kernel with split-K |
| **Bit widths** | 2, 3, 4 | 2, 3, 4, 5 |
| **Weight format** | Bit-slice decomposed, offline restructured | Bit-plane (from `__ballot_sync`), tiled |
| **Scale format** | FP16 group scales | E4M4 absmax (1 byte per block of 32) |
| **Block size** | Configurable (32, 64, 128, 256) | Fixed at 32 |
| **Target GPU** | Ampere | Ampere + Hopper |

---

## 13. Detailed Comparison: FLUTE vs. Bitsandbytes kbit

This section provides a side-by-side analysis of every major design decision,
referencing the actual bitsandbytes kbit implementation on the
`feature/kbit-quantization` branch (`csrc/ops.cu` lines 649-869) and the planned
GEMM kernel design from `agents/kbit_gemm_context.md`.

### 13.1 Codebook Lookup Mechanism

This is the single biggest architectural difference between the two kernels.

**FLUTE: Vectorized shared memory LUT with duplication**

FLUTE stores the lookup table in shared memory. To reduce the number of shared
memory transactions, it creates a "vectorized" table containing every possible
*pair* of consecutive indices. For 4-bit quantization:

- Original table: 16 entries × 2 bytes (half) = 32 bytes
- Vectorized table: 256 entries × 4 bytes (half2) = 1024 bytes

The kernel extracts pairs of 4-bit indices from packed weight data, forms an
8-bit combined index, and fetches a `half2` from shared memory in one transaction.
This halves the number of shared memory reads.

To handle bank conflicts (up to 8-way for 4-bit), FLUTE duplicates the entire
vectorized table multiple times in shared memory at different base addresses,
shifting bank alignment. The duplication count is auto-tuned per shape/GPU.
Worst case: 8 copies × 1 KB = 8 KB of shared memory for the table alone.

Modes in `packbits_utils.hpp`:
```cpp
enum QuantMapModeEnum {
    Basic,           // Per-element LUT lookup
    Vectorized,      // Vectorized half2 lookup (default)
    WarpShuffle,     // __shfl_sync-based (register)
    Marlin           // lop3 arithmetic dequant
};
```

**kbit: Register shuffle via `__shfl_sync`**

The bitsandbytes kbit kernel stores the codebook in a single register per lane:

```cpp
// ops.cu line ~766 (standalone dequant), GEMM plan uses same pattern:
float cb = (lane_id < (1 << K)) ? codebook[lane_id] : 0.0f;
// ...
float val = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;
```

For the GEMM kernel, the codebook is pre-converted to half at kernel start:
```cpp
half cb_h = (lane < (1 << K_BITS))
    ? __float2half(codebook[lane]) : __float2half(0.0f);
// In inner loop:
half val = __shfl_sync(0xFFFFFFFF, cb_h, idx);
```

Each lane holds one codebook entry in a register. Lookup is a warp shuffle with
arbitrary per-thread source lane selection. Cost: 1 cycle on the shuffle unit,
zero memory bandwidth consumed.

**Why kbit's approach is better for our use case:**

- Our codebooks have at most 2^5 = 32 entries (K=2..5), fitting exactly in a
  32-lane warp. No shared memory needed at all.
- Shuffle is 1 cycle with zero bank conflicts by definition.
- No shared memory space consumed by the table — more room for A and B tiles.
- No duplication/tuning complexity.
- The shuffle approach is already proven in the existing standalone dequant
  kernel (`ops.cu` line 783).

FLUTE needs shared memory because it's designed to be generic — it supports
arbitrary table sizes that could exceed 32 entries. For exactly this reason,
FLUTE also offers a `WarpShuffle` mode, but it isn't the default.

### 13.2 Weight Packing Format

**FLUTE: Contiguous K-bit packing with bit-slice decomposition**

FLUTE packs quantized indices contiguously. For 4-bit: two 4-bit indices per
`uint8`, or 8 per `uint32`. The packed `int16` values are loaded via 128-bit
async copies.

For 3-bit (which doesn't divide evenly into 128-bit words), FLUTE uses
**bit-slice decomposition**: split each 3-bit index into a 1-bit MSB and a
2-bit LSB, store them in separate arrays, load each with clean 128-bit copies,
and combine in registers:

```
combined_index = (bit_slice_1 << 2) | bit_slice_2
```

The offline restructuring permutes packed weights so that after loading and
dequantization, values land in the exact register positions that `m16n8k16`
tensor cores expect. This means the kernel never does runtime reordering.

**kbit: Bit-plane format via `__ballot_sync`**

The bitsandbytes quantize kernel (`ops.cu` line 706) produces K separate
`uint32` bit-plane words per block of 32 elements:

```cpp
// pack_kbit_warp<K>:
for (int bit = 0; bit < K; bit++)
    packed_words[bit] = __ballot_sync(0xFFFFFFFF, (qval >> bit) & 1);
```

Bit-plane 0 contains bit 0 of all 32 elements, bit-plane 1 contains bit 1, etc.
The GEMM repack kernel retiles this from flat sequential into
`[k_tile][n_tile][col][k_block][bit_plane]` order for coalesced tile loads.

To extract an index in the GEMM kernel:
```cpp
for (int b = 0; b < K_BITS; b++)
    idx |= ((planes[b] >> row) & 1) << b;
```

**Comparison:**

| Aspect | FLUTE | kbit |
|---|---|---|
| Storage unit | Contiguous K-bit fields in int16 | K separate uint32 bit-plane words |
| 3-bit handling | Bit-slice split (1+2), two separate loads | Natural: K=3 bit-planes, same as K=2,4,5 |
| 5-bit handling | Not supported | Natural: K=5 bit-planes |
| Extraction cost | Shift+mask to isolate K-bit field from packed word | K shift+mask+OR to assemble index from planes |
| Memory footprint | K bits per element | K bits per element (identical) |
| Runtime reordering | None (offline permutation matches tensor core layout) | None (repack kernel produces tile-aligned layout) |

The bit-plane format's key advantage is uniformity: K=2,3,4,5 all work
identically with no special cases. FLUTE needs separate code paths for 3-bit
(the bit-slice decomposition). The bit-plane extraction cost (K INT32 ops per
element) runs on integer ALU concurrent with tensor core MMA, so it's
effectively hidden.

### 13.3 Scale/Absmax Format and Application

**FLUTE: FP16 group scales**

FLUTE uses standard half-precision scales with configurable group sizes
(32, 64, 128, 256). Dequantization is: `value = table[index] * scale`.

The scales are loaded from global → shared memory alongside the packed weights,
with their own pipeline stage (`StagesG`). Inside the fragment loop, scale values
are applied via `__hmul2()` paired half multiplication.

Storage overhead per element: 2 bytes / group_size. For group_size=128: 0.0156
bytes/element. For group_size=32: 0.0625 bytes/element.

**kbit: E4M4 absmax (1 byte per block of 32)**

The kbit system uses a custom 8-bit floating point format for the per-block
absmax value (`ops.cu` line 722):

```cpp
// E4M4: 4-bit exponent (bias=11) + 4-bit mantissa
// Normal: 2^(e-11) * (1 + m/16), range ~[6.1e-5, 31.0]
// Decode: construct IEEE 754 float via bit manipulation
unsigned int ieee = (unsigned int)(e - E4M4_BIAS + 127) << 23
                  | (unsigned int)m << 19;
return __uint_as_float(ieee);
```

Dequantization is: `value = codebook[index] * absmax`. The absmax is always
per-block (blocksize=32), giving fine-grained scaling.

Storage overhead: 1 byte / 32 = 0.03125 bytes/element. This is:
- 2× less than FLUTE with group_size=32 (0.0625 bytes/element)
- Same as FLUTE with group_size=64 in absolute bytes, but kbit gets
  per-32-element granularity vs FLUTE's per-64-element granularity
- Max relative error from E4M4: 6.25% (1/16 from 4-bit mantissa)

In the GEMM kernel, absmax decode happens once per block-of-32 per column per
K-tile (256 decodes total for TILE_N=128, TILE_K=64). The decode is ~5 integer
ALU ops, negligible compared to MMA throughput.

**Why E4M4 matters:**

At K=2 (2-bit quantization), each element is 2 bits = 0.25 bytes. FLUTE's FP16
scale at group_size=128 adds 0.0156 bytes/element (6.25% overhead). kbit's E4M4
at blocksize=32 adds 0.03125 bytes/element (12.5% overhead) but with 4× finer
granularity — and in 1 byte instead of 2. The finer granularity typically
improves quantization quality more than the coarser group hurts it.

### 13.4 Work Distribution and Split-K

**FLUTE: Stream-K via CUTLASS**

FLUTE uses CUTLASS's built-in Stream-K decomposition (`tile_scheduler_utils.hpp`).
All (M,N,K) tiles are linearized into a 1D work sequence and distributed evenly
across `num_blocks` threadblocks:

```cpp
tiles_per_block = total_tiles / num_blocks;
blocks_special = total_tiles % num_blocks;  // get +1 tile
```

When multiple blocks contribute to the same output tile (different K-ranges),
the `FixupHelper` coordinates via `cutlass::Barrier` primitives. Partial sums
are stored in FP32 in a global workspace; the finishing block reduces and
converts to FP16.

Grid launch: `dim3(num_blocks)` for Stream-K mode.

**kbit: Persistent kernel with linearized work assignment**

The kbit GEMM plan launches exactly `num_SMs` blocks. Work items are linearized
as (m_tile, n_tile, k_chunk) triples, ordered so that all k_chunks for a given
(m,n) output tile are contiguous:

```cpp
int work_per_block = div_ceil(total_work, gridDim.x);
int my_start = blockIdx.x * work_per_block;
int my_end = min(my_start + work_per_block, total_work);
```

Key optimization: when consecutive work items share the same output tile, the
block keeps accumulators in registers across k_chunks — no intermediate write.
The pipeline restarts between chunks (~2-tile cost), but accumulators persist.

Output write uses a three-way branch:
- Full K-range ownership → write FP16 directly (common case for large M)
- First contributor → write FP32 to workspace (overwrite, acts as zero+write)
- Subsequent contributors → atomicAdd FP32 to workspace

A per-tile atomic counter tracks when the last contributor finishes, which
then converts FP32 → FP16 in the final output.

**Comparison:**

| Aspect | FLUTE (Stream-K) | kbit (Persistent) |
|---|---|---|
| Implementation | CUTLASS built-in | Hand-written |
| Launch config | `dim3(num_blocks)` | `dim3(num_SMs)` |
| Granularity | Per K-tile | Per k_chunk (multiple K-tiles) |
| Sync mechanism | `cutlass::Barrier` semaphores | `atomicAdd` + atomic counter |
| Accumulator reuse | Each block handles isolated work items | Consecutive same-(m,n) items share accumulators |
| Reduction | Finishing block reduces all partials | Last contributor (via counter) converts to FP16 |
| Dependency | Requires CUTLASS | Self-contained |

The persistent kernel's accumulator-reuse optimization is significant: for
problems where each block handles multiple k_chunks for the same output tile,
it avoids writing and re-reading intermediate FP32 partials. Stream-K doesn't
have this optimization — each block writes its partial to global memory.

### 13.5 Bit-Width Support

| Bits | FLUTE | kbit |
|---|---|---|
| 2-bit | Yes (build from source) | Yes |
| 3-bit | Yes (bit-slice decomposition) | Yes (bit-plane, no special case) |
| 4-bit | Yes (primary target) | Yes |
| 5-bit | No | Yes |

FLUTE's lack of 5-bit support is likely because the bit-slice approach would
need a 2+3 or 1+4 split, adding another code path. The kbit bit-plane format
handles K=5 identically to K=2,3,4.

### 13.6 Implementation Framework

**FLUTE: CUTLASS 3 / CuTe templates**

- All tiling, pipelining, and MMA via CUTLASS abstractions
- Shared memory layouts use CuTe's swizzle patterns (3×3×3)
- Async copies via `cp.async` managed by CUTLASS pipeline stages
- `TiledCopy` and `TiledMma` handle thread-to-data mapping
- `GemmConfig` template encodes the full kernel configuration
- Code generation produces template instantiations per (shape, bits, GPU)

Pros: Less custom infrastructure to write, well-tested pipeline/sync code.
Cons: Massive template expansion, slow compile, CUTLASS version dependency
(pinned to v3.4.1), shape-specialized binaries.

**kbit: Hand-written CUDA**

- Custom tiling with explicit loop structures
- Manual `cp.async` pipeline (2-stage double buffer)
- Inline PTX for `ldmatrix` and `mma.sync` instructions
- No external dependencies beyond CUDA toolkit
- Single compilation unit (`kernels.cu`) with template params `<K_BITS, M_BLOCKS>`
- Kernel config selected at launch time based on M dimension

Pros: Full control over register allocation and scheduling, no dependency
management, single binary works for all shapes of the same (K, M_BLOCKS).
Cons: Must implement all infrastructure manually, more potential for bugs in
pipeline/sync code.

### 13.7 Tensor Core Usage

Both kernels use the same fundamental MMA instruction: `m16n8k16` with FP16
inputs and FP32 accumulation.

**FLUTE**: CuTe's `SM80_16x8x16_F32F16F16F32` atom, configured via `TiledMma`
with customizable thread layout (`MmaTheM × MmaTheN × MmaTheK`) and
permutation (`MmaPrmM × MmaPrmN × MmaPrmK`).

**kbit**: Direct inline PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
instruction. Thread-to-fragment mapping hand-computed:
- 4 threads per column (lane/4 = column index)
- Row indices: {2i, 2i+1, 2i+8, 2i+9} where i = lane%4
- FragA: M_BLOCKS × half2[2] per k-sub-tile
- FragB: half2[2] per N-block (dequantized on the fly, not stored)

The kbit design explicitly exploits the 4-threads-per-column property for
shared memory access: when loading bit-plane words, 4 threads read the same
K addresses, getting a free 4-way broadcast with zero bank conflicts. FLUTE
doesn't need this optimization because its offline restructuring already
places data in the correct register positions.

### 13.8 Pipeline Design

**FLUTE**: Configurable multi-stage pipeline (2-4 stages, auto-tuned).
Separate pipeline stages for different data streams:
- `Stages`: Main pipeline depth for A and Q tiles
- `StagesG`: Separate depth for scale factor loading
- `StagesGView`: View stages for handling GroupSize/TileK relationships

Circular shared memory buffers managed by CUTLASS pipeline abstractions.

**kbit**: 2-stage double-buffered pipeline (fixed).
- Stage 0 and Stage 1 alternate in shared memory
- `cp_async_fence()` and `cp_async_wait<1>()` for synchronization
- Pipeline restarts when switching k_chunks (2-tile cost)

The kbit approach is simpler but less flexible. FLUTE's ability to tune the
pipeline depth per shape can yield better performance in specific cases.

### 13.9 Offline Weight Preparation

Both require offline weight restructuring, but the details differ.

**FLUTE offline restructuring:**

1. Quantize weights to K-bit indices using a codebook (NF or custom)
2. Pack indices contiguously (for 3-bit: split into 1+2 bit-slices)
3. **Permute** packed words so that after loading and dequantization, values
   land directly in tensor core register positions
4. The permutation encodes: thread-to-element MMA mapping + ldmatrix layout +
   bit-slice separation

This is a single combined permutation that folds multiple concerns together.

**kbit offline restructuring:**

1. Quantize weights via `kQuantizeBlockwise_kbit` → flat bit-plane format
   (K uint32 words per block of 32 elements, sequential)
2. Encode absmax from float32 to E4M4 uint8
3. **Retile** bit-planes from flat → `[k_tile][n_tile][col][k_block][bit_plane]`
4. **Retile** absmax from flat → `[k_tile][n_tile][col][k_block]`

The kbit repack is a simpler gather/permutation — it only changes the tile
layout, not the data format within tiles. No MMA-layout-aware permutation is
needed because the GEMM kernel handles the thread-to-element mapping at runtime
via the bit-plane extraction + `__shfl_sync` codebook lookup.

### 13.10 Summary: When to Prefer Which Approach

**FLUTE is better when:**
- You need arbitrary codebook sizes (> 32 entries)
- You want to leverage CUTLASS's tested infrastructure
- You need auto-tuning across many different matrix shapes
- You need Stream-K's sophisticated edge-case handling
- 3-bit and 4-bit are the primary targets

**kbit is better when:**
- Codebooks are ≤ 32 entries (K ≤ 5) — register shuffle is strictly faster
- You need 5-bit support
- You want zero external dependencies
- Fine-grained E4M4 absmax (per-32-element) is important
- You need a single binary that works across all shapes (no re-tuning)
- You want Hopper GPU support from the start
- The bit-plane format naturally handles all K values uniformly

---

## 14. Limitations and Known Issues

1. **Shape specialization**: Each matrix shape requires separate tuning and
   compilation. Different tensor parallel configurations create different shapes,
   limiting supported models. (Partial mitigation via auto-tune as of Jan 2025.)

2. **Ampere-only optimization**: Not yet leveraging Hopper features (TMA, warp
   specialization, distributed shared memory). Runs on H100 but not at peak.

3. **bfloat16 performance**: Slower than float16 on Ampere due to lack of
   hardware-accelerated bfloat16 atomic-add (needed for Stream-K reduction).

4. **Large batch degradation**: Performance advantage diminishes at batch > 32
   as the GEMM becomes compute-bound rather than memory-bandwidth-bound.

5. **Numerical issues**: Some instability reported with 4-bit, group-size=256
   on A100.

6. **No 5-bit support**: FLUTE supports 2, 3, 4-bit only. The kbit design
   supports 5-bit as well.

---

## 15. Links and References

### Primary Sources

- **Paper (ArXiv)**: https://arxiv.org/abs/2407.10960
- **Paper (PDF)**: https://arxiv.org/pdf/2407.10960
- **Paper (HTML)**: https://arxiv.org/html/2407.10960
- **Paper (ACL Anthology)**: https://aclanthology.org/2024.findings-emnlp.724/
- **GitHub Repository**: https://github.com/HanGuo97/flute
- **HuggingFace Paper Page**: https://huggingface.co/papers/2407.10960

### Source Code (Key Files)

- **Main kernel**: https://github.com/HanGuo97/flute/blob/main/flute/csrc/qgemm_kernel.hpp
- **Configuration**: https://github.com/HanGuo97/flute/blob/main/flute/csrc/config.hpp
- **Dequantization**: https://github.com/HanGuo97/flute/blob/main/flute/csrc/packbits_utils.hpp
- **Tile scheduling**: https://github.com/HanGuo97/flute/blob/main/flute/csrc/tile_scheduler_utils.hpp
- **Weight packing**: https://github.com/HanGuo97/flute/blob/main/flute/packbits_utils.py
- **NF utilities**: https://github.com/HanGuo97/flute/blob/main/flute/nf_utils.py
- **Auto-tuning**: https://github.com/HanGuo97/flute/blob/main/flute/tune.py
- **Ops/dispatch**: https://github.com/HanGuo97/flute/blob/main/flute/ops.py

### Pre-Quantized Models

- **HuggingFace Hub**: Models under the `HanGuo97` organization
  - LLaMA-3.1: 8B, 70B, 405B (base + instruct, NFL W4G64 default)
  - LLaMA-3: 8B, 70B
  - Gemma-2: 9B, 27B

### Related Projects

- **CUTLASS 3.x**: https://github.com/NVIDIA/cutlass (required dependency, v3.4.1)
- **HIGGS**: Vector dequantization extension, NAACL 2025
- **HadaCore**: Hadamard transform integration
- **Marlin**: https://github.com/IST-DASLab/marlin (comparison kernel for uniform INT4)
- **LUT-GEMM**: Earlier work on lookup-table-based GEMM kernels
- **LUT Tensor Core (arxiv 2408.06003)**: Hardware/software co-design for LUT operations

### Blog Posts and Analysis

- **MarkTechPost**: https://www.marktechpost.com/2024/07/26/flute-a-cuda-kernel-designed-for-fused-quantized-matrix-multiplications-to-accelerate-llm-inference/
- **Semantic Scholar**: https://www.semanticscholar.org/paper/Fast-Matrix-Multiplications-for-Lookup-LLMs-Guo-Brandon/be66705b36912679ea373184aaf057aa365d292a
- **AlphaXiv Discussion**: https://www.alphaxiv.org/abs/2407.10960

### Installation

```bash
# Default (CUDA 12.1)
pip install flute-kernel

# CUDA 11.8
pip install flute-kernel -i https://flute-ai.github.io/whl/cu118

# CUDA 12.4
pip install flute-kernel -i https://flute-ai.github.io/whl/cu124

# From source (required for 2-bit)
git clone https://github.com/HanGuo97/flute.git
cd flute
pip install -e .
```

### Citation

```bibtex
@inproceedings{guo2024flute,
  title={Fast Matrix Multiplications for Lookup Table-Quantized LLMs},
  author={Guo, Han and Brandon, William and Cholakov, Radostin and
          Ragan-Kelley, Jonathan and Xing, Eric P. and Kim, Yoon},
  booktitle={Findings of EMNLP},
  year={2024}
}
```
