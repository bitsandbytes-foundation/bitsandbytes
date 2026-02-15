# kbit Scalar GEMV Optimization Guide

## Overview

This guide describes how to build a high-performance scalar GEMV (matrix-vector
multiply) kernel for kbit-quantized weights. The kernel multiplies a small
activation matrix A [M, K] by a quantized weight matrix B [N, K] to produce
C [M, N], where M is 1-4 (batch size during autoregressive decoding).

The target model is **Qwen3-Coder-Next** (the only model we optimize for), which
has both dense and mixture-of-experts (MoE) layers. The kernel must support all
kbit widths from 2 to 5 bits.

The approach: start with a kernel that achieves 100% memory throughput using only
vector loads, then incrementally add quantization logic while maintaining
performance.

---

## Table of Contents

1.  [Target Model: Qwen3-Coder-Next](#1-target-model-qwen3-coder-next)
2.  [GEMM Shapes](#2-gemm-shapes)
3.  [Reference Implementation: bnb gemv_4bit](#3-reference-implementation-bnb-gemv_4bit)
4.  [kbit Quantization Format](#4-kbit-quantization-format)
5.  [Data Layout: Repack Tiling](#5-data-layout-repack-tiling)
6.  [RTX 4090 Hardware Parameters](#6-rtx-4090-hardware-parameters)
7.  [Theoretical Performance Targets](#7-theoretical-performance-targets)
8.  [Build System: Only Compile What You Need](#8-build-system-only-compile-what-you-need)
9.  [ncu Benchmarking: The Only Benchmark That Matters](#9-ncu-benchmarking-the-only-benchmark-that-matters)
10. [Step-by-Step Kernel Development](#10-step-by-step-kernel-development)
11. [Testing: Correctness at the End](#11-testing-correctness-at-the-end)
12. [Current Kernel State](#12-current-kernel-state)
13. [Known Issues and Pitfalls](#13-known-issues-and-pitfalls)

---

## 1. Target Model: Qwen3-Coder-Next

Config from `https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json`:

```
hidden_size:                   2048
intermediate_size:             5120
num_attention_heads:           16
num_key_value_heads:           2
head_dim:                      256
num_hidden_layers:             48

num_experts:                   512
num_experts_per_tok:           10
moe_intermediate_size:         512
shared_expert_intermediate_size: 512

linear_num_key_heads:          16
linear_num_value_heads:        32
linear_key_head_dim:           128
linear_value_head_dim:         128
```

This is a hybrid dense + MoE architecture. Every layer has attention (dense) plus
an MLP that is either dense or MoE (decoder_sparse_step=1 means every layer is
MoE). There are also "linear attention" projections with separate key/value head
configurations.


## 2. GEMM Shapes

Every linear layer in the model produces a GEMM of the form:

    C[M, N] = A[M, K] * W^T[K, N]

where W is stored quantized as [N, K]. During autoregressive decoding, M = 1-4
(batch size / number of concurrent sequences). The weight matrix dominates memory
traffic since it is much larger than A or C.

### All unique shapes from Qwen3-Coder-Next

| Layer                  | K_dim | N     | Data (K=4, bytes) | Notes                    |
|------------------------|------:|------:|------------------:|--------------------------|
| Q projection           | 2048  | 4096  | 4.25 MB           | 16 heads * 256 head_dim  |
| K projection           | 2048  | 512   | 0.53 MB           | 2 KV heads * 256         |
| V projection           | 2048  | 512   | 0.53 MB           | 2 KV heads * 256         |
| O projection           | 4096  | 2048  | 4.25 MB           | 16*256 -> 2048           |
| Linear key proj        | 2048  | 2048  | 2.13 MB           | 16 heads * 128           |
| Linear value proj      | 2048  | 4096  | 4.25 MB           | 32 heads * 128           |
| Dense gate_proj        | 2048  | 5120  | 5.31 MB           | SiLU gate                |
| Dense up_proj          | 2048  | 5120  | 5.31 MB           | (gate and up are separate)|
| Dense down_proj        | 5120  | 2048  | 5.31 MB           |                          |
| MoE gate_proj (per expert) | 2048 | 512 | 0.53 MB          | 512 experts, top-10      |
| MoE up_proj (per expert)   | 2048 | 512 | 0.53 MB          |                          |
| MoE down_proj (per expert) | 512  | 2048 | 0.53 MB          |                          |
| Shared expert gate/up  | 2048  | 512   | 0.53 MB           |                          |
| Shared expert down     | 512   | 2048  | 0.53 MB           |                          |

### Data size calculation

For a weight matrix W[N, K_dim] quantized at k bits with blocksize 32:

```
B_packed:  N * K_dim / 32 * k * 4 bytes   (k uint32 bit-plane words per 32-element block)
B_absmax:  N * K_dim / 32 bytes            (1 byte E4M4 absmax per block)
A:         M * K_dim * 2 bytes             (fp16/bf16, negligible for M<=4)
Total:     N * K_dim * (k/8 + 1/32) bytes  (dominated by B_packed)
```

For k=4: `N * K_dim * (4/8 + 1/32) = N * K_dim * 0.53125 bytes`.

### Shape categories

1. **Large** (>= 4 MB): Q proj, O proj, linear value, dense gate/up/down.
   These have enough parallelism to saturate memory bandwidth.

2. **Medium** (~2 MB): Linear key (2048x2048).
   Borderline — needs careful occupancy management.

3. **Small** (~0.5 MB): K/V proj, all MoE expert layers, shared expert.
   Fundamentally limited by kernel launch overhead (~2-3 us). Even at perfect
   bandwidth (1 TB/s), 0.5 MB takes only 0.5 us. The MoE expert shapes should
   use the **grouped GEMV kernel** which batches multiple experts into one launch.


## 3. Reference Implementation: bnb gemv_4bit

The existing bitsandbytes 4-bit GEMV kernel (`kgemm_4bit_inference_naive` in
`bitsandbytes/csrc/kernels.cu`) achieves ~4x speedup over dequantize+cuBLAS.
It is the direct inspiration for our kbit kernel.

### Architecture

```
Grid:   (N + 3) / 4 blocks   (each block handles 4 output rows)
Block:  128 threads = 4 warps
        Each warp handles ONE output row (column of W^T)
        32 lanes split the K dimension
```

### Key design principles

1. **One warp per output element.** Each warp computes one dot product
   C[0, n] = sum_k(A[0, k] * W[n, k]). The 32 lanes split K into chunks
   and reduce via `CUB::WarpReduce`.

2. **Vector loads everywhere.** The critical loads use `int4` (16 bytes):
   - B (weights): `reinterpret_cast<int4*>(B)[offset]` — loads 16 bytes of
     packed 4-bit weights (32 nibbles) in one instruction.
   - A (activations): `reinterpret_cast<int4*>(A)[offset]` — loads 8 fp16
     values (16 bytes) in one instruction.

3. **Codebook in shared memory.** The 16-entry NF4 codebook is loaded into
   `__shared__ T quant_map[16]` once, then accessed via nibble index:
   `quant_map[local_B_4bit[j] >> 4]` and `quant_map[local_B_4bit[j] & 0xF]`.

4. **Register-file computation.** All computation happens in registers:
   `local_B_4bit[16]` (packed bytes), `local_B[8]` (dequantized values),
   `local_A[8]` (activation values), `local_C` (float32 accumulator).

5. **No shared memory for tiles.** Unlike our kbit kernel, the bnb kernel
   does NOT tile into shared memory. Each thread loads directly from global
   memory into registers. This works because:
   - The data access pattern is already coalesced (32 lanes read consecutive K
     elements)
   - Each thread processes `num_values_4bit = 32` elements per K-iteration
   - The codebook is tiny (16 entries)

### Per-iteration data flow

```
Each lane processes 32 elements per K-iteration, in 4 sub-iterations of 8:

  for each K chunk (32 lanes * 32 elements = 1024 K elements per iter):
    1. Vector-load 16 bytes of packed B → local_B_4bit[16]  (one int4)
    2. Load absmax for this block                             (one float)
    for i in 0..3:  (4 sub-iterations)
      3. Dequantize 8 nibbles → local_B[8]   (codebook lookup * absmax)
      4. Vector-load 8 fp16 A values → local_A[8]  (one int4)
      5. Dot product: local_C += sum(local_A[k] * local_B[k])

  WarpReduce(local_C)  → output
```

### Why this matters for our kernel

Our kbit kernel should follow the same philosophy:
- **Vector loads** for all large data (B_packed via int4 or cp.async)
- **Register-file computation** for dequantization
- **Warp-level parallelism** with one warp per output column
- **Minimal shared memory** — only what's necessary

The main difference: our bit-plane format requires different dequantization
(bit extraction from K uint32 planes + shuffle-based codebook lookup instead
of nibble extraction + shared memory codebook lookup).


## 4. kbit Quantization Format

### Bit-plane packing

Unlike NF4 which packs two 4-bit values per byte (nibble packing), the kbit
format uses **bit-plane packing**. For k-bit quantization of a 32-element block:

```
Block of 32 values, each quantized to k bits (indices i0, i1, ..., i31):

Bit-plane 0:  uint32 where bit j = bit 0 of index[j]
Bit-plane 1:  uint32 where bit j = bit 1 of index[j]
...
Bit-plane k-1: uint32 where bit j = bit (k-1) of index[j]
```

So each 32-element block produces **k uint32 words** (k * 4 bytes). This is the
"flat" packed format output by `quantize_kbit`.

### Extracting an index

To recover the k-bit index for element j in a block:

```c
int idx = 0;
for (int b = 0; b < k; b++)
    idx |= ((planes[b] >> j) & 1) << b;
```

This produces k shift+mask+or operations. For k=4, that's 12 ALU ops per element.

### Codebook lookup via warp shuffle

The codebook has `2^k` entries (4 for k=2, 32 for k=5). Since `2^k <= 32`
(the warp size), we store the codebook in **registers** and use `__shfl_sync`
to broadcast:

```c
// Each lane loads its codebook entry once at kernel start
float cb = (lane_id < (1 << k)) ? codebook[lane_id] : 0.0f;

// In the inner loop, look up index via shuffle
float weight = __shfl_sync(0xFFFFFFFF, cb, idx);
```

This is faster than shared memory lookup because shuffle is a single-cycle
register-to-register operation with no bank conflicts.

### Absmax: E4M4 encoding

Each 32-element block has an absmax scale factor. We encode it as a single byte
using E4M4 format (4-bit exponent, 4-bit mantissa, custom bias of 11):

```
Normal:    value = 2^(e - 11) * (1 + m/16)    for e > 0
Subnormal: value = 2^(-10) * (m/16)           for e = 0
```

Decoding uses the branchless version `decode_e4m4_absmax_branchless()` in the
inner loop to avoid warp divergence.

### Full dequantization formula

```
dequantized_weight = codebook[idx] * absmax
```

Where `idx` is the k-bit index extracted from the bit-planes, `codebook` is
the quantization codebook (typically normal-distribution quantiles), and `absmax`
is the E4M4-decoded per-block scale factor.


## 5. Data Layout: Repack Tiling

The flat bit-plane format has poor memory access patterns for the GEMV kernel.
The **repack** step reorganizes data into tiles that enable coalesced vector loads.

### Tile dimensions (compile-time constants)

```c
KBIT_TILE_K = 64      // 64 elements in K dimension per tile = 2 quantization blocks
KBIT_TILE_N = 128     // 128 columns (output channels) per tile
KBIT_BLOCKSIZE = 32   // quantization block size (always 32)
```

### Tile memory layout

Within each tile, data is stored as `[col][kb][bit]`:

```
For a tile with 128 columns and 2 k-blocks:
  col_0, kb_0, bit_0    ← uint32 word
  col_0, kb_0, bit_1
  ...
  col_0, kb_0, bit_{k-1}
  col_0, kb_1, bit_0
  col_0, kb_1, bit_1
  ...
  col_0, kb_1, bit_{k-1}
  col_1, kb_0, bit_0    ← next column starts here
  ...
  col_127, kb_1, bit_{k-1}
```

Each column occupies `k_blocks_per_tile * k` contiguous uint32 words.
For k=4: `2 * 4 = 8` words = 32 bytes per column per tile.

### Tile indexing

Tiles are indexed as `(k_tile, n_tile)` and stored in memory as:

```
tile_index = k_tile * n_tiles + n_tile
B_packed[tile_index * words_per_tile + col * k_blocks_per_tile * k + kb * k + bit]
```

Where:
- `words_per_tile = TILE_N * k_blocks_per_tile * k`
- `n_tiles = N / TILE_N`
- `k_tiles = K_dim / TILE_K`

### Absmax tiling

Same tile structure but 1 byte per (col, kb) pair:

```
absmax_per_tile = TILE_N * k_blocks_per_tile
absmax[tile_index * absmax_per_tile + col * k_blocks_per_tile + kb]
```

### Sub-tile access for TILE_N < 128

Because columns are stored contiguously within a tile, a sub-tile of 64 columns
(the first or second half) is a contiguous block of memory. This means cp.async
int4 vector loads work for sub-tiles:

```
First 64 columns:  offset = 0
Second 64 columns: offset = 64 * k_blocks_per_tile * k  (in uint32 words)
```

The repack kernel is in `csrc/ops.cu` at the `kRepackKbit` function (~line 877).
The repack is a one-time cost during weight loading — not on the inference
critical path.


## 6. RTX 4090 Hardware Parameters

```
GPU:                    NVIDIA GeForce RTX 4090
Architecture:           Ada Lovelace (sm_89)
SMs:                    128
Max threads/SM:         1536 (48 warps)
Max threads/block:      1024
Warp size:              32
Registers/SM:           65536
Max registers/thread:   255
Shared memory/SM:       100 KB (configurable up to 100 KB)
L2 cache:               72 MB
Memory bandwidth:       1008 GB/s (theoretical peak)
Memory bus:             384-bit GDDR6X
Clock (boost):          ~2520 MHz
```

### Occupancy calculation

For a kernel with R registers/thread and B threads/block:

```
Registers/block = R * B
Max blocks from registers = 65536 / (R * B)
Max blocks from warps = 48 / (B / 32)
Max blocks from shmem = 100KB / shmem_per_block
Actual max blocks/SM = min(all three)
```

For 128 threads (4 warps) with 40 registers:
- From registers: 65536 / (40 * 128) = 12
- From warps: 48 / 4 = 12
- Maximum occupancy: 12 blocks/SM * 4 warps = 48 warps = 100%

For 64 threads (2 warps) with 40 registers:
- From registers: 65536 / (40 * 64) = 25
- From warps: 48 / 2 = 24
- Maximum occupancy: 24 blocks/SM * 2 warps = 48 warps = 100%

**Key insight:** Register count matters. Each additional register per thread
reduces the number of blocks that fit on an SM. Going from 40 to 48 registers
per thread with 128-thread blocks drops max blocks from 12 to 10. That is a 17%
reduction in theoretical occupancy.


## 7. Theoretical Performance Targets

The kernel is **memory-bandwidth-bound**. The weight matrix B dominates memory
traffic. The activation A and output C are negligible (a few KB vs several MB).

### Target: achievable memory bandwidth

On RTX 4090, achievable DRAM bandwidth for streaming workloads is typically
**750-850 GB/s** (75-85% of the 1008 GB/s theoretical peak). The remaining 15-25%
is lost to:
- DRAM refresh cycles
- Memory controller overhead
- Address translation
- Imperfect occupancy / latency hiding

**Our target: 750+ GB/s sustained for large shapes.**

### Per-shape theoretical minimum time

At 800 GB/s (conservative achievable target):

| Shape              | Data (k=4) | Min time @ 800 GB/s |
|--------------------|-----------|---------------------|
| 2048 x 5120        | 5.31 MB   | 6.6 us              |
| 5120 x 2048        | 5.31 MB   | 6.6 us              |
| 2048 x 4096        | 4.25 MB   | 5.3 us              |
| 4096 x 2048        | 4.25 MB   | 5.3 us              |
| 2048 x 2048        | 2.13 MB   | 2.7 us              |
| 2048 x 512         | 0.53 MB   | 0.66 us             |
| 512 x 2048         | 0.53 MB   | 0.66 us             |

Small shapes (0.5 MB) will be dominated by launch overhead (2-3 us) and can never
reach their bandwidth limit. These are batched via the grouped GEMV kernel.


## 8. Build System: Only Compile What You Need

Full compilation of `ops.cu` takes a long time because it contains many template
instantiations for all kernel variants (MMA kernels, dequantize kernels, quantize
kernels, etc.) across multiple architectures.

### Fast rebuild for scalar GEMV development

The project uses CMake with a build directory at `build/`. To rebuild only what
changed after modifying the scalar GEMV kernel in `csrc/ops.cu`:

```bash
cd /home/tim/git/bnb-kbit-gemm/build
cmake --build . --config Release 2>&1 | tail -5
```

**Tip:** If you are only modifying the scalar GEMV kernel code (not adding new
template instantiations or changing headers), the incremental rebuild only
recompiles `ops.cu`. This is still slow (~60-90 seconds) because the entire file
is one compilation unit.

### Reducing compile time

To iterate faster on the kernel, you can:

1. **Only compile for sm_89** (the RTX 4090). Edit `CMakeLists.txt` or pass
   `-DCOMPUTE_CAPABILITY=89` to cmake. This avoids compiling for sm_75, sm_80,
   sm_86, sm_90, etc.

2. **Minimize template instantiations.** The scalar GEMV kernel is instantiated
   for all combinations of:
   - k = 2, 3, 4, 5 (bit widths)
   - M_VAL = 1, 2, 4 (batch size templates)
   - scalar_t = half, __nv_bfloat16 (data types)
   - N_TILE = 64, 128 (tile sizes)

   That is `4 * 3 * 2 * 2 = 48` instantiations. During development, you can
   temporarily reduce this to just k=4, M_VAL=1, half, N_TILE=128 (1 variant)
   and add back the others when the kernel is working. The instantiations are
   near the end of `ops.cu` — look for `LAUNCH_SCALAR_GEMV` and the explicit
   template instantiations of `kbitScalarGemv`.

3. **Use `ccache`** if available — it caches compilation results.


## 9. ncu Benchmarking: The Only Benchmark That Matters

**Do NOT use Python-side benchmarking** (torch.cuda.Event timing). Python
dispatch overhead is 30-40 us, which completely dominates the 5-15 us kernel time.
Python benchmarks tell you nothing about kernel performance.

**Only use NVIDIA Nsight Compute (ncu).**

### The profiling script

Create `/tmp/ncu_scalar_gemv.py`:

```python
"""Minimal ncu profiling script for scalar GEMV kernel."""
import os, sys, torch
sys.path.insert(0, "/home/tim/git/bnb-kbit-gemm")
import bitsandbytes
from bitsandbytes import _ops
from scipy.stats import norm

def create_cb(k):
    n_levels = 1 << k
    quantiles = torch.linspace(0.5/n_levels, 1.0 - 0.5/n_levels, n_levels)
    values = torch.tensor(norm.ppf(quantiles.numpy()), dtype=torch.float32)
    return (values / values.abs().max()).cuda()

# Select shape from environment
shapes = [
    ("dense_gateup", 2048, 5120),
    ("dense_down",   5120, 2048),
    ("Q_proj",       2048, 4096),
    ("O_proj",       4096, 2048),
    ("KV_proj",      2048, 512),
    ("linear_key",   2048, 2048),
    ("MoE_gateup",   2048, 512),
    ("MoE_down",     512,  2048),
]
shape_idx = int(os.environ.get("SHAPE_IDX", "0"))
name, K_dim, N = shapes[shape_idx]
k = int(os.environ.get("K_BITS", "4"))
M = int(os.environ.get("M_VAL", "1"))

print(f"Shape: {name} K={K_dim} N={N} M={M} k={k}", file=sys.stderr)

cb = create_cb(k)
W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
pf, am = torch.ops.bitsandbytes.quantize_kbit(W.reshape(-1), cb, k)
pt, at = torch.ops.bitsandbytes.repack_kbit(pf, am.cuda(), K_dim, N, k)
A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
C = torch.empty(M, N, device="cuda", dtype=torch.float16)

# Warmup
for _ in range(5):
    torch.ops.bitsandbytes.kbit_scalar_gemv(A, pt, at, cb, K_dim, N, k, 0, out=C)
torch.cuda.synchronize()

# Profiled call
torch.ops.bitsandbytes.kbit_scalar_gemv(A, pt, at, cb, K_dim, N, k, 0, out=C)
torch.cuda.synchronize()
```

### Quick ncu command: one shape, key metrics

```bash
SHAPE_IDX=0 ncu --kernel-name "kbit_scalar_gemv" \
    --launch-skip 5 --launch-count 1 \
    --metrics "gpu__time_duration.avg,\
dram__throughput.avg_pct_of_peak_sustained_elapsed,\
sm__throughput.avg_pct_of_peak_sustained_elapsed,\
sm__warps_active.avg_pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__grid_size,launch__block_size,\
launch__shared_mem_per_block_dynamic" \
    python /tmp/ncu_scalar_gemv.py
```

### Full ncu profile (when you need stall reasons, occupancy details)

```bash
SHAPE_IDX=0 ncu --kernel-name "kbit_scalar_gemv" \
    --launch-skip 5 --launch-count 1 \
    --set full \
    python /tmp/ncu_scalar_gemv.py
```

The `--set full` output includes:
- **GPU Speed Of Light**: DRAM throughput %, compute throughput %, duration
- **Memory Workload Analysis**: sectors, bank conflicts, L1/L2 hit rates
- **Warp State Statistics**: stall reasons, IPC, eligible warps
- **Occupancy**: theoretical vs achieved, limiting factors
- **Source Counters**: per-line stall attribution

### Profile all shapes at once

```bash
for i in 0 1 2 3 4 5 6 7; do
  result=$(SHAPE_IDX=$i ncu --kernel-name "kbit_scalar_gemv" \
      --launch-skip 5 --launch-count 1 \
      --metrics "gpu__time_duration.avg,launch__grid_size,launch__block_size,\
launch__registers_per_thread,dram__throughput.avg_pct_of_peak_sustained_elapsed" \
      python /tmp/ncu_scalar_gemv.py 2>&1)
  name=$(echo "$result" | grep "Shape:" | sed 's/Shape: //')
  time=$(echo "$result" | grep "gpu__time_duration.avg" | awk '{print $NF}')
  grid=$(echo "$result" | grep "launch__grid_size" | awk '{print $NF}')
  bw=$(echo "$result" | grep "dram__throughput" | awk '{print $NF}')
  echo "$name: ${time} us, grid=$grid, DRAM=${bw}%"
done
```

### Profile across all k values (2-5)

```bash
for k in 2 3 4 5; do
  result=$(SHAPE_IDX=0 K_BITS=$k ncu --kernel-name "kbit_scalar_gemv" \
      --launch-skip 5 --launch-count 1 \
      --metrics "gpu__time_duration.avg,dram__throughput.avg_pct_of_peak_sustained_elapsed" \
      python /tmp/ncu_scalar_gemv.py 2>&1)
  time=$(echo "$result" | grep "gpu__time_duration.avg" | awk '{print $NF}')
  bw=$(echo "$result" | grep "dram__throughput" | awk '{print $NF}')
  echo "k=$k: ${time} us, DRAM=${bw}%"
done
```

### What to look at in ncu output

The metrics to focus on, in order of importance:

1. **`gpu__time_duration.avg`** — wall-clock kernel time in microseconds.
   This is the number you are optimizing.

2. **`dram__throughput.avg_pct_of_peak_sustained_elapsed`** — percentage of peak
   DRAM bandwidth achieved. Target: 75%+. If this is low, you are not issuing
   enough memory requests or are stalling too much.

3. **`launch__registers_per_thread`** — register count. Directly determines max
   blocks per SM. Keep at 40 or below for 128-thread blocks (gives 12 blocks/SM).

4. **`launch__grid_size`** — number of blocks launched. Must be >= num_SMs (128)
   for any occupancy. Ideally >= 12 * 128 = 1536 for full occupancy.

5. **`sm__warps_active.avg_pct_of_peak_sustained_active`** — achieved occupancy.
   Low occupancy means not enough warps to hide memory latency.

6. **Stall reasons** (from `--set full`): Look for "scoreboard" stalls (waiting
   for memory) and "barrier" stalls (waiting for __syncthreads). These tell you
   what to fix.


## 10. Step-by-Step Kernel Development

Build the kernel incrementally. Each step should be profiled with ncu before
moving to the next. **Do not test correctness until Step 5.**

### Step 1: Vector Load Skeleton — Achieve 100% Memory Throughput

**Goal:** A kernel that reads all the B_packed data using vector loads and does
nothing with it. This establishes the memory throughput ceiling.

```c
// Pseudocode for Step 1
__global__ void kbit_scalar_gemv_step1(
    const unsigned int* B_packed,
    scalar_t* C,
    int K_dim, int N
) {
    // One warp per output column (like bnb gemv_4bit)
    // Each warp reads all K elements for its column via int4 vector loads
    // Accumulate into a dummy variable to prevent optimization
    // WarpReduce and write result
}
```

Key design decisions:
- **Block size:** 128 threads = 4 warps. Each warp handles one output column.
  Grid = N / 4 blocks. For N=5120: 1280 blocks.
- **Vector loads:** Use `int4` (16 bytes) loads for B_packed. Each int4 loads
  4 uint32 words = 4 bit-plane words. For k=4, this is exactly one column's
  data for one k-block.
- **No shared memory needed** for this step — load directly from global memory
  into registers (like the bnb kernel).
- **No tiling needed** — each warp independently streams through all K data for
  its column.

**Expected result:** Kernel time should be close to `data_size / 800 GB/s`.
DRAM throughput should be 75-85%. If not, the grid is too small (need more
blocks or split-K) or the loads are not coalesced.

#### Occupancy considerations for Step 1

For N=5120: grid = 1280, capacity = 12 * 128 = 1536. Waves = 0.83. Not great.
For N=512: grid = 128, capacity = 1536. Waves = 0.08. Terrible.

**Split-K** is needed for small shapes: split the K dimension across multiple
warps, each processing a subset of K, then atomicAdd partial results. This
increases the grid size proportionally.

### Step 2: Add Bit-Plane Extraction

Add the bit extraction logic to convert bit-planes into k-bit indices.

```c
// In the inner loop, after loading k uint32 planes:
int idx = 0;
for (int b = 0; b < k; b++)
    idx |= ((planes[b] >> j) & 1) << b;
```

Profile again. The additional ALU instructions should not significantly impact
a memory-bound kernel. If DRAM throughput drops, the extra instructions are
stalling the memory pipeline — you need more warps (higher occupancy) to hide
the compute latency.

### Step 3: Add Codebook Lookup via Shuffle

Add the shuffle-based codebook lookup:

```c
float cb = (lane_id < (1 << k)) ? codebook[lane_id] : 0.0f;
// ...
float weight = __shfl_sync(0xFFFFFFFF, cb, idx);
```

The shuffle is 1 cycle and should have negligible impact.

### Step 4: Add Absmax Decoding and Scale

Add the E4M4 absmax decode and multiply:

```c
float amax = decode_e4m4_absmax_branchless(absmax_byte);
float dequantized_weight = weight * amax;
```

At this point you have full dequantization. Profile to confirm memory throughput
is maintained.

### Step 5: Add A Loading and FMA — Complete Kernel

Add the activation vector load and FMA accumulation:

```c
// Load A values (vector load, 8 fp16 at a time)
// FMA: accumulator += dequantized_weight * a_value
```

Add warp reduction and output write.

**Now test correctness.** Run the full test suite:

```bash
pytest tests/test_scalar_gemv.py -v --tb=short -x
```

### Step 6: Optimize

Once the kernel is correct and you understand the ncu profile at each step,
optimize:

1. **Reduce register count** if above 40 (use `__launch_bounds__` if needed)
2. **Fix bank conflicts** if shared memory is used
3. **Tune split-K** for each shape category
4. **Consider cp.async** for loading B to overlap with compute
5. **Tune TILE_N** (64 vs 128) per shape for better grid occupancy

### Important: test all k values

Every optimization must work for **k = 2, 3, 4, and 5**. The data sizes,
register usage, and loop trip counts all change with k. A kernel that is fast
for k=4 but broken for k=2 is useless.

When profiling, always check at least k=2, k=4, and k=5 to cover the range:

```bash
for k in 2 3 4 5; do
  echo "--- k=$k ---"
  K_BITS=$k ncu --kernel-name "kbit_scalar_gemv" --launch-skip 5 --launch-count 1 \
      --metrics "gpu__time_duration.avg,launch__registers_per_thread" \
      python /tmp/ncu_scalar_gemv.py 2>&1 | grep -E "time_duration|registers"
done
```


## 11. Testing: Correctness at the End

**Do not test correctness until the kernel is complete (Step 5).** Partial
kernels produce garbage output — testing them wastes time.

### Test suite

The test file is `tests/test_scalar_gemv.py`. Run with:

```bash
pytest tests/test_scalar_gemv.py -v --tb=short -x -p no:randomly
```

The `-p no:randomly` flag disables test randomization so failures are
reproducible.

### What the tests cover

- **`test_basic_correctness`**: k=2,3,4,5 x M=1,2,3,4 at shape (2048, 512).
  Compares against the MMA kernel (`kbit_gemm_prod`).
- **`test_various_shapes`**: Multiple (K, N) combinations at k=4, M=1.
  Covers 2048x5120, 5120x2048, 2048x4096, 512x2048.
- **`test_no_splitk`**: Forced k_chunks=1 (no split-K) for k=1,2,3,4.
- **`test_dtype`**: fp16 and bf16 at k=4, M=2.
- **`test_grouped_*`**: Grouped GEMV (MoE batching) tests.

### k=2 through k=5 coverage

The `test_basic_correctness` test is parametrized over `k=[2,3,4,5]` and
`M=[1,2,3,4]`. This gives 16 test cases that cover all kbit/batch combinations.
**All 16 must pass.** Do not ship a kernel that fails for any k value.

### Common correctness issues

1. **Stale split-K workspace.** The `C_workspace` and `tile_counters` tensors
   are cached and reused across calls. They MUST be zeroed before each call.
   The Python side (`_kbit_scalar_gemv_impl` in `backends/cuda/ops.py`) does
   `C_workspace.zero_()` and `tile_counters.zero_()`.

2. **tile_counters size.** If you change TILE_N dynamically (e.g., TILE_N=64
   for small shapes), the number of n_tiles changes. The tile_counters array
   must be large enough for the maximum possible n_tiles. Currently allocated
   as `N // 64` entries (covering both TILE_N=64 and TILE_N=128).

3. **Repack tile size mismatch.** The repack kernel uses KBIT_TILE_K=64 and
   KBIT_TILE_N=128 (hardcoded constants at line ~872 of ops.cu). If you change
   the GEMV kernel's tile sizes, you must either:
   - Keep reading from the 128-column repack tiles (using sub-tile offsets), or
   - Change the repack kernel to match (requires re-quantizing all weights).

4. **A tile loading for M > 1.** The activation matrix A is [M, K_dim] in
   row-major layout. When loading a tile of A, rows are NOT contiguous — each
   row is K_dim elements apart. Do NOT use flat cp.async / memcpy for A when
   M > 1. Use per-element loads with proper row indexing.


## 12. Current Kernel State

The kernel in `csrc/ops.cu` (search for `kbit_scalar_gemv`) currently implements:

### Dense scalar GEMV (`kbit_scalar_gemv`)

- Template parameters: `K_BITS` (2-5), `M_VAL` (1/2/4), `N_TILE` (64/128),
  `scalar_t` (half/bf16).
- TILE_K = 64, matching the repack layout.
- Single-buffered shared memory: loads B tile + absmax + A tile into shmem,
  syncs, computes, syncs, next tile.
- B loaded via cp.async int4 vector loads (bypasses L1 cache).
- A loaded via regular loads with bounds checking.
- Codebook in registers via warp shuffle.
- Split-K with atomicAdd and tile_counters for reduction.
- Persistent work loop (grid-stride loop over work items).
- Dynamic TILE_N selection: 64 for small shapes, 128 for large shapes.

### Grouped scalar GEMV (`kbit_grouped_scalar_gemv`)

- For MoE: batches multiple experts into one kernel launch.
- Each block handles one (expert, n_tile) pair.
- Binary search to find expert ID from flattened work index.
- Double-buffered cp.async pipeline.
- No split-K needed (enough parallelism from multiple experts).

### ncu Performance (as of last measurement, M=1, k=4)

| Shape              | GPU time  | DRAM throughput | Grid  | Registers |
|--------------------|-----------|----------------|-------|-----------|
| 2048 x 5120        | 14.85 us  | ~54%           | 1280  | 40        |
| 5120 x 2048        | 15.74 us  | ~54%           | 1280  | 40        |
| 2048 x 4096        | 12.29 us  | ~54%           | 1024  | 40        |
| 4096 x 2048        | 13.06 us  | ~54%           | 1024  | 40        |
| 2048 x 512          | 4.58 us  | ~11%           | 256   | 48        |
| 2048 x 2048        | 8.29 us   | ~24%           | 512   | 40        |
| 512 x 2048          | 4.70 us  | ~11%           | 256   | 48        |

### Gap to theoretical target

| Shape              | Current   | Target @800 GB/s | Gap   |
|--------------------|-----------|-------------------|-------|
| 2048 x 5120        | 14.85 us  | 6.6 us            | 2.2x  |
| 2048 x 4096        | 12.29 us  | 5.3 us            | 2.3x  |
| 2048 x 2048        | 8.29 us   | 2.7 us            | 3.1x  |
| 2048 x 512         | 4.58 us   | 0.66 us           | 6.9x  |

The large shapes are at ~54% of peak DRAM bandwidth. The main bottleneck is
the shared-memory-based tiling approach with syncthreads barriers. The bnb
reference kernel avoids shared memory entirely.

**Recommendation:** Consider rewriting following the bnb pattern — direct
register-file loads from global memory, warp-level parallelism, no shared
memory tiles, no syncthreads. This eliminates the barrier overhead that
currently costs ~45% of peak bandwidth.


## 13. Known Issues and Pitfalls

### Register pressure with higher k

Higher k values (k=5) require more registers for the bit-plane words:
- k=2: 2 uint32 registers for planes
- k=4: 4 uint32 registers
- k=5: 5 uint32 registers

Plus the loop generates more ALU instructions for index extraction. Monitor
`launch__registers_per_thread` across all k values — if k=5 pushes registers
above 42 (with 128-thread blocks), max blocks/SM drops below 12.

### Bank conflicts in shared memory

The current tiled layout can cause bank conflicts when threads in a warp read
from shmem addresses that map to the same bank. With the `[col][kb][bit]`
layout and 128 threads reading `sh_b[col * B_COL_WORDS + kb * k + b]`:

- For k=4: B_COL_WORDS = 8. Thread 0 reads word 0, thread 1 reads word 8,
  thread 4 reads word 32 = same bank as word 0 (32 banks, 4 bytes each).
  This causes 4-way bank conflicts with k=4.

If you stay with shared memory, consider adding +1 padding to eliminate bank
conflicts: `sh_b[col * (B_COL_WORDS + 1) + ...]`.

### The "same waves" problem with TILE_N

Reducing TILE_N from 128 to 64 doubles the number of n_tiles but also doubles
the SM block capacity (from 12 to 24 blocks/SM). The ratio
`total_work / capacity` stays the same. This means:

- TILE_N=64 does NOT improve occupancy in terms of warps
- It does give more blocks (better load balancing for uneven work)
- It does incur higher register usage (48 vs 40) due to sub-tile offset math

Choose TILE_N=64 only when N is not divisible by 128, or when you need the
load-balancing benefit (marginal).

### cp.async alignment requirements

`cp.async.cg.shared.global` requires 16-byte alignment for both source and
destination addresses. When computing sub-tile offsets into the repacked B data,
verify that `sub_col_offset * B_COL_WORDS * sizeof(uint32)` is a multiple of 16.

For the common cases:
- k=2, B_COL_WORDS=4: 64 * 4 * 4 = 1024 bytes. 1024 % 16 = 0. OK.
- k=3, B_COL_WORDS=6: 64 * 6 * 4 = 1536 bytes. 1536 % 16 = 0. OK.
- k=4, B_COL_WORDS=8: 64 * 8 * 4 = 2048 bytes. 2048 % 16 = 0. OK.
- k=5, B_COL_WORDS=10: 64 * 10 * 4 = 2560 bytes. 2560 % 16 = 0. OK.

All fine because `64 * k * 2 * 4` is always a multiple of 16 for k >= 2.

### Python-side caching

The split-K workspace and tile counters are cached in a Python dict keyed by
`(device, M, N)`. If you change the kernel's tiling such that different shapes
need different workspace sizes, the cache may return a too-small tensor. Either:
- Always allocate for the worst case (current approach: `N // 64`)
- Clear the cache when shapes change
- Don't cache at all (minor overhead from allocation)

### Compile time explosion

The scalar GEMV kernel is instantiated for every combination of:
- k = 2, 3, 4, 5
- M_VAL = 1, 2, 4
- N_TILE = 64, 128
- scalar_t = half, bf16

That is 48 kernel variants. Each takes ~1-2 seconds to compile. To iterate
faster during development, temporarily reduce to k=4, M_VAL=1, half, N_TILE=128
only (1 variant). The instantiation macros are near the end of `ops.cu` — search
for `LAUNCH_SCALAR_GEMV` and the explicit template instantiations.

---

## Appendix A: File Map

| File | Purpose |
|------|---------|
| `csrc/ops.cu` | All CUDA kernels (quantize, repack, GEMM, GEMV) |
| `csrc/ops.cuh` | C++ launcher declarations |
| `csrc/pythonInterface.cpp` | C-linkage wrappers called from Python |
| `bitsandbytes/_ops.py` | PyTorch op definitions (schema, fake implementations) |
| `bitsandbytes/backends/cuda/ops.py` | CUDA backend: Python → C++ bridge |
| `tests/test_scalar_gemv.py` | Test suite for dense + grouped scalar GEMV |
| `benchmarks/bench_scalar_gemv.py` | Python-side benchmark (for reference only) |

### Key locations in ops.cu

| Line (approx) | Content |
|----------------|---------|
| 724 | `decode_e4m4_absmax` / `decode_e4m4_absmax_branchless` |
| 762 | `encode_e4m4_absmax` |
| 872 | Repack tile constants (`KBIT_TILE_K=64`, `KBIT_TILE_N=128`) |
| 877 | `kRepackKbit` kernel |
| 1161 | cp.async helper functions |
| 2563 | `kbit_scalar_gemv` kernel |
| 2737 | Launcher: `kbitScalarGemvLaunchTiled` |
| 2805 | Launcher: `kbitScalarGemvLaunch` (TILE_N selection) |
| 2833 | Public entry: `kbitScalarGemv` (M_VAL dispatch) |
| 2874 | `kbit_grouped_scalar_gemv` kernel (MoE) |


## Appendix B: Quick Reference — ncu One-Liners

Profile the largest shape (dense gate/up 2048x5120), full metrics:
```bash
SHAPE_IDX=0 ncu --kernel-name "kbit_scalar_gemv" --launch-skip 5 --launch-count 1 --set full python /tmp/ncu_scalar_gemv.py
```

Profile KV proj (small shape, 2048x512):
```bash
SHAPE_IDX=4 ncu --kernel-name "kbit_scalar_gemv" --launch-skip 5 --launch-count 1 --set full python /tmp/ncu_scalar_gemv.py
```

Profile with k=2 (minimum bit width):
```bash
SHAPE_IDX=0 K_BITS=2 ncu --kernel-name "kbit_scalar_gemv" --launch-skip 5 --launch-count 1 --set full python /tmp/ncu_scalar_gemv.py
```

Profile with M=4 (maximum batch size):
```bash
SHAPE_IDX=0 M_VAL=4 ncu --kernel-name "kbit_scalar_gemv" --launch-skip 5 --launch-count 1 --set full python /tmp/ncu_scalar_gemv.py
```


## Appendix C: The bnb Kernel Constants

For reference, the upstream bnb `kgemm_4bit_inference_naive` kernel uses:

```c
#define num_values_4bit 32      // elements processed per K-iteration per lane
THREADS = 128                   // 4 warps per block
BITS = 16                       // fp16 = 16 bits per A element
```

Per lane per K-iteration:
- Reads 16 bytes of packed B (32 nibbles = 32 4-bit values via one int4 load)
- Reads 4 x 16 bytes of A (4 sub-iterations, 8 fp16 values each via int4 loads)
- Processes 32 weight elements total
- Loads 1 float32 absmax
- Grid: `(N + 3) / 4` blocks (4 output rows per block = 4 warps)

The kernel achieves ~4x speedup over dequantize-then-cuBLAS for M=1 inference.
Our kbit kernel should aim for similar or better speedup at all k values (2-5).
