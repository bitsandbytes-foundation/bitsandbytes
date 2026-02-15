# Scalar GEMV Kernel Implementation Guide

**Location:** `agents/scalar_gemv_guide.md`
**Referenced from:** `optimization.md` Section 4 (P0: Scalar Kernel)
**Key files to modify:**
- `csrc/ops.cu` — CUDA kernel + launcher
- `csrc/pythonInterface.cpp` — C wrappers
- `bitsandbytes/_ops.py` — torch.library op definitions
- `bitsandbytes/backends/cuda/ops.py` — Python dispatch
- `tests/test_kbit_quantization.py` — correctness tests

**Context documents:** `progress.md` (full dev record), `optimization.md` (kernel strategy)

---

## 1. What This Kernel Does

Computes `C[M, N] = A[M, K_dim] * W_kbit[K_dim, N]^T` for M=1-4 using
scalar FMA instead of tensor core MMA. Supports both:
- **Single-matrix** (dense layers): one weight matrix
- **Grouped** (MoE experts): multiple expert weight matrices in one launch

Uses the same tiled kbit data format as the existing MMA kernels — no
repack changes needed.

### Why it's needed

At M=1, the MMA kernel wastes 93.75% of tensor core work (TILE_M=16,
only 1 row has data). cuBLAS uses an optimized GEMV at M=1, achieving
69% of peak DRAM bandwidth. Our MMA kernel achieves only 31%. The scalar
kernel eliminates MMA waste entirely and should achieve ~50-60% bandwidth
efficiency, translating the 3.6x data compression into a 2.5-3.5x speedup
over cuBLAS.

---

## 2. Architecture

### Thread/block organization

- **Block size:** 256 threads (8 warps), same as MMA kernel
- **TILE_N:** 128 output columns per block (same as MMA kernel)
- **TILE_K:** 64 (same as MMA kernel, matches tiled data format)
- **No TILE_M concept** — M is a runtime parameter (1-4), not tiled

Thread assignment for M=1:
- 256 threads, 128 columns → 2 threads per column
- Thread `t` and thread `t+128` split the K-dimension reduction
- Thread `t` handles even k_tiles, thread `t+128` handles odd k_tiles
- After all k_tiles: `__shfl_xor_sync` to reduce partial sums

Thread assignment for M=2-4:
- Each thread owns one column, processes all M rows
- 256 threads / 128 columns = 2 threads per column (split K)
- Each thread maintains M accumulators (`float acc[M_VAL]`)
- Dequant done once per element, weight reused across M rows

### Data flow per k_tile

1. **Load B tile** (kbit packed + absmax) into shared memory via cp.async
   - Same cp.async pipeline as MMA kernel (double-buffered)
   - B data: TILE_N × KB_PER_TILE × K_BITS uint32 words = 1024 words for K=4
   - Absmax: TILE_N × KB_PER_TILE = 256 bytes
2. **Load A values** directly into registers from global memory
   - M × TILE_K × sizeof(half) = 128-512 bytes (tiny, no shared memory needed)
   - Simple coalesced load, no XOR swizzle needed
3. **Dequant + FMA** in registers:
   - Read bit-plane words from shared memory
   - Extract K-bit index using bit manipulation
   - Codebook lookup via `__shfl_sync`
   - Scale by absmax
   - FMA: `acc[m] += weight * A_reg[m][k]`
4. **Store output** directly to global memory

### Codebook lookup

Same technique as the dequant kernel: codebook entries stored in lane
registers, lookup via `__shfl_sync`:

```cuda
// At kernel start: load codebook into lane registers
float cb = (lane_id < (1 << K_BITS)) ? codebook[lane_id] : 0.0f;

// During dequant: lookup by index
float val = __shfl_sync(0xFFFFFFFF, cb, idx);
float weight = val * amax;
```

This is register-to-register (~5 cycles), no shared memory needed for
the codebook.

### Shared memory budget

Per stage (one of two double-buffer slots):
- B tile: 128 × 2 × K × 4 bytes = 4096 bytes (K=4)
- Absmax: 256 bytes (aligned to 272)
- A tile: NOT in shared memory (loaded directly to registers)
- Total per stage: ~4368 bytes
- Double-buffered: ~8736 bytes

Much less than the MMA kernel (~15-20 KB), so occupancy will be higher.

---

## 3. Inner Loop Detail

For each k_tile, each thread processes its assigned column's k-blocks:

```cuda
// Thread owns column 'col', handles k-blocks based on thread assignment
// For M=1 with K-split: thread t handles k_blocks 0,2,4,...
//                        thread t+128 handles k_blocks 1,3,5,...
// (or split by k_tile: thread t does even k_tiles, t+128 odd k_tiles)

const int col = threadIdx.x % 128;  // output column
const int k_split_id = threadIdx.x / 128;  // 0 or 1

// After shared memory is ready for this k_tile:
unsigned int* b_ptr = sh_b(stage);
unsigned char* abs_ptr = sh_abs(stage);

#pragma unroll
for (int kb = 0; kb < KB_PER_TILE; kb++) {  // KB_PER_TILE = 2
    // Load K bit-plane words for this column's k-block
    unsigned int planes[K_BITS];
    int b_addr = col * B_COL_WORDS + kb * K_BITS;
    #pragma unroll
    for (int b = 0; b < K_BITS; b++)
        planes[b] = b_ptr[b_addr + b];

    float amax = decode_e4m4_absmax_branchless(abs_ptr[col * KB_PER_TILE + kb]);

    int k_base_local = kb * 32;  // within the k_tile
    int k_global = kt * TILE_K + k_base_local;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        // Extract K-bit index
        int idx = 0;
        #pragma unroll
        for (int b = 0; b < K_BITS; b++)
            idx |= ((planes[b] >> j) & 1) << b;

        float w = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;

        // FMA for each M row (dequant done once, reused)
        #pragma unroll
        for (int m = 0; m < M_VAL; m++)
            acc[m] += w * A_vals[m][k_global + j];
    }
}
```

### A value loading strategy

For M=1-4, A values are tiny. Two options:

**Option A (simpler, recommended for first version):**
Pre-load ALL A values for the full K_dim into registers at kernel start.
For M=1, K=2048: 2048 fp16 = 4 KB. At M=4: 16 KB. This exceeds register
file capacity, so use local memory (L1-cached, effectively free for
sequential access). Access pattern: `A_vals[m][k]`.

**Option B (more efficient):**
Load A values per k_tile into registers. For M=1, TILE_K=64: 64 fp16 =
128 bytes = 32 registers. Fits easily. Load from global memory at the
start of each k_tile iteration (while waiting for cp.async of B data).

Option B is better for register pressure. Implementation:
```cuda
// At start of each k_tile iteration:
half A_local[M_VAL][TILE_K];
for (int m = 0; m < M_VAL; m++)
    for (int i = 0; i < TILE_K; i += 8) {
        // Vectorized load: 8 halves = 16 bytes
        int k = kt * TILE_K + i;
        if (k + 7 < K_dim)
            *(int4*)&A_local[m][i] = *(const int4*)&A[m * K_dim + k];
    }
```

---

## 4. Work Distribution

### Single-matrix (dense layers)

Grid: one block per n_tile. For N=5120: 40 blocks. Each block processes
all K_dim for its 128 output columns.

For shapes with few n_tiles (N=512 → 4 blocks), use K-splitting:
launch more blocks, each handles a subset of k_tiles, atomicAdd partial
results to workspace. Same split-K logic as the production MMA kernel.

### Grouped (MoE experts)

Same as `kbit_grouped_gemm_prod`: persistent kernel with work_offsets,
binary search to find expert_id. Each work item is one (expert, n_tile).
No split-K (grouping provides enough parallelism).

The launcher computes work_offsets on the CPU side (tiny: num_experts+1
ints copied from device), same pattern as the existing grouped GEMM.

---

## 5. Template Parameters

```cuda
template <int K_BITS, int M_VAL, typename scalar_t>
__global__ void kbit_scalar_gemv(
    const scalar_t* __restrict__ A,
    const unsigned int* __restrict__ B_packed,
    const unsigned char* __restrict__ B_absmax,
    const float* __restrict__ codebook,
    scalar_t* __restrict__ C,
    float* __restrict__ C_workspace,      // for split-K
    int* __restrict__ tile_counters,       // for split-K
    const int M, const int K_dim, const int N,
    const int k_splits, const int total_work
);
```

- `K_BITS`: 2, 3, 4, 5 (compile-time, same as MMA kernel)
- `M_VAL`: 1, 2, 3, 4 (compile-time, controls unrolling)
- `scalar_t`: half, __nv_bfloat16

Grouped variant:
```cuda
template <int K_BITS, int M_VAL, typename scalar_t>
__global__ void kbit_grouped_scalar_gemv(
    const scalar_t* __restrict__ A_concat,
    const unsigned int* __restrict__ B_packed_all,
    const unsigned char* __restrict__ B_absmax_all,
    const float* __restrict__ codebook,
    scalar_t* __restrict__ C_concat,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ work_offsets,
    const int K_dim, const int N,
    const int num_experts, const int total_work
);
```

### Instantiations needed

For each K in {2,3,4,5} × M_VAL in {1,2,4} × scalar_t in {half, bf16}:
- 4 × 3 × 2 = 24 instantiations per kernel variant
- Start with K=4, M=1, fp16 only for initial testing (1 instantiation)
- Add remaining after correctness verified

---

## 6. Implementation Steps

### Step 1: CUDA kernel (`csrc/ops.cu`)

Add after the grouped GEMM code (around line 2547):

1. `kbit_scalar_gemv` kernel function (single-matrix with split-K)
2. `kbit_grouped_scalar_gemv` kernel function (grouped, no split-K)
3. `kbitScalarGemvLaunch` launcher (handles split-K grid sizing)
4. `kbitScalarGemv` public entry (M_VAL dispatch + SM query)
5. `kbitGroupedScalarGemv` public entry (M_VAL dispatch + work_offsets)
6. Template instantiations at end of file

### Step 2: C interface (`csrc/pythonInterface.cpp`)

Add forward declarations and extern C wrappers:
```cpp
// Forward declarations
#define MAKE_KBIT_SCALAR_GEMV_DECL(K) \
    void kbit_scalar_gemv_fp16_k##K(...); \
    void kbit_scalar_gemv_bf16_k##K(...);

// Extern C wrappers
#define MAKE_CKBIT_SCALAR_GEMV(K) \
    void ckbit_scalar_gemv_fp16_k##K(...) { \
        kbit_scalar_gemv_fp16_k##K(...); \
    }
```

Same pattern for grouped variant.

### Step 3: Python op registration (`bitsandbytes/_ops.py`)

Register two new ops:
```python
torch.library.define("bitsandbytes::kbit_scalar_gemv",
    "(Tensor A, Tensor B_packed, Tensor B_absmax, Tensor codebook, "
    "int K_dim, int N, int k) -> Tensor")

torch.library.define("bitsandbytes::kbit_grouped_scalar_gemv",
    "(Tensor A, Tensor B_packed_all, Tensor B_absmax_all, Tensor codebook, "
    "Tensor expert_offsets, int K_dim, int N, int k, int num_experts) -> Tensor")
```

### Step 4: Python dispatch (`bitsandbytes/backends/cuda/ops.py`)

Implement the CUDA backend kernels. Key: auto-select M_VAL template
based on actual M:
```python
@register_kernel("bitsandbytes::kbit_scalar_gemv", "cuda")
def _(A, B_packed, B_absmax, codebook, K_dim, N, k):
    M = A.shape[0]
    assert M <= 4
    # Allocate output, workspace, tile_counters
    # Call ckbit_scalar_gemv_{dtype}_k{k}
```

### Step 5: Correctness test

Add to `tests/test_kbit_quantization.py`:
```python
@pytest.mark.parametrize("K_dim,N", [(2048, 512), (2048, 5120), (5120, 2048)])
@pytest.mark.parametrize("M", [1, 2, 4])
@pytest.mark.parametrize("k", [4])
def test_scalar_gemv_correctness(K_dim, N, M, k):
    # Quantize weight, compute reference via dequant + torch.mm
    # Compare against kbit_scalar_gemv output
    # Tolerance: same as existing GEMM tests
```

### Step 6: Benchmark

Extend `benchmarks/bench_crossover.py` to include scalar GEMV in the
comparison table. Key comparison: scalar GEMV vs cuBLAS at M=1,2,4.

---

## 7. Expected Performance

Based on roofline analysis (see `optimization.md` Section 6):

| Shape | Scalar est (M=1) | cuBLAS (M=1) | Projected speedup |
|-------|------------------:|-------------:|------------------:|
| gate/up 2048×5120 | ~5us | ~25us | ~5x |
| down 5120×2048 | ~5us | ~25us | ~5x |
| Q proj 2048×4096 | ~4us | ~17us | ~4x |
| shared gate/up 2048×10240 | ~10us | ~55us | ~5.5x |
| MoE expert 2048×512 (×8) | ~4us | ~17us | ~4x |

Full model per-layer (all projections combined):
- Qwen3 batch=1: ~27us kbit vs ~141us cuBLAS = **5.3x**
- GLM4.7 batch=1: ~37us kbit vs ~157us cuBLAS = **4.3x**

These use a 1.8x overhead factor over theoretical bandwidth minimum.
The actual speedup depends on achieved bandwidth efficiency.

---

## 8. Key Differences from MMA Kernel

| Aspect | MMA kernel | Scalar kernel |
|--------|-----------|---------------|
| Inner compute | `mma.sync.aligned.m16n8k16` | Scalar FMA loop |
| A data | Shared memory + ldmatrix + XOR swizzle | Registers (direct global load) |
| B dequant output | Pack into MMA fragments (uint32) | Float value, used directly |
| Thread→output mapping | Complex (gid/tid fragment layout) | Simple (thread % 128 = column) |
| M handling | TILE_M=16, zero-padded | M_VAL template, no padding |
| Registers/thread | ~128 (MMA fragments) | ~30-40 |
| Occupancy | Low (register-limited) | High |
| Shared memory | A tile + B tile + absmax (~15-20 KB) | B tile + absmax only (~9 KB) |

---

## 9. Risks and Mitigations

1. **Shared memory bank conflicts on B reads.** Multiple threads reading
   the same column's bit-plane words from shared memory. Mitigation:
   with 2 threads per column (K-split), only 2-way conflict. Acceptable.

2. **Codebook shuffle across warp boundaries.** `__shfl_sync` only works
   within a warp. Threads in different warps processing the same column
   need independent codebook registers. This is already handled: each
   thread loads `cb = codebook[lane_id]` at kernel start.

3. **Register spill for M=4.** Each thread needs 4 accumulators + A values
   + packed words + temporaries. Estimate: ~40 registers. Fine for sm_89
   (255 max registers per thread).

4. **K-split reduction overhead.** For single-matrix with N=512 (4 blocks),
   need split-K to fill 128 SMs. atomicAdd overhead for the split-K
   reduction adds ~5-10us. Still much faster than MMA kernel. For grouped
   dispatch, split-K is unnecessary (enough experts to fill SMs).
