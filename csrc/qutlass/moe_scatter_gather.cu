/*
 * Scatter and gather kernels for MoE batched NVFP4 GEMM pipeline.
 *
 * Scatter: copies packed FP4/uint8 data from concatenated token layout to
 *          padded per-expert batched layout. Zero-fills padding rows.
 *          Works for both packed FP4 activations (row_bytes = K/2) and
 *          scale factors (same kernel, different row_bytes).
 *
 * Gather: copies BF16 results from padded per-expert batched layout
 *         back to concatenated token layout.
 *
 * Weighted gather: fused gather + multiply by expert gating weight +
 *         atomicAdd into output. Single kernel replaces gather + scale + sum.
 *
 * All kernels use one threadblock per expert with vectorized 128-bit
 * (uint4) loads/stores for bandwidth efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// Scatter: concatenated FP4 → padded per-expert batched FP4
// =========================================================================
// Grid: (num_experts, chunks_per_expert).  Each block handles a byte-range
// slice of one expert's total_bytes = max_M * row_bytes, splitting work
// across multiple SMs for bandwidth saturation on wide GPUs (B200: 160 SMs).
//
// Data layout:
//   Input:  packed_concat [total_tokens * row_bytes] contiguous
//   Output: packed_batched [num_experts * max_M * row_bytes] with zero padding
//
// row_bytes = K / 2 (packed FP4: 2 values per byte)
__global__ void kMoeScatterNVFP4(
    const uint8_t* __restrict__ input,    // [total_tokens * row_bytes]
    uint8_t* __restrict__ output,          // [num_experts * max_M * row_bytes]
    const int* __restrict__ expert_offsets, // [num_experts + 1] cumulative token offsets
    int max_M,                             // padded max tokens per expert
    int row_bytes,                         // K / 2
    int chunks_per_expert                  // gridDim.y
) {
    int expert = blockIdx.x;
    int chunk  = blockIdx.y;

    int start = expert_offsets[expert];
    int end = expert_offsets[expert + 1];
    int n_tokens = end - start;

    const uint8_t* src = input + (long long)start * row_bytes;
    uint8_t* dst = output + (long long)expert * max_M * row_bytes;

    long long total_bytes = (long long)max_M * row_bytes;
    long long data_bytes = (long long)n_tokens * row_bytes;

    // This block's byte range (aligned to 16 for vectorization)
    long long bytes_per_chunk = ((total_bytes + chunks_per_expert - 1) / chunks_per_expert + 15) & ~15LL;
    long long my_start = (long long)chunk * bytes_per_chunk;
    long long my_end = min(my_start + bytes_per_chunk, total_bytes);
    if (my_start >= total_bytes) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Process byte range [my_start, my_end) — copy from src where < data_bytes, zero otherwise
    // Use uint4 (16-byte) vectorization
    long long vec_start = (my_start + 15) / 16;  // first full uint4 in range
    long long vec_end = my_end / 16;              // last full uint4 in range

    // Scalar head bytes
    for (long long i = my_start + tid; i < min(vec_start * 16, my_end); i += stride) {
        dst[i] = (i < data_bytes) ? src[i] : 0;
    }

    // Vectorized middle
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    uint4 zero4 = make_uint4(0, 0, 0, 0);
    long long data_vec_boundary = data_bytes / 16;  // last full uint4 within data

    for (long long i = vec_start + tid; i < vec_end; i += stride) {
        if (i < data_vec_boundary) {
            dst4[i] = src4[i];
        } else if (i * 16 >= data_bytes) {
            dst4[i] = zero4;
        } else {
            // Straddles data/padding boundary — byte-by-byte
            uint8_t tmp[16];
            const uint8_t* s = src + i * 16;
            for (int b = 0; b < 16; b++) {
                long long pos = i * 16 + b;
                tmp[b] = (pos < data_bytes) ? s[b] : 0;
            }
            dst4[i] = *reinterpret_cast<uint4*>(tmp);
        }
    }

    // Scalar tail bytes
    for (long long i = vec_end * 16 + tid; i < my_end; i += stride) {
        dst[i] = (i < data_bytes) ? src[i] : 0;
    }
}


// =========================================================================
// Gather: padded per-expert BF16 → concatenated BF16
// =========================================================================
// Grid: (num_experts, chunks_per_expert).  Each block handles a byte-range
// slice of one expert's data_bytes = n_tokens * row_bytes.
//
// Data layout:
//   Input:  D_batched [num_experts * max_M * N] bf16
//   Output: D_concat [total_tokens * N] bf16
//
// row_bytes = N * 2 (bf16 = 2 bytes per element)
__global__ void kMoeGatherBF16(
    const uint8_t* __restrict__ input,     // [num_experts * max_M * row_bytes]
    uint8_t* __restrict__ output,           // [total_tokens * row_bytes]
    const int* __restrict__ expert_offsets, // [num_experts + 1]
    int max_M,
    int row_bytes,                          // N * 2
    int chunks_per_expert                   // gridDim.y
) {
    int expert = blockIdx.x;
    int chunk  = blockIdx.y;

    int start = expert_offsets[expert];
    int end = expert_offsets[expert + 1];
    int n_tokens = end - start;
    if (n_tokens <= 0) return;

    const uint8_t* src = input + (long long)expert * max_M * row_bytes;
    uint8_t* dst = output + (long long)start * row_bytes;

    long long data_bytes = (long long)n_tokens * row_bytes;

    // This block's byte range (aligned to 16)
    long long bytes_per_chunk = ((data_bytes + chunks_per_expert - 1) / chunks_per_expert + 15) & ~15LL;
    long long my_start = (long long)chunk * bytes_per_chunk;
    long long my_end = min(my_start + bytes_per_chunk, data_bytes);
    if (my_start >= data_bytes) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Vectorized uint4 copy over [my_start, my_end)
    long long vec_start = (my_start + 15) / 16;
    long long vec_end = my_end / 16;

    // Scalar head
    for (long long i = my_start + tid; i < min(vec_start * 16, my_end); i += stride) {
        dst[i] = src[i];
    }

    // Vectorized middle
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (long long i = vec_start + tid; i < vec_end; i += stride) {
        dst4[i] = src4[i];
    }

    // Scalar tail
    for (long long i = vec_end * 16 + tid; i < my_end; i += stride) {
        dst[i] = src[i];
    }
}


// =========================================================================
// Weighted gather: padded per-expert BF16 → FP32 accumulate → BF16 output
// =========================================================================
// Two-phase operation (both launched from one extern "C" call):
//   Phase 1: kMoeWeightedGatherAccum — read BF16 expert output, multiply by
//            gating weight, atomicAdd into FP32 workspace.
//   Phase 2: kConvertFP32ToBF16 — convert FP32 workspace to BF16 output.
//
// Uses a token-parallel layout: grid = (total_assignments,) where each
// assignment is a (token_id, expert_id, weight) triple. Atomic contention
// is minimal — at most top_k experts write to the same token row, and with
// N=4096 elements spread across 256 threads, collisions are rare.
//
// FP32 accumulation avoids BF16 rounding error across top_k additions.
// The final conversion to BF16 rounds once at the end.

__global__ void kMoeWeightedGatherAccum(
    const __nv_bfloat16* __restrict__ D_batched, // [num_experts * max_M * N]
    float* __restrict__ workspace,                // [num_tokens * N] fp32, zero-initialized
    const int* __restrict__ token_ids,            // [total_assignments]
    const int* __restrict__ expert_ids,           // [total_assignments]
    const int* __restrict__ slot_ids,             // [total_assignments]
    const float* __restrict__ weights,            // [total_assignments]
    int max_M,
    int N
) {
    int assign = blockIdx.x;
    int token_id = token_ids[assign];
    int expert_id = expert_ids[assign];
    int slot_id = slot_ids[assign];
    float w = weights[assign];

    const __nv_bfloat16* src = D_batched + ((long long)expert_id * max_M + slot_id) * N;
    float* dst = workspace + (long long)token_id * N;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float val = __bfloat162float(src[i]) * w;
        atomicAdd(&dst[i], val);
    }
}

__global__ void kConvertFP32ToBF16(
    const float* __restrict__ input,       // [n_elements]
    __nv_bfloat16* __restrict__ output,    // [n_elements]
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}


// =========================================================================
// extern "C" launchers
// =========================================================================

// Target enough total blocks to saturate GPU SMs.
// B200 has 160 SMs; 2× oversubscription hides latency.
static constexpr int kTargetBlocks = 320;

extern "C" void cmoe_scatter_nvfp4(
    const void* input,
    void* output,
    const int* expert_offsets,
    int max_M,
    int K,
    int num_experts,
    cudaStream_t stream
) {
    int row_bytes = K / 2;  // packed FP4: 2 values per byte
    int chunks = max(1, kTargetBlocks / max(num_experts, 1));

    dim3 grid(num_experts, chunks);
    dim3 block(256);

    kMoeScatterNVFP4<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        expert_offsets,
        max_M,
        row_bytes,
        chunks
    );
}

extern "C" void cmoe_gather_bf16(
    const void* input,
    void* output,
    const int* expert_offsets,
    int max_M,
    int N,
    int num_experts,
    cudaStream_t stream
) {
    int row_bytes = N * 2;  // bf16: 2 bytes per element
    int chunks = max(1, kTargetBlocks / max(num_experts, 1));

    dim3 grid(num_experts, chunks);
    dim3 block(256);

    kMoeGatherBF16<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        expert_offsets,
        max_M,
        row_bytes,
        chunks
    );
}

extern "C" void cmoe_weighted_gather_bf16(
    const void* D_batched,      // [num_experts * max_M * N] bf16
    void* output_bf16,          // [num_tokens * N] bf16, final output
    float* workspace_fp32,      // [num_tokens * N] fp32, caller-managed scratch
    const int* token_ids,       // [total_assignments]
    const int* expert_ids,      // [total_assignments]
    const int* slot_ids,        // [total_assignments]
    const float* weights,       // [total_assignments]
    int total_assignments,
    int num_tokens,
    int max_M,
    int N,
    cudaStream_t stream
) {
    if (total_assignments <= 0) return;

    int n_elements = num_tokens * N;

    // Zero the FP32 workspace
    cudaMemsetAsync(workspace_fp32, 0, (size_t)n_elements * sizeof(float), stream);

    // Phase 1: weighted accumulate into FP32 workspace
    {
        dim3 grid(total_assignments);
        dim3 block(256);

        kMoeWeightedGatherAccum<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(D_batched),
            workspace_fp32,
            token_ids,
            expert_ids,
            slot_ids,
            weights,
            max_M,
            N
        );
    }

    // Phase 2: convert FP32 → BF16
    {
        int threads = 256;
        int blocks = (n_elements + threads - 1) / threads;

        kConvertFP32ToBF16<<<blocks, threads, 0, stream>>>(
            workspace_fp32,
            static_cast<__nv_bfloat16*>(output_bf16),
            n_elements
        );
    }
}
