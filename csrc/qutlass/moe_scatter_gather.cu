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
// Each threadblock handles one expert. Threads cooperatively copy
// n_tokens * row_bytes from the concatenated source to the padded
// destination, then zero-fill padding rows.
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
    int row_bytes                          // K / 2
) {
    int expert = blockIdx.x;
    int start = expert_offsets[expert];
    int end = expert_offsets[expert + 1];
    int n_tokens = end - start;

    // Source: contiguous in concatenated buffer
    const uint8_t* src = input + (long long)start * row_bytes;

    // Destination: padded slot for this expert
    uint8_t* dst = output + (long long)expert * max_M * row_bytes;

    // Total bytes to process for this expert (data + padding)
    long long total_bytes = (long long)max_M * row_bytes;
    long long data_bytes = (long long)n_tokens * row_bytes;

    // Use vectorized uint4 (16-byte) copies where possible
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Copy data rows using uint4 vectorization
    long long vec_data_bytes = (data_bytes / 16) * 16;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    long long n_vec = vec_data_bytes / 16;

    for (long long i = tid; i < n_vec; i += stride) {
        dst4[i] = src4[i];
    }

    // Handle remaining bytes in data region
    for (long long i = vec_data_bytes + tid; i < data_bytes; i += stride) {
        dst[i] = src[i];
    }

    // Zero-fill padding region using uint4
    long long pad_start = data_bytes;
    long long pad_bytes = total_bytes - pad_start;

    if (pad_bytes > 0) {
        // Align pad_start up to 16-byte boundary for vectorized zeroing
        long long aligned_pad_start = ((pad_start + 15) / 16) * 16;

        // Zero unaligned bytes at start of padding
        for (long long i = pad_start + tid; i < aligned_pad_start && i < total_bytes; i += stride) {
            dst[i] = 0;
        }

        // Vectorized zero-fill
        uint4 zero4 = make_uint4(0, 0, 0, 0);
        long long vec_pad_end = (total_bytes / 16) * 16;
        uint4* dst4_pad = reinterpret_cast<uint4*>(dst);
        long long vec_start = aligned_pad_start / 16;
        long long vec_end = vec_pad_end / 16;

        for (long long i = vec_start + tid; i < vec_end; i += stride) {
            dst4_pad[i] = zero4;
        }

        // Zero remaining bytes at end
        for (long long i = vec_pad_end + tid; i < total_bytes; i += stride) {
            dst[i] = 0;
        }
    }
}


// =========================================================================
// Gather: padded per-expert BF16 → concatenated BF16
// =========================================================================
// Each threadblock handles one expert. Threads cooperatively copy
// n_tokens * row_elems BF16 values from the padded batched output
// to the concatenated result.
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
    int row_bytes                           // N * 2
) {
    int expert = blockIdx.x;
    int start = expert_offsets[expert];
    int end = expert_offsets[expert + 1];
    int n_tokens = end - start;

    if (n_tokens <= 0) return;

    // Source: padded slot for this expert
    const uint8_t* src = input + (long long)expert * max_M * row_bytes;

    // Destination: contiguous in concatenated buffer
    uint8_t* dst = output + (long long)start * row_bytes;

    long long data_bytes = (long long)n_tokens * row_bytes;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Vectorized uint4 copy
    long long vec_bytes = (data_bytes / 16) * 16;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    long long n_vec = vec_bytes / 16;

    for (long long i = tid; i < n_vec; i += stride) {
        dst4[i] = src4[i];
    }

    // Handle remaining bytes
    for (long long i = vec_bytes + tid; i < data_bytes; i += stride) {
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

    // One threadblock per expert, 256 threads
    dim3 grid(num_experts);
    dim3 block(256);

    kMoeScatterNVFP4<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        expert_offsets,
        max_M,
        row_bytes
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

    dim3 grid(num_experts);
    dim3 block(256);

    kMoeGatherBF16<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        expert_offsets,
        max_M,
        row_bytes
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
