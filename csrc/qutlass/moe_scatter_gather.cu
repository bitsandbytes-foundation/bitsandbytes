/*
 * Scatter and gather kernels for MoE batched NVFP4 GEMM pipeline.
 *
 * Scatter: copies packed FP4 data from concatenated token layout to
 *          padded per-expert batched layout. Zero-fills padding rows.
 *
 * Gather: copies BF16 results from padded per-expert batched layout
 *         back to concatenated token layout.
 *
 * Both kernels use one threadblock per expert with vectorized 128-bit
 * (uint4) loads/stores for bandwidth efficiency.
 */

#include <cuda_runtime.h>
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
