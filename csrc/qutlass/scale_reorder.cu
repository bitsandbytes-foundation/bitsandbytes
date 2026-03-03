/*
 * Scale factor reordering for CUTLASS block-scaled GEMM.
 *
 * Converts flat row-major block scale factors into the swizzled layout
 * expected by CUTLASS's Sm1xx block-scaled MMA operations.
 *
 * Reference: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
 *
 * The swizzle pattern within a 128×4 block of scale factors maps
 * (row, col) → (row % 32, (row / 32) * 4 + col) in a 32×16 output block.
 * This pattern is hardware-defined and independent of GEMM tile configuration.
 */

#include <cstdint>
#include <cuda_runtime.h>

// =========================================================================
// to_blocked: row-major scales → CUTLASS block-scaled layout
// =========================================================================
// Input:  flat row-major scale tensor of shape (H, W) where H and W are
//         padded to multiples of 128 and 4 respectively.
// Output: swizzled flat buffer of size (ceil(H/128) * ceil(W/4) * 128 * 4)
//         in CUTLASS block-scaled format.
//
// Each thread block handles one 128×4 block of the input.
__global__ void kScaleToBlocked(
    const uint8_t* __restrict__ input, // (H, W) row-major
    uint8_t* __restrict__ output,      // flat swizzled output
    int H, int W
) // scale tensor dimensions
{
    // Block indices
    int block_row = blockIdx.x; // which 128-row block
    int block_col = blockIdx.y; // which 4-col block

    int n_col_blocks = (W + 3) / 4;

    // Thread computes one element within the 128×4 block
    int local_idx = threadIdx.x; // 0..511 (128 * 4 = 512 threads)
    int r = local_idx / 4;       // row within block [0..127]
    int c = local_idx % 4;       // col within block [0..3]

    int global_r = block_row * 128 + r;
    int global_c = block_col * 4 + c;

    // Load input (zero if out of bounds)
    uint8_t val = 0;
    if (global_r < H && global_c < W) {
        val = input[global_r * W + global_c];
    }

    // Swizzle: (r, c) → position in 32×16 output block
    int r_mod_32 = r % 32;
    int r_div_32 = r / 32;
    int dest_in_block = r_mod_32 * 16 + r_div_32 * 4 + c;

    // Output block offset: blocks are stored sequentially
    // Block order: iterate col blocks first, then row blocks
    int block_idx = block_row * n_col_blocks + block_col;
    int block_size = 128 * 4; // 512 elements per block
    int output_idx = block_idx * block_size + dest_in_block;

    output[output_idx] = val;
}

// =========================================================================
// from_blocked: CUTLASS block-scaled layout → row-major scales
// =========================================================================
// Inverse of to_blocked. Used by dequantize to read swizzled scales.
__global__ void kScaleFromBlocked(
    const uint8_t* __restrict__ input, // flat swizzled input
    uint8_t* __restrict__ output,      // (H, W) row-major output
    int H, int W
) // scale tensor dimensions
{
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    int n_col_blocks = (W + 3) / 4;

    int local_idx = threadIdx.x;
    int r = local_idx / 4;
    int c = local_idx % 4;

    int global_r = block_row * 128 + r;
    int global_c = block_col * 4 + c;

    // Compute swizzled index (same as to_blocked)
    int r_mod_32 = r % 32;
    int r_div_32 = r / 32;
    int dest_in_block = r_mod_32 * 16 + r_div_32 * 4 + c;

    int block_idx = block_row * n_col_blocks + block_col;
    int block_size = 128 * 4;
    int input_idx = block_idx * block_size + dest_in_block;

    // Read from swizzled, write to row-major
    uint8_t val = input[input_idx];

    if (global_r < H && global_c < W) {
        output[global_r * W + global_c] = val;
    }
}

// =========================================================================
// extern "C" launchers
// =========================================================================

extern "C" void cscale_to_blocked(
    const void* input, // (H, W) row-major uint8 scales
    void* output,      // flat swizzled output
    int H, int W,      // scale tensor dimensions
    cudaStream_t stream
) {
    int n_row_blocks = (H + 127) / 128;
    int n_col_blocks = (W + 3) / 4;

    dim3 grid(n_row_blocks, n_col_blocks);
    dim3 block(512); // 128 * 4 threads per block

    kScaleToBlocked<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output), H, W
    );
}

extern "C" void cscale_from_blocked(
    const void* input, // flat swizzled input
    void* output,      // (H, W) row-major uint8 output
    int H, int W,      // scale tensor dimensions
    cudaStream_t stream
) {
    int n_row_blocks = (H + 127) / 128;
    int n_col_blocks = (W + 3) / 4;

    dim3 grid(n_row_blocks, n_col_blocks);
    dim3 block(512);

    kScaleFromBlocked<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output), H, W
    );
}
