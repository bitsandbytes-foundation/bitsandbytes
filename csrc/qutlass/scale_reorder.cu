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
// Batched per-expert to_blocked: row-major scales → per-expert swizzled
// =========================================================================
// For grouped/MoE GEMM: takes a concatenated row-major scale tensor
// and expert offsets, produces independently-swizzled per-expert outputs
// in a single kernel launch.
//
// Grid: (max_row_blocks, n_col_blocks, num_experts)
// Each block handles one 128×4 tile for one expert.
// Output: contiguous buffer with per-expert swizzled blocks at precomputed offsets.
__global__ void kScaleToBlockedBatched(
    const uint8_t* __restrict__ input,          // (total_rows, W) row-major
    uint8_t* __restrict__ output,               // contiguous output for all experts
    const int* __restrict__ expert_row_offsets,  // [num_experts] start row per expert
    const int* __restrict__ expert_M,            // [num_experts] rows per expert
    const int* __restrict__ expert_out_offsets,  // [num_experts] byte offset in output per expert
    int W,                                       // scale columns (K/16)
    int num_experts
) {
    int expert = blockIdx.z;
    if (expert >= num_experts) return;

    int M_e = expert_M[expert];
    if (M_e <= 0) return;

    int block_row = blockIdx.x;  // which 128-row block within this expert
    int block_col = blockIdx.y;  // which 4-col block

    int n_row_blocks_e = (M_e + 127) / 128;
    if (block_row >= n_row_blocks_e) return;

    int n_col_blocks = (W + 3) / 4;

    // Thread within the 128×4 block
    int local_idx = threadIdx.x;  // 0..511
    int r = local_idx / 4;        // row within block [0..127]
    int c = local_idx % 4;        // col within block [0..3]

    // Global coordinates in the concatenated input
    int row_offset = expert_row_offsets[expert];
    int global_r = row_offset + block_row * 128 + r;
    int global_c = block_col * 4 + c;

    // Local row within expert (for bounds checking)
    int local_r = block_row * 128 + r;

    // Load input (zero if out of bounds)
    uint8_t val = 0;
    if (local_r < M_e && global_c < W) {
        val = input[global_r * W + global_c];
    }

    // Swizzle: same pattern as kScaleToBlocked
    int r_mod_32 = r % 32;
    int r_div_32 = r / 32;
    int dest_in_block = r_mod_32 * 16 + r_div_32 * 4 + c;

    // Output offset: expert's base + block index within expert
    int block_idx = block_row * n_col_blocks + block_col;
    int block_size = 128 * 4;  // 512 bytes per block
    int out_base = expert_out_offsets[expert];
    int output_idx = out_base + block_idx * block_size + dest_in_block;

    output[output_idx] = val;
}


// =========================================================================
// extern "C" launchers
// =========================================================================

extern "C" void cscale_to_blocked_batched(
    const void* input,                   // (total_rows, W) row-major uint8 scales
    void* output,                        // contiguous output buffer for all experts
    const int* expert_row_offsets,        // [num_experts] start row per expert (device)
    const int* expert_M,                 // [num_experts] rows per expert (device)
    const int* expert_out_offsets,        // [num_experts] byte offset in output (device)
    int W,                               // scale columns
    int num_experts,
    int max_row_blocks,                  // max ceil(M_e/128) across all experts
    cudaStream_t stream
) {
    int n_col_blocks = (W + 3) / 4;

    dim3 grid(max_row_blocks, n_col_blocks, num_experts);
    dim3 block(512);

    kScaleToBlockedBatched<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        expert_row_offsets,
        expert_M,
        expert_out_offsets,
        W,
        num_experts
    );
}


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
