// NVFP4 Block-Scaled GEMM Kernel for SM_120a (Blackwell Consumer GPUs)
// Uses: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Must be compiled with: -gencode=arch=compute_120a,code=sm_120a
//
// Computes: D = A * B (NVFP4 inputs with block scales, BF16 output)
// A: M x K (row-major packed FP4, 2 values per byte)
// B: K x N (column-major packed FP4, 2 values per byte)
// SFA: M x (K/16) UE4M3 block scales for A
// SFB: N x (K/16) UE4M3 block scales for B
// D: M x N BF16 output (first version: BF16 output, not NVFP4)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// MMA wrapper: m16n8k64 E2M1 x E2M1 -> F32 with UE4M3 block scales
// ============================================================================
__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float c0, float c1, float c2, float c3, uint32_t sfa, uint32_t sfb
) {
    uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
                 ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13},"
                 "{%14},"
                 "{%15, %16},"
                 "{%17},"
                 "{%18, %19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa),
                   "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
}

// ============================================================================
// Simple NVFP4 GEMM kernel (correctness-first, not performance-optimized)
//
// This kernel is designed for correctness verification first.
// Each warp computes one m16n8 output tile, iterating over K.
//
// Layout assumptions:
//   A: M x K, row-major, packed FP4 (2 per byte). Byte [i * K/2 + k/2]
//   B: N x K, "column-major" meaning B is stored as N rows of K (B^T in memory).
//      Packed FP4. Byte [j * K/2 + k/2]. This matches TN layout for MMA.
//   SFA: M x (K/16), row-major UE4M3. Byte [i * (K/16) + k/16]
//   SFB: N x (K/16), row-major UE4M3. Byte [j * (K/16) + k/16]
//
// MMA register mapping (SM80_16x8_Row for C/D):
//   Thread tid (0-31), octet = tid/4, quad = tid%4
//   d[0] = C[octet*2,   quad*2]
//   d[1] = C[octet*2,   quad*2+1]
//   d[2] = C[octet*2+1, quad*2]
//   d[3] = C[octet*2+1, quad*2+1]
//
// A register mapping (from CUTLASS ALayout for m16n8k64):
//   Thread tid, 4 regs of 8 nibbles each = 32 values per thread
//   The layout is complex; we use ldmatrix or manual packing.
//
// For this first version, we use a SIMPLER approach:
//   - Load A and B tiles into shared memory
//   - Use ldmatrix.x4 to load from shared memory to registers
//   - This avoids needing to understand the exact register layout
//
// Actually, ldmatrix doesn't support FP4. So we need to understand the
// register layout and pack data manually.
//
// MMA A register layout for m16n8k64 (from CUTLASS):
//   ALayout = Layout<Shape<Shape<_4,_8>, Shape<_8,_2,_2>>,
//                    Stride<Stride<_128,_1>, Stride<_16,_8,_512>>>
//   This maps (T32, V32) -> element index in M16xK64 tile (row-major)
//
//   For thread t, value v:
//     t0 = t/8, t1 = t%8   (thread decomposition)
//     v0 = v%8, v1 = (v/8)%2, v2 = v/16   (value decomposition)
//     element_idx = t0*128 + t1*1 + v0*16 + v1*8 + v2*512
//     row = element_idx / 64 (M dimension)
//     col = element_idx % 64 (K dimension)
//
//   Since values are packed 8 per uint32 register:
//     reg[0] = values v=0..7, reg[1] = v=8..15, reg[2] = v=16..23, reg[3] = v=24..31
//
// MMA B register layout for m16n8k64 (from CUTLASS):
//   BLayout = Layout<Shape<Shape<_4,_8>, Shape<_8,_2>>,
//                    Stride<Stride<_64,_1>, Stride<_8,_256>>>
//   For thread t, value v:
//     t0 = t/8, t1 = t%8
//     v0 = v%8, v1 = v/8
//     element_idx = t0*64 + t1*1 + v0*8 + v1*256
//     row = element_idx / 64 (N dimension)
//     col = element_idx % 64 (K dimension)
//
// SFA register layout:
//   SFALayout = Layout<Shape<Shape<_2,_2,_8>,_64>,
//                      Stride<Stride<_8,_0,_1>,_16>>
//   (T32,V64) -> (M16, K64) scale factor index
//   The _0 stride means dimension 1 is broadcast
//   For thread t: t0 = t/16, t1 = (t/8)%2, t2 = t%8
//   Scale idx = t0*8 + t2*1 + v*16 where v=0..3 (4 SFs per row)
//   But with _0 stride: pairs of threads read same scales
//
// For this first implementation, we pack A/B/SF registers in the host
// launcher and pass them via shared memory with the correct layout.
// ============================================================================

// Helper: extract 4-bit nibble from packed byte array
__device__ __forceinline__ uint32_t pack_8_nibbles(const unsigned char* data, int start_idx) {
    // Pack 8 consecutive 4-bit values from data starting at element index start_idx
    // data is packed 2 per byte (low nibble = even index, high nibble = odd index)
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        int elem_idx = start_idx + i;
        int byte_idx = elem_idx / 2;
        uint32_t nibble;
        if (elem_idx % 2 == 0) {
            nibble = data[byte_idx] & 0x0F;
        } else {
            nibble = (data[byte_idx] >> 4) & 0x0F;
        }
        result |= (nibble << (i * 4));
    }
    return result;
}

// Simple GEMM kernel: one warp per m16n8 output tile
// Each warp iterates over K in steps of 64
__global__ void kGemmNVFP4_simple(
    const unsigned char* __restrict__ A,   // M x K/2 packed FP4 (row-major)
    const unsigned char* __restrict__ B,   // N x K/2 packed FP4 (B transposed, row-major)
    const unsigned char* __restrict__ SFA, // M x K/16 UE4M3 scales
    const unsigned char* __restrict__ SFB, // N x K/16 UE4M3 scales
    float* __restrict__ D,                 // M x N output (F32)
    int M, int N, int K
) {
    // Warp-level tiling: each warp computes one m16n8 output tile
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    // Map warp to output tile
    int num_n_tiles = (N + 7) / 8;
    int tile_m = (warp_id / num_n_tiles) * 16;
    int tile_n = (warp_id % num_n_tiles) * 8;

    if (tile_m >= M || tile_n >= N)
        return;

    // Accumulator registers
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // CuTE thread decomposition: Shape<_4,_8> means first mode is fastest
    // T = t0 + t1*4, so t0 = T%4 (0-3), t1 = T/4 (0-7)
    int t0 = lane_id % 4; // 0-3
    int t1 = lane_id / 4; // 0-7

    // Iterate over K dimension in steps of 64
    for (int k_start = 0; k_start < K; k_start += 64) {
        // Load A registers: 4 x uint32 (32 E2M1 values per thread)
        // ALayout: coord = t0*128 + t1 + v0*16 + v1*8 + v2*512
        // CuTE coord space is column-major in tile: m = coord%16, k = coord/16
        // Value decomposition: v = v0 + v1*8 + v2*16 (v0=0..7, v1=0..1, v2=0..1)
        //
        // CRITICAL: CuTE column-major M-index interleaves rows [0,8], [1,9], ...
        // but the SM80_16x8 output layout expects consecutive row pairs [0,1], [2,3], ...
        // We remap: actual_m = (cute_m % 8) * 2 + cute_m / 8
        // so CuTE m=0 → actual 0, m=8 → actual 1, m=1 → actual 2, m=9 → actual 3, etc.
        uint32_t a_regs[4];
        for (int reg = 0; reg < 4; reg++) {
            uint32_t packed = 0;
            for (int nib = 0; nib < 8; nib++) {
                int v = reg * 8 + nib;
                int v0 = v % 8;
                int v1 = (v / 8) % 2;
                int v2 = v / 16;

                int coord = t0 * 128 + t1 + v0 * 16 + v1 * 8 + v2 * 512;
                int cute_m = coord % 16;   // CuTE M index (interleaved)
                int tile_col = coord / 16; // K index within tile
                // Remap from CuTE interleaved to sequential row order
                int tile_row = (cute_m % 8) * 2 + cute_m / 8;

                int global_m = tile_m + tile_row;
                int global_k = k_start + tile_col;

                uint32_t nibble = 0;
                if (global_m < M && global_k < K) {
                    int byte_idx = global_m * (K / 2) + global_k / 2;
                    if (global_k % 2 == 0) {
                        nibble = A[byte_idx] & 0x0F;
                    } else {
                        nibble = (A[byte_idx] >> 4) & 0x0F;
                    }
                }
                packed |= (nibble << (nib * 4));
            }
            a_regs[reg] = packed;
        }

        // Load B registers: 2 x uint32 (16 E2M1 values per thread)
        // BLayout: coord = t0*64 + t1 + v0*8 + v1*256
        // CuTE coord space is column-major: n = coord%8, k = coord/8
        uint32_t b_regs[2];
        for (int reg = 0; reg < 2; reg++) {
            uint32_t packed = 0;
            for (int nib = 0; nib < 8; nib++) {
                int v = reg * 8 + nib;
                int v0 = v % 8;
                int v1 = v / 8;

                int coord = t0 * 64 + t1 + v0 * 8 + v1 * 256;
                int tile_row = coord % 8; // N index within tile (column-major)
                int tile_col = coord / 8; // K index within tile

                int global_n = tile_n + tile_row;
                int global_k = k_start + tile_col;

                uint32_t nibble = 0;
                if (global_n < N && global_k < K) {
                    int byte_idx = global_n * (K / 2) + global_k / 2;
                    if (global_k % 2 == 0) {
                        nibble = B[byte_idx] & 0x0F;
                    } else {
                        nibble = (B[byte_idx] >> 4) & 0x0F;
                    }
                }
                packed |= (nibble << (nib * 4));
            }
            b_regs[reg] = packed;
        }

        // Load SFA: 1 x uint32 (4 packed UE4M3 bytes)
        // SFALayout: Shape<Shape<_2,_2,_8>,_64>, Stride<Stride<_8,_0,_1>,_16>
        // CuTE: T = t0 + t1*2 + t2*4, so t0=T%2, t1=(T/2)%2, t2=T/4
        // Strides: (8, 0, 1). t1 has stride 0 (broadcast).
        // sf_thread_contrib = t0*8 + t2 = (lane%2)*8 + (lane/4)
        // SF coord = sf_thread_contrib + v*16 (column-major: m=coord%16, k_blk=coord/16)
        uint32_t sfa_packed = 0;
        {
            int sf_thread_idx = (lane_id % 2) * 8 + (lane_id / 4);
            for (int sf_v = 0; sf_v < 4; sf_v++) {
                int sf_element = sf_thread_idx + sf_v * 16;
                int cute_sf_m = sf_element % 16; // CuTE M index (interleaved)
                int sf_col = sf_element / 16;    // K/16 index in tile
                // Same remapping as A data: CuTE interleaved → sequential
                int sf_row = (cute_sf_m % 8) * 2 + cute_sf_m / 8;

                int global_m = tile_m + sf_row;
                int global_k_block = k_start / 16 + sf_col;

                unsigned char sf_val = 0;
                if (global_m < M && global_k_block < K / 16) {
                    sf_val = SFA[global_m * (K / 16) + global_k_block];
                }
                sfa_packed |= ((uint32_t)sf_val << (sf_v * 8));
            }
        }

        // Load SFB: 1 x uint32 (4 packed UE4M3 bytes)
        // SFBLayout: Shape<Shape<_4,_8>,_64>, Stride<Stride<_0,_1>,_8>
        // CuTE: T = t0 + t1*4, so t0=T%4 (stride=0, broadcast), t1=T/4
        // sf_thread_contrib = t1 = lane/4
        // SF coord = sf_thread_contrib + v*8 (column-major: n=coord%8, k_blk=coord/8)
        uint32_t sfb_packed = 0;
        {
            int sf_thread_idx = lane_id / 4;
            for (int sf_v = 0; sf_v < 4; sf_v++) {
                int sf_element = sf_thread_idx + sf_v * 8;
                int sf_row = sf_element % 8; // N index in tile
                int sf_col = sf_element / 8; // K/16 index in tile

                int global_n = tile_n + sf_row;
                int global_k_block = k_start / 16 + sf_col;

                unsigned char sf_val = 0;
                if (global_n < N && global_k_block < K / 16) {
                    sf_val = SFB[global_n * (K / 16) + global_k_block];
                }
                sfb_packed |= ((uint32_t)sf_val << (sf_v * 8));
            }
        }

        // Execute MMA
        mma_nvfp4_m16n8k64(
            acc0, acc1, acc2, acc3, a_regs[0], a_regs[1], a_regs[2], a_regs[3], b_regs[0], b_regs[1], acc0, acc1, acc2,
            acc3, sfa_packed, sfb_packed
        );
    }

    // Write output using SM80_16x8_Row layout
    // Thread tid, octet = tid/4, quad = tid%4
    // d[0] = C[octet*2,   quad*2]
    // d[1] = C[octet*2,   quad*2+1]
    // d[2] = C[octet*2+1, quad*2]
    // d[3] = C[octet*2+1, quad*2+1]
    int octet = lane_id / 4;
    int quad = lane_id % 4;

    int out_row0 = tile_m + octet * 2;
    int out_row1 = tile_m + octet * 2 + 1;
    int out_col0 = tile_n + quad * 2;
    int out_col1 = tile_n + quad * 2 + 1;

    if (out_row0 < M && out_col0 < N)
        D[out_row0 * N + out_col0] = acc0;
    if (out_row0 < M && out_col1 < N)
        D[out_row0 * N + out_col1] = acc1;
    if (out_row1 < M && out_col0 < N)
        D[out_row1 * N + out_col0] = acc2;
    if (out_row1 < M && out_col1 < N)
        D[out_row1 * N + out_col1] = acc3;
}

// Host-side launcher
extern "C" void cgemm_nvfp4(
    const unsigned char* A, const unsigned char* B, const unsigned char* SFA, const unsigned char* SFB, float* D, int M,
    int N, int K
) {
    // Each warp handles one m16n8 output tile
    int num_m_tiles = (M + 15) / 16;
    int num_n_tiles = (N + 7) / 8;
    int total_warps = num_m_tiles * num_n_tiles;

    // 4 warps per block (128 threads)
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * 32;
    int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    kGemmNVFP4_simple<<<num_blocks, threads_per_block>>>(A, B, SFA, SFB, D, M, N, K);
}
