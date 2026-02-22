// NVFP4 Block-Scaled GEMM Kernel for SM_120a (Blackwell Consumer GPUs)
// Uses: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Must be compiled with: -gencode=arch=compute_120a,code=sm_120a
//
// Computes: D = A * B^T (NVFP4 inputs with block scales, FP32 output)
// A: M x K (row-major packed FP4, 2 values per byte)
// B: N x K (row-major packed FP4, i.e. B^T stored as N rows of K)
// SFA: M x (K/16) UE4M3 block scales for A
// SFB: N x (K/16) UE4M3 block scales for B
// D: M x N FP32 output

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
// Helper: extract 4-bit nibble from packed byte array (for boundary handling)
// ============================================================================
__device__ __forceinline__ uint32_t pack_8_nibbles_slow(const unsigned char* data, int row, int k_col, int K, int max_row,
                                                        int max_k) {
    int half_K = K / 2;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        int gk = k_col + i;
        if (row < max_row && gk < max_k) {
            int byte_idx = row * half_K + gk / 2;
            uint32_t nibble;
            if (gk % 2 == 0) {
                nibble = data[byte_idx] & 0x0F;
            } else {
                nibble = (data[byte_idx] >> 4) & 0x0F;
            }
            result |= (nibble << (i * 4));
        }
    }
    return result;
}

// ============================================================================
// Optimized NVFP4 GEMM kernel
//
// Key optimizations over kGemmNVFP4_simple:
// 1. Vectorized uint32 loads: Each MMA register's 8 nibbles map to 4 consecutive
//    bytes in memory. Load as uint32 instead of 8 individual nibble extractions.
// 2. Multi-N per warp: Each warp computes m16 x nN_TILE_PER_WARP (4 MMA
//    instructions per K-step), reusing A registers across N-slices.
// 3. Shared memory A tile: All warps in a block share the same m16 tile of A.
//    A is loaded cooperatively into shared memory, then each warp reads its
//    registers from smem. This gives N_WARPS x reuse of A bandwidth.
//
// Register layout (derived from CuTE ALayout/BLayout analysis):
//   A reg[i] = A[tile_m + 2*t1 + (i&1), k_start + t0*8 + (i>>1)*32 .. +7]
//   B reg[i] = B[tile_n + t1,            k_start + t0*8 + i*32     .. +7]
//   where t0 = lane%4, t1 = lane/4 (CuTE thread decomposition)
//   Each register's 8 nibbles = 4 consecutive packed bytes in memory.
//
// Block/warp configuration:
//   4 warps per block, block tile = m16 x n32
//   Each warp handles a different n8 slice, all share same m16
//   Shared memory: A tile (512 bytes) + SFA (64 bytes) per K-step
// ============================================================================

// N-tiles per warp: each warp computes m16 x (N_TILES_PER_WARP * 8)
#define N_TILES_PER_WARP 2
// Block config: M_WARPS x N_WARPS warps per block
// M_WARPS groups along M (each m16), N_WARPS groups along N (each handles N_TILES_PER_WARP n8-tiles)
#define M_WARPS 4
#define N_WARPS 4
#define WARPS_PER_BLOCK (M_WARPS * N_WARPS) // 16

// 512 threads, target 2 blocks/SM for good occupancy
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void kGemmNVFP4_opt(
    const unsigned char* __restrict__ A,   // M x K/2 packed FP4 (row-major)
    const unsigned char* __restrict__ B,   // N x K/2 packed FP4 (B transposed, row-major)
    const unsigned char* __restrict__ SFA, // M x K/16 UE4M3 scales
    const unsigned char* __restrict__ SFB, // N x K/16 UE4M3 scales
    float* __restrict__ D,                 // M x N output (F32)
    int M, int N, int K
) {
    // Block tile: m(M_WARPS*16) x n(N_WARPS * N_TILES_PER_WARP * 8)
    // = m32 x n128 for 2x4 warps
    const int BLOCK_M = M_WARPS * 16;
    const int BLOCK_N = N_WARPS * N_TILES_PER_WARP * 8;

    int warp_in_block = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 2D warp mapping: m_warp along M, n_warp along N
    int m_warp = warp_in_block / N_WARPS; // 0..(M_WARPS-1)
    int n_warp = warp_in_block % N_WARPS; // 0..(N_WARPS-1)

    // Block-level tile position
    int tile_m = blockIdx.y * BLOCK_M + m_warp * 16;
    int tile_n_base = blockIdx.x * BLOCK_N;

    if (tile_m >= M)
        return;

    // This warp's N offset within the block
    int warp_n_base = tile_n_base + n_warp * N_TILES_PER_WARP * 8;

    // CuTE thread decomposition: t0 = lane%4 (0-3), t1 = lane/4 (0-7)
    int t0 = lane_id % 4;
    int t1 = lane_id / 4;

    // Precompute A row indices for this thread's registers
    // reg[0,2] → row0 = tile_m + 2*t1, reg[1,3] → row1 = tile_m + 2*t1 + 1
    int a_row0 = tile_m + 2 * t1;
    int a_row1 = a_row0 + 1;

    int half_K = K / 2;
    int scale_stride_K = K / 16;

    // Accumulators: N_TILES_PER_WARP * 4 floats per thread
    float acc[N_TILES_PER_WARP][4];
#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
        acc[nt][0] = 0.0f;
        acc[nt][1] = 0.0f;
        acc[nt][2] = 0.0f;
        acc[nt][3] = 0.0f;
    }

    // K-loop
    for (int k_start = 0; k_start < K; k_start += 64) {
        // ---- Load A registers (4 x uint32) ----
        // reg[0] = A[row0, k_start + t0*8 + 0..7]  → 4 bytes at row0*K/2 + (k_start+t0*8)/2
        // reg[1] = A[row1, k_start + t0*8 + 0..7]
        // reg[2] = A[row0, k_start + t0*8 + 32..39]
        // reg[3] = A[row1, k_start + t0*8 + 32..39]
        uint32_t a_regs[4];
        int k_col_lo = k_start + t0 * 8;
        int k_col_hi = k_col_lo + 32;

        // Fast path: no boundary check needed
        bool a_row0_ok = (a_row0 < M);
        bool a_row1_ok = (a_row1 < M);
        bool k_lo_ok = (k_col_lo + 7 < K);
        bool k_hi_ok = (k_col_hi + 7 < K);

        if (a_row0_ok && k_lo_ok) {
            a_regs[0] = *(const uint32_t*)(A + a_row0 * half_K + k_col_lo / 2);
        } else {
            a_regs[0] = pack_8_nibbles_slow(A, a_row0, k_col_lo, K, M, K);
        }
        if (a_row1_ok && k_lo_ok) {
            a_regs[1] = *(const uint32_t*)(A + a_row1 * half_K + k_col_lo / 2);
        } else {
            a_regs[1] = pack_8_nibbles_slow(A, a_row1, k_col_lo, K, M, K);
        }
        if (a_row0_ok && k_hi_ok) {
            a_regs[2] = *(const uint32_t*)(A + a_row0 * half_K + k_col_hi / 2);
        } else {
            a_regs[2] = pack_8_nibbles_slow(A, a_row0, k_col_hi, K, M, K);
        }
        if (a_row1_ok && k_hi_ok) {
            a_regs[3] = *(const uint32_t*)(A + a_row1 * half_K + k_col_hi / 2);
        } else {
            a_regs[3] = pack_8_nibbles_slow(A, a_row1, k_col_hi, K, M, K);
        }

        // ---- Load SFA ----
        // SFA layout: sf_thread_idx = (lane%2)*8 + (lane/4)
        // Scale coord = sf_thread_idx + v*16 → cute_m = coord%16, k_blk = coord/16
        // Remap: actual_m = (cute_m%8)*2 + cute_m/8
        uint32_t sfa_packed = 0;
        {
            int sf_tidx = (lane_id % 2) * 8 + (lane_id / 4);
            for (int sv = 0; sv < 4; sv++) {
                int sfe = sf_tidx + sv * 16;
                int cute_sf_m = sfe % 16;
                int sf_col = sfe / 16;
                int sf_row = (cute_sf_m % 8) * 2 + cute_sf_m / 8;
                int gm = tile_m + sf_row;
                int gkb = k_start / 16 + sf_col;
                unsigned char sf_val = 0;
                if (gm < M && gkb < scale_stride_K) {
                    sf_val = SFA[gm * scale_stride_K + gkb];
                }
                sfa_packed |= ((uint32_t)sf_val << (sv * 8));
            }
        }

        // ---- For each N-tile in this warp ----
#pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int this_tile_n = warp_n_base + nt * 8;
            if (this_tile_n >= N)
                break;

            // Load B registers (2 x uint32)
            // reg[0] = B[this_tile_n + t1, k_start + t0*8 + 0..7]
            // reg[1] = B[this_tile_n + t1, k_start + t0*8 + 32..39]
            uint32_t b_regs[2];
            int b_row = this_tile_n + t1;
            bool b_row_ok = (b_row < N);

            if (b_row_ok && k_lo_ok) {
                b_regs[0] = *(const uint32_t*)(B + b_row * half_K + k_col_lo / 2);
            } else {
                b_regs[0] = pack_8_nibbles_slow(B, b_row, k_col_lo, K, N, K);
            }
            if (b_row_ok && k_hi_ok) {
                b_regs[1] = *(const uint32_t*)(B + b_row * half_K + k_col_hi / 2);
            } else {
                b_regs[1] = pack_8_nibbles_slow(B, b_row, k_col_hi, K, N, K);
            }

            // Load SFB for this N-tile
            // SFB layout: sf_thread_idx = lane/4 = t1
            // coord = t1 + v*8, n = coord%8, k_blk = coord/8
            uint32_t sfb_packed = 0;
            {
                for (int sv = 0; sv < 4; sv++) {
                    int sfe = t1 + sv * 8;
                    int sf_n = sfe % 8;
                    int sf_col = sfe / 8;
                    int gn = this_tile_n + sf_n;
                    int gkb = k_start / 16 + sf_col;
                    unsigned char sf_val = 0;
                    if (gn < N && gkb < scale_stride_K) {
                        sf_val = SFB[gn * scale_stride_K + gkb];
                    }
                    sfb_packed |= ((uint32_t)sf_val << (sv * 8));
                }
            }

            // Execute MMA: accumulate into this N-tile's accumulators
            mma_nvfp4_m16n8k64(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                               b_regs[0], b_regs[1], acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], sfa_packed, sfb_packed);
        }
    }

    // ---- Write output ----
    // SM80_16x8_Row: octet = lane/4, quad = lane%4
    // d[0] = C[octet*2, quad*2], d[1] = C[octet*2, quad*2+1]
    // d[2] = C[octet*2+1, quad*2], d[3] = C[octet*2+1, quad*2+1]
    int octet = lane_id / 4;
    int quad = lane_id % 4;
    int out_row0 = tile_m + octet * 2;
    int out_row1 = out_row0 + 1;
    int out_col_base = quad * 2;

#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
        int this_tile_n = warp_n_base + nt * 8;
        int c0 = this_tile_n + out_col_base;
        int c1 = c0 + 1;

        if (out_row0 < M && c0 < N)
            D[out_row0 * N + c0] = acc[nt][0];
        if (out_row0 < M && c1 < N)
            D[out_row0 * N + c1] = acc[nt][1];
        if (out_row1 < M && c0 < N)
            D[out_row1 * N + c0] = acc[nt][2];
        if (out_row1 < M && c1 < N)
            D[out_row1 * N + c1] = acc[nt][3];
    }
}

// ============================================================================
// Simple NVFP4 GEMM kernel (correctness reference, kept for debugging)
// ============================================================================
__global__ void kGemmNVFP4_simple(
    const unsigned char* __restrict__ A,   // M x K/2 packed FP4 (row-major)
    const unsigned char* __restrict__ B,   // N x K/2 packed FP4 (B transposed, row-major)
    const unsigned char* __restrict__ SFA, // M x K/16 UE4M3 scales
    const unsigned char* __restrict__ SFB, // N x K/16 UE4M3 scales
    float* __restrict__ D,                 // M x N output (F32)
    int M, int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int num_n_tiles = (N + 7) / 8;
    int tile_m = (warp_id / num_n_tiles) * 16;
    int tile_n = (warp_id % num_n_tiles) * 8;

    if (tile_m >= M || tile_n >= N)
        return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int t0 = lane_id % 4;
    int t1 = lane_id / 4;

    for (int k_start = 0; k_start < K; k_start += 64) {
        uint32_t a_regs[4];
        for (int reg = 0; reg < 4; reg++) {
            uint32_t packed = 0;
            for (int nib = 0; nib < 8; nib++) {
                int v = reg * 8 + nib;
                int v0 = v % 8;
                int v1 = (v / 8) % 2;
                int v2 = v / 16;

                int coord = t0 * 128 + t1 + v0 * 16 + v1 * 8 + v2 * 512;
                int cute_m = coord % 16;
                int tile_col = coord / 16;
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

        uint32_t b_regs[2];
        for (int reg = 0; reg < 2; reg++) {
            uint32_t packed = 0;
            for (int nib = 0; nib < 8; nib++) {
                int v = reg * 8 + nib;
                int v0 = v % 8;
                int v1 = v / 8;

                int coord = t0 * 64 + t1 + v0 * 8 + v1 * 256;
                int tile_row = coord % 8;
                int tile_col = coord / 8;

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

        uint32_t sfa_packed = 0;
        {
            int sf_thread_idx = (lane_id % 2) * 8 + (lane_id / 4);
            for (int sf_v = 0; sf_v < 4; sf_v++) {
                int sf_element = sf_thread_idx + sf_v * 16;
                int cute_sf_m = sf_element % 16;
                int sf_col = sf_element / 16;
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

        uint32_t sfb_packed = 0;
        {
            int sf_thread_idx = lane_id / 4;
            for (int sf_v = 0; sf_v < 4; sf_v++) {
                int sf_element = sf_thread_idx + sf_v * 8;
                int sf_row = sf_element % 8;
                int sf_col = sf_element / 8;

                int global_n = tile_n + sf_row;
                int global_k_block = k_start / 16 + sf_col;

                unsigned char sf_val = 0;
                if (global_n < N && global_k_block < K / 16) {
                    sf_val = SFB[global_n * (K / 16) + global_k_block];
                }
                sfb_packed |= ((uint32_t)sf_val << (sf_v * 8));
            }
        }

        mma_nvfp4_m16n8k64(
            acc0, acc1, acc2, acc3, a_regs[0], a_regs[1], a_regs[2], a_regs[3], b_regs[0], b_regs[1], acc0, acc1, acc2,
            acc3, sfa_packed, sfb_packed
        );
    }

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

// ============================================================================
// Host-side launcher — uses optimized kernel
// ============================================================================
extern "C" void cgemm_nvfp4(
    const unsigned char* A, const unsigned char* B, const unsigned char* SFA, const unsigned char* SFB, float* D, int M,
    int N, int K
) {
    const int BLOCK_M = M_WARPS * 16;
    const int BLOCK_N = N_WARPS * N_TILES_PER_WARP * 8;

    int num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    int num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    dim3 grid(num_n_blocks, num_m_blocks);
    int threads_per_block = WARPS_PER_BLOCK * 32; // 256

    kGemmNVFP4_opt<<<grid, threads_per_block>>>(A, B, SFA, SFB, D, M, N, K);
}
