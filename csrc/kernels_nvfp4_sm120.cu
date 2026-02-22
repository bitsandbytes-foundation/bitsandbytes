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
// Shared-memory NVFP4 GEMM kernel
//
// Key optimizations:
// 1. Cooperative tiling: All threads cooperatively load A/B/SFA/SFB tiles from
//    global memory into shared memory with coalesced access patterns.
// 2. Data reuse: A tile shared across N_WARPS (4x saving), B tile shared
//    across M_WARPS (2x saving). Total ~2.4x bandwidth reduction.
// 3. Fast register packing: MMA registers read from smem as uint32 loads.
//    SFA/SFB packed registers loaded as single uint32 (all 4 bytes are
//    consecutive in the same row — proven by CuTE layout analysis).
// 4. Vectorized global loads: A uses uint32 (4B), B uses uint4 (16B).
//
// Register layout (from CuTE ALayout/BLayout):
//   A reg[i] = A[tile_m + 2*t1 + (i&1), k_start + t0*8 + (i>>1)*32 .. +7]
//   B reg[i] = B[tile_n + t1,            k_start + t0*8 + i*32     .. +7]
//   SFA packed = SFA[actual_m, k_blk 0..3] (consecutive in memory)
//   SFB packed = SFB[tile_n + t1, k_blk 0..3] (consecutive in memory)
//
// Block tile: m32 x n128 (M_WARPS=2, N_WARPS=4, N_TILES_PER_WARP=4)
// Shared memory per K-step: 1024 + 4096 + 128 + 512 = 5760 bytes
// ============================================================================

// N-tiles per warp: each warp computes m16 x (N_TILES_PER_WARP * 8)
#define N_TILES_PER_WARP 4
// Block config: M_WARPS x N_WARPS warps per block
#define M_WARPS 2
#define N_WARPS 4
#define WARPS_PER_BLOCK (M_WARPS * N_WARPS) // 8

// Block tile dimensions
#define BLOCK_M_DIM (M_WARPS * 16)                   // 32
#define BLOCK_N_DIM (N_WARPS * N_TILES_PER_WARP * 8) // 128

// Shared memory sizes (bytes per K-step)
#define SMEM_A_BYTES (BLOCK_M_DIM * 32)   // 1024
#define SMEM_B_BYTES (BLOCK_N_DIM * 32)   // 4096
#define SMEM_SFA_BYTES (BLOCK_M_DIM * 4)  // 128
#define SMEM_SFB_BYTES (BLOCK_N_DIM * 4)  // 512
#define SMEM_TOTAL (SMEM_A_BYTES + SMEM_B_BYTES + SMEM_SFA_BYTES + SMEM_SFB_BYTES)

// 256 threads, target 4 blocks/SM for occupancy
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4) void kGemmNVFP4_smem(
    const unsigned char* __restrict__ A,   // M x K/2 packed FP4 (row-major)
    const unsigned char* __restrict__ B,   // N x K/2 packed FP4 (B transposed, row-major)
    const unsigned char* __restrict__ SFA, // M x K/16 UE4M3 scales
    const unsigned char* __restrict__ SFB, // N x K/16 UE4M3 scales
    float* __restrict__ D,                 // M x N output (F32)
    int M, int N, int K
) {
    // Shared memory: 16-byte aligned for uint4 stores
    __shared__ __align__(16) unsigned char smem[SMEM_TOTAL]; // 5760 bytes
    unsigned char* smem_A = smem;
    unsigned char* smem_B = smem + SMEM_A_BYTES;
    unsigned char* smem_SFA = smem + SMEM_A_BYTES + SMEM_B_BYTES;
    unsigned char* smem_SFB = smem + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_SFA_BYTES;

    const int tid = threadIdx.x;
    const int warp_in_block = tid / 32;
    const int lane_id = tid % 32;
    const int m_warp = warp_in_block / N_WARPS; // 0..1
    const int n_warp = warp_in_block % N_WARPS; // 0..3

    const int block_m = blockIdx.y * BLOCK_M_DIM;
    const int block_n = blockIdx.x * BLOCK_N_DIM;
    const int tile_m = block_m + m_warp * 16;
    const int warp_n_base = block_n + n_warp * N_TILES_PER_WARP * 8;

    const int t0 = lane_id % 4;
    const int t1 = lane_id / 4;
    const int half_K = K / 2;
    const int scale_K = K / 16;

    // Accumulators
    float acc[N_TILES_PER_WARP][4];
#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    // Precompute smem row indices for A register reads
    const int a_local_row0 = m_warp * 16 + 2 * t1;
    const int a_local_row1 = a_local_row0 + 1;

    // Precompute SFA row for this thread (all 4 bytes come from the same row)
    // sf_tidx = (lane%2)*8 + lane/4; cute_m_0 = sf_tidx % 16
    // actual_m = (cute_m_0 % 8)*2 + cute_m_0/8
    const int sf_tidx = (lane_id % 2) * 8 + (lane_id / 4);
    const int cute_sf_m0 = sf_tidx % 16;
    const int sfa_local_row = m_warp * 16 + (cute_sf_m0 % 8) * 2 + cute_sf_m0 / 8;

    // K-loop
    for (int k_start = 0; k_start < K; k_start += 64) {
        const int k_byte = k_start / 2;
        const int k_scale = k_start / 16;

        // ================================================================
        // Phase 1: Cooperative load from global → shared memory
        // ================================================================

        // ---- A tile: BLOCK_M×32 = 1024 bytes, 256 threads × 4 bytes each ----
        {
            const int off = tid * 4;       // byte offset in smem_A (0..1020)
            const int row = off >> 5;      // off / 32 → local row (0..31)
            const int col = off & 31;      // off % 32 → byte col (0,4,...,28)
            const int gm = block_m + row;

            uint32_t val = 0;
            if (gm < M) {
                const int gaddr = gm * half_K + k_byte + col;
                if (k_byte + col + 3 < half_K) {
                    val = *(const uint32_t*)(A + gaddr);
                } else {
                    // K-boundary: byte-by-byte
                    for (int b = 0; b < 4; b++) {
                        if (k_byte + col + b < half_K)
                            val |= ((uint32_t)A[gaddr + b]) << (b * 8);
                    }
                }
            }
            *(uint32_t*)(smem_A + off) = val;
        }

        // ---- B tile: BLOCK_N×32 = 4096 bytes, 256 threads × 16 bytes each ----
        {
            const int off = tid * 16;      // byte offset in smem_B (0..4080)
            const int row = off >> 5;      // local row (0..127)
            const int col = off & 31;      // 0 or 16
            const int gn = block_n + row;

            if (gn < N) {
                const int gaddr = gn * half_K + k_byte + col;
                if (k_byte + col + 15 < half_K) {
                    *(uint4*)(smem_B + off) = *(const uint4*)(B + gaddr);
                } else {
                    // K-boundary: byte-by-byte
                    for (int b = 0; b < 16; b++) {
                        smem_B[off + b] = (k_byte + col + b < half_K) ? B[gaddr + b] : 0;
                    }
                }
            } else {
                // N-boundary: zero-fill
                *(uint4*)(smem_B + off) = make_uint4(0, 0, 0, 0);
            }
        }

        // ---- SFA: BLOCK_M×4 = 128 bytes. First 32 threads load 4 bytes each ----
        if (tid < BLOCK_M_DIM) {
            const int gm = block_m + tid;
            uint32_t val = 0;
            if (gm < M) {
                const int base = gm * scale_K + k_scale;
                if (k_scale + 3 < scale_K) {
                    val = *(const uint32_t*)(SFA + base);
                } else {
                    for (int b = 0; b < 4; b++) {
                        if (k_scale + b < scale_K)
                            val |= ((uint32_t)SFA[base + b]) << (b * 8);
                    }
                }
            }
            *(uint32_t*)(smem_SFA + tid * 4) = val;
        }

        // ---- SFB: BLOCK_N×4 = 512 bytes. First 128 threads load 4 bytes each ----
        if (tid < BLOCK_N_DIM) {
            const int gn = block_n + tid;
            uint32_t val = 0;
            if (gn < N) {
                const int base = gn * scale_K + k_scale;
                if (k_scale + 3 < scale_K) {
                    val = *(const uint32_t*)(SFB + base);
                } else {
                    for (int b = 0; b < 4; b++) {
                        if (k_scale + b < scale_K)
                            val |= ((uint32_t)SFB[base + b]) << (b * 8);
                    }
                }
            }
            *(uint32_t*)(smem_SFB + tid * 4) = val;
        }

        __syncthreads();

        // ================================================================
        // Phase 2: Read MMA registers from smem and compute
        // ================================================================

        // A registers (shared across all N-tiles in this warp)
        uint32_t a_regs[4];
        a_regs[0] = *(const uint32_t*)(smem_A + a_local_row0 * 32 + t0 * 4);
        a_regs[1] = *(const uint32_t*)(smem_A + a_local_row1 * 32 + t0 * 4);
        a_regs[2] = *(const uint32_t*)(smem_A + a_local_row0 * 32 + t0 * 4 + 16);
        a_regs[3] = *(const uint32_t*)(smem_A + a_local_row1 * 32 + t0 * 4 + 16);

        // SFA: single uint32 load (all 4 bytes are in the same row)
        uint32_t sfa_packed = *(const uint32_t*)(smem_SFA + sfa_local_row * 4);

        // Per-N-tile: read B and SFB from smem, execute MMA
#pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int local_n = n_warp * N_TILES_PER_WARP * 8 + nt * 8;
            int b_row = local_n + t1;

            // B registers: 2 × uint32 from smem
            uint32_t b_regs[2];
            b_regs[0] = *(const uint32_t*)(smem_B + b_row * 32 + t0 * 4);
            b_regs[1] = *(const uint32_t*)(smem_B + b_row * 32 + t0 * 4 + 16);

            // SFB: single uint32 load (4 consecutive bytes at row t1)
            uint32_t sfb_packed = *(const uint32_t*)(smem_SFB + (local_n + t1) * 4);

            mma_nvfp4_m16n8k64(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                               b_regs[0], b_regs[1], acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], sfa_packed, sfb_packed);
        }

        __syncthreads(); // Barrier before next K-step's smem writes
    }

    // ---- Write output ----
    // SM80_16x8_Row: octet = lane/4, quad = lane%4
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
// Host-side launcher — uses shared memory kernel
// ============================================================================
extern "C" void cgemm_nvfp4(
    const unsigned char* A, const unsigned char* B, const unsigned char* SFA, const unsigned char* SFB, float* D, int M,
    int N, int K
) {
    int num_m_blocks = (M + BLOCK_M_DIM - 1) / BLOCK_M_DIM;
    int num_n_blocks = (N + BLOCK_N_DIM - 1) / BLOCK_N_DIM;

    dim3 grid(num_n_blocks, num_m_blocks);
    int threads_per_block = WARPS_PER_BLOCK * 32; // 256

    kGemmNVFP4_smem<<<grid, threads_per_block>>>(A, B, SFA, SFB, D, M, N, K);
}
