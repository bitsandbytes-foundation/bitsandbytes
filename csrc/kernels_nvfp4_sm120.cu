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
// Double-buffered shared-memory NVFP4 GEMM kernel with cp.async
//
// Key optimizations:
// 1. Cooperative tiling with coalesced global→smem loads
// 2. Data reuse: A shared across N_WARPS (4x), B shared across M_WARPS (2x)
// 3. Double buffering: overlaps global→smem loads for K-step k+1 with
//    MMA compute for K-step k. Uses register-based pipelining: global loads
//    go to registers first, compute runs, then registers write to smem.
// 4. Fast register packing from smem (uint32 loads, no nibble loops)
// 5. SFA/SFB packed as single uint32 loads (proved consecutive by CuTE analysis)
//
// Block tile: m32 x n128 (M_WARPS=2, N_WARPS=4, N_TILES_PER_WARP=4)
// Shared memory: 2 × 5760 = 11520 bytes (double-buffered)
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

// 256 threads, target 3 blocks/SM (need ~80 regs for double-buffer registers)
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 3) void kGemmNVFP4_smem(
    const unsigned char* __restrict__ A,   // M x K/2 packed FP4 (row-major)
    const unsigned char* __restrict__ B,   // N x K/2 packed FP4 (B transposed, row-major)
    const unsigned char* __restrict__ SFA, // M x K/16 UE4M3 scales
    const unsigned char* __restrict__ SFB, // N x K/16 UE4M3 scales
    float* __restrict__ D,                 // M x N output (F32)
    int M, int N, int K
) {
    // Double-buffered shared memory
    __shared__ __align__(16) unsigned char smem[2 * SMEM_TOTAL]; // 11520 bytes

    const int tid = threadIdx.x;
    const int warp_in_block = tid / 32;
    const int lane_id = tid % 32;
    const int m_warp = warp_in_block / N_WARPS;
    const int n_warp = warp_in_block % N_WARPS;

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

    // Precompute smem read indices
    const int a_local_row0 = m_warp * 16 + 2 * t1;
    const int a_local_row1 = a_local_row0 + 1;
    const int sf_tidx = (lane_id % 2) * 8 + (lane_id / 4);
    const int cute_sf_m0 = sf_tidx % 16;
    const int sfa_local_row = m_warp * 16 + (cute_sf_m0 % 8) * 2 + cute_sf_m0 / 8;

    // Precompute cooperative load addresses (per-thread, reused each K-step)
    const int a_off = tid * 4;
    const int a_load_row = a_off >> 5;
    const int a_load_col = a_off & 31;
    const int a_gm = block_m + a_load_row;

    const int b_off = tid * 16;
    const int b_load_row = b_off >> 5;
    const int b_load_col = b_off & 31;
    const int b_gn = block_n + b_load_row;

    // ================================================================
    // Macro for cooperative load into a buffer
    // ================================================================
#define COOP_LOAD(BUF, K_BYTE, K_SCALE)                                                                                   \
    do {                                                                                                                   \
        unsigned char* _sA = (BUF);                                                                                        \
        unsigned char* _sB = (BUF) + SMEM_A_BYTES;                                                                         \
        unsigned char* _sSFA = (BUF) + SMEM_A_BYTES + SMEM_B_BYTES;                                                        \
        unsigned char* _sSFB = (BUF) + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_SFA_BYTES;                                       \
        /* A tile */                                                                                                       \
        {                                                                                                                  \
            uint32_t _av = 0;                                                                                              \
            if (a_gm < M) {                                                                                                \
                int _ga = a_gm * half_K + (K_BYTE) + a_load_col;                                                           \
                if ((K_BYTE) + a_load_col + 3 < half_K)                                                                    \
                    _av = *(const uint32_t*)(A + _ga);                                                                     \
                else                                                                                                       \
                    for (int _b = 0; _b < 4; _b++)                                                                         \
                        if ((K_BYTE) + a_load_col + _b < half_K)                                                           \
                            _av |= ((uint32_t)A[_ga + _b]) << (_b * 8);                                                   \
            }                                                                                                              \
            *(uint32_t*)(_sA + a_off) = _av;                                                                               \
        }                                                                                                                  \
        /* B tile */                                                                                                       \
        {                                                                                                                  \
            if (b_gn < N) {                                                                                                \
                int _gb = b_gn * half_K + (K_BYTE) + b_load_col;                                                           \
                if ((K_BYTE) + b_load_col + 15 < half_K)                                                                   \
                    *(uint4*)(_sB + b_off) = *(const uint4*)(B + _gb);                                                     \
                else                                                                                                       \
                    for (int _b = 0; _b < 16; _b++)                                                                        \
                        _sB[b_off + _b] = ((K_BYTE) + b_load_col + _b < half_K) ? B[_gb + _b] : 0;                        \
            } else                                                                                                         \
                *(uint4*)(_sB + b_off) = make_uint4(0, 0, 0, 0);                                                          \
        }                                                                                                                  \
        /* SFA */                                                                                                          \
        if (tid < BLOCK_M_DIM) {                                                                                           \
            int _gm = block_m + tid;                                                                                       \
            uint32_t _sv = 0;                                                                                              \
            if (_gm < M) {                                                                                                 \
                int _bs = _gm * scale_K + (K_SCALE);                                                                       \
                if ((K_SCALE) + 3 < scale_K)                                                                               \
                    _sv = *(const uint32_t*)(SFA + _bs);                                                                   \
                else                                                                                                       \
                    for (int _b = 0; _b < 4; _b++)                                                                         \
                        if ((K_SCALE) + _b < scale_K)                                                                      \
                            _sv |= ((uint32_t)SFA[_bs + _b]) << (_b * 8);                                                 \
            }                                                                                                              \
            *(uint32_t*)(_sSFA + tid * 4) = _sv;                                                                           \
        }                                                                                                                  \
        /* SFB */                                                                                                          \
        if (tid < BLOCK_N_DIM) {                                                                                           \
            int _gn = block_n + tid;                                                                                       \
            uint32_t _sv = 0;                                                                                              \
            if (_gn < N) {                                                                                                 \
                int _bs = _gn * scale_K + (K_SCALE);                                                                       \
                if ((K_SCALE) + 3 < scale_K)                                                                               \
                    _sv = *(const uint32_t*)(SFB + _bs);                                                                   \
                else                                                                                                       \
                    for (int _b = 0; _b < 4; _b++)                                                                         \
                        if ((K_SCALE) + _b < scale_K)                                                                      \
                            _sv |= ((uint32_t)SFB[_bs + _b]) << (_b * 8);                                                 \
            }                                                                                                              \
            *(uint32_t*)(_sSFB + tid * 4) = _sv;                                                                           \
        }                                                                                                                  \
    } while (0)

    // ================================================================
    // Macro for compute step from a buffer
    // ================================================================
#define COMPUTE_STEP(BUF)                                                                                                  \
    do {                                                                                                                   \
        const unsigned char* _cA = (BUF);                                                                                  \
        const unsigned char* _cB = (BUF) + SMEM_A_BYTES;                                                                   \
        const unsigned char* _cSFA = (BUF) + SMEM_A_BYTES + SMEM_B_BYTES;                                                  \
        const unsigned char* _cSFB = (BUF) + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_SFA_BYTES;                                 \
        uint32_t _ar[4];                                                                                                   \
        _ar[0] = *(const uint32_t*)(_cA + a_local_row0 * 32 + t0 * 4);                                                    \
        _ar[1] = *(const uint32_t*)(_cA + a_local_row1 * 32 + t0 * 4);                                                    \
        _ar[2] = *(const uint32_t*)(_cA + a_local_row0 * 32 + t0 * 4 + 16);                                               \
        _ar[3] = *(const uint32_t*)(_cA + a_local_row1 * 32 + t0 * 4 + 16);                                               \
        uint32_t _sf = *(const uint32_t*)(_cSFA + sfa_local_row * 4);                                                      \
        _Pragma("unroll") for (int _nt = 0; _nt < N_TILES_PER_WARP; _nt++) {                                               \
            int _ln = n_warp * N_TILES_PER_WARP * 8 + _nt * 8;                                                            \
            int _br = _ln + t1;                                                                                            \
            uint32_t _b0 = *(const uint32_t*)(_cB + _br * 32 + t0 * 4);                                                   \
            uint32_t _b1 = *(const uint32_t*)(_cB + _br * 32 + t0 * 4 + 16);                                              \
            uint32_t _sb = *(const uint32_t*)(_cSFB + (_ln + t1) * 4);                                                     \
            mma_nvfp4_m16n8k64(acc[_nt][0], acc[_nt][1], acc[_nt][2], acc[_nt][3], _ar[0], _ar[1], _ar[2], _ar[3], _b0,   \
                               _b1, acc[_nt][0], acc[_nt][1], acc[_nt][2], acc[_nt][3], _sf, _sb);                         \
        }                                                                                                                  \
    } while (0)

    // ================================================================
    // Prologue: load first K-step into buffer 0
    // ================================================================
    COOP_LOAD(smem, 0, 0);
    __syncthreads();

    int cur = 0;
    for (int k_start = 0; k_start < K; k_start += 64) {
        int nxt = 1 - cur;
        unsigned char* cur_buf = smem + cur * SMEM_TOTAL;
        unsigned char* nxt_buf = smem + nxt * SMEM_TOTAL;

        // Issue loads for NEXT K-step into other buffer
        if (k_start + 64 < K) {
            COOP_LOAD(nxt_buf, (k_start + 64) / 2, (k_start + 64) / 16);
        }

        // Compute with CURRENT buffer (overlaps with loads above)
        COMPUTE_STEP(cur_buf);

        // Single sync: ensures both loads and compute are done
        __syncthreads();
        cur = nxt;
    }

#undef COOP_LOAD
#undef COMPUTE_STEP

    // ---- Write output ----
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
