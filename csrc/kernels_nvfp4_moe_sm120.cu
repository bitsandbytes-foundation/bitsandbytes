// Batched NVFP4 MoE GEMM for SM_120a (consumer Blackwell: RTX 5090, RTX 6000)
//
// CUDA-graph friendly: fixed grid (n_tiles, m_tiles, num_experts), no dynamic
// routing, no host-device sync. All experts compute max_M rows; padded rows
// produce ignored output that the caller discards.
//
// Uses the same hand-written PTX mma.sync as the existing grouped kernel,
// but with batched layout instead of concatenated+binary-search.
//
// Data layout:
//   A_batched:   (num_experts, max_M, K/2)  packed FP4, row-major per expert
//   B_all:       (num_experts, N, K/2)      packed FP4, row-major per expert
//   SFA_batched: (num_experts, sfa_per_expert_bytes) per-expert swizzled scales
//   SFB_all:     (num_experts, sfb_per_expert_bytes) per-expert swizzled scales
//   D_batched:   (num_experts, max_M, N)    BF16 output, row-major per expert
//
// No alpha epilogue — tensor scales applied post-hoc in Python (same as
// existing SM_120 grouped kernel).
//
// Must be compiled with: -gencode=arch=compute_120a,code=sm_120a

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

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
// Swizzled scale index computation (same as existing kernel)
// ============================================================================
__device__ __forceinline__ int swizzled_scale_offset(int row, int col, int n_col_blocks) {
    int block_row = row >> 7;         // row / 128
    int block_col = col >> 2;         // col / 4
    int r = row & 127;               // row % 128
    int c = col & 3;                 // col % 4
    int block_idx = block_row * n_col_blocks + block_col;
    return block_idx * 512 + (r & 31) * 16 + (r >> 5) * 4 + c;
}

// ============================================================================
// Output conversion helpers
// ============================================================================
template <typename T> __device__ __forceinline__ T float_to_out(float v);
template <> __device__ __forceinline__ float float_to_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 float_to_out<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}
template <> __device__ __forceinline__ half float_to_out<half>(float v) { return __float2half(v); }

// ============================================================================
// Block tile dimensions (same as existing kernel)
// ============================================================================
#define MOE_N_TILES_PER_WARP 4
#define MOE_M_WARPS 2
#define MOE_N_WARPS 4
#define MOE_WARPS_PER_BLOCK (MOE_M_WARPS * MOE_N_WARPS) // 8
#define MOE_BLOCK_M_DIM (MOE_M_WARPS * 16)                   // 32
#define MOE_BLOCK_N_DIM (MOE_N_WARPS * MOE_N_TILES_PER_WARP * 8) // 128
#define MOE_SMEM_A_BYTES (MOE_BLOCK_M_DIM * 32)  // 1024
#define MOE_SMEM_B_BYTES (MOE_BLOCK_N_DIM * 32)  // 4096
#define MOE_SMEM_SFA_BYTES (MOE_BLOCK_M_DIM * 4) // 128
#define MOE_SMEM_SFB_BYTES (MOE_BLOCK_N_DIM * 4) // 512
#define MOE_SMEM_TOTAL (MOE_SMEM_A_BYTES + MOE_SMEM_B_BYTES + MOE_SMEM_SFA_BYTES + MOE_SMEM_SFB_BYTES)

// ============================================================================
// Batched MoE GEMM kernel
//
// Grid: (num_n_tiles, num_m_tiles, num_experts)
//   blockIdx.x = n_tile index
//   blockIdx.y = m_tile index
//   blockIdx.z = expert index
//
// Each expert computes max_M × N output with its own activation/weight data.
// Padded rows (beyond actual token count) produce garbage that the caller
// discards during the gather step.
// ============================================================================
template <typename OutT>
__global__ __launch_bounds__(MOE_WARPS_PER_BLOCK * 32, 4) void kBatchedMoeGemmNVFP4(
    const unsigned char* __restrict__ A_batched,   // (num_experts, max_M, K/2)
    const unsigned char* __restrict__ B_all,        // (num_experts, N, K/2)
    const unsigned char* __restrict__ SFA_batched,  // per-expert swizzled act scales
    const unsigned char* __restrict__ SFB_all,      // per-expert swizzled wt scales
    OutT* __restrict__ D_batched,                   // (num_experts, max_M, N)
    int max_M, int N, int K,
    int sfa_per_expert_bytes,                       // size of one expert's SFA block
    int sfb_per_expert_bytes                        // size of one expert's SFB block
) {
    const int expert = blockIdx.z;
    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;

    const int half_K = K / 2;
    const int scale_K = K / 16;
    const int scale_n_col_blocks = (scale_K + 3) / 4;

    // Point to this expert's data in the batched layout
    const unsigned char* A   = A_batched   + (size_t)expert * max_M * half_K;
    const unsigned char* B   = B_all       + (size_t)expert * N * half_K;
    const unsigned char* SFA = SFA_batched + (size_t)expert * sfa_per_expert_bytes;
    const unsigned char* SFB = SFB_all     + (size_t)expert * sfb_per_expert_bytes;
    OutT*                D   = D_batched   + (size_t)expert * max_M * N;
    const int M = max_M;  // compute all rows including padding

    // --- Standard tile GEMM (same core logic as kGemmNVFP4_smem) ---
    __shared__ __align__(16) unsigned char smem[MOE_SMEM_TOTAL];
    unsigned char* smem_A   = smem;
    unsigned char* smem_B   = smem + MOE_SMEM_A_BYTES;
    unsigned char* smem_SFA = smem + MOE_SMEM_A_BYTES + MOE_SMEM_B_BYTES;
    unsigned char* smem_SFB = smem + MOE_SMEM_A_BYTES + MOE_SMEM_B_BYTES + MOE_SMEM_SFA_BYTES;

    const int tid = threadIdx.x;
    const int warp_in_block = tid / 32;
    const int lane_id = tid % 32;
    const int m_warp = warp_in_block / MOE_N_WARPS;
    const int n_warp = warp_in_block % MOE_N_WARPS;

    const int block_m = m_tile * MOE_BLOCK_M_DIM;
    const int block_n = n_tile * MOE_BLOCK_N_DIM;
    const int tile_m = block_m + m_warp * 16;
    const int warp_n_base = block_n + n_warp * MOE_N_TILES_PER_WARP * 8;

    const int t0 = lane_id % 4;
    const int t1 = lane_id / 4;

    float acc[MOE_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int nt = 0; nt < MOE_N_TILES_PER_WARP; nt++) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    const int a_local_row0 = m_warp * 16 + 2 * t1;
    const int a_local_row1 = a_local_row0 + 1;
    const int sf_tidx = (lane_id % 2) * 8 + (lane_id / 4);
    const int cute_sf_m0 = sf_tidx % 16;
    const int sfa_local_row = m_warp * 16 + (cute_sf_m0 % 8) * 2 + cute_sf_m0 / 8;

    const int a_off = tid * 4;
    const int a_load_row = a_off >> 5;
    const int a_load_col = a_off & 31;
    const int a_gm = block_m + a_load_row;

    const int b_off = tid * 16;
    const int b_load_row = b_off >> 5;
    const int b_load_col = b_off & 31;
    const int b_gn = block_n + b_load_row;

    const bool a_gm_ok = (a_gm < M);
    const bool b_gn_ok = (b_gn < N);
    const int a_row_base = a_gm * half_K;
    const int b_row_base = b_gn * half_K;

    // Pipeline registers
    uint32_t pipe_a = 0;
    uint4 pipe_b = make_uint4(0, 0, 0, 0);
    uint32_t pipe_sfa = 0, pipe_sfb = 0;

    // Load helper — uses per-expert local offsets for scales
    auto do_load = [&](int k_byte, int k_scale) {
        pipe_a = 0;
        if (a_gm_ok) {
            int ga = a_row_base + k_byte + a_load_col;
            if (k_byte + a_load_col + 3 < half_K)
                pipe_a = *(const uint32_t*)(A + ga);
            else
                for (int i = 0; i < 4; i++)
                    if (k_byte + a_load_col + i < half_K)
                        pipe_a |= ((uint32_t)A[ga + i]) << (i * 8);
        }
        if (b_gn_ok) {
            int gb = b_row_base + k_byte + b_load_col;
            if (k_byte + b_load_col + 15 < half_K) {
                uint4 bv = *(const uint4*)(B + gb);
                pipe_b.x = bv.x; pipe_b.y = bv.y; pipe_b.z = bv.z; pipe_b.w = bv.w;
            } else {
                unsigned char buf[16] = {};
                for (int i = 0; i < 16; i++)
                    if (k_byte + b_load_col + i < half_K) buf[i] = B[gb + i];
                pipe_b = *(uint4*)buf;
            }
        } else { pipe_b = make_uint4(0, 0, 0, 0); }

        // SFA: per-expert swizzled layout (local row indices within this expert)
        pipe_sfa = 0;
        if (tid < MOE_BLOCK_M_DIM) {
            int gm = block_m + tid;
            if (gm < M) {
                int bs = swizzled_scale_offset(gm, k_scale, scale_n_col_blocks);
                if (k_scale + 3 < scale_K)
                    pipe_sfa = *(const uint32_t*)(SFA + bs);
                else
                    for (int i = 0; i < 4; i++)
                        if (k_scale + i < scale_K)
                            pipe_sfa |= ((uint32_t)SFA[bs + i]) << (i * 8);
            }
        }
        // SFB: per-expert swizzled layout (local row indices within this expert)
        pipe_sfb = 0;
        if (tid < MOE_BLOCK_N_DIM) {
            int gn = block_n + tid;
            if (gn < N) {
                int bs = swizzled_scale_offset(gn, k_scale, scale_n_col_blocks);
                if (k_scale + 3 < scale_K)
                    pipe_sfb = *(const uint32_t*)(SFB + bs);
                else
                    for (int i = 0; i < 4; i++)
                        if (k_scale + i < scale_K)
                            pipe_sfb |= ((uint32_t)SFB[bs + i]) << (i * 8);
            }
        }
    };

    auto do_store = [&]() {
        *(uint32_t*)(smem_A + a_off) = pipe_a;
        *(uint4*)(smem_B + b_off) = pipe_b;
        if (tid < MOE_BLOCK_M_DIM) *(uint32_t*)(smem_SFA + tid * 4) = pipe_sfa;
        if (tid < MOE_BLOCK_N_DIM) *(uint32_t*)(smem_SFB + tid * 4) = pipe_sfb;
    };

    auto do_compute = [&]() {
        uint32_t ar[4];
        ar[0] = *(const uint32_t*)(smem_A + a_local_row0 * 32 + t0 * 4);
        ar[1] = *(const uint32_t*)(smem_A + a_local_row1 * 32 + t0 * 4);
        ar[2] = *(const uint32_t*)(smem_A + a_local_row0 * 32 + t0 * 4 + 16);
        ar[3] = *(const uint32_t*)(smem_A + a_local_row1 * 32 + t0 * 4 + 16);
        uint32_t sf = *(const uint32_t*)(smem_SFA + sfa_local_row * 4);
        #pragma unroll
        for (int nt = 0; nt < MOE_N_TILES_PER_WARP; nt++) {
            int ln = n_warp * MOE_N_TILES_PER_WARP * 8 + nt * 8;
            int br = ln + t1;
            uint32_t b0 = *(const uint32_t*)(smem_B + br * 32 + t0 * 4);
            uint32_t b1 = *(const uint32_t*)(smem_B + br * 32 + t0 * 4 + 16);
            uint32_t sb = *(const uint32_t*)(smem_SFB + (ln + t1) * 4);
            mma_nvfp4_m16n8k64(
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                ar[0], ar[1], ar[2], ar[3], b0, b1,
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], sf, sb
            );
        }
    };

    // Load first K-step
    do_load(0, 0);
    do_store();
    __syncthreads();

    for (int k_start = 0; k_start < K; k_start += 64) {
        bool has_next = (k_start + 64 < K);
        if (has_next) do_load((k_start + 64) / 2, (k_start + 64) / 16);
        do_compute();
        __syncthreads();
        if (has_next) { do_store(); __syncthreads(); }
    }

    // Write output (no split-K, direct store)
    int octet = lane_id / 4;
    int quad = lane_id % 4;
    int out_row0 = tile_m + octet * 2;
    int out_row1 = out_row0 + 1;
    int out_col_base = quad * 2;

    #pragma unroll
    for (int nt = 0; nt < MOE_N_TILES_PER_WARP; nt++) {
        int this_tile_n = warp_n_base + nt * 8;
        int c0 = this_tile_n + out_col_base;
        int c1 = c0 + 1;
        if (out_row0 < M && c0 < N) D[out_row0 * N + c0] = float_to_out<OutT>(acc[nt][0]);
        if (out_row0 < M && c1 < N) D[out_row0 * N + c1] = float_to_out<OutT>(acc[nt][1]);
        if (out_row1 < M && c0 < N) D[out_row1 * N + c0] = float_to_out<OutT>(acc[nt][2]);
        if (out_row1 < M && c1 < N) D[out_row1 * N + c1] = float_to_out<OutT>(acc[nt][3]);
    }
}

// ============================================================================
// Launcher and C interface
// ============================================================================

template <typename OutT>
static void launch_batched_moe_gemm_nvfp4(
    const unsigned char* A_batched, const unsigned char* B_all,
    const unsigned char* SFA_batched, const unsigned char* SFB_all,
    OutT* D_batched,
    int max_M, int N, int K, int num_experts,
    int sfa_per_expert_bytes, int sfb_per_expert_bytes,
    cudaStream_t stream
) {
    int num_m_tiles = (max_M + MOE_BLOCK_M_DIM - 1) / MOE_BLOCK_M_DIM;
    int num_n_tiles = (N + MOE_BLOCK_N_DIM - 1) / MOE_BLOCK_N_DIM;
    int threads_per_block = MOE_WARPS_PER_BLOCK * 32;

    dim3 grid(num_n_tiles, num_m_tiles, num_experts);
    kBatchedMoeGemmNVFP4<OutT><<<grid, threads_per_block, 0, stream>>>(
        A_batched, B_all, SFA_batched, SFB_all, D_batched,
        max_M, N, K, sfa_per_expert_bytes, sfb_per_expert_bytes
    );
}

extern "C" void cgemm_nvfp4_moe_bf16(
    const unsigned char* A_batched,
    const unsigned char* B_all,
    const unsigned char* SFA_batched,
    const unsigned char* SFB_all,
    __nv_bfloat16* D_batched,
    int max_M, int N, int K, int num_experts,
    int sfa_per_expert_bytes, int sfb_per_expert_bytes,
    cudaStream_t stream
) {
    launch_batched_moe_gemm_nvfp4<__nv_bfloat16>(
        A_batched, B_all, SFA_batched, SFB_all, D_batched,
        max_M, N, K, num_experts,
        sfa_per_expert_bytes, sfb_per_expert_bytes, stream
    );
}
