#pragma once

// CUDA tensor-core (mma.sync m16n8k*) types and epilogue for the 4-bit GEMM
// kernels. Included ONLY by the CUDA MMA translation units (gemm_4bit_sm75.cu,
// gemm_4bit_sm80.cu), which are compiled exclusively by nvcc and never on ROCm
// (see CMakeLists.txt). Keeping this code out of gemm_4bit_common.cuh means the
// HIP compiler never parses these PTX-dependent bodies, so no HIP-compat stubs
// are needed here. The AMD matrix-core (WMMA) path uses a different fragment
// layout and will live in its own gemm_4bit_wmma.cuh.

#include "gemm_4bit_common.cuh"

#include <cstdint>
#include <type_traits>

// MMA accumulator fragment for m16n8k*
struct FragC {
    float x[4];
};

/// @brief Compile-time layout of warps for mma.sync m16n8k* instructions
///
/// Warps are split `WARPS_M x WARPS_N` where `WARPS_M * WARPS_N == NUM_WARPS_VAL`.
/// Default split: MT<=32 -> 2Mx(N/2); MT>32 -> 4Mx(N/4).
///
/// @tparam MT Number of rows  (e.g. 32, 64, 128)
/// @tparam NT Number of columns (e.g. 32, 64, 128, 256)
/// @tparam NUM_WARPS_VAL Total number of warps (e.g. 4, 8, 16); must be divisible by WARPS_M
/// @tparam MMA_M_VAL MMA M dimension (e.g. 16)
/// @tparam MMA_N_VAL MMA N dimension (e.g. 8)
template <int MT, int NT, int NUM_WARPS_VAL = 8, int MMA_M_VAL = 16, int MMA_N_VAL = 8> struct MmaWarpLayout {
    static constexpr int WARPS_M = (MT <= 32) ? 2 : 4;
    static constexpr int WARPS_N = NUM_WARPS_VAL / WARPS_M;
    static constexpr int WARP_M = MT / WARPS_M;
    static constexpr int WARP_MMA_M = WARP_M / MMA_M_VAL;
    static constexpr int WARP_N = NT / WARPS_N;
    static constexpr int WARP_MMA_N = WARP_N / MMA_N_VAL;
};

// Two f32 accumulators -> packed uint32 of two T values, via PTX cvt.
template <typename T> __device__ __forceinline__ uint32_t cvt_f32x2_to_packed(float hi, float lo) {
    static_assert(is_16bit_float_v<T>, "cvt_f32x2_to_packed: T must be __nv_bfloat16 or half");
    uint32_t result;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(hi), "f"(lo));
    } else {
        // cvt.rn.f16x2.f32 requires sm80+; __floats2half2_rn is used instead
        // so the compiler picks the right instruction per target.
        // lo -> bits [15:0], hi -> bits [31:16].
        const half2 h2 = __floats2half2_rn(lo, hi);
        result = *reinterpret_cast<const uint32_t*>(&h2);
    }
    return result;
}

/// @brief MMA epilogue: store f32 accumulators to C with optional bias, for m16n8k* layouts.
///
/// D-fragment layout (mma.sync m16n8k*):
///   group = lane/4,  tid = lane%4
///   d[0] -> C[group,         tid*2]    d[1] -> C[group,         tid*2+1]
///   d[2] -> C[group+MMA_M/2, tid*2]    d[3] -> C[group+MMA_M/2, tid*2+1]
///
/// @tparam T Output type (__nv_bfloat16 or half)
/// @tparam WARP_MMA_M Number of M-direction MMA tiles per warp
/// @tparam WARP_MMA_N Number of N-direction MMA tiles per warp
/// @tparam MMA_M_VAL MMA M dimension (default 16)
/// @tparam MMA_N_VAL MMA N dimension (default 8)
/// @param C Output pointer [M, N]
/// @param accum f32 accumulator fragments [WARP_MMA_M][WARP_MMA_N]
/// @param bm Block M offset (blockIdx.x * MT)
/// @param bn Block N offset (blockIdx.y * NT)
/// @param wm_off Warp M offset within block
/// @param wn_off Warp N offset within block
/// @param M Total output rows
/// @param N Total output columns
/// @param lane_id Lane index within warp (0-31)
/// @param bias Optional bias [N]; nullptr = no bias
template <typename T, int WARP_MMA_M, int WARP_MMA_N, int MMA_M_VAL = 16, int MMA_N_VAL = 8>
__device__ __forceinline__ void mma_store_accum(
    T* C, const FragC (&accum)[WARP_MMA_M][WARP_MMA_N], int bm, int bn, int wm_off, int wn_off, int M, int N,
    int lane_id,
    const T* bias = nullptr // [N], optional; accumulated in fp32 before downcast
) {
    constexpr int ROW_STRIDE = MMA_M_VAL / 2;
    const int group = lane_id / 4;
    const int tid = lane_id % 4;

#pragma unroll
    for (int wm = 0; wm < WARP_MMA_M; wm++) {
#pragma unroll
        for (int wn = 0; wn < WARP_MMA_N; wn++) {
            const int base_m = bm + wm_off + wm * MMA_M_VAL;
            const int base_n = bn + wn_off + wn * MMA_N_VAL;
            const int m0 = base_m + group;
            const int m1 = base_m + group + ROW_STRIDE;
            const int n0 = base_n + tid * 2;
            const int n1 = n0 + 1;

            // Bias is added in fp32 before the downcast.
            const float b0 = bias ? static_cast<float>(bias[n0]) : 0.0f;
            const float b1 = (bias && n1 < N) ? static_cast<float>(bias[n1]) : 0.0f;

            if (m0 < M) {
                if (__builtin_expect(n1 < N, 1)) {
                    const uint32_t c01 = cvt_f32x2_to_packed<T>(accum[wm][wn].x[1] + b1, accum[wm][wn].x[0] + b0);
                    // clang-format off
                    asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(&C[m0 * N + n0]), "r"(c01));
                    // clang-format on
                } else if (n0 < N) {
                    C[m0 * N + n0] = static_cast<T>(accum[wm][wn].x[0] + b0);
                }
            }
            if (m1 < M) {
                if (__builtin_expect(n1 < N, 1)) {
                    const uint32_t c23 = cvt_f32x2_to_packed<T>(accum[wm][wn].x[3] + b1, accum[wm][wn].x[2] + b0);
                    // clang-format off
                    asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(&C[m1 * N + n0]), "r"(c23));
                    // clang-format on
                } else if (n0 < N) {
                    C[m1 * N + n0] = static_cast<T>(accum[wm][wn].x[2] + b0);
                }
            }
        }
    }
}
