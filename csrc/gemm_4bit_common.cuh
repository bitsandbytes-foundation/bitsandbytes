#pragma once

// Shared types and utilities for 4bit GEMM kernels.

// GPU properties queried once per device and cached in gemm_4bit.cu.
// Passed through dispatch into MMA launchers to avoid repeated cudaGetDevice calls.
struct GpuProps {
    int device_index, num_sms, cc_major, cc_minor;
};

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

// NF4 dequantization LUT
static __device__ __constant__ float NF4_LUT_F32[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

// FP4 dequantization LUT
static __device__ __constant__ float FP4_LUT_F32[16] = {
    0.0f,             // 0b0000
    0.005208333333f,  // 0b0001
    0.66666667f,      // 0b0010
    1.0f,             // 0b0011
    0.33333333f,      // 0b0100
    0.5f,             // 0b0101
    0.16666667f,      // 0b0110
    0.25f,            // 0b0111
    -0.0f,            // 0b1000
    -0.005208333333f, // 0b1001
    -0.66666667f,     // 0b1010
    -1.0f,            // 0b1011
    -0.33333333f,     // 0b1100
    -0.5f,            // 0b1101
    -0.16666667f,     // 0b1110
    -0.25f,           // 0b1111
};

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

// Indicates whether `T` is a 16-bit float type supported in our kernels (currently `__nv_bfloat16` or `half`).
template <typename T> constexpr bool is_16bit_float_v = std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>;

// Convert two floats to a T vector-pair (half2 or __nv_bfloat162), with rounding.
template <typename T> __device__ __forceinline__ auto make_vec2(float a, float b) {
    static_assert(is_16bit_float_v<T>, "make_vec2: T must be __nv_bfloat16 or half");
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __floats2bfloat162_rn(a, b);
    else
        return __floats2half2_rn(a, b);
}

// Single float broadcast into both lanes of a T vector-pair.
template <typename T> __device__ __forceinline__ auto broadcast_vec2(float x) {
    static_assert(is_16bit_float_v<T>, "broadcast_vec2: T must be __nv_bfloat16 or half");
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __float2bfloat162_rn(x);
    else
        return __float2half2_rn(x);
}

// T vector-pair -> float2.
template <typename T2> __device__ __forceinline__ float2 vec2_to_float2(T2 v) {
    if constexpr (std::is_same_v<T2, __nv_bfloat162>)
        // __bfloat1622float2 is gated behind sm80+ in CUDA < 12.2. Two
        // __bfloat162float calls emit identical PTX (cvt.f32.bf16 on sm90+,
        // mov.b32 on earlier targets).
        return {__bfloat162float(v.x), __bfloat162float(v.y)};
    else
        return __half22float2(v);
}

// Two uint32 values holding uint16 bit patterns -> T vector-pair.
// Used to reassemble warp-shuffled LUT entries.
template <typename T> __device__ __forceinline__ auto vec2_from_u16bits(uint32_t hi, uint32_t lo) {
    static_assert(is_16bit_float_v<T>, "vec2_from_u16bits: T must be __nv_bfloat16 or half");
    using T2 = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, __nv_bfloat162, half2>;
    const uint16_t hi16 = (uint16_t)hi, lo16 = (uint16_t)lo;
    return T2{*reinterpret_cast<const T*>(&hi16), *reinterpret_cast<const T*>(&lo16)};
}

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
