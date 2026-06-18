#pragma once

// Shared types and utilities for 4bit GEMM kernels.

#include "compat.cuh"

#include <cstdint>
#include <type_traits>


// GPU properties queried once per device and cached in gemm_4bit.cu.
// Passed through dispatch into MMA launchers to avoid repeated cudaGetDevice calls.
struct GpuProps {
    int device_index, num_sms, cc_major, cc_minor;
};

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


// Indicates whether `T` is a 16-bit float type supported in our kernels (currently `bnb_bfloat16` or `half`).
template <typename T> constexpr bool is_16bit_float_v = std::is_same_v<T, bnb_bfloat16> || std::is_same_v<T, half>;

// Convert two floats to a T vector-pair (half2 or bnb_bfloat162), with rounding.
template <typename T> __device__ __forceinline__ auto make_vec2(float a, float b) {
    static_assert(is_16bit_float_v<T>, "make_vec2: T must be bnb_bfloat16 or half");
    if constexpr (std::is_same_v<T, bnb_bfloat16>)
#if BNB_HIP
        return bnb_bfloat162{__float2bfloat16(a), __float2bfloat16(b)};
#else
        return __floats2bfloat162_rn(a, b);
#endif
    else
        return __floats2half2_rn(a, b);
}

// Single float broadcast into both lanes of a T vector-pair.
template <typename T> __device__ __forceinline__ auto broadcast_vec2(float x) {
    static_assert(is_16bit_float_v<T>, "broadcast_vec2: T must be bnb_bfloat16 or half");
    if constexpr (std::is_same_v<T, bnb_bfloat16>)
#if BNB_HIP
        return bnb_bfloat162{__float2bfloat16(x), __float2bfloat16(x)};
#else
        return __float2bfloat162_rn(x);
#endif
    else
        return __float2half2_rn(x);
}

// T vector-pair -> float2.
template <typename T2> __device__ __forceinline__ float2 vec2_to_float2(T2 v) {
    if constexpr (std::is_same_v<T2, bnb_bfloat162>)
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
    static_assert(is_16bit_float_v<T>, "vec2_from_u16bits: T must be bnb_bfloat16 or half");
    using T2 = std::conditional_t<std::is_same_v<T, bnb_bfloat16>, bnb_bfloat162, half2>;
    const uint16_t hi16 = (uint16_t)hi, lo16 = (uint16_t)lo;
    return T2{*reinterpret_cast<const T*>(&hi16), *reinterpret_cast<const T*>(&lo16)};
}
