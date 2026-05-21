// SIMT 4-bit GEMM kernel. Compiles for all architectures (sm60+).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "gemm_4bit_common.cuh"
#include "gemm_4bit_simt.cuh"

// Warps per block; each warp owns one N-column. CTA size = WARPS_PER_BLOCK * 32.
static constexpr int WARPS_PER_BLOCK = 4;

// Element-wise multiply of a half2 vector pair.
__device__ __forceinline__ half2 vec2_mul(half2 a, half2 b) { return __hmul2(a, b); }

// Element-wise multiply of a __nv_bfloat162 vector pair.
__device__ __forceinline__ __nv_bfloat162 vec2_mul(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && __CUDA_VERSION__ < 12020
    // Falls back to fp32 round-trip on <= sm75 + CUDA < 12.2, which lacks a __hmul2
    // overload for bf16 on Turing; identical to what CUDA 12.2+ emits for that target.
    return __floats2bfloat162_rn(
        __bfloat162float(a.x) * __bfloat162float(b.x), __bfloat162float(a.y) * __bfloat162float(b.y)
    );
#else
    return __hmul2(a, b);
#endif
}

/// @brief Fused 4-bit dequantize + GEMM with optional nested absmax and bias.
///        Computes C[M,N] = A[M,K] @ B[N,K]^T + bias.
///
/// One warp owns one N-column. All 32 lanes split K in parallel, then warp-reduce.
/// B is dequantized once per outer K step and reused across all M rows.
/// Supports single-level and double-quantized (nested) absmax.
///
/// Dtype paths:
///   bf16/fp16: LUT as uint16-in-uint32 for warp shuffle; T2 pair math; 1x uint4 A load per sub-iter
///   fp32:      LUT as float for warp shuffle; scalar multiply; 2x uint4 A loads per sub-iter
///
/// Grid: (ceil(N/WARPS_PER_BLOCK), ceil(M/M_BLOCK))
///
/// @tparam T       Input/output dtype (`__nv_bfloat16`, `half`, or `float`)
/// @tparam M_BLOCK M rows per block
template <typename T, int M_BLOCK = 1>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32) gemm_4bit_simt(
    // clang-format off
    const T*       __restrict__ A,             // inputs [M, K]
    const uint8_t* __restrict__ B,             // packed 4-bit weights [N, K/2]
    const float*   __restrict__ absmax,        // fp32 absmax [N, K/blocksize] or
                                               //   [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* __restrict__ absmax_8bit,   // [N, K/blocksize] uint8 compressed absmax;
                                               //   nullptr = non-nested
    const float*   __restrict__ absmax_code,   // [256] codebook for 8bit absmax
    const float*   __restrict__ absmax_offset, // scalar; nullptr = non-nested
    T*             __restrict__ C,             // [M, N]
    const T*       __restrict__ bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,                       // problem shape
    int blocksize,                             // elements per quantization block
    int quant_type                             // 1 = FP4, 2 = NF4
    // clang-format on
) {
    // T2 is the vector-pair type; for float it resolves to float2 but is unused.
    using T2 = std::conditional_t<
        std::is_same_v<T, __nv_bfloat16>, __nv_bfloat162, std::conditional_t<std::is_same_v<T, half>, half2, float2>>;

    const float absmax_offset_f = absmax_8bit ? __ldg(absmax_offset) : 0.0f;

    const int blocksize_log2 = __ffs(blocksize) - 1;

    constexpr int NUM_VAL = 32;            // 4-bit elements per lane per outer K step
    constexpr int K_STRIDE = 32 * NUM_VAL; // = 1024 K elements per outer step

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int base_m = blockIdx.y * M_BLOCK;

    if (warp_n >= N)
        return;

    // Each lane loads its LUT entry (lane_id < 16) for warp-shuffle dequant.
    // bf16/fp16: centroid as uint16-in-uint32 for __shfl_sync over uint32.
    // fp32: centroid as float for __shfl_sync over float.
    [[maybe_unused]] uint32_t my_lut_u32 = 0u;
    [[maybe_unused]] float my_lut_f32_shfl = 0.0f;

    if (lane_id < 16) {
        const float* lut = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
        if constexpr (std::is_same_v<T, float>)
            my_lut_f32_shfl = lut[lane_id];
        else {
            const T centroid = static_cast<T>(lut[lane_id]);
            my_lut_u32 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&centroid));
        }
    }

    const int blk_per_row = K >> blocksize_log2;

    // Per-M accumulators. M_BLOCK is compile-time so the loop fully unrolls.
    float acc[M_BLOCK];
#pragma unroll
    for (int m = 0; m < M_BLOCK; m++)
        acc[m] = 0.f;

    const int m_valid = min(M_BLOCK, max(0, M - base_m));

    // All 32 lanes run the same number of K iterations for `__shfl_sync` convergence.
    // Inactive lanes load b_packed4={0}, which decodes to 0 and contributes nothing.
    const int num_groups = (K + K_STRIDE - 1) / K_STRIDE;

    for (int g = 0; g < num_groups; g++) {
        const int inner_k = g * K_STRIDE + lane_id * NUM_VAL;
        const bool lane_active = (inner_k < K);

        // Scale: one absmax per lane per outer K step.
        float scale_f = 0.0f;
        if (lane_active) {
            const int blk_idx = warp_n * blk_per_row + (inner_k >> blocksize_log2);
            if (absmax_8bit) {
                // absmax_8bit[blk_idx] is a uint8 index into absmax_code.
                // absmax[blk_idx >> 8] is the fp32 state2 scale (one per 256 blocks).
                // absmax_offset is subtracted at quantize time and re-added here.
                scale_f = __ldg(&absmax_code[absmax_8bit[blk_idx]]) * __ldg(&absmax[blk_idx >> 8]) + absmax_offset_f;
            } else {
                scale_f = __ldg(&absmax[blk_idx]);
            }
        }

        // Load 16 packed 4-bit bytes (.cs: streaming, bypasses L1).
        // Inactive lanes keep b_packed4 = {0}.
        uint4 b_packed4 = {0u, 0u, 0u, 0u};
        if (lane_active) {
            const uint32_t* bptr = reinterpret_cast<const uint32_t*>(B + warp_n * (K / 2) + inner_k / 2);
            // clang-format off
            asm volatile(
                "ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(b_packed4.x), "=r"(b_packed4.y),
                  "=r"(b_packed4.z), "=r"(b_packed4.w)
                : "l"(bptr)
            );
            // clang-format on
        }
        const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b_packed4);

        // Decode B and accumulate.
        // bf16/fp16: uint16-in-uint32 LUT, hmul2 vector math, 1x uint4 A load per sub-iter.
        // fp32:      float LUT, scalar multiply, 2x uint4 A loads per sub-iter.
        [[maybe_unused]] T2 scale_x2{};
        if constexpr (!std::is_same_v<T, float>)
            scale_x2 = broadcast_vec2<T>(scale_f);

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            // Decode 4 B bytes (8 nibbles) into 8 dequantized values.
            // hi nibble (>>4) = lower K index, lo nibble (&0xf) = higher K index.
            [[maybe_unused]] T2 b_chunk[4];
            [[maybe_unused]] float b_dq[8];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint8_t byte = b_bytes[sub * 4 + j];
                if constexpr (std::is_same_v<T, float>) {
                    b_dq[j * 2] = __shfl_sync(0xffffffff, my_lut_f32_shfl, byte >> 4) * scale_f;
                    b_dq[j * 2 + 1] = __shfl_sync(0xffffffff, my_lut_f32_shfl, byte & 0x0f) * scale_f;
                } else {
                    const uint32_t hi = __shfl_sync(0xffffffff, my_lut_u32, byte >> 4);
                    const uint32_t lo = __shfl_sync(0xffffffff, my_lut_u32, byte & 0x0f);
                    b_chunk[j] = vec2_mul(vec2_from_u16bits<T>(hi, lo), scale_x2);
                }
            }

            if (lane_active) {
#pragma unroll
                for (int m = 0; m < M_BLOCK; m++) {
                    if (m >= m_valid)
                        break;
                    const int m_global = base_m + m;
                    const int a_k = inner_k + sub * 8;

                    if constexpr (std::is_same_v<T, float>) {
                        // 8 floats = 2x uint4 (cached; A rows reused across N-tiles)
                        const uint4 a4a = *reinterpret_cast<const uint4*>(&A[m_global * K + a_k]);
                        const uint4 a4b = *reinterpret_cast<const uint4*>(&A[m_global * K + a_k + 4]);
                        const float* fa = reinterpret_cast<const float*>(&a4a);
                        const float* fb = reinterpret_cast<const float*>(&a4b);
#pragma unroll
                        for (int k = 0; k < 4; k++) {
                            acc[m] += fa[k] * b_dq[k];
                            acc[m] += fb[k] * b_dq[k + 4];
                        }
                    } else {
                        // 8 T elements as uint4 (cached; A rows reused across N-tiles).
                        // sm90+: asm fence gives better occupancy.
#if __CUDA_ARCH__ >= 900
                        uint4 a_packed4;
                        // clang-format off
                        asm volatile(
                            "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                            : "=r"(a_packed4.x), "=r"(a_packed4.y),
                              "=r"(a_packed4.z), "=r"(a_packed4.w)
                            : "l"(&A[m_global * K + a_k])
                        );
                        // clang-format on
#else
                        const uint4 a_packed4 = *reinterpret_cast<const uint4*>(&A[m_global * K + a_k]);
#endif
                        const T2* a_pairs = reinterpret_cast<const T2*>(&a_packed4);
#pragma unroll
                        for (int k = 0; k < 4; k++) {
                            const float2 p = vec2_to_float2(vec2_mul(a_pairs[k], b_chunk[k]));
                            acc[m] += p.x + p.y;
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    // Warp reduce: sum acc[m] across all 32 lanes.
    for (int m = 0; m < M_BLOCK; m++) {
        if (m >= m_valid)
            break;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
    }

    if (lane_id == 0) {
        const float bias_f = bias ? static_cast<float>(bias[warp_n]) : 0.0f;
#pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            if (m >= m_valid)
                break;
            C[(base_m + m) * N + warp_n] = static_cast<T>(acc[m] + bias_f);
        }
    }
}

/// @brief Host launcher for the SIMT 4-bit GEMM kernel. Selects M_BLOCK at compile
///        time based on M (exact for M=1..8, clamped to 8 above).
/// @tparam T Input/output dtype (`__nv_bfloat16`, `half`, or `float`)
template <typename T>
void launch_gemm_4bit_simt(
    // clang-format off
    const T*       A,             // inputs [M, K]
    const uint8_t* B,             // packed 4-bit weights [N, K/2]
    const float*   absmax,        // fp32 absmax [N*K/blocksize] or [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* absmax_8bit,   // [N*K/blocksize] uint8 compressed absmax; nullptr = non-nested
    const float*   absmax_code,   // [256] codebook for 8bit absmax
    const float*   absmax_offset, // scalar; nullptr = non-nested
    T*             C,             // output [M, N]
    const T*       bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,          // problem shape
    int blocksize,                // elements per quantization block
    int quant_type,               // 1 = FP4, 2 = NF4
    cudaStream_t stream           // CUDA stream
    // clang-format on
) {
    const int n_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // M=1..8: M_BLOCK == M, so the inner m-loop fully unrolls at compile time.
    // M>8: M_BLOCK=8, ceil(M/8) grid rows.
    auto launch = [&](auto mb_tag) {
        constexpr int MB = decltype(mb_tag)::value;
        const int grid_y = (M + MB - 1) / MB;
        gemm_4bit_simt<T, MB><<<dim3(n_blocks, grid_y), WARPS_PER_BLOCK * 32, 0, stream>>>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type
        );
    };

    // clang-format off
    switch (min(M, 8)) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        default: launch(std::integral_constant<int, 8>{}); break;
    }
    // clang-format on
}

// Explicit instantiations for supported dtypes.
template void launch_gemm_4bit_simt<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, __nv_bfloat16*,
    const __nv_bfloat16*, int, int, int, int, int, cudaStream_t
);
template void launch_gemm_4bit_simt<half>(
    const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int, int,
    int, int, int, cudaStream_t
);
template void launch_gemm_4bit_simt<float>(
    const float*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, float*, const float*, int,
    int, int, int, int, cudaStream_t
);
