// SIMT 4-bit GEMM kernel. Compiles for all CUDA architectures (sm60+) and for
// ROCm/HIP (RDNA wave32 and CDNA wave64) via the compat shims in gemm_4bit_common.cuh.

#include <cstdint>
#include <type_traits>

#include "gemm_4bit_common.cuh"
#include "gemm_4bit_simt.cuh"

// Warps per block; each warp owns one N-column. CTA size = WARPS_PER_BLOCK * 32.
static constexpr int WARPS_PER_BLOCK = 4;

// fp32 takes the LDS centroid LUT only on RDNA3 (gfx11); it regresses on RDNA4
// (gfx12), which keeps the warp shuffle.
#if BNB_HIP && defined(__GFX11__)
#define BNB_HIP_FP32_LDS_LUT 1
#else
#define BNB_HIP_FP32_LDS_LUT 0
#endif

// Element-wise multiply of a half2 vector pair.
__device__ __forceinline__ half2 vec2_mul(half2 a, half2 b) { return __hmul2(a, b); }

// Element-wise multiply of a bnb_bfloat162 vector pair.
__device__ __forceinline__ bnb_bfloat162 vec2_mul(bnb_bfloat162 a, bnb_bfloat162 b) {
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

// Sum a per-lane float across the 32-lane warp; result valid in lane 0.
// HIP: DPP row-shift reduction + xor-16 exchange (cheaper than shfl on RDNA).
// CUDA/others: standard __shfl_down_sync tree.
__device__ __forceinline__ float simt_warp_reduce_sum(float v) {
#if BNB_HIP
    v += __builtin_amdgcn_mov_dpp(v, 0x108, 0xf, 0xf, 1); // row_shr:8
    v += __builtin_amdgcn_mov_dpp(v, 0x104, 0xf, 0xf, 1); // row_shr:4
    v += __builtin_amdgcn_mov_dpp(v, 0x102, 0xf, 0xf, 1); // row_shr:2
    v += __builtin_amdgcn_mov_dpp(v, 0x101, 0xf, 0xf, 1); // row_shr:1
    return v + __shfl_xor(v, 16, 32);
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(BNB_FULL_WARP_MASK, v, offset, 32);
    return v;
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
/// @tparam T       Input/output dtype (`bnb_bfloat16`, `half`, or `float`)
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
        std::is_same_v<T, bnb_bfloat16>, bnb_bfloat162, std::conditional_t<std::is_same_v<T, half>, half2, float2>>;

    const float absmax_offset_f = absmax_8bit ? __ldg(absmax_offset) : 0.0f;

    const int blocksize_log2 = __ffs(blocksize) - 1;

    constexpr int NUM_VAL = 32;            // 4-bit elements per lane per outer K step
    constexpr int K_STRIDE = 32 * NUM_VAL; // = 1024 K elements per outer step

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int base_m = blockIdx.y * M_BLOCK;

    // HIP bf16 uses the native v_dot2_f32_bf16 dot product with an LDS centroid
    // LUT; all other dtype/platform combinations keep the warp-shuffle + pair-mul
    // path. Compile-time so the unused paths are dropped.
    constexpr bool HIP_BF16_VDOT2 = BNB_HIP && std::is_same_v<T, bnb_bfloat16>;

    // Stage the fp16/fp32 centroid LUT in LDS instead of warp shuffle (ds_bpermute):
    // fp16 on all RDNA, fp32 only on RDNA3 (BNB_HIP_FP32_LDS_LUT). bf16 uses the
    // VDOT2 LDS LUT above.
    [[maybe_unused]] constexpr bool HIP_LDS_LUT_SHFL =
        BNB_HIP && !HIP_BF16_VDOT2 && (std::is_same_v<T, half> || (std::is_same_v<T, float> && BNB_HIP_FP32_LDS_LUT));

    // Each lane loads its LUT entry (lane_id < 16) for warp-shuffle dequant.
    // bf16/fp16: centroid as uint16-in-uint32 for __shfl_sync over uint32.
    // fp32: centroid as float for __shfl_sync over float.
    // (The VDOT2 path uses the shared LDS LUT below instead of my_lut_u32.)
    [[maybe_unused]] uint32_t my_lut_u32 = 0u;
    [[maybe_unused]] float my_lut_f32_shfl = 0.0f;

    if (lane_id < 16) {
        const float* lut = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
        if constexpr (std::is_same_v<T, float>)
            my_lut_f32_shfl = lut[lane_id];
        else if constexpr (!HIP_BF16_VDOT2) {
            const T centroid = static_cast<T>(lut[lane_id]);
            my_lut_u32 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&centroid));
        }
    }

    // Centroid LUTs in LDS, populated by all warps before the warp_n >= N early-out
    // so every warp reaches the barrier. bf16 -> quant_map_bf16 (VDOT2 path);
    // fp16/fp32 -> quant_map_shfl (fp16 = uint16 bits, fp32 = float). The
    // HIP_BF16_VDOT2 / HIP_LDS_LUT_SHFL gates select which dtype/arch passes use them.
#if BNB_HIP
    using shfl_lut_t = std::conditional_t<std::is_same_v<T, float>, float, uint16_t>;
    __shared__ uint16_t quant_map_bf16[16];
    __shared__ shfl_lut_t quant_map_shfl[16];
    if constexpr (HIP_BF16_VDOT2) {
        if (lane_id < 16) {
            const float* lut = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
            const bnb_bfloat16 c = static_cast<bnb_bfloat16>(lut[lane_id]);
            quant_map_bf16[lane_id] = *reinterpret_cast<const uint16_t*>(&c);
        }
        __syncthreads();
    }

    if constexpr (HIP_LDS_LUT_SHFL) {
        if (lane_id < 16) {
            const float* lut = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
            if constexpr (std::is_same_v<T, float>) {
                quant_map_shfl[lane_id] = lut[lane_id];
            } else {
                const T c = static_cast<T>(lut[lane_id]);
                quant_map_shfl[lane_id] = *reinterpret_cast<const uint16_t*>(&c);
            }
        }
        __syncthreads();
    }
#endif
    if (warp_n >= N)
        return;

    const int blk_per_row = K >> blocksize_log2;

    // Per-M accumulators. M_BLOCK is compile-time so the loop fully unrolls.
    float acc[M_BLOCK];
#pragma unroll
    for (int m = 0; m < M_BLOCK; m++)
        acc[m] = 0.f;

    const int m_valid = min(M_BLOCK, max(0, M - base_m));

    // All 32 lanes run the same number of K iterations for warp-shuffle convergence.
    // Inactive lanes load b_packed4={0}, which decodes to 0 and contributes nothing.
    const int num_groups = (K + K_STRIDE - 1) / K_STRIDE;

#if BNB_HIP
    // Per-lane fp32 scale for the K group at inner_k (single-level or nested absmax).
    auto compute_scale = [&](int ik) -> float {
        const int blk_idx = warp_n * blk_per_row + (ik >> blocksize_log2);
        if (absmax_8bit)
            return __ldg(&absmax_code[absmax_8bit[blk_idx]]) * __ldg(&absmax[blk_idx >> 8]) + absmax_offset_f;
        return __ldg(&absmax[blk_idx]);
    };

    // Register prefetch of the packed B + scale for the next K group, overlapping
    // the global loads with the current group's decode/accumulate.
    int pref_inner_k = lane_id * NUM_VAL;
    bool pref_lane_active = (pref_inner_k < K);
    float pref_scale_f = 0.0f;
    uint4 pref_b_packed4 = {0u, 0u, 0u, 0u};
    if (pref_lane_active) {
        pref_scale_f = compute_scale(pref_inner_k);
        pref_b_packed4 =
            *reinterpret_cast<const uint4*>(reinterpret_cast<const uint32_t*>(B + warp_n * (K / 2) + pref_inner_k / 2));
    }
#endif

    for (int g = 0; g < num_groups; g++) {
        const int inner_k = g * K_STRIDE + lane_id * NUM_VAL;

#if BNB_HIP
        const bool lane_active = pref_lane_active;
        const float scale_f = pref_scale_f;
        const uint4 b_packed4 = pref_b_packed4;

        // Advance the prefetch to the next group.
        const int next_inner_k = (g + 1) * K_STRIDE + lane_id * NUM_VAL;
        pref_lane_active = (g + 1 < num_groups) && (next_inner_k < K);
        pref_scale_f = 0.0f;
        pref_b_packed4 = {0u, 0u, 0u, 0u};
        if (pref_lane_active) {
            pref_scale_f = compute_scale(next_inner_k);
            pref_b_packed4 = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const uint32_t*>(B + warp_n * (K / 2) + next_inner_k / 2)
            );
        }
#else
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
#endif
        const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b_packed4);

        // Decode B and accumulate.
        // HIP bf16:   bf16 LDS-LUT centroids, native v_dot2_f32_bf16, scale once per sub.
        // bf16/fp16:  uint16-in-uint32 LUT, hmul2 vector math, 1x uint4 A load per sub-iter.
        // fp32:       float LUT, scalar multiply, 2x uint4 A loads per sub-iter.
        [[maybe_unused]] T2 scale_x2{};
        if constexpr (!std::is_same_v<T, float> && !HIP_BF16_VDOT2)
            scale_x2 = broadcast_vec2<T>(scale_f);

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            // Decode 4 B bytes (8 nibbles) into 8 dequantized values.
            // hi nibble (>>4) = lower K index, lo nibble (&0xf) = higher K index.
            [[maybe_unused]] T2 b_chunk[4];
            [[maybe_unused]] float b_dq[8];
#if BNB_HIP
            // VDOT2 centroid pairs (no scale baked in); scale applied once below.
            using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));
            [[maybe_unused]] bf16x2_t b_chunk_d[4];
#endif
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint8_t byte = b_bytes[sub * 4 + j];
                if constexpr (std::is_same_v<T, float>) {
                    if constexpr (HIP_LDS_LUT_SHFL) {
                        b_dq[j * 2] = quant_map_shfl[byte >> 4] * scale_f;
                        b_dq[j * 2 + 1] = quant_map_shfl[byte & 0x0f] * scale_f;
                    } else {
                        b_dq[j * 2] = __shfl_sync(BNB_FULL_WARP_MASK, my_lut_f32_shfl, byte >> 4, 32) * scale_f;
                        b_dq[j * 2 + 1] = __shfl_sync(BNB_FULL_WARP_MASK, my_lut_f32_shfl, byte & 0x0f, 32) * scale_f;
                    }
                } else if constexpr (HIP_BF16_VDOT2) {
#if BNB_HIP
                    const uint16_t hi16 = quant_map_bf16[byte >> 4];
                    const uint16_t lo16 = quant_map_bf16[byte & 0x0f];
                    b_chunk_d[j] =
                        bf16x2_t{*reinterpret_cast<const __bf16*>(&hi16), *reinterpret_cast<const __bf16*>(&lo16)};
#endif
                } else {
                    // fp16 on HIP uses the LDS LUT; fp16/bf16 on CUDA use the shuffle.
                    uint32_t hi, lo;
                    if constexpr (HIP_LDS_LUT_SHFL) {
                        hi = quant_map_shfl[byte >> 4];
                        lo = quant_map_shfl[byte & 0x0f];
                    } else {
                        hi = __shfl_sync(BNB_FULL_WARP_MASK, my_lut_u32, byte >> 4, 32);
                        lo = __shfl_sync(BNB_FULL_WARP_MASK, my_lut_u32, byte & 0x0f, 32);
                    }
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
                    } else if constexpr (HIP_BF16_VDOT2) {
#if BNB_HIP
                        // Native v_dot2_f32_bf16 over 8 K elements (4 pairs),
                        // accumulating in fp32; multiply by the scale once. This
                        // avoids the bf16-multiply round-to-nearest-even emulation on RDNA.
                        const uint4 a_packed4 = *reinterpret_cast<const uint4*>(&A[m_global * K + a_k]);
                        const bf16x2_t* a_pairs_d = reinterpret_cast<const bf16x2_t*>(&a_packed4);
                        float partial = 0.0f;
#pragma unroll
                        for (int k = 0; k < 4; k++)
                            partial = __builtin_amdgcn_fdot2_f32_bf16(a_pairs_d[k], b_chunk_d[k], partial, false);
                        acc[m] += partial * scale_f;
#endif
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
        acc[m] = simt_warp_reduce_sum(acc[m]);
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
/// @tparam T Input/output dtype (`bnb_bfloat16`, `half`, or `float`)
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
    bnb_stream_t stream           // CUDA/HIP stream
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
template void launch_gemm_4bit_simt<bnb_bfloat16>(
    const bnb_bfloat16*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, bnb_bfloat16*,
    const bnb_bfloat16*, int, int, int, int, int, bnb_stream_t
);
template void launch_gemm_4bit_simt<half>(
    const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int, int,
    int, int, int, bnb_stream_t
);
template void launch_gemm_4bit_simt<float>(
    const float*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, float*, const float*, int,
    int, int, int, int, bnb_stream_t
);
