// C API for the custom 4-bit GEMM.
// Dispatches to SIMT, MMA (CUDA) or WMMA (ROCm RDNA3/RDNA4) kernels based on GPU architecture and shape.
// Computes out[M, N] = A[M, K] @ B[N, K]^T. All pointers are device memory.

#include <climits>
#include <cstdint>
#include <type_traits>

#include "compat.cuh"
#include "gemm_4bit_simt.cuh"
#if BNB_HIP
#include "gemm_4bit_wmma.cuh"
#endif
#if defined(BNB_HAS_GEMM4BIT_SM75) || defined(BNB_HAS_GEMM4BIT_SM80)
// Included only when an MMA-capable arch (sm75/sm80+) is in the build.
#include "gemm_4bit_common.cuh" // GpuProps, used only by the MMA dispatch below
#include "gemm_4bit_sm75.cuh"
#include "gemm_4bit_sm80.cuh"

// 16-entry cache indexed by device ID. num_sms==0 means not yet populated.
// Static storage is zero-initialized, so all entries start unpopulated (num_sms==0).
GpuProps get_gpu_props() {
    static GpuProps cache[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);

    if (dev < 16 && cache[dev].num_sms != 0)
        return cache[dev];

    GpuProps props = {};
    props.device_index = dev;
    cudaDeviceGetAttribute(&props.num_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&props.cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&props.cc_minor, cudaDevAttrComputeCapabilityMinor, dev);

    if (dev < 16)
        cache[dev] = props;

    return props;
}
#endif

/// @brief Fused 4-bit dequantize + GEMM. Computes out[M,N] = A[M,K] @ B[N,K]^T + bias.
///
/// Dispatches to SIMT (sm60+, all ROCm), MMA (sm75 fp16, sm80+ bf16/fp16) or WMMA (RDNA3/RDNA4
/// bf16/fp16/fp32) based on GPU arch and shape. fp32 uses SIMT except on RDNA3/RDNA4. Supports
/// single-level and double-quantized (nested) absmax.
///
/// @tparam T Input/output dtype (`bnb_bfloat16`, `half`, or `float`)
template <typename T>
static void gemm_4bit(
    // clang-format off
    const T*       A,             // inputs [M, K]
    const uint8_t* B,             // packed 4-bit weights [N, K/2]
    const float*   absmax,        // fp32 absmax [N*K/blocksize] or [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* absmax_8bit,   // [N*K/blocksize] uint8 compressed absmax; nullptr = non-nested
    const float*   absmax_code,   // [256] codebook for 8bit absmax
    const float*   absmax_offset, // scalar; nullptr = non-nested
    T*             out,           // [M, N]
    const T*       bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,          // problem shape
    int blocksize,                // elements per quantization block
    int quant_type,               // 1 = FP4, 2 = NF4
    bnb_stream_t stream           // CUDA/HIP stream
    // clang-format on
) {
    constexpr bool is_fp32 = std::is_same_v<T, float>;

#if BNB_HIP
    // WMMA needs K % 64 == 0 and at least 32-element quantization blocks.
    // RDNA3 (gfx110x) and RDNA3.5 (gfx115x, gfx11 kernels): on the 40 calibrated LLM shapes
    // WMMA beats SIMT for bf16/fp16 at every M (gfx1100: at M=1 by ~8% in total time, mostly
    // on the larger weights).
    // RDNA4 (gfx1201): each WMMA covers 16 rows (gfx11: 8), so SIMT keeps the smallest M.
    // On gfx1201 WMMA is 7-10% faster than SIMT in total time over the 40 shapes from M=2
    // for bf16/fp16 (2-3% slower at M=1).
    // fp32 (six split-bf16 WMMAs per tile) picks by M and N, see below.
    constexpr int GFX12_MIN_M_16BIT = 2;
    const int wmma_arch = K % 64 == 0 && blocksize >= 32 ? gemm_4bit_wmma_arch() : 0;
    bool use_wmma =
        (wmma_arch == 11 || wmma_arch == 113 || wmma_arch == 115) || (wmma_arch == 12 && M >= GFX12_MIN_M_16BIT);
    if (wmma_arch != 0 && is_fp32) {
        // fp32: with few output columns the split-bf16 WMMA tiles leave too few blocks to
        // stream the weights, so SIMT stays faster up to a few rows (gfx12's 16-row tiles: a
        // few dozen). The first row with M <= max_m decides: WMMA iff N >= min_n; past the
        // last row WMMA always. Fitted per target from SIMT and WMMA timings at every M up
        // to 32:
        // - gfx1100 (W7900, pinned clocks): within 2.6% of the per-shape faster kernel in total
        //   time at M = 4-5 and within 1.7% from M = 6 (SIMT through M = 5 was 1.10x slower at
        //   M = 5); WMMA at M = 4 lost to SIMT on tall-K shapes by up to 1.34x.
        // - gfx1201 (R9700, pinned clocks): within 0.8% in total at M = 5-32; single shapes up
        //   to 1.14x slower near the N cutoffs (WMMA alone was up to 1.6x slower at M = 6).
        struct Cutoff {
            int max_m, min_n;
        };

        // Narrow weights stay on SIMT longer (dense M scan, pinned clocks): with N <= 1024 the
        // split-bf16 WMMA tiles leave too few blocks, up to 2.4x slower than SIMT at M = 52.
        static constexpr Cutoff gfx11_fp32_wmma[] = {{4, INT_MAX}, {5, 3072},  {7, 2816},
                                                     {8, 2048},    {12, 2816}, {64, 1024}};
        static constexpr Cutoff gfx115_fp32_wmma[] = {{3, INT_MAX}, {5, 2560}, {7, 1024}};
        // gfx1103 (Radeon 760M, 8 CUs, pinned clocks): SIMT through M = 2, WMMA from M = 3 (by N
        // through M = 4): 1.12x faster in total at M = 3 and 1.29x at M = 4 than the gfx1100 table,
        // which kept SIMT through M = 4. WMMA for wide N at M = 2 measured 2% slower.
        static constexpr Cutoff gfx1103_fp32_wmma[] = {{2, INT_MAX}, {4, 3840}};
        static constexpr Cutoff gfx12_fp32_wmma[] = {{4, INT_MAX}, {5, 8192},  {6, 4096},
                                                     {9, 2048},    {15, 1408}, {31, 1024}};
        auto pick = [&](const auto& table) {
            for (const auto& row : table)
                if (M <= row.max_m)
                    return N >= row.min_n;
            return true;
        };
        use_wmma = wmma_arch == 12    ? pick(gfx12_fp32_wmma)
                   : wmma_arch == 115 ? pick(gfx115_fp32_wmma)
                   : wmma_arch == 113 ? pick(gfx1103_fp32_wmma)
                                      : pick(gfx11_fp32_wmma);
    }
    if (use_wmma && launch_gemm_4bit_wmma<T>(
                        wmma_arch, A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize,
                        quant_type, stream
                    ))
        return;
#endif

    // Otherwise fp32 and M<=3 are always SIMT -- skip the props lookup.
    if (is_fp32 || M <= 3) {
        launch_gemm_4bit_simt<T>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
        );
        return;
    }

#if defined(BNB_HAS_GEMM4BIT_SM75) || defined(BNB_HAS_GEMM4BIT_SM80)
    const GpuProps gpu = get_gpu_props();
    const int num_sms = gpu.num_sms;
    const int cc_maj = gpu.cc_major;
    const int cc_min = gpu.cc_minor;

    const bool hbm_arch = (cc_maj == 8 && cc_min == 0) || cc_maj == 9 || cc_maj == 10;
    const bool gddr_arch = !hbm_arch && cc_maj >= 8;
    const int mma_blocks = ((M + 31) / 32) * ((N + 63) / 64);

    // sm86/sm89/sm120 with >= 48 SMs: high GDDR bandwidth means SIMT wins at M=4.
    const bool highbw_gddr = (cc_maj == 8 && (cc_min == 6 || cc_min == 9) && num_sms >= 48) ||
                             (cc_maj == 12 && cc_min == 0 && num_sms >= 48);

    // Below 2/3-wave at M<=8: SIMT keeps more warps in flight.
    const bool undersubscribed =
        (M <= 8 && mma_blocks * 3 <= num_sms * 2) || (hbm_arch && M == 4 && mma_blocks <= num_sms);

    // sm89 (>=60 SMs) and sm120/sm121 (>=48 SMs): at M<=6 with wide N, SIMT saturates
    // bandwidth more efficiently than blocked MMA.
    const bool wide_n_simt =
        M <= 6 && mma_blocks >= num_sms &&
        ((cc_maj == 8 && cc_min == 9 && num_sms >= 60) || (cc_maj == 12 && cc_min == 0 && num_sms >= 48) ||
         (cc_maj == 12 && cc_min == 1 && num_sms >= 48));

    // GDDR tall-K (K>N): K-loop too long relative to output tile at small M.
    const bool tall_k_simt = gddr_arch && K > N && M <= 17 && mma_blocks * 3 < num_sms;

    const bool use_simt = (M == 4 && highbw_gddr) || undersubscribed || wide_n_simt || tall_k_simt ||
                          (M <= 16 && mma_blocks * 4 <= num_sms) || (M <= 32 && mma_blocks * 8 <= num_sms) ||
                          (K % 64 != 0); // MMA requirement

    if (!use_simt) {
#if defined(BNB_HAS_GEMM4BIT_SM80)
        if (cc_maj >= 8) {
            if constexpr (!is_fp32) {
                launch_gemm_4bit_sm80_m16n8k16<T>(
                    A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type,
                    gpu, stream
                );
                return;
            }
        }
#endif
#if defined(BNB_HAS_GEMM4BIT_SM75)
        if (cc_maj == 7 && cc_min >= 5) {
            // bf16 has no sm75 tensor core support; falls through to SIMT.
            if constexpr (std::is_same_v<T, half>) {
                launch_gemm_4bit_sm75_m16n8k8<T>(
                    A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type,
                    gpu, stream
                );
                return;
            }
        }
#endif
    }
#endif // BNB_HAS_GEMM4BIT_SM75 || BNB_HAS_GEMM4BIT_SM80

    launch_gemm_4bit_simt<T>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

extern "C" {

void cgemm_4bit_bf16(
    const bnb_bfloat16* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, bnb_bfloat16* out, const bnb_bfloat16* bias, int M, int N, int K, int blocksize,
    int quant_type, bnb_stream_t stream
) {
    gemm_4bit<bnb_bfloat16>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

void cgemm_4bit_fp16(
    const half* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, half* out, const half* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
) {
    gemm_4bit<half>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

void cgemm_4bit_fp32(
    const float* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, float* out, const float* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
) {
    gemm_4bit<float>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

} // extern "C"
