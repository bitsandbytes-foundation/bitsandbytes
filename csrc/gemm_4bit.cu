// C API for the custom 4-bit GEMM.
// Dispatches to SIMT or MMA kernels based on GPU architecture and shape.
// Computes out[M, N] = A[M, K] @ B[N, K]^T. All pointers are device memory.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "gemm_4bit_simt.cuh"
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

/// @brief Fused 4-bit dequantize + GEMM. Computes out[M,N] = A[M,K] @ B[N,K]^T + bias.
///
/// Dispatches to SIMT (sm60+) or MMA (sm75 fp16, sm80+ bf16/fp16) based on GPU arch and shape.
/// fp32 always uses SIMT. Supports single-level and double-quantized (nested) absmax.
///
/// @tparam T Input/output dtype (`__nv_bfloat16`, `half`, or `float`)
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
    cudaStream_t stream           // CUDA stream
    // clang-format on
) {
    constexpr bool is_fp32 = std::is_same_v<T, float>;

    // fp32 and M<=3 are always SIMT regardless of GPU -- skip the props lookup.
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

    // sm89 (>=60 SMs) and sm120 (>=48 SMs): at M<=6 with wide N, SIMT saturates
    // bandwidth more efficiently than blocked MMA.
    const bool wide_n_simt =
        M <= 6 && mma_blocks >= num_sms &&
        ((cc_maj == 8 && cc_min == 9 && num_sms >= 60) || (cc_maj == 12 && cc_min == 0 && num_sms >= 48));

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
    const __nv_bfloat16* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, __nv_bfloat16* out, const __nv_bfloat16* bias, int M, int N, int K, int blocksize,
    int quant_type, cudaStream_t stream
) {
    gemm_4bit<__nv_bfloat16>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

void cgemm_4bit_fp16(
    const half* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, half* out, const half* bias, int M, int N, int K, int blocksize, int quant_type,
    cudaStream_t stream
) {
    gemm_4bit<half>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

void cgemm_4bit_fp32(
    const float* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, float* out, const float* bias, int M, int N, int K, int blocksize, int quant_type,
    cudaStream_t stream
) {
    gemm_4bit<float>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream
    );
}

} // extern "C"
