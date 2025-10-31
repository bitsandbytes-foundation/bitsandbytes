#include <BinSearch.h>
#include <common.h>
#include <cpu_ops.h>
#include <thread>

using namespace BinSearch;

#define __AVX512F__

#if defined(__AVX512F__)
#include <immintrin.h>

inline __m256i cvt_fp32_to_fp16(const __m512 src) {
    return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
    #if defined(__AVX512BF16__)
      return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
    #else
      __m512i value = _mm512_castps_si512(src);
      __m512i nan = _mm512_set1_epi32(0xffff);
      auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
      __m512i ones = _mm512_set1_epi32(0x1);
      __m512i vec_bias = _mm512_set1_epi32(0x7fff);
      // uint32_t lsb = (input >> 16) & 1;
      auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
      // uint32_t rounding_bias = 0x7fff + lsb;
      t_value = _mm512_add_epi32(t_value, vec_bias);
      // input += rounding_bias;
      t_value = _mm512_add_epi32(t_value, value);
      // input = input >> 16;
      t_value = _mm512_srli_epi32(t_value, 16);
      // Check NaN before converting back to bf16
      t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
      return _mm512_cvtusepi32_epi16(t_value);
    #endif
}

static inline __m512 set_nf4_lut() {
    return _mm512_set_ps(
        1.0f,
        0.7229568362236023,
        0.5626170039176941,
        0.44070982933044434,
        0.33791524171829224,
        0.24611230194568634,
        0.16093020141124725,
        0.07958029955625534,
        0.0f,
        -0.09105003625154495,
        -0.18477343022823334,
        -0.28444138169288635,
        -0.39491748809814453,
        -0.5250730514526367,
        -0.6961928009986877,
        -1.0f);
}
static inline __m512 set_fp4_lut() {
    return _mm512_set_ps(
        0.0000f,
        5.208333333e-03f,
        0.66666667f,
        1.0000f,
        0.33333333f,
        0.5000f,
        0.16666667f,
        0.2500f,
        0.0000f,
        -5.208333333e-03f,
        -0.66666667f,
        -1.0000f,
        -0.33333333f,
        -0.5000f,
        -0.16666667f,
        -0.2500f);
}
#endif

// 4-bit (FP4 / NF4) dequantization helper extracted from the original else branch.
// DATA_TYPE: 1 = FP4, 0 = NF4
template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(unsigned char* A,
                                const float* absmax,
                                T* out,
                                long long blocksize,
                                long long m,
                                long long n) {
    static_assert(DATA_TYPE == 0 || DATA_TYPE == 1,
                  "dequantizeBlockwise4bitCpu called with non 4-bit DATA_TYPE");
    if (blocksize <= 0 || m < 0 || n <= 0) return;

#if defined(__AVX512F__)
    long long dim_0 = m;
    long long dim_1 = n;
    long long input_dim_1 = dim_1 >> 1;
    long long absmax_dim_1 = dim_1 / blocksize;
    using Tcomp = float;
    constexpr auto VEC_LEN = sizeof(__m512i) / sizeof(Tcomp); // 16
    if (dim_1 % VEC_LEN == 0 && blocksize >= VEC_LEN) {
        __m512 lut = DATA_TYPE == 1 ? set_fp4_lut() : set_nf4_lut();
        constexpr auto k_step = VEC_LEN / 2; // 8
        #pragma omp parallel for
        for (int block_idx = 0; block_idx < dim_0; ++block_idx) {
            for (int k = 0; k < input_dim_1; k += k_step) {
                // Load 64 bits of nf4 data and a single scale data
                uint8_t* p = &A[block_idx * input_dim_1 + k];
                uint64_t packed;
                std::memcpy(&packed, p, sizeof(uint64_t));
                auto scale_idx = k * 2 / blocksize;
                auto vscales = _mm512_set1_ps((float)absmax[block_idx * absmax_dim_1 + scale_idx]);
                // unpack nf4 data to 32-bit integers
                uint64_t high = 0;
                uint64_t low = 0;
                for (int i = 0; i < 4; ++i) {
                    low |= ((packed >> (2*i * 4)) & 0xf) << ((2*i+1) * 8);
                    low |= ((packed >> ((2*i+1) * 4)) & 0xf) << (2*i * 8);
                    high |= ((packed >> (2*i * 4 + 32)) & 0xf) << ((2*i+1) * 8);
                    high |= ((packed >> ((2*i+1) * 4 + 32)) & 0xf) << (2*i * 8);
                }
                __m128i packed_128 = _mm_set_epi64x(high, low);
                __m512i vint32 = _mm512_cvtepu8_epi32(packed_128);
                // Table look-up
                __m512 vout = _mm512_permutexvar_ps(vint32, lut);
                // Apply scale
                vout = _mm512_mul_ps(vout, vscales);
                // Store results
                T* pout = &out[block_idx * dim_1 + k * 2];
                if constexpr (std::is_same<T, float>()) {
                _mm512_storeu_ps(pout, vout);
                } else if constexpr (std::is_same<T, bf16_t>()) {
                _mm256_storeu_si256(
                    (__m256i*)pout, cvt_fp32_to_bf16(vout));
                } else if constexpr (std::is_same<T, fp16_t>()) {
                _mm256_storeu_si256(
                    (__m256i*)pout, cvt_fp32_to_fp16(vout));
                }
            }
        }
    }
#else
    // Scalar fallback branch
    long long total = m * n;
    #pragma omp parallel for
    for (long long block_idx = 0; block_idx < total; block_idx += blocksize) {
        long long valid_items = (total - block_idx >= blocksize ? blocksize : total - block_idx);
        float scale = absmax[block_idx / blocksize];
        for (long long i = 0; i < valid_items; i += 2) {
            long long byte_index = (block_idx + i) >> 1;
            unsigned char byte = A[byte_index];

            // High nibble first (matches previous code logic)
            float v0 = (DATA_TYPE == 1 ? dDequantizeFP4(byte >> 4)
                                       : dDequantizeNF4(byte >> 4)) * scale;
            // Low nibble second
            float v1 = (DATA_TYPE == 1 ? dDequantizeFP4(byte & 0x0F)
                                       : dDequantizeNF4(byte & 0x0F)) * scale;

            if constexpr (std::is_same<T, bf16_t>::value) {
                out[block_idx + i] = float_to_bf16(v0);
            } else {
                out[block_idx + i] = static_cast<T>(v0);
            }

            if (i + 1 < valid_items) {
                if constexpr (std::is_same<T, bf16_t>::value) {
                    out[block_idx + i + 1] = float_to_bf16(v1);
                } else {
                    out[block_idx + i + 1] = static_cast<T>(v1);
                }
            }
        }
    }
#endif
}


template <typename T>
void dequantizeBlockwise8bitCpu(float* code,
                            unsigned char* A,
                            const float* absmax,
                            T* out,
                            long long blocksize,
                            long long n) {
    if (blocksize <= 0 || n <= 0) return;
    // 8-bit path
    #pragma omp parallel for
    for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
        long long valid_items = (n - block_idx >= blocksize ? blocksize : n - block_idx);
        long long block_end   = block_idx + valid_items;
        float scale = absmax[block_idx / blocksize];
        for (long long i = block_idx; i < block_end; ++i) {
            float v = code[A[i]] * scale;
            if constexpr (std::is_same<T, bf16_t>::value) {
                out[i] = float_to_bf16(v);
            } else {
                out[i] = static_cast<T>(v);
            }
        }
    }
}


void quantize_cpu(float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n) {

    // the default code is has range [-0.993, 1.0] which can cause an error in the binary search algorithm used below
    code[0] = -1.0f;

    long long num_blocks = n / blocksize;
    num_blocks += n % blocksize == 0 ? 0 : 1;

    const uint32 elements_code = 256;
    BinAlgo<Scalar, float, Direct2> bin_searcher(code, elements_code);

    int thread_wave_size = 256;
    // we chunk the threads into waves of 256 since the max limit is
    // between 16k and 64k on Linux (we reach this when running BLOOM-176B with a large batch size)
    for (long long offset = 0; offset < num_blocks; offset += thread_wave_size) {
        long long valid_chunks = num_blocks - offset >= thread_wave_size ? thread_wave_size : num_blocks - offset;
        std::vector<std::thread> threads(valid_chunks);
        std::vector<quantize_block_args> args(valid_chunks);

        int chunks_processed = 0;
        for (long long block_idx = offset * blocksize; block_idx < n; block_idx += blocksize) {
            long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
            long long block_end = block_idx + valid_items;

            struct quantize_block_args& arg = args[chunks_processed];
            arg.bin_searcher = &bin_searcher;
            arg.code = code;
            arg.A = A;
            arg.absmax = absmax;
            arg.out = out;
            arg.block_end = block_end;
            arg.block_idx = block_idx;
            arg.threadidx = block_idx / blocksize;
            arg.blocksize = blocksize;

            threads[chunks_processed] = std::thread([arg] { quantize_block(arg); });
            chunks_processed += 1;
            if (chunks_processed == valid_chunks) {
                break;
            }
        }

        for (int i = 0; i < valid_chunks; i++)
            threads[i].join();
    }
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void dequantizeBlockwise8bitCpu<float>(
    float* code, unsigned char* A, const float* absmax, float* out, long long blocksize, long long n);
template void dequantizeBlockwise8bitCpu<fp16_t>(
    float* code, unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long n);
template void dequantizeBlockwise8bitCpu<bf16_t>(
    float* code, unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long n);

template void dequantizeBlockwise4bitCpu<float, FP4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n);
template void dequantizeBlockwise4bitCpu<float, NF4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n);

template void dequantizeBlockwise4bitCpu<fp16_t, FP4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n);
template void dequantizeBlockwise4bitCpu<fp16_t, NF4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n);

template void dequantizeBlockwise4bitCpu<bf16_t, FP4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n);
template void dequantizeBlockwise4bitCpu<bf16_t, NF4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n);

// template void gemv_4bit_inference<fp16_t, 16>(
//     int m, int n, int k, fp16_t* A, unsigned char* B, float* absmax, float* datatype, fp16_t* out,
//     int lda, int ldb, int ldc, int blocksize);

// template void gemv_4bit_inference<bf16_t, 16>(
//     int m, int n, int k, bf16_t* A, unsigned char* B, float* absmax, float* datatype, bf16_t* out,
//     int lda, int ldb, int ldc, int blocksize);

// template void gemv_4bit_inference<float, 32>(
//     int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out,
//     int lda, int ldb, int ldc, int blocksize);
