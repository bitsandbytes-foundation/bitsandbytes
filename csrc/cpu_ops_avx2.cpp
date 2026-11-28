// AVX2 kernels for x86-64 CPUs without AVX-512. Compiled without the -mavx512*
// flags (see CMakeLists.txt) so the compiler cannot emit EVEX-encoded instructions
// that would SIGILL on those CPUs; a target attribute is not enough because the
// <immintrin.h> wrappers inherit the file-wide target and then fail to inline.
#include "cpu_ops.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)

#if defined(_MSC_VER)
#define BNB_NOINLINE __declspec(noinline)
#else
#define BNB_NOINLINE __attribute__((noinline))
#endif

namespace {

constexpr float fp4_lut[16] = {
    0.0f,  0.005208333333f,  0.66666667f,  1.0f,  0.33333333f,  0.5f,  0.16666667f,  0.25f,
    -0.0f, -0.005208333333f, -0.66666667f, -1.0f, -0.33333333f, -0.5f, -0.16666667f, -0.25f,
};
constexpr float nf4_lut[16] = {
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

// 16-entry table lookup: permutevar8x32 uses the low 3 index bits, bit 3 selects the half.
inline __m256 lut16_lookup(__m256i idx, __m256 lut_lo, __m256 lut_hi) {
    __m256 lo = _mm256_permutevar8x32_ps(lut_lo, idx);
    __m256 hi = _mm256_permutevar8x32_ps(lut_hi, idx);
    __m256 select_hi = _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28));
    return _mm256_blendv_ps(lo, hi, select_hi);
}

// fp32 -> bf16 (round-to-nearest-even, NaN -> 0xffff as in the AVX-512 path), low 16 bits of each lane.
inline __m256i cvt_fp32_to_bf16_epi32(__m256 src) {
    __m256i value = _mm256_castps_si256(src);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(value, 16), _mm256_set1_epi32(1));
    __m256i rounding_bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff));
    __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(value, rounding_bias), 16);
    __m256 ordered = _mm256_cmp_ps(src, src, _CMP_ORD_Q);
    __m256i nan = _mm256_set1_epi32(0xffff);
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(nan), _mm256_castsi256_ps(rounded), ordered));
}

template <typename T> inline void store16(T* pout, __m256 va, __m256 vb) {
    if constexpr (std::is_same<T, float>::value) {
        _mm256_storeu_ps(pout, va);
        _mm256_storeu_ps(pout + 8, vb);
    } else if constexpr (std::is_same<T, bf16_t>::value) {
        __m256i a = cvt_fp32_to_bf16_epi32(va);
        __m256i b = cvt_fp32_to_bf16_epi32(vb);
        // packus works per 128-bit lane; restore element order.
        __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pout), packed);
    } else {
        __m128i a = _mm256_cvtps_ph(va, _MM_FROUND_TO_NEAREST_INT);
        __m128i b = _mm256_cvtps_ph(vb, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pout), _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1));
    }
}

// Scalar tail store with the same rounding as the vector path.
template <typename T> inline void store1(T* pout, float v) {
    if constexpr (std::is_same<T, float>::value) {
        *pout = v;
    } else if constexpr (std::is_same<T, bf16_t>::value) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        if (v != v) {
            *pout = bf16_t{static_cast<uint16_t>(0xffff)};
        } else {
            uint32_t r = bits + 0x7fff + ((bits >> 16) & 1);
            *pout = bf16_t{static_cast<uint16_t>(r >> 16)};
        }
    } else {
        __m128i h = _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT);
        *pout = fp16_t{static_cast<uint16_t>(_mm_extract_epi16(h, 0))};
    }
}

} // namespace

// Blocks are indexed over the flattened tensor like the scalar path; 16 values per
// iteration with a scalar tail. noinline keeps LTO from inlining this into an
// AVX-512 caller, which would re-open the EVEX problem.
template <typename T, int DATA_TYPE>
BNB_NOINLINE void
    dequantize_4bit_avx2(const unsigned char* A, const float* absmax, T* out, long long blocksize, long long total) {
    const float* lut = DATA_TYPE == 1 ? fp4_lut : nf4_lut;
    const __m256 lut_lo = _mm256_loadu_ps(lut);
    const __m256 lut_hi = _mm256_loadu_ps(lut + 8);
    const __m128i mask4 = _mm_set1_epi8(0x0f);

    BNB_OMP_PARALLEL_FOR
    for (long long block_idx = 0; block_idx < total; block_idx += blocksize) {
        const long long valid_items = (total - block_idx >= blocksize) ? blocksize : total - block_idx;
        const float scale = absmax[block_idx / blocksize];
        const __m256 vscale = _mm256_set1_ps(scale);
        const unsigned char* pin = A + (block_idx >> 1);
        T* pout = out + block_idx;

        long long i = 0;
        for (; i + 16 <= valid_items; i += 16) {
            // 8 packed bytes -> 16 nibble indices, high nibble first
            __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pin + (i >> 1)));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
            __m128i lo = _mm_and_si128(raw, mask4);
            __m128i idx16 = _mm_unpacklo_epi8(hi, lo);
            __m256i idx_a = _mm256_cvtepu8_epi32(idx16);
            __m256i idx_b = _mm256_cvtepu8_epi32(_mm_srli_si128(idx16, 8));
            __m256 va = _mm256_mul_ps(lut16_lookup(idx_a, lut_lo, lut_hi), vscale);
            __m256 vb = _mm256_mul_ps(lut16_lookup(idx_b, lut_lo, lut_hi), vscale);
            store16(pout + i, va, vb);
        }
        for (; i < valid_items; i += 2) {
            unsigned char byte = pin[i >> 1];
            store1(pout + i, lut[byte >> 4] * scale);
            if (i + 1 < valid_items) {
                store1(pout + i + 1, lut[byte & 0x0F] * scale);
            }
        }
    }
}

template void dequantize_4bit_avx2<float, FP4>(const unsigned char*, const float*, float*, long long, long long);
template void dequantize_4bit_avx2<float, NF4>(const unsigned char*, const float*, float*, long long, long long);
template void dequantize_4bit_avx2<bf16_t, FP4>(const unsigned char*, const float*, bf16_t*, long long, long long);
template void dequantize_4bit_avx2<bf16_t, NF4>(const unsigned char*, const float*, bf16_t*, long long, long long);
template void dequantize_4bit_avx2<fp16_t, FP4>(const unsigned char*, const float*, fp16_t*, long long, long long);
template void dequantize_4bit_avx2<fp16_t, NF4>(const unsigned char*, const float*, fp16_t*, long long, long long);

#endif // (__x86_64__ || _M_X64) && __AVX2__
