#include <BinSearch.h>
#include <cpu_ops.h>
#include <thread>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef HAS_OPENMP
#include <omp.h>
#define BNB_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define BNB_OMP_PARALLEL_FOR
#endif

namespace {

constexpr int kCodebookSize = 256;

inline unsigned char lookup_code_index(const float* codebook, float value) {
    value = std::clamp(value, -1.0f, 1.0f);
    const float* begin = codebook;
    const float* end = codebook + kCodebookSize;
    const float* right = std::lower_bound(begin, end, value);
    if (right == begin) {
        return 0;
    }
    if (right == end) {
        return static_cast<unsigned char>(kCodebookSize - 1);
    }
    const float* left = right - 1;
    const float dist_left = std::fabs(value - *left);
    const float dist_right = std::fabs(*right - value);
    const unsigned char idx = static_cast<unsigned char>(right - begin);
    return dist_right < dist_left ? idx : idx - 1;
}

} // namespace

#if defined(__AVX512F__)
#include <immintrin.h>

inline __m256i cvt_fp32_to_fp16(const __m512 src) {
    return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if defined(__AVX512BF16__)
    if (has_avx512bf16()) {
        return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
    }
#endif
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
}

static inline __m512 set_nf4_lut() {
    return _mm512_set_ps(
        1.0f, 0.7229568362236023, 0.5626170039176941, 0.44070982933044434, 0.33791524171829224, 0.24611230194568634,
        0.16093020141124725, 0.07958029955625534, 0.0f, -0.09105003625154495, -0.18477343022823334,
        -0.28444138169288635, -0.39491748809814453, -0.5250730514526367, -0.6961928009986877, -1.0f
    );
}

static inline __m512 set_fp4_lut() {
    return _mm512_set_ps(
        -0.2500f, -0.16666667f, -0.5000f, -0.33333333f, -1.0000f, -0.66666667f, -5.208333333e-03f, 0.0000f, 0.2500f,
        0.16666667f, 0.5000f, 0.33333333f, 1.0000f, 0.66666667f, 5.208333333e-03f, 0.0000f
    );
}
#endif

// 4-bit (FP4 / NF4) dequantization helper extracted from the original else branch.
// DATA_TYPE: 1 = FP4, 2 = NF4
template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(
    unsigned char* A, const float* absmax, T* out, long long blocksize, long long m, long long n
) {
    static_assert(DATA_TYPE == 1 || DATA_TYPE == 2, "dequantizeBlockwise4bitCpu called with non 4-bit DATA_TYPE");
    if (blocksize <= 0 || m < 0 || n <= 0)
        return;

#if defined(__AVX512F__)
    if (has_avx512f()) {
        long long dim_0 = m;
        long long dim_1 = n;
        long long input_dim_1 = dim_1 >> 1;
        long long absmax_dim_1 = dim_1 / blocksize;
        using Tcomp = float;
        constexpr auto VEC_LEN = sizeof(__m512i) / sizeof(Tcomp); // 16
        if (dim_1 % VEC_LEN == 0 && blocksize >= VEC_LEN) {
            __m512 lut = DATA_TYPE == 1 ? set_fp4_lut() : set_nf4_lut();
            constexpr auto k_step = VEC_LEN / 2; // 8
            BNB_OMP_PARALLEL_FOR
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
                        low |= ((packed >> (2 * i * 4)) & 0xf) << ((2 * i + 1) * 8);
                        low |= ((packed >> ((2 * i + 1) * 4)) & 0xf) << (2 * i * 8);
                        high |= ((packed >> (2 * i * 4 + 32)) & 0xf) << ((2 * i + 1) * 8);
                        high |= ((packed >> ((2 * i + 1) * 4 + 32)) & 0xf) << (2 * i * 8);
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
                        _mm256_storeu_si256((__m256i*)pout, cvt_fp32_to_bf16(vout));
                    } else if constexpr (std::is_same<T, fp16_t>()) {
                        _mm256_storeu_si256((__m256i*)pout, cvt_fp32_to_fp16(vout));
                    }
                }
            }
            return;
        }
    }
#endif
    // Scalar fallback branch
    long long total = m * n;
    BNB_OMP_PARALLEL_FOR
    for (long long block_idx = 0; block_idx < total; block_idx += blocksize) {
        long long valid_items = (total - block_idx >= blocksize ? blocksize : total - block_idx);
        float scale = absmax[block_idx / blocksize];
        for (long long i = 0; i < valid_items; i += 2) {
            long long byte_index = (block_idx + i) >> 1;
            unsigned char byte = A[byte_index];

            // High nibble first (matches previous code logic)
            float v0 = (DATA_TYPE == 1 ? dDequantizeFP4(byte >> 4) : dDequantizeNF4(byte >> 4)) * scale;
            // Low nibble second
            float v1 = (DATA_TYPE == 1 ? dDequantizeFP4(byte & 0x0F) : dDequantizeNF4(byte & 0x0F)) * scale;

            if constexpr (std::is_same<T, bf16_t>::value) {
                out[block_idx + i] = float_to_bf16(v0);
            } else if constexpr (std::is_same<T, fp16_t>::value) {
                out[block_idx + i] = float_to_fp16(v0);
            } else {
                out[block_idx + i] = static_cast<T>(v0);
            }

            if (i + 1 < valid_items) {
                if constexpr (std::is_same<T, bf16_t>::value) {
                    out[block_idx + i + 1] = float_to_bf16(v1);
                } else if constexpr (std::is_same<T, fp16_t>::value) {
                    out[block_idx + i + 1] = float_to_fp16(v1);
                } else {
                    out[block_idx + i + 1] = static_cast<T>(v1);
                }
            }
        }
    }
}

template <typename T>
void dequantizeBlockwise8bitCpu(
    float* code, unsigned char* A, const float* absmax, T* out, long long blocksize, long long n
) {
    if (blocksize <= 0 || n <= 0)
        return;
    // 8-bit path
    BNB_OMP_PARALLEL_FOR
    for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
        long long valid_items = (n - block_idx >= blocksize ? blocksize : n - block_idx);
        long long block_end = block_idx + valid_items;
        float scale = absmax[block_idx / blocksize];
        for (long long i = block_idx; i < block_end; ++i) {
            float v = code[A[i]] * scale;
            if constexpr (std::is_same<T, bf16_t>::value) {
                out[i] = float_to_bf16(v);
            } else if constexpr (std::is_same<T, fp16_t>::value) {
                out[i] = float_to_fp16(v);
            } else {
                out[i] = static_cast<T>(v);
            }
        }
    }
}

void quantize_cpu(float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n) {

    if (blocksize <= 0 || n <= 0)
        return;

    // Ensure we cover the full expected dynamic range of the codebook.
    code[0] = -1.0f;

    const auto process_block = [&](long long block_start, long long block_end) {
        float absmax_block = 0.0f;
        for (long long i = block_start; i < block_end; ++i) {
            absmax_block = std::max(absmax_block, std::fabs(A[i]));
        }

        long long absmax_idx = block_start / blocksize;
        absmax[absmax_idx] = absmax_block;

        if (absmax_block == 0.0f) {
            std::fill(out + block_start, out + block_end, 0);
            return;
        }

        const float inv_absmax = 1.0f / absmax_block;
        for (long long i = block_start; i < block_end; ++i) {
            float normed_value = A[i] * inv_absmax;
            out[i] = lookup_code_index(code, normed_value);
        }
    };

    const long long num_blocks = (n + blocksize - 1) / blocksize;
    const int thread_wave_size = 256;

    // We chunk the threads into waves of 256 since the max limit is between 16k and 64k on Linux
    // (we reach this when running BLOOM-176B with a large batch size).
    for (long long offset = 0; offset < num_blocks; offset += thread_wave_size) {
        const long long wave_blocks = std::min<long long>(thread_wave_size, num_blocks - offset);
        std::vector<std::thread> threads;
        threads.reserve(wave_blocks);

        const long long first_block_start = offset * blocksize;
        for (long long b = 0; b < wave_blocks; ++b) {
            const long long block_start = first_block_start + b * blocksize;
            if (block_start >= n)
                break;
            const long long block_end = std::min(block_start + blocksize, n);
            threads.emplace_back(process_block, block_start, block_end);
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BF16__)

#define CVT_BF16_TO_FP32(a) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int DATA_TYPE> struct tinygemm_kernel_nn {
    static inline void apply(
        const scalar_t*, const unsigned char*, scalar_t*, const scalar_t*, int64_t, int, int64_t, int64_t, int64_t,
        int64_t, int64_t
    ) {
        static_assert(sizeof(scalar_t) == 0, "tinygemm_kernel_nn primary template should never be instantiated");
    }
};

template <int BLOCK_M, int BLOCK_N, int DATA_TYPE> struct tinygemm_kernel_nn<bf16_t, BLOCK_M, BLOCK_N, DATA_TYPE> {
    static inline void apply(
        const bf16_t* __restrict__ A, const unsigned char* __restrict__ B, bf16_t* __restrict__ C,
        const bf16_t* __restrict__ Bs, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
        int64_t strideBz, int64_t strideBs
    ) {
        static_assert(BLOCK_N % 32 == 0);
        constexpr int ROWS = BLOCK_M;      // 32
        constexpr int COLS = BLOCK_N / 16; // 2

        // prefetch distance
        constexpr int PREFETCH_SIZE_K = 16 * 4;

        __m512bh va;
        __m512bh vb[COLS];
        __m512 vc[ROWS * COLS];
        __m512 vc_master[ROWS * COLS];

        __m256i mask = _mm256_set1_epi8(0xF); // lower 4 bit
        __m256i fifteen = _mm256_set1_epi8(15);
        __m512i lut = DATA_TYPE == 1
                          ? _mm512_set_epi16(
                                0x0000, -0x4180, -0x41D5, -0x4100, -0x4155, -0x4080, -0x40D5, -0x4455, 0x0000, 0x3E80,
                                0x3E2B, 0x3F00, 0x3EAB, 0x3F80, 0x3F2B, 0x3BAB, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
                            )
                          : _mm512_set_epi16(
                                0x0000, 0x3F80, 0x3F39, 0x3F10, 0x3EE2, 0x3EAD, 0x3E7C, 0x3E25, 0x3DA3, 0x0000, -0x4246,
                                -0x41C3, -0x416E, -0x4136, -0x40FA, -0x40CE, -0x4080, 0x0000, 0x0000, 0x0000, 0x0000,
                                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
                            );
        __m512 scales[COLS];
        const int64_t K2 = K >> 1;
        const int64_t lda2 = lda >> 1;
        const int64_t ldb2 = ldb;            // ldb * 2 >> 1;
        const int64_t gs2 = group_size >> 1; // 64 / 2 = 32
        const float* a_ptr = reinterpret_cast<const float*>(A);

        auto loadc = [&](auto i) {
            constexpr int col = i % COLS;
            vc_master[i] = _mm512_set1_ps(0.f);
        };
        Unroll<ROWS * COLS>{}(loadc);

        auto pre_compute = [&](auto i, int64_t kgs) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = _mm512_set1_ps(0.f); // reset accumulator

            // load scales
            if constexpr (row == 0 && col % 2 == 0) {
                // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
                __m512i tmp = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(Bs + kgs * strideBs + col * 16));
                scales[col] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 0));
                scales[col + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 1));
            }
        };
        auto compute = [&](auto i, int64_t k) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
            }
            if constexpr (row == 0 && col % 2 == 0) {
                __m256i vb_u4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + k * ldb + col * 16));

                // deinterleave and lookup to BF16
                __m256i vb_i8_lo = vb_u4 & mask;
                __m256i vb_i8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
                vb_i8_lo = _mm256_add_epi8(vb_i8_lo, fifteen);
                vb_i8_hi = _mm256_add_epi8(vb_i8_hi, fifteen);
                vb[col] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_lo), lut);
                vb[col + 1] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_hi), lut);

                if constexpr (PREFETCH_SIZE_K > 0) {
                    _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
                }
            }
            vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
        };
        auto post_compute = [&](auto i, int64_t kgs) {
            vc_master[i] = _mm512_fmadd_ps(vc[i], scales[i % COLS], vc_master[i]);
        };
        for (int64_t k = 0; k < K2; k += gs2) {
            Unroll<ROWS * COLS>{}(pre_compute, k / gs2);
            for (int64_t k_offset = 0; k_offset < gs2; ++k_offset) {
                Unroll<ROWS * COLS>{}(compute, k + k_offset);
            }
            Unroll<ROWS * COLS>{}(post_compute, k / gs2);
        }

        auto storec = [&](auto i) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            if constexpr (col % 2 == 0) {
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(C + row * ldc + col * 16),
                    (__m512i)(_mm512_cvtne2ps_pbh(vc_master[i + 1], vc_master[i]))
                );
            }
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE, DATA_TYPE)                                                         \
    tinygemm_kernel_nn<scalar_t, MB_SIZE, NB_SIZE, DATA_TYPE>::apply(                                                  \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start, Bs + nb_start, K, group_size, lda, ldb, ldc,  \
        strideBz, strideBs                                                                                             \
    );

template <typename scalar_t, int DATA_TYPE>
void tinygemm_kernel(
    const scalar_t* __restrict__ A, const unsigned char* __restrict__ B, scalar_t* __restrict__ C,
    const scalar_t* __restrict__ Bs, scalar_t* __restrict__ Btmp, float* __restrict__ Ctmp, int64_t M, int64_t N,
    int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBz, int64_t strideBs
) {
    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 64;
    const int64_t MB = div_up(M, BLOCK_M);
    const int64_t NB = div_up(N, BLOCK_N);
    for (int mb = 0; mb < MB; ++mb) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(BLOCK_M, M - mb_start);
        for (int64_t nb = 0; nb < NB; ++nb) {
            int64_t nb_start = nb * BLOCK_N;
            int64_t nb_size = std::min(BLOCK_N, N - nb_start);

            switch (mb_size << 4 | nb_size >> 4) {
            // mb_size = 1
            case 0x12:
                LAUNCH_TINYGEMM_KERNEL_NN(1, 32, DATA_TYPE);
                break;
            case 0x14:
                LAUNCH_TINYGEMM_KERNEL_NN(1, 64, DATA_TYPE);
                break;
            // mb_size = 2
            case 0x22:
                LAUNCH_TINYGEMM_KERNEL_NN(2, 32, DATA_TYPE);
                break;
            case 0x24:
                LAUNCH_TINYGEMM_KERNEL_NN(2, 64, DATA_TYPE);
                break;
            // mb_size = 3
            case 0x32:
                LAUNCH_TINYGEMM_KERNEL_NN(3, 32, DATA_TYPE);
                break;
            case 0x34:
                LAUNCH_TINYGEMM_KERNEL_NN(3, 64, DATA_TYPE);
                break;
            // mb_size = 4
            case 0x42:
                LAUNCH_TINYGEMM_KERNEL_NN(4, 32, DATA_TYPE);
                break;
            case 0x44:
                LAUNCH_TINYGEMM_KERNEL_NN(4, 64, DATA_TYPE);
                break;
            default: {
                std::fprintf(
                    stderr, "[bitsandbytes] Unexpected block size %lldx%lld\n", (long long)mb_size, (long long)nb_size
                );
                std::abort(); // or return; if you prefer silent exit
            }
            }
        }
    }
}

template <typename T, int DATA_TYPE>
void gemv_4bit_inference(
    int64_t M, int64_t N, int64_t K, const T* __restrict__ x, const unsigned char* __restrict__ w,
    const T* __restrict__ absmax, T* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
) {
    constexpr int64_t BLOCK_M = block_size_m(); // 32
    constexpr int64_t BLOCK_N = block_size_n(); // 32
    const int64_t MB = div_up(M, BLOCK_M);      // （x + y -1）/ y, res = 1 when M <= 32
    const int64_t NB = div_up(N, BLOCK_N);
    // TODO: enable brgemm in the future.
    // const bool use_brgemm = M > 4;
    // const bool use_brgemm_dequant_out = M > 512;
    // T* Btmp_start = nullptr;
    // l2 cache block for n
    int64_t cache_blocks_nb = get_cache_blocks<T>(BLOCK_N * K);
    parallel_2d(MB, NB, [&](int64_t begin_mb, int64_t end_mb, int64_t begin_nb, int64_t end_nb) {
        // for brgemm, use float32 for accumulate
        alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
        alignas(64) T Btmp_inner[BLOCK_N * BLOCK_K]; // BLOCK_K = 128
        for (int64_t nbb = begin_nb; nbb < end_nb; nbb += cache_blocks_nb) {
            for (int64_t mb = begin_mb; mb < end_mb; ++mb) { // 0-1
                for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, end_nb); ++nb) {
                    int64_t mb_start = mb * BLOCK_M; // 0
                    int64_t mb_size = std::min(M - mb_start, BLOCK_M);
                    int64_t nb_start = nb * BLOCK_N;
                    int64_t nb_size = std::min(N - nb_start, BLOCK_N);
                    tinygemm_kernel<T, DATA_TYPE>(
                        /*   A  */ x + mb_start * x_stride,
                        /*   B  */ w + nb_start * K / 2, // divide by 2 since w is u4 packed in u8, K is w.size(1) * 2
                        /*   C  */ out + mb_start * out_stride + nb_start,
                        /*  Bs  */ absmax + nb_start,
                        /* Btmp */ Btmp_inner,
                        /* Ctmp */ Ctmp,
                        /*   M  */ mb_size,
                        /*   N  */ nb_size,
                        /*   K  */ K,
                        /*  gs  */ blocksize, // group_size
                        /* lda  */ x_stride,
                        /* ldb  */ nb_size,
                        /* ldc  */ out_stride,
                        /* sBz  */ N,
                        /* sBs  */ N
                    );
                }
            }
        }
        // if (use_brgemm) {
        //     at::native::cpublas::brgemm_release();
        // }
    });
}
#endif

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void dequantizeBlockwise8bitCpu<float>(
    float* code, unsigned char* A, const float* absmax, float* out, long long blocksize, long long n
);
template void dequantizeBlockwise8bitCpu<fp16_t>(
    float* code, unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long n
);
template void dequantizeBlockwise8bitCpu<bf16_t>(
    float* code, unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long n
);

template void dequantizeBlockwise4bitCpu<float, FP4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<float, NF4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
);

template void dequantizeBlockwise4bitCpu<fp16_t, FP4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<fp16_t, NF4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
);

template void dequantizeBlockwise4bitCpu<bf16_t, FP4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<bf16_t, NF4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
);

#if defined(__AVX512F__) && defined(__AVX512BF16__)
template void gemv_4bit_inference<bf16_t, FP4>(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
);
template void gemv_4bit_inference<bf16_t, NF4>(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
);
#endif
