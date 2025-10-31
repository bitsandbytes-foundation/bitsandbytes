#include <BinSearch.h>
#include <common.h>
#include <cpu_ops.h>
#include <thread>

using namespace BinSearch;


// 4-bit (FP4 / NF4) dequantization helper extracted from the original else branch.
// DATA_TYPE: 1 = FP4, 2 = NF4
template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(unsigned char* A,
                                const float* absmax,
                                T* out,
                                long long blocksize,
                                long long m,
                                long long n) {
    static_assert(DATA_TYPE == 0 || DATA_TYPE == 1,
                  "dequantizeBlockwise4bitCpu called with non 4-bit DATA_TYPE");
    if (blocksize <= 0 || n <= 0) return;

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(TEST_BUG)
    // AVX512 optimized branch (placeholder)
    // DATA_TYPE: 1 = FP4, 2 = NF4
    if (1 == 0) {return;}
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
