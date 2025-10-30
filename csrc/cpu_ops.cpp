#include <BinSearch.h>
#include <common.h>
#include <cpu_ops.h>
#include <thread>

using namespace BinSearch;


template <typename T, int DATA_TYPE>
void dequantizeBlockwiseCpu(float* code, unsigned char* A, float* absmax, T* out,
                            long long blocksize, long long n) {
    if (DATA_TYPE > 0) {
        #pragma omp parallel for
        for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
            long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
            long long block_end = block_idx + valid_items;
            float scale = absmax[block_idx / blocksize];
            for (long long i = block_idx; i < block_end; i++) {
                float v = code[A[i]] * scale;
                if constexpr (std::is_same<T, bf16_t>::value) {
                    out[i] = float_to_bf16(v);
                } else {
                    out[i] = static_cast<T>(v);
                }
            }
        }
    } else {
        // 4bit path
        dequantizeBlockwise4bitCpu<T, DATA_TYPE>(code, A, absmax, out, blocksize, n);
    }
}

template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(float* code, unsigned char* A, float* absmax, T* out, long long blocksize, long long n) {
    return;
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

template void dequantizeBlockwiseCpu<float, General8bit>(
    float* code, unsigned char* A, float* absmax, float* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<float, FP4>(
    float* code, unsigned char* A, float* absmax, float* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<float, NF4>(
    float* code, unsigned char* A, float* absmax, float* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<fp16_t, General8bit>(
    float* code, unsigned char* A, float* absmax, fp16_t* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<fp16_t, FP4>(
    float* code, unsigned char* A, float* absmax, fp16_t* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<fp16_t, NF4>(
    float* code, unsigned char* A, float* absmax, fp16_t* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<bf16_t, General8bit>(
    float* code, unsigned char* A, float* absmax, bf16_t* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<bf16_t, FP4>(
    float* code, unsigned char* A, float* absmax, bf16_t* out, long long blocksize, long long n);

template void dequantizeBlockwiseCpu<bf16_t, NF4>(
    float* code, unsigned char* A, float* absmax, bf16_t* out, long long blocksize, long long n);

// template void gemv_4bit_inference<fp16_t, 16>(
//     int m, int n, int k, fp16_t* A, unsigned char* B, float* absmax, float* datatype, fp16_t* out,
//     int lda, int ldb, int ldc, int blocksize);

// template void gemv_4bit_inference<bf16_t, 16>(
//     int m, int n, int k, bf16_t* A, unsigned char* B, float* absmax, float* datatype, bf16_t* out,
//     int lda, int ldb, int ldc, int blocksize);

// template void gemv_4bit_inference<float, 32>(
//     int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out,
//     int lda, int ldb, int ldc, int blocksize);
