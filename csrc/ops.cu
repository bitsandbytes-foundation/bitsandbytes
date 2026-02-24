// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <kernels.cuh>
#include <limits>
#include <ops.cuh>

#define ERR_NOT_IMPLEMENTED 100

using std::cout;
using std::endl;

template <typename T, int STOCHASTIC, int DATA_TYPE>
void quantizeBlockwise(
    float* code, T* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
) {
    int num_blocks = n / blocksize;
    num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;

    if (blocksize == 4096)
        kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, DATA_TYPE>
            <<<num_blocks, 1024>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 2048)
        kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE><<<num_blocks, 512>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 1024)
        kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 512)
        kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 256)
        kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE><<<num_blocks, 128>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 128)
        kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE><<<num_blocks, 64>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 64) {
#if BNB_HIP
        // On HIP with 64-wide warps (CDNA), use specialized kernel for 4-bit types
        if constexpr (DATA_TYPE > 0) {
            kQuantizeBlockwiseSmall<T, DATA_TYPE>
                <<<(num_blocks + 1) / 2, 64>>>(code, A, absmax, out, rand, rand_offset, n);
        } else {
            kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, rand, rand_offset, n);
        }
#else
        kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, rand, rand_offset, n);
#endif
    } else if (blocksize == 32) {
        // For 4-bit: use specialized kernel that processes 2 blocks per warp
        // Each CUDA block handles 2 quantization blocks, so divide num_blocks by 2
        if constexpr (DATA_TYPE > 0) {
            int num_blocks_adjusted = (num_blocks + 1) / 2;
            kQuantizeBlockwiseSmall<T, DATA_TYPE>
                <<<num_blocks_adjusted, 32>>>(code, A, absmax, out, rand, rand_offset, n);
        }
    }

    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code, unsigned char* A, float* absmax, T* out, int blocksize, const int n, bnb_stream_t stream
) {
    constexpr int tile_size = (DATA_TYPE > 0) ? 1024 : 512;

    // Upcast to int64 to avoid overflow for large n
    int grid_blocks = ((int64_t)n + tile_size - 1) / tile_size;

    if (DATA_TYPE > 0)
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>
            <<<grid_blocks, 64, 0, stream>>>(code, A, absmax, out, blocksize / 2, n);
    else
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>
            <<<grid_blocks, 64, 0, stream>>>(code, A, absmax, out, blocksize, n);

    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

template <typename T, int OPTIMIZER>
void optimizer32bit(
    T* g, T* p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm, const float beta1,
    const float beta2, const float beta3, const float alpha, const float eps, const float weight_decay, const int step,
    const float lr, const float gnorm_scale, bool skip_zeros, const int n
) {
    int num_blocks = n / 4096;
    num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
    switch (OPTIMIZER) {
    case ADAM:
    case ADEMAMIX:
        if (max_unorm > 0.0f) {
            BNB_CHECK_RETURN(BNB_DEVICE_MEMSET(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(
                g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n
            );
            BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        }
        kOptimizer32bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, beta3, alpha, eps, weight_decay, step, lr,
            gnorm_scale, skip_zeros, n
        );
        BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
        if (max_unorm > 0.0f) {
            BNB_CHECK_RETURN(BNB_DEVICE_MEMSET(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        }

        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        break;
    case LION:
        // in lion, the momentum update after the parameter update
        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());

        if (max_unorm > 0.0f) {
            BNB_CHECK_RETURN(BNB_DEVICE_MEMSET(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        }
        break;
    }
}

#define BLOCKSIZE_2STATE 256
#define NUM_2STATE 1
#define BLOCKSIZE_1STATE 256
#define NUM_1STATE 1

template <typename T, int OPTIMIZER>
void optimizerStatic8bitBlockwise(
    T* p, T* g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3, float alpha,
    float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1, float* absmax2,
    float weight_decay, const float gnorm_scale, bool skip_zeros, int n
) {

    int num_blocks = 0;
    switch (OPTIMIZER) {
    case ADAM:
    case ADEMAMIX:
        num_blocks = n / BLOCKSIZE_2STATE;
        num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
        kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>
            <<<num_blocks, BLOCKSIZE_2STATE / NUM_2STATE>>>(
                p, g, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, quantiles1, quantiles2, absmax1,
                absmax2, weight_decay, gnorm_scale, skip_zeros, n
            );
        BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
    case LION:
        num_blocks = n / BLOCKSIZE_1STATE;
        num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
        kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>
            <<<num_blocks, BLOCKSIZE_1STATE / NUM_1STATE>>>(
                p, g, state1, beta1, beta2, eps, step, lr, quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n
            );
        BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
        break;
    }
}

void gemmex(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc
) {
    const int falpha = 1;
    const int fbeta = 0;
    const void* alpha = &falpha;
    const void* beta = &fbeta;

#if BNB_HIP
    hipblasStatus_t status;

#if hipblasVersionMajor >= 3
    status = hipblasGemmEx(
        context->m_handle, transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N, transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k,
        alpha, A, HIP_R_8I, lda, B, HIP_R_8I, ldb, beta, C, HIP_R_32I, ldc, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT
    );
#else
    status = hipblasGemmEx(
        context->m_handle, transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N, transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k,
        alpha, A, HIPBLAS_R_8I, lda, B, HIPBLAS_R_8I, ldb, beta, C, HIPBLAS_R_32I, ldc, HIPBLAS_R_32I,
        HIPBLAS_GEMM_DEFAULT
    );
#endif

    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cout << "HIPBLAS ERROR: Status " << status << std::endl;
    }
#else
    cublasStatus_t status;

    status = cublasGemmEx(
        context->m_handle, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
        alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, beta, C, CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }
#endif
}

void strided_gemmex(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc, long long int strideA, long long int strideB, long long int strideC, int batchCount
) {
    const int falpha = 1;
    const int fbeta = 0;
    const void* alpha = &falpha;
    const void* beta = &fbeta;

#if BNB_HIP
    hipblasStatus_t status;

#if hipblasVersionMajor >= 3
    status = hipblasGemmStridedBatchedEx(
        context->m_handle, transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N, transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k,
        alpha, A, HIP_R_8I, lda, (long long int)strideA, B, HIP_R_8I, ldb, (long long int)strideB, beta, C, HIP_R_32I,
        ldc, (long long int)strideC, batchCount, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT
    );
#else
    status = hipblasGemmStridedBatchedEx(
        context->m_handle, transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N, transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k,
        alpha, A, HIPBLAS_R_8I, lda, (long long int)strideA, B, HIPBLAS_R_8I, ldb, (long long int)strideB, beta, C,
        HIPBLAS_R_32I, ldc, (long long int)strideC, batchCount, HIPBLAS_R_32I, HIPBLAS_GEMM_DEFAULT
    );
#endif

    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cout << "HIPBLAS ERROR: Status " << status << std::endl;
    }
#else
    cublasStatus_t status;

    status = cublasGemmStridedBatchedEx(
        context->m_handle, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
        alpha, A, CUDA_R_8I, lda, (long long int)strideA, B, CUDA_R_8I, ldb, (long long int)strideB, beta, C,
        CUDA_R_32I, ldc, (long long int)strideC, batchCount, CUDA_R_32I, CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }
#endif
}

int roundoff(int v, int d) { return (v + d - 1) / d * d; }

template <int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(
    bnb_blasLt_handle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, bnb_stream_t stream
) {

#if BNB_HIP && defined(NO_HIPBLASLT)
    return ERR_NOT_IMPLEMENTED;
#else

    // Calculate C = A^T @ B, in col-major layout.
    //
    // Use the IMMA kernels requires:
    // * A must be transposed and B must be non-transposed.
    // * Dimensions m and k must be multiples of 4.
    // * All pointers must be 4-byte aligned; 16-byte alignment preferred.

    int has_error = 0;

    bnb_blasLt_matmul_desc_t matmulDesc;
    bnb_blasLt_layout_t aDesc, bDesc, cDesc;
    auto opT = BNB_BLASLT_OP_T;

    auto outType = DTYPE_OUT == 32 ? BNB_R_32I : BNB_R_8I;
    auto scaleType = DTYPE_OUT == 32 ? BNB_R_32I : BNB_R_32F;

    auto pointerMode = BNB_BLASLT_PTR_MODE_ALPHA_VEC;

    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&aDesc, BNB_R_8I, m, k, lda));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&bDesc, BNB_R_8I, m, n, ldb));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&cDesc, outType, k, n, ldc));

    // Default layout order is col major

    has_error |= checkBlasLtStatus(bnb_blasLtMatmulDescCreate(&matmulDesc, BNB_BLASLT_COMPUTE_32I, scaleType));
    has_error |= checkBlasLtStatus(bnb_blasLtMatmulDescSetAttr(matmulDesc, BNB_BLASLT_DESC_TRANSA, &opT, sizeof(opT)));

    if (DTYPE_OUT == 32) {
#if BNB_HIP
        // HIP requires heuristic algo selection
        const int64_t max_workspace_size = 0; // set to 0 to avoid choosing GSU kernel

        bnb_blasLt_preference_t pref;
        checkBlasLtStatus(bnb_blasLtPrefCreate(&pref));
        checkBlasLtStatus(
            bnb_blasLtPrefSetAttr(pref, BNB_BLASLT_PREF_MAX_WORKSPACE, &max_workspace_size, sizeof(max_workspace_size))
        );

        const int request_solutions = 1;
        bnb_blasLt_heuristic_t heuristicResult[request_solutions];
        int returnedAlgoCount = 0;
        checkBlasLtStatus(bnb_blasLtAlgoGetHeuristic(
            ltHandle, matmulDesc, aDesc, bDesc, cDesc, cDesc, pref, request_solutions, heuristicResult,
            &returnedAlgoCount
        ));

        if (returnedAlgoCount == 0) {
            has_error = 1;
            fprintf(stderr, "Error: Matmul Algo Heuristic didn't return algorithms\n");
        } else {
            int alpha = 1, beta = 0;
            has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
                ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int32_t*)C, cDesc, (int32_t*)C, cDesc,
                &heuristicResult[0].algo, NULL, 0, stream
            ));
        }
#else
        int alpha = 1, beta = 0;
        has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
            ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int32_t*)C, cDesc, (int32_t*)C, cDesc, NULL, NULL,
            0, stream
        ));
#endif
    } else {
        // This path is unlikely to be used, as 8-bit accumulation can lead to likely overflows.

        if (!SCALE_ROWS) {
            float alpha = 1.0f, beta = 0.0f;
            has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
                ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int8_t*)C, cDesc, (int8_t*)C, cDesc, NULL,
                NULL, 0, stream
            ));
        } else {
            auto alphaVec = BNB_BLASLT_PTR_MODE_ALPHA_VEC;
            float beta = 0.0f;
            has_error |= checkBlasLtStatus(
                bnb_blasLtMatmulDescSetAttr(matmulDesc, BNB_BLASLT_DESC_POINTER_MODE, &pointerMode, sizeof(alphaVec))
            );
            has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
                ltHandle, matmulDesc, row_scale, A, aDesc, B, bDesc, &beta, (int8_t*)C, cDesc, (int8_t*)C, cDesc, NULL,
                NULL, 0, stream
            ));
        }
    }

    has_error |= checkBlasLtStatus(bnb_blasLtLayoutDestroy(cDesc));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutDestroy(bDesc));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutDestroy(aDesc));
    has_error |= checkBlasLtStatus(bnb_blasLtMatmulDescDestroy(matmulDesc));

    if (has_error == 1)
        printf("error detected");

    return has_error;
#endif // NO_HIPBLASLT
}

int fill_up_to_nearest_multiple(int value, int multiple) {
    return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(
    int* A, float* rowStats, float* colStats, half* out, half* bias, int numRows, int numCols, bnb_stream_t stream
) {
    const int threads = 512;
    const int num_per_thread = 4;
    const int num_per_block = threads * num_per_thread;
    const int n = numRows * numCols;
    const int num_blocks = (n + num_per_block - 1) / num_per_block;

    kdequant_mm_int32_fp16<num_per_thread, threads>
        <<<num_blocks, threads, 0, stream>>>(A, rowStats, colStats, out, bias, numRows, numCols, n);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

void int8VectorQuant(
    half* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols, bnb_stream_t stream
) {
    if (threshold == 0.0) {
        kInt8VectorQuant<half, 1024, 0><<<rows, 1024, 0, stream>>>(A, out, rowStats, threshold, rows, cols);
    } else {
        kInt8VectorQuant<half, 1024, 1><<<rows, 1024, 0, stream>>>(A, out, rowStats, threshold, rows, cols);
    }
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

template <typename T, int BITS>
void gemm_4bit_inference_naive(
    int m, int n, int k, T* A, unsigned char* B, float* absmax, float* datatype, T* out, int lda, int ldb, int ldc,
    int blocksize, bnb_stream_t stream
) {

    int num_blocks = (m + 3) / 4;
#if BNB_HIP
    // On 64-wide warp architectures, each warp processes 2 rows instead of 4
    if (BNB_WARP_SIZE == 64) {
        num_blocks = (m + 1) / 2;
    }
#endif

    kgemm_4bit_inference_naive<T, 128, BITS>
        <<<num_blocks, 128, 0, stream>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

template <typename T, int FUNC> void func(T* A, T* B, T value, long n) {
    int threads = 512;
    int blocks = n / threads;
    blocks = n % threads == 0 ? blocks : blocks + 1;
    blocks = blocks > 65535 ? 65535 : blocks;
    kfunc<T, FUNC><<<blocks, 512>>>(A, B, value, n);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void func<float, FILL>(float* A, float* B, float value, long n);
template void func<unsigned char, FILL>(unsigned char* A, unsigned char* B, unsigned char value, long n);
template void func<float, ARANGE>(float* A, float* B, float value, long n);
template void func<float, _MUL>(float* A, float* B, float value, long n);

template void gemm_4bit_inference_naive<half, 16>(
    int m, int n, int k, half* A, unsigned char* B, float* absmax, float* datatype, half* out, int lda, int ldb,
    int ldc, int blocksize, bnb_stream_t stream
);
template void gemm_4bit_inference_naive<bnb_bfloat16, 16>(
    int m, int n, int k, bnb_bfloat16* A, unsigned char* B, float* absmax, float* datatype, bnb_bfloat16* out, int lda,
    int ldb, int ldc, int blocksize, bnb_stream_t stream
);
template void gemm_4bit_inference_naive<float, 32>(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, bnb_stream_t stream
);

template int igemmlt<32, 0>(
    bnb_blasLt_handle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, bnb_stream_t stream
);
template int igemmlt<8, 0>(
    bnb_blasLt_handle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, bnb_stream_t stream
);
template int igemmlt<8, 1>(
    bnb_blasLt_handle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, bnb_stream_t stream
);

template void quantizeBlockwise<half, 1, General8bit>(
    float* code, half* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<half, 0, General8bit>(
    float* code, half* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<half, 0, FP4>(
    float* code, half* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<half, 0, NF4>(
    float* code, half* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<float, 1, General8bit>(
    float* code, float* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<float, 0, General8bit>(
    float* code, float* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<float, 0, FP4>(
    float* code, float* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<float, 0, NF4>(
    float* code, float* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize, const int n
);
template void quantizeBlockwise<bnb_bfloat16, 1, General8bit>(
    float* code, bnb_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<bnb_bfloat16, 0, General8bit>(
    float* code, bnb_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<bnb_bfloat16, 0, FP4>(
    float* code, bnb_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<bnb_bfloat16, 0, NF4>(
    float* code, bnb_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);

template void dequantizeBlockwise<float, General8bit>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<float, FP4>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<float, NF4>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<half, General8bit>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<half, FP4>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<half, NF4>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<bnb_bfloat16, General8bit>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<bnb_bfloat16, FP4>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, int blocksize, const int n, bnb_stream_t stream
);
template void dequantizeBlockwise<bnb_bfloat16, NF4>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, int blocksize, const int n, bnb_stream_t stream
);

#define MAKE_optimizer32bit(name, gtype)                                                                               \
    template void optimizer32bit<gtype, name>(                                                                         \
        gtype * g, gtype * p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm,           \
        const float beta1, const float beta2, const float beta3, const float alpha, const float eps,                   \
        const float weight_decay, const int step, const float lr, const float gnorm_scale, const bool skip_zeros,      \
        const int n                                                                                                    \
    );

MAKE_optimizer32bit(ADAM, half) MAKE_optimizer32bit(ADAM, float) MAKE_optimizer32bit(ADAM, bnb_bfloat16) MAKE_optimizer32bit(MOMENTUM, half) MAKE_optimizer32bit(MOMENTUM, float) MAKE_optimizer32bit(MOMENTUM, bnb_bfloat16) MAKE_optimizer32bit(RMSPROP, half) MAKE_optimizer32bit(RMSPROP, float) MAKE_optimizer32bit(RMSPROP, bnb_bfloat16) MAKE_optimizer32bit(
    LION, half
) MAKE_optimizer32bit(LION, float) MAKE_optimizer32bit(LION, bnb_bfloat16) MAKE_optimizer32bit(ADAGRAD, half) MAKE_optimizer32bit(ADAGRAD, float) MAKE_optimizer32bit(ADAGRAD, bnb_bfloat16) MAKE_optimizer32bit(ADEMAMIX, half) MAKE_optimizer32bit(ADEMAMIX, bnb_bfloat16) MAKE_optimizer32bit(ADEMAMIX, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name)                                                           \
    template void optimizerStatic8bitBlockwise<gtype, optim_name>(                                                     \
        gtype * p, gtype * g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3,     \
        float alpha, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1,              \
        float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n                            \
    );

    MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, LION);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(half, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(bnb_bfloat16, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(float, ADEMAMIX);
