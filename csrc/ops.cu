// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cub/device/device_scan.cuh>
#include <kernels.cuh>
#include <limits>
#include <ops.cuh>

#define ERR_NOT_IMPLEMENTED 100

using std::cout;
using std::endl;

void quantize(float* code, float* A, unsigned char* out, int n) {
    int num_blocks = n / 1024;
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    kQuantize<<<num_blocks, 1024>>>(code, A, out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void dequantize(float* code, unsigned char* A, float* out, int n, cudaStream_t stream) {
    int num_blocks = n / 1024;
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    kDequantize<<<num_blocks, 1024, 0, stream>>>(code, A, out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

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
    else if (blocksize == 64)
        kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE><<<num_blocks, 32>>>(code, A, absmax, out, rand, rand_offset, n);
    else if (blocksize == 32) {
        // For 4-bit: use specialized kernel (kQuantizeBlockwise32) that processes 2 blocks per warp
        // Each CUDA block handles 2 quantization blocks, so divide num_blocks by 2
        if (DATA_TYPE > 0) {
            int num_blocks_adjusted = (num_blocks + 1) / 2;
            kQuantizeBlockwise32<T, DATA_TYPE><<<num_blocks_adjusted, 32>>>(code, A, absmax, out, rand, rand_offset, n);
        }
    }

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code, unsigned char* A, float* absmax, T* out, int blocksize, const int n, cudaStream_t stream
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

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
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
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(
                g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n
            );
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }
        kOptimizer32bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, beta3, alpha, eps, weight_decay, step, lr,
            gnorm_scale, skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
        if (max_unorm > 0.0f) {
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }

        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case LION:
        // in lion, the momentum update after the parameter update
        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        if (max_unorm > 0.0f) {
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }
        break;
    }
}

template <typename T, int OPTIMIZER>
void optimizerStatic8bit(
    T* p, T* g, unsigned char* state1, unsigned char* state2, float* unorm, float max_unorm, float param_norm,
    float beta1, float beta2, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* max1,
    float* max2, float* new_max1, float* new_max2, float weight_decay, const float gnorm_scale, int n
) {
    int num_blocks = n / 4096;
    num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;

    if (max_unorm > 0.0f) {
        CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
    }

    switch (OPTIMIZER) {
    case ADAM:
        CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1 * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemset(new_max2, 0, 1 * sizeof(float)));
        kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 256>>>(
            p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1,
            new_max2, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        kOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2,
            max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
        CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1 * sizeof(float)));
        kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 256>>>(
            p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        kOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1,
            weight_decay, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case LION:
        // in lion, the momentum update happens after the parameter update
        kOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1,
            weight_decay, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        CUDA_CHECK_RETURN(cudaMemset(new_max1, 0, 1 * sizeof(float)));
        kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 256>>>(
            p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    default:
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
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
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
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    }
}

template <typename T> void percentileClipping(T* g, float* gnorm_vec, int step, const int n) {
    int num_blocks = n / 2048;
    num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
    CUDA_CHECK_RETURN(cudaMemset(&gnorm_vec[step % 100], 0, 1 * sizeof(float)));
    kPercentileClipping<T, 2048, 4><<<num_blocks, 512>>>(g, gnorm_vec, step, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void gemmex(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc
) {
    const int falpha = 1;
    const int fbeta = 0;
    const void* alpha = &falpha;
    const void* beta = &fbeta;
    cublasStatus_t status;

    status = cublasGemmEx(
        context->m_handle, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
        alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, beta, C, CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }
}

void strided_gemmex(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc, long long int strideA, long long int strideB, long long int strideC, int batchCount
) {
    const int falpha = 1;
    const int fbeta = 0;
    const void* alpha = &falpha;
    const void* beta = &fbeta;
    cublasStatus_t status;

    // cout << transposeA << transposeB << endl;
    // printf("%i %i %i\n", m,n,k);
    // printf("%i %i %i\n", lda,ldb,ldc);
    // printf("%i %i %i\n", strideA, strideB, strideC);
    // printf("%i\n", batchCount);

    status = cublasGemmStridedBatchedEx(
        context->m_handle, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
        alpha, A, CUDA_R_8I, lda, (long long int)strideA, B, CUDA_R_8I, ldb, (long long int)strideB, beta, C,
        CUDA_R_32I, ldc, (long long int)strideC, batchCount, CUDA_R_32I, CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }
}

int roundoff(int v, int d) { return (v + d - 1) / d * d; }

template <int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
) {

    // Calculate C = A^T @ B, in col-major layout.
    //
    // Use the IMMA kernels requires:
    // * A must be transposed and B must be non-transposed.
    // * Dimensions m and k must be multiples of 4.
    // * All pointers must be 4-byte aligned; 16-byte alignment preferred.

    int has_error = 0;

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc;
    cublasOperation_t opT = CUBLAS_OP_T;

    cudaDataType_t outType = DTYPE_OUT == 32 ? CUDA_R_32I : CUDA_R_8I;
    cudaDataType_t scaleType = DTYPE_OUT == 32 ? CUDA_R_32I : CUDA_R_32F;

    cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;

    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_8I, m, k, lda));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_8I, m, n, ldb));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&cDesc, outType, k, n, ldc));

    // Default layout order is col major

    has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, scaleType));
    has_error |=
        checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));

    if (DTYPE_OUT == 32) {
        int alpha = 1, beta = 0;
        has_error |= checkCublasStatus(cublasLtMatmul(
            ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int32_t*)C, cDesc, (int32_t*)C, cDesc, NULL, NULL,
            0, stream
        ));
    } else {
        // This path is unlikely to be used, as 8-bit accumulation can lead to likely overflows.

        if (!SCALE_ROWS) {
            float alpha = 1.0f, beta = 0.0f;
            has_error |= checkCublasStatus(cublasLtMatmul(
                ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int8_t*)C, cDesc, (int8_t*)C, cDesc, NULL,
                NULL, 0, stream
            ));
        } else {
            cublasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            float beta = 0.0f;
            has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(alphaVec)
            ));
            has_error |= checkCublasStatus(cublasLtMatmul(
                ltHandle, matmulDesc, row_scale, A, aDesc, B, bDesc, &beta, (int8_t*)C, cDesc, (int8_t*)C, cDesc, NULL,
                NULL, 0, stream
            ));
        }
    }

    has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(cDesc));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    has_error |= checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));

    if (has_error == 1)
        printf("error detected");

    return has_error;
}

int fill_up_to_nearest_multiple(int value, int multiple) {
    return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(
    int* A, float* rowStats, float* colStats, half* out, half* bias, int numRows, int numCols, cudaStream_t stream
) {
    const int threads = 512;
    const int num_per_thread = 4;
    const int num_per_block = threads * num_per_thread;
    const int n = numRows * numCols;
    const int num_blocks = (n + num_per_block - 1) / num_per_block;

    kdequant_mm_int32_fp16<num_per_thread, threads>
        <<<num_blocks, threads, 0, stream>>>(A, rowStats, colStats, out, bias, numRows, numCols, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void int8VectorQuant(
    half* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols, cudaStream_t stream
) {
    if (threshold == 0.0) {
        kInt8VectorQuant<half, 1024, 0><<<rows, 1024, 0, stream>>>(A, out, rowStats, threshold, rows, cols);
    } else {
        kInt8VectorQuant<half, 1024, 1><<<rows, 1024, 0, stream>>>(A, out, rowStats, threshold, rows, cols);
    }
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void spmm_coo(
    cusparseHandle_t handle, int* A_rowidx, int* A_colidx, half* A_vals, int A_nnz, int A_rows, int A_cols, int B_cols,
    int ldb, half* B, int ldc, half* C, bool transposed_B
) {
    cusparseSpMatDescr_t descA;
    cusparseDnMatDescr_t descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUSPARSE(cusparseCreateCoo(
        &descA, A_rows, A_cols, A_nnz, A_rowidx, A_colidx, A_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_16F
    ));
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&descC, A_rows, B_cols, ldc, C, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    // Create dense matrix B
    if (transposed_B) {
        int tmp = A_cols;
        A_cols = B_cols;
        B_cols = tmp;
    }

    CHECK_CUSPARSE(cusparseCreateDnMat(&descB, A_cols, B_cols, ldb, B, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        transposed_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descA, descB, &beta,
        descC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize
    ));
    CUDA_CHECK_RETURN(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        transposed_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descA, descB, &beta,
        descC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer
    ));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(descA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(descB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(descC));
    CUDA_CHECK_RETURN(cudaFree(dBuffer));
}

template <typename T, int BITS>
void spmm_coo_very_sparse_naive(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, T* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
) {

    kspmm_coo_very_sparse_naive<T, 8, BITS><<<nnz_rows, 256>>>(
        max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int BITS>
void gemm_4bit_inference_naive(
    int m, int n, int k, T* A, unsigned char* B, float* absmax, float* datatype, T* out, int lda, int ldb, int ldc,
    int blocksize, cudaStream_t stream
) {

    int num_blocks = (m + 3) / 4;
    kgemm_4bit_inference_naive<T, 128, BITS>
        <<<num_blocks, 128, 0, stream>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T, int FUNC> void func(T* A, T* B, T value, long n) {
    int threads = 512;
    int blocks = n / threads;
    blocks = n % threads == 0 ? blocks : blocks + 1;
    blocks = blocks > 65535 ? 65535 : blocks;
    kfunc<T, FUNC><<<blocks, 512>>>(A, B, value, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
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
    int ldc, int blocksize, cudaStream_t stream
);
template void gemm_4bit_inference_naive<__nv_bfloat16, 16>(
    int m, int n, int k, __nv_bfloat16* A, unsigned char* B, float* absmax, float* datatype, __nv_bfloat16* out,
    int lda, int ldb, int ldc, int blocksize, cudaStream_t stream
);
template void gemm_4bit_inference_naive<float, 32>(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, cudaStream_t stream
);

template void spmm_coo_very_sparse_naive<half, 16>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
);
template void spmm_coo_very_sparse_naive<signed char, 8>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
);

template int igemmlt<32, 0>(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
);
template int igemmlt<8, 0>(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
);
template int igemmlt<8, 1>(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
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
template void quantizeBlockwise<__nv_bfloat16, 1, General8bit>(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<__nv_bfloat16, 0, General8bit>(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<__nv_bfloat16, 0, FP4>(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);
template void quantizeBlockwise<__nv_bfloat16, 0, NF4>(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, float* rand, int rand_offset, int blocksize,
    const int n
);

template void dequantizeBlockwise<float, General8bit>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<float, FP4>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<float, NF4>(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<half, General8bit>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<half, FP4>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<half, NF4>(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<__nv_bfloat16, General8bit>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<__nv_bfloat16, FP4>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
);
template void dequantizeBlockwise<__nv_bfloat16, NF4>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
);

#define MAKE_optimizer32bit(name, gtype)                                                                               \
    template void optimizer32bit<gtype, name>(                                                                         \
        gtype * g, gtype * p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm,           \
        const float beta1, const float beta2, const float beta3, const float alpha, const float eps,                   \
        const float weight_decay, const int step, const float lr, const float gnorm_scale, const bool skip_zeros,      \
        const int n                                                                                                    \
    );

MAKE_optimizer32bit(ADAM, half) MAKE_optimizer32bit(ADAM, float) MAKE_optimizer32bit(ADAM, __nv_bfloat16) MAKE_optimizer32bit(MOMENTUM, half) MAKE_optimizer32bit(MOMENTUM, float) MAKE_optimizer32bit(
    MOMENTUM, __nv_bfloat16
) MAKE_optimizer32bit(RMSPROP, half) MAKE_optimizer32bit(RMSPROP, float) MAKE_optimizer32bit(RMSPROP, __nv_bfloat16) MAKE_optimizer32bit(LION, half) MAKE_optimizer32bit(LION, float) MAKE_optimizer32bit(LION, __nv_bfloat16) MAKE_optimizer32bit(ADAGRAD, half) MAKE_optimizer32bit(ADAGRAD, float) MAKE_optimizer32bit(ADAGRAD, __nv_bfloat16) MAKE_optimizer32bit(ADEMAMIX, half) MAKE_optimizer32bit(ADEMAMIX, __nv_bfloat16) MAKE_optimizer32bit(ADEMAMIX, float)

#define MAKE_optimizerStatic8bit(name, gtype)                                                                          \
    template void optimizerStatic8bit<gtype, name>(                                                                    \
        gtype * p, gtype * g, unsigned char* state1, unsigned char* state2, float* unorm, float max_unorm,             \
        float param_norm, float beta1, float beta2, float eps, int step, float lr, float* quantiles1,                  \
        float* quantiles2, float* max1, float* max2, float* new_max1, float* new_max2, float weight_decay,             \
        const float gnorm_scale, int n                                                                                 \
    );

    MAKE_optimizerStatic8bit(ADAM, half) MAKE_optimizerStatic8bit(ADAM, float) MAKE_optimizerStatic8bit(MOMENTUM, half) MAKE_optimizerStatic8bit(MOMENTUM, float) MAKE_optimizerStatic8bit(
        RMSPROP, half
    ) MAKE_optimizerStatic8bit(RMSPROP, float) MAKE_optimizerStatic8bit(LION, half) MAKE_optimizerStatic8bit(LION, float) MAKE_optimizerStatic8bit(ADAGRAD, half) MAKE_optimizerStatic8bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name)                                                           \
    template void optimizerStatic8bitBlockwise<gtype, optim_name>(                                                     \
        gtype * p, gtype * g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3,     \
        float alpha, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1,              \
        float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n                            \
    );

        MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, LION);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(half, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(float, ADEMAMIX);

template void percentileClipping(float* g, float* gnorm_vec, int step, const int n);
template void percentileClipping(half* g, float* gnorm_vec, int step, const int n);

// ===========================================================================
// K-bit blockwise quantization/dequantization (blocksize=32, K=2..5)
//
// Kernel definitions and launch wrappers in the same compilation unit
// to avoid RDC device linking issues with template instantiations.
// ===========================================================================

// ---- Device helpers ----

__device__ __forceinline__ float warp_reduce_absmax_kbit(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

template <int K> __device__ __forceinline__ void pack_kbit_warp(unsigned char qval, unsigned int* packed_words) {
#pragma unroll
    for (int bit = 0; bit < K; bit++)
        packed_words[bit] = __ballot_sync(0xFFFFFFFF, (qval >> bit) & 1);
}

template <int K>
__device__ __forceinline__ unsigned char unpack_kbit_warp(const unsigned int* packed_words, int lane_id) {
    unsigned char val = 0;
#pragma unroll
    for (int bit = 0; bit < K; bit++)
        val |= ((packed_words[bit] >> lane_id) & 1) << bit;
    return val;
}

// ---- Stage 4: Full quantize kernel ----

template <typename T, int K>
__global__ void kQuantizeBlockwise_kbit(
    const float* __restrict__ codebook, const T* __restrict__ A, float* __restrict__ absmax,
    unsigned int* __restrict__ packed_out, const int n
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int block_start = warp_id * 32;
    if (block_start >= n)
        return;
    float val = (block_start + lane_id < n) ? (float)A[block_start + lane_id] : 0.0f;
    float amax = warp_reduce_absmax_kbit(fabsf(val));
    float amax_safe = fmaxf(amax, 1e-8f);
    if (lane_id == 0)
        absmax[warp_id] = amax;
    float normalized = val / amax_safe;
    float cb = (lane_id < (1 << K)) ? codebook[lane_id] : 0.0f;
    unsigned char best_idx = 0;
    float best_dist = 1e10f;
#pragma unroll
    for (int i = 0; i < (1 << K); i++) {
        float cb_val = __shfl_sync(0xFFFFFFFF, cb, i);
        float dist = fabsf(normalized - cb_val);
        bool closer = (dist < best_dist);
        best_dist = closer ? dist : best_dist;
        best_idx = closer ? (unsigned char)i : best_idx;
    }
    unsigned int packed[K];
    pack_kbit_warp<K>(best_idx, packed);
    if (lane_id < K)
        packed_out[warp_id * K + lane_id] = packed[lane_id];
}

// ---- E4M4 absmax decode ----
// uint8 -> float: E4M4 format with configurable bias and IEEE-style subnormals.
// Normal   (e > 0): 2^(e - BIAS) * (1 + m/16)
// Subnormal (e = 0): 2^(1 - BIAS) * (m/16)
// Zero     (e = 0, m = 0): 0.0
constexpr int E4M4_BIAS = 11;

__device__ __forceinline__ float decode_e4m4_absmax(unsigned char raw) {
    if (raw == 0)
        return 0.0f;
    int e = raw >> 4;
    int m = raw & 0xF;
    if (e == 0) {
        // Subnormal (extremely rare in practice): 2^(1-BIAS) * m/16
        return ldexpf((float)m, 1 - E4M4_BIAS - 4);
    }
    // Normal: construct IEEE 754 float directly via bit manipulation.
    // Target: 2^(e - BIAS) * (1 + m/16)
    // IEEE 754: exponent_field = (e - BIAS) + 127, mantissa_field = m << 19
    unsigned int ieee = (unsigned int)(e - E4M4_BIAS + 127) << 23 | (unsigned int)m << 19;
    return __uint_as_float(ieee);
}

// ---- E4M4 absmax encode ----
// float -> uint8: inverse of decode_e4m4_absmax.
// Normal   (e_biased > 0): e_biased = floor(log2(val)) + BIAS, m = round((val/2^e_unbiased - 1) * 16)
// Subnormal (e_biased == 0): m = round(val / 2^(1-BIAS) * 16)
__device__ __forceinline__ unsigned char encode_e4m4_absmax(float val) {
    if (val <= 0.0f)
        return 0;
    int e_unbiased = (int)floorf(log2f(val));
    int e_biased = e_unbiased + E4M4_BIAS;
    if (e_biased < 0)
        e_biased = 0;
    if (e_biased > 15)
        e_biased = 15;
    int m;
    if (e_biased == 0) {
        // Subnormal: val = 2^(1-BIAS) * (m/16)  =>  m = val / 2^(1-BIAS) * 16
        float subnormal_scale = ldexpf(1.0f, 1 - E4M4_BIAS);
        m = __float2int_rn(val / subnormal_scale * 16.0f);
    } else {
        // Normal: val = 2^e_unbiased * (1 + m/16)  =>  m = (val/2^e_unbiased - 1) * 16
        float scale = ldexpf(1.0f, e_unbiased);
        m = __float2int_rn((val / scale - 1.0f) * 16.0f);
    }
    if (m < 0)
        m = 0;
    if (m > 15)
        m = 15;
    return (unsigned char)((e_biased << 4) | m);
}

// Template helper: convert ABSMAX_T to float.
// Specialization for unsigned char uses E4M4 decode.
template <typename ABSMAX_T> __device__ __forceinline__ float load_absmax(const ABSMAX_T* absmax, int idx) {
    return (float)absmax[idx];
}

template <> __device__ __forceinline__ float load_absmax<unsigned char>(const unsigned char* absmax, int idx) {
    return decode_e4m4_absmax(absmax[idx]);
}

// ---- Stage 5: Full dequantize kernel ----

// Vectorized version: each warp processes BLOCKS_PER_WARP quant blocks,
// amortizing codebook load across multiple blocks.
// Templated on T (output type) and ABSMAX_T (absmax format).
template <typename T, int K, int BLOCKS_PER_WARP, typename ABSMAX_T>
__global__ void kDequantizeBlockwise_kbit_vec(
    const unsigned int* __restrict__ packed_in, const float* __restrict__ codebook, const ABSMAX_T* __restrict__ absmax,
    T* __restrict__ out, const int n
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int base_block = warp_id * BLOCKS_PER_WARP;

    if (base_block * 32 >= n)
        return;

    // Load codebook into lane registers (one-time, amortized across BLOCKS_PER_WARP blocks)
    float cb = (lane_id < (1 << K)) ? codebook[lane_id] : 0.0f;

#pragma unroll
    for (int b = 0; b < BLOCKS_PER_WARP; b++) {
        const int block_id = base_block + b;
        const int block_start = block_id * 32;
        if (block_start >= n)
            break;

        float amax = load_absmax(absmax, block_id);
        unsigned int packed[K];
#pragma unroll
        for (int bit = 0; bit < K; bit++) {
            unsigned int word = (lane_id == bit) ? packed_in[block_id * K + bit] : 0;
            packed[bit] = __shfl_sync(0xFFFFFFFF, word, bit);
        }
        unsigned char idx = unpack_kbit_warp<K>(packed, lane_id);
        float val = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;

        if (block_start + lane_id < n)
            out[block_start + lane_id] = (T)val;
    }
}

// ---- Launch wrappers ----

#define KBIT_WARPS_PER_BLOCK 8
#define KBIT_THREADS_PER_BLOCK (KBIT_WARPS_PER_BLOCK * 32) // 256

// ---- Production kernel launchers (Stage 4-5) ----

template <typename T, int K>
void quantizeBlockwise_kbit(const float* codebook, const T* A, float* absmax, unsigned int* packed_out, int n) {
    int num_blocks_quant = (n + 31) / 32;
    int num_cuda_blocks = (num_blocks_quant + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kQuantizeBlockwise_kbit<T, K><<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK>>>(codebook, A, absmax, packed_out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Generic dequant launcher: supports all output types and absmax formats.
template <typename T, int K, typename ABSMAX_T>
void dequantizeBlockwise_kbit(
    const unsigned int* packed_in, const float* codebook, const ABSMAX_T* absmax, T* out, int n, cudaStream_t stream
) {
    constexpr int BPW = 4; // blocks per warp
    int num_blocks_quant = (n + 31) / 32;
    int num_warps = (num_blocks_quant + BPW - 1) / BPW;
    int num_cuda_blocks = (num_warps + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kDequantizeBlockwise_kbit_vec<T, K, BPW, ABSMAX_T>
        <<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(packed_in, codebook, absmax, out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ---- Stage 2: Repack kernel (flat bit-plane -> GEMM-tiled layout) ----

// Tile sizes matching the GEMM kernel design (compile-time constants).
constexpr int KBIT_TILE_K = 64;
constexpr int KBIT_TILE_N = 128;
constexpr int KBIT_BLOCKSIZE = 32;

template <int K>
__global__ void kRepackKbit(
    const unsigned int* __restrict__ packed_flat, const float* __restrict__ absmax_flat,
    unsigned int* __restrict__ packed_tiled, unsigned char* __restrict__ absmax_tiled, const int K_dim, const int N
) {
    // Each thread handles one (n_idx, k_block_idx) pair.
    const int total_k_blocks = K_dim / KBIT_BLOCKSIZE;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * total_k_blocks)
        return;

    const int n_idx = idx / total_k_blocks;
    const int k_block_idx = idx % total_k_blocks;
    const int k_start = k_block_idx * KBIT_BLOCKSIZE;

    // Source: flat block ID from W[N, K_dim] row-major layout.
    // Element (n, k) at flat_index = n * K_dim + k; block_id = flat_index / 32.
    const int flat_block_id = n_idx * (K_dim / KBIT_BLOCKSIZE) + k_block_idx;

    // Destination: tiled position.
    const int k_tile = k_start / KBIT_TILE_K;
    const int n_tile = n_idx / KBIT_TILE_N;
    const int col = n_idx % KBIT_TILE_N;
    const int kb = (k_start % KBIT_TILE_K) / KBIT_BLOCKSIZE;

    const int n_tiles = N / KBIT_TILE_N;
    constexpr int k_blocks_per_tile = KBIT_TILE_K / KBIT_BLOCKSIZE; // 2
    constexpr int words_per_tile = KBIT_TILE_N * k_blocks_per_tile * K;
    constexpr int absmax_per_tile = KBIT_TILE_N * k_blocks_per_tile;

    const int tile_base = k_tile * n_tiles + n_tile;
    const int dst_word_base = tile_base * words_per_tile + (col * k_blocks_per_tile + kb) * K;
    const int src_word_base = flat_block_id * K;

// Copy K bit-plane words
#pragma unroll
    for (int bit = 0; bit < K; bit++)
        packed_tiled[dst_word_base + bit] = packed_flat[src_word_base + bit];

    // Encode absmax to E4M4 and copy
    const int dst_abs_idx = tile_base * absmax_per_tile + col * k_blocks_per_tile + kb;
    absmax_tiled[dst_abs_idx] = encode_e4m4_absmax(absmax_flat[flat_block_id]);
}

// Repack launcher
template <int K>
void repackKbit(
    const unsigned int* packed_flat, const float* absmax_flat, unsigned int* packed_tiled,
    unsigned char* absmax_tiled, int K_dim, int N
) {
    int total_work = N * (K_dim / KBIT_BLOCKSIZE);
    int block_size = 256;
    int grid_size = (total_work + block_size - 1) / block_size;
    kRepackKbit<K><<<grid_size, block_size>>>(packed_flat, absmax_flat, packed_tiled, absmax_tiled, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ---- Stage 3: Minimal fused kbit dequant + GEMM kernel ----
// No cp.async pipeline, no persistent kernel, no split-K.
// Validates: tiled addressing, bit-plane extraction, codebook lookup, MMA, output write.
// C[M, N] = A[M, K_dim] * W^T where W[N, K_dim] is kbit-quantized in tiled format.
//
// Grid: (n_tiles, m_tiles), 256 threads (8 warps) per block.
// For M_BLOCKS=1 (TILE_M=16): all 8 warps span N, each warp handles 16 columns.

template <int K_BITS>
__global__ void kbit_gemm_minimal(
    const half* __restrict__ A, const unsigned int* __restrict__ B_packed, const unsigned char* __restrict__ B_absmax,
    const float* __restrict__ codebook, half* __restrict__ C, const int M, const int K_dim, const int N
) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;                 // 2
    constexpr int B_COL_STRIDE = KB_PER_TILE * K_BITS + 1;  // +1 padding for bank conflicts
    constexpr int N_BLOCKS = 2;                              // 16 cols per warp / 8 cols per MMA

    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int gid = lane_id / 4;  // group_id (0-7): maps to MMA row (A/C) or column (B)
    const int tid = lane_id % 4;  // tid_in_group (0-3): maps to MMA column pairs

    const int warp_n_base = warp_id * (TILE_N / 8);  // 16 cols per warp

    // Shared memory: A tile | B tile (padded) | absmax tile
    extern __shared__ char smem[];
    half* sh_a = reinterpret_cast<half*>(smem);
    unsigned int* sh_b = reinterpret_cast<unsigned int*>(sh_a + TILE_M * TILE_K);
    unsigned char* sh_abs = reinterpret_cast<unsigned char*>(sh_b + TILE_N * B_COL_STRIDE);

    // Codebook in register (one half per lane, lanes 0..2^K-1 hold valid entries)
    half cb_h = (lane_id < (1 << K_BITS)) ? __float2half(codebook[lane_id]) : __float2half(0.0f);

    // Accumulators: N_BLOCKS MMA positions, 4 floats each
    float frag_c[N_BLOCKS][4];
#pragma unroll
    for (int nb = 0; nb < N_BLOCKS; nb++)
        frag_c[nb][0] = frag_c[nb][1] = frag_c[nb][2] = frag_c[nb][3] = 0.0f;

    const int m_base = m_tile * TILE_M;

    for (int kt = 0; kt < k_tiles; kt++) {
        const int k_base = kt * TILE_K;

        // ---- Load A tile to shared memory (synchronous) ----
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gr = m_base + row;
            int gc = k_base + col;
            sh_a[row * TILE_K + col] = (gr < M && gc < K_dim) ? A[gr * K_dim + gc] : __float2half(0.0f);
        }

        // ---- Load B tile to shared memory (with +1 column padding) ----
        const int tile_idx = kt * n_tiles + n_tile;
        const int b_global_base = tile_idx * (TILE_N * KB_PER_TILE * K_BITS);
        const int abs_global_base = tile_idx * (TILE_N * KB_PER_TILE);

        for (int i = threadIdx.x; i < TILE_N * KB_PER_TILE * K_BITS; i += blockDim.x) {
            int col = i / (KB_PER_TILE * K_BITS);
            int rem = i % (KB_PER_TILE * K_BITS);
            int kb = rem / K_BITS;
            int bit = rem % K_BITS;
            sh_b[col * B_COL_STRIDE + kb * K_BITS + bit] = B_packed[b_global_base + i];
        }

        // ---- Load absmax ----
        for (int i = threadIdx.x; i < TILE_N * KB_PER_TILE; i += blockDim.x)
            sh_abs[i] = B_absmax[abs_global_base + i];

        __syncthreads();

        // ---- Process 4 k-sub-tiles (each 16 elements) ----
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int k_block = ks / 2;   // which 32-element block (0 or 1)
            const int half_idx = ks % 2;  // which half within block (0: bits 0-15, 1: bits 16-31)

            // Load A fragment from shared memory
            // m16n8k16 register order (from Turing m16n8k8 decomposition):
            //   a[0]: row_lo (gid), k_lo (tid*2..tid*2+1)
            //   a[1]: row_hi (gid+8), k_lo (tid*2..tid*2+1)
            //   a[2]: row_lo (gid), k_hi (tid*2+8..tid*2+9)
            //   a[3]: row_hi (gid+8), k_hi (tid*2+8..tid*2+9)
            uint32_t frag_a[4];
            {
                const int kc0 = ks * 16 + tid * 2;
                const int kc1 = ks * 16 + tid * 2 + 8;
                const int r0 = gid;
                const int r1 = gid + 8;
                half2 h_rlo_klo = __halves2half2(
                    (r0 < TILE_M) ? sh_a[r0 * TILE_K + kc0] : __float2half(0.0f),
                    (r0 < TILE_M) ? sh_a[r0 * TILE_K + kc0 + 1] : __float2half(0.0f));
                half2 h_rhi_klo = __halves2half2(
                    (r1 < TILE_M) ? sh_a[r1 * TILE_K + kc0] : __float2half(0.0f),
                    (r1 < TILE_M) ? sh_a[r1 * TILE_K + kc0 + 1] : __float2half(0.0f));
                half2 h_rlo_khi = __halves2half2(
                    (r0 < TILE_M) ? sh_a[r0 * TILE_K + kc1] : __float2half(0.0f),
                    (r0 < TILE_M) ? sh_a[r0 * TILE_K + kc1 + 1] : __float2half(0.0f));
                half2 h_rhi_khi = __halves2half2(
                    (r1 < TILE_M) ? sh_a[r1 * TILE_K + kc1] : __float2half(0.0f),
                    (r1 < TILE_M) ? sh_a[r1 * TILE_K + kc1 + 1] : __float2half(0.0f));
                frag_a[0] = *reinterpret_cast<uint32_t*>(&h_rlo_klo);
                frag_a[1] = *reinterpret_cast<uint32_t*>(&h_rhi_klo);
                frag_a[2] = *reinterpret_cast<uint32_t*>(&h_rlo_khi);
                frag_a[3] = *reinterpret_cast<uint32_t*>(&h_rhi_khi);
            }

            // For each N-block (2 per warp)
#pragma unroll
            for (int nb = 0; nb < N_BLOCKS; nb++) {
                // Column in the tile for this thread's B fragment
                // B fragment layout for m16n8k16: column = gid (0-7)
                int col = warp_n_base + nb * 8 + gid;

                // Load K bit-plane words from shared memory
                unsigned int planes[K_BITS];
                int b_addr = col * B_COL_STRIDE + k_block * K_BITS;
#pragma unroll
                for (int b = 0; b < K_BITS; b++)
                    planes[b] = sh_b[b_addr + b];

                // Decode absmax for this column and block
                half scale = __float2half(decode_e4m4_absmax(sh_abs[col * KB_PER_TILE + k_block]));

                // Extract indices and dequantize 4 fragment values
                // B fragment rows: {2*tid, 2*tid+1, 2*tid+8, 2*tid+9} within the 16-element sub-tile
                // Bit position in the 32-bit plane word: half_idx*16 + row
                const int bit_offset = half_idx * 16;
                const int rows[4] = {2 * tid, 2 * tid + 1, 2 * tid + 8, 2 * tid + 9};
                half vals[4];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    int bit_pos = bit_offset + rows[r];
                    int idx = 0;
#pragma unroll
                    for (int b = 0; b < K_BITS; b++)
                        idx |= ((planes[b] >> bit_pos) & 1) << b;
                    vals[r] = __hmul(__shfl_sync(0xFFFFFFFF, cb_h, idx), scale);
                }

                // Construct B fragment as uint32_t registers
                uint32_t frag_b[2];
                {
                    half2 b0 = __halves2half2(vals[0], vals[1]);
                    half2 b1 = __halves2half2(vals[2], vals[3]);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&b0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&b1);
                }

                // MMA: C += A * B (m16n8k16, fp16 inputs, fp32 accumulator)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5, %6, %7}, "
                             "{%8, %9}, "
                             "{%10, %11, %12, %13};\n"
                             : "=f"(frag_c[nb][0]), "=f"(frag_c[nb][1]), "=f"(frag_c[nb][2]),
                               "=f"(frag_c[nb][3])
                             : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                               "r"(frag_b[0]), "r"(frag_b[1]),
                               "f"(frag_c[nb][0]), "f"(frag_c[nb][1]), "f"(frag_c[nb][2]),
                               "f"(frag_c[nb][3]));
            }
        }
        __syncthreads();
    }

    // ---- Write output ----
    // C fragment layout for m16n8k16:
    //   c[0] = C[gid,   tid*2],   c[1] = C[gid,   tid*2+1]
    //   c[2] = C[gid+8, tid*2],   c[3] = C[gid+8, tid*2+1]
#pragma unroll
    for (int nb = 0; nb < N_BLOCKS; nb++) {
        int c_col = n_tile * TILE_N + warp_n_base + nb * 8 + tid * 2;
        int m_row0 = m_base + gid;
        int m_row1 = m_base + gid + 8;
        if (m_row0 < M) {
            C[m_row0 * N + c_col] = __float2half(frag_c[nb][0]);
            C[m_row0 * N + c_col + 1] = __float2half(frag_c[nb][1]);
        }
        if (m_row1 < M) {
            C[m_row1 * N + c_col] = __float2half(frag_c[nb][2]);
            C[m_row1 * N + c_col + 1] = __float2half(frag_c[nb][3]);
        }
    }
}

// Stage 3 GEMM launcher
template <int K>
void kbitGemmMinimal(
    const half* A, const unsigned int* B_packed, const unsigned char* B_absmax, const float* codebook, half* C, int M,
    int K_dim, int N
) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_STRIDE = KB_PER_TILE * K + 1;

    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int n_tiles = N / TILE_N;

    dim3 grid(n_tiles, m_tiles);
    dim3 block(256);

    int smem_size = TILE_M * TILE_K * sizeof(half) + TILE_N * B_COL_STRIDE * sizeof(unsigned int)
                    + TILE_N * KB_PER_TILE * sizeof(unsigned char);

    kbit_gemm_minimal<K><<<grid, block, smem_size>>>(A, B_packed, B_absmax, codebook, C, M, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ---- Debug: Simple MMA test kernel ----
// Takes fp16 A[16,16] and fp16 B[16,8] (B stored row-major), outputs fp32 C[16,8].
__global__ void test_mma_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C) {
    int lane_id = threadIdx.x % 32;
    int gid = lane_id / 4;
    int tid = lane_id % 4;

    // Load A fragment: A is [16,16] row-major
    // m16n8k16 register order (from Turing m16n8k8 decomposition):
    //   a[0]: row_lo (gid), k_lo (tid*2..tid*2+1)
    //   a[1]: row_hi (gid+8), k_lo (tid*2..tid*2+1)
    //   a[2]: row_lo (gid), k_hi (tid*2+8..tid*2+9)
    //   a[3]: row_hi (gid+8), k_hi (tid*2+8..tid*2+9)
    uint32_t frag_a[4];
    {
        half2 h_rlo_klo = __halves2half2(A[gid * 16 + tid * 2], A[gid * 16 + tid * 2 + 1]);
        half2 h_rhi_klo = __halves2half2(A[(gid + 8) * 16 + tid * 2], A[(gid + 8) * 16 + tid * 2 + 1]);
        half2 h_rlo_khi = __halves2half2(A[gid * 16 + tid * 2 + 8], A[gid * 16 + tid * 2 + 9]);
        half2 h_rhi_khi = __halves2half2(A[(gid + 8) * 16 + tid * 2 + 8], A[(gid + 8) * 16 + tid * 2 + 9]);
        frag_a[0] = *reinterpret_cast<uint32_t*>(&h_rlo_klo);
        frag_a[1] = *reinterpret_cast<uint32_t*>(&h_rhi_klo);
        frag_a[2] = *reinterpret_cast<uint32_t*>(&h_rlo_khi);
        frag_a[3] = *reinterpret_cast<uint32_t*>(&h_rhi_khi);
    }

    // Load B fragment: B is [16,8] row-major. MMA B is col-major, so B_col[k,n] = B_row[k,n].
    uint32_t frag_b[2];
    {
        half2 b0 = __halves2half2(B[(tid * 2) * 8 + gid], B[(tid * 2 + 1) * 8 + gid]);
        half2 b1 = __halves2half2(B[(tid * 2 + 8) * 8 + gid], B[(tid * 2 + 9) * 8 + gid]);
        frag_b[0] = *reinterpret_cast<uint32_t*>(&b0);
        frag_b[1] = *reinterpret_cast<uint32_t*>(&b1);
    }

    float c[4] = {0, 0, 0, 0};
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                   "r"(frag_b[0]), "r"(frag_b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // Write C[16,8] row-major
    C[gid * 8 + tid * 2] = c[0];
    C[gid * 8 + tid * 2 + 1] = c[1];
    C[(gid + 8) * 8 + tid * 2] = c[2];
    C[(gid + 8) * 8 + tid * 2 + 1] = c[3];
}

void testMMA(const half* A, const half* B, float* C) {
    test_mma_kernel<<<1, 32>>>(A, B, C);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


// ---- Template instantiations ----

#define INSTANTIATE_KBIT_QUANT(T, K)                                                                                   \
    template void quantizeBlockwise_kbit<T, K>(const float*, const T*, float*, unsigned int*, int);

INSTANTIATE_KBIT_QUANT(half, 2)
INSTANTIATE_KBIT_QUANT(half, 3)
INSTANTIATE_KBIT_QUANT(half, 4)
INSTANTIATE_KBIT_QUANT(half, 5)
INSTANTIATE_KBIT_QUANT(__nv_bfloat16, 2)
INSTANTIATE_KBIT_QUANT(__nv_bfloat16, 3)
INSTANTIATE_KBIT_QUANT(__nv_bfloat16, 4)
INSTANTIATE_KBIT_QUANT(__nv_bfloat16, 5)
INSTANTIATE_KBIT_QUANT(float, 2)
INSTANTIATE_KBIT_QUANT(float, 3)
INSTANTIATE_KBIT_QUANT(float, 4)
INSTANTIATE_KBIT_QUANT(float, 5)

// Dequant instantiations: all output types × absmax types × K values
#define INSTANTIATE_KBIT_DEQUANT(T, K, ABSMAX_T)                                                                       \
    template void dequantizeBlockwise_kbit<T, K, ABSMAX_T>(                                                            \
        const unsigned int*, const float*, const ABSMAX_T*, T*, int, cudaStream_t                                      \
    );

// uint8 E4M4 absmax (default)
INSTANTIATE_KBIT_DEQUANT(half, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT(half, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT(half, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT(half, 5, unsigned char)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 5, unsigned char)
INSTANTIATE_KBIT_DEQUANT(float, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT(float, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT(float, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT(float, 5, unsigned char)

// fp16 absmax (option)
INSTANTIATE_KBIT_DEQUANT(half, 2, half)
INSTANTIATE_KBIT_DEQUANT(half, 3, half)
INSTANTIATE_KBIT_DEQUANT(half, 4, half)
INSTANTIATE_KBIT_DEQUANT(half, 5, half)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 2, half)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 3, half)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 4, half)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 5, half)
INSTANTIATE_KBIT_DEQUANT(float, 2, half)
INSTANTIATE_KBIT_DEQUANT(float, 3, half)
INSTANTIATE_KBIT_DEQUANT(float, 4, half)
INSTANTIATE_KBIT_DEQUANT(float, 5, half)

// Repack instantiations: one per K value
#define INSTANTIATE_KBIT_REPACK(K) template void repackKbit<K>(const unsigned int*, const float*, unsigned int*, unsigned char*, int, int);

INSTANTIATE_KBIT_REPACK(2)
INSTANTIATE_KBIT_REPACK(3)
INSTANTIATE_KBIT_REPACK(4)
INSTANTIATE_KBIT_REPACK(5)

// GEMM instantiations: one per K value (fp16 only for Stage 3)
#define INSTANTIATE_KBIT_GEMM(K) template void kbitGemmMinimal<K>(const half*, const unsigned int*, const unsigned char*, const float*, half*, int, int, int);

INSTANTIATE_KBIT_GEMM(2)
INSTANTIATE_KBIT_GEMM(3)
INSTANTIATE_KBIT_GEMM(4)
INSTANTIATE_KBIT_GEMM(5)
