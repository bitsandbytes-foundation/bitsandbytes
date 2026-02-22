// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cub/device/device_scan.cuh>
#include <kernels.cuh>
#include <limits>
#include <ops.cuh>
#include <type_traits>
#include <vector>


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

// Branchless version for the GEMM inner loop.  Eliminates BSSY/BSYNC
// divergence-handling pairs that the branchy version generates.
// Subnormals (e==0) are treated as normal-path (produces a small wrong
// value, but no real weight block has absmax < 2^-10).
__device__ __forceinline__ float decode_e4m4_absmax_branchless(unsigned char raw) {
    int e = raw >> 4;
    int m = raw & 0xF;
    // Normal path: construct IEEE 754 directly.
    // When raw==0 (e==0, m==0) this produces 2^(0-11+127)<<23 | 0 which
    // is some small positive float; we select 0.0 below via predicate.
    unsigned int ieee = (unsigned int)(e - E4M4_BIAS + 127) << 23 | (unsigned int)m << 19;
    float result = __uint_as_float(ieee);
    // Zero-out for raw==0 using predicated select (no branch).
    // PTXAS emits a FSEL instruction (1 cycle, no divergence).
    return (raw != 0) ? result : 0.0f;
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

// ---- Stage 4: Full quantize kernel ----

template <typename T, int K>
__global__ void kQuantizeBlockwise_kbit(
    const float* __restrict__ codebook, const T* __restrict__ A, unsigned char* __restrict__ absmax,
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
        absmax[warp_id] = encode_e4m4_absmax(amax);
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
void quantizeBlockwise_kbit(
    const float* codebook, const T* A, unsigned char* absmax, unsigned int* packed_out, int n, cudaStream_t stream
) {
    int num_blocks_quant = (n + 31) / 32;
    int num_cuda_blocks = (num_blocks_quant + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kQuantizeBlockwise_kbit<T, K><<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(codebook, A, absmax, packed_out, n);
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

// Tiled-layout dequantize: reads from repack_kbit output (tiled bit-plane layout),
// writes to flat [N, K_dim] row-major output for cuBLAS matmul.
// block_id maps to (n_idx, k_block_idx) coordinates, then computes tiled read addresses.
template <typename T, int K, int BLOCKS_PER_WARP, typename ABSMAX_T>
__global__ void kDequantizeBlockwise_kbit_tiled(
    const unsigned int* __restrict__ packed_in, const float* __restrict__ codebook, const ABSMAX_T* __restrict__ absmax,
    T* __restrict__ out, const int K_dim, const int N
) {
    constexpr int BS = 32; // quantization block size
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int KB_PER_TILE = TILE_K / BS; // 2
    constexpr int WORDS_PER_TILE = TILE_N * KB_PER_TILE * K;
    constexpr int ABS_PER_TILE = TILE_N * KB_PER_TILE;

    const int total_k_blocks = K_dim / BS;
    const int n_tiles = N / TILE_N;
    const int n = N * K_dim; // total elements

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int base_block = warp_id * BLOCKS_PER_WARP;

    if (base_block * 32 >= n)
        return;

    float cb = (lane_id < (1 << K)) ? codebook[lane_id] : 0.0f;

#pragma unroll
    for (int b = 0; b < BLOCKS_PER_WARP; b++) {
        const int block_id = base_block + b;
        const int block_start = block_id * 32;
        if (block_start >= n)
            break;

        // Decompose linear block_id into matrix coordinates
        const int n_idx = block_id / total_k_blocks;
        const int k_block_idx = block_id % total_k_blocks;

        // Compute tiled addresses
        const int k_tile = k_block_idx / KB_PER_TILE;
        const int kb = k_block_idx % KB_PER_TILE;
        const int n_tile = n_idx / TILE_N;
        const int col_in_tile = n_idx % TILE_N;
        const int tile_base = k_tile * n_tiles + n_tile;
        const int word_base = tile_base * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + kb) * K;
        const int abs_idx = tile_base * ABS_PER_TILE + col_in_tile * KB_PER_TILE + kb;

        float amax = load_absmax(absmax, abs_idx);
        unsigned int packed[K];
#pragma unroll
        for (int bit = 0; bit < K; bit++) {
            unsigned int word = (lane_id == bit) ? packed_in[word_base + bit] : 0;
            packed[bit] = __shfl_sync(0xFFFFFFFF, word, bit);
        }
        unsigned char idx = unpack_kbit_warp<K>(packed, lane_id);
        float val = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;

        if (block_start + lane_id < n)
            out[block_start + lane_id] = (T)val;
    }
}

// Tiled dequant launcher
template <typename T, int K, typename ABSMAX_T>
void dequantizeBlockwise_kbit_tiled(
    const unsigned int* packed_in, const float* codebook, const ABSMAX_T* absmax, T* out, int K_dim, int N,
    cudaStream_t stream
) {
    constexpr int BPW = 4;
    int n = N * K_dim;
    int num_blocks_quant = (n + 31) / 32;
    int num_warps = (num_blocks_quant + BPW - 1) / BPW;
    int num_cuda_blocks = (num_warps + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kDequantizeBlockwise_kbit_tiled<T, K, BPW, ABSMAX_T>
        <<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(packed_in, codebook, absmax, out, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ---- Stage 2: Repack kernel (flat bit-plane -> GEMM-tiled layout) ----

// Tile sizes matching the GEMM kernel design (compile-time constants).
constexpr int KBIT_TILE_K = 64;
constexpr int KBIT_TILE_N = 128;
constexpr int KBIT_BLOCKSIZE = 32;

template <int K>
__global__ void kRepackKbit(
    const unsigned int* __restrict__ packed_flat, const unsigned char* __restrict__ absmax_flat,
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

    // Copy absmax byte (already E4M4 encoded from quantize_kbit)
    const int dst_abs_idx = tile_base * absmax_per_tile + col * k_blocks_per_tile + kb;
    absmax_tiled[dst_abs_idx] = absmax_flat[flat_block_id];
}

// Repack launcher
template <int K>
void repackKbit(
    const unsigned int* packed_flat, const unsigned char* absmax_flat, unsigned int* packed_tiled,
    unsigned char* absmax_tiled, int K_dim, int N, cudaStream_t stream
) {
    int total_work = N * (K_dim / KBIT_BLOCKSIZE);
    int block_size = 256;
    int grid_size = (total_work + block_size - 1) / block_size;
    kRepackKbit<K><<<grid_size, block_size, 0, stream>>>(packed_flat, absmax_flat, packed_tiled, absmax_tiled, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Datacenter GPU detection: Hopper (sm_90) and Blackwell datacenter (sm_100).
// NOTE: sm_120 (RTX 5090, Blackwell consumer) lacks TMA/wgmma — must NOT match.
#if defined(__CUDA_ARCH__)
#define BNB_DATACENTER_GPU (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#else
#define BNB_DATACENTER_GPU 0
#endif

// L2 prefetch hint (datacenter GPUs only — consumer GPUs ignore it)
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
#if BNB_DATACENTER_GPU
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
#else
    (void)ptr;
#endif
}

// cp.async helpers (sm_80+) — used by production MMA and grouped MMA kernels
__device__ __forceinline__ void cp_async_cg_16(void* __restrict__ smem, const void* __restrict__ gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---- Stage 6: Production kernel with bf16 support ----
// Templates on scalar_t (half or __nv_bfloat16) and K_BITS.
// Uses the same split-K architecture as Stage 5.

// Helper: type-specific operations
template <typename scalar_t> struct ScalarOps {
    __device__ static scalar_t from_float(float f);
    __device__ static float to_float(scalar_t v);
    __device__ static scalar_t mul(scalar_t a, scalar_t b);
};

template <> struct ScalarOps<half> {
    __device__ static half from_float(float f) { return __float2half(f); }

    __device__ static float to_float(half v) { return __half2float(v); }

    __device__ static half mul(half a, half b) { return __hmul(a, b); }
};

template <> struct ScalarOps<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_float(float f) { return __float2bfloat16(f); }

    __device__ static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

    __device__ static __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
};

// Helper: MMA instruction dispatch based on scalar_t
template <typename scalar_t>
__device__ __forceinline__ void mma_m16n8k16(uint32_t (&frag_a)[4], uint32_t (&frag_b)[2], float (&frag_c)[4]) {
    if constexpr (std::is_same_v<scalar_t, half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11, %12, %13};\n"
                     : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
                     : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]), "r"(frag_b[0]), "r"(frag_b[1]),
                       "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11, %12, %13};\n"
                     : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
                     : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]), "r"(frag_b[0]), "r"(frag_b[1]),
                       "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3]));
    }
}

// Helper: pack two scalar_t values into a uint32 (for MMA fragment register)
template <typename scalar_t> __device__ __forceinline__ uint32_t pack_two(scalar_t a, scalar_t b) {
    if constexpr (std::is_same_v<scalar_t, half>) {
        half2 v = __halves2half2(a, b);
        return *reinterpret_cast<uint32_t*>(&v);
    } else {
        __nv_bfloat162 v = __halves2bfloat162(a, b);
        return *reinterpret_cast<uint32_t*>(&v);
    }
}

template <int K_BITS, int M_BLOCKS, int TILE_N_VAL = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(TILE_N_VAL <= 64 ? 128 : 256, TILE_N_VAL <= 64 ? 12 : 1) kbit_gemm_prod(
    const scalar_t* __restrict__ A, const unsigned int* __restrict__ B_packed, const ABSMAX_T* __restrict__ B_absmax,
    const float* __restrict__ codebook, scalar_t* __restrict__ C, float* __restrict__ C_workspace,
    int* __restrict__ tile_counters, const int M, const int K_dim, const int N, const int k_splits, const int total_work
) {
    using Ops = ScalarOps<scalar_t>;
    constexpr int TILE_M = M_BLOCKS * 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = TILE_N_VAL;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_WORDS = KB_PER_TILE * K_BITS;
    constexpr int N_BLOCKS = 2;

    constexpr int A_STAGE_ELEMS = TILE_M * TILE_K;
    constexpr int B_STAGE_WORDS = TILE_N * B_COL_WORDS;
    constexpr int ABS_STAGE_ELEMS = TILE_N * KB_PER_TILE;
    constexpr int ABS_STAGE_BYTES = ABS_STAGE_ELEMS * (int)sizeof(ABSMAX_T);

    constexpr int A_STAGE_BYTES = A_STAGE_ELEMS * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES_VAL = B_STAGE_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES_VAL + ABS_STAGE_ALIGNED;

    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    const int tiles_per_split = (k_tiles + k_splits - 1) / k_splits;

    constexpr int COLS_PER_WARP = N_BLOCKS * 8; // 16: each warp handles 2 MMA n-blocks of 8 cols
    constexpr int NUM_WARPS = TILE_N / COLS_PER_WARP;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int gid = lane_id / 4;
    const int tid = lane_id % 4;
    const int warp_n_base = warp_id * COLS_PER_WARP;

    // Double-buffered shared memory
    extern __shared__ char smem[];
    auto sh_a = [&](int stage) -> scalar_t* { return reinterpret_cast<scalar_t*>(smem + stage * STAGE_BYTES); };
    auto sh_b = [&](int stage) -> unsigned int* {
        return reinterpret_cast<unsigned int*>(smem + stage * STAGE_BYTES + A_STAGE_BYTES);
    };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(smem + stage * STAGE_BYTES + A_STAGE_BYTES + B_STAGE_BYTES_VAL);
    };

    // Codebook in registers (converted to scalar_t)
    scalar_t cb_val = (lane_id < (1 << K_BITS)) ? Ops::from_float(codebook[lane_id]) : Ops::from_float(0.0f);

    float frag_c[M_BLOCKS][N_BLOCKS][4];

    // Persistent work loop: each block processes multiple (m,n,k_split) items
    // Work items are ordered k-split-last: work_id = mn_id * k_splits + ks_id
    // This groups k-splits for the same (m,n) tile together.
    for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
        const int mn_id = work_id / k_splits;
        const int ks_id = work_id % k_splits;
        const int n_tile = mn_id % n_tiles;
        const int m_tile = mn_id / n_tiles;
        const int m_base = m_tile * TILE_M;

        const int kt_start = ks_id * tiles_per_split;
        const int kt_end = min(kt_start + tiles_per_split, k_tiles);
        if (kt_start >= k_tiles)
            continue;

        // Zero accumulators for this work item
#pragma unroll
        for (int mb = 0; mb < M_BLOCKS; mb++)
#pragma unroll
            for (int nb = 0; nb < N_BLOCKS; nb++)
                frag_c[mb][nb][0] = frag_c[mb][nb][1] = frag_c[mb][nb][2] = frag_c[mb][nb][3] = 0.0f;

        // Fetch tile lambda (captures n_tile, m_base from loop)
        auto fetch_tile = [&](int stage, int kt) {
            const int k_base = kt * TILE_K;
            const int tile_idx = kt * n_tiles + n_tile;

            // B tile via cp.async
            const int b_global_base = tile_idx * B_STAGE_WORDS;
            constexpr int B_INT4S = B_STAGE_BYTES_VAL / 16;
            const int4* b_src = reinterpret_cast<const int4*>(B_packed + b_global_base);
            int4* b_dst = reinterpret_cast<int4*>(sh_b(stage));
            for (int i = threadIdx.x; i < B_INT4S; i += blockDim.x)
                cp_async_cg_16(&b_dst[i], &b_src[i]);

            // Absmax via cp.async
            const int abs_global_base = tile_idx * ABS_STAGE_ELEMS;
            constexpr int ABS_INT4S = (ABS_STAGE_BYTES + 15) / 16;
            const int4* abs_src = reinterpret_cast<const int4*>(B_absmax + abs_global_base);
            int4* abs_dst = reinterpret_cast<int4*>(sh_abs(stage));
            for (int i = threadIdx.x; i < ABS_INT4S; i += blockDim.x)
                cp_async_cg_16(&abs_dst[i], &abs_src[i]);

            // A tile via cp.async with XOR swizzle
            scalar_t* a_dst = sh_a(stage);
            constexpr int A_GROUPS = A_STAGE_ELEMS / 8;
            const bool a_interior = (m_base + TILE_M <= M) && (k_base + TILE_K <= K_dim);

            if (a_interior) {
                for (int i = threadIdx.x; i < A_GROUPS; i += blockDim.x) {
                    int row = i / (TILE_K / 8);
                    int col_group = i % (TILE_K / 8);
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * TILE_K + swizzled_group * 8]);
                    const int4* src =
                        reinterpret_cast<const int4*>(&A[(m_base + row) * K_dim + k_base + col_group * 8]);
                    cp_async_cg_16(dst, src);
                }
            } else {
                for (int i = threadIdx.x; i < A_GROUPS; i += blockDim.x) {
                    int row = i / (TILE_K / 8);
                    int col_group = i % (TILE_K / 8);
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * TILE_K + swizzled_group * 8]);
                    int gr = m_base + row;
                    int gc = k_base + col_group * 8;
                    if (gr < M && gc < K_dim) {
                        const int4* src = reinterpret_cast<const int4*>(&A[gr * K_dim + gc]);
                        cp_async_cg_16(dst, src);
                    } else {
                        *dst = make_int4(0, 0, 0, 0);
                    }
                }
            }
        };

        // Compute tile: inline dequant interleaved with MMA
        auto compute_tile = [&](int stage) {
            scalar_t* a_ptr = sh_a(stage);
            unsigned int* b_ptr = sh_b(stage);
            ABSMAX_T* abs_ptr = sh_abs(stage);

#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int k_block = ks / 2;
                const int half_idx = ks % 2;

                uint32_t frag_a[M_BLOCKS][4];
#pragma unroll
                for (int mb = 0; mb < M_BLOCKS; mb++) {
                    const int mb_row_offset = mb * 16;
                    const int matrix_id = lane_id / 8;
                    const int row_in_matrix = lane_id % 8;
                    const int a_row = mb_row_offset + row_in_matrix + (matrix_id % 2) * 8;
                    const int col_start = ks * 16 + (matrix_id / 2) * 8;
                    const int col_group = col_start / 8;
                    const int swizzled_group = col_group ^ (a_row % 8);
                    const int swizzled_col_start = swizzled_group * 8;

                    const scalar_t* addr = &a_ptr[a_row * TILE_K + swizzled_col_start];
                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(frag_a[mb][0]), "=r"(frag_a[mb][1]), "=r"(frag_a[mb][2]), "=r"(frag_a[mb][3])
                                 : "r"(smem_addr));
                }

#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int col = warp_n_base + nb * 8 + gid;
                    unsigned int planes[K_BITS];
                    int b_addr = col * B_COL_WORDS + k_block * K_BITS;
#pragma unroll
                    for (int b = 0; b < K_BITS; b++)
                        planes[b] = b_ptr[b_addr + b];

                    scalar_t scale = Ops::from_float(load_absmax<ABSMAX_T>(abs_ptr, col * KB_PER_TILE + k_block));

                    const int bit_offset = half_idx * 16;
                    const int rows[4] = {2 * tid, 2 * tid + 1, 2 * tid + 8, 2 * tid + 9};

                    int bp0 = bit_offset + rows[0];
                    int bp1 = bit_offset + rows[1];
                    int bp2 = bit_offset + rows[2];
                    int bp3 = bit_offset + rows[3];

                    int idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
#pragma unroll
                    for (int b = 0; b < K_BITS; b++) {
                        unsigned int p = planes[b];
                        idx0 |= ((p >> bp0) & 1) << b;
                        idx1 |= ((p >> bp1) & 1) << b;
                        idx2 |= ((p >> bp2) & 1) << b;
                        idx3 |= ((p >> bp3) & 1) << b;
                    }

                    scalar_t vals[4];
                    vals[0] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx0), scale);
                    vals[1] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx1), scale);
                    vals[2] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx2), scale);
                    vals[3] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx3), scale);

                    uint32_t frag_b[2];
                    frag_b[0] = pack_two<scalar_t>(vals[0], vals[1]);
                    frag_b[1] = pack_two<scalar_t>(vals[2], vals[3]);

#pragma unroll
                    for (int mb = 0; mb < M_BLOCKS; mb++) {
                        mma_m16n8k16<scalar_t>(frag_a[mb], frag_b, frag_c[mb][nb]);
                    }
                }
            }
        };

        // Pipeline: double-buffered cp.async
        fetch_tile(0, kt_start);
        cp_async_fence();

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % 2;
            if (kt + 1 < kt_end) {
                fetch_tile((kt + 1 - kt_start) % 2, kt + 1);
                cp_async_fence();
                // L2 prefetch for tile kt+2 (warms L2 before next fetch_tile issues cp.async)
                if (kt + 2 < kt_end) {
                    const int pf_tile = (kt + 2) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            compute_tile(cur);
            __syncthreads();
        }

        // Write output for this work item
        if (k_splits == 1) {
            // Direct write — this block owns the full K reduction
#pragma unroll
            for (int mb = 0; mb < M_BLOCKS; mb++) {
#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int c_col = n_tile * TILE_N + warp_n_base + nb * 8 + tid * 2;
                    int m_row0 = m_base + mb * 16 + gid;
                    int m_row1 = m_base + mb * 16 + gid + 8;
                    if (m_row0 < M) {
                        C[m_row0 * N + c_col] = Ops::from_float(frag_c[mb][nb][0]);
                        C[m_row0 * N + c_col + 1] = Ops::from_float(frag_c[mb][nb][1]);
                    }
                    if (m_row1 < M) {
                        C[m_row1 * N + c_col] = Ops::from_float(frag_c[mb][nb][2]);
                        C[m_row1 * N + c_col + 1] = Ops::from_float(frag_c[mb][nb][3]);
                    }
                }
            }
        } else {
            // Partial K — atomicAdd to workspace, last block converts to output
#pragma unroll
            for (int mb = 0; mb < M_BLOCKS; mb++) {
#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int c_col = n_tile * TILE_N + warp_n_base + nb * 8 + tid * 2;
                    int m_row0 = m_base + mb * 16 + gid;
                    int m_row1 = m_base + mb * 16 + gid + 8;
                    if (m_row0 < M) {
                        atomicAdd(&C_workspace[m_row0 * N + c_col], frag_c[mb][nb][0]);
                        atomicAdd(&C_workspace[m_row0 * N + c_col + 1], frag_c[mb][nb][1]);
                    }
                    if (m_row1 < M) {
                        atomicAdd(&C_workspace[m_row1 * N + c_col], frag_c[mb][nb][2]);
                        atomicAdd(&C_workspace[m_row1 * N + c_col + 1], frag_c[mb][nb][3]);
                    }
                }
            }

            __threadfence();

            __shared__ int is_last;
            if (threadIdx.x == 0) {
                int done = atomicAdd(&tile_counters[mn_id], 1);
                is_last = (done == k_splits - 1) ? 1 : 0;
            }
            __syncthreads();

            if (is_last) {
                for (int i = threadIdx.x; i < TILE_M * TILE_N; i += blockDim.x) {
                    int row = m_base + i / TILE_N;
                    int col = n_tile * TILE_N + i % TILE_N;
                    if (row < M)
                        C[row * N + col] = Ops::from_float(C_workspace[row * N + col]);
                }
            }
        }
    } // end persistent work loop
}

// Production GEMM launcher — persistent kernel with auto k_splits
template <int K, int MB, int TN = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char>
static void kbitGemmProdLaunch(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int num_sms, cudaStream_t stream
) {
    constexpr int TILE_M = MB * 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = TN;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_WORDS = KB_PER_TILE * K;
    constexpr int N_BLOCKS = 2;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8); // TN=128→8, TN=64→4
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    constexpr int A_STAGE_BYTES = TILE_M * TILE_K * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES = TILE_N * B_COL_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_BYTES = TILE_N * KB_PER_TILE * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int n_tiles = N / TILE_N;
    int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    int mn_tiles = m_tiles * n_tiles;

    // k_splits heuristic: target enough blocks for good SM occupancy.
    // With BLOCK_DIM threads/block, we want ~4 blocks/SM for latency hiding.
    // BLOCK_DIM=128 (TN=64): 4 blocks/SM → 16 warps → 33% occupancy
    // BLOCK_DIM=256 (TN=128): 1 block/SM → 8 warps → 16% occupancy (ok for large M)
    constexpr int TARGET_BLOCKS_PER_SM = (BLOCK_DIM <= 128) ? 4 : 1;
    int target_blocks = num_sms * TARGET_BLOCKS_PER_SM;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    // Grid: launch enough blocks to fill target occupancy.
    // Multiple blocks per SM is fine — GPU schedules them concurrently.
    int grid_size = (k_splits == 1) ? total_work : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int smem_size = 2 * STAGE_BYTES;

    kbit_gemm_prod<K, MB, TN, scalar_t, ABSMAX_T><<<grid_size, block, smem_size, stream>>>(
        A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, k_splits, total_work
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <int K, typename scalar_t, typename ABSMAX_T = unsigned char>
void kbitGemmProd(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int k_chunks, cudaStream_t stream
) {
    // Query SM count for persistent kernel grid sizing and M_BLOCKS dispatch
    int dev;
    cudaGetDevice(&dev);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    // Choose M_BLOCKS. With the persistent kernel, the grid always has
    // num_SMs blocks, so the SM utilization concern is gone. Choose the
    // largest M_BLOCKS that fits the M dimension.
    int m_blocks = 1;
    if (M > 48)
        m_blocks = 4;
    else if (M > 32)
        m_blocks = 3;
    else if (M > 16)
        m_blocks = 2;

    // Choose TILE_N: use 64 for M<=16 (m_blocks==1) to double n_tiles
    // and improve SM utilization. Use 128 for larger M where there's
    // already enough M-dimension parallelism.
    const bool use_tn64 = (m_blocks == 1) && (N % 64 == 0);

    if (use_tn64) {
        // TILE_N=64: 4 warps (128 threads), 2x more n-tiles
        kbitGemmProdLaunch<K, 1, 64, scalar_t, ABSMAX_T>(
            A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
        );
    } else {
        // TILE_N=128: original path
        switch (m_blocks) {
        case 4:
            kbitGemmProdLaunch<K, 4, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 3:
            kbitGemmProdLaunch<K, 3, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 2:
            kbitGemmProdLaunch<K, 2, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        default:
            kbitGemmProdLaunch<K, 1, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        }
    }
}

// ---- Grouped Expert GEMM ----
// Batches multiple MoE expert GEMM invocations into one kernel launch.
// All experts share K_dim, N, k, codebook. Each expert has its own
// B weights and a variable number of tokens (M_i).
// Supports TILE_N=64/128 and optional split-K for SM utilization.

template <int K_BITS, int M_BLOCKS, int TN, typename scalar_t, typename ABSMAX_T = unsigned char>
__global__ void kbit_grouped_gemm_prod(
    const scalar_t* __restrict__ A_concat, const unsigned int* __restrict__ B_packed_all,
    const ABSMAX_T* __restrict__ B_absmax_all, const float* __restrict__ codebook, scalar_t* __restrict__ C_concat,
    float* __restrict__ C_workspace, int* __restrict__ tile_counters, const int* __restrict__ expert_offsets,
    const int K_dim, const int N, const int num_experts, const int k_splits, const int total_work
) {
    using Ops = ScalarOps<scalar_t>;
    constexpr int TILE_M = M_BLOCKS * 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = TN;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_WORDS = KB_PER_TILE * K_BITS;
    constexpr int N_BLOCKS = 2;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8);
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    constexpr int A_STAGE_ELEMS = TILE_M * TILE_K;
    constexpr int B_STAGE_WORDS = TILE_N * B_COL_WORDS;
    constexpr int ABS_STAGE_ELEMS = TILE_N * KB_PER_TILE;
    constexpr int ABS_STAGE_BYTES = ABS_STAGE_ELEMS * (int)sizeof(ABSMAX_T);

    constexpr int A_STAGE_BYTES = A_STAGE_ELEMS * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES_VAL = B_STAGE_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES_VAL + ABS_STAGE_ALIGNED;

    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    const int tiles_per_split = (k_tiles + k_splits - 1) / k_splits;

    // Per-expert B data sizes (same for all experts since K_dim, N are shared)
    const int b_packed_per_expert = k_tiles * n_tiles * B_STAGE_WORDS;
    const int b_absmax_per_expert = k_tiles * n_tiles * ABS_STAGE_ELEMS;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int gid = lane_id / 4;
    const int tid = lane_id % 4;
    const int warp_n_base = warp_id * (TILE_N / NUM_WARPS);

    // Double-buffered shared memory
    extern __shared__ char smem[];
    auto sh_a = [&](int stage) -> scalar_t* { return reinterpret_cast<scalar_t*>(smem + stage * STAGE_BYTES); };
    auto sh_b = [&](int stage) -> unsigned int* {
        return reinterpret_cast<unsigned int*>(smem + stage * STAGE_BYTES + A_STAGE_BYTES);
    };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(smem + stage * STAGE_BYTES + A_STAGE_BYTES + B_STAGE_BYTES_VAL);
    };

    // Codebook in registers
    scalar_t cb_val = (lane_id < (1 << K_BITS)) ? Ops::from_float(codebook[lane_id]) : Ops::from_float(0.0f);

    float frag_c[M_BLOCKS][N_BLOCKS][4];

    // Persistent work loop
    // Work items: mn_tiles_total * k_splits, ordered k-split-last per expert tile
    for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
        // Decompose work_id into (expert_id, m_tile, n_tile, ks_id).
        // Linear scan to find expert_id from expert_offsets.
        int expert_id = 0;
        int tiles_so_far = 0;
        int mn_tiles_e = 0;
        for (int e = 0; e < num_experts; e++) {
            int M_e_tmp = expert_offsets[e + 1] - expert_offsets[e];
            int m_tiles_e = (M_e_tmp + TILE_M - 1) / TILE_M;
            mn_tiles_e = m_tiles_e * n_tiles;
            int expert_total = mn_tiles_e * k_splits;
            if (work_id < tiles_so_far + expert_total) {
                expert_id = e;
                break;
            }
            tiles_so_far += expert_total;
        }

        const int local_work_id = work_id - tiles_so_far;
        const int mn_local = local_work_id / k_splits;
        const int ks_id = local_work_id % k_splits;
        const int n_tile = mn_local % n_tiles;
        const int m_tile = mn_local / n_tiles;

        // K-tile range for this split
        const int kt_start = ks_id * tiles_per_split;
        const int kt_end = min(kt_start + tiles_per_split, k_tiles);
        if (kt_start >= k_tiles)
            continue;

        // Per-expert parameters
        const int a_row_offset = expert_offsets[expert_id];
        const int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
        const int m_base = m_tile * TILE_M;

        // Expert-specific pointers
        const scalar_t* A = A_concat + a_row_offset * K_dim;
        const unsigned int* B_packed = B_packed_all + expert_id * b_packed_per_expert;
        const ABSMAX_T* B_absmax = B_absmax_all + expert_id * b_absmax_per_expert;
        scalar_t* C = C_concat + a_row_offset * N;
        float* C_ws = (k_splits > 1) ? C_workspace + a_row_offset * N : nullptr;

        // Zero accumulators
#pragma unroll
        for (int mb = 0; mb < M_BLOCKS; mb++)
#pragma unroll
            for (int nb = 0; nb < N_BLOCKS; nb++)
                frag_c[mb][nb][0] = frag_c[mb][nb][1] = frag_c[mb][nb][2] = frag_c[mb][nb][3] = 0.0f;

        // Fetch tile lambda
        auto fetch_tile = [&](int stage, int kt) {
            const int k_base = kt * TILE_K;
            const int tile_idx = kt * n_tiles + n_tile;

            // B tile via cp.async
            const int b_global_base = tile_idx * B_STAGE_WORDS;
            constexpr int B_INT4S = B_STAGE_BYTES_VAL / 16;
            const int4* b_src = reinterpret_cast<const int4*>(B_packed + b_global_base);
            int4* b_dst = reinterpret_cast<int4*>(sh_b(stage));
            for (int i = threadIdx.x; i < B_INT4S; i += BLOCK_DIM)
                cp_async_cg_16(&b_dst[i], &b_src[i]);

            // Absmax via cp.async
            const int abs_global_base = tile_idx * ABS_STAGE_ELEMS;
            constexpr int ABS_INT4S = (ABS_STAGE_BYTES + 15) / 16;
            const int4* abs_src = reinterpret_cast<const int4*>(B_absmax + abs_global_base);
            int4* abs_dst = reinterpret_cast<int4*>(sh_abs(stage));
            for (int i = threadIdx.x; i < ABS_INT4S; i += BLOCK_DIM)
                cp_async_cg_16(&abs_dst[i], &abs_src[i]);

            // A tile via cp.async with XOR swizzle
            scalar_t* a_dst = sh_a(stage);
            constexpr int A_GROUPS = A_STAGE_ELEMS / 8;
            const bool a_interior = (m_base + TILE_M <= M_e) && (k_base + TILE_K <= K_dim);

            if (a_interior) {
                for (int i = threadIdx.x; i < A_GROUPS; i += BLOCK_DIM) {
                    int row = i / (TILE_K / 8);
                    int col_group = i % (TILE_K / 8);
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * TILE_K + swizzled_group * 8]);
                    const int4* src =
                        reinterpret_cast<const int4*>(&A[(m_base + row) * K_dim + k_base + col_group * 8]);
                    cp_async_cg_16(dst, src);
                }
            } else {
                for (int i = threadIdx.x; i < A_GROUPS; i += BLOCK_DIM) {
                    int row = i / (TILE_K / 8);
                    int col_group = i % (TILE_K / 8);
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * TILE_K + swizzled_group * 8]);
                    int gr = m_base + row;
                    int gc = k_base + col_group * 8;
                    if (gr < M_e && gc < K_dim) {
                        const int4* src = reinterpret_cast<const int4*>(&A[gr * K_dim + gc]);
                        cp_async_cg_16(dst, src);
                    } else {
                        *dst = make_int4(0, 0, 0, 0);
                    }
                }
            }
        };

        // Compute tile lambda
        auto compute_tile = [&](int stage) {
            scalar_t* a_ptr = sh_a(stage);
            unsigned int* b_ptr = sh_b(stage);
            ABSMAX_T* abs_ptr = sh_abs(stage);

#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int k_block = ks / 2;
                const int half_idx = ks % 2;

                uint32_t frag_a[M_BLOCKS][4];
#pragma unroll
                for (int mb = 0; mb < M_BLOCKS; mb++) {
                    const int mb_row_offset = mb * 16;
                    const int matrix_id = lane_id / 8;
                    const int row_in_matrix = lane_id % 8;
                    const int a_row = mb_row_offset + row_in_matrix + (matrix_id % 2) * 8;
                    const int col_start = ks * 16 + (matrix_id / 2) * 8;
                    const int col_group = col_start / 8;
                    const int swizzled_group = col_group ^ (a_row % 8);
                    const int swizzled_col_start = swizzled_group * 8;

                    const scalar_t* addr = &a_ptr[a_row * TILE_K + swizzled_col_start];
                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(frag_a[mb][0]), "=r"(frag_a[mb][1]), "=r"(frag_a[mb][2]), "=r"(frag_a[mb][3])
                                 : "r"(smem_addr));
                }

#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int col = warp_n_base + nb * 8 + gid;
                    unsigned int planes[K_BITS];
                    int b_addr = col * B_COL_WORDS + k_block * K_BITS;
#pragma unroll
                    for (int b = 0; b < K_BITS; b++)
                        planes[b] = b_ptr[b_addr + b];

                    scalar_t scale = Ops::from_float(load_absmax<ABSMAX_T>(abs_ptr, col * KB_PER_TILE + k_block));

                    const int bit_offset = half_idx * 16;
                    const int rows[4] = {2 * tid, 2 * tid + 1, 2 * tid + 8, 2 * tid + 9};

                    int bp0 = bit_offset + rows[0];
                    int bp1 = bit_offset + rows[1];
                    int bp2 = bit_offset + rows[2];
                    int bp3 = bit_offset + rows[3];

                    int idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
#pragma unroll
                    for (int b = 0; b < K_BITS; b++) {
                        unsigned int p = planes[b];
                        idx0 |= ((p >> bp0) & 1) << b;
                        idx1 |= ((p >> bp1) & 1) << b;
                        idx2 |= ((p >> bp2) & 1) << b;
                        idx3 |= ((p >> bp3) & 1) << b;
                    }

                    scalar_t vals[4];
                    vals[0] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx0), scale);
                    vals[1] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx1), scale);
                    vals[2] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx2), scale);
                    vals[3] = Ops::mul(__shfl_sync(0xFFFFFFFF, cb_val, idx3), scale);

                    uint32_t frag_b[2];
                    frag_b[0] = pack_two<scalar_t>(vals[0], vals[1]);
                    frag_b[1] = pack_two<scalar_t>(vals[2], vals[3]);

#pragma unroll
                    for (int mb = 0; mb < M_BLOCKS; mb++) {
                        mma_m16n8k16<scalar_t>(frag_a[mb], frag_b, frag_c[mb][nb]);
                    }
                }
            }
        };

        // Pipeline: double-buffered cp.async over this split's k-tile range
        fetch_tile(0, kt_start);
        cp_async_fence();

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % 2;
            if (kt + 1 < kt_end) {
                fetch_tile((kt - kt_start + 1) % 2, kt + 1);
                cp_async_fence();
                // L2 prefetch for tile kt+2
                if (kt + 2 < kt_end) {
                    const int pf_tile = (kt + 2) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            compute_tile(cur);
            __syncthreads();
        }

        // Write output
        if (k_splits == 1) {
            // Direct write — this block owns the full K reduction
#pragma unroll
            for (int mb = 0; mb < M_BLOCKS; mb++) {
#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int c_col = n_tile * TILE_N + warp_n_base + nb * 8 + tid * 2;
                    int m_row0 = m_base + mb * 16 + gid;
                    int m_row1 = m_base + mb * 16 + gid + 8;
                    if (m_row0 < M_e) {
                        C[m_row0 * N + c_col] = Ops::from_float(frag_c[mb][nb][0]);
                        C[m_row0 * N + c_col + 1] = Ops::from_float(frag_c[mb][nb][1]);
                    }
                    if (m_row1 < M_e) {
                        C[m_row1 * N + c_col] = Ops::from_float(frag_c[mb][nb][2]);
                        C[m_row1 * N + c_col + 1] = Ops::from_float(frag_c[mb][nb][3]);
                    }
                }
            }
        } else {
            // Partial K — atomicAdd to fp32 workspace, last block converts to output
            // mn_id is the global (expert, m_tile, n_tile) index for tile_counters
            int mn_id = tiles_so_far / k_splits + mn_local;
#pragma unroll
            for (int mb = 0; mb < M_BLOCKS; mb++) {
#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int c_col = n_tile * TILE_N + warp_n_base + nb * 8 + tid * 2;
                    int m_row0 = m_base + mb * 16 + gid;
                    int m_row1 = m_base + mb * 16 + gid + 8;
                    if (m_row0 < M_e) {
                        atomicAdd(&C_ws[m_row0 * N + c_col], frag_c[mb][nb][0]);
                        atomicAdd(&C_ws[m_row0 * N + c_col + 1], frag_c[mb][nb][1]);
                    }
                    if (m_row1 < M_e) {
                        atomicAdd(&C_ws[m_row1 * N + c_col], frag_c[mb][nb][2]);
                        atomicAdd(&C_ws[m_row1 * N + c_col + 1], frag_c[mb][nb][3]);
                    }
                }
            }

            __threadfence();

            __shared__ int is_last;
            if (threadIdx.x == 0) {
                int done = atomicAdd(&tile_counters[mn_id], 1);
                is_last = (done == k_splits - 1) ? 1 : 0;
            }
            __syncthreads();

            if (is_last) {
                for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_DIM) {
                    int row = m_base + i / TILE_N;
                    int col = n_tile * TILE_N + i % TILE_N;
                    if (row < M_e)
                        C[row * N + col] = Ops::from_float(C_ws[row * N + col]);
                }
            }
        }
    } // end persistent work loop
}

// [REMOVED: Warp-specialized and dequant-once grouped GEMM kernels.
//  Both were correct but slower than the baseline on Ada (sm_89) due to
//  register pressure from multiple accumulator sets. See moe-kernel-spec.md
//  for the full analysis. Code removed in dead-code cleanup.]

// Grouped GEMM launcher — supports TILE_N=64/128 and auto k_splits
template <int K, int MB, int TN, typename scalar_t, typename ABSMAX_T = unsigned char>
static void kbitGroupedGemmProdLaunch(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all, const float* codebook,
    scalar_t* C_concat, float* C_workspace, int* tile_counters, const int* expert_offsets, int K_dim, int N,
    int num_experts, int max_M, int num_sms, cudaStream_t stream
) {
    constexpr int TILE_M = MB * 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = TN;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_WORDS = KB_PER_TILE * K;
    constexpr int N_BLOCKS = 2;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8);
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    constexpr int A_STAGE_BYTES = TILE_M * TILE_K * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES = TILE_N * B_COL_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_BYTES = TILE_N * KB_PER_TILE * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    int n_tiles = N / TILE_N;
    int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    int m_tiles_per_expert = (max_M + TILE_M - 1) / TILE_M;
    int mn_tiles = num_experts * m_tiles_per_expert * n_tiles;

    // k_splits heuristic: target enough blocks for good SM occupancy
    constexpr int TARGET_BLOCKS_PER_SM = (BLOCK_DIM <= 128) ? 4 : 1;
    int target_blocks = num_sms * TARGET_BLOCKS_PER_SM;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    int grid_size = (k_splits == 1) ? min(num_sms, total_work) : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int smem_size = 2 * STAGE_BYTES;

    kbit_grouped_gemm_prod<K, MB, TN, scalar_t, ABSMAX_T><<<grid_size, block, smem_size, stream>>>(
        A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, expert_offsets, K_dim, N,
        num_experts, k_splits, total_work
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry point: caller passes max_M, workspace, and tile_counters.
// Chooses TILE_N=64 for small M (m_blocks==1) to improve SM utilization,
// and auto-selects k_splits when there aren't enough MN tiles.
template <int K, typename scalar_t, typename ABSMAX_T = unsigned char>
void kbitGroupedGemmProd(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all, const float* codebook,
    scalar_t* C_concat, float* C_workspace, int* tile_counters, const int* d_expert_offsets, int K_dim, int N,
    int num_experts, int max_M, cudaStream_t stream
) {
    if (max_M == 0 || N == 0)
        return;

    int dev;
    cudaGetDevice(&dev);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int m_blocks = 1;
    if (max_M > 48)
        m_blocks = 4;
    else if (max_M > 32)
        m_blocks = 3;
    else if (max_M > 16)
        m_blocks = 2;

    // Choose TILE_N: use 64 for m_blocks==1 to double n_tiles and improve SM utilization
    const bool use_tn64 = (m_blocks == 1) && (N % 64 == 0);

    if (use_tn64) {
        kbitGroupedGemmProdLaunch<K, 1, 64, scalar_t, ABSMAX_T>(
            A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
            K_dim, N, num_experts, max_M, num_sms, stream
        );
    } else {
        switch (m_blocks) {
        case 4:
            kbitGroupedGemmProdLaunch<K, 4, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        case 3:
            kbitGroupedGemmProdLaunch<K, 3, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        case 2:
            kbitGroupedGemmProdLaunch<K, 2, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        default:
            kbitGroupedGemmProdLaunch<K, 1, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        }
    }
}

// ===================================================================
// Scalar GEMV kernel: C[M,N] = A[M,K_dim] * W_kbit^T  (M=1..4)
// ===================================================================
//
// C=1 architecture: 1 output column per block, 2 warps (64 threads) split K.
// Grid = N (direct mapping). No split-K, no workspace.
// int4 vector loads for A, dequant-once loop: weights decoded once, FMA'd across M rows.
// Fully unrolled with __launch_bounds__ controlling register budget.
// Two-phase shared memory reduction (warp shuffle + shmem).
// Supports both flat (quantize_kbit) and tiled (repack_kbit) B layouts.

template <int K_BITS, int M_VAL, bool TILED = false, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(64, M_VAL <= 2 ? 24 : 16) kbit_scalar_gemv(
    const scalar_t* __restrict__ A,
    const unsigned int* __restrict__ B_packed, // flat or tiled
    const ABSMAX_T* __restrict__ B_absmax,     // flat or tiled
    const float* __restrict__ codebook, scalar_t* __restrict__ C, const int M, const int K_dim, const int N
) {
    constexpr int BS = 32; // quantization block size
    constexpr int BLOCK_SIZE = 64;
    constexpr int NUM_WARPS = 2;
    constexpr int M_MAX = 4;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int col = blockIdx.x;

    const int num_k_blocks = K_dim / BS;

    // Codebook in registers (shuffle-based lookup)
    float cb = (lane_id < (1 << K_BITS)) ? codebook[lane_id] : 0.0f;

    // Tiled layout constants (only used when TILED=true)
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int KB_PER_TILE = TILE_K / BS; // 2
    constexpr int WORDS_PER_TILE = TILE_N * KB_PER_TILE * K_BITS;
    constexpr int ABS_PER_TILE = TILE_N * KB_PER_TILE;

    // Flat layout: column base pointers
    const unsigned int* B_col = nullptr;
    const ABSMAX_T* abs_col = nullptr;

    // Tiled layout: per-column tile coordinates
    int n_tile = 0, col_in_tile = 0, n_tiles = 0;

    if constexpr (!TILED) {
        B_col = B_packed + col * num_k_blocks * K_BITS;
        abs_col = B_absmax + col * num_k_blocks;
    } else {
        n_tiles = N / TILE_N;
        n_tile = col / TILE_N;
        col_in_tile = col % TILE_N;
    }

    // Accumulators
    float acc[M_VAL];
#pragma unroll
    for (int m = 0; m < M_VAL; m++)
        acc[m] = 0.0f;

    // 64 threads stride through K blocks: thread t handles blocks t, t+64, t+128, ...
    // max_iters ensures all lanes iterate the same number of times (no warp divergence at __shfl_sync).
    const int max_iters = (num_k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < max_iters; iter++) {
        const int block_idx = threadIdx.x + iter * BLOCK_SIZE;
        const bool valid = (block_idx < num_k_blocks);

        // Compute word base address for this K-block's bit-plane data
        int word_base;
        int abs_idx;
        if constexpr (!TILED) {
            word_base = block_idx * K_BITS;
            abs_idx = block_idx;
        } else {
            const int k_tile = block_idx / KB_PER_TILE;
            const int kb = block_idx % KB_PER_TILE;
            const int tile_base = k_tile * n_tiles + n_tile;
            word_base = tile_base * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + kb) * K_BITS;
            abs_idx = tile_base * ABS_PER_TILE + col_in_tile * KB_PER_TILE + kb;
        }

        // L2 prefetch for next iteration's B data
        {
            const int next_block_idx = block_idx + BLOCK_SIZE;
            if (next_block_idx < num_k_blocks) {
                if constexpr (!TILED) {
                    prefetch_l2(&B_col[next_block_idx * K_BITS]);
                } else {
                    const int nk_tile = next_block_idx / KB_PER_TILE;
                    const int nkb = next_block_idx % KB_PER_TILE;
                    const int ntb = nk_tile * n_tiles + n_tile;
                    prefetch_l2(&B_packed[ntb * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + nkb) * K_BITS]);
                }
            }
        }

        // Load k bit-plane words (guarded; invalid threads get 0)
        // Vector loads for power-of-2 K_BITS, scalar for others.
        const unsigned int* B_src = TILED ? B_packed : B_col;
        unsigned int planes[K_BITS];
        if constexpr (K_BITS == 2) {
            uint2 pv = valid ? *reinterpret_cast<const uint2*>(&B_src[word_base]) : make_uint2(0u, 0u);
            planes[0] = pv.x;
            planes[1] = pv.y;
        } else if constexpr (K_BITS == 4) {
            int4 pv;
            if (valid)
                pv = *reinterpret_cast<const int4*>(&B_src[word_base]);
            else {
                pv.x = 0;
                pv.y = 0;
                pv.z = 0;
                pv.w = 0;
            }
            planes[0] = (unsigned int)pv.x;
            planes[1] = (unsigned int)pv.y;
            planes[2] = (unsigned int)pv.z;
            planes[3] = (unsigned int)pv.w;
        } else {
#pragma unroll
            for (int b = 0; b < K_BITS; b++)
                planes[b] = valid ? B_src[word_base + b] : 0u;
        }

        // Load absmax (guarded; invalid threads get 0)
        const ABSMAX_T* abs_src = TILED ? B_absmax : abs_col;
        float amax = valid ? load_absmax(abs_src, abs_idx) : 0.0f;

        const int k_base = block_idx * BS;

// Dequant-once loop: decode weight once per element, FMA across all M rows.
// sub iterates 4 groups of 8 elements within the 32-element quant block.
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            // Load A for all M rows (int4 = 8 fp16 values each)
            int4 av[M_VAL];
#pragma unroll
            for (int m = 0; m < M_VAL; m++) {
                if (valid)
                    av[m] = *reinterpret_cast<const int4*>(&A[m * K_dim + k_base + sub * 8]);
            }

// Dequant each element once, then FMA across M rows
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int idx = 0;
#pragma unroll
                for (int b = 0; b < K_BITS; b++)
                    idx |= ((planes[b] >> (sub * 8 + j)) & 1) << b;
                float w = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;

#pragma unroll
                for (int m = 0; m < M_VAL; m++) {
                    const scalar_t* ap = reinterpret_cast<const scalar_t*>(&av[m]);
                    if (valid)
                        acc[m] += w * ScalarOps<scalar_t>::to_float(ap[j]);
                }
            }
        }
    }

// Phase 1: Intra-warp reduction via shuffle
#pragma unroll
    for (int m = 0; m < M_VAL; m++) {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2)
            acc[m] += __shfl_down_sync(0xFFFFFFFF, acc[m], offset);
    }

    // Phase 2: Inter-warp reduction via shared memory (2 warps)
    __shared__ float s_partial[NUM_WARPS * M_MAX];

    if (lane_id == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++)
            s_partial[warp_id * M_MAX + m] = acc[m];
    }
    __syncthreads();

    // Thread 0 sums all warps and writes output
    if (threadIdx.x == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++) {
            if (m < M) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    sum += s_partial[w * M_MAX + m];
                C[m * N + col] = ScalarOps<scalar_t>::from_float(sum);
            }
        }
    }
}

// ---- Scalar GEMV launcher ----
template <int K, int MV, bool TILED, typename scalar_t, typename ABSMAX_T>
static void kbitScalarGemvLaunch(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    int M, int K_dim, int N, cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 64;
    int grid_size = N;

    kbit_scalar_gemv<K, MV, TILED, scalar_t, ABSMAX_T>
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(A, B_packed, B_absmax, codebook, C, M, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry point: selects M_VAL template (flat layout)
template <int K, typename scalar_t, typename ABSMAX_T>
void kbitScalarGemv(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    int M, int K_dim, int N, cudaStream_t stream
) {
#define LAUNCH_SCALAR_GEMV(MV)                                                                                         \
    kbitScalarGemvLaunch<K, MV, false, scalar_t, ABSMAX_T>(A, B_packed, B_absmax, codebook, C, M, K_dim, N, stream)

    if (M <= 1) {
        LAUNCH_SCALAR_GEMV(1);
    } else if (M <= 2) {
        LAUNCH_SCALAR_GEMV(2);
    } else if (M <= 3) {
        LAUNCH_SCALAR_GEMV(3);
    } else {
        LAUNCH_SCALAR_GEMV(4);
    }

#undef LAUNCH_SCALAR_GEMV
}

// Public entry point: selects M_VAL template (tiled layout)
template <int K, typename scalar_t, typename ABSMAX_T>
void kbitScalarGemvTiled(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    int M, int K_dim, int N, cudaStream_t stream
) {
#define LAUNCH_SCALAR_GEMV_TILED(MV)                                                                                   \
    kbitScalarGemvLaunch<K, MV, true, scalar_t, ABSMAX_T>(A, B_packed, B_absmax, codebook, C, M, K_dim, N, stream)

    if (M <= 1) {
        LAUNCH_SCALAR_GEMV_TILED(1);
    } else if (M <= 2) {
        LAUNCH_SCALAR_GEMV_TILED(2);
    } else if (M <= 3) {
        LAUNCH_SCALAR_GEMV_TILED(3);
    } else {
        LAUNCH_SCALAR_GEMV_TILED(4);
    }

#undef LAUNCH_SCALAR_GEMV_TILED
}

// ---- Tiled Scalar GEMV v2 ----
// Cooperative tile loading into shared memory with split-K for occupancy.
// Grid = n_tiles * k_splits, Block = 128 threads (4 warps).
// Each thread handles one column within an N-tile.
// Double-buffered cp.async pipeline for B + absmax tiles.
// A loaded directly from global memory (L1 broadcast across columns).

template <int K_BITS, int M_VAL, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(128, 8) kbit_scalar_gemv_tiled_v2(
    const scalar_t* __restrict__ A,
    const unsigned int* __restrict__ B_packed,
    const ABSMAX_T* __restrict__ B_absmax,
    const float* __restrict__ codebook,
    scalar_t* __restrict__ C,
    float* __restrict__ C_workspace,
    int* __restrict__ tile_counters,
    const int M, const int K_dim, const int N, const int k_splits
) {
    constexpr int BS = 32;            // quantization block size
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int BLOCK_DIM = 128;    // threads per block
    constexpr int NUM_WARPS = 4;
    constexpr int M_MAX = 4;
    constexpr int KB_PER_TILE = TILE_K / BS;  // 2
    constexpr int B_COL_WORDS = KB_PER_TILE * K_BITS;
    constexpr int B_STAGE_WORDS = TILE_N * B_COL_WORDS;
    constexpr int B_STAGE_BYTES = B_STAGE_WORDS * (int)sizeof(unsigned int);
    constexpr int ABS_STAGE_ELEMS = TILE_N * KB_PER_TILE;
    constexpr int ABS_STAGE_BYTES = ABS_STAGE_ELEMS * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    const int tiles_per_split = (k_tiles + k_splits - 1) / k_splits;

    // Work item: which N-tile and K-split
    const int work_id = blockIdx.x;
    const int n_tile = work_id / k_splits;
    const int ks_id = work_id % k_splits;
    const int n_base = n_tile * TILE_N;

    const int kt_start = ks_id * tiles_per_split;
    const int kt_end = min(kt_start + tiles_per_split, k_tiles);
    if (kt_start >= k_tiles) return;

    // This thread's column within the tile
    const int col_in_tile = threadIdx.x;  // 0..127
    const int col = n_base + col_in_tile;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Codebook in registers (shuffle-based lookup)
    float cb = (lane_id < (1 << K_BITS)) ? codebook[lane_id] : 0.0f;

    // Double-buffered shared memory
    extern __shared__ char smem[];
    auto sh_b = [&](int stage) -> unsigned int* {
        return reinterpret_cast<unsigned int*>(smem + stage * STAGE_BYTES);
    };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(smem + stage * STAGE_BYTES + B_STAGE_BYTES);
    };

    // Accumulators
    float acc[M_VAL];
    #pragma unroll
    for (int m = 0; m < M_VAL; m++) acc[m] = 0.0f;

    // Fetch tile: cooperative cp.async loading of B + absmax
    auto fetch_tile = [&](int stage, int kt) {
        const int tile_idx = kt * n_tiles + n_tile;  // K-major tile ordering

        // B tile via cp.async (all 128 threads cooperatively load)
        const int b_global_base = tile_idx * B_STAGE_WORDS;
        constexpr int B_INT4S = B_STAGE_BYTES / 16;
        const int4* b_src = reinterpret_cast<const int4*>(B_packed + b_global_base);
        int4* b_dst = reinterpret_cast<int4*>(sh_b(stage));
        for (int i = threadIdx.x; i < B_INT4S; i += BLOCK_DIM)
            cp_async_cg_16(&b_dst[i], &b_src[i]);

        // Absmax via cp.async
        const int abs_global_base = tile_idx * ABS_STAGE_ELEMS;
        constexpr int ABS_INT4S = (ABS_STAGE_BYTES + 15) / 16;
        const int4* abs_src = reinterpret_cast<const int4*>(B_absmax + abs_global_base);
        int4* abs_dst = reinterpret_cast<int4*>(sh_abs(stage));
        for (int i = threadIdx.x; i < ABS_INT4S; i += BLOCK_DIM)
            cp_async_cg_16(&abs_dst[i], &abs_src[i]);
    };

    // Compute tile: each thread reads its column from shared memory
    auto compute_tile = [&](int stage, int kt) {
        unsigned int* b_ptr = sh_b(stage);
        ABSMAX_T* abs_ptr = sh_abs(stage);
        const int k_base = kt * TILE_K;

        // Process KB_PER_TILE (=2) K-blocks within this tile
        #pragma unroll
        for (int kb = 0; kb < KB_PER_TILE; kb++) {
            const int block_k_base = k_base + kb * BS;
            if (block_k_base >= K_dim) continue;

            // Read bit-planes from shared memory for this column
            int b_addr = col_in_tile * B_COL_WORDS + kb * K_BITS;
            unsigned int planes[K_BITS];
            if constexpr (K_BITS == 2) {
                uint2 pv = *reinterpret_cast<const uint2*>(&b_ptr[b_addr]);
                planes[0] = pv.x; planes[1] = pv.y;
            } else if constexpr (K_BITS == 4) {
                int4 pv = *reinterpret_cast<const int4*>(&b_ptr[b_addr]);
                planes[0] = (unsigned int)pv.x; planes[1] = (unsigned int)pv.y;
                planes[2] = (unsigned int)pv.z; planes[3] = (unsigned int)pv.w;
            } else {
                #pragma unroll
                for (int b = 0; b < K_BITS; b++)
                    planes[b] = b_ptr[b_addr + b];
            }

            // Load absmax from shared memory
            float amax = load_absmax(abs_ptr, col_in_tile * KB_PER_TILE + kb);

            // Dequant-once loop: decode weight once, FMA across M rows
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                // Load A for all M rows (int4 = 8 fp16 values)
                int4 av[M_VAL];
                #pragma unroll
                for (int m = 0; m < M_VAL; m++)
                    av[m] = *reinterpret_cast<const int4*>(&A[m * K_dim + block_k_base + sub * 8]);

                // Dequant each element once, then FMA across M rows
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int idx = 0;
                    #pragma unroll
                    for (int b = 0; b < K_BITS; b++)
                        idx |= ((planes[b] >> (sub * 8 + j)) & 1) << b;
                    float w = __shfl_sync(0xFFFFFFFF, cb, idx) * amax;

                    #pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        const scalar_t* ap = reinterpret_cast<const scalar_t*>(&av[m]);
                        acc[m] += w * ScalarOps<scalar_t>::to_float(ap[j]);
                    }
                }
            }
        }
    };

    // Pipeline: double-buffered cp.async
    fetch_tile(0, kt_start);
    cp_async_fence();

    for (int kt = kt_start; kt < kt_end; kt++) {
        int cur = (kt - kt_start) % 2;
        if (kt + 1 < kt_end) {
            fetch_tile((kt + 1 - kt_start) % 2, kt + 1);
            cp_async_fence();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        compute_tile(cur, kt);
        __syncthreads();
    }

    // Write output
    if (k_splits == 1) {
        // Direct write — this block owns the full K reduction
        #pragma unroll
        for (int m = 0; m < M_VAL; m++) {
            if (m < M && col < N)
                C[m * N + col] = ScalarOps<scalar_t>::from_float(acc[m]);
        }
    } else {
        // Partial K — atomicAdd to workspace
        #pragma unroll
        for (int m = 0; m < M_VAL; m++) {
            if (m < M && col < N)
                atomicAdd(&C_workspace[m * N + col], acc[m]);
        }

        __threadfence();

        // Last-arriving split converts workspace to output
        __shared__ int is_last;
        if (threadIdx.x == 0) {
            int done = atomicAdd(&tile_counters[n_tile], 1);
            is_last = (done == k_splits - 1) ? 1 : 0;
        }
        __syncthreads();

        if (is_last) {
            for (int i = threadIdx.x; i < M_VAL * TILE_N; i += BLOCK_DIM) {
                int m = i / TILE_N;
                int c = n_base + i % TILE_N;
                if (m < M && c < N)
                    C[m * N + c] = ScalarOps<scalar_t>::from_float(C_workspace[m * N + c]);
            }
        }
    }
}

// ---- Tiled GEMV v2 launcher ----
template <int K, int MV, typename scalar_t, typename ABSMAX_T>
static void kbitScalarGemvTiledV2Launch(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax,
    const float* codebook, scalar_t* C, float* C_workspace, int* tile_counters,
    int M, int K_dim, int N, int num_sms, cudaStream_t stream
) {
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    constexpr int BLOCK_DIM = 128;
    constexpr int BS = 32;
    constexpr int KB_PER_TILE = TILE_K / BS;
    constexpr int B_COL_WORDS = KB_PER_TILE * K;
    constexpr int B_STAGE_BYTES = TILE_N * B_COL_WORDS * (int)sizeof(unsigned int);
    constexpr int ABS_STAGE_BYTES = TILE_N * KB_PER_TILE * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    int n_tiles = N / TILE_N;
    int k_tiles = (K_dim + TILE_K - 1) / TILE_K;

    // Choose k_splits to achieve ~4 blocks per SM.
    // Recompute from tiles_per_split to guarantee no empty splits
    // (empty splits would skip the tile_counters atomicAdd, breaking the last-block check).
    int target_blocks = num_sms * 4;
    int k_splits = max(1, (target_blocks + n_tiles - 1) / n_tiles);
    k_splits = min(k_splits, k_tiles);
    int tiles_per_split = (k_tiles + k_splits - 1) / k_splits;
    k_splits = (k_tiles + tiles_per_split - 1) / tiles_per_split;  // no empty splits

    int grid_size = n_tiles * k_splits;
    int smem_size = 2 * STAGE_BYTES;

    kbit_scalar_gemv_tiled_v2<K, MV, scalar_t, ABSMAX_T>
        <<<grid_size, BLOCK_DIM, smem_size, stream>>>(
            A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters,
            M, K_dim, N, k_splits
        );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry point: selects M_VAL template, queries num_sms internally
template <int K, typename scalar_t, typename ABSMAX_T>
void kbitScalarGemvTiledV2(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax,
    const float* codebook, scalar_t* C, float* C_workspace, int* tile_counters,
    int M, int K_dim, int N, cudaStream_t stream
) {
    int dev;
    cudaGetDevice(&dev);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

#define LAUNCH_GEMV_V2(MV) \
    kbitScalarGemvTiledV2Launch<K, MV, scalar_t, ABSMAX_T>( \
        A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, \
        M, K_dim, N, num_sms, stream)

    if (M <= 1)      { LAUNCH_GEMV_V2(1); }
    else if (M <= 2) { LAUNCH_GEMV_V2(2); }
    else if (M <= 3) { LAUNCH_GEMV_V2(3); }
    else             { LAUNCH_GEMV_V2(4); }

#undef LAUNCH_GEMV_V2
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
                 : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]), "r"(frag_b[0]), "r"(frag_b[1]),
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
    template void quantizeBlockwise_kbit<T, K>(const float*, const T*, unsigned char*, unsigned int*, int, cudaStream_t);

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

// float32 absmax (from quantize_kbit output directly)
INSTANTIATE_KBIT_DEQUANT(half, 2, float)
INSTANTIATE_KBIT_DEQUANT(half, 3, float)
INSTANTIATE_KBIT_DEQUANT(half, 4, float)
INSTANTIATE_KBIT_DEQUANT(half, 5, float)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 2, float)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 3, float)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 4, float)
INSTANTIATE_KBIT_DEQUANT(__nv_bfloat16, 5, float)
INSTANTIATE_KBIT_DEQUANT(float, 2, float)
INSTANTIATE_KBIT_DEQUANT(float, 3, float)
INSTANTIATE_KBIT_DEQUANT(float, 4, float)
INSTANTIATE_KBIT_DEQUANT(float, 5, float)

// Tiled dequant instantiations: all output types × absmax types × K values
#define INSTANTIATE_KBIT_DEQUANT_TILED(T, K, ABSMAX_T)                                                                 \
    template void dequantizeBlockwise_kbit_tiled<T, K, ABSMAX_T>(                                                      \
        const unsigned int*, const float*, const ABSMAX_T*, T*, int, int, cudaStream_t                                 \
    );

// uint8 E4M4 absmax
INSTANTIATE_KBIT_DEQUANT_TILED(half, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 5, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 5, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 2, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 3, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 4, unsigned char)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 5, unsigned char)

// fp16 absmax
INSTANTIATE_KBIT_DEQUANT_TILED(half, 2, half)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 3, half)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 4, half)
INSTANTIATE_KBIT_DEQUANT_TILED(half, 5, half)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 2, half)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 3, half)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 4, half)
INSTANTIATE_KBIT_DEQUANT_TILED(__nv_bfloat16, 5, half)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 2, half)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 3, half)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 4, half)
INSTANTIATE_KBIT_DEQUANT_TILED(float, 5, half)

// Repack instantiations: one per K value
#define INSTANTIATE_KBIT_REPACK(K)                                                                                     \
    template void repackKbit<K>(const unsigned int*, const unsigned char*, unsigned int*, unsigned char*, int, int, cudaStream_t);

INSTANTIATE_KBIT_REPACK(2)
INSTANTIATE_KBIT_REPACK(3)
INSTANTIATE_KBIT_REPACK(4)
INSTANTIATE_KBIT_REPACK(5)

// Production kernel instantiations — uint8 E4M4 absmax (default)
#define INSTANTIATE_KBIT_GEMM_PROD_U8(K)                                                                               \
    template void kbitGemmProd<K, half, unsigned char>(                                                                \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, float*, int*, int, int, int, int, \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void kbitGemmProd<K, __nv_bfloat16, unsigned char>(                                                       \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, float*, int*,   \
        int, int, int, int, cudaStream_t                                                                               \
    );
INSTANTIATE_KBIT_GEMM_PROD_U8(2)
INSTANTIATE_KBIT_GEMM_PROD_U8(3)
INSTANTIATE_KBIT_GEMM_PROD_U8(4)
INSTANTIATE_KBIT_GEMM_PROD_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_GEMM_PROD_FP16(K)                                                                             \
    template void kbitGemmProd<K, half, half>(                                                                         \
        const half*, const unsigned int*, const half*, const float*, half*, float*, int*, int, int, int, int,          \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void kbitGemmProd<K, __nv_bfloat16, half>(                                                                \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, float*, int*, int, int,  \
        int, int, cudaStream_t                                                                                         \
    );
INSTANTIATE_KBIT_GEMM_PROD_FP16(2)
INSTANTIATE_KBIT_GEMM_PROD_FP16(3)
INSTANTIATE_KBIT_GEMM_PROD_FP16(4)
INSTANTIATE_KBIT_GEMM_PROD_FP16(5)

// Grouped expert GEMM instantiations — uint8 E4M4 absmax (default)
#define INSTANTIATE_KBIT_GROUPED_GEMM_PROD_U8(K)                                                                       \
    template void kbitGroupedGemmProd<K, half, unsigned char>(                                                         \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, float*, int*, const int*, int,    \
        int, int, int, cudaStream_t                                                                                    \
    );                                                                                                                 \
    template void kbitGroupedGemmProd<K, __nv_bfloat16, unsigned char>(                                                \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, float*, int*,   \
        const int*, int, int, int, int, cudaStream_t                                                                   \
    );
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_U8(2)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_U8(3)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_U8(4)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_GROUPED_GEMM_PROD_FP16(K)                                                                     \
    template void kbitGroupedGemmProd<K, half, half>(                                                                  \
        const half*, const unsigned int*, const half*, const float*, half*, float*, int*, const int*, int, int, int,   \
        int, cudaStream_t                                                                                              \
    );                                                                                                                 \
    template void kbitGroupedGemmProd<K, __nv_bfloat16, half>(                                                         \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, float*, int*,            \
        const int*, int, int, int, int, cudaStream_t                                                                   \
    );
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_FP16(2)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_FP16(3)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_FP16(4)
INSTANTIATE_KBIT_GROUPED_GEMM_PROD_FP16(5)

// Scalar GEMV instantiations — flat layout, C=1
// uint8 E4M4 absmax (default)
#define INSTANTIATE_KBIT_SCALAR_GEMV_U8(K)                                                                             \
    template void kbitScalarGemv<K, half, unsigned char>(                                                              \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, int, int, int, cudaStream_t       \
    );                                                                                                                 \
    template void kbitScalarGemv<K, __nv_bfloat16, unsigned char>(                                                     \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, int, int, int,  \
        cudaStream_t                                                                                                   \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_U8(2)
INSTANTIATE_KBIT_SCALAR_GEMV_U8(3)
INSTANTIATE_KBIT_SCALAR_GEMV_U8(4)
INSTANTIATE_KBIT_SCALAR_GEMV_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_FP16(K)                                                                           \
    template void kbitScalarGemv<K, half, half>(                                                                       \
        const half*, const unsigned int*, const half*, const float*, half*, int, int, int, cudaStream_t                \
    );                                                                                                                 \
    template void kbitScalarGemv<K, __nv_bfloat16, half>(                                                              \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, int, int, int,           \
        cudaStream_t                                                                                                   \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_FP16(2)
INSTANTIATE_KBIT_SCALAR_GEMV_FP16(3)
INSTANTIATE_KBIT_SCALAR_GEMV_FP16(4)
INSTANTIATE_KBIT_SCALAR_GEMV_FP16(5)
// Scalar GEMV instantiations — tiled layout
// uint8 E4M4 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_TILED_U8(K)                                                                       \
    template void kbitScalarGemvTiled<K, half, unsigned char>(                                                         \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, int, int, int, cudaStream_t       \
    );                                                                                                                 \
    template void kbitScalarGemvTiled<K, __nv_bfloat16, unsigned char>(                                                \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, int, int, int,  \
        cudaStream_t                                                                                                   \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_U8(2)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_U8(3)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_U8(4)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_TILED_FP16(K)                                                                     \
    template void kbitScalarGemvTiled<K, half, half>(                                                                  \
        const half*, const unsigned int*, const half*, const float*, half*, int, int, int, cudaStream_t                \
    );                                                                                                                 \
    template void kbitScalarGemvTiled<K, __nv_bfloat16, half>(                                                         \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, int, int, int,           \
        cudaStream_t                                                                                                   \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_FP16(2)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_FP16(3)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_FP16(4)
INSTANTIATE_KBIT_SCALAR_GEMV_TILED_FP16(5)
// Scalar GEMV v2 (tiled with shared memory) instantiations — uint8 E4M4 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(K)                                                                          \
    template void kbitScalarGemvTiledV2<K, half, unsigned char>(                                                        \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, float*, int*,                      \
        int, int, int, cudaStream_t                                                                                    \
    );                                                                                                                 \
    template void kbitScalarGemvTiledV2<K, __nv_bfloat16, unsigned char>(                                              \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, float*, int*,    \
        int, int, int, cudaStream_t                                                                                    \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(2)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(3)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(4)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(K)                                                                        \
    template void kbitScalarGemvTiledV2<K, half, half>(                                                                \
        const half*, const unsigned int*, const half*, const float*, half*, float*, int*,                              \
        int, int, int, cudaStream_t                                                                                    \
    );                                                                                                                 \
    template void kbitScalarGemvTiledV2<K, __nv_bfloat16, half>(                                                       \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, float*, int*,            \
        int, int, int, cudaStream_t                                                                                    \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(2)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(3)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(4)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(5)
