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

// ---- VQ Generalized Template Infrastructure ----
// VQTraits: compile-time constants for all VQ kernel configurations.
// Parameterized on P_VAL (vector dimension) and INDEX_BITS (codebook index width).
//
// Target configurations:
//   8-bit/p=4  → 2.00 bits/wt, BS=32, 256 entries, 2 KB shmem
//   8-bit/p=3  → 2.67 bits/wt, BS=48, 256 entries, 2 KB shmem
//  10-bit/p=3  → 3.33 bits/wt, BS=48, 1024 entries, 8 KB shmem
//   8-bit/p=2  → 4.00 bits/wt, BS=32, 256 entries, 1 KB shmem
//  10-bit/p=2  → 5.00 bits/wt, BS=32, 1024 entries, 4 KB shmem

template <int P_VAL, int INDEX_BITS = 8>
struct VQTraits {
    static constexpr int BS = (P_VAL == 3) ? 48 : 32;
    static constexpr int CB_ENTRIES = 1 << INDEX_BITS;        // 256 (8-bit) or 1024 (10-bit)
    static constexpr int GROUPS = BS / P_VAL;                 // always exact division
    static constexpr int TOTAL_BITS = GROUPS * INDEX_BITS;
    static constexpr int WORDS = (TOTAL_BITS + 31) / 32;     // uint32 words per block
    static constexpr int CB_PLANES = (P_VAL + 1) / 2;        // 1 for p=2, 2 for p=3/4
    static constexpr int CB_SHMEM_BYTES = CB_PLANES * CB_ENTRIES * (int)sizeof(half2);
    static constexpr int TILE_K = 2 * BS;                     // 64 (BS=32) or 96 (BS=48)
    static constexpr int TILE_N = 128;
    static constexpr int KB_PER_TILE = 2;
    static constexpr int WORDS_PER_TILE = TILE_N * KB_PER_TILE * WORDS;
    static constexpr int ABS_PER_TILE = TILE_N * KB_PER_TILE;
};

// extract_index: extract the i-th codebook index from packed uint32 words.
// 8-bit: fast byte extraction. 10-bit: general bit-shift with cross-boundary OR.
template <int INDEX_BITS>
__device__ __forceinline__ int vq_extract_index(const unsigned int* words, int i) {
    if constexpr (INDEX_BITS == 8) {
        return (words[i >> 2] >> ((i & 3) << 3)) & 0xFF;
    } else {
        constexpr unsigned int MASK = (1u << INDEX_BITS) - 1u;
        const int bit = i * INDEX_BITS;
        const int w = bit >> 5;       // bit / 32
        const int off = bit & 31;     // bit % 32
        unsigned int val = words[w] >> off;
        if (off > 32 - INDEX_BITS)    // crosses uint32 boundary
            val |= words[w + 1] << (32 - off);
        return (int)(val & MASK);
    }
}

// cb_lookup: read P_VAL fp16 values from the shared memory codebook.
// Codebook layout in shared memory (contiguous, padded to 4 bytes or 8 bytes):
//   p=2: half2[CB_ENTRIES] — 1 read (4 bytes)
//   p=3: contiguous 8-byte records: (val0,val1,val2,pad) — 1 int2 read (8 bytes)
//   p=4: contiguous 8-byte records: (val0,val1,val2,val3) — 1 int2 read (8 bytes)
template <int P_VAL, int CB_ENTRIES>
__device__ __forceinline__ void vq_cb_lookup(const half2* s_cb, int idx, float* out) {
    if constexpr (P_VAL == 2) {
        half2 v0 = s_cb[idx];
        out[0] = __half2float(v0.x);
        out[1] = __half2float(v0.y);
    } else {
        // p=3/4: single 8-byte read from contiguous padded layout
        const int2* cb_i2 = reinterpret_cast<const int2*>(s_cb);
        int2 packed = cb_i2[idx];
        half2 v0, v1;
        v0 = *reinterpret_cast<half2*>(&packed.x);
        v1 = *reinterpret_cast<half2*>(&packed.y);
        out[0] = __half2float(v0.x);
        out[1] = __half2float(v0.y);
        out[2] = __half2float(v1.x);
        if constexpr (P_VAL == 4)
            out[3] = __half2float(v1.y);
    }
}

// load_codebook: load the codebook from global memory into shared memory.
// Contiguous padded layout: each entry is 8 bytes for p>=3 (padded to 4 halves).
template <int P_VAL, int CB_ENTRIES, int BLOCK_SIZE>
__device__ __forceinline__ void vq_load_codebook(half2* s_cb, const half* codebook) {
    if constexpr (P_VAL == 2) {
        // [CB_ENTRIES, 2] fp16 viewed as half2[CB_ENTRIES]
        const half2* cb_src = reinterpret_cast<const half2*>(codebook);
        for (int i = threadIdx.x; i < CB_ENTRIES; i += BLOCK_SIZE)
            s_cb[i] = cb_src[i];
    } else if constexpr (P_VAL == 3) {
        // [CB_ENTRIES, 3] fp16 → contiguous 8-byte records: (val0, val1, val2, pad=0)
        const half* cb_half = codebook;
        for (int i = threadIdx.x; i < CB_ENTRIES; i += BLOCK_SIZE) {
            half h0 = cb_half[i * 3 + 0];
            half h1 = cb_half[i * 3 + 1];
            half h2 = cb_half[i * 3 + 2];
            s_cb[i * 2]     = __halves2half2(h0, h1);
            s_cb[i * 2 + 1] = __halves2half2(h2, __float2half(0.0f));
        }
    } else {
        // p=4: [CB_ENTRIES, 4] fp16 → contiguous 8-byte records: (val0, val1, val2, val3)
        const half2* cb_src = reinterpret_cast<const half2*>(codebook);
        for (int i = threadIdx.x; i < CB_ENTRIES; i += BLOCK_SIZE) {
            s_cb[i * 2]     = cb_src[i * 2];
            s_cb[i * 2 + 1] = cb_src[i * 2 + 1];
        }
    }
}

// Static assertions to verify VQTraits for all 5 target configurations
static_assert(VQTraits<4, 8>::BS == 32 && VQTraits<4, 8>::CB_ENTRIES == 256 &&
    VQTraits<4, 8>::GROUPS == 8 && VQTraits<4, 8>::WORDS == 2 &&
    VQTraits<4, 8>::CB_PLANES == 2 && VQTraits<4, 8>::CB_SHMEM_BYTES == 2048 &&
    VQTraits<4, 8>::TILE_K == 64, "VQTraits<4,8> mismatch");

static_assert(VQTraits<3, 8>::BS == 48 && VQTraits<3, 8>::CB_ENTRIES == 256 &&
    VQTraits<3, 8>::GROUPS == 16 && VQTraits<3, 8>::WORDS == 4 &&
    VQTraits<3, 8>::CB_PLANES == 2 && VQTraits<3, 8>::CB_SHMEM_BYTES == 2048 &&
    VQTraits<3, 8>::TILE_K == 96, "VQTraits<3,8> mismatch");

static_assert(VQTraits<3, 10>::BS == 48 && VQTraits<3, 10>::CB_ENTRIES == 1024 &&
    VQTraits<3, 10>::GROUPS == 16 && VQTraits<3, 10>::WORDS == 5 &&
    VQTraits<3, 10>::CB_PLANES == 2 && VQTraits<3, 10>::CB_SHMEM_BYTES == 8192 &&
    VQTraits<3, 10>::TILE_K == 96, "VQTraits<3,10> mismatch");

static_assert(VQTraits<2, 8>::BS == 32 && VQTraits<2, 8>::CB_ENTRIES == 256 &&
    VQTraits<2, 8>::GROUPS == 16 && VQTraits<2, 8>::WORDS == 4 &&
    VQTraits<2, 8>::CB_PLANES == 1 && VQTraits<2, 8>::CB_SHMEM_BYTES == 1024 &&
    VQTraits<2, 8>::TILE_K == 64, "VQTraits<2,8> mismatch");

static_assert(VQTraits<2, 10>::BS == 32 && VQTraits<2, 10>::CB_ENTRIES == 1024 &&
    VQTraits<2, 10>::GROUPS == 16 && VQTraits<2, 10>::WORDS == 5 &&
    VQTraits<2, 10>::CB_PLANES == 1 && VQTraits<2, 10>::CB_SHMEM_BYTES == 4096 &&
    VQTraits<2, 10>::TILE_K == 64, "VQTraits<2,10> mismatch");

// Dummy kernel to verify helpers instantiate for all 5 (P_VAL, INDEX_BITS) configs
template <int P_VAL, int INDEX_BITS>
__global__ void __launch_bounds__(64)
vq_verify_helpers_dummy(const unsigned int* words_in, const half* cb_in, float* out) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    __shared__ half2 s_cb[Traits::CB_PLANES * Traits::CB_ENTRIES];
    vq_load_codebook<P_VAL, Traits::CB_ENTRIES, 64>(s_cb, cb_in);
    __syncthreads();
    int idx = vq_extract_index<INDEX_BITS>(words_in, threadIdx.x % Traits::GROUPS);
    float vals[P_VAL];
    vq_cb_lookup<P_VAL, Traits::CB_ENTRIES>(s_cb, idx, vals);
    float sum = 0;
    for (int d = 0; d < P_VAL; d++) sum += vals[d];
    out[threadIdx.x] = sum;
}

// Force instantiation for all 5 configs
template __global__ void vq_verify_helpers_dummy<4, 8>(const unsigned int*, const half*, float*);
template __global__ void vq_verify_helpers_dummy<3, 8>(const unsigned int*, const half*, float*);
template __global__ void vq_verify_helpers_dummy<3, 10>(const unsigned int*, const half*, float*);
template __global__ void vq_verify_helpers_dummy<2, 8>(const unsigned int*, const half*, float*);
template __global__ void vq_verify_helpers_dummy<2, 10>(const unsigned int*, const half*, float*);

// ---- VQ (Vector Quantization) kernels ----
// VQ replaces bit-plane format with byte-indexed codebook lookup.
// Each 8-bit index maps to P_VAL fp16 weight values from a 256-entry codebook.
// P_VAL=2: 4 bits/weight (256 entries of half2), P_VAL=4: 2 bits/weight (256 entries of 4×half).

// VQ quantize: find nearest codebook entry for each group of P_VAL weights.
// One warp per BS-element quantization block. Not performance-critical (offline quantization).
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits.
template <int P_VAL, int INDEX_BITS, typename scalar_t>
__global__ void kQuantize_VQ(
    const half* __restrict__ codebook,      // [CB_ENTRIES, P_VAL] codebook in fp16
    const scalar_t* __restrict__ A,         // input weights (flat)
    unsigned char* __restrict__ absmax_out,  // E4M4 encoded absmax per block
    unsigned int* __restrict__ packed_out,   // packed indices
    const int n                             // total number of weight elements
) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = Traits::BS;
    constexpr int CB_ENTRIES = Traits::CB_ENTRIES;
    constexpr int GROUPS = Traits::GROUPS;
    constexpr int WORDS = Traits::WORDS;
    constexpr int ELEMS_PER_LANE = (BS + 31) / 32; // 1 for BS=32, 2 for BS=48

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int local_warp = threadIdx.x / 32;
    const int block_start = warp_id * BS;

    if (block_start >= n)
        return;

    // Load codebook into shared memory
    __shared__ half cb[CB_ENTRIES * P_VAL];
    for (int i = threadIdx.x; i < CB_ENTRIES * P_VAL; i += blockDim.x)
        cb[i] = codebook[i];

    // Per-warp shared memory for normalized weights and indices
    __shared__ float norm_shmem[8][BS];
    __shared__ int idx_shmem[8][32]; // max GROUPS is 16

    __syncthreads();

    // Load elements — each lane handles up to ELEMS_PER_LANE for BS>32
    float vals[ELEMS_PER_LANE];
    float my_max = 0.0f;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        int idx = lane_id + e * 32;
        vals[e] = (idx < BS && block_start + idx < n) ? (float)A[block_start + idx] : 0.0f;
        my_max = fmaxf(my_max, fabsf(vals[e]));
    }

    // Compute absmax via warp reduction
    float amax = warp_reduce_absmax_kbit(my_max);
    float amax_safe = fmaxf(amax, 1e-8f);

    // Store E4M4 absmax
    if (lane_id == 0)
        absmax_out[warp_id] = encode_e4m4_absmax(amax);

    // Normalize into shared memory
#pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        int idx = lane_id + e * 32;
        if (idx < BS)
            norm_shmem[local_warp][idx] = vals[e] / amax_safe;
    }
    __syncwarp();

    // Find nearest codebook entry for each group
    if (lane_id < GROUPS) {
        float w[4]; // max P_VAL=4
#pragma unroll
        for (int d = 0; d < P_VAL; d++)
            w[d] = norm_shmem[local_warp][lane_id * P_VAL + d];

        int best_idx = 0;
        float best_dist = 1e10f;

        for (int c = 0; c < CB_ENTRIES; c++) {
            float dist = 0.0f;
#pragma unroll
            for (int d = 0; d < P_VAL; d++) {
                float diff = w[d] - __half2float(cb[c * P_VAL + d]);
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }
        idx_shmem[local_warp][lane_id] = best_idx;
    }
    __syncwarp();

    // Pack indices into uint32 words
    if (lane_id < WORDS) {
        unsigned int word = 0;
        if constexpr (INDEX_BITS == 8) {
            // Byte packing: 4 indices per word
#pragma unroll
            for (int b = 0; b < 4; b++) {
                int gi = lane_id * 4 + b;
                if (gi < GROUPS)
                    word |= ((unsigned int)idx_shmem[local_warp][gi] & 0xFF) << (b * 8);
            }
        } else {
            // General bit packing for 10-bit (or any INDEX_BITS)
            int word_bit_start = lane_id * 32;
            int gi_start = word_bit_start / INDEX_BITS;
            int gi_end = min(GROUPS, (word_bit_start + 31) / INDEX_BITS + 1);
            for (int gi = gi_start; gi < gi_end; gi++) {
                unsigned int idx_val = (unsigned int)idx_shmem[local_warp][gi];
                int bit_pos = gi * INDEX_BITS;
                int shift = bit_pos - word_bit_start;
                if (shift >= 0)
                    word |= idx_val << shift;
                else
                    word |= idx_val >> (-shift);
            }
        }
        packed_out[warp_id * WORDS + lane_id] = word;
    }
}

// VQ dequantize (flat layout): read packed indices, look up codebook, write fp16/bf16.
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits.
template <int P_VAL, int INDEX_BITS, typename T, typename ABSMAX_T>
__global__ void kDequantize_VQ(
    const unsigned int* __restrict__ packed_in, // packed indices
    const half* __restrict__ codebook,          // [CB_ENTRIES, P_VAL] codebook in fp16
    const ABSMAX_T* __restrict__ absmax,        // absmax per block
    T* __restrict__ out,                        // output weights (flat)
    const int n                                 // total number of weight elements
) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = Traits::BS;
    constexpr int CB_ENTRIES = Traits::CB_ENTRIES;
    constexpr int GROUPS = Traits::GROUPS;
    constexpr int WORDS = Traits::WORDS;
    constexpr int ELEMS_PER_LANE = (BS + 31) / 32;

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int block_start = warp_id * BS;

    if (block_start >= n)
        return;

    // Load codebook into shared memory
    __shared__ half cb[CB_ENTRIES * P_VAL];
    for (int i = threadIdx.x; i < CB_ENTRIES * P_VAL; i += blockDim.x)
        cb[i] = codebook[i];
    __syncthreads();

    float amax = load_absmax(absmax, warp_id);

    // Load packed words via warp shuffle broadcast
    unsigned int words[WORDS];
#pragma unroll
    for (int w = 0; w < WORDS; w++) {
        unsigned int word_val = (lane_id == w) ? packed_in[warp_id * WORDS + w] : 0;
        words[w] = __shfl_sync(0xFFFFFFFF, word_val, w);
    }

    // Each lane handles up to ELEMS_PER_LANE elements
#pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        int elem = lane_id + e * 32;
        if (elem < BS && block_start + elem < n) {
            int group = elem / P_VAL;
            int component = elem % P_VAL;

            // Extract index using general method
            int idx = vq_extract_index<INDEX_BITS>(words, group);

            // Codebook lookup
            float val = __half2float(cb[idx * P_VAL + component]) * amax;
            out[block_start + elem] = (T)val;
        }
    }
}


// ---- VQ tiled dequantize kernel ----
// Reads from tiled VQ layout (from repack_vq output), writes flat [N, K_dim] row-major.
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits.

template <int P_VAL, int INDEX_BITS, typename T, typename ABSMAX_T>
__global__ void kDequantize_VQ_tiled(
    const unsigned int* __restrict__ packed_tiled,
    const half* __restrict__ codebook,
    const ABSMAX_T* __restrict__ absmax_tiled,
    T* __restrict__ out,
    const int K_dim, const int N
) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = Traits::BS;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = Traits::TILE_N;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS = Traits::WORDS;
    constexpr int WORDS_PER_TILE = TILE_N * KB_PER_TILE * WORDS;
    constexpr int ABS_PER_TILE = TILE_N * KB_PER_TILE;
    constexpr int GROUPS = Traits::GROUPS;

    const int n_tiles = N / TILE_N;

    // Each thread handles one element in the [N, K_dim] output
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K_dim;
    if (idx >= total)
        return;

    const int n_idx = idx / K_dim;
    const int k_idx = idx % K_dim;
    const int k_block = k_idx / BS;
    const int elem_in_block = k_idx % BS;

    // Tiled addressing
    const int k_tile = k_block / KB_PER_TILE;
    const int kb = k_block % KB_PER_TILE;
    const int n_tile = n_idx / TILE_N;
    const int col_in_tile = n_idx % TILE_N;
    const int tile_base = k_tile * n_tiles + n_tile;

    // Load absmax
    const int abs_idx = tile_base * ABS_PER_TILE + col_in_tile * KB_PER_TILE + kb;
    float amax = load_absmax(absmax_tiled, abs_idx);

    // Find the index for this element's group
    const int group = elem_in_block / P_VAL;
    const int component = elem_in_block % P_VAL;

    // Load words for this block (thread-per-element, so we load individually)
    const int word_base = tile_base * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + kb) * WORDS;

    if constexpr (INDEX_BITS == 8) {
        // Fast path: byte extraction
        const int word_in_block = group / 4;
        const int byte_in_word = group % 4;
        unsigned int word_val = packed_tiled[word_base + word_in_block];
        int cb_idx = (word_val >> (byte_in_word * 8)) & 0xFF;
        float val = __half2float(codebook[cb_idx * P_VAL + component]) * amax;
        out[idx] = (T)val;
    } else {
        // General bit extraction for 10-bit indices
        constexpr unsigned int MASK = (1u << INDEX_BITS) - 1u;
        const int bit = group * INDEX_BITS;
        const int w = bit >> 5;
        const int off = bit & 31;
        unsigned int val = packed_tiled[word_base + w] >> off;
        if (off > 32 - INDEX_BITS)
            val |= packed_tiled[word_base + w + 1] << (32 - off);
        int cb_idx = (int)(val & MASK);
        float fval = __half2float(codebook[cb_idx * P_VAL + component]) * amax;
        out[idx] = (T)fval;
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
    kQuantizeBlockwise_kbit<T, K>
        <<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(codebook, A, absmax, packed_out, n);
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

// ---- VQ kernel launchers ----

template <int P_VAL, int INDEX_BITS, typename T>
void quantize_vq(
    const half* codebook, const T* A, unsigned char* absmax, unsigned int* packed_out, int n, cudaStream_t stream
) {
    constexpr int BS = VQTraits<P_VAL, INDEX_BITS>::BS;
    int num_blocks_quant = (n + BS - 1) / BS;
    int num_cuda_blocks = (num_blocks_quant + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kQuantize_VQ<P_VAL, INDEX_BITS, T>
        <<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(codebook, A, absmax, packed_out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <int P_VAL, int INDEX_BITS, typename T, typename ABSMAX_T>
void dequantize_vq(
    const unsigned int* packed_in, const half* codebook, const ABSMAX_T* absmax, T* out, int n, cudaStream_t stream
) {
    constexpr int BS = VQTraits<P_VAL, INDEX_BITS>::BS;
    int num_blocks_quant = (n + BS - 1) / BS;
    int num_cuda_blocks = (num_blocks_quant + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kDequantize_VQ<P_VAL, INDEX_BITS, T, ABSMAX_T>
        <<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(packed_in, codebook, absmax, out, n);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <int P_VAL, int INDEX_BITS, typename T, typename ABSMAX_T>
void dequantize_vq_tiled(
    const unsigned int* packed_tiled, const half* codebook, const ABSMAX_T* absmax_tiled,
    T* out, int K_dim, int N, cudaStream_t stream
) {
    int total = N * K_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kDequantize_VQ_tiled<P_VAL, INDEX_BITS, T, ABSMAX_T>
        <<<blocks, threads, 0, stream>>>(packed_tiled, codebook, absmax_tiled, out, K_dim, N);
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
    kRepackKbit<K>
        <<<grid_size, block_size, 0, stream>>>(packed_flat, absmax_flat, packed_tiled, absmax_tiled, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ---- VQ Repack (flat VQ indices -> tiled layout) ----
// Copies packed words and absmax from flat to tiled layout for GEMM kernels.
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits.

template <int P_VAL, int INDEX_BITS>
__global__ void kRepackVQ(
    const unsigned int* __restrict__ packed_flat, const unsigned char* __restrict__ absmax_flat,
    unsigned int* __restrict__ packed_tiled, unsigned char* __restrict__ absmax_tiled, const int K_dim, const int N
) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = Traits::BS;
    constexpr int WORDS = Traits::WORDS;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = Traits::TILE_N;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS_PER_TILE = TILE_N * KB_PER_TILE * WORDS;
    constexpr int ABS_PER_TILE = TILE_N * KB_PER_TILE;

    const int total_k_blocks = K_dim / BS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * total_k_blocks)
        return;

    const int n_idx = idx / total_k_blocks;
    const int k_block_idx = idx % total_k_blocks;
    const int k_start = k_block_idx * BS;

    // Source: flat layout
    const int flat_block_id = n_idx * total_k_blocks + k_block_idx;

    // Destination: tiled layout
    const int k_tile = k_start / TILE_K;
    const int n_tile = n_idx / TILE_N;
    const int col = n_idx % TILE_N;
    const int kb = (k_start % TILE_K) / BS;

    const int n_tiles = N / TILE_N;
    const int tile_base = k_tile * n_tiles + n_tile;
    const int dst_word_base = tile_base * WORDS_PER_TILE + (col * KB_PER_TILE + kb) * WORDS;
    const int src_word_base = flat_block_id * WORDS;

#pragma unroll
    for (int w = 0; w < WORDS; w++)
        packed_tiled[dst_word_base + w] = packed_flat[src_word_base + w];

    const int dst_abs_idx = tile_base * ABS_PER_TILE + col * KB_PER_TILE + kb;
    absmax_tiled[dst_abs_idx] = absmax_flat[flat_block_id];
}

// VQ Repack launcher
template <int P_VAL, int INDEX_BITS>
void repackVQ(
    const unsigned int* packed_flat, const unsigned char* absmax_flat,
    unsigned int* packed_tiled, unsigned char* absmax_tiled,
    int K_dim, int N, cudaStream_t stream
) {
    constexpr int BS = VQTraits<P_VAL, INDEX_BITS>::BS;
    int total_work = N * (K_dim / BS);
    int block_size = 256;
    int grid_size = (total_work + block_size - 1) / block_size;
    kRepackVQ<P_VAL, INDEX_BITS>
        <<<grid_size, block_size, 0, stream>>>(packed_flat, absmax_flat, packed_tiled, absmax_tiled, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// ===========================================================================
// Hadamard rotation kernel (in-place, blocksize-templated)
//
// Applies a randomized Walsh-Hadamard transform (H*D) to contiguous blocks
// of BLOCK_SIZE elements. D is a diagonal sign-flip matrix (optional).
// Used to spread outliers before kbit quantization.
// Since H*D is orthogonal, rotating both weights and activations preserves
// the GEMM result: (H*D)(A) @ (H*D)(B)^T = A @ B^T.
//
// One warp per rotation block:
//   BLOCK_SIZE=32:  1 elem/thread, 5 shuffle stages
//   BLOCK_SIZE=64:  2 elem/thread, 1 register + 5 shuffle stages
//   BLOCK_SIZE=128: 4 elem/thread, 2 register + 5 shuffle stages
//   BLOCK_SIZE=256: 8 elem/thread, 3 register + 5 shuffle stages
//
// signs: optional bitmask of BLOCK_SIZE/32 uint32 words. If non-null, bit i
// set means element i is negated before the Hadamard butterfly. Same sign
// vector is applied to every block.
// ===========================================================================

template <int BLOCK_SIZE, typename T>
__global__ void kHadamardRotate(T* __restrict__ data, const int n, const unsigned int* __restrict__ signs) {
    constexpr int ELEMS_PER_THREAD = BLOCK_SIZE / 32;
    static_assert(BLOCK_SIZE >= 32 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be a power of 2 >= 32");

    const int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int block_start = warp_idx * BLOCK_SIZE;

    if (block_start >= n)
        return;

    // Load ELEMS_PER_THREAD elements per thread.
    // Thread t holds elements at global positions: block_start + t, t+32, t+64, ...
    float vals[ELEMS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        int idx = block_start + lane_id + j * 32;
        vals[j] = (idx < n) ? (float)data[idx] : 0.0f;
    }

    // Apply random sign flips (D matrix) before butterfly.
    // Element at position lane_id + j*32 uses word j, bit lane_id.
    if (signs != nullptr) {
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            if (signs[j] & (1u << lane_id))
                vals[j] = -vals[j];
        }
    }

    // In-register butterfly stages (strides >= 32).
    // Stride S in global space corresponds to element index s = S/32.
    // Element j pairs with element j ^ s (both in the same thread).
#pragma unroll
    for (int s = ELEMS_PER_THREAD / 2; s >= 1; s >>= 1) {
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            int partner = j ^ s;
            if (partner > j) {
                float a = vals[j], b = vals[partner];
                vals[j] = a + b;
                vals[partner] = a - b;
            }
        }
    }

    // Shuffle butterfly stages (strides 16, 8, 4, 2, 1).
    // Each stage exchanges values between lanes within the warp.
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
#pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, vals[j], s);
            vals[j] = (lane_id & s) ? (other - vals[j]) : (vals[j] + other);
        }
    }

    // Normalize by 1/sqrt(BLOCK_SIZE).
    const float norm = rsqrtf((float)BLOCK_SIZE);
#pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++)
        vals[j] *= norm;

    // Store back.
#pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        int idx = block_start + lane_id + j * 32;
        if (idx < n)
            data[idx] = (T)vals[j];
    }
}

// ---- Hadamard rotation launch wrapper ----

template <int BLOCK_SIZE, typename T>
void hadamardRotate(T* data, int n, const unsigned int* signs, cudaStream_t stream) {
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_cuda_blocks = (num_blocks + KBIT_WARPS_PER_BLOCK - 1) / KBIT_WARPS_PER_BLOCK;
    kHadamardRotate<BLOCK_SIZE, T><<<num_cuda_blocks, KBIT_THREADS_PER_BLOCK, 0, stream>>>(data, n, signs);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Explicit instantiations: 4 block sizes x 2 dtypes
#define INSTANTIATE_HADAMARD(BS)                                                                                       \
    template void hadamardRotate<BS, half>(half*, int, const unsigned int*, cudaStream_t);                             \
    template void hadamardRotate<BS, __nv_bfloat16>(__nv_bfloat16*, int, const unsigned int*, cudaStream_t);

INSTANTIATE_HADAMARD(32)
INSTANTIATE_HADAMARD(64)
INSTANTIATE_HADAMARD(128)
INSTANTIATE_HADAMARD(256)

#undef INSTANTIATE_HADAMARD

// ===========================================================================
// Full-dimension Hadamard rotation kernel.
// One thread block processes one row of DIM elements using 3-4 butterfly levels:
//   1. In-thread butterfly (strides 1..kNElts/2)
//   2. Warp shuffle butterfly (strides kNElts..kNElts*16)
//   3. Cross-warp butterfly via shared memory (strides across warps)
//   4. Cross-chunk butterfly in registers (when kNChunks > 1)
//
// Grid: (num_rows,). Signs: DIM/32 uint32 words (one per full row, not per block).
// ===========================================================================

template <int kLogDim, int kNThreads, typename T>
__global__ void kHadamardRotateFull(T* __restrict__ data, const int num_rows, const unsigned int* __restrict__ signs) {
    constexpr int DIM = 1 << kLogDim;
    constexpr int kNElts = 8; // elements per thread per chunk
    constexpr int kNChunks = DIM / (kNThreads * kNElts);
    constexpr int kNWarps = kNThreads / 32;

    static_assert(DIM == kNThreads * kNElts * kNChunks, "dimension decomposition mismatch");
    static_assert(kNElts == 8, "kNElts must be 8");
    static_assert((kNThreads & (kNThreads - 1)) == 0, "kNThreads must be power of 2");

    const int row = blockIdx.x;
    if (row >= num_rows)
        return;

    T* row_data = data + (long long)row * DIM;

    // Shared memory for cross-warp butterfly (only needed when kNWarps > 1).
    // Use char[] to match other kernels in this TU, then cast to float*.
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // ---- Load elements (contiguous per thread) ----
    float vals[kNChunks][kNElts];
#pragma unroll
    for (int c = 0; c < kNChunks; c++) {
        const int base = c * kNThreads * kNElts + tid * kNElts;
#pragma unroll
        for (int i = 0; i < kNElts; i++) {
            vals[c][i] = (float)row_data[base + i];
        }
    }

    // ---- Apply sign flips (D matrix) before butterfly ----
    // 8 contiguous elements at position 'base' always fit within one uint32 word
    // since base is always a multiple of 8.
    if (signs != nullptr) {
#pragma unroll
        for (int c = 0; c < kNChunks; c++) {
            const int linear = c * kNThreads + tid; // which group of 8
            const int word_idx = linear / 4;
            const int byte_pos = (linear % 4) * 8;
            const unsigned int byte_bits = (signs[word_idx] >> byte_pos) & 0xFFu;
#pragma unroll
            for (int i = 0; i < kNElts; i++) {
                if (byte_bits & (1u << i))
                    vals[c][i] = -vals[c][i];
            }
        }
    }

    // ---- Level 1: In-thread butterfly (strides 1, 2, 4) ----
#pragma unroll
    for (int c = 0; c < kNChunks; c++) {
#pragma unroll
        for (int s = 1; s < kNElts; s <<= 1) {
#pragma unroll
            for (int i = 0; i < kNElts; i++) {
                int partner = i ^ s;
                if (partner > i) {
                    float a = vals[c][i], b = vals[c][partner];
                    vals[c][i] = a + b;
                    vals[c][partner] = a - b;
                }
            }
        }
    }

    // ---- Level 2: Warp shuffle butterfly (shfl_xor s=1..16) ----
#pragma unroll
    for (int s = 1; s <= 16; s <<= 1) {
#pragma unroll
        for (int c = 0; c < kNChunks; c++) {
#pragma unroll
            for (int i = 0; i < kNElts; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[c][i], s);
                vals[c][i] = (lane_id & s) ? (other - vals[c][i]) : (vals[c][i] + other);
            }
        }
    }

    // ---- Level 3: Cross-warp butterfly via shared memory ----
    if constexpr (kNWarps > 1) {
        constexpr int VALS_PER_THREAD = kNChunks * kNElts;
        // smem layout: smem[tid * VALS_PER_THREAD + c * kNElts + i]
#pragma unroll
        for (int ws = 1; ws < kNWarps; ws <<= 1) {
            // Write my values to shared memory
#pragma unroll
            for (int c = 0; c < kNChunks; c++) {
#pragma unroll
                for (int i = 0; i < kNElts; i++) {
                    smem[tid * VALS_PER_THREAD + c * kNElts + i] = vals[c][i];
                }
            }
            __syncthreads();

            // Read partner warp's values
            const int partner_tid = (warp_id ^ ws) * 32 + lane_id;
            const bool negate = (warp_id & ws) != 0;
#pragma unroll
            for (int c = 0; c < kNChunks; c++) {
#pragma unroll
                for (int i = 0; i < kNElts; i++) {
                    float pval = smem[partner_tid * VALS_PER_THREAD + c * kNElts + i];
                    vals[c][i] = negate ? (pval - vals[c][i]) : (vals[c][i] + pval);
                }
            }
            __syncthreads();
        }
    }

    // ---- Level 4: Cross-chunk butterfly (in-register, no communication) ----
    if constexpr (kNChunks > 1) {
#pragma unroll
        for (int cs = 1; cs < kNChunks; cs <<= 1) {
#pragma unroll
            for (int c = 0; c < kNChunks; c++) {
                int pc = c ^ cs;
                if (pc > c) {
#pragma unroll
                    for (int i = 0; i < kNElts; i++) {
                        float a = vals[c][i], b = vals[pc][i];
                        vals[c][i] = a + b;
                        vals[pc][i] = a - b;
                    }
                }
            }
        }
    }

    // ---- Normalize by 1/sqrt(DIM) ----
    const float norm = rsqrtf((float)DIM);
#pragma unroll
    for (int c = 0; c < kNChunks; c++) {
#pragma unroll
        for (int i = 0; i < kNElts; i++)
            vals[c][i] *= norm;
    }

    // ---- Store back ----
#pragma unroll
    for (int c = 0; c < kNChunks; c++) {
        const int base = c * kNThreads * kNElts + tid * kNElts;
#pragma unroll
        for (int i = 0; i < kNElts; i++) {
            row_data[base + i] = (T)vals[c][i];
        }
    }
}

// ---- Full-dimension Hadamard launch wrapper ----
// kLogDim must match the dimension. kNThreads is the thread block size.

template <int kLogDim, int kNThreads, typename T>
void hadamardRotateFull(T* data, int num_rows, const unsigned int* signs, cudaStream_t stream) {
    constexpr int DIM = 1 << kLogDim;
    constexpr int kNElts = 8;
    constexpr int kNChunks = DIM / (kNThreads * kNElts);
    constexpr int smem_bytes = kNThreads * kNChunks * kNElts * sizeof(float);
    kHadamardRotateFull<kLogDim, kNThreads, T><<<num_rows, kNThreads, smem_bytes, stream>>>(data, num_rows, signs);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Explicit instantiations: dim 512..8192, 2 dtypes
#define INSTANTIATE_HADAMARD_FULL(LOG_DIM, NTHREADS)                                                                   \
    template void hadamardRotateFull<LOG_DIM, NTHREADS, half>(half*, int, const unsigned int*, cudaStream_t);          \
    template void hadamardRotateFull<LOG_DIM, NTHREADS, __nv_bfloat16>(                                                \
        __nv_bfloat16*, int, const unsigned int*, cudaStream_t                                                         \
    );

INSTANTIATE_HADAMARD_FULL(9, 64)   // dim=512
INSTANTIATE_HADAMARD_FULL(10, 128) // dim=1024
INSTANTIATE_HADAMARD_FULL(11, 256) // dim=2048
INSTANTIATE_HADAMARD_FULL(12, 256) // dim=4096
INSTANTIATE_HADAMARD_FULL(13, 256) // dim=8192

#undef INSTANTIATE_HADAMARD_FULL

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

// Helper: FP8 e4m3 MMA instruction (m16n8k32)
// A: 16x32 (e4m3), B: 32x8 (e4m3), C/D: 16x8 (f32)
__device__ __forceinline__ void mma_m16n8k32_fp8(uint32_t (&frag_a)[4], uint32_t (&frag_b)[2], float (&frag_c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
                 : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                   "r"(frag_b[0]), "r"(frag_b[1]),
                   "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3]));
}

// Convert float to e4m3 byte
__device__ __forceinline__ unsigned char float_to_e4m3(float val) {
    __nv_fp8_e4m3 fp8(val);
    return *reinterpret_cast<unsigned char*>(&fp8);
}

// Pack 4 float values as e4m3 into a uint32
__device__ __forceinline__ uint32_t pack_fp8x4(float v0, float v1, float v2, float v3) {
    return (unsigned int)float_to_e4m3(v0)
         | ((unsigned int)float_to_e4m3(v1) << 8)
         | ((unsigned int)float_to_e4m3(v2) << 16)
         | ((unsigned int)float_to_e4m3(v3) << 24);
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

    // Pipeline depth: 4 stages on datacenter GPUs (228KB shmem), 2 on consumer (100KB)
#if BNB_DATACENTER_GPU
    constexpr int NUM_STAGES = 4;
#else
    constexpr int NUM_STAGES = 2;
#endif

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

    // Multi-stage shared memory (NUM_STAGES stages)
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

        // Pipeline: NUM_STAGES-deep cp.async (2 on consumer, 4 on datacenter)
        // Pre-fill first (NUM_STAGES - 1) tiles
        {
            int prefill_end = kt_start + NUM_STAGES - 1;
            if (prefill_end > kt_end)
                prefill_end = kt_end;
            for (int pf = kt_start; pf < prefill_end; pf++) {
                fetch_tile((pf - kt_start) % NUM_STAGES, pf);
                cp_async_fence();
            }
        }

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % NUM_STAGES;
            int fetch_kt = kt + NUM_STAGES - 1;
            if (fetch_kt < kt_end) {
                fetch_tile((fetch_kt - kt_start) % NUM_STAGES, fetch_kt);
                cp_async_fence();
                // L2 prefetch for tile beyond the pipeline
                if (fetch_kt + 1 < kt_end) {
                    const int pf_tile = (fetch_kt + 1) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<NUM_STAGES - 1>();
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

// Cached SM count — queried once per process, safe for CUDA graph capture.
static int cachedNumSMs() {
    static int cached = -1;
    if (cached < 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

// Pipeline stage count: 4 on datacenter GPUs (more shmem), 2 on consumer.
static int pipelineNumStages() {
    static int cached = -1;
    if (cached < 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
        int sm = major * 10 + minor;
        cached = (sm == 90 || sm == 100) ? 4 : 2;
    }
    return cached;
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
    // Datacenter GPUs (H100) have higher bandwidth and can sustain more concurrent blocks.
    // TN=64: 4 blocks/SM (consumer), 6 blocks/SM (datacenter) for better latency hiding
    // TN=128: 1 block/SM (consumer), 2 blocks/SM (datacenter) to exploit larger shmem
    int target_blocks_per_sm;
    if constexpr (BLOCK_DIM <= 128)
        target_blocks_per_sm = (num_sms > 130) ? 6 : 4; // H100: 132 SMs
    else
        target_blocks_per_sm = (num_sms > 130) ? 2 : 1;
    int target_blocks = num_sms * target_blocks_per_sm;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    // Grid: launch enough blocks to fill target occupancy.
    // Multiple blocks per SM is fine — GPU schedules them concurrently.
    int grid_size = (k_splits == 1) ? total_work : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int num_stages = pipelineNumStages();
    int smem_size = num_stages * STAGE_BYTES;

    // If shared memory exceeds default 48KB limit, increase it
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kbit_gemm_prod<K, MB, TN, scalar_t, ABSMAX_T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

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
    const int num_sms = cachedNumSMs();

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

// ===========================================================================
// VQ Codebook MMA kernel: vq_gemm_prod (Generalized Template)
// Parameterized on (P_VAL, INDEX_BITS) via VQTraits for all 5 target configs.
// Uses tensor core m16n8k16 MMA instructions.
// ===========================================================================

template <int P_VAL, int INDEX_BITS, int M_BLOCKS, int TILE_N_VAL = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char, bool USE_FP8 = false>
__global__ void __launch_bounds__(TILE_N_VAL <= 64 ? 128 : 256, TILE_N_VAL <= 64 ? 12 : 1) vq_gemm_prod(
    const scalar_t* __restrict__ A, const unsigned int* __restrict__ B_packed, const ABSMAX_T* __restrict__ B_absmax,
    const half* __restrict__ codebook, scalar_t* __restrict__ C, float* __restrict__ C_workspace,
    int* __restrict__ tile_counters, const int M, const int K_dim, const int N, const int k_splits, const int total_work
) {
    using Ops = ScalarOps<scalar_t>;
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int TILE_M = M_BLOCKS * 16;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = TILE_N_VAL;
    constexpr int BS = Traits::BS;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS = Traits::WORDS;
    constexpr int GROUPS = Traits::GROUPS;
    constexpr int CB_ENTRIES = Traits::CB_ENTRIES;
    constexpr int B_COL_WORDS = KB_PER_TILE * WORDS;
    constexpr int N_BLOCKS = 2;
    constexpr int K_STEPS_PER_BLOCK = BS / 16;
    constexpr int TOTAL_K_STEPS = KB_PER_TILE * K_STEPS_PER_BLOCK;

    // FP8 MMA constants (m16n8k32: process 32 K elements per step)
    constexpr int FP8_K_STEP = 32;
    constexpr int FP8_TOTAL_STEPS = TILE_K / FP8_K_STEP;  // 3 for p=3, 2 for p=2/4

    // A stride must be padded to next power-of-2 multiple of 8 so that the XOR
    // swizzle (col_group ^ (row % 8)) never exceeds the allocated row width.
    // For TILE_K=64: stride=64 (no change).  For TILE_K=96: stride=128.
    static constexpr int _next_p2_groups = []() constexpr {
        int g = TILE_K / 8;
        int p2 = 1;
        while (p2 < g) p2 *= 2;
        return p2;
    }();
    constexpr int A_STRIDE_K = _next_p2_groups * 8;

    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE_K;
    constexpr int B_STAGE_WORDS = TILE_N * B_COL_WORDS;
    constexpr int ABS_STAGE_ELEMS = TILE_N * KB_PER_TILE;
    constexpr int ABS_STAGE_BYTES = ABS_STAGE_ELEMS * (int)sizeof(ABSMAX_T);

    constexpr int A_STAGE_BYTES = A_STAGE_ELEMS * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES_VAL = B_STAGE_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES_VAL + ABS_STAGE_ALIGNED;

    // Codebook in shared memory (persistent, not part of pipeline)
    constexpr int CB_BYTES = Traits::CB_SHMEM_BYTES;
    constexpr int CB_ALIGNED = (CB_BYTES + 15) & ~15;

#if BNB_DATACENTER_GPU
    constexpr int NUM_STAGES = 4;
#else
    constexpr int NUM_STAGES = 2;
#endif

    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    const int tiles_per_split = (k_tiles + k_splits - 1) / k_splits;

    constexpr int COLS_PER_WARP = N_BLOCKS * 8;
    constexpr int NUM_WARPS = TILE_N / COLS_PER_WARP;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int gid = lane_id / 4;
    const int tid = lane_id % 4;
    const int warp_n_base = warp_id * COLS_PER_WARP;

    // Shared memory layout: [codebook | stage0 | stage1 | ...]
    extern __shared__ char smem[];
    char* stage_base = smem + CB_ALIGNED;

    auto sh_a = [&](int stage) -> scalar_t* { return reinterpret_cast<scalar_t*>(stage_base + stage * STAGE_BYTES); };
    auto sh_b = [&](int stage) -> unsigned int* {
        return reinterpret_cast<unsigned int*>(stage_base + stage * STAGE_BYTES + A_STAGE_BYTES);
    };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(stage_base + stage * STAGE_BYTES + A_STAGE_BYTES + B_STAGE_BYTES_VAL);
    };

    // Load codebook into shared memory (once, persistent)
    half2* cb_shmem = reinterpret_cast<half2*>(smem);
    vq_load_codebook<P_VAL, CB_ENTRIES, TILE_N_VAL <= 64 ? 128 : 256>(cb_shmem, codebook);
    __syncthreads();

    float frag_c[M_BLOCKS][N_BLOCKS][4];

    // Persistent work loop
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

        // Zero accumulators
#pragma unroll
        for (int mb = 0; mb < M_BLOCKS; mb++)
#pragma unroll
            for (int nb = 0; nb < N_BLOCKS; nb++)
                frag_c[mb][nb][0] = frag_c[mb][nb][1] = frag_c[mb][nb][2] = frag_c[mb][nb][3] = 0.0f;

        // Fetch tile (identical to kbit version except B_COL_WORDS differs)
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

            // A tile via cp.async with XOR swizzle.
            // Uses A_STRIDE_K (power-of-2 padded) as row pitch so XOR stays in bounds.
            scalar_t* a_dst = sh_a(stage);
            constexpr int A_GROUPS_TOTAL = A_STAGE_ELEMS / 8;
            constexpr int A_K_GROUPS = A_STRIDE_K / 8;   // groups per row (may exceed TILE_K/8)
            constexpr int REAL_K_GROUPS = TILE_K / 8;     // actual data groups per row
            const bool a_interior = (m_base + TILE_M <= M) && (k_base + TILE_K <= K_dim)
                                    && (A_STRIDE_K == TILE_K);

            if (a_interior) {
                for (int i = threadIdx.x; i < A_GROUPS_TOTAL; i += blockDim.x) {
                    int row = i / A_K_GROUPS;
                    int col_group = i % A_K_GROUPS;
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * A_STRIDE_K + swizzled_group * 8]);
                    const int4* src =
                        reinterpret_cast<const int4*>(&A[(m_base + row) * K_dim + k_base + col_group * 8]);
                    cp_async_cg_16(dst, src);
                }
            } else {
                for (int i = threadIdx.x; i < A_GROUPS_TOTAL; i += blockDim.x) {
                    int row = i / A_K_GROUPS;
                    int col_group = i % A_K_GROUPS;
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * A_STRIDE_K + swizzled_group * 8]);
                    int gr = m_base + row;
                    int gc = k_base + col_group * 8;
                    if (gr < M && gc < K_dim && col_group < REAL_K_GROUPS) {
                        const int4* src = reinterpret_cast<const int4*>(&A[gr * K_dim + gc]);
                        cp_async_cg_16(dst, src);
                    } else {
                        *dst = make_int4(0, 0, 0, 0);
                    }
                }
            }
        };

        // Compute tile: VQ dequant via generalized index extraction + codebook lookup
        auto compute_tile = [&](int stage) {
            scalar_t* a_ptr = sh_a(stage);
            unsigned int* b_ptr = sh_b(stage);
            ABSMAX_T* abs_ptr = sh_abs(stage);

          if constexpr (USE_FP8) {
            // ---- FP8 MMA path: m16n8k32, process 32 K elements per step ----
            // FP8_TOTAL_STEPS = TILE_K/32 (3 for p=3, 2 for p=2/4).
            // Each step decodes 8 weight values per thread (2 fragments × 4 FP8 each).
            // A fragments loaded from shared memory (FP16→FP8 conversion in registers).
#pragma unroll
            for (int fp8_ks = 0; fp8_ks < FP8_TOTAL_STEPS; fp8_ks++) {
                // Load A fragments: FP16 from shared memory → convert to FP8 → pack
                // Thread mapping for m16n8k32 FP8:
                //   row = gid (0..7), tid (0..3) selects 4 consecutive k positions
                //   frag[0]: row,   k = fp8_ks*32 + 4*tid + 0..3
                //   frag[1]: row,   k = fp8_ks*32 + 4*tid + 16..19
                //   frag[1]: row+8, k = fp8_ks*32 + 4*tid + 0..3
                //   frag[2]: row,   k = fp8_ks*32 + 4*tid + 16..19
                //   frag[3]: row+8, k = fp8_ks*32 + 4*tid + 16..19
                uint32_t frag_a_fp8[M_BLOCKS][4];
#pragma unroll
                for (int mb = 0; mb < M_BLOCKS; mb++) {
#pragma unroll
                    for (int frag = 0; frag < 4; frag++) {
                        int row = mb * 16 + gid + (frag & 1) * 8;
                        int k_off = 4 * tid + (frag >= 2 ? 16 : 0);
                        int k_abs = fp8_ks * FP8_K_STEP + k_off;

                        int k_group = k_abs / 8;
                        int k_within = k_abs % 8;
                        int swizzled_group = k_group ^ (row % 8);

                        const scalar_t* addr = &a_ptr[row * A_STRIDE_K + swizzled_group * 8 + k_within];
                        float v0 = Ops::to_float(addr[0]);
                        float v1 = Ops::to_float(addr[1]);
                        float v2 = Ops::to_float(addr[2]);
                        float v3 = Ops::to_float(addr[3]);
                        frag_a_fp8[mb][frag] = pack_fp8x4(v0, v1, v2, v3);
                    }
                }

                // Decode B weights and execute FP8 MMA
#pragma unroll
                for (int nb = 0; nb < N_BLOCKS; nb++) {
                    int col = warp_n_base + nb * 8 + gid;

                    // Each half of the 32-element step may span a different quant block
                    uint32_t frag_b_fp8[2];
#pragma unroll
                    for (int half_idx = 0; half_idx < 2; half_idx++) {
                        int k_tile_base = fp8_ks * FP8_K_STEP + half_idx * 16;
                        int k_block = k_tile_base / BS;
                        int k_in_block_base = k_tile_base % BS;

                        float scale_f = Ops::to_float(Ops::from_float(
                            load_absmax<ABSMAX_T>(abs_ptr, col * KB_PER_TILE + k_block)));

                        // Load packed words for this quantization block
                        int word_base = col * B_COL_WORDS + k_block * WORDS;
                        unsigned int words_local[5];
#pragma unroll
                        for (int w = 0; w < WORDS; w++)
                            words_local[w] = b_ptr[word_base + w];

                        // Decode 4 values at k_in_block_base + 4*tid + 0..3
                        float vals[4];
                        float cb_vals[4];
#pragma unroll
                        for (int v = 0; v < 4; v++) {
                            int k_in_block = k_in_block_base + 4 * tid + v;
                            int gi = k_in_block / P_VAL;
                            int di = k_in_block % P_VAL;

                            int idx = vq_extract_index<INDEX_BITS>(words_local, gi);
                            vq_cb_lookup<P_VAL, CB_ENTRIES>(cb_shmem, idx, cb_vals);
                            vals[v] = cb_vals[di] * scale_f;
                        }

                        frag_b_fp8[half_idx] = pack_fp8x4(vals[0], vals[1], vals[2], vals[3]);
                    }

#pragma unroll
                    for (int mb = 0; mb < M_BLOCKS; mb++) {
                        mma_m16n8k32_fp8(frag_a_fp8[mb], frag_b_fp8, frag_c[mb][nb]);
                    }
                }
            }
          } else {
            // ---- FP16 MMA path: m16n8k16, process 16 K elements per step ----
            // Nested loop: outer over quant blocks, inner over sub-steps.
            // For p=2/4 (K_STEPS_PER_BLOCK=2): inner fully unrolled.
            // For p=3   (K_STEPS_PER_BLOCK=3): inner not unrolled to reduce reg pressure.
#pragma unroll
            for (int k_block = 0; k_block < KB_PER_TILE; k_block++) {

#pragma unroll(K_STEPS_PER_BLOCK <= 2 ? K_STEPS_PER_BLOCK : 1)
                for (int sub_step = 0; sub_step < K_STEPS_PER_BLOCK; sub_step++) {
                    const int ks = k_block * K_STEPS_PER_BLOCK + sub_step;

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

                        const scalar_t* addr = &a_ptr[a_row * A_STRIDE_K + swizzled_col_start];
                        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));

                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(frag_a[mb][0]), "=r"(frag_a[mb][1]), "=r"(frag_a[mb][2]), "=r"(frag_a[mb][3])
                                     : "r"(smem_addr));
                    }

#pragma unroll
                    for (int nb = 0; nb < N_BLOCKS; nb++) {
                        int col = warp_n_base + nb * 8 + gid;

                        scalar_t scale = Ops::from_float(load_absmax<ABSMAX_T>(abs_ptr, col * KB_PER_TILE + k_block));
                        float scale_f = Ops::to_float(scale);

                        int word_base_addr = col * B_COL_WORDS + k_block * WORDS;
                        unsigned int words_local[5]; // max WORDS is 5
#pragma unroll
                        for (int w = 0; w < WORDS; w++)
                            words_local[w] = b_ptr[word_base_addr + w];

                        // Decode 4 weight values for this thread's MMA positions.
                        scalar_t vals[4];
                        float cb_vals[4];
#pragma unroll
                        for (int v = 0; v < 4; v++) {
                            const int pos_off = (v < 2) ? v : (v + 6); // {0, 1, 8, 9}
                            int k_in_block = sub_step * 16 + 2 * tid + pos_off;
                            int gi = k_in_block / P_VAL;
                            int di = k_in_block % P_VAL;

                            int idx = vq_extract_index<INDEX_BITS>(words_local, gi);
                            vq_cb_lookup<P_VAL, CB_ENTRIES>(cb_shmem, idx, cb_vals);
                            vals[v] = Ops::from_float(cb_vals[di] * scale_f);
                        }

                        uint32_t frag_b[2];
                        frag_b[0] = pack_two<scalar_t>(vals[0], vals[1]);
                        frag_b[1] = pack_two<scalar_t>(vals[2], vals[3]);

#pragma unroll
                        for (int mb = 0; mb < M_BLOCKS; mb++) {
                            mma_m16n8k16<scalar_t>(frag_a[mb], frag_b, frag_c[mb][nb]);
                        }
                    }
                }
            }
          } // end USE_FP8
        };

        // Pipeline: NUM_STAGES-deep cp.async
        {
            int prefill_end = kt_start + NUM_STAGES - 1;
            if (prefill_end > kt_end)
                prefill_end = kt_end;
            for (int pf = kt_start; pf < prefill_end; pf++) {
                fetch_tile((pf - kt_start) % NUM_STAGES, pf);
                cp_async_fence();
            }
        }

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % NUM_STAGES;
            int fetch_kt = kt + NUM_STAGES - 1;
            if (fetch_kt < kt_end) {
                fetch_tile((fetch_kt - kt_start) % NUM_STAGES, fetch_kt);
                cp_async_fence();
                if (fetch_kt + 1 < kt_end) {
                    const int pf_tile = (fetch_kt + 1) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<NUM_STAGES - 1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            compute_tile(cur);
            __syncthreads();
        }

        // Write output
        if (k_splits == 1) {
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

// VQ GEMM launcher
template <int P, int IB, int MB, int TN = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char, bool USE_FP8 = false>
static void vqGemmProdLaunch(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const half* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int num_sms, cudaStream_t stream
) {
    using Traits = VQTraits<P, IB>;
    constexpr int TILE_M = MB * 16;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = TN;
    constexpr int BS = Traits::BS;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS = Traits::WORDS;
    constexpr int B_COL_WORDS = KB_PER_TILE * WORDS;
    constexpr int N_BLOCKS = 2;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8);
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    // Match A_STRIDE_K in kernel: pad to next power-of-2 of (TILE_K/8) groups
    static constexpr int _launch_p2_groups = []() constexpr {
        int g = TILE_K / 8;
        int p2 = 1;
        while (p2 < g) p2 *= 2;
        return p2;
    }();
    constexpr int A_STRIDE_K = _launch_p2_groups * 8;

    constexpr int A_STAGE_BYTES = TILE_M * A_STRIDE_K * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES = TILE_N * B_COL_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_BYTES = TILE_N * KB_PER_TILE * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    constexpr int CB_BYTES = Traits::CB_SHMEM_BYTES;
    constexpr int CB_ALIGNED = (CB_BYTES + 15) & ~15;

    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int n_tiles = N / TILE_N;
    int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    int mn_tiles = m_tiles * n_tiles;

    int target_blocks_per_sm;
    if constexpr (BLOCK_DIM <= 128)
        target_blocks_per_sm = (num_sms > 130) ? 6 : 4;
    else
        target_blocks_per_sm = (num_sms > 130) ? 2 : 1;
    int target_blocks = num_sms * target_blocks_per_sm;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    int grid_size = (k_splits == 1) ? total_work : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int num_stages = pipelineNumStages();
    int smem_size = CB_ALIGNED + num_stages * STAGE_BYTES;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            vq_gemm_prod<P, IB, MB, TN, scalar_t, ABSMAX_T, USE_FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    vq_gemm_prod<P, IB, MB, TN, scalar_t, ABSMAX_T, USE_FP8><<<grid_size, block, smem_size, stream>>>(
        A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, k_splits, total_work
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <int P, int IB, typename scalar_t, typename ABSMAX_T = unsigned char>
void vqGemmProd(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const half* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int k_chunks, cudaStream_t stream
) {
    const int num_sms = cachedNumSMs();

    int m_blocks = 1;
    if (M > 48)
        m_blocks = 4;
    else if (M > 32)
        m_blocks = 3;
    else if (M > 16)
        m_blocks = 2;

    const bool use_tn64 = (m_blocks == 1) && (N % 64 == 0);

    if (use_tn64) {
        vqGemmProdLaunch<P, IB, 1, 64, scalar_t, ABSMAX_T>(
            A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
        );
    } else {
        switch (m_blocks) {
        case 4:
            vqGemmProdLaunch<P, IB, 4, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 3:
            vqGemmProdLaunch<P, IB, 3, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 2:
            vqGemmProdLaunch<P, IB, 2, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        default:
            vqGemmProdLaunch<P, IB, 1, 128, scalar_t, ABSMAX_T>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        }
    }
}

// VQ GEMM FP8 dispatcher (same as vqGemmProd but with USE_FP8=true)
template <int P, int IB, typename scalar_t, typename ABSMAX_T = unsigned char>
void vqGemmProdFP8(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const half* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int k_chunks, cudaStream_t stream
) {
    const int num_sms = cachedNumSMs();

    int m_blocks = 1;
    if (M > 48)
        m_blocks = 4;
    else if (M > 32)
        m_blocks = 3;
    else if (M > 16)
        m_blocks = 2;

    const bool use_tn64 = (m_blocks == 1) && (N % 64 == 0);

    if (use_tn64) {
        vqGemmProdLaunch<P, IB, 1, 64, scalar_t, ABSMAX_T, true>(
            A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
        );
    } else {
        switch (m_blocks) {
        case 4:
            vqGemmProdLaunch<P, IB, 4, 128, scalar_t, ABSMAX_T, true>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 3:
            vqGemmProdLaunch<P, IB, 3, 128, scalar_t, ABSMAX_T, true>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        case 2:
            vqGemmProdLaunch<P, IB, 2, 128, scalar_t, ABSMAX_T, true>(
                A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream
            );
            break;
        default:
            vqGemmProdLaunch<P, IB, 1, 128, scalar_t, ABSMAX_T, true>(
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

    // Pipeline depth: 4 stages on datacenter GPUs, 2 on consumer
#if BNB_DATACENTER_GPU
    constexpr int NUM_STAGES = 4;
#else
    constexpr int NUM_STAGES = 2;
#endif

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

    // Multi-stage shared memory (NUM_STAGES stages)
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

        // Pipeline: NUM_STAGES-deep cp.async (2 on consumer, 4 on datacenter)
        {
            int prefill_end = kt_start + NUM_STAGES - 1;
            if (prefill_end > kt_end)
                prefill_end = kt_end;
            for (int pf = kt_start; pf < prefill_end; pf++) {
                fetch_tile((pf - kt_start) % NUM_STAGES, pf);
                cp_async_fence();
            }
        }

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % NUM_STAGES;
            int fetch_kt = kt + NUM_STAGES - 1;
            if (fetch_kt < kt_end) {
                fetch_tile((fetch_kt - kt_start) % NUM_STAGES, fetch_kt);
                cp_async_fence();
                // L2 prefetch for tile beyond the pipeline
                if (fetch_kt + 1 < kt_end) {
                    const int pf_tile = (fetch_kt + 1) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<NUM_STAGES - 1>();
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
    int target_blocks_per_sm;
    if constexpr (BLOCK_DIM <= 128)
        target_blocks_per_sm = (num_sms > 130) ? 6 : 4;
    else
        target_blocks_per_sm = (num_sms > 130) ? 2 : 1;
    int target_blocks = num_sms * target_blocks_per_sm;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    int grid_size = (k_splits == 1) ? min(num_sms, total_work) : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int num_stages = pipelineNumStages();
    int smem_size = num_stages * STAGE_BYTES;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kbit_grouped_gemm_prod<K, MB, TN, scalar_t, ABSMAX_T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

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

    const int num_sms = cachedNumSMs();

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

// ===========================================================================
// VQ Codebook Grouped GEMM kernel: vq_grouped_gemm_prod
// Fuses all MoE expert GEMMs into a single persistent kernel launch.
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits helpers.
// ===========================================================================

template <int P_VAL, int INDEX_BITS, int M_BLOCKS, int TN = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(TN <= 64 ? 128 : 256, TN <= 64 ? 12 : 1) vq_grouped_gemm_prod(
    const scalar_t* __restrict__ A_concat, const unsigned int* __restrict__ B_packed_all,
    const ABSMAX_T* __restrict__ B_absmax_all, const half* __restrict__ codebook, scalar_t* __restrict__ C_concat,
    float* __restrict__ C_workspace, int* __restrict__ tile_counters, const int* __restrict__ expert_offsets,
    const int K_dim, const int N, const int num_experts, const int k_splits, const int total_work
) {
    using Ops = ScalarOps<scalar_t>;
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int TILE_M = M_BLOCKS * 16;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = TN;
    constexpr int BS = Traits::BS;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS = Traits::WORDS;
    constexpr int GROUPS = Traits::GROUPS;
    constexpr int CB_ENTRIES = Traits::CB_ENTRIES;
    constexpr int B_COL_WORDS = KB_PER_TILE * WORDS;
    constexpr int N_BLOCKS = 2;
    constexpr int K_STEPS_PER_BLOCK = BS / 16;
    constexpr int TOTAL_K_STEPS = KB_PER_TILE * K_STEPS_PER_BLOCK;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8);
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    // A stride padded to next power-of-2 multiple of 8 for XOR swizzle safety
    static constexpr int _next_p2_groups = []() constexpr {
        int g = TILE_K / 8;
        int p2 = 1;
        while (p2 < g) p2 *= 2;
        return p2;
    }();
    constexpr int A_STRIDE_K = _next_p2_groups * 8;

    constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE_K;
    constexpr int B_STAGE_WORDS = TILE_N * B_COL_WORDS;
    constexpr int ABS_STAGE_ELEMS = TILE_N * KB_PER_TILE;
    constexpr int ABS_STAGE_BYTES = ABS_STAGE_ELEMS * (int)sizeof(ABSMAX_T);

    constexpr int A_STAGE_BYTES = A_STAGE_ELEMS * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES_VAL = B_STAGE_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES_VAL + ABS_STAGE_ALIGNED;

    // Codebook in shared memory (persistent, not part of pipeline)
    constexpr int CB_BYTES = Traits::CB_SHMEM_BYTES;
    constexpr int CB_ALIGNED = (CB_BYTES + 15) & ~15;

#if BNB_DATACENTER_GPU
    constexpr int NUM_STAGES = 4;
#else
    constexpr int NUM_STAGES = 2;
#endif

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

    // Shared memory layout: [codebook | stage0 | stage1 | ...]
    extern __shared__ char smem[];
    char* stage_base = smem + CB_ALIGNED;

    auto sh_a = [&](int stage) -> scalar_t* { return reinterpret_cast<scalar_t*>(stage_base + stage * STAGE_BYTES); };
    auto sh_b = [&](int stage) -> unsigned int* {
        return reinterpret_cast<unsigned int*>(stage_base + stage * STAGE_BYTES + A_STAGE_BYTES);
    };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(stage_base + stage * STAGE_BYTES + A_STAGE_BYTES + B_STAGE_BYTES_VAL);
    };

    // Load codebook into shared memory (once, persistent)
    half2* cb_shmem = reinterpret_cast<half2*>(smem);
    vq_load_codebook<P_VAL, CB_ENTRIES, BLOCK_DIM>(cb_shmem, codebook);
    __syncthreads();

    float frag_c[M_BLOCKS][N_BLOCKS][4];

    // Persistent work loop
    for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
        // Decompose work_id into (expert_id, m_tile, n_tile, ks_id)
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

        // Fetch tile lambda — uses A_STRIDE_K for XOR swizzle safety
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

            // A tile via cp.async with XOR swizzle (padded stride)
            scalar_t* a_dst = sh_a(stage);
            constexpr int A_GROUPS_TOTAL = A_STAGE_ELEMS / 8;
            constexpr int A_K_GROUPS = A_STRIDE_K / 8;
            constexpr int REAL_K_GROUPS = TILE_K / 8;
            const bool a_interior = (m_base + TILE_M <= M_e) && (k_base + TILE_K <= K_dim)
                                    && (A_STRIDE_K == TILE_K);

            if (a_interior) {
                for (int i = threadIdx.x; i < A_GROUPS_TOTAL; i += BLOCK_DIM) {
                    int row = i / A_K_GROUPS;
                    int col_group = i % A_K_GROUPS;
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * A_STRIDE_K + swizzled_group * 8]);
                    const int4* src =
                        reinterpret_cast<const int4*>(&A[(m_base + row) * K_dim + k_base + col_group * 8]);
                    cp_async_cg_16(dst, src);
                }
            } else {
                for (int i = threadIdx.x; i < A_GROUPS_TOTAL; i += BLOCK_DIM) {
                    int row = i / A_K_GROUPS;
                    int col_group = i % A_K_GROUPS;
                    int swizzled_group = col_group ^ (row % 8);
                    int4* dst = reinterpret_cast<int4*>(&a_dst[row * A_STRIDE_K + swizzled_group * 8]);
                    int gr = m_base + row;
                    int gc = k_base + col_group * 8;
                    if (gr < M_e && gc < K_dim && col_group < REAL_K_GROUPS) {
                        const int4* src = reinterpret_cast<const int4*>(&A[gr * K_dim + gc]);
                        cp_async_cg_16(dst, src);
                    } else {
                        *dst = make_int4(0, 0, 0, 0);
                    }
                }
            }
        };

        // Compute tile: VQ dequant via generalized index extraction + codebook lookup
        // Nested loop: outer over quantization blocks (KB_PER_TILE=2),
        // inner over sub-steps (K_STEPS_PER_BLOCK = BS/16).
        // For p=2/4 (K_STEPS_PER_BLOCK=2): inner fully unrolled.
        // For p=3   (K_STEPS_PER_BLOCK=3): inner not unrolled to reduce reg pressure.
        auto compute_tile = [&](int stage) {
            scalar_t* a_ptr = sh_a(stage);
            unsigned int* b_ptr = sh_b(stage);
            ABSMAX_T* abs_ptr = sh_abs(stage);

#pragma unroll
            for (int k_block = 0; k_block < KB_PER_TILE; k_block++) {

#pragma unroll(K_STEPS_PER_BLOCK <= 2 ? K_STEPS_PER_BLOCK : 1)
                for (int sub_step = 0; sub_step < K_STEPS_PER_BLOCK; sub_step++) {
                    const int ks = k_block * K_STEPS_PER_BLOCK + sub_step;

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

                        const scalar_t* addr = &a_ptr[a_row * A_STRIDE_K + swizzled_col_start];
                        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));

                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(frag_a[mb][0]), "=r"(frag_a[mb][1]), "=r"(frag_a[mb][2]), "=r"(frag_a[mb][3])
                                     : "r"(smem_addr));
                    }

#pragma unroll
                    for (int nb = 0; nb < N_BLOCKS; nb++) {
                        int col = warp_n_base + nb * 8 + gid;

                        scalar_t scale = Ops::from_float(load_absmax<ABSMAX_T>(abs_ptr, col * KB_PER_TILE + k_block));
                        float scale_f = Ops::to_float(scale);

                        int word_base_addr = col * B_COL_WORDS + k_block * WORDS;
                        unsigned int words_local[5]; // max WORDS is 5
#pragma unroll
                        for (int w = 0; w < WORDS; w++)
                            words_local[w] = b_ptr[word_base_addr + w];

                        // cb_vals declared outside v-loop to prevent compiler from
                        // keeping 4 copies alive during unrolled iterations.
                        scalar_t vals[4];
                        float cb_vals[4]; // reused across v iterations
#pragma unroll
                        for (int v = 0; v < 4; v++) {
                            const int pos_off = (v < 2) ? v : (v + 6);
                            int k_in_block = sub_step * 16 + 2 * tid + pos_off;
                            int gi = k_in_block / P_VAL;
                            int di = k_in_block % P_VAL;

                            int idx = vq_extract_index<INDEX_BITS>(words_local, gi);
                            vq_cb_lookup<P_VAL, CB_ENTRIES>(cb_shmem, idx, cb_vals);
                            vals[v] = Ops::from_float(cb_vals[di] * scale_f);
                        }

                        uint32_t frag_b[2];
                        frag_b[0] = pack_two<scalar_t>(vals[0], vals[1]);
                        frag_b[1] = pack_two<scalar_t>(vals[2], vals[3]);

#pragma unroll
                        for (int mb = 0; mb < M_BLOCKS; mb++) {
                            mma_m16n8k16<scalar_t>(frag_a[mb], frag_b, frag_c[mb][nb]);
                        }
                    }
                }
            }
        };

        // Pipeline: NUM_STAGES-deep cp.async
        {
            int prefill_end = kt_start + NUM_STAGES - 1;
            if (prefill_end > kt_end)
                prefill_end = kt_end;
            for (int pf = kt_start; pf < prefill_end; pf++) {
                fetch_tile((pf - kt_start) % NUM_STAGES, pf);
                cp_async_fence();
            }
        }

        for (int kt = kt_start; kt < kt_end; kt++) {
            int cur = (kt - kt_start) % NUM_STAGES;
            int fetch_kt = kt + NUM_STAGES - 1;
            if (fetch_kt < kt_end) {
                fetch_tile((fetch_kt - kt_start) % NUM_STAGES, fetch_kt);
                cp_async_fence();
                if (fetch_kt + 1 < kt_end) {
                    const int pf_tile = (fetch_kt + 1) * n_tiles + n_tile;
                    prefetch_l2(B_packed + pf_tile * B_STAGE_WORDS);
                    prefetch_l2(B_absmax + pf_tile * ABS_STAGE_ELEMS);
                }
                cp_async_wait<NUM_STAGES - 1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            compute_tile(cur);
            __syncthreads();
        }

        // Write output
        if (k_splits == 1) {
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

// VQ Grouped GEMM launcher — supports TILE_N=64/128 and auto k_splits
template <int P, int IB, int MB, int TN = 128, typename scalar_t = half, typename ABSMAX_T = unsigned char>
static void vqGroupedGemmProdLaunch(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all, const half* codebook,
    scalar_t* C_concat, float* C_workspace, int* tile_counters, const int* expert_offsets, int K_dim, int N,
    int num_experts, int max_M, int num_sms, cudaStream_t stream
) {
    using Traits = VQTraits<P, IB>;
    constexpr int TILE_M = MB * 16;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = TN;
    constexpr int BS = Traits::BS;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS = Traits::WORDS;
    constexpr int B_COL_WORDS = KB_PER_TILE * WORDS;
    constexpr int N_BLOCKS = 2;
    constexpr int NUM_WARPS = TILE_N / (N_BLOCKS * 8);
    constexpr int BLOCK_DIM = NUM_WARPS * 32;

    // A stride padded to next power-of-2 multiple of 8 (same as kernel)
    static constexpr int _next_p2_groups = []() constexpr {
        int g = TILE_K / 8;
        int p2 = 1;
        while (p2 < g) p2 *= 2;
        return p2;
    }();
    constexpr int A_STRIDE_K = _next_p2_groups * 8;

    constexpr int A_STAGE_BYTES = TILE_M * A_STRIDE_K * sizeof(scalar_t);
    constexpr int B_STAGE_BYTES = TILE_N * B_COL_WORDS * sizeof(unsigned int);
    constexpr int ABS_STAGE_BYTES = TILE_N * KB_PER_TILE * (int)sizeof(ABSMAX_T);
    constexpr int ABS_STAGE_ALIGNED = (ABS_STAGE_BYTES + 15) & ~15;
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + ABS_STAGE_ALIGNED;

    constexpr int CB_BYTES = Traits::CB_SHMEM_BYTES;
    constexpr int CB_ALIGNED = (CB_BYTES + 15) & ~15;

    int n_tiles = N / TILE_N;
    int k_tiles = (K_dim + TILE_K - 1) / TILE_K;
    int m_tiles_per_expert = (max_M + TILE_M - 1) / TILE_M;
    int mn_tiles = num_experts * m_tiles_per_expert * n_tiles;

    int target_blocks_per_sm;
    if constexpr (BLOCK_DIM <= 128)
        target_blocks_per_sm = (num_sms > 130) ? 6 : 4;
    else
        target_blocks_per_sm = (num_sms > 130) ? 2 : 1;
    int target_blocks = num_sms * target_blocks_per_sm;

    int k_splits = 1;
    if (mn_tiles < target_blocks && k_tiles > 1) {
        k_splits = min(k_tiles, (target_blocks + mn_tiles - 1) / mn_tiles);
    }

    int total_work = mn_tiles * k_splits;
    int grid_size = (k_splits == 1) ? min(num_sms, total_work) : min(target_blocks, total_work);

    dim3 block(BLOCK_DIM);
    int num_stages = pipelineNumStages();
    int smem_size = CB_ALIGNED + num_stages * STAGE_BYTES;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            vq_grouped_gemm_prod<P, IB, MB, TN, scalar_t, ABSMAX_T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    vq_grouped_gemm_prod<P, IB, MB, TN, scalar_t, ABSMAX_T><<<grid_size, block, smem_size, stream>>>(
        A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, expert_offsets, K_dim, N,
        num_experts, k_splits, total_work
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// VQ Grouped GEMM public entry point
template <int P, int IB, typename scalar_t, typename ABSMAX_T = unsigned char>
void vqGroupedGemmProd(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all, const half* codebook,
    scalar_t* C_concat, float* C_workspace, int* tile_counters, const int* d_expert_offsets, int K_dim, int N,
    int num_experts, int max_M, cudaStream_t stream
) {
    if (max_M == 0 || N == 0)
        return;

    const int num_sms = cachedNumSMs();

    int m_blocks = 1;
    if (max_M > 48)
        m_blocks = 4;
    else if (max_M > 32)
        m_blocks = 3;
    else if (max_M > 16)
        m_blocks = 2;

    const bool use_tn64 = (m_blocks == 1) && (N % 64 == 0);

    if (use_tn64) {
        vqGroupedGemmProdLaunch<P, IB, 1, 64, scalar_t, ABSMAX_T>(
            A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
            K_dim, N, num_experts, max_M, num_sms, stream
        );
    } else {
        switch (m_blocks) {
        case 4:
            vqGroupedGemmProdLaunch<P, IB, 4, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        case 3:
            vqGroupedGemmProdLaunch<P, IB, 3, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        case 2:
            vqGroupedGemmProdLaunch<P, IB, 2, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        default:
            vqGroupedGemmProdLaunch<P, IB, 1, 128, scalar_t, ABSMAX_T>(
                A_concat, B_packed_all, B_absmax_all, codebook, C_concat, C_workspace, tile_counters, d_expert_offsets,
                K_dim, N, num_experts, max_M, num_sms, stream
            );
            break;
        }
    }
}

// ===========================================================================
// VQ Codebook Grouped Scalar GEMV: vq_grouped_scalar_gemv
// Fuses all MoE expert scalar GEMV calls into a single kernel launch.
// Uses the same tiled B layout as vq_grouped_gemm_prod / vq_scalar_gemv.
// Optimized for M=1-4 (typical MoE decode batch per expert).
// Grid = num_experts * N, one block per (expert, output column) pair.
// Generalized for all (P_VAL, INDEX_BITS) configs via VQTraits.
// ===========================================================================

template <int P_VAL, int INDEX_BITS, int M_VAL, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(64, M_VAL <= 2 ? 24 : 16) vq_grouped_scalar_gemv(
    const scalar_t* __restrict__ A_concat,
    const unsigned int* __restrict__ B_packed_all,
    const ABSMAX_T* __restrict__ B_absmax_all,
    const half* __restrict__ codebook,
    scalar_t* __restrict__ C_concat,
    const int* __restrict__ expert_offsets,
    const int K_dim, const int N, const int num_experts
) {
    using T = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = T::BS;
    constexpr int BLOCK_SIZE = 64;
    constexpr int NUM_WARPS = 2;
    constexpr int M_MAX = 4;
    constexpr int GROUPS = T::GROUPS;
    constexpr int WORDS = T::WORDS;
    constexpr int CB_ENTRIES = T::CB_ENTRIES;

    // Tiled layout constants
    constexpr int TILE_K = T::TILE_K;
    constexpr int TILE_N = T::TILE_N;
    constexpr int KB_PER_TILE = T::KB_PER_TILE;
    constexpr int WORDS_PER_TILE = T::WORDS_PER_TILE;
    constexpr int ABS_PER_TILE = T::ABS_PER_TILE;

    // Block → (expert, column) mapping
    const int expert_id = blockIdx.x / N;
    const int col = blockIdx.x % N;
    if (expert_id >= num_experts) return;

    // Expert boundaries from offset array
    const int a_row_offset = expert_offsets[expert_id];
    const int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    if (M_e == 0) return;

    // Per-expert pointers
    const scalar_t* A = A_concat + a_row_offset * K_dim;
    scalar_t* C = C_concat + a_row_offset * N;

    const int num_k_blocks = K_dim / BS;
    const int n_tiles = N / TILE_N;
    const int k_tiles = (K_dim + TILE_K - 1) / TILE_K;

    // Per-expert B data sizes (same for all experts since K_dim, N are shared)
    const int b_packed_per_expert = k_tiles * n_tiles * WORDS_PER_TILE;
    const int b_absmax_per_expert = k_tiles * n_tiles * ABS_PER_TILE;
    const unsigned int* B_packed = B_packed_all + expert_id * b_packed_per_expert;
    const ABSMAX_T* B_absmax = B_absmax_all + expert_id * b_absmax_per_expert;

    // Tiled layout addressing for this column
    const int n_tile = col / TILE_N;
    const int col_in_tile = col % TILE_N;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared memory: codebook + partial reduction
    constexpr int CB_SHMEM_BYTES = T::CB_SHMEM_BYTES;
    extern __shared__ char smem_raw[];
    half2* s_cb = reinterpret_cast<half2*>(smem_raw);
    float* s_partial = reinterpret_cast<float*>(smem_raw + CB_SHMEM_BYTES);

    // Load codebook into shared memory
    vq_load_codebook<P_VAL, CB_ENTRIES, BLOCK_SIZE>(s_cb, codebook);
    __syncthreads();

    // Accumulators
    float acc[M_VAL];
#pragma unroll
    for (int m = 0; m < M_VAL; m++)
        acc[m] = 0.0f;

    // 64 threads stride through K blocks
    const int max_iters = (num_k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < max_iters; iter++) {
        const int block_idx = threadIdx.x + iter * BLOCK_SIZE;
        const bool valid = (block_idx < num_k_blocks);

        // Tiled B addressing
        const int k_tile = block_idx / KB_PER_TILE;
        const int kb = block_idx % KB_PER_TILE;
        const int tile_base = k_tile * n_tiles + n_tile;
        const int word_base = tile_base * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + kb) * WORDS;
        const int abs_idx = tile_base * ABS_PER_TILE + col_in_tile * KB_PER_TILE + kb;

        // L2 prefetch for next iteration
        {
            const int next_block_idx = block_idx + BLOCK_SIZE;
            if (next_block_idx < num_k_blocks) {
                const int nk_tile = next_block_idx / KB_PER_TILE;
                const int nkb = next_block_idx % KB_PER_TILE;
                const int ntb = nk_tile * n_tiles + n_tile;
                prefetch_l2(&B_packed[ntb * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + nkb) * WORDS]);
            }
        }

        // Load packed words
        unsigned int words[WORDS];
        if constexpr (WORDS == 2) {
            uint2 pv = valid ? *reinterpret_cast<const uint2*>(&B_packed[word_base]) : make_uint2(0u, 0u);
            words[0] = pv.x;
            words[1] = pv.y;
        } else if constexpr (WORDS == 4) {
            int4 pv;
            if (valid)
                pv = *reinterpret_cast<const int4*>(&B_packed[word_base]);
            else {
                pv.x = 0; pv.y = 0; pv.z = 0; pv.w = 0;
            }
            words[0] = (unsigned int)pv.x;
            words[1] = (unsigned int)pv.y;
            words[2] = (unsigned int)pv.z;
            words[3] = (unsigned int)pv.w;
        } else {
            // WORDS == 5 (10-bit, p=2): scalar loads to avoid alignment issues
            if (valid) {
                for (int w = 0; w < WORDS; w++)
                    words[w] = B_packed[word_base + w];
            } else {
                for (int w = 0; w < WORDS; w++)
                    words[w] = 0u;
            }
        }

        // Load absmax
        float amax = valid ? load_absmax(B_absmax, abs_idx) : 0.0f;

        const int k_base = block_idx * BS;

        // Dequant + FMA with vectorized A loads.
        // 8-bit indices: iterate by word (4 indices/word), pre-load A as int4/uint2.
        // 10-bit indices: fall back to group-based loop (indices cross word boundaries).
        if constexpr (INDEX_BITS == 8) {
            constexpr int INDICES_PER_WORD = 4;
            constexpr int ELEMS_PER_WORD = INDICES_PER_WORD * P_VAL;
            // WORDS_PER_BLOCK = GROUPS / 4
            constexpr int WORDS_PER_BLOCK = GROUPS / INDICES_PER_WORD;

#pragma unroll
            for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                if constexpr (P_VAL == 2) {
                    // 8 elements per word → 1 int4 load (16 bytes = 8 fp16)
                    int4 av[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid)
                            av[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                const scalar_t* ap = reinterpret_cast<const scalar_t*>(&av[m]);
                                acc[m] += w0 * ScalarOps<scalar_t>::to_float(ap[b * 2])
                                        + w1 * ScalarOps<scalar_t>::to_float(ap[b * 2 + 1]);
                            }
                        }
                    }
                } else if constexpr (P_VAL == 4) {
                    // 16 elements per word → 2 int4 loads (32 bytes = 16 fp16)
                    int4 av0[M_VAL], av1[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid) {
                            av0[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                            av1[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD + 8]);
                        }
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
                        float w2 = cb[2] * amax;
                        float w3 = cb[3] * amax;
                        int elem_in_word = b * 4;
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                const scalar_t* ap;
                                int off;
                                if (elem_in_word < 8) {
                                    ap = reinterpret_cast<const scalar_t*>(&av0[m]);
                                    off = elem_in_word;
                                } else {
                                    ap = reinterpret_cast<const scalar_t*>(&av1[m]);
                                    off = elem_in_word - 8;
                                }
                                acc[m] += w0 * ScalarOps<scalar_t>::to_float(ap[off])
                                        + w1 * ScalarOps<scalar_t>::to_float(ap[off + 1])
                                        + w2 * ScalarOps<scalar_t>::to_float(ap[off + 2])
                                        + w3 * ScalarOps<scalar_t>::to_float(ap[off + 3]);
                            }
                        }
                    }
                } else {
                    // P_VAL == 3: 12 elements per word → 3 uint2 loads (24 bytes = 12 fp16)
                    uint2 av0[M_VAL], av1[M_VAL], av2[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid) {
                            const uint2* base = reinterpret_cast<const uint2*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                            av0[m] = base[0];  // elements 0-3
                            av1[m] = base[1];  // elements 4-7
                            av2[m] = base[2];  // elements 8-11
                        }
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
                        float w2 = cb[2] * amax;
                        // Element offset within the 12-element word
                        int elem = b * 3;
                        // Map element to (vector_idx, offset_in_vector):
                        //   b=0: elem=0  → av0, off=0
                        //   b=1: elem=3  → av0, off=3 (w0); av1, off=0 (w1,w2)
                        //   b=2: elem=6  → av1, off=2
                        //   b=3: elem=9  → av2, off=1
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                // Select the right vector and offset for each of the 3 elements
                                const scalar_t* v0_ptr = reinterpret_cast<const scalar_t*>(&av0[m]);
                                const scalar_t* v1_ptr = reinterpret_cast<const scalar_t*>(&av1[m]);
                                const scalar_t* v2_ptr = reinterpret_cast<const scalar_t*>(&av2[m]);

                                // Use a flat 12-element view: elem 0-3 in av0, 4-7 in av1, 8-11 in av2
                                float a0, a1, a2;
                                if (elem < 4) {
                                    a0 = ScalarOps<scalar_t>::to_float(v0_ptr[elem]);
                                } else if (elem < 8) {
                                    a0 = ScalarOps<scalar_t>::to_float(v1_ptr[elem - 4]);
                                } else {
                                    a0 = ScalarOps<scalar_t>::to_float(v2_ptr[elem - 8]);
                                }
                                if (elem + 1 < 4) {
                                    a1 = ScalarOps<scalar_t>::to_float(v0_ptr[elem + 1]);
                                } else if (elem + 1 < 8) {
                                    a1 = ScalarOps<scalar_t>::to_float(v1_ptr[elem + 1 - 4]);
                                } else {
                                    a1 = ScalarOps<scalar_t>::to_float(v2_ptr[elem + 1 - 8]);
                                }
                                if (elem + 2 < 4) {
                                    a2 = ScalarOps<scalar_t>::to_float(v0_ptr[elem + 2]);
                                } else if (elem + 2 < 8) {
                                    a2 = ScalarOps<scalar_t>::to_float(v1_ptr[elem + 2 - 4]);
                                } else {
                                    a2 = ScalarOps<scalar_t>::to_float(v2_ptr[elem + 2 - 8]);
                                }
                                acc[m] += w0 * a0 + w1 * a1 + w2 * a2;
                            }
                        }
                    }
                }
            }
        } else {
            // 10-bit indices: group-based loop (indices cross word boundaries)
#pragma unroll
            for (int g = 0; g < GROUPS; g++) {
                int idx = vq_extract_index<INDEX_BITS>(words, g);
                float wt[P_VAL];
                vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, wt);
#pragma unroll
                for (int e = 0; e < P_VAL; e++)
                    wt[e] *= amax;
                const int k_elem = k_base + g * P_VAL;
#pragma unroll
                for (int m = 0; m < M_VAL; m++) {
                    if (valid) {
#pragma unroll
                        for (int e = 0; e < P_VAL; e++) {
                            acc[m] += wt[e] * ScalarOps<scalar_t>::to_float(A[m * K_dim + k_elem + e]);
                        }
                    }
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

    // Phase 2: Inter-warp reduction via shared memory
    if (lane_id == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++)
            s_partial[warp_id * M_MAX + m] = acc[m];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++) {
            if (m < M_e) {
                float sum = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    sum += s_partial[ww * M_MAX + m];
                C[m * N + col] = ScalarOps<scalar_t>::from_float(sum);
            }
        }
    }
}

// ---- VQ Grouped Scalar GEMV launcher ----
template <int P, int IB, int MV, typename scalar_t, typename ABSMAX_T>
static void vqGroupedScalarGemvLaunch(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all,
    const half* codebook, scalar_t* C_concat, const int* expert_offsets,
    int K_dim, int N, int num_experts, cudaStream_t stream
) {
    using T = VQTraits<P, IB>;
    constexpr int BLOCK_SIZE = 64;
    constexpr int M_MAX = 4;
    int smem_size = T::CB_SHMEM_BYTES + 2 * M_MAX * sizeof(float);

    int grid_size = num_experts * N;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            vq_grouped_scalar_gemv<P, IB, MV, scalar_t, ABSMAX_T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    vq_grouped_scalar_gemv<P, IB, MV, scalar_t, ABSMAX_T>
        <<<grid_size, BLOCK_SIZE, smem_size, stream>>>(
            A_concat, B_packed_all, B_absmax_all, codebook, C_concat,
            expert_offsets, K_dim, N, num_experts);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry point: dispatches on max_M (1-4)
template <int P, int IB, typename scalar_t, typename ABSMAX_T = unsigned char>
void vqGroupedScalarGemv(
    const scalar_t* A_concat, const unsigned int* B_packed_all, const ABSMAX_T* B_absmax_all,
    const half* codebook, scalar_t* C_concat, const int* expert_offsets,
    int K_dim, int N, int num_experts, int max_M, cudaStream_t stream
) {
    if (max_M == 0 || N == 0 || num_experts == 0)
        return;

#define LAUNCH_VQ_GROUPED_GEMV(MV) \
    vqGroupedScalarGemvLaunch<P, IB, MV, scalar_t, ABSMAX_T>( \
        A_concat, B_packed_all, B_absmax_all, codebook, C_concat, \
        expert_offsets, K_dim, N, num_experts, stream)
    if (max_M <= 1)      { LAUNCH_VQ_GROUPED_GEMV(1); }
    else if (max_M <= 2) { LAUNCH_VQ_GROUPED_GEMV(2); }
    else if (max_M <= 3) { LAUNCH_VQ_GROUPED_GEMV(3); }
    else                 { LAUNCH_VQ_GROUPED_GEMV(4); }
#undef LAUNCH_VQ_GROUPED_GEMV
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

// ---- VQ Scalar GEMV (Generalized Template) ----
// Vector Quantization codebook-based scalar GEMV for M=1-4.
// Parameterized on (P_VAL, INDEX_BITS) via VQTraits for all 5 target configs:
//   8-bit/p=4 (2.00 bits/wt), 8-bit/p=3 (2.67), 10-bit/p=3 (3.33),
//   8-bit/p=2 (4.00), 10-bit/p=2 (5.00)
// 64 threads (2 warps), one output column per block, grid = N.

// Compute launch_bounds max_blocks from shmem size at compile time
template <int P_VAL, int INDEX_BITS, int M_VAL>
struct VQGemvLaunchBounds {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    // SM has 100KB shmem on sm_89. Each block needs CB_SHMEM + reduction shmem.
    static constexpr int TOTAL_SHMEM = Traits::CB_SHMEM_BYTES + 2 * 4 * (int)sizeof(float); // 2 warps * M_MAX * float
    // max_blocks = min(24, 100KB / total_shmem). Cap at 24 (hw limit for 64 threads/block).
    static constexpr int MAX_BLOCKS_SHMEM = 102400 / TOTAL_SHMEM;
    static constexpr int MAX_BLOCKS = (MAX_BLOCKS_SHMEM < 24) ? MAX_BLOCKS_SHMEM : 24;
    // For M>2, reduce occupancy target to save registers
    static constexpr int VALUE = (M_VAL <= 2) ? MAX_BLOCKS : ((MAX_BLOCKS < 16) ? MAX_BLOCKS : 16);
};

template <int P_VAL, int INDEX_BITS, int M_VAL, bool TILED, typename scalar_t, typename ABSMAX_T>
__global__ void __launch_bounds__(64, VQGemvLaunchBounds<P_VAL, INDEX_BITS, M_VAL>::VALUE) vq_scalar_gemv(
    const scalar_t* __restrict__ A,
    const unsigned int* __restrict__ B_packed,
    const ABSMAX_T* __restrict__ B_absmax,
    const half* __restrict__ codebook,
    scalar_t* __restrict__ C,
    const int M, const int K_dim, const int N
) {
    using Traits = VQTraits<P_VAL, INDEX_BITS>;
    constexpr int BS = Traits::BS;
    constexpr int BLOCK_SIZE = 64;
    constexpr int NUM_WARPS = 2;
    constexpr int M_MAX = 4;
    constexpr int WORDS = Traits::WORDS;
    constexpr int GROUPS = Traits::GROUPS;
    constexpr int CB_ENTRIES = Traits::CB_ENTRIES;
    constexpr int TILE_K = Traits::TILE_K;
    constexpr int TILE_N = Traits::TILE_N;
    constexpr int KB_PER_TILE = Traits::KB_PER_TILE;
    constexpr int WORDS_PER_TILE = Traits::WORDS_PER_TILE;
    constexpr int ABS_PER_TILE = Traits::ABS_PER_TILE;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int col = blockIdx.x;
    const int num_k_blocks = K_dim / BS;

    // Shared memory: codebook + partial reduction
    extern __shared__ char smem_raw[];
    half2* s_cb = reinterpret_cast<half2*>(smem_raw);
    float* s_partial = reinterpret_cast<float*>(smem_raw + Traits::CB_SHMEM_BYTES);

    // Load codebook into shared memory using generalized helper
    vq_load_codebook<P_VAL, CB_ENTRIES, BLOCK_SIZE>(s_cb, codebook);
    __syncthreads();

    // Layout-dependent addressing
    const unsigned int* B_col = nullptr;
    const ABSMAX_T* abs_col = nullptr;
    int n_tile = 0, col_in_tile = 0, n_tiles = 0;

    if constexpr (!TILED) {
        B_col = B_packed + col * num_k_blocks * WORDS;
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

    // 64 threads stride through K blocks
    const int max_iters = (num_k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < max_iters; iter++) {
        const int block_idx = threadIdx.x + iter * BLOCK_SIZE;
        const bool valid = (block_idx < num_k_blocks);

        // Compute word base address
        int word_base;
        int abs_idx;
        if constexpr (!TILED) {
            word_base = block_idx * WORDS;
            abs_idx = block_idx;
        } else {
            const int k_tile = block_idx / KB_PER_TILE;
            const int kb = block_idx % KB_PER_TILE;
            const int tile_base = k_tile * n_tiles + n_tile;
            word_base = tile_base * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + kb) * WORDS;
            abs_idx = tile_base * ABS_PER_TILE + col_in_tile * KB_PER_TILE + kb;
        }

        // L2 prefetch for next iteration
        {
            const int next_block_idx = block_idx + BLOCK_SIZE;
            if (next_block_idx < num_k_blocks) {
                if constexpr (!TILED) {
                    prefetch_l2(&B_col[next_block_idx * WORDS]);
                } else {
                    const int nk_tile = next_block_idx / KB_PER_TILE;
                    const int nkb = next_block_idx % KB_PER_TILE;
                    const int ntb = nk_tile * n_tiles + n_tile;
                    prefetch_l2(&B_packed[ntb * WORDS_PER_TILE + (col_in_tile * KB_PER_TILE + nkb) * WORDS]);
                }
            }
        }

        // Load packed index words
        const unsigned int* B_src = TILED ? B_packed : B_col;
        unsigned int words[WORDS];
        if constexpr (WORDS == 5) {
            // 10-bit: 5 words = 20 bytes. Blocks are at 20-byte intervals,
            // so int4 (16-byte aligned) loads would fault on odd blocks.
            // Use scalar loads instead.
            #pragma unroll
            for (int i = 0; i < 5; i++)
                words[i] = valid ? B_src[word_base + i] : 0u;
        } else if constexpr (WORDS == 4) {
            int4 pv;
            if (valid)
                pv = *reinterpret_cast<const int4*>(&B_src[word_base]);
            else { pv.x = 0; pv.y = 0; pv.z = 0; pv.w = 0; }
            words[0] = (unsigned int)pv.x; words[1] = (unsigned int)pv.y;
            words[2] = (unsigned int)pv.z; words[3] = (unsigned int)pv.w;
        } else if constexpr (WORDS == 2) {
            uint2 pv = valid ? *reinterpret_cast<const uint2*>(&B_src[word_base]) : make_uint2(0u, 0u);
            words[0] = pv.x; words[1] = pv.y;
        } else {
            // Generic fallback
            #pragma unroll
            for (int i = 0; i < WORDS; i++)
                words[i] = valid ? B_src[word_base + i] : 0u;
        }

        // Load absmax
        const ABSMAX_T* abs_src = TILED ? B_absmax : abs_col;
        float amax = valid ? load_absmax(abs_src, abs_idx) : 0.0f;

        const int k_base = block_idx * BS;

        // Dequant + FMA with vectorized A loads.
        // 8-bit indices: iterate by word (4 indices/word), pre-load A as int4/uint2.
        // 10-bit indices: fall back to group-based loop (indices cross word boundaries).
        if constexpr (INDEX_BITS == 8) {
            constexpr int INDICES_PER_WORD = 4;
            constexpr int ELEMS_PER_WORD = INDICES_PER_WORD * P_VAL;
            constexpr int WORDS_PER_BLOCK = GROUPS / INDICES_PER_WORD;

#pragma unroll
            for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                if constexpr (P_VAL == 2) {
                    // 8 elements per word → 1 int4 load (16 bytes = 8 fp16)
                    int4 av[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid)
                            av[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                const scalar_t* ap = reinterpret_cast<const scalar_t*>(&av[m]);
                                acc[m] += w0 * ScalarOps<scalar_t>::to_float(ap[b * 2])
                                        + w1 * ScalarOps<scalar_t>::to_float(ap[b * 2 + 1]);
                            }
                        }
                    }
                } else if constexpr (P_VAL == 4) {
                    // 16 elements per word → 2 int4 loads (32 bytes = 16 fp16)
                    int4 av0[M_VAL], av1[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid) {
                            av0[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                            av1[m] = *reinterpret_cast<const int4*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD + 8]);
                        }
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
                        float w2 = cb[2] * amax;
                        float w3 = cb[3] * amax;
                        int elem_in_word = b * 4;
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                const scalar_t* ap;
                                int off;
                                if (elem_in_word < 8) {
                                    ap = reinterpret_cast<const scalar_t*>(&av0[m]);
                                    off = elem_in_word;
                                } else {
                                    ap = reinterpret_cast<const scalar_t*>(&av1[m]);
                                    off = elem_in_word - 8;
                                }
                                acc[m] += w0 * ScalarOps<scalar_t>::to_float(ap[off])
                                        + w1 * ScalarOps<scalar_t>::to_float(ap[off + 1])
                                        + w2 * ScalarOps<scalar_t>::to_float(ap[off + 2])
                                        + w3 * ScalarOps<scalar_t>::to_float(ap[off + 3]);
                            }
                        }
                    }
                } else {
                    // P_VAL == 3: 12 elements per word → 3 uint2 loads (24 bytes = 12 fp16)
                    uint2 av0[M_VAL], av1[M_VAL], av2[M_VAL];
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid) {
                            const uint2* base = reinterpret_cast<const uint2*>(
                                &A[m * K_dim + k_base + w * ELEMS_PER_WORD]);
                            av0[m] = base[0];  // elements 0-3
                            av1[m] = base[1];  // elements 4-7
                            av2[m] = base[2];  // elements 8-11
                        }
                    }
#pragma unroll
                    for (int b = 0; b < INDICES_PER_WORD; b++) {
                        int idx = (words[w] >> (b * 8)) & 0xFF;
                        float cb[P_VAL];
                        vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
                        float w0 = cb[0] * amax;
                        float w1 = cb[1] * amax;
                        float w2 = cb[2] * amax;
                        int elem = b * 3;
#pragma unroll
                        for (int m = 0; m < M_VAL; m++) {
                            if (valid) {
                                const scalar_t* v0_ptr = reinterpret_cast<const scalar_t*>(&av0[m]);
                                const scalar_t* v1_ptr = reinterpret_cast<const scalar_t*>(&av1[m]);
                                const scalar_t* v2_ptr = reinterpret_cast<const scalar_t*>(&av2[m]);

                                float a0, a1, a2;
                                if (elem < 4) {
                                    a0 = ScalarOps<scalar_t>::to_float(v0_ptr[elem]);
                                } else if (elem < 8) {
                                    a0 = ScalarOps<scalar_t>::to_float(v1_ptr[elem - 4]);
                                } else {
                                    a0 = ScalarOps<scalar_t>::to_float(v2_ptr[elem - 8]);
                                }
                                if (elem + 1 < 4) {
                                    a1 = ScalarOps<scalar_t>::to_float(v0_ptr[elem + 1]);
                                } else if (elem + 1 < 8) {
                                    a1 = ScalarOps<scalar_t>::to_float(v1_ptr[elem + 1 - 4]);
                                } else {
                                    a1 = ScalarOps<scalar_t>::to_float(v2_ptr[elem + 1 - 8]);
                                }
                                if (elem + 2 < 4) {
                                    a2 = ScalarOps<scalar_t>::to_float(v0_ptr[elem + 2]);
                                } else if (elem + 2 < 8) {
                                    a2 = ScalarOps<scalar_t>::to_float(v1_ptr[elem + 2 - 4]);
                                } else {
                                    a2 = ScalarOps<scalar_t>::to_float(v2_ptr[elem + 2 - 8]);
                                }
                                acc[m] += w0 * a0 + w1 * a1 + w2 * a2;
                            }
                        }
                    }
                }
            }
        } else {
            // 10-bit indices: group-based loop (indices cross word boundaries)
#pragma unroll
            for (int gi = 0; gi < GROUPS; gi++) {
                const int wpos = gi * P_VAL;
                int idx = vq_extract_index<INDEX_BITS>(words, gi);
                float cb[P_VAL];
                vq_cb_lookup<P_VAL, CB_ENTRIES>(s_cb, idx, cb);
#pragma unroll
                for (int d = 0; d < P_VAL; d++) {
                    float w = cb[d] * amax;
#pragma unroll
                    for (int m = 0; m < M_VAL; m++) {
                        if (valid) {
                            float a = ScalarOps<scalar_t>::to_float(A[m * K_dim + k_base + wpos + d]);
                            acc[m] += w * a;
                        }
                    }
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

    // Phase 2: Inter-warp reduction via shared memory
    if (lane_id == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++)
            s_partial[warp_id * M_MAX + m] = acc[m];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (int m = 0; m < M_VAL; m++) {
            if (m < M) {
                float sum = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    sum += s_partial[ww * M_MAX + m];
                C[m * N + col] = ScalarOps<scalar_t>::from_float(sum);
            }
        }
    }
}

// ---- VQ Scalar GEMV launchers ----
template <int P, int IB, int MV, bool TILED, typename scalar_t, typename ABSMAX_T>
static void vqScalarGemvLaunch(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax,
    const half* codebook, scalar_t* C, int M, int K_dim, int N, cudaStream_t stream
) {
    using Traits = VQTraits<P, IB>;
    constexpr int BLOCK_SIZE = 64;
    constexpr int M_MAX = 4;
    int smem_size = Traits::CB_SHMEM_BYTES + 2 * M_MAX * (int)sizeof(float);

    vq_scalar_gemv<P, IB, MV, TILED, scalar_t, ABSMAX_T>
        <<<N, BLOCK_SIZE, smem_size, stream>>>(A, B_packed, B_absmax, codebook, C, M, K_dim, N);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry: flat layout
template <int P, int IB, typename scalar_t, typename ABSMAX_T>
void vqScalarGemv(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax,
    const half* codebook, scalar_t* C, int M, int K_dim, int N, cudaStream_t stream
) {
#define LAUNCH_VQ_GEMV(MV) \
    vqScalarGemvLaunch<P, IB, MV, false, scalar_t, ABSMAX_T>(A, B_packed, B_absmax, codebook, C, M, K_dim, N, stream)
    if (M <= 1)      { LAUNCH_VQ_GEMV(1); }
    else if (M <= 2) { LAUNCH_VQ_GEMV(2); }
    else if (M <= 3) { LAUNCH_VQ_GEMV(3); }
    else             { LAUNCH_VQ_GEMV(4); }
#undef LAUNCH_VQ_GEMV
}

// Public entry: tiled layout
template <int P, int IB, typename scalar_t, typename ABSMAX_T>
void vqScalarGemvTiled(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax,
    const half* codebook, scalar_t* C, int M, int K_dim, int N, cudaStream_t stream
) {
#define LAUNCH_VQ_GEMV_TILED(MV) \
    vqScalarGemvLaunch<P, IB, MV, true, scalar_t, ABSMAX_T>(A, B_packed, B_absmax, codebook, C, M, K_dim, N, stream)
    if (M <= 1)      { LAUNCH_VQ_GEMV_TILED(1); }
    else if (M <= 2) { LAUNCH_VQ_GEMV_TILED(2); }
    else if (M <= 3) { LAUNCH_VQ_GEMV_TILED(3); }
    else             { LAUNCH_VQ_GEMV_TILED(4); }
#undef LAUNCH_VQ_GEMV_TILED
}

// ---- Tiled Scalar GEMV v2 ----
// Cooperative tile loading into shared memory with split-K for occupancy.
// Grid = n_tiles * k_splits, Block = 128 threads (4 warps).
// Each thread handles one column within an N-tile.
// Double-buffered cp.async pipeline for B + absmax tiles.
// A loaded directly from global memory (L1 broadcast across columns).

template <int K_BITS, int M_VAL, typename scalar_t = half, typename ABSMAX_T = unsigned char>
__global__ void __launch_bounds__(128, 8) kbit_scalar_gemv_tiled_v2(
    const scalar_t* __restrict__ A, const unsigned int* __restrict__ B_packed, const ABSMAX_T* __restrict__ B_absmax,
    const float* __restrict__ codebook, scalar_t* __restrict__ C, float* __restrict__ C_workspace,
    int* __restrict__ tile_counters, const int M, const int K_dim, const int N, const int k_splits
) {
    constexpr int BS = 32; // quantization block size
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 128;
    constexpr int BLOCK_DIM = 128; // threads per block
    constexpr int NUM_WARPS = 4;
    constexpr int M_MAX = 4;
    constexpr int KB_PER_TILE = TILE_K / BS; // 2
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
    if (kt_start >= k_tiles)
        return;

    // This thread's column within the tile
    const int col_in_tile = threadIdx.x; // 0..127
    const int col = n_base + col_in_tile;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Codebook in registers (shuffle-based lookup)
    float cb = (lane_id < (1 << K_BITS)) ? codebook[lane_id] : 0.0f;

    // Double-buffered shared memory
    extern __shared__ char smem[];
    auto sh_b = [&](int stage) -> unsigned int* { return reinterpret_cast<unsigned int*>(smem + stage * STAGE_BYTES); };
    auto sh_abs = [&](int stage) -> ABSMAX_T* {
        return reinterpret_cast<ABSMAX_T*>(smem + stage * STAGE_BYTES + B_STAGE_BYTES);
    };

    // Accumulators
    float acc[M_VAL];
#pragma unroll
    for (int m = 0; m < M_VAL; m++)
        acc[m] = 0.0f;

    // Fetch tile: cooperative cp.async loading of B + absmax
    auto fetch_tile = [&](int stage, int kt) {
        const int tile_idx = kt * n_tiles + n_tile; // K-major tile ordering

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
            if (block_k_base >= K_dim)
                continue;

            // Read bit-planes from shared memory for this column
            int b_addr = col_in_tile * B_COL_WORDS + kb * K_BITS;
            unsigned int planes[K_BITS];
            if constexpr (K_BITS == 2) {
                uint2 pv = *reinterpret_cast<const uint2*>(&b_ptr[b_addr]);
                planes[0] = pv.x;
                planes[1] = pv.y;
            } else if constexpr (K_BITS == 4) {
                int4 pv = *reinterpret_cast<const int4*>(&b_ptr[b_addr]);
                planes[0] = (unsigned int)pv.x;
                planes[1] = (unsigned int)pv.y;
                planes[2] = (unsigned int)pv.z;
                planes[3] = (unsigned int)pv.w;
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
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, int num_sms, cudaStream_t stream
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
    k_splits = (k_tiles + tiles_per_split - 1) / tiles_per_split; // no empty splits

    int grid_size = n_tiles * k_splits;
    int smem_size = 2 * STAGE_BYTES;

    kbit_scalar_gemv_tiled_v2<K, MV, scalar_t, ABSMAX_T><<<grid_size, BLOCK_DIM, smem_size, stream>>>(
        A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, k_splits
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// Public entry point: selects M_VAL template, queries num_sms internally
template <int K, typename scalar_t, typename ABSMAX_T>
void kbitScalarGemvTiledV2(
    const scalar_t* A, const unsigned int* B_packed, const ABSMAX_T* B_absmax, const float* codebook, scalar_t* C,
    float* C_workspace, int* tile_counters, int M, int K_dim, int N, cudaStream_t stream
) {
    const int num_sms = cachedNumSMs();

#define LAUNCH_GEMV_V2(MV)                                                                                             \
    kbitScalarGemvTiledV2Launch<K, MV, scalar_t, ABSMAX_T>(                                                            \
        A, B_packed, B_absmax, codebook, C, C_workspace, tile_counters, M, K_dim, N, num_sms, stream                   \
    )

    if (M <= 1) {
        LAUNCH_GEMV_V2(1);
    } else if (M <= 2) {
        LAUNCH_GEMV_V2(2);
    } else if (M <= 3) {
        LAUNCH_GEMV_V2(3);
    } else {
        LAUNCH_GEMV_V2(4);
    }

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
    template void quantizeBlockwise_kbit<T, K>(                                                                        \
        const float*, const T*, unsigned char*, unsigned int*, int, cudaStream_t                                       \
    );

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
    template void repackKbit<K>(                                                                                       \
        const unsigned int*, const unsigned char*, unsigned int*, unsigned char*, int, int, cudaStream_t               \
    );

INSTANTIATE_KBIT_REPACK(2)
INSTANTIATE_KBIT_REPACK(3)
INSTANTIATE_KBIT_REPACK(4)
INSTANTIATE_KBIT_REPACK(5)

// VQ repack: (P_VAL, INDEX_BITS)
#define INSTANTIATE_VQ_REPACK(P, IB)                                                                                   \
    template void repackVQ<P, IB>(                                                                                     \
        const unsigned int*, const unsigned char*, unsigned int*, unsigned char*, int, int, cudaStream_t                \
    );
INSTANTIATE_VQ_REPACK(4, 8)
INSTANTIATE_VQ_REPACK(3, 8)
INSTANTIATE_VQ_REPACK(3, 10)
INSTANTIATE_VQ_REPACK(2, 8)
INSTANTIATE_VQ_REPACK(2, 10)

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

// VQ GEMM prod instantiations — uint8 E4M4 absmax
#define INSTANTIATE_VQ_GEMM_PROD_U8(P, IB)                                                                             \
    template void vqGemmProd<P, IB, half, unsigned char>(                                                              \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, float*, int*, int, int, int, int,  \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void vqGemmProd<P, IB, __nv_bfloat16, unsigned char>(                                                    \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const half*, __nv_bfloat16*, float*, int*,    \
        int, int, int, int, cudaStream_t                                                                               \
    );
INSTANTIATE_VQ_GEMM_PROD_U8(4, 8)
INSTANTIATE_VQ_GEMM_PROD_U8(3, 8)
INSTANTIATE_VQ_GEMM_PROD_U8(3, 10)
INSTANTIATE_VQ_GEMM_PROD_U8(2, 8)
INSTANTIATE_VQ_GEMM_PROD_U8(2, 10)

// VQ GEMM FP8 MMA prod instantiations
#define INSTANTIATE_VQ_GEMM_PROD_FP8_U8(P, IB)                                                                       \
    template void vqGemmProdFP8<P, IB, half, unsigned char>(                                                          \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, float*, int*, int, int, int, int,  \
        cudaStream_t                                                                                                   \
    );
INSTANTIATE_VQ_GEMM_PROD_FP8_U8(3, 8)
INSTANTIATE_VQ_GEMM_PROD_FP8_U8(2, 8)

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

// VQ Grouped expert GEMM instantiations — uint8 E4M4 absmax
#define INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(P, IB)                                                                    \
    template void vqGroupedGemmProd<P, IB, half, unsigned char>(                                                       \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, float*, int*, const int*, int,      \
        int, int, int, cudaStream_t                                                                                    \
    );                                                                                                                 \
    template void vqGroupedGemmProd<P, IB, __nv_bfloat16, unsigned char>(                                              \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const half*, __nv_bfloat16*, float*, int*,     \
        const int*, int, int, int, int, cudaStream_t                                                                   \
    );
INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(2, 8)
INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(2, 10)
INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(3, 8)
INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(3, 10)
INSTANTIATE_VQ_GROUPED_GEMM_PROD_U8(4, 8)

// VQ Grouped Scalar GEMV instantiations — uint8 E4M4 absmax
#define INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(P, IB)                                                                   \
    template void vqGroupedScalarGemv<P, IB, half, unsigned char>(                                                     \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, const int*, int, int,               \
        int, int, cudaStream_t                                                                                         \
    );                                                                                                                 \
    template void vqGroupedScalarGemv<P, IB, __nv_bfloat16, unsigned char>(                                            \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const half*, __nv_bfloat16*, const int*, int,  \
        int, int, int, cudaStream_t                                                                                    \
    );
INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(2, 8)
INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(2, 10)
INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(3, 8)
INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(3, 10)
INSTANTIATE_VQ_GROUPED_SCALAR_GEMV_U8(4, 8)

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
    template void kbitScalarGemvTiledV2<K, half, unsigned char>(                                                       \
        const half*, const unsigned int*, const unsigned char*, const float*, half*, float*, int*, int, int, int,      \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void kbitScalarGemvTiledV2<K, __nv_bfloat16, unsigned char>(                                              \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const float*, __nv_bfloat16*, float*, int*,   \
        int, int, int, cudaStream_t                                                                                    \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(2)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(3)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(4)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_U8(5)
// fp16 absmax
#define INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(K)                                                                        \
    template void kbitScalarGemvTiledV2<K, half, half>(                                                                \
        const half*, const unsigned int*, const half*, const float*, half*, float*, int*, int, int, int, cudaStream_t  \
    );                                                                                                                 \
    template void kbitScalarGemvTiledV2<K, __nv_bfloat16, half>(                                                       \
        const __nv_bfloat16*, const unsigned int*, const half*, const float*, __nv_bfloat16*, float*, int*, int, int,  \
        int, cudaStream_t                                                                                              \
    );
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(2)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(3)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(4)
INSTANTIATE_KBIT_SCALAR_GEMV_V2_FP16(5)

// ---- VQ template instantiations ----
// quantize_vq: (P_VAL, INDEX_BITS) × scalar_t
#define INSTANTIATE_VQ_QUANT(P, IB)                                                                                    \
    template void quantize_vq<P, IB, half>(const half*, const half*, unsigned char*, unsigned int*, int, cudaStream_t); \
    template void quantize_vq<P, IB, __nv_bfloat16>(                                                                  \
        const half*, const __nv_bfloat16*, unsigned char*, unsigned int*, int, cudaStream_t                             \
    );                                                                                                                 \
    template void quantize_vq<P, IB, float>(const half*, const float*, unsigned char*, unsigned int*, int, cudaStream_t);
INSTANTIATE_VQ_QUANT(4, 8)
INSTANTIATE_VQ_QUANT(3, 8)
INSTANTIATE_VQ_QUANT(3, 10)
INSTANTIATE_VQ_QUANT(2, 8)
INSTANTIATE_VQ_QUANT(2, 10)

// dequantize_vq: (P_VAL, INDEX_BITS) × T × ABSMAX_T
#define INSTANTIATE_VQ_DEQUANT(P, IB)                                                                                  \
    template void dequantize_vq<P, IB, half, unsigned char>(                                                           \
        const unsigned int*, const half*, const unsigned char*, half*, int, cudaStream_t                                \
    );                                                                                                                 \
    template void dequantize_vq<P, IB, __nv_bfloat16, unsigned char>(                                                  \
        const unsigned int*, const half*, const unsigned char*, __nv_bfloat16*, int, cudaStream_t                       \
    );                                                                                                                 \
    template void dequantize_vq<P, IB, half, float>(                                                                   \
        const unsigned int*, const half*, const float*, half*, int, cudaStream_t                                        \
    );                                                                                                                 \
    template void dequantize_vq<P, IB, __nv_bfloat16, float>(                                                         \
        const unsigned int*, const half*, const float*, __nv_bfloat16*, int, cudaStream_t                               \
    );
INSTANTIATE_VQ_DEQUANT(4, 8)
INSTANTIATE_VQ_DEQUANT(3, 8)
INSTANTIATE_VQ_DEQUANT(3, 10)
INSTANTIATE_VQ_DEQUANT(2, 8)
INSTANTIATE_VQ_DEQUANT(2, 10)

// dequantize_vq_tiled: (P_VAL, INDEX_BITS) × T × ABSMAX_T
#define INSTANTIATE_VQ_DEQUANT_TILED(P, IB)                                                                            \
    template void dequantize_vq_tiled<P, IB, half, unsigned char>(                                                     \
        const unsigned int*, const half*, const unsigned char*, half*, int, int, cudaStream_t                           \
    );                                                                                                                 \
    template void dequantize_vq_tiled<P, IB, __nv_bfloat16, unsigned char>(                                            \
        const unsigned int*, const half*, const unsigned char*, __nv_bfloat16*, int, int, cudaStream_t                  \
    );                                                                                                                 \
    template void dequantize_vq_tiled<P, IB, half, float>(                                                             \
        const unsigned int*, const half*, const float*, half*, int, int, cudaStream_t                                   \
    );                                                                                                                 \
    template void dequantize_vq_tiled<P, IB, __nv_bfloat16, float>(                                                    \
        const unsigned int*, const half*, const float*, __nv_bfloat16*, int, int, cudaStream_t                          \
    );
INSTANTIATE_VQ_DEQUANT_TILED(4, 8)
INSTANTIATE_VQ_DEQUANT_TILED(3, 8)
INSTANTIATE_VQ_DEQUANT_TILED(3, 10)
INSTANTIATE_VQ_DEQUANT_TILED(2, 8)
INSTANTIATE_VQ_DEQUANT_TILED(2, 10)

// vq_scalar_gemv: (P_VAL, INDEX_BITS) × scalar_t × ABSMAX_T (flat + tiled)
#define INSTANTIATE_VQ_SCALAR_GEMV_U8(P, IB)                                                                           \
    template void vqScalarGemv<P, IB, half, unsigned char>(                                                            \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, int, int, int, cudaStream_t         \
    );                                                                                                                 \
    template void vqScalarGemv<P, IB, __nv_bfloat16, unsigned char>(                                                   \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const half*, __nv_bfloat16*, int, int, int,    \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void vqScalarGemvTiled<P, IB, half, unsigned char>(                                                       \
        const half*, const unsigned int*, const unsigned char*, const half*, half*, int, int, int, cudaStream_t         \
    );                                                                                                                 \
    template void vqScalarGemvTiled<P, IB, __nv_bfloat16, unsigned char>(                                              \
        const __nv_bfloat16*, const unsigned int*, const unsigned char*, const half*, __nv_bfloat16*, int, int, int,    \
        cudaStream_t                                                                                                   \
    );
// All 5 VQ configs
INSTANTIATE_VQ_SCALAR_GEMV_U8(2, 8)
INSTANTIATE_VQ_SCALAR_GEMV_U8(2, 10)
INSTANTIATE_VQ_SCALAR_GEMV_U8(3, 8)
INSTANTIATE_VQ_SCALAR_GEMV_U8(3, 10)
INSTANTIATE_VQ_SCALAR_GEMV_U8(4, 8)

#define INSTANTIATE_VQ_SCALAR_GEMV_F32(P, IB)                                                                          \
    template void vqScalarGemv<P, IB, half, float>(                                                                    \
        const half*, const unsigned int*, const float*, const half*, half*, int, int, int, cudaStream_t                 \
    );                                                                                                                 \
    template void vqScalarGemv<P, IB, __nv_bfloat16, float>(                                                           \
        const __nv_bfloat16*, const unsigned int*, const float*, const half*, __nv_bfloat16*, int, int, int,            \
        cudaStream_t                                                                                                   \
    );                                                                                                                 \
    template void vqScalarGemvTiled<P, IB, half, float>(                                                               \
        const half*, const unsigned int*, const float*, const half*, half*, int, int, int, cudaStream_t                 \
    );                                                                                                                 \
    template void vqScalarGemvTiled<P, IB, __nv_bfloat16, float>(                                                      \
        const __nv_bfloat16*, const unsigned int*, const float*, const half*, __nv_bfloat16*, int, int, int,            \
        cudaStream_t                                                                                                   \
    );
// All 5 VQ configs
INSTANTIATE_VQ_SCALAR_GEMV_F32(2, 8)
INSTANTIATE_VQ_SCALAR_GEMV_F32(2, 10)
INSTANTIATE_VQ_SCALAR_GEMV_F32(3, 8)
INSTANTIATE_VQ_SCALAR_GEMV_F32(3, 10)
INSTANTIATE_VQ_SCALAR_GEMV_F32(4, 8)
