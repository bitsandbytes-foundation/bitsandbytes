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

// ============================================================================
// NVFP4 quantize/dequantize host-side launchers
// ============================================================================

template <typename T>
void quantizeNVFP4(
    const T* input, unsigned char* output, unsigned char* block_scales,
    float tensor_scale, const int n
) {
    // Each thread handles 2 elements, so we need n/2 threads
    const int threads_per_block = 256;
    const int num_threads = (n + 1) / 2;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    kQuantizeNVFP4<T><<<num_blocks, threads_per_block>>>(
        input, output, block_scales, tensor_scale, n
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T>
void dequantizeNVFP4(
    const unsigned char* input, const unsigned char* block_scales,
    float tensor_scale, T* output, const int n, cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int num_threads = (n + 1) / 2;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    kDequantizeNVFP4<T><<<num_blocks, threads_per_block, 0, stream>>>(
        input, block_scales, tensor_scale, output, n
    );
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

// NVFP4 template instantiations
template void quantizeNVFP4<half>(
    const half* input, unsigned char* output, unsigned char* block_scales,
    float tensor_scale, const int n
);
template void quantizeNVFP4<__nv_bfloat16>(
    const __nv_bfloat16* input, unsigned char* output, unsigned char* block_scales,
    float tensor_scale, const int n
);
template void quantizeNVFP4<float>(
    const float* input, unsigned char* output, unsigned char* block_scales,
    float tensor_scale, const int n
);
template void dequantizeNVFP4<half>(
    const unsigned char* input, const unsigned char* block_scales,
    float tensor_scale, half* output, const int n, cudaStream_t stream
);
template void dequantizeNVFP4<__nv_bfloat16>(
    const unsigned char* input, const unsigned char* block_scales,
    float tensor_scale, __nv_bfloat16* output, const int n, cudaStream_t stream
);
template void dequantizeNVFP4<float>(
    const unsigned char* input, const unsigned char* block_scales,
    float tensor_scale, float* output, const int n, cudaStream_t stream
);

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
