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

int fill_up_to_nearest_multiple(int value, int multiple) {
    return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
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

#define MAKE_optimizerStatic8bit(name, gtype)                                                                          \
    template void optimizerStatic8bit<gtype, name>(                                                                    \
        gtype * p, gtype * g, unsigned char* state1, unsigned char* state2, float* unorm, float max_unorm,             \
        float param_norm, float beta1, float beta2, float eps, int step, float lr, float* quantiles1,                  \
        float* quantiles2, float* max1, float* max2, float* new_max1, float* new_max2, float weight_decay,             \
        const float gnorm_scale, int n                                                                                 \
    );

MAKE_optimizerStatic8bit(ADAM, half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, half)
MAKE_optimizerStatic8bit(RMSPROP, float)
MAKE_optimizerStatic8bit(LION, half)
MAKE_optimizerStatic8bit(LION, float)
MAKE_optimizerStatic8bit(ADAGRAD, half)
MAKE_optimizerStatic8bit(ADAGRAD, float)

template void percentileClipping(float* g, float* gnorm_vec, int step, const int n);
template void percentileClipping(half* g, float* gnorm_vec, int step, const int n);
