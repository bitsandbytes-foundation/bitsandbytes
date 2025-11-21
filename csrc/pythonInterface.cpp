// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#if BUILD_CUDA
#include <cuda_runtime_api.h>
#include <ops.cuh>
#endif
#if BUILD_HIP
#include <ops_hip.cuh>
#endif
#if BUILD_MPS
// #include <mps_ops.h>
#endif
#if BUILD_XPU
#include <xpu_ops.h>
#endif
#include <cpu_ops.h>

// Compatibility between HIP/CUDA APIs
#if BUILD_HIP
#define cudaStream_t hipStream_t
#define __nv_bfloat16 hip_bfloat16
#define cublasLtHandle_t hipblasLtHandle_t
#define ContextCusparse ContextHipsparse
#define cusparseHandle_t hipsparseHandle_t
#define cudaMallocManaged hipMallocManaged
#define cudaMemAttachHost hipMemAttachHost
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#endif

// We cannot call templated code from C, so we wrap the template in a C compatible call here if necessary.
// We use macro functions to expand all the different optimizers. Looks ugly, and is ugly, but its better than to
// maintain all that boilerplate
//===================================================================================
//                               UNMANGLED CALLS
//===================================================================================

#if BUILD_CUDA || BUILD_HIP

// void gemm_host_fp32(int M, int N, int K, float * A,  float* B,  float * out,  int lda, int ldb, int ldc)
//{ gemm_host<float>(M, N, K, A, B, out, lda, ldb, ldc, 32); }
void gemm_host_fp16(int M, int N, int K, half* A, half* B, half* out, int lda, int ldb, int ldc) {
    gemm_host<half>(M, N, K, A, B, out, lda, ldb, ldc, 16);
}

void gemm_4bit_inference(
    int m, int n, int k, half* A, unsigned char* B, float* absmax, half* out, int lda, int ldb, int ldc, int blocksize
) {
    gemm_4bit_inference<half>(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize);
}

void gemm_4bit_inference_naive_fp16(
    int m, int n, int k, half* A, unsigned char* B, float* absmax, float* datatype, half* out, int lda, int ldb,
    int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive<half, 16>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void gemm_4bit_inference_naive_bf16(
    int m, int n, int k, __nv_bfloat16* A, unsigned char* B, float* absmax, float* datatype, __nv_bfloat16* out,
    int lda, int ldb, int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive<__nv_bfloat16, 16>(
        m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream
    );
}

void gemm_4bit_inference_naive_fp32(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive<float, 32>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

#define MAKE_ELEMENTWISE_FUNC(fname, type_name, ctype, FUNC)                                                           \
    void fname##_##type_name(ctype* A, ctype* B, ctype value, long n) { func<ctype, FUNC>(A, B, value, n); }

MAKE_ELEMENTWISE_FUNC(fill, fp32, float, FILL)
MAKE_ELEMENTWISE_FUNC(fill, uint8, unsigned char, FILL)
MAKE_ELEMENTWISE_FUNC(arange, fp32, float, ARANGE)
MAKE_ELEMENTWISE_FUNC(_mul, fp32, float, _MUL)

#define MAKE_FUNC32(fname, oname, gtype, gbits)                                                                        \
    void fname##32bit_grad_##gbits(                                                                                    \
        gtype* g, gtype* p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm,             \
        const float beta1, const float beta2, const float beta3, const float alpha, const float eps,                   \
        const float weight_decay, const int step, const float lr, float gnorm_scale, bool skip_zeros, const int n      \
    ) {                                                                                                                \
        optimizer32bit<gtype, oname>(                                                                                  \
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, beta3, alpha, eps, weight_decay, step,   \
            lr, gnorm_scale, skip_zeros, n                                                                             \
        );                                                                                                             \
    }

MAKE_FUNC32(momentum, MOMENTUM, float, 32)
MAKE_FUNC32(momentum, MOMENTUM, half, 16)
MAKE_FUNC32(adam, ADAM, float, fp32)
MAKE_FUNC32(adam, ADAM, half, fp16)
MAKE_FUNC32(adam, ADAM, __nv_bfloat16, bf16)
MAKE_FUNC32(rmsprop, RMSPROP, float, 32)
MAKE_FUNC32(rmsprop, RMSPROP, half, 16)
MAKE_FUNC32(lion, LION, float, fp32)
MAKE_FUNC32(lion, LION, half, fp16)
MAKE_FUNC32(lion, LION, __nv_bfloat16, bf16)
MAKE_FUNC32(adagrad, ADAGRAD, float, 32)
MAKE_FUNC32(adagrad, ADAGRAD, half, 16)
MAKE_FUNC32(ademamix, ADEMAMIX, float, fp32)
MAKE_FUNC32(ademamix, ADEMAMIX, half, fp16)
MAKE_FUNC32(ademamix, ADEMAMIX, __nv_bfloat16, bf16)

#define MAKE_FUNC8(fname, oname, gtype, gbits)                                                                         \
    void fname##_static_8bit_grad_##gbits(                                                                             \
        gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, float* unorm, float max_unorm,               \
        float param_norm, float beta1, float beta2, float eps, int step, float lr, float* quantiles1,                  \
        float* quantiles2, float* max1, float* max2, float* new_max1, float* new_max2, float weight_decay,             \
        float gnorm_scale, int n                                                                                       \
    ) {                                                                                                                \
        optimizerStatic8bit<gtype, oname>(                                                                             \
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2,   \
            max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n                                               \
        );                                                                                                             \
    }

MAKE_FUNC8(adam, ADAM, float, 32)
MAKE_FUNC8(adam, ADAM, half, 16)
MAKE_FUNC8(momentum, MOMENTUM, float, 32)
MAKE_FUNC8(momentum, MOMENTUM, half, 16)
MAKE_FUNC8(rmsprop, RMSPROP, float, 32)
MAKE_FUNC8(rmsprop, RMSPROP, half, 16)
MAKE_FUNC8(lion, LION, float, 32)
MAKE_FUNC8(lion, LION, half, 16)

#define MAKE_BLOCKWISE8(fname, optim_name, gtype, gbits)                                                               \
    void fname##_8bit_blockwise_grad_##gbits(                                                                          \
        gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3,       \
        float alpha, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1,              \
        float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n                            \
    ) {                                                                                                                \
        optimizerStatic8bitBlockwise<gtype, optim_name>(                                                               \
            p, g, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, \
            weight_decay, gnorm_scale, skip_zeros, n                                                                   \
        );                                                                                                             \
    }

MAKE_BLOCKWISE8(adam, ADAM, half, fp16)
MAKE_BLOCKWISE8(adam, ADAM, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(adam, ADAM, float, fp32)
MAKE_BLOCKWISE8(momentum, MOMENTUM, half, fp16)
MAKE_BLOCKWISE8(momentum, MOMENTUM, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(momentum, MOMENTUM, float, fp32)
MAKE_BLOCKWISE8(rmsprop, RMSPROP, half, fp16)
MAKE_BLOCKWISE8(rmsprop, RMSPROP, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(rmsprop, RMSPROP, float, fp32)
MAKE_BLOCKWISE8(adagrad, ADAGRAD, half, fp16)
MAKE_BLOCKWISE8(adagrad, ADAGRAD, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(adagrad, ADAGRAD, float, fp32)
MAKE_BLOCKWISE8(lion, LION, half, fp16)
MAKE_BLOCKWISE8(lion, LION, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(lion, LION, float, fp32)
MAKE_BLOCKWISE8(ademamix, ADEMAMIX, half, fp16)
MAKE_BLOCKWISE8(ademamix, ADEMAMIX, __nv_bfloat16, bf16)
MAKE_BLOCKWISE8(ademamix, ADEMAMIX, float, fp32)

void percentileClipping_g32(float* g, float* gnorm_vec, int step, const int n) {
    percentileClipping<float>(g, gnorm_vec, step, n);
}

void percentileClipping_g16(half* g, float* gnorm_vec, int step, const int n) {
    percentileClipping<half>(g, gnorm_vec, step, n);
}

void quantizeBlockwise_fp16(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<half, 0, General8bit>(code, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_fp16_fp4(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<half, 0, FP4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_fp16_nf4(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<half, 0, NF4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_bf16(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise<__nv_bfloat16, 0, General8bit>(code, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_bf16_fp4(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise<__nv_bfloat16, 0, FP4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_bf16_nf4(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise<__nv_bfloat16, 0, NF4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_fp32(float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<float, 0, General8bit>(code, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_fp32_fp4(float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<float, 0, FP4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void quantizeBlockwise_fp32_nf4(float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<float, 0, NF4>(nullptr, A, absmax, out, nullptr, 0, blocksize, n);
}

void dequantizeBlockwise_fp16(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<half, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp16_fp4(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<half, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp16_nf4(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<half, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32_fp4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<float, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32_nf4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<float, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<__nv_bfloat16, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16_fp4(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<__nv_bfloat16, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16_nf4(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise<__nv_bfloat16, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

int igemmlt_32(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt<32, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

int igemmlt_8(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt<8, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

int igemmlt_8_rowscale(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt<8, 1>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

void spmm_coo_very_sparse_naive_fp16(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
) {
    spmm_coo_very_sparse_naive<half, 16>(
        max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB,
        colsB
    );
}

void spmm_coo_very_sparse_naive_int8(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
) {
    spmm_coo_very_sparse_naive<signed char, 8>(
        max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB,
        colsB
    );
}
#endif

#if BUILD_XPU

void dequantizeBlockwise_fp16(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<sycl::half, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp16_fp4(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<sycl::half, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp16_nf4(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<sycl::half, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<float, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32_fp4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<float, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_fp32_nf4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise<float, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise<sycl::ext::oneapi::bfloat16, General8bit>(code, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16_fp4(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise<sycl::ext::oneapi::bfloat16, FP4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void dequantizeBlockwise_bf16_nf4(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise<sycl::ext::oneapi::bfloat16, NF4>(nullptr, A, absmax, out, blocksize, n, stream);
}

void gemv_4bit_inference_fp16(
    int m, int n, int k, sycl::half* A, unsigned char* B, float* absmax, float* datatype, sycl::half* out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference<sycl::half, 16>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void gemv_4bit_inference_bf16(
    int m, int n, int k, sycl::ext::oneapi::bfloat16* A, unsigned char* B, float* absmax, float* datatype,
    sycl::ext::oneapi::bfloat16* out, int lda, int ldb, int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference<sycl::ext::oneapi::bfloat16, 16>(
        m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream
    );
}

void gemv_4bit_inference_fp32(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference<float, 32>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

#endif

extern "C" {
#if BUILD_CUDA || BUILD_HIP
void cquantize(float* code, float* A, unsigned char* out, int n) { quantize(code, A, out, n); }

void cdequantize(float* code, unsigned char* A, float* out, int n, cudaStream_t stream) {
    dequantize(code, A, out, n, stream);
}

void cdequantize_blockwise_fp16_fp4(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp16(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp16(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp16_nf4(
    float* code, unsigned char* A, float* absmax, half* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp16_nf4(code, A, absmax, out, blocksize, n, stream);
}

void cquantize_blockwise_fp16(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise_fp16(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp16_fp4(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp16_nf4(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise_fp16_nf4(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp32(float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise_fp32(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp32_fp4(
    float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp32_nf4(
    float* code, float* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp32(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp32_fp4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp32_nf4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n, stream);
}

void cquantize_blockwise_bf16(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise_bf16(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_bf16_fp4(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n);
}

void cquantize_blockwise_bf16_nf4(
    float* code, __nv_bfloat16* A, float* absmax, unsigned char* out, int blocksize, const int n
) {
    quantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_bf16(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_bf16(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_bf16_fp4(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_bf16_nf4(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, int blocksize, const int n, cudaStream_t stream
) {
    dequantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n, stream);
}

#define MAKE_CFUNC32(name, gtype, gbits)                                                                               \
    void c##name##32bit_grad_##gbits(                                                                                  \
        gtype* g, gtype* p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm,             \
        const float beta1, const float beta2, const float beta3, const float alpha, const float eps,                   \
        const float weight_decay, const int step, const float lr, const float gnorm_scale, bool skip_zeros,            \
        const int n                                                                                                    \
    ) {                                                                                                                \
        name##32bit_grad_##gbits(                                                                                      \
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, beta3, alpha, eps, weight_decay, step,   \
            lr, gnorm_scale, skip_zeros, n                                                                             \
        );                                                                                                             \
    }

MAKE_CFUNC32(adam, float, fp32)
MAKE_CFUNC32(adam, half, fp16)
MAKE_CFUNC32(adam, __nv_bfloat16, bf16)
MAKE_CFUNC32(momentum, float, 32)
MAKE_CFUNC32(momentum, half, 16)
MAKE_CFUNC32(rmsprop, float, 32)
MAKE_CFUNC32(rmsprop, half, 16)
MAKE_CFUNC32(lion, float, fp32)
MAKE_CFUNC32(lion, half, fp16)
MAKE_CFUNC32(lion, __nv_bfloat16, bf16)
MAKE_CFUNC32(adagrad, float, 32)
MAKE_CFUNC32(adagrad, half, 16)
MAKE_CFUNC32(ademamix, float, fp32)
MAKE_CFUNC32(ademamix, half, fp16)
MAKE_CFUNC32(ademamix, __nv_bfloat16, bf16)

#define MAKE_CFUNC8(name, gtype, gbits)                                                                                \
    void c##name##_static_8bit_grad_##gbits(                                                                           \
        gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, float* unorm, float max_unorm,               \
        float param_norm, float beta1, float beta2, float eps, int step, float lr, float* quantiles1,                  \
        float* quantiles2, float* max1, float* max2, float* new_max1, float* new_max2, float weight_decay,             \
        float gnorm_scale, int n                                                                                       \
    ) {                                                                                                                \
        name##_static_8bit_grad_##gbits(                                                                               \
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2,   \
            max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n                                               \
        );                                                                                                             \
    }

MAKE_CFUNC8(adam, float, 32)
MAKE_CFUNC8(adam, half, 16)
MAKE_CFUNC8(momentum, float, 32)
MAKE_CFUNC8(momentum, half, 16)
MAKE_CFUNC8(rmsprop, float, 32)
MAKE_CFUNC8(rmsprop, half, 16)
MAKE_CFUNC8(lion, float, 32)
MAKE_CFUNC8(lion, half, 16)

#define MAKE_CBLOCKWISE8(fname, optim_name, gtype, gbits)                                                              \
    void c##fname##_8bit_blockwise_grad_##gbits(                                                                       \
        gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3,       \
        float alpha, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1,              \
        float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n                            \
    ) {                                                                                                                \
        fname##_8bit_blockwise_grad_##gbits(                                                                           \
            p, g, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, \
            weight_decay, gnorm_scale, skip_zeros, n                                                                   \
        );                                                                                                             \
    }

MAKE_CBLOCKWISE8(adam, ADAM, half, fp16)
MAKE_CBLOCKWISE8(adam, ADAM, float, fp32)
MAKE_CBLOCKWISE8(adam, ADAM, __nv_bfloat16, bf16)
MAKE_CBLOCKWISE8(momentum, MOMENTUM, half, fp16)
MAKE_CBLOCKWISE8(momentum, MOMENTUM, float, fp32)
MAKE_CBLOCKWISE8(momentum, MOMENTUM, __nv_bfloat16, bf16)
MAKE_CBLOCKWISE8(rmsprop, RMSPROP, half, fp16)
MAKE_CBLOCKWISE8(rmsprop, RMSPROP, float, fp32)
MAKE_CBLOCKWISE8(rmsprop, RMSPROP, __nv_bfloat16, bf16)
MAKE_CBLOCKWISE8(adagrad, ADAGRAD, half, fp16)
MAKE_CBLOCKWISE8(adagrad, ADAGRAD, float, fp32)
MAKE_CBLOCKWISE8(adagrad, ADAGRAD, __nv_bfloat16, bf16)
MAKE_CBLOCKWISE8(lion, LION, half, fp16)
MAKE_CBLOCKWISE8(lion, LION, float, fp32)
MAKE_CBLOCKWISE8(lion, LION, __nv_bfloat16, bf16)
MAKE_CBLOCKWISE8(ademamix, ADEMAMIX, half, fp16)
MAKE_CBLOCKWISE8(ademamix, ADEMAMIX, float, fp32)
MAKE_CBLOCKWISE8(ademamix, ADEMAMIX, __nv_bfloat16, bf16)

void cpercentile_clipping_g32(float* g, float* gnorm_vec, int step, const int n) {
    percentileClipping_g32(g, gnorm_vec, step, n);
}

void cpercentile_clipping_g16(half* g, float* gnorm_vec, int step, const int n) {
    percentileClipping_g16(g, gnorm_vec, step, n);
}

void cigemm(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc
) {
    gemmex(context, transposeA, transposeB, m, n, k, A, B, C, lda, ldb, ldc);
}

void cbatched_igemm(
    Context* context, bool transposeA, bool transposeB, int m, int n, int k, void* A, void* B, void* C, int lda,
    int ldb, int ldc, long strideA, long strideB, long strideC, int batchCount
) {
    strided_gemmex(
        context, transposeA, transposeB, m, n, k, A, B, C, lda, ldb, ldc, strideA, strideB, strideC, batchCount
    );
}

Context* get_context() { return new Context(); }

ContextCusparse* get_cusparse() { return new ContextCusparse(); }

int cigemmlt_32(
    Context* context, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale, int lda,
    int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt_32((cublasLtHandle_t)context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

int cigemmlt_8(
    Context* context, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale, int lda,
    int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt_8((cublasLtHandle_t)context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

int cigemmlt_8_rowscale(
    Context* context, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale, int lda,
    int ldb, int ldc, cudaStream_t stream
) {
    return igemmlt_8_rowscale((cublasLtHandle_t)context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc, stream);
}

void cdequant_mm_int32_fp16(
    int* A, float* rowStats, float* colStats, half* out, half* bias, int numRows, int numCols, cudaStream_t stream
) {
    dequant_mm_int32_fp16(A, rowStats, colStats, out, bias, numRows, numCols, stream);
}

void cint8_vector_quant(
    half* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols, cudaStream_t stream
) {
    int8VectorQuant(A, out, rowStats, threshold, rows, cols, stream);
}

void cspmm_coo(
    ContextCusparse* context, int* A_rowidx, int* A_colidx, half* A_vals, int A_nnz, int A_rows, int A_cols, int B_cols,
    int ldb, half* B, int ldc, half* C, bool transposed_B
) {
    spmm_coo(
        (cusparseHandle_t)context->m_handle, A_rowidx, A_colidx, A_vals, A_nnz, A_rows, A_cols, B_cols, ldb, B, ldc, C,
        transposed_B
    );
}

void cspmm_coo_very_sparse_naive_fp16(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
) {
    spmm_coo_very_sparse_naive_fp16(
        max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB,
        colsB
    );
}

void cspmm_coo_very_sparse_naive_int8(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB
) {
    spmm_coo_very_sparse_naive_int8(
        max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB,
        colsB
    );
}

// void cgemm_host_fp32(int M, int N, int K, float * A,  float* B,  float * out,  int lda, int ldb, int ldc)
//{ gemm_host_fp32(M, N, K, A, B, out, lda, ldb, ldc); }

void cgemm_host_fp16(int M, int N, int K, half* A, half* B, half* out, int lda, int ldb, int ldc) {
    gemm_host_fp16(M, N, K, A, B, out, lda, ldb, ldc);
}

void cgemm_4bit_inference(
    int m, int n, int k, half* A, unsigned char* B, float* absmax, half* out, int lda, int ldb, int ldc, int blocksize
) {
    gemm_4bit_inference(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize);
}

void* cget_managed_ptr(size_t bytes) {
    void* ptr;
    CUDA_CHECK_RETURN(cudaMallocManaged(&ptr, bytes, cudaMemAttachHost));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    return ptr;
}

void cprefetch(void* ptr, size_t bytes, int device) {

    int hasPrefetch = 0;
    CUDA_CHECK_RETURN(
        cudaDeviceGetAttribute(&hasPrefetch, cudaDevAttrConcurrentManagedAccess, device)
    ); // 40ns overhead
    if (hasPrefetch == 0)
        return;

#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
    CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, loc, 0u, 0));
#else
    CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, device, 0));
#endif

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

#define CMAKE_ELEMENTWISE_FUNC(fname, type_name, ctype, FUNC)                                                          \
    void c##fname##_##type_name(ctype* A, ctype* B, ctype value, long n) { fname##_##type_name(A, B, value, n); }

CMAKE_ELEMENTWISE_FUNC(fill, fp32, float, FILL)
CMAKE_ELEMENTWISE_FUNC(fill, uint8, unsigned char, FILL)
CMAKE_ELEMENTWISE_FUNC(arange, fp32, float, ARANGE)
CMAKE_ELEMENTWISE_FUNC(_mul, fp32, float, _MUL)

void cgemm_4bit_inference_naive_fp16(
    int m, int n, int k, half* A, unsigned char* B, float* absmax, float* datatype, half* out, int lda, int ldb,
    int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive_fp16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void cgemm_4bit_inference_naive_bf16(
    int m, int n, int k, __nv_bfloat16* A, unsigned char* B, float* absmax, float* datatype, __nv_bfloat16* out,
    int lda, int ldb, int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive_bf16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void cgemm_4bit_inference_naive_fp32(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, cudaStream_t stream
) {
    gemm_4bit_inference_naive_fp32(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

#endif

#if BUILD_XPU

void cdequantize_blockwise_fp16_fp4(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp16_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp16(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp16(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp16_nf4(
    float* code, unsigned char* A, float* absmax, sycl::half* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp16_nf4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp32(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp32(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp32_fp4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp32_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_fp32_nf4(
    float* code, unsigned char* A, float* absmax, float* out, int blocksize, const int n, sycl::queue* stream
) {
    dequantizeBlockwise_fp32_nf4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_bf16(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise_bf16(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_bf16_fp4(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise_bf16_fp4(code, A, absmax, out, blocksize, n, stream);
}

void cdequantize_blockwise_bf16_nf4(
    float* code, unsigned char* A, float* absmax, sycl::ext::oneapi::bfloat16* out, int blocksize, const int n,
    sycl::queue* stream
) {
    dequantizeBlockwise_bf16_nf4(code, A, absmax, out, blocksize, n, stream);
}

void cgemv_4bit_inference_fp16(
    int m, int n, int k, sycl::half* A, unsigned char* B, float* absmax, float* datatype, sycl::half* out, int lda,
    int ldb, int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference_fp16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void cgemv_4bit_inference_bf16(
    int m, int n, int k, sycl::ext::oneapi::bfloat16* A, unsigned char* B, float* absmax, float* datatype,
    sycl::ext::oneapi::bfloat16* out, int lda, int ldb, int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference_bf16(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

void cgemv_4bit_inference_fp32(
    int m, int n, int k, float* A, unsigned char* B, float* absmax, float* datatype, float* out, int lda, int ldb,
    int ldc, int blocksize, sycl::queue* stream
) {
    gemv_4bit_inference_fp32(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, stream);
}

#endif

void cquantize_blockwise_cpu_fp32(
    float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n
) {
    quantize_cpu(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_cpu_fp32(
    float* code, unsigned char* A, const float* absmax, float* out, long long blocksize, long long n
) {
    dequantizeBlockwise8bitCpu<float>(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_cpu_bf16(
    float* code, unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long n
) {
    dequantizeBlockwise8bitCpu<bf16_t>(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_cpu_fp16(
    float* code, unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long n
) {
    dequantizeBlockwise8bitCpu<fp16_t>(code, A, absmax, out, blocksize, n);
}

void cdequantize_blockwise_cpu_fp4_fp32(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<float, FP4>(A, absmax, out, blocksize, m, n);
}

void cdequantize_blockwise_cpu_fp4_bf16(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<bf16_t, FP4>(A, absmax, out, blocksize, m, n);
}

void cdequantize_blockwise_cpu_fp4_fp16(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<fp16_t, FP4>(A, absmax, out, blocksize, m, n);
}

void cdequantize_blockwise_cpu_nf4_fp32(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<float, NF4>(A, absmax, out, blocksize, m, n);
}

void cdequantize_blockwise_cpu_nf4_bf16(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<bf16_t, NF4>(A, absmax, out, blocksize, m, n);
}

void cdequantize_blockwise_cpu_nf4_fp16(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
) {
    dequantizeBlockwise4bitCpu<fp16_t, NF4>(A, absmax, out, blocksize, m, n);
}

#if defined(__AVX512F__) && defined(__AVX512BF16__)
void gemv_4bit_inference_cpu_fp4_bf16(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
) {
    gemv_4bit_inference<bf16_t, FP4>(M, N, K, x, w, absmax, out, blocksize, x_stride, out_stride);
}

void gemv_4bit_inference_cpu_nf4_bf16(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
) {
    gemv_4bit_inference<bf16_t, NF4>(M, N, K, x, w, absmax, out, blocksize, x_stride, out_stride);
}
#endif
#if defined(__AVX512F__)
bool has_avx512f_cpu() { return has_avx512f(); }
#if defined(__AVX512BF16__)
bool has_avx512bf16_cpu() { return has_avx512bf16(); }
#endif
#endif
}
