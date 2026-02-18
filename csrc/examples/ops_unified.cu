// ops_unified.cu — EXAMPLE of merged host wrappers for CUDA/HIP
//
// This replaces both csrc/ops.cu and csrc/ops.hip. Shows representative
// functions covering all categories of differences.
//
// Key points:
//   - <<<grid, block>>> works on both CUDA and HIP (no hipLaunchKernelGGL needed)
//   - BNB_CHECK_RETURN replaces CUDA_CHECK_RETURN / hip equivalent
//   - bnb_stream_t replaces cudaStream_t / hipStream_t
//   - #if BNB_HIP only for genuinely different library code (igemmlt, spmm_coo)

#include "common.cuh"
#include "compat.cuh"
#include "kernels.cuh"
#include "ops_unified.cuh"

#include <cassert>
#include <common.h>
#include <limits>

#if !BNB_HIP
#include <cub/device/device_scan.cuh>
#endif

#define ERR_NOT_IMPLEMENTED 100

using std::cout;
using std::endl;

// ============================================================================
// Quantize / Dequantize — fully shared, <<<>>> works on both platforms
// ============================================================================

void quantize(float* code, float* A, unsigned char* out, int n) {
    int num_blocks = n / 1024;
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    kQuantize<<<num_blocks, 1024>>>(code, A, out, n);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

void dequantize(float* code, unsigned char* A, float* out, int n, bnb_stream_t stream) {
    int num_blocks = n / 1024;
    num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
    kDequantize<<<num_blocks, 1024, 0, stream>>>(code, A, out, n);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

// ============================================================================
// quantizeBlockwise — mostly shared, small warp-size dispatch difference
// ============================================================================

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
    // Smallest blocksize: uses unified kQuantizeBlockwiseSmall
    // BNB_WARP_SIZE is the compile-time block size (32 on CUDA, 32 or 64 on HIP)
    else if (blocksize == BNB_WARP_SIZE) {
        if constexpr (DATA_TYPE > 0) {
            int num_blocks_adjusted = (num_blocks + 1) / 2;
            kQuantizeBlockwiseSmall<T, DATA_TYPE>
                <<<num_blocks_adjusted, BNB_WARP_SIZE>>>(code, A, absmax, out, rand, rand_offset, n);
        }
    }

    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

// ============================================================================
// dequantizeBlockwise — fully shared
// ============================================================================

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code, unsigned char* A, float* absmax, T* out, int blocksize, const int n, bnb_stream_t stream
) {
    constexpr int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
    int grid_blocks = ((int64_t)n + tile_size - 1) / tile_size;

    if (DATA_TYPE > 0)
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>
            <<<grid_blocks, 64, 0, stream>>>(code, A, absmax, out, blocksize / 2, n);
    else
        kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>
            <<<grid_blocks, 64, 0, stream>>>(code, A, absmax, out, blocksize, n);

    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

// ============================================================================
// gemm_4bit_inference_naive — small warp-size difference in block count
// ============================================================================

template <typename T, int BITS>
void gemm_4bit_inference_naive(
    int m, int n, int k, T* A, unsigned char* B, float* absmax, float* datatype, T* out, int lda, int ldb, int ldc,
    int blocksize, bnb_stream_t stream
) {
    // Warp size affects how many rows each block processes
    int num_blocks;
    if constexpr (BNB_WARP_SIZE == 64)
        num_blocks = (m + 1) / 2;
    else
        num_blocks = (m + 3) / 4;

    kgemm_4bit_inference_naive<T, 128, BITS>
        <<<num_blocks, 128, 0, stream>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

// ============================================================================
// igemmlt — BLAS library calls genuinely differ between cuBLAS and hipBLAS
//
// This is one of the few functions requiring substantial #if BNB_HIP blocks.
// The algorithm is the same but hipBLAS requires explicit heuristic selection
// while cuBLAS auto-selects.
// ============================================================================

template <int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(
    bnb_blasLt_handle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, bnb_stream_t stream
) {
#if BNB_HIP && defined(NO_HIPBLASLT)
    return ERR_NOT_IMPLEMENTED;
#else
    int has_error = 0;

    bnb_blasLt_matmul_desc_t matmulDesc;
    bnb_blasLt_layout_t aDesc, bDesc, cDesc;

    auto outType = DTYPE_OUT == 32 ? BNB_R_32I : BNB_R_8I;
    auto scaleType = DTYPE_OUT == 32 ? BNB_R_32I : BNB_R_32F;
    auto opT = BNB_BLASLT_OP_T;

    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&aDesc, BNB_R_8I, m, k, lda));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&bDesc, BNB_R_8I, m, n, ldb));
    has_error |= checkBlasLtStatus(bnb_blasLtLayoutCreate(&cDesc, outType, k, n, ldc));

    has_error |= checkBlasLtStatus(bnb_blasLtMatmulDescCreate(&matmulDesc, BNB_BLASLT_COMPUTE_32I, scaleType));
    has_error |= checkBlasLtStatus(bnb_blasLtMatmulDescSetAttr(matmulDesc, BNB_BLASLT_DESC_TRANSA, &opT, sizeof(opT)));

    if (DTYPE_OUT == 32) {
        int alpha = 1, beta = 0;

#if BNB_HIP
        // HIP requires explicit algorithm heuristic selection
        bnb_blasLt_preference_t pref;
        const int64_t max_workspace_size = 0;
        checkBlasLtStatus(bnb_blasLtPrefCreate(&pref));
        checkBlasLtStatus(
            bnb_blasLtPrefSetAttr(pref, BNB_BLASLT_PREF_MAX_WORKSPACE, &max_workspace_size, sizeof(max_workspace_size))
        );

        bnb_blasLt_heuristic_t heuristicResult[1];
        int returnedAlgoCount = 0;
        checkBlasLtStatus(bnb_blasLtAlgoGetHeuristic(
            ltHandle, matmulDesc, aDesc, bDesc, cDesc, cDesc, pref, 1, heuristicResult, &returnedAlgoCount
        ));

        if (returnedAlgoCount == 0) {
            has_error = 1;
            fprintf(stderr, "Error: Matmul Algo Heuristic didn't return algorithms\n");
        } else {
            has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
                ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int32_t*)C, cDesc, (int32_t*)C, cDesc,
                &heuristicResult[0].algo, NULL, 0, stream
            ));
        }
#else
        // CUDA: cuBLAS auto-selects algorithm
        has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
            ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int32_t*)C, cDesc, (int32_t*)C, cDesc, NULL, NULL,
            0, stream
        ));
#endif
    } else {
        if (!SCALE_ROWS) {
            float alpha = 1.0f, beta = 0.0f;
            has_error |= checkBlasLtStatus(bnb_blasLtMatmul(
                ltHandle, matmulDesc, &alpha, A, aDesc, B, bDesc, &beta, (int8_t*)C, cDesc, (int8_t*)C, cDesc, NULL,
                NULL, 0, stream
            ));
        } else {
            auto pointerMode = BNB_BLASLT_PTR_MODE_ALPHA_VEC;
            float beta = 0.0f;
            has_error |= checkBlasLtStatus(
                bnb_blasLtMatmulDescSetAttr(matmulDesc, BNB_BLASLT_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode))
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
#endif
}

// ============================================================================
// spmm_coo — sparse library calls differ but structure is identical
// Uses unified CHECK_SPARSE and bnb_sparse* macros from compat.cuh
// ============================================================================

void spmm_coo(
    bnb_sparse_handle_t handle, int* A_rowidx, int* A_colidx, half* A_vals, int A_nnz, int A_rows, int A_cols,
    int B_cols, int ldb, half* B, int ldc, half* C, bool transposed_B
) {
#if BNB_HIP && defined(NO_HIPBLASLT)
    // No sparse support on older ROCm
#else
    float alpha = 1.0f;
    float beta = 0.0f;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Note: all of these use the bnb_sparse* macros from compat.cuh
    // which resolve to cusparse* or hipsparse* as appropriate

    // bnb_sparseCreateCoo → cusparseCreateCoo / hipsparseCreateCoo
    // BNB_R_16F → CUDA_R_16F / HIP_R_16F
    // etc.

    // Omitting the body as it would be identical to what compat.cuh provides
    // (see full macro mappings in compat.cuh)

    CHECK_SPARSE(bnb_sparseCreateCoo(
        NULL, A_rows, A_cols, A_nnz, A_rowidx, A_colidx, A_vals, BNB_SPARSE_INDEX_32I, BNB_SPARSE_INDEX_BASE_ZERO,
        BNB_R_16F
    ));

    // ... (rest of spmm_coo using bnb_sparse* macros — same pattern)
#endif
}

// ============================================================================
// Simple kernel launchers — fully shared
// ============================================================================

void dequant_mm_int32_fp16(
    int* A, float* rowStats, float* colStats, half* out, half* bias, int numRows, int numCols, bnb_stream_t stream
) {
    const int threads = 512;
    const int num_per_thread = 4;
    const int n = numRows * numCols;
    const int num_blocks = (n + threads * num_per_thread - 1) / (threads * num_per_thread);

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

template <typename T, int FUNC> void func(T* A, T* B, T value, long n) {
    int threads = 512;
    int blocks = n / threads;
    blocks = n % threads == 0 ? blocks : blocks + 1;
    blocks = blocks > 65535 ? 65535 : blocks;
    kfunc<T, FUNC><<<blocks, 512>>>(A, B, value, n);
    BNB_CHECK_RETURN(BNB_PEEK_LAST_ERROR());
}

// ============================================================================
// Template instantiations
// ============================================================================

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
