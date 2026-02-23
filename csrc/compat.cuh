// compat.cuh — Platform abstraction layer for CUDA/HIP portability
//
// This header resolves ALL mechanical differences between CUDA and HIP.
// Kernel code should include this header and use the bnb_* types/macros
// instead of cuda*/hip* identifiers directly.
//
// The guard macro is BNB_HIP, which is defined when compiling for ROCm/HIP
// (set via CMakeLists.txt's add_compile_definitions(__HIP_PLATFORM_AMD__)).

#pragma once

// Platform detection

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define BNB_HIP 1
#else
#define BNB_HIP 0
#endif

// Runtime and FP16/BF16 headers

#if BNB_HIP

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

#else // CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#endif

// Stream and error types

#if BNB_HIP

using bnb_stream_t = hipStream_t;
using bnb_error_t = hipError_t;

#define BNB_SUCCESS hipSuccess
#define BNB_PEEK_LAST_ERROR() hipPeekAtLastError()
#define BNB_GET_ERROR_STRING(e) hipGetErrorString(e)
#define BNB_DEVICE_MALLOC(p, s) hipMalloc(p, s)
#define BNB_DEVICE_FREE(p) hipFree(p)
#define BNB_DEVICE_MEMSET(p, v, s) hipMemset(p, v, s)

#else // CUDA

using bnb_stream_t = cudaStream_t;
using bnb_error_t = cudaError_t;

#define BNB_SUCCESS cudaSuccess
#define BNB_PEEK_LAST_ERROR() cudaPeekAtLastError()
#define BNB_GET_ERROR_STRING(e) cudaGetErrorString(e)
#define BNB_DEVICE_MALLOC(p, s) cudaMalloc(p, s)
#define BNB_DEVICE_FREE(p) cudaFree(p)
#define BNB_DEVICE_MEMSET(p, v, s) cudaMemset(p, v, s)

#endif

// Error checking

#define BNB_CHECK_RETURN(value)                                                                                        \
    {                                                                                                                  \
        bnb_error_t _bnb_stat = value;                                                                                 \
        if (_bnb_stat != BNB_SUCCESS) {                                                                                \
            fprintf(stderr, "Error %s at line %d in file %s\n", BNB_GET_ERROR_STRING(_bnb_stat), __LINE__, __FILE__);  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

// Keep backward compat for existing code during migration
#define CUDA_CHECK_RETURN(value) BNB_CHECK_RETURN(value)

// Warp synchronization
//
// HIP warps are always in lockstep (no independent thread scheduling),
// so __syncwarp() is a no-op. CUDA needs it for warp convergence.

#if BNB_HIP
#define __syncwarp()                                                                                                   \
    do {                                                                                                               \
    } while (0)
#endif

// BFloat16 type alias

#if BNB_HIP
using bnb_bfloat16 = hip_bfloat16;
#else
using bnb_bfloat16 = __nv_bfloat16;
#endif

// Data type enum aliases for BLAS/Sparse libraries

#if BNB_HIP

#define BNB_R_16F HIP_R_16F
#define BNB_R_32F HIP_R_32F
#define BNB_R_8I HIP_R_8I
#define BNB_R_32I HIP_R_32I

#else // CUDA

#define BNB_R_16F CUDA_R_16F
#define BNB_R_32F CUDA_R_32F
#define BNB_R_8I CUDA_R_8I
#define BNB_R_32I CUDA_R_32I

#endif

// BLAS Lt types and functions

#if BNB_HIP

#ifndef NO_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

using bnb_blasLt_handle_t = hipblasLtHandle_t;
using bnb_blasLt_matmul_desc_t = hipblasLtMatmulDesc_t;
using bnb_blasLt_layout_t = hipblasLtMatrixLayout_t;
using bnb_blasLt_preference_t = hipblasLtMatmulPreference_t;

#define BNB_BLASLT_OP_T HIPBLAS_OP_T
#define BNB_BLASLT_COMPUTE_32I HIPBLAS_COMPUTE_32I

#define bnb_blasLtCreate hipblasLtCreate
#define bnb_blasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define bnb_blasLtMatmulDescSetAttr hipblasLtMatmulDescSetAttribute
#define bnb_blasLtLayoutCreate hipblasLtMatrixLayoutCreate
#define bnb_blasLtLayoutDestroy hipblasLtMatrixLayoutDestroy
#define bnb_blasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define bnb_blasLtMatmul hipblasLtMatmul
#define bnb_blasLtPrefCreate hipblasLtMatmulPreferenceCreate
#define bnb_blasLtPrefSetAttr hipblasLtMatmulPreferenceSetAttribute
#define bnb_blasLtAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic

#define BNB_BLASLT_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define BNB_BLASLT_DESC_POINTER_MODE HIPBLASLT_MATMUL_DESC_POINTER_MODE
#define BNB_BLASLT_PREF_MAX_WORKSPACE HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define BNB_BLASLT_PTR_MODE_ALPHA_VEC HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST

using bnb_blasLt_heuristic_t = hipblasLtMatmulHeuristicResult_t;
using bnb_blas_status_t = hipblasStatus_t;
#define BNB_BLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#else // CUDA

#include <cublasLt.h>
#include <cublas_v2.h>

using bnb_blasLt_handle_t = cublasLtHandle_t;
using bnb_blasLt_matmul_desc_t = cublasLtMatmulDesc_t;
using bnb_blasLt_layout_t = cublasLtMatrixLayout_t;

#define BNB_BLASLT_OP_T CUBLAS_OP_T
#define BNB_BLASLT_COMPUTE_32I CUBLAS_COMPUTE_32I

#define bnb_blasLtCreate cublasLtCreate
#define bnb_blasLtMatmulDescCreate cublasLtMatmulDescCreate
#define bnb_blasLtMatmulDescSetAttr cublasLtMatmulDescSetAttribute
#define bnb_blasLtLayoutCreate cublasLtMatrixLayoutCreate
#define bnb_blasLtLayoutDestroy cublasLtMatrixLayoutDestroy
#define bnb_blasLtMatmulDescDestroy cublasLtMatmulDescDestroy
#define bnb_blasLtMatmul cublasLtMatmul

#define BNB_BLASLT_DESC_TRANSA CUBLASLT_MATMUL_DESC_TRANSA
#define BNB_BLASLT_DESC_POINTER_MODE CUBLASLT_MATMUL_DESC_POINTER_MODE
#define BNB_BLASLT_PTR_MODE_ALPHA_VEC CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO

using bnb_blas_status_t = cublasStatus_t;
#define BNB_BLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#endif

// Sparse library types

#if BNB_HIP

#include <hipsparse/hipsparse.h>

using bnb_sparse_handle_t = hipsparseHandle_t;
using bnb_sparseSpMatDescr_t = hipsparseSpMatDescr_t;
using bnb_sparseDnMatDescr_t = hipsparseDnMatDescr_t;

#define bnb_sparseCreate hipsparseCreate
#define bnb_sparseCreateCoo hipsparseCreateCoo
#define bnb_sparseCreateDnMat hipsparseCreateDnMat
#define bnb_sparseSpMM_bufSize hipsparseSpMM_bufferSize
#define bnb_sparseSpMM hipsparseSpMM
#define bnb_sparseDestroySpMat hipsparseDestroySpMat
#define bnb_sparseDestroyDnMat hipsparseDestroyDnMat

#define BNB_SPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define BNB_SPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define BNB_SPARSE_ORDER_ROW HIPSPARSE_ORDER_ROW
#define BNB_SPARSE_OP_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define BNB_SPARSE_OP_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define BNB_SPARSE_SPMM_ALG_DEFAULT HIPSPARSE_SPMM_ALG_DEFAULT

#define CHECK_SPARSE(value)                                                                                            \
    {                                                                                                                  \
        hipsparseStatus_t _stat = value;                                                                               \
        if (_stat != HIPSPARSE_STATUS_SUCCESS) {                                                                       \
            fprintf(stderr, "Error %s at line %d in file %s\n", hipsparseGetErrorString(_stat), __LINE__, __FILE__);   \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#else // CUDA

#include <cusparse.h>

using bnb_sparse_handle_t = cusparseHandle_t;
using bnb_sparseSpMatDescr_t = cusparseSpMatDescr_t;
using bnb_sparseDnMatDescr_t = cusparseDnMatDescr_t;

#define bnb_sparseCreate cusparseCreate
#define bnb_sparseCreateCoo cusparseCreateCoo
#define bnb_sparseCreateDnMat cusparseCreateDnMat
#define bnb_sparseSpMM_bufSize cusparseSpMM_bufferSize
#define bnb_sparseSpMM cusparseSpMM
#define bnb_sparseDestroySpMat cusparseDestroySpMat
#define bnb_sparseDestroyDnMat cusparseDestroyDnMat

#define BNB_SPARSE_INDEX_32I CUSPARSE_INDEX_32I
#define BNB_SPARSE_INDEX_BASE_ZERO CUSPARSE_INDEX_BASE_ZERO
#define BNB_SPARSE_ORDER_ROW CUSPARSE_ORDER_ROW
#define BNB_SPARSE_OP_NON_TRANSPOSE CUSPARSE_OPERATION_NON_TRANSPOSE
#define BNB_SPARSE_OP_TRANSPOSE CUSPARSE_OPERATION_TRANSPOSE
#define BNB_SPARSE_SPMM_ALG_DEFAULT CUSPARSE_SPMM_ALG_DEFAULT

#define CHECK_SPARSE(value)                                                                                            \
    {                                                                                                                  \
        cusparseStatus_t _stat = value;                                                                                \
        if (_stat != CUSPARSE_STATUS_SUCCESS) {                                                                        \
            fprintf(stderr, "Error %s at line %d in file %s\n", cusparseGetErrorString(_stat), __LINE__, __FILE__);    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#endif
