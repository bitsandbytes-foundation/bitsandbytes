#include "common.cuh"
#include "ops.cuh" // For CUDA_CHECK_RETURN, checkCublasStatus
#include <cub/cub.cuh>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <float.h> // For FLT_MIN/FLT_MAX

#define MM_DEQUANT_CONST 6.200012e-05f // 1.0f/(127.0f*127.0f)

// from kernels.cu
// TODO move somewhere like common.cuh or cub_utils.cuh etc
#if CCCL_VERSION >= 2008002
#include <cuda/std/functional>
#define CUB_REDUCTIONOP_MAX                                                                                            \
    cuda::maximum<> {}
#else
#define CUB_REDUCTIONOP_MAX cub::Max()
#endif

// Inputs:
//  A [rows, cols]
// Outputs:
//  rowStats [rows]
//  out [rows, cols]
template <typename T, int THREADS, int SPARSE_DECOMP>
__launch_bounds__(1024, BNB_MAX_THREADS_PER_SM / 1024) __global__
    void kInt8VectorQuant(T* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols) {

    using BlockReduceT = cub::BlockReduce<T, THREADS>;

    // One block per row.
    // Threads load column values in a striped arrangement.
    // e.g. t0 reads row[0], row[0+nthreads], ..
    // and  t1 reads row[1], row[1+nthreads], ..
    // Each thread will determine its local absmax.
    // We then do a blockwise reduction to determine the row's absmax.

    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ T smem_row_absmax;

    const int row_id = blockIdx.x;
    const T* row_data = A + (row_id * cols);

    // Threads will read the row values in a striped access pattern and find a local absmax.
    T row_local_absmax = -FLT_MIN;
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        const T absval = fabsf(__ldcs(&(row_data[i])));

        // For sparse decomposition, values outside of the threshold are not to be
        // included when calculating the row's absmax.
        if constexpr (SPARSE_DECOMP) {
            row_local_absmax = fmaxf(row_local_absmax, absval < T(threshold) ? absval : row_local_absmax);
        } else {
            row_local_absmax = fmaxf(row_local_absmax, absval);
        }
    }

    // Reduce thread-local absmax across the block.
    const T row_absmax = BlockReduceT(temp_storage).Reduce(row_local_absmax, CUB_REDUCTIONOP_MAX, cols);
    if (threadIdx.x == 0) {
        // Save our block's absmax to shared memory for the quantization step.
        rowStats[row_id] = smem_row_absmax = row_absmax;
    }
    __syncthreads();

    // Quantize row-wise.
    const float scale = __fdividef(127.0f, smem_row_absmax);
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        float val = row_data[i];

        if constexpr (SPARSE_DECOMP) {
            // For sparse decomposition, we do not want to quantize the outliers.
            // Instead they're zeroed out.
            out[row_id * cols + i] = fabs(val) < threshold ? __float2int_rn(val * scale) : 0;
        } else {
            out[row_id * cols + i] = __float2int_rn(val * scale);
        }
    }
}

template <int ITEMS_PER_THREAD, int THREADS>
__global__ void kdequant_mm_int32_fp16(
    int* __restrict__ const A, float* __restrict__ const rowStats, float* __restrict__ const colStats, half* out,
    half* __restrict__ const bias, const int numRows, const int numCols, const int n
) {
    const int n_out = numRows * numCols;

    int block_offset = blockIdx.x * THREADS * ITEMS_PER_THREAD;
    int thread_offset = threadIdx.x * ITEMS_PER_THREAD;

    int local_values[ITEMS_PER_THREAD];
    half local_output[ITEMS_PER_THREAD];

    float local_rowStats[ITEMS_PER_THREAD];
    float local_colStats[ITEMS_PER_THREAD];
    float local_biasValue[ITEMS_PER_THREAD];

    typedef cub::BlockLoad<int, THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE> LoadInt32;
    __shared__ typename LoadInt32::TempStorage loadint32;

    int row_idx, col_idx;

#pragma unroll ITEMS_PER_THREAD
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {

        row_idx = (block_offset + thread_offset + j) / numCols;
        col_idx = (block_offset + thread_offset + j) % numCols;

        local_colStats[j] = col_idx >= numCols ? 0.0f : __ldg(&colStats[col_idx]);
        local_rowStats[j] = row_idx >= numRows ? 0.0f : __ldg(&rowStats[row_idx]);
        local_biasValue[j] = ((bias == nullptr) || col_idx >= numCols) ? 0.0f : __half2float(bias[col_idx]);
    }

    // Each block loads THREADS * ITEMS_PER_THREAD values from A
    int valid_items =
        block_offset + THREADS * ITEMS_PER_THREAD < n_out ? THREADS * ITEMS_PER_THREAD : n_out - block_offset;
    LoadInt32(loadint32).Load(&(A[block_offset]), local_values, valid_items, 0);

#pragma unroll ITEMS_PER_THREAD
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        local_output[j] = __float2half(
            fmaf(local_values[j] * local_rowStats[j] * local_colStats[j], MM_DEQUANT_CONST, local_biasValue[j])
        );
    }

#pragma unroll ITEMS_PER_THREAD
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int outIdx = block_offset + thread_offset + j;
        if (outIdx < n_out) {
            out[outIdx] = local_output[j];
        }
    }
}

// launch/host code
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

// Launch templates
// TODO: we don't really need the 8bit accumulation path, can simplify
// TODO create C API here?

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
