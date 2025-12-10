#include <cublasLt.h>
#include <cuda_fp16.h>

// was in ops.cuh
void dequant_mm_int32_fp16(
    int* A, float* rowStats, float* colStats, half* out, half* bias, int numRows, int numCols, cudaStream_t stream
);
void int8VectorQuant(
    half* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols, cudaStream_t stream
);

template <int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(
    cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t* A, const int8_t* B, void* C, float* row_scale,
    int lda, int ldb, int ldc, cudaStream_t stream
);
