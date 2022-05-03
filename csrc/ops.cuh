
#ifndef ops_H
#define ops_H

#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <vector>
#include <functional>

using std::cout;
using std::endl;

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#define CHECK_CUSPARSE(value) {                      \
  cusparseStatus_t _m_cudaStat = value;                    \
  if (_m_cudaStat != CUSPARSE_STATUS_SUCCESS) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cusparseGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }


#define THREADS_PER_BLOCKS (512)


inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}


template <typename InType, typename OutType = InType, typename ComputeType = OutType>
struct TestBench {

    using SampleRunner = std::function<void()>;
    TestBench(int m, int n, int k, ComputeType alpha = 0.0f, ComputeType beta = 0.0f, size_t workspaceSize = 1024 * 1024 * 4, int N = 1) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(m * k * N), Bhost(n * k * N),
        Chost(m * n * N), biasHost(m * N) {
        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        fillData();
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    void fillData() {
        for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        for (int i = 0; i < m * N; i++) biasHost[i] = InType(i + 1);
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpy(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
				cout << "copy" << endl;
    }

    void copyDataFromDevice() {
        checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Chost, biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Cdev, *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
};

typedef enum Operations_t
{
	ksmul = 0,
} Operations_t;

typedef enum Optimizer_t
{
	ADAM = 0,
	MOMENTUM = 1,
  RMSPROP = 2,
  LARS = 3,
} Optimizer_t;

typedef enum Transform_t
{
	ROW = 0,
	COL = 1,
  COL32 = 2,
  COL_TURING = 3,
  COL_AMPERE = 4,
} Transform_t;

class Context
{
    public:
				cublasHandle_t m_handle;

				Context()
				{
					cublasHandle_t handle;
					cublasCreate_v2(&handle);
					m_handle = handle;
				}

};

class ContextLt
{
    public:
				cublasLtHandle_t m_handle;

				ContextLt()
				{
					cublasLtHandle_t handle;
					cublasLtCreate(&handle);
					m_handle = handle;
				}

};

class ContextCusparse
{
    public:
				cusparseHandle_t m_handle;

				ContextCusparse()
				{
					cusparseHandle_t handle;
					cusparseCreate(&handle);
					m_handle = handle;
				}

};


template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n);

void quantize(float *code, float *A, unsigned char *out, int n);
void dequantize(float *code, unsigned char *A, float *out, int n);
template <typename T, int STOCHASTIC> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float* rand, int rand_offset, const int n);
template<typename T> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int block_size, const int n);

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p, 
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2, float eps, float weight_decay,
                int step, float lr, const float gnorm_scale, int n);

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g, unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr, 
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n);

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr, 
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, int n);

template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n);

//void igemmLt(Context *context, bool transposeA, bool transposeB, int m, int n, int k, const void *A, const void *B, void *C, int lda, int ldb, int ldc);
void gemmex(Context * context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc);
void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc, 
                    long long int strideA, long long int strideB, long long int strideC, int batchCount);


template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> void igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2);
void cutlass_igemm(bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc);
void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, int numRows, int numCols);
void getColRowStats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols);
void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed,
                       int *rowidx, int *colidx, half *val, int *nnz_block_ptr, float threshold, int rows, int cols);

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols);

void spmm_coo(cusparseHandle_t handle, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B);

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);


#endif







