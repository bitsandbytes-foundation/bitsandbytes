#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <cuda_profiler_api.h>
#include <cub/device/device_scan.cuh>
#include <limits>

using std::cout;
using std::endl;

template void elementWise<ksmul>(float *A, float *out, int n, float scalar);
template <int action> void elementWise(float *A, float *out, int n, float scalar)
{
  kElementWise<action><<<n/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A, NULL, out,scalar, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);
template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
  kEstimateQuantiles<T><<<blocks, 1024>>>(A, code, offset, std::numeric_limits<T>::max(), n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int blocks = n/1024;
  blocks = n % 1024 == 0 ? blocks : blocks + 1;
  kQuantize<<<blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int blocks = n/1024;
  blocks = n % 1024 == 0 ? blocks : blocks + 1;
  kDequantize<<<blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void optimizer_32bit_2State<half, adam>(half* g, half* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const int n);
template void optimizer_32bit_2State<float, adam>(float* g, float* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const int n);
template<typename T, int OPTIMIZER> void optimizer_32bit_2State(T* g, T* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const bool is_sparse, const int n)
{
  int blocks = n/4096;
  blocks = n % 4096 == 0 ? blocks : blocks + 1;
  kOptimizer_32bit_2State<T, adam><<<blocks, 1024>>>(g, p, state1, state2, beta1, beta2, eps, weight_decay, step, lr, is_sparse, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}
