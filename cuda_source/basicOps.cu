#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <hdf5.h>
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
