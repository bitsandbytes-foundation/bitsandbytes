#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <hdf5.h>
#include <cuda_profiler_api.h>
#include <cub/device/device_scan.cuh>

using std::cout;
using std::endl;

template void elementWise<ksmul>(float *A, float *out, int n, float scalar);
template <int action> void elementWise(float *A, float *out, int n, float scalar)
{
  kElementWise<action><<<n/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A, NULL, out,scalar, n);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}
