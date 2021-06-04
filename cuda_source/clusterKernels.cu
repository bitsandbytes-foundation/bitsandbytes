#include <clusterKernels.cuh>
//#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_discontinuity.cuh>

template __global__ void kElementWise<ksmul>(const float *A, const float *B, float *out, const float scalar, int size);
template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  //this switch operation will be removed by the compiler upon instantiation of the template
       switch(operation)
	   {
         case ksmul: out[i] = A[i] * scalar; break;
	   }
  }
}
