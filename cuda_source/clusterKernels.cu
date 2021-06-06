#include <clusterKernels.cuh>
//#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_store.cuh>

#define HLF_MAX 65504

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


/**
 * @brief Quantizes x with the quantization map smem_code.
 *
 * @return returns quantized number.
 */
__device__ unsigned char quantize(float* smem_code, float x)
{
    unsigned char pivot = 127;
    unsigned char upper_pivot = 255;
    unsigned char lower_pivot = 0;

    // i>>=1 = {64, 32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > smem_code[pivot])
        {
            lower_pivot = pivot;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            pivot-=i;
        }
    }

    if(pivot == upper_pivot)
    {
        if(fabsf(smem_code[pivot]-x) <= fabsf(smem_code[lower_pivot]-x))
            return pivot;
        else
            return lower_pivot;
    }
    else if(pivot == lower_pivot)
    {
        if(fabsf(smem_code[upper_pivot]-x) <= fabsf(smem_code[lower_pivot]-x))
            return upper_pivot;
        else
            return lower_pivot;
    }
    else
    {
        if(fabsf(smem_code[pivot]-x) <= fabsf(smem_code[upper_pivot]-x))
            if(fabsf(smem_code[pivot]-x) <= fabsf(smem_code[lower_pivot]-x))
                return pivot;
            else
                return lower_pivot;
        else
            if(fabsf(smem_code[upper_pivot]-x) <= fabsf(smem_code[lower_pivot]-x))
                return upper_pivot;
            else
                return lower_pivot;
    }
}

#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

template __global__ void kEstimateQuantiles(float *__restrict__ const A, float *code, const float offset, const float max_val, const int n);
template __global__ void kEstimateQuantiles(half *__restrict__ const A, float *code, const float offset, const half max_val, const int n);
template<typename T>
__launch_bounds__(TH, 1)
__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n)
{
  const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  int valid_items = (blockIdx.x+1 == gridDim.x) ? n - (blockIdx.x*NUM_BLOCK) : NUM_BLOCK;
  const int base_idx = (blockIdx.x * NUM_BLOCK);
  const float reciprocal_num_blocks = 1.0f/(n < 4096 ? 1.0f : (n/NUM_BLOCK));

  T vals[NUM];

  typedef cub::BlockRadixSort<T, TH, NUM, cub::NullType, 4, true, cub::BLOCK_SCAN_RAKING> BlockRadixSort;
  typedef cub::BlockLoad<T, TH, NUM, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;

  __shared__ union {
      typename LoadFloat::TempStorage loadf;
      typename BlockRadixSort::TempStorage sort;
      int smem_qidx[NUM_BLOCK];
  } temp_storage;

  if(threadIdx.x < 256 && blockIdx.x == 0)
    code[threadIdx.x] = 0.0f;

  __syncthreads();

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_BLOCK)
  {
      valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

      // do not process half-blocks
      if(valid_items < NUM_BLOCK && n > NUM_BLOCK){ continue; }

      #pragma unroll 4
      for(int j = 0; j < NUM; j++)
          vals[j] = max_val;

      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(A[i]), vals, valid_items);

      #pragma unroll 4
      for(int j = 0; j < NUM; j++)
          vals[j] *= reciprocal_num_blocks;


      __syncthreads();
      // sort into striped pattern to mitigate bank conflicts
      // striped pattern index for thread 0 [0, 1024, 2048, 3096]
      // striped pattern index for thread 1 [1, 1025, 2049, 3097]
      BlockRadixSort(temp_storage.sort).SortBlockedToStriped(vals);

      __syncthreads();
      for(int j = threadIdx.x; j < NUM_BLOCK; j+=blockDim.x)
          temp_storage.smem_qidx[j] = -1;

      if(threadIdx.x < 256)
      {
          float q_interval = (1.0f-(2.0f*offset))/255.0f;
          int local_idx = round(((offset+(threadIdx.x*q_interval))*(valid_items-1)));
          temp_storage.smem_qidx[local_idx] = threadIdx.x;
      }

      __syncthreads();

      for(int i = threadIdx.x; i < NUM_BLOCK; i+=blockDim.x)
      {
          if(temp_storage.smem_qidx[i] != -1)
              atomicAdd(&code[temp_storage.smem_qidx[i]], vals[i/TH]);
      }
  }
}


__launch_bounds__(TH, 4)
__global__ void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n)
{
  const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  int valid_items = (blockIdx.x+1 == gridDim.x) ? n - (blockIdx.x*NUM_BLOCK) : NUM_BLOCK;
  const int base_idx = (blockIdx.x * NUM_BLOCK);

  float vals[NUM];
  unsigned char qvals[NUM];

  typedef cub::BlockLoad<float, TH, NUM, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  typedef cub::BlockStore<unsigned char, TH, NUM, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;
  __shared__ float smem_code[256];

  if(threadIdx.x < 256)
    smem_code[threadIdx.x] = code[threadIdx.x];

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_BLOCK)
  {
      valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

      __syncthreads();
      LoadFloat(loadf).Load(&(A[i]), vals, valid_items);

     #pragma unroll 4
     for(int j = 0; j < NUM; j++)
        qvals[j] = quantize(smem_code, vals[j]);

      __syncthreads();
      StoreChar(storec).Store(&(out[i]), qvals, valid_items);
  }
}

__global__ void kDequantize(float *code, unsigned char *A, float *out, const int n)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ float smem_code[256];
	if(threadIdx.x < 256)
	{
		smem_code[threadIdx.x] = code[threadIdx.x];
	}

	__syncthreads();

	for (int i = idx;i < n; i += numThreads)
	{
		out[i] = smem_code[A[i]];
	}
}



#define NUM_PER_THREAD 4

template __global__ void kOptimizer_32bit_2State<float, adam>(float* g, float* p, float* state1, float* state2,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const int n);
template<typename T, int OPTIMIZER>
__launch_bounds__(TH, 1)
__global__ void kOptimizer_32bit_2State(T* g, T* p, 
                float* state1, float* state2,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const int n)
{

  const int n_full = ((TH*NUM_PER_THREAD)*(n/(TH*NUM_PER_THREAD))) + (n % (TH*NUM_PER_THREAD) == 0 ? 0 : (TH*NUM_PER_THREAD));
  const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
  int valid_items = 0;

  T g_vals[NUM_PER_THREAD];
  T p_vals[NUM_PER_THREAD];

  float s1_vals[NUM_PER_THREAD];
  float s2_vals[NUM_PER_THREAD];

  const float correction1 = 1.0f - powf(beta1, step);
  const float correction2 = sqrtf(1.0f - powf(beta2, step));
  const float step_size = -lr*correction2/correction1;

  typedef cub::BlockLoad<T, TH, NUM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
  typedef cub::BlockStore<T, TH, NUM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> Store;

  typedef cub::BlockLoad<float, TH, NUM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  typedef cub::BlockStore<float, TH, NUM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreFloat;

  __shared__ union {
      typename Load::TempStorage load;
      typename Store::TempStorage store;
      typename LoadFloat::TempStorage loadf;
      typename StoreFloat::TempStorage storef;
  } temp_storage;

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*TH*NUM_PER_THREAD)
  {
      valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : i + (TH*NUM_PER_THREAD) - n;

      __syncthreads();
      Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items);
      __syncthreads();
      Load(temp_storage.load).Load(&(p[i]), p_vals, valid_items);

      if((i + (threadIdx.x*NUM_PER_THREAD) + NUM_PER_THREAD) <= n)
      {
        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
        {
            switch(OPTIMIZER)
            {
                case adam: 
                    s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
                    s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
                    p_vals[j] = ((float)p_vals[j]) + (step_size*(s1_vals[j]/(sqrtf(s2_vals[j])+(eps*correction2))));
                    break;
                case momentum: 
                    break;
            }
        }
      }

      __syncthreads();
      Store(temp_storage.store).Store(&(p[i]), p_vals, valid_items);
      __syncthreads();
      StoreFloat(temp_storage.storef).Store(&(state1[i]), s1_vals, valid_items);
      __syncthreads();
      StoreFloat(temp_storage.storef).Store(&(state2[i]), s2_vals, valid_items);
        __syncthreads();
  }
}
