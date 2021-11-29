// Copyright (c) Facebook, Inc. and its affiliates. 
//   
// This source code is licensed under the MIT license found in the 
// LICENSE file in the root directory of this source tree.

#include <kernels.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <math_constants.h>

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

// source: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ float atomicMax(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        reinterpret_cast<int*>(address), assumed,
        __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ float atomicMin(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        reinterpret_cast<int*>(address), assumed,
        __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template <int STOCHASTIC>
__device__ unsigned char dQuantize(float* smem_code, const float rand, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(!STOCHASTIC)
    {
      if(x > val)
      {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
        {
          return upper_pivot;
        }
        else
          return pivot;
      }
      else
      {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
          return lower_pivot;
        else
          return pivot;
      }
    }
    else
    {
      if(x > val)
      {
        float dist_to_upper = fabsf(upper-x);
        float dist_full = upper-val;
        if(rand >= dist_to_upper/dist_full) return upper_pivot;
        else return pivot;
      }
      else
      {
        float dist_to_lower = fabsf(lower-x);
        float dist_full = val-lower;
        if(rand >= dist_to_lower/dist_full) return lower_pivot;
        else return pivot;
      }
    }
}

template <int SIGNED>
__device__ __forceinline__ unsigned char quantize_2D(float *__restrict__ quadrants, float *__restrict__ const smem_code, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = SIGNED ? -1.0f : 0.0f;
    float upper = 1.0f;
    float midpoint;
    float val = quadrants[1];
    int local_pivot = 1;
    int offset = 1;

    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
            //val = i == 64 ? quadrants[2] : smem_code[pivot];
            local_pivot += offset;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
            //val = i == 64 ? quadrants[0] : smem_code[pivot];
            local_pivot -= offset;
        }
        val = i >= 64 ? quadrants[local_pivot] : smem_code[pivot];
        offset -= 1;
    }

    if(x > val)
    {
      midpoint = (upper+val)*0.5f;
      if(x > midpoint)
        return upper_pivot;
      else
        return pivot;
    }
    else
    {
      midpoint = (lower+val)*0.5f;
      if(x < midpoint)
        return lower_pivot;
      else
        return pivot;
    }
}

template <int SIGNED>
__device__ __forceinline__ unsigned char quantize_quadrant(int QUADRANT, float *__restrict__ const smem_code, float x, float lower, float midpoint, float upper)
{
    int lower_pivot = QUADRANT*16-1 - 0;
    int pivot = QUADRANT*16-1 + 16;
    int upper_pivot = QUADRANT*16-1 + 31;

    float val = midpoint;

    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 16; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(x > val)
    {
      midpoint = (upper+val)*0.5f;
      if(x > midpoint)
        return upper_pivot;
      else
        return pivot;
    }
    else
    {
      midpoint = (lower+val)*0.5f;
      if(x < midpoint)
        return lower_pivot;
      else
        return pivot;
    }
}

__global__ void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n)
{
  const int tid = threadIdx.x + (blockDim.x*blockIdx.x);
  const int numThreads = blockDim.x*gridDim.x;

  for(int i = tid; i < n; i+=numThreads)
  {
      int idx = (index1[i]*maxidx1) + index2[i];
      atomicAdd(&histogram[idx], src[i]);
  }
}

template<typename T, int BLOCK_SIZE, int NUM_MAX>
__global__ void kCompressMax(T * __restrict__ const A, T* out, unsigned char* out_idx, const int n)
{
  typedef cub::WarpReduce<T> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  typedef cub::BlockLoad<T, BLOCK_SIZE/8 , 8, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
  __shared__ typename LoadT::TempStorage loadt;

  const int warp_idx = threadIdx.x/32;
  const int valid_items = n - (blockIdx.x*BLOCK_SIZE) > BLOCK_SIZE ? BLOCK_SIZE : n - (blockIdx.x*BLOCK_SIZE);

  //  BLOCK_SIZE/32 == number of warps
  __shared__ int smem_max_indices[8*BLOCK_SIZE/32];
  __shared__ float smem_max_values[8*BLOCK_SIZE/32];

  T values[8];
  T max1 = -64000.0f;
  T max2 = -64000.0f;
  int max_idx1 = -1;
  int max_idx2 = -1;
  int sign1 = -1;
  int sign2 = -1;

  // 1. load 8 values per thread
  // 2. compute 2-max in registers (64 max per warp)
  // 3. do warp reduction + broadcast back
  // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
  // 5. Repeat (3) 8 times for top 8 values in 256
  // 6. store with byte index

  LoadT(loadt).Load(&(A[(blockIdx.x*BLOCK_SIZE)]), values, valid_items, (T)0.0f);
  #pragma unroll 8
  for(int i = 0; i < 8; i++)
  {
    T absval = fabsf(values[i]);
    if(absval > max1)
    {
      max1 = values[i];
      sign1 = signbit(values[i]);
      max_idx1 = 8*threadIdx.x + i;
    }
    else if(absval > max2)
    {
      max2 = values[i];
      sign2 = signbit(values[i]);
      max_idx2 = 8*threadIdx.x + i;
    }
  }

  float warp_max;
  for(int i = 0; i < 8; i++)
  {
    // 3. do warp reduction + broadcast back
    warp_max = WarpReduce(temp_storage).Reduce(max1, cub::Max());
    warp_max = cub::ShuffleIndex<32>(warp_max, 0, 0xffffffff);

    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    if(warp_max == max1)
    {
      smem_max_values[warp_idx*8 + i] = sign1 != 0 ? -max1 : max1;
      smem_max_indices[warp_idx*8 + i] = max_idx1;

      sign1 = sign2;
      max1 = max2;
      max_idx1 = max_idx2;

      max2 = -64000.0f;
    }
    __syncwarp();
  }

  if(threadIdx.x % 32 < 8)
  {
    // offset: 8 values per 256 input values
    // 
    int offset = BLOCK_SIZE*blockIdx.x*BLOCK_SIZE/32*8;
  }

}

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 4096

template<typename T>
__launch_bounds__(THREADS_ESTIMATE, 1)
__global__ void kEstimateQuantiles(T *__restrict__ const A, float *code, const float offset, const T max_val, const int n)
{
  const int n_full = (BLOCK_ESTIMATE*(n/BLOCK_ESTIMATE)) + (n % BLOCK_ESTIMATE == 0 ? 0 : BLOCK_ESTIMATE);
  int valid_items = (blockIdx.x+1 == gridDim.x) ? n - (blockIdx.x*BLOCK_ESTIMATE) : BLOCK_ESTIMATE;
  const int base_idx = (blockIdx.x * BLOCK_ESTIMATE);
  const float reciprocal_num_blocks = 1.0f/(n < 4096 ? 1.0f : (n/BLOCK_ESTIMATE));

  T vals[NUM_ESTIMATE];

  typedef cub::BlockRadixSort<T, THREADS_ESTIMATE, NUM_ESTIMATE, cub::NullType, 4, true, cub::BLOCK_SCAN_RAKING> BlockRadixSort;
  typedef cub::BlockLoad<T, THREADS_ESTIMATE, NUM_ESTIMATE, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;

  __shared__ union {
      typename LoadFloat::TempStorage loadf;
      typename BlockRadixSort::TempStorage sort;
      int smem_qidx[BLOCK_ESTIMATE];
  } temp_storage;

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_ESTIMATE)
  {
      valid_items = n - i > BLOCK_ESTIMATE ? BLOCK_ESTIMATE : n - i;

      // do not process half-blocks
      if(valid_items < BLOCK_ESTIMATE && n > BLOCK_ESTIMATE){ continue; }

      #pragma unroll 4
      for(int j = 0; j < NUM_ESTIMATE; j++)
          vals[j] = max_val;

      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(A[i]), vals, valid_items);

      #pragma unroll 4
      for(int j = 0; j < NUM_ESTIMATE; j++)
          vals[j] = ((float)vals[j]) * reciprocal_num_blocks;


      __syncthreads();
      // sort into striped pattern to mitigate bank conflicts
      // striped pattern index for thread 0 [0, 1024, 2048, 3096]
      // striped pattern index for thread 1 [1, 1025, 2049, 3097]
      BlockRadixSort(temp_storage.sort).SortBlockedToStriped(vals);

      __syncthreads();
      for(int j = threadIdx.x; j < BLOCK_ESTIMATE; j+=blockDim.x)
          temp_storage.smem_qidx[j] = -1;

      if(threadIdx.x < 256)
      {
          float q_interval = (1.0f-(2.0f*offset))/255.0f;
          int local_idx = round(((offset+(threadIdx.x*q_interval))*(valid_items-1)));
          temp_storage.smem_qidx[local_idx] = threadIdx.x;
      }

      __syncthreads();

      for(int i = threadIdx.x; i < BLOCK_ESTIMATE; i+=blockDim.x)
      {
          if(temp_storage.smem_qidx[i] != -1)
              atomicAdd(&code[temp_storage.smem_qidx[i]], vals[i/THREADS_ESTIMATE]);
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
  //const int lane_id = threadIdx.x % 2;

  typedef cub::BlockLoad<float, TH, NUM, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  typedef cub::BlockStore<unsigned char, TH, NUM, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;
  __shared__ float smem_code[256];
  //__shared__ float smem_code[2][257];

  if(threadIdx.x < 256)
  {
    smem_code[threadIdx.x] = code[threadIdx.x];
    //smem_code[0][threadIdx.x] = code[threadIdx.x];
    //smem_code[1][threadIdx.x] = smem_code[0][threadIdx.x];
  }


  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_BLOCK)
  {
      // number of values already processed in blocks +
      // number of values already processed in this block +
      // rand_offset % mod value
      valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

      __syncthreads();
      LoadFloat(loadf).Load(&(A[i]), vals, valid_items);


      #pragma unroll 4
      for(int j = 0; j < NUM; j++)
          qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

      __syncthreads();
      StoreChar(storec).Store(&(out[i]), qvals, valid_items);
  }
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC>
__launch_bounds__(TH, 4)
__global__ void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n)
{
  const int n_full = gridDim.x * BLOCK_SIZE;
  int valid_items = 0;
  const int base_idx = (blockIdx.x * BLOCK_SIZE);

  T vals[NUM];
  float rand_vals[NUM];
  unsigned char qvals[NUM];
  //float local_abs_max = -FLT_MAX;
  float local_abs_max = 0.0f;
  int local_rand_idx = 0;

  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
  typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;
  typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;

  __shared__ typename LoadT::TempStorage loadt;
  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;
  __shared__ typename BlockReduce::TempStorage reduce;
  __shared__ float smem_code[256];
  __shared__ float smem_absmax_value[1];

  if(threadIdx.x < 256)
    smem_code[threadIdx.x] = code[threadIdx.x];

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
    valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
    local_abs_max = -FLT_MAX;

    __syncthreads();
    LoadT(loadt).Load(&(A[i]), vals, valid_items, (T)0.0f);

    // 1. compute local max
    // 2. broadcast local max
    // 3. normalize inputs and quantize

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
       local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max(), valid_items);

    if(threadIdx.x == 0)
      smem_absmax_value[0] = local_abs_max;

    __syncthreads();

    if(threadIdx.x == 0)
      absmax[i/BLOCK_SIZE] = local_abs_max;
    else
      local_abs_max = smem_absmax_value[0];

    __syncwarp();

    local_abs_max = 1.0f/local_abs_max;

    if(STOCHASTIC)
    {
      local_rand_idx = ((blockIdx.x*NUM_BLOCK) + (threadIdx.x*NUM) + rand_offset) % (1024-4);
      LoadFloat(loadf).Load(&rand[local_rand_idx], rand_vals, BLOCK_SIZE, 0);
    }

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
    {
      if(!STOCHASTIC)
       qvals[j] = dQuantize<0>(smem_code, 0.0f, ((float)vals[j])*local_abs_max);
      else
       qvals[j] = dQuantize<1>(smem_code, rand_vals[j], ((float)vals[j])*local_abs_max);
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[i]), qvals, valid_items);
  }
}

template<typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH>
__global__ void kDequantizeBlockwise(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, T *out, const int n)
{

  const int n_full = gridDim.x * BLOCK_SIZE;
  int valid_items = 0;
  const int base_idx = (blockIdx.x * BLOCK_SIZE);

  T vals[NUM];
  unsigned char qvals[NUM];
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;
  __shared__ float smem_code[256];

  if(threadIdx.x < 256)
    smem_code[threadIdx.x] = code[threadIdx.x];

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
      valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
      local_abs_max = absmax[i/BLOCK_SIZE];

      __syncthreads();
      LoadChar(loadchar).Load(&(A[i]), qvals, valid_items, 128);

      #pragma unroll NUM_PER_TH
      for(int j = 0; j < NUM_PER_TH; j++)
        vals[j] = smem_code[qvals[j]]*local_abs_max;

      __syncthreads();
      StoreT(storet).Store(&(out[i]), vals, valid_items);
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



template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__launch_bounds__(BLOCK_SIZE/NUM_VALS, 1)
__global__ void kPreconditionOptimizer32bit2State(T* g, T* p, 
                float* state1, float* state2, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n)
{

  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  const int base_idx = (blockIdx.x * blockDim.x * NUM_VALS);
  int valid_items = 0;

  T g_vals[NUM_VALS];

  float s1_vals[NUM_VALS];
  float s2_vals[NUM_VALS];

  const float correction1 = 1.0f/(1.0f - powf(beta1, step));
  const float correction2 = 1.0f/(1.0f - powf(beta2, step));

  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
  typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_VALS> BlockReduce;

  __shared__ union {
      typename Load::TempStorage load;
      typename LoadFloat::TempStorage loadf;
      typename BlockReduce::TempStorage reduce;
  } temp_storage;

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
      valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

      __syncthreads();
      Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items, 0.0f);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
      {
          switch(OPTIMIZER)
          {
              case ADAM: 
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
                  s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
                  s1_vals[j] *= correction1;
                  s2_vals[j] *= correction2;
                  s1_vals[j] = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
                  s1_vals[j] *= s1_vals[j]; // update l2 norm (update*update)
                  break;
          }
      }

      # pragma unroll NUM_VALS-1
      for(unsigned int j = 1; j < NUM_VALS; j++)
          s1_vals[0] += s1_vals[j];

      __syncthreads();
      s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0]);

      if(threadIdx.x == 0)
        atomicAdd(&unorm[0], s1_vals[0]);

      __syncwarp();
  }
}



#define NUM_PER_THREAD 4

template<typename T, int OPTIMIZER>
__launch_bounds__(TH, 1)
__global__ void kOptimizer32bit2State(T* g, T* p, 
                float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n)
{

  const int n_full = ((TH*NUM_PER_THREAD)*(n/(TH*NUM_PER_THREAD))) + (n % (TH*NUM_PER_THREAD) == 0 ? 0 : (TH*NUM_PER_THREAD));
  const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
  int valid_items = 0;
  float update_scale = 0.0f;
  T g_vals[NUM_PER_THREAD];
  T p_vals[NUM_PER_THREAD];

  float s1_vals[NUM_PER_THREAD];
  float s2_vals[NUM_PER_THREAD];

  const float correction1 = 1.0f - powf(beta1, step);
  const float correction2 = sqrtf(1.0f - powf(beta2, step));
  const float step_size = -lr*correction2/correction1;

  if(max_unorm > 0.0f)
  {
    update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
    if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
    else{ update_scale = 1.0f; }
  }
  else{ update_scale = 1.0f; }

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
      valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

      __syncthreads();
      Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items);
      __syncthreads();
      Load(temp_storage.load).Load(&(p[i]), p_vals, valid_items);

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
          switch(OPTIMIZER)
          {
              case ADAM: 
									if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
									{
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
										s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
										p_vals[j] = ((float)p_vals[j]) + (update_scale*step_size*(s1_vals[j]/(sqrtf(s2_vals[j])+(eps*correction2))));

                    if(weight_decay > 0.0f)
                        p_vals[j] = ((float)p_vals[j])*(1.0f-(lr*weight_decay));
									}
                  break;
          }
      }

      __syncthreads();
      Store(temp_storage.store).Store(&(p[i]), p_vals, valid_items);
      __syncthreads();
      StoreFloat(temp_storage.storef).Store(&(state1[i]), s1_vals, valid_items);
      __syncthreads();
      StoreFloat(temp_storage.storef).Store(&(state2[i]), s2_vals, valid_items);
  }
}

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__launch_bounds__(BLOCK_SIZE/NUM_VALS, 1)
__global__ void kPreconditionOptimizer32bit1State(T* g, T* p, 
                float* state1, float *unorm,
                const float beta1, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n)
{

  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  const int base_idx = (blockIdx.x * blockDim.x * NUM_VALS);
  int valid_items = 0;

  T g_vals[NUM_VALS];

  float s1_vals[NUM_VALS];

  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
  typedef cub::BlockLoad<float, BLOCK_SIZE/NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_VALS> BlockReduce;

  __shared__ union {
      typename Load::TempStorage load;
      typename LoadFloat::TempStorage loadf;
      typename BlockReduce::TempStorage reduce;
  } temp_storage;

  for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
      valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

      __syncthreads();
      Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
      {
          switch(OPTIMIZER)
          {
              case MOMENTUM: 
                  if(step == 1)
                    s1_vals[j] = (float)g_vals[j]; // state update
                  else
                    s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]); // state update
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case RMSPROP: 
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j])); // state update
                  s1_vals[j] = __fdividef((float)g_vals[j],sqrtf(s1_vals[j])+eps); // update value
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case ADAGRAD: 
                  s1_vals[j] = s1_vals[j] + ((float)g_vals[j])*((float)g_vals[j]); // state update
                  s1_vals[j] = __fdividef((float)g_vals[j],sqrtf(s1_vals[j])+eps); // update value
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
          }
      }

      # pragma unroll
      for(unsigned int j = 1; j < NUM_VALS; j++)
        s1_vals[0] += s1_vals[j];

      __syncthreads();
      s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0], valid_items);

      if(threadIdx.x == 0)
        atomicAdd(&unorm[0], s1_vals[0]);

      __syncwarp();
  }
}

template<typename T, int OPTIMIZER>
__launch_bounds__(TH, 1)
__global__ void kOptimizer32bit1State(T *g, T *p, 
                float *state1, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n)
{

  const int n_full = ((TH*NUM_PER_THREAD)*(n/(TH*NUM_PER_THREAD))) + (n % (TH*NUM_PER_THREAD) == 0 ? 0 : (TH*NUM_PER_THREAD));
  const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
  int valid_items = 0;
  float update_scale = 0.0f;

  if(max_unorm > 0.0f)
  {
    update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
    if(update_scale > max_unorm*param_norm+eps){ update_scale = (max_unorm*param_norm+eps)/update_scale; }
    else{ update_scale = 1.0f; }
  }
  else{ update_scale = 1.0f; }

  T g_vals[NUM_PER_THREAD];
  T p_vals[NUM_PER_THREAD];

  float s1_vals[NUM_PER_THREAD];

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
      valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

      __syncthreads();
      Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items);
      __syncthreads();
      LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items);
      __syncthreads();
      Load(temp_storage.load).Load(&(p[i]), p_vals, valid_items);

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
        g_vals[j] = gnorm_scale*((float)g_vals[j]);
        if(weight_decay > 0.0f)
          g_vals[j] = (float)g_vals[j] + (((float)p_vals[j])*weight_decay);
      }

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
					if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
					{
						switch(OPTIMIZER)
						{
								case MOMENTUM: 
										if(step == 1)
											s1_vals[j] = (float)g_vals[j];
										else
											s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);

										p_vals[j] = ((float)p_vals[j]) + update_scale*(-lr*(s1_vals[j]));
										break;
								case RMSPROP: 
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j]));
										p_vals[j] = ((float)p_vals[j]) - update_scale*(lr*__fdividef((float)g_vals[j],sqrtf((float)s1_vals[j])+eps));
										break;
								case ADAGRAD: 
										s1_vals[j] = s1_vals[j] + ((float)g_vals[j])*((float)g_vals[j]);
										p_vals[j] = ((float)p_vals[j]) - lr*__fdividef((float)g_vals[j],sqrtf((float)s1_vals[j])+eps);
										break;
						}
					}
      }

      __syncthreads();
      Store(temp_storage.store).Store(&(p[i]), p_vals, valid_items);
      __syncthreads();
      StoreFloat(temp_storage.storef).Store(&(state1[i]), s1_vals, valid_items);
  }
}


#define NUM8BIT 16
#define NUM_THREADS 256
#define NUM_PER_BLOCK 4096

template<typename T, int OPTIMIZER>
__global__ void
__launch_bounds__(NUM_THREADS, 2)
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const float gnorm_scale, const int n)
{
    const int n_full = gridDim.x * NUM_PER_BLOCK;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items = n - (blockIdx.x*NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (blockIdx.x*NUM_PER_BLOCK);
    float g_val = 0.0f;
    float local_max_s1 = -FLT_MAX;
    float local_max_s2 = -FLT_MAX;
    float local_unorm = 0.0f;

    float s2_vals[NUM8BIT];
    float s1_vals[NUM8BIT];
    T g_vals[NUM8BIT];
    unsigned char m_c1[NUM8BIT];
    unsigned char r_c2[NUM8BIT];

    typedef cub::BlockLoad<T, NUM_THREADS, NUM8BIT, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, NUM_THREADS, NUM8BIT, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadUInt8;
    typedef cub::BlockReduce<float, NUM_THREADS> BlockReduce;


    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadUInt8::TempStorage loadc;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    __shared__ float smem_quantiles1[256];
    __shared__ float smem_quantiles2[256];

    if(threadIdx.x < 256)
    {
        smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];
        smem_quantiles2[threadIdx.x] = quantiles2[threadIdx.x];
    }

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += NUM_THREADS*gridDim.x*NUM8BIT)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state1[i]), m_c1, valid_items, 128);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state2[i]), r_c2, valid_items, 128);
        __syncthreads();

        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]]*max1[0]*beta1;
            s1_vals[j] += (1.0f-beta1)*g_val;
            local_max_s1 = fmaxf(local_max_s1, fabsf(s1_vals[j]));
        }

        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s2_vals[j] = smem_quantiles2[r_c2[j]]*max2[0]*beta2;
            s2_vals[j] += (1.0f-beta2)*g_val*g_val;
            local_max_s2 = fmaxf(local_max_s2, fabsf(s2_vals[j]));
        }

        if(unorm != NULL)
        {
          #pragma unroll 16
          for(int j = 0; j < NUM8BIT; j++)
          {
            float correction1 = __fdividef(1.0f, 1.0f - powf(beta1, step));
            float correction2 = __fdividef(1.0f, 1.0f - powf(beta2, step));
            s1_vals[j] *= correction1;
            s2_vals[j] *= correction2;
            float update_val = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
            local_unorm += update_val*update_val;
          }
        }
    }

    __syncthreads();
    local_max_s1 = BlockReduce(temp_storage.reduce).Reduce(local_max_s1, cub::Max(), valid_items);
    __syncthreads();
    local_max_s2 = BlockReduce(temp_storage.reduce).Reduce(local_max_s2, cub::Max(), valid_items);
    if(unorm != NULL)
    {
      __syncthreads();
      local_unorm = BlockReduce(temp_storage.reduce).Reduce(local_unorm, cub::Sum(), valid_items);
    }

    if(threadIdx.x == 0)
    {
        atomicMax(&new_max1[0], local_max_s1);
        atomicMax(&new_max2[0], local_max_s2);
        if(unorm != NULL){ atomicAdd(&unorm[0], local_unorm); }
    }
}

#define NUM_PER_THREAD2 4
#define NUM_THREADS2 1024
#define NUM_PER_BLOCK2 4096

template<typename T, int OPTIMIZER>
__global__ void
__launch_bounds__(NUM_THREADS2, 1)
kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, const int n)
{

    const int n_full = (blockDim.x * gridDim.x)*NUM_PER_THREAD2;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float s2_vals[NUM_PER_THREAD2];
    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2 = sqrtf(1.0f - powf(beta2, step));
    const float step_size = -lr*correction2/correction1;
    //const float step_size = -lr*correction2/correction1;
    float new_max_val1 = 1.0f/new_max1[0];
    float new_max_val2 = 1.0f/new_max2[0];
    float update_scale = 1.0f;

    if(max_unorm > 0.0f)
    {
      update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
      if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
      else{ update_scale = 1.0f; }
    }
    else{ update_scale = 1.0f; }

    unsigned char c1s[NUM_PER_THREAD2];
    unsigned char c2s[NUM_PER_THREAD2];
    T p_vals[NUM_PER_THREAD2];
    T g_vals[NUM_PER_THREAD2];
    typedef cub::BlockLoad<T, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[256];
    __shared__ float smem_quantiles2[256];

    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadChar::TempStorage loadc;
        typename StoreChar::TempStorage storec;
        typename StoreT::TempStorage storeh;
    } temp_storage;

    if(threadIdx.x < 512)
    {
        if(threadIdx.x < 256)
            smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];
        else
            smem_quantiles2[threadIdx.x-256] = quantiles2[threadIdx.x-256];
    }

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_THREADS2*NUM_PER_THREAD2)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state2[i]), c2s, valid_items, 0);
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items);

        if((i + (threadIdx.x*NUM_PER_THREAD2) + NUM_PER_THREAD2) > n){ continue; }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[c1s[j]];
            s1_vals[j] = s1_vals[j]*max1[0];

            s1_vals[j] = (s1_vals[j]*beta1) + (((1.0f-beta1)*g_val));

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j]*new_max_val1);

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(signbit(smem_quantiles1[c1s[j]]) != signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }

            s2_vals[j] = smem_quantiles2[c2s[j]];
            s2_vals[j] = s2_vals[j]*max2[0];
            s2_vals[j] = (s2_vals[j]*beta2) + (((1.0f-beta2)*g_val*g_val));
            c2s[j] = dQuantize<0>(smem_quantiles2, 0.0f, s2_vals[j]*new_max_val2);
        }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            p_vals[j] = (T)(((float)p_vals[j]) + ((update_scale*step_size*(s1_vals[j]/(sqrtf(s2_vals[j])+(correction2*eps))))));
            if(weight_decay > 0.0f)
                p_vals[j] = update_scale*((float)p_vals[j])*(1.0f-(lr*weight_decay));
        }

        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state2[i]), c2s, valid_items);
        __syncthreads();
    }
}


template<typename T, int OPTIMIZER>
__global__ void
__launch_bounds__(NUM_THREADS, 2)
kPreconditionOptimizerStatic8bit1State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, 
                float *unorm,
                const float beta1, 
                const float eps, const int step,
                float* __restrict__ const quantiles1, 
                float* max1, float* new_max1, 
                const float weight_decay,
                const float gnorm_scale, const int n)
{
    const int n_full = gridDim.x * NUM_PER_BLOCK;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items = n - (blockIdx.x*NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (blockIdx.x*NUM_PER_BLOCK);
    float g_val = 0.0f;
    float local_max_s1 = -FLT_MAX;
    float local_unorm = 0.0f;

    float s1_vals[NUM8BIT];
    T g_vals[NUM8BIT];
    unsigned char m_c1[NUM8BIT];

    typedef cub::BlockLoad<T, NUM_THREADS, NUM8BIT, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, NUM_THREADS, NUM8BIT, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadUInt8;
    typedef cub::BlockReduce<float, NUM_THREADS> BlockReduce;


    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadUInt8::TempStorage loadc;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    __shared__ float smem_quantiles1[256];

    if(threadIdx.x < 256)
      smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_THREADS*NUM8BIT)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state1[i]), m_c1, valid_items, 128);

        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]]*max1[0];
            switch(OPTIMIZER)
            {
                case MOMENTUM: 
                    if(step == 1)
                      s1_vals[j] = (float)g_vals[j];
                    else
                      s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);
                    if(unorm != NULL)
                      local_unorm += s1_vals[j]*s1_vals[j];
                    break;
              case RMSPROP: 
                    s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
                  break;
            }

            local_max_s1 = fmaxf(local_max_s1, fabsf(s1_vals[j]));
        }
    }

    __syncthreads();
    local_max_s1 = BlockReduce(temp_storage.reduce).Reduce(local_max_s1, cub::Max(), valid_items);
    if(threadIdx.x == 0){ atomicMax(&new_max1[0], local_max_s1); }
    if(unorm != NULL)
    {
      __syncthreads();
      local_unorm = BlockReduce(temp_storage.reduce).Reduce(local_unorm, cub::Sum(), valid_items);
      if(threadIdx.x == 0){ atomicAdd(&unorm[0], local_unorm); }
    }

}

template<typename T, int OPTIMIZER>
__global__ void
kOptimizerStatic8bit1State(T* p, T* const g, unsigned char* state1,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, 
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, 
                float* max1, float* new_max1, 
                float weight_decay,
                const float gnorm_scale, const int n)
{

    const int n_full = (blockDim.x * gridDim.x)*NUM_PER_THREAD2;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float new_max_val1 = 1.0f/new_max1[0];
    float update_scale = 1.0f;

    if(max_unorm > 0.0f)
    {
      update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
      if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
      else{ update_scale = 1.0f; }
    }
    else{ update_scale = 1.0f; }

    unsigned char c1s[NUM_PER_THREAD2];
    T p_vals[NUM_PER_THREAD2];
    T g_vals[NUM_PER_THREAD2];
    typedef cub::BlockLoad<T, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, NUM_THREADS2, NUM_PER_THREAD2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[256];

    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadChar::TempStorage loadc;
        typename StoreChar::TempStorage storec;
        typename StoreT::TempStorage storeh;
    } temp_storage;

    if(threadIdx.x < 256)
        smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*NUM_THREADS2*NUM_PER_THREAD2)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items);

        if((i + (threadIdx.x*NUM_PER_THREAD2) + NUM_PER_THREAD2) > n){ continue; }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            if(weight_decay > 0.0f)
              g_val += ((float)p_vals[j])*weight_decay;
            s1_vals[j] = smem_quantiles1[c1s[j]]*max1[0];

            switch(OPTIMIZER)
            {
                case MOMENTUM: 
                  if(step == 1)
                    s1_vals[j] = g_vals[j];
                  else
                    s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);

                  p_vals[j] = ((float)p_vals[j]) + (-lr*update_scale*(s1_vals[j]));
                  break;
              case RMSPROP: 
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
                  p_vals[j] = ((float)p_vals[j]) - (lr*__fdividef(g_val,sqrtf(s1_vals[j])+eps));
                  break;
            }

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j]*new_max_val1);

            // make sure state1 term has still the same sign after quantization
            if(signbit(smem_quantiles1[c1s[j]]) != signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
        __syncthreads();
    }
}


template<typename T, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPercentileClipping(T * __restrict__ g, float *gnorm_vec, int step, const int n)
{
  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  int valid_items = 0;

  typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_VALS> BlockReduce;
  typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;

  __shared__ typename BlockReduce::TempStorage reduce;

  __shared__ typename LoadT::TempStorage loadT;
  T vals[NUM_VALS];
  float local_sum = 0.0f;

  for (unsigned int i = (blockIdx.x * BLOCK_SIZE); i < n_full; i += gridDim.x*BLOCK_SIZE)
  {
      valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
      local_sum = 0.0f;

      __syncthreads();
      LoadT(loadT).Load(&(g[i]), vals, valid_items, (T)0.0f);

     #pragma unroll NUM_VALS
     for(int j = 0; j < NUM_VALS; j++)
       local_sum += ((float)vals[j])*((float)vals[j]);

    local_sum = BlockReduce(reduce).Sum(local_sum, valid_items);
    if(threadIdx.x == 0)
    {
      if(step == 1)
      {
        // initialize with the same norm for all positions
        //#pragma unroll 10
        for(int j = 0; j < 100; j++)
          atomicAdd(&gnorm_vec[j], local_sum);
      }
      else
          atomicAdd(&gnorm_vec[step % 100], local_sum);
    }

  }
}


#define LANES 2
#define QUAD 3
template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void
kOptimizerStatic8bit2StateBlockwise(T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* absmax1, float* absmax2, 
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n)
{

    //const int n_full = n + (n%BLOCK_SIZE);
    const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = (blockIdx.x * BLOCK_SIZE);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];
    // 2-5%
    const float correction1 = 1.0f - __powf(beta1, step);
    const float correction2 = sqrtf(1.0f -__powf(beta2, step));
    const float step_size = __fdividef(-lr*correction2,correction1);
    const int lane_id = threadIdx.x % LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float quadrants1[QUAD];
    float quadrants2[QUAD];

    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    T g_vals[N_PER_TH];
    typedef cub::BlockLoad<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[LANES][257];
    __shared__ float smem_quantiles2[LANES][257];
    typedef cub::BlockReduce<float, BLOCK_SIZE/N_PER_TH> BlockReduce1;
    typedef cub::BlockReduce<float, BLOCK_SIZE/N_PER_TH> BlockReduce2;
    __shared__ typename BlockReduce1::TempStorage reduce1;
    __shared__ typename BlockReduce2::TempStorage reduce2;
    __shared__ float smem_exchange1[1];
    __shared__ float smem_exchange2[1];

    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadChar::TempStorage loadc;
        typename StoreChar::TempStorage storec;
        typename StoreT::TempStorage storeh;
    } temp_storage;
    // init: 0.2 -> 0.23

    // 0.23 -> 0.23
      smem_quantiles1[0][threadIdx.x] = quantiles1[threadIdx.x];
      smem_quantiles2[0][threadIdx.x] = quantiles2[threadIdx.x];
      # pragma unroll
      for(unsigned int j = 1; j < LANES; j++)
      {
        smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];
        smem_quantiles2[j][threadIdx.x] = smem_quantiles2[0][threadIdx.x];
      }

    __syncthreads();

    #pragma unroll
    for(int k = 0; k < QUAD; k++)
    {
      quadrants1[k] = smem_quantiles1[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];
      quadrants2[k] = smem_quantiles2[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];
    }


    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
    {
        // loads: 0.23 -> 0.85/1.44
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state2[i]), c2s, valid_items, 0);

        new_local_abs_max1 = -FLT_MAX;
        new_local_abs_max2 = -FLT_MAX;

        //  update: 2.48/1.57 -> 2.51/1.60
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
						if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
						{
							s1_vals[j] = smem_quantiles1[lane_id][c1s[j]]*absmax1[i/BLOCK_SIZE];
							s1_vals[j] = (s1_vals[j]*beta1) + (((1.0f-beta1)*g_val));

							s2_vals[j] = smem_quantiles2[lane_id][c2s[j]]*absmax2[i/BLOCK_SIZE];
							s2_vals[j] = (s2_vals[j]*beta2) + (((1.0f-beta2)*g_val*g_val));
						}

            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));
        }


        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = BlockReduce1(reduce1).Reduce(new_local_abs_max1, cub::Max());
        new_local_abs_max2 = BlockReduce2(reduce2).Reduce(new_local_abs_max2, cub::Max());

        if(threadIdx.x == 0)
        {
          smem_exchange1[0] = new_local_abs_max1;
          smem_exchange2[0] = new_local_abs_max2;
        }

        __syncthreads();

        if(threadIdx.x == 0)
        {
          absmax1[i/BLOCK_SIZE] = new_local_abs_max1;
          absmax2[i/BLOCK_SIZE] = new_local_abs_max2;
        }
        else
        {
          new_local_abs_max1 = smem_exchange1[0];
          new_local_abs_max2 = smem_exchange2[0];
        }

        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), g_vals, valid_items, (T)0.0f);
        //  reduce: 2.67/1.69 -> 2.67/1.70
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
						if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
						{
							g_vals[j] = (T)(((float)g_vals[j]) + ((step_size*(__fdividef(s1_vals[j],(sqrtf(s2_vals[j])+(correction2*eps)))))));
							if(weight_decay > 0.0f)
									g_vals[j] = ((float)g_vals[j])*(1.0f-(lr*weight_decay));
						}
        }

        //  store: 0.85/1.44 -> 2.48/1.57
        __syncthreads();
        StoreT(temp_storage.storeh).Store(&(p[i]), g_vals, valid_items);

        //  quantizaztion: 2.67/1.70  -> 3.4/3.3
        # pragma unroll N_PER_TH 
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], __fdividef(s1_vals[j],new_local_abs_max1));
            c2s[j] = quantize_2D<0>(quadrants2, smem_quantiles2[lane_id], __fdividef(s2_vals[j],new_local_abs_max2));

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state2[i]), c2s, valid_items);
    }
}


#define LANES 2
#define QUAD 3
template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void
kOptimizerStatic8bit1StateBlockwise(T* p, T* __restrict__ const g, unsigned char* state1,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* absmax1,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n)
{

    //const int n_full = n + (n%BLOCK_SIZE);
    const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = (blockIdx.x * BLOCK_SIZE);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    // 2-5%
    const int lane_id = threadIdx.x % LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float quadrants1[QUAD];

    unsigned char c1s[N_PER_TH];
    T g_vals[N_PER_TH];
		T p_vals[N_PER_TH];

    typedef cub::BlockLoad<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[LANES][257];
    typedef cub::BlockReduce<float, BLOCK_SIZE/N_PER_TH> BlockReduce1;
    __shared__ typename BlockReduce1::TempStorage reduce1;
    __shared__ float smem_exchange1[1];

    __shared__ union {
        typename LoadT::TempStorage loadh;
        typename LoadChar::TempStorage loadc;
        typename StoreChar::TempStorage storec;
        typename StoreT::TempStorage storeh;
    } temp_storage;
    // init: 0.2 -> 0.23

    // 0.23 -> 0.23
		smem_quantiles1[0][threadIdx.x] = quantiles1[threadIdx.x];
		# pragma unroll
		for(unsigned int j = 1; j < LANES; j++)
			smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];

    __syncthreads();

    #pragma unroll
    for(int k = 0; k < QUAD; k++)
      quadrants1[k] = smem_quantiles1[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
    {
        // loads: 0.23 -> 0.85/1.44
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items, (T)0.0f);

        new_local_abs_max1 = -FLT_MAX;

        //  update: 2.48/1.57 -> 2.51/1.60
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
						if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
						{
							if(weight_decay > 0.0f)
								g_val += ((float)p_vals[j])*weight_decay;

							s1_vals[j] = smem_quantiles1[lane_id][c1s[j]]*absmax1[i/BLOCK_SIZE];

							switch(OPTIMIZER)
							{
									case MOMENTUM: 
										if(step == 1)
											s1_vals[j] = g_val;
										else
											s1_vals[j] = (s1_vals[j]*beta1) + g_val;
										break;
									case RMSPROP: 
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
										break;
									case ADAGRAD: 
										s1_vals[j] = s1_vals[j] + (g_val*g_val);
										break;
							}
						}

            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
        }


        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = BlockReduce1(reduce1).Reduce(new_local_abs_max1, cub::Max());

        if(threadIdx.x == 0)
          smem_exchange1[0] = new_local_abs_max1;

        __syncthreads();

        if(threadIdx.x == 0)
          absmax1[i/BLOCK_SIZE] = new_local_abs_max1;
        else
          new_local_abs_max1 = smem_exchange1[0];

        //  reduce: 2.67/1.69 -> 2.67/1.70
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
				{
						if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
						{
							switch(OPTIMIZER)
							{
									case MOMENTUM: 
										p_vals[j] = ((float)p_vals[j]) - lr*(s1_vals[j]);
										break;
									case RMSPROP: 
										g_val = g_vals[j];
										p_vals[j] = ((float)p_vals[j]) - lr*(__fdividef(g_val, sqrtf(s1_vals[j])+eps));
										break;
									case ADAGRAD: 
										g_val = g_vals[j];
										p_vals[j] = ((float)p_vals[j]) - lr*(__fdividef(g_val, sqrtf(s1_vals[j])+eps));
										break;
							}
						}
				}

        //  store: 0.85/1.44 -> 2.48/1.57
        __syncthreads();
        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);

        //  quantizaztion: 2.67/1.70  -> 3.4/3.3
        # pragma unroll N_PER_TH 
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], __fdividef(s1_vals[j],new_local_abs_max1));

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
    }
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template __device__ unsigned char dQuantize<0>(float* smem_code, const float rand, float x);
template __device__ unsigned char dQuantize<1>(float* smem_code, const float rand, float x);

template __global__ void kEstimateQuantiles(float *__restrict__ const A, float *code, const float offset, const float max_val, const int n);
template __global__ void kEstimateQuantiles(half *__restrict__ const A, float *code, const float offset, const half max_val, const int n);

#define MAKE_PreconditionOptimizer32bit1State(oname, gtype) \
template __global__ void kPreconditionOptimizer32bit1State<gtype, oname, 4096, 8>(gtype* g, gtype* p, \
                float* state1, float *unorm, \
                const float beta1, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const int n); \

MAKE_PreconditionOptimizer32bit1State(MOMENTUM, half)
MAKE_PreconditionOptimizer32bit1State(MOMENTUM, float)
MAKE_PreconditionOptimizer32bit1State(RMSPROP, half)
MAKE_PreconditionOptimizer32bit1State(RMSPROP, float)
MAKE_PreconditionOptimizer32bit1State(ADAGRAD, half)
MAKE_PreconditionOptimizer32bit1State(ADAGRAD, float)

#define MAKE_Optimizer32bit1State(oname, gtype) \
template __global__ void kOptimizer32bit1State<gtype, oname>(gtype* g, gtype* p, float* state1, float *unorm, const float max_unorm, const float param_norm, \
    const float beta1, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n); \

MAKE_Optimizer32bit1State(MOMENTUM, half)
MAKE_Optimizer32bit1State(MOMENTUM, float)
MAKE_Optimizer32bit1State(RMSPROP, half)
MAKE_Optimizer32bit1State(RMSPROP, float)
MAKE_Optimizer32bit1State(ADAGRAD, half)
MAKE_Optimizer32bit1State(ADAGRAD, float)

#define MAKE_PreconditionOptimizer32bit2State(oname, gtype) \
template __global__ void kPreconditionOptimizer32bit2State<gtype, oname, 4096, 8>(gtype* g, gtype* p,  \
                float* state1, float* state2, float *unorm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const int n); \

MAKE_PreconditionOptimizer32bit2State(ADAM, half)
MAKE_PreconditionOptimizer32bit2State(ADAM, float)

template __global__ void kOptimizer32bit2State<half, ADAM>(half* g, half* p, float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);
template __global__ void kOptimizer32bit2State<float, ADAM>(float* g, float* p, float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

#define MAKE_PreconditionStatic8bit1State(oname, gtype) \
template __global__ void kPreconditionOptimizerStatic8bit1State<gtype, oname>(gtype* p, gtype* __restrict__ const g, unsigned char*__restrict__  const state1,  \
                float *unorm,  \
                const float beta1,  \
                const float eps, const int step,  \
                float* __restrict__ const quantiles1,  \
                float* max1, float* new_max1,  \
                const float weight_decay, \
                const float gnorm_scale,  \
                const int n); \

MAKE_PreconditionStatic8bit1State(MOMENTUM, half)
MAKE_PreconditionStatic8bit1State(MOMENTUM, float)
MAKE_PreconditionStatic8bit1State(RMSPROP, half)
MAKE_PreconditionStatic8bit1State(RMSPROP, float)

#define MAKE_optimizerStatic8bit1State(oname, gtype) \
template __global__ void kOptimizerStatic8bit1State<gtype, oname>(gtype* p, gtype* const g, unsigned char* state1,  \
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1,  \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1,  \
                float* max1, float* new_max1,  \
                float weight_decay, \
                const float gnorm_scale,  \
                const int n); \

MAKE_optimizerStatic8bit1State(MOMENTUM, half)
MAKE_optimizerStatic8bit1State(MOMENTUM, float)
MAKE_optimizerStatic8bit1State(RMSPROP, half)
MAKE_optimizerStatic8bit1State(RMSPROP, float)

#define MAKE_PreconditionStatic8bit2State(oname, gtype) \
template __global__ void kPreconditionOptimizerStatic8bit2State<gtype, oname>(gtype* p, gtype* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2, \
                float *unorm, \
                const float beta1, const float beta2, \
                const float eps, const int step,  \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                const float gnorm_scale,  \
                const int n); \

MAKE_PreconditionStatic8bit2State(ADAM, half)
MAKE_PreconditionStatic8bit2State(ADAM, float)

#define MAKE_optimizerStatic8bit2State(oname, gtype) \
template __global__ void kOptimizerStatic8bit2State<gtype, oname>(gtype* p, gtype* const g, unsigned char* state1, unsigned char* state2, \
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale,  \
                const int n); \

MAKE_optimizerStatic8bit2State(ADAM, half)
MAKE_optimizerStatic8bit2State(ADAM, float)

template __global__ void kPercentileClipping<float, 2048, 4>(float * __restrict__ g, float *gnorm_vec, int step, const int n);
template __global__ void kPercentileClipping<half, 2048, 4>(half * __restrict__ g, float *gnorm_vec, int step, const int n);

template __global__ void kQuantizeBlockwise<half, 4096, 4, 0>(float * code, half * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template __global__ void kQuantizeBlockwise<float, 4096, 4, 0>(float * code, float * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template __global__ void kQuantizeBlockwise<half, 4096, 4, 1>(float * code, half * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);
template __global__ void kQuantizeBlockwise<float, 4096, 4, 1>(float * code, float * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n);

template __global__ void kDequantizeBlockwise<half, 4096, 1024, 4>(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, half *out, const int n);
template __global__ void kDequantizeBlockwise<float, 4096, 1024, 4>(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, float *out, const int n);
template __global__ void kDequantizeBlockwise<half, 2048, 512, 4>(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, half *out, const int n);
template __global__ void kDequantizeBlockwise<float, 2048, 512, 4>(float *code, unsigned char * __restrict__ const A, float * __restrict__ const absmax, float *out, const int n);



#define MAKE_OptimizerStatic8bit2StateBlockwise(oname, gtype, block_size, num_per_thread) \
template __global__ void kOptimizerStatic8bit2StateBlockwise<gtype, oname, block_size, num_per_thread>(gtype* p, gtype* __restrict__ const g, unsigned char* state1, unsigned char* state2, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* absmax1, float* absmax2,  \
                float weight_decay, \
                const float gnorm_scale, const bool skip_zeros, const int n); \

MAKE_OptimizerStatic8bit2StateBlockwise(ADAM, float, 2048, 8)
MAKE_OptimizerStatic8bit2StateBlockwise(ADAM, half, 2048, 8)

#define MAKE_OptimizerStatic8bit1StateBlockwise(oname, gtype, block_size, num_per_thread) \
template __global__ void kOptimizerStatic8bit1StateBlockwise<gtype, oname, block_size, num_per_thread>( \
		gtype* p, gtype* __restrict__ const g, unsigned char* state1, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, \
                float* absmax1, \
                float weight_decay, \
                const float gnorm_scale, const bool skip_zeros, const int n); \

MAKE_OptimizerStatic8bit1StateBlockwise(MOMENTUM, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(MOMENTUM, half, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(RMSPROP, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(RMSPROP, half, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(ADAGRAD, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(ADAGRAD, half, 2048, 8)
