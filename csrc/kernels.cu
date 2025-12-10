// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "common.cuh"
#include "kernels.cuh"
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

#if CCCL_VERSION >= 2008002
#include <cuda/std/functional>
#define CUB_REDUCTIONOP_MAX                                                                                            \
    cuda::maximum<> {}
#else
#define CUB_REDUCTIONOP_MAX cub::Max()
#endif

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

__device__ static float fp4_dequantization_lut[8] = {
    0.0f,            // 0b000
    0.005208333333f, // 0b001
    0.66666667f,     // 0b010
    1.0f,            // 0b011
    0.33333333f,     // 0b100
    0.5f,            // 0b101
    0.16666667f,     // 0b110
    0.25f            // 0b111
};

__device__ static float nf4_dequantization_lut[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};

// source: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ float atomicMax(float* address, float val) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(reinterpret_cast<int*>(address), assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float dDequantizeFP4Tree(unsigned char val) {
    float sign = 1.0f - 2 * ((val & 0b1000) >> 3);
    return fp4_dequantization_lut[val & 0b111] * sign;
}

__device__ __forceinline__ float dDequantizeNF4(unsigned char val) { return nf4_dequantization_lut[val & 0x0F]; }

// sign function for lion
// taken from https://stackoverflow.com/a/4609795, but not sure if there's a proper way to do this in CUDA

template <typename T> __device__ int sgn(T val) { return (T(0) < val) - (val < T(0)); }

template <int STOCHASTIC> __device__ unsigned char dQuantize(float* smem_code, const float rand, float x) {
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for (int i = 64; i > 0; i >>= 1) {
        if (x > val) {
            lower_pivot = pivot;
            lower = val;
            pivot += i;
        } else {
            upper_pivot = pivot;
            upper = val;
            pivot -= i;
        }
        val = smem_code[pivot];
    }

    if (upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if (lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if (!STOCHASTIC) {
        if (x > val) {
            float midpoint = (upper + val) * 0.5f;
            if (x > midpoint) {
                return upper_pivot;
            } else
                return pivot;
        } else {
            float midpoint = (lower + val) * 0.5f;
            if (x < midpoint)
                return lower_pivot;
            else
                return pivot;
        }
    } else {
        if (x > val) {
            float dist_to_upper = fabsf(upper - x);
            float dist_full = upper - val;
            if (rand >= dist_to_upper / dist_full)
                return upper_pivot;
            else
                return pivot;
        } else {
            float dist_to_lower = fabsf(lower - x);
            float dist_full = val - lower;
            if (rand >= dist_to_lower / dist_full)
                return lower_pivot;
            else
                return pivot;
        }
    }
}

__launch_bounds__(TH, 4) __global__
    void kQuantize(float* code, float* __restrict__ const A, unsigned char* out, const int n) {
    const int n_full = (NUM_BLOCK * (n / NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
    int valid_items = (blockIdx.x + 1 == gridDim.x) ? n - (blockIdx.x * NUM_BLOCK) : NUM_BLOCK;
    const int base_idx = (blockIdx.x * NUM_BLOCK);

    float vals[NUM];
    unsigned char qvals[NUM];
    // const int lane_id = threadIdx.x % 2;

    typedef cub::BlockLoad<float, TH, NUM, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    typedef cub::BlockStore<unsigned char, TH, NUM, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

    __shared__ typename LoadFloat::TempStorage loadf;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ float smem_code[256];
    //__shared__ float smem_code[2][257];

    if (threadIdx.x < 256) {
        smem_code[threadIdx.x] = code[threadIdx.x];
        // smem_code[0][threadIdx.x] = code[threadIdx.x];
        // smem_code[1][threadIdx.x] = smem_code[0][threadIdx.x];
    }

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * NUM_BLOCK) {
        // number of values already processed in blocks +
        // number of values already processed in this block +
        // rand_offset % mod value
        valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

        __syncthreads();
        LoadFloat(loadf).Load(&(A[i]), vals, valid_items);

#pragma unroll 4
        for (int j = 0; j < NUM; j++)
            qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

        __syncthreads();
        StoreChar(storec).Store(&(out[i]), qvals, valid_items);
    }
}

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void
    kDequantizeBlockwise(float* code, unsigned char* A, float* absmax, T* out, const int blocksize, const int n) {

    const int n_load = (gridDim.x * TILE_SIZE);
    int valid_items_load = 0;
    int valid_items_store = 0;
    const int base_idx = (blockIdx.x * TILE_SIZE);

    T vals[NUM_PER_TH * ((DATA_TYPE > 0) ? 2 : 1)];
    unsigned char qvals[NUM_PER_TH];
    float local_abs_max = -FLT_MAX;

    typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    for (int i = base_idx; i < n_load; i += gridDim.x * TILE_SIZE) {
        if (DATA_TYPE > 0) {
            // Cast n to int64_t to avoid overflow for large n
            valid_items_load = min(TILE_SIZE, static_cast<int>((static_cast<int64_t>(n) + 1) / 2) - i);
            valid_items_store = min(TILE_SIZE * 2, n - i * 2);
        } else {
            valid_items_load = min(TILE_SIZE, n - i);
            valid_items_store = valid_items_load;
        }

        // Since blocksize will always be a power-of-2, we avoid more expensive
        // division by the blocksize and instead use a shift operation.
        // This is equivalent to (i+threadId.x*NUM_PER_TH)/blocksize.
        local_abs_max = __ldg(&absmax[(i + threadIdx.x * NUM_PER_TH) >> (31 - __clz(blocksize))]);

        __syncthreads();
        LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

        switch (DATA_TYPE) {
        case General8bit:
// load code through read-only cache via __ldg
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++)
                vals[j] = __ldg(&code[qvals[j]]) * local_abs_max;
            break;
        case FP4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                vals[j * 2] = dDequantizeFP4Tree(qvals[j] >> 4) * local_abs_max;
                vals[j * 2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F) * local_abs_max;
            }
            break;
        case NF4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                vals[j * 2] = dDequantizeNF4(qvals[j] >> 4) * local_abs_max;
                vals[j * 2 + 1] = dDequantizeNF4(qvals[j] & 0x0F) * local_abs_max;
            }
            break;
        }

        __syncthreads();
        StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i * 2 : i]), vals, valid_items_store);
    }
}

__global__ void kDequantize(float* code, unsigned char* A, float* out, const int n) {
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    __shared__ float smem_code[256];
    if (threadIdx.x < 256) {
        smem_code[threadIdx.x] = code[threadIdx.x];
    }

    __syncthreads();

    for (int i = idx; i < n; i += numThreads) {
        out[i] = smem_code[A[i]];
    }
}

#define NUM_PER_THREAD 4

#define NUM8BIT 16
#define NUM_THREADS 256
#define NUM_PER_BLOCK 4096

template <typename T, int OPTIMIZER>
__global__ void __launch_bounds__(NUM_THREADS, 2) kPreconditionOptimizerStatic8bit2State(
    T* p, T* __restrict__ const g, unsigned char* __restrict__ const state1, unsigned char* __restrict__ const state2,
    float* unorm, const float beta1, const float beta2, const float eps, const int step,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* max1, float* max2,
    float* new_max1, float* new_max2, const float gnorm_scale, const int n
) {
    const int n_full = gridDim.x * NUM_PER_BLOCK;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items =
        n - (blockIdx.x * NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (blockIdx.x * NUM_PER_BLOCK);
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

    if (threadIdx.x < 256) {
        smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];
        smem_quantiles2[threadIdx.x] = quantiles2[threadIdx.x];
    }

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += NUM_THREADS * gridDim.x * NUM8BIT) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;

        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state1[i]), m_c1, valid_items, 128);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state2[i]), r_c2, valid_items, 128);
        __syncthreads();

#pragma unroll 16
        for (int j = 0; j < NUM8BIT; j++) {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]] * max1[0] * beta1;
            s1_vals[j] += (1.0f - beta1) * g_val;
            local_max_s1 = fmaxf(local_max_s1, fabsf(s1_vals[j]));
        }

#pragma unroll 16
        for (int j = 0; j < NUM8BIT; j++) {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s2_vals[j] = smem_quantiles2[r_c2[j]] * max2[0] * beta2;
            s2_vals[j] += (1.0f - beta2) * g_val * g_val;
            local_max_s2 = fmaxf(local_max_s2, fabsf(s2_vals[j]));
        }

        if (unorm != NULL) {
#pragma unroll 16
            for (int j = 0; j < NUM8BIT; j++) {
                float correction1 = __fdividef(1.0f, 1.0f - powf(beta1, step));
                float correction2 = __fdividef(1.0f, 1.0f - powf(beta2, step));
                s1_vals[j] *= correction1;
                s2_vals[j] *= correction2;
                float update_val = s1_vals[j] / (sqrtf(s2_vals[j]) + eps); // update
                local_unorm += update_val * update_val;
            }
        }
    }

    __syncthreads();
    local_max_s1 = BlockReduce(temp_storage.reduce).Reduce(local_max_s1, CUB_REDUCTIONOP_MAX, valid_items);
    __syncthreads();
    local_max_s2 = BlockReduce(temp_storage.reduce).Reduce(local_max_s2, CUB_REDUCTIONOP_MAX, valid_items);
    if (unorm != NULL) {
        __syncthreads();
        local_unorm = BlockReduce(temp_storage.reduce).Sum(local_unorm, valid_items);
    }

    if (threadIdx.x == 0) {
        atomicMax(&new_max1[0], local_max_s1);
        atomicMax(&new_max2[0], local_max_s2);
        if (unorm != NULL) {
            atomicAdd(&unorm[0], local_unorm);
        }
    }
}

#define NUM_PER_THREAD2 4
#define NUM_THREADS2 1024
#define NUM_PER_BLOCK2 4096

template <typename T, int OPTIMIZER>
__global__ void __launch_bounds__(NUM_THREADS2, 1) kOptimizerStatic8bit2State(
    T* p, T* const g, unsigned char* state1, unsigned char* state2, const float* unorm, const float max_unorm,
    const float param_norm, const float beta1, const float beta2, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* max1, float* max2,
    float* new_max1, float* new_max2, float weight_decay, const float gnorm_scale, const int n
) {

    const int n_full = (blockDim.x * gridDim.x) * NUM_PER_THREAD2;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float s2_vals[NUM_PER_THREAD2];
    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2 = sqrtf(1.0f - powf(beta2, step));
    const float step_size = -lr * correction2 / correction1;
    // const float step_size = -lr*correction2/correction1;
    float new_max_val1 = 1.0f / new_max1[0];
    float new_max_val2 = 1.0f / new_max2[0];
    float update_scale = 1.0f;

    if (max_unorm > 0.0f) {
        update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
        if (update_scale > max_unorm * param_norm) {
            update_scale = (max_unorm * param_norm) / update_scale;
        } else {
            update_scale = 1.0f;
        }
    } else {
        update_scale = 1.0f;
    }

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

    if (threadIdx.x < 512) {
        if (threadIdx.x < 256)
            smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];
        else
            smem_quantiles2[threadIdx.x - 256] = quantiles2[threadIdx.x - 256];
    }

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * NUM_THREADS2 * NUM_PER_THREAD2) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state2[i]), c2s, valid_items, 0);
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items);

        if ((i + (threadIdx.x * NUM_PER_THREAD2) + NUM_PER_THREAD2) > n) {
            continue;
        }

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD2; j++) {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[c1s[j]];
            s1_vals[j] = s1_vals[j] * max1[0];

            s1_vals[j] = (s1_vals[j] * beta1) + (((1.0f - beta1) * g_val));

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j] * new_max_val1);

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if (signbit(smem_quantiles1[c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f)
                    c1s[j] += 1;
                else
                    c1s[j] -= 1;
            }

            s2_vals[j] = smem_quantiles2[c2s[j]];
            s2_vals[j] = s2_vals[j] * max2[0];
            s2_vals[j] = (s2_vals[j] * beta2) + (((1.0f - beta2) * g_val * g_val));
            c2s[j] = dQuantize<0>(smem_quantiles2, 0.0f, s2_vals[j] * new_max_val2);
        }

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD2; j++) {
            p_vals[j] = (T)(((float)p_vals[j]) +
                            ((update_scale * step_size * (s1_vals[j] / (sqrtf(s2_vals[j]) + (correction2 * eps))))));
            if (weight_decay > 0.0f)
                p_vals[j] = update_scale * ((float)p_vals[j]) * (1.0f - (lr * weight_decay));
        }

        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state2[i]), c2s, valid_items);
        __syncthreads();
    }
}

template <typename T, int OPTIMIZER>
__global__ void __launch_bounds__(NUM_THREADS, 2) kPreconditionOptimizerStatic8bit1State(
    T* p, T* __restrict__ const g, unsigned char* __restrict__ const state1, float* unorm, const float beta1,
    const float beta2, const float eps, const int step, float* __restrict__ const quantiles1, float* max1,
    float* new_max1, const float weight_decay, const float gnorm_scale, const int n
) {
    const int n_full = gridDim.x * NUM_PER_BLOCK;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items =
        n - (blockIdx.x * NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (blockIdx.x * NUM_PER_BLOCK);
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

    if (threadIdx.x < 256)
        smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * NUM_THREADS * NUM8BIT) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;

        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadUInt8(temp_storage.loadc).Load(&(state1[i]), m_c1, valid_items, 128);

#pragma unroll 16
        for (int j = 0; j < NUM8BIT; j++) {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]] * max1[0];
            switch (OPTIMIZER) {
            case ADAGRAD:
            case MOMENTUM:
                if (step == 1)
                    s1_vals[j] = (float)g_vals[j];
                else
                    s1_vals[j] = s1_vals[j] * beta1 + ((float)g_vals[j]);
                if (unorm != NULL)
                    local_unorm += s1_vals[j] * s1_vals[j];
                break;
            case LION:
                s1_vals[j] = s1_vals[j] * beta2 + ((1.0f - beta2) * g_val);
                break;
            case RMSPROP:
                s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * (g_val * g_val));
                break;
            }

            local_max_s1 = fmaxf(local_max_s1, fabsf(s1_vals[j]));
        }
    }

    __syncthreads();
    local_max_s1 = BlockReduce(temp_storage.reduce).Reduce(local_max_s1, CUB_REDUCTIONOP_MAX, valid_items);
    if (threadIdx.x == 0) {
        atomicMax(&new_max1[0], local_max_s1);
    }
    if (unorm != NULL) {
        __syncthreads();
        local_unorm = BlockReduce(temp_storage.reduce).Sum(local_unorm, valid_items);
        if (threadIdx.x == 0) {
            atomicAdd(&unorm[0], local_unorm);
        }
    }
}

template <typename T, int OPTIMIZER>
__global__ void __launch_bounds__(1024, 1) kOptimizerStatic8bit1State(
    T* p, T* const g, unsigned char* state1, const float* unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* max1, float* new_max1, float weight_decay, const float gnorm_scale,
    const int n
) {

    const int n_full = (blockDim.x * gridDim.x) * NUM_PER_THREAD2;
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float new_max_val1 = 1.0f / new_max1[0];
    float update_scale = 1.0f;

    if (max_unorm > 0.0f) {
        update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
        if (update_scale > max_unorm * param_norm) {
            update_scale = (max_unorm * param_norm) / update_scale;
        } else {
            update_scale = 1.0f;
        }
    } else {
        update_scale = 1.0f;
    }

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

    if (threadIdx.x < 256)
        smem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];

    __syncthreads();

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * NUM_THREADS2 * NUM_PER_THREAD2) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items);

        if ((i + (threadIdx.x * NUM_PER_THREAD2) + NUM_PER_THREAD2) > n) {
            continue;
        }

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD2; j++) {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;

            if (weight_decay > 0.0f) {
                switch (OPTIMIZER) {
                case ADAGRAD:
                case MOMENTUM:
                case RMSPROP:
                    g_val += ((float)p_vals[j]) * weight_decay;
                    break;
                case LION:
                    p_vals[j] = ((float)p_vals[j]) * (1.0f - lr * weight_decay);
                    break;
                }
            }

            s1_vals[j] = smem_quantiles1[c1s[j]] * max1[0];

            switch (OPTIMIZER) {
            case ADAGRAD:
            case MOMENTUM:
                if (step == 1)
                    s1_vals[j] = g_vals[j];
                else
                    s1_vals[j] = s1_vals[j] * beta1 + ((float)g_vals[j]);

                p_vals[j] = ((float)p_vals[j]) + (-lr * update_scale * (s1_vals[j]));
                break;
            case LION:
                p_vals[j] =
                    ((float)p_vals[j]) - (lr * sgn(((float)s1_vals[j]) * beta1 + ((1.0f - beta1) * ((float)g_val))));
                s1_vals[j] = s1_vals[j] * beta2 + ((1.0f - beta2) * g_val);
                break;
            case RMSPROP:
                s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * (g_val * g_val));
                p_vals[j] = ((float)p_vals[j]) - (lr * __fdividef(g_val, sqrtf(s1_vals[j]) + eps));
                break;
            }

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j] * new_max_val1);

            // make sure state1 term has still the same sign after quantization
            if (signbit(smem_quantiles1[c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f)
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

template <typename T, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPercentileClipping(T* __restrict__ g, float* gnorm_vec, int step, const int n) {
    const int n_full = (BLOCK_SIZE * (n / BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
    int valid_items = 0;

    typedef cub::BlockReduce<float, BLOCK_SIZE / NUM_VALS> BlockReduce;
    typedef cub::BlockLoad<T, BLOCK_SIZE / NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;

    __shared__ typename BlockReduce::TempStorage reduce;

    __shared__ typename LoadT::TempStorage loadT;
    T vals[NUM_VALS];
    float local_sum = 0.0f;

    for (unsigned int i = (blockIdx.x * BLOCK_SIZE); i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
        local_sum = 0.0f;

        __syncthreads();
        LoadT(loadT).Load(&(g[i]), vals, valid_items, (T)0.0f);

#pragma unroll NUM_VALS
        for (int j = 0; j < NUM_VALS; j++)
            local_sum += ((float)vals[j]) * ((float)vals[j]);

        local_sum = BlockReduce(reduce).Sum(local_sum, valid_items);
        if (threadIdx.x == 0) {
            if (step == 1) {
                // initialize with the same norm for all positions
                // #pragma unroll 10
                for (int j = 0; j < 100; j++)
                    atomicAdd(&gnorm_vec[j], local_sum);
            } else
                atomicAdd(&gnorm_vec[step % 100], local_sum);
        }
    }
}

#define DENORM 1.0f / 127.0f
#define MAX_SPARSE_COUNT 32
#define SMEM_SIZE 8 * 256

template <typename T, int SPMM_ITEMS, int BITS>
__global__ void kspmm_coo_very_sparse_naive(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, T* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
) {

    // 0. load balancing: We process rows with most columns first (count_vec)and we process one row per block
    //    If a block finishes, the next one is scheduled. Since the last blocks like have fewer
    //    elements they finish faster "fillin up" the gaps left by larger blocks

    // without tensor cores
    // 1. use rowidx_length to find what to load (as many blocks as there are rows)
    // 2. Load A into registers
    // 3. each warp loads all required rows of B but each warp is offset by k
    // 4. Do mma operations that accumulate into registers
    // 5. Each warp stores its output row into matrix C

    const int count = max_count[blockIdx.x];
    const int local_max_idx = max_idx[blockIdx.x];
    const int offset = local_max_idx == 0 ? 0 : offset_rowidx[local_max_idx - 1];
    const int local_row_idx = rowidx[offset];

    const int warp_id = threadIdx.x / 32;
    const int warp_idx = threadIdx.x % 32;
    const int warp_offset = (warp_id * 32) * SPMM_ITEMS;
    const int num_items = BITS == 8 ? 8 : 8;
    int idx_col_B = warp_offset;
    int local_idx_col_B_offset = 0;

    half local_valA[MAX_SPARSE_COUNT];
    int local_colidxA[MAX_SPARSE_COUNT];
    half local_valC[SPMM_ITEMS];
    T local_valsB[num_items];
    half local_valOut[num_items];
    // 128 byte loads per warp == 4 bytes per thread

    // 2. Load A into registers
    for (int j = 0; j < MAX_SPARSE_COUNT; j++) {
        local_valA[j] = j < count ? values[offset + j] : __float2half(0.0f);
        local_colidxA[j] = j < count ? colidx[offset + j] : 0;
    }

    // each thread processes SPMM_ITEMS=32 per iteration. We have 256 threads. 32*256=x192
    // we expect each warp to be SPMM_ITEMS*32 apart
    // we have a total of 128 bytes for the bank with a bank size of 4 bytes
    // added 3 bytes = 6 values between warps should reduce bank conflicts
    __shared__ half smem_dequant_stats[SMEM_SIZE];

    while (idx_col_B < colsB) {

        if (dequant_stats != NULL) {
            for (int i = threadIdx.x; i < SMEM_SIZE; i += blockDim.x)
                if ((idx_col_B + i - local_idx_col_B_offset) < colsB)
                    smem_dequant_stats[i] = dequant_stats[idx_col_B + i - local_idx_col_B_offset];

            __syncthreads();
        }

#pragma unroll SPMM_ITEMS
        for (int j = 0; j < SPMM_ITEMS; j++)
            local_valC[j] = 0.0f;

#pragma unroll
        for (int i = 0; i < count; i++) {
            // 3. each warp loads all required rows of B but each warp is offset by k
            int row_offset = colsB * local_colidxA[i];

#pragma unroll SPMM_ITEMS
            for (int j = 0; j < SPMM_ITEMS; j += num_items) {
                // 4. Multiply the tile -> accumulate outputs in shared memory until 128 bytes it reached
                int idx = idx_col_B + (warp_idx * SPMM_ITEMS) + j;
                if (idx >= colsB) {
                    break;
                }
                if ((idx + num_items < colsB)) {
                    if (BITS == 8)
                        reinterpret_cast<float2(&)[num_items]>(local_valsB)[0] =
                            reinterpret_cast<float2*>(B)[(row_offset + idx) / num_items];
                    else
                        reinterpret_cast<float4(&)[num_items]>(local_valsB)[0] =
                            reinterpret_cast<float4*>(B)[(row_offset + idx) / num_items];
                } else {
#pragma unroll num_items
                    for (int k = 0; k < num_items; k++)
                        if (idx + k < colsB)
                            local_valsB[k] = B[row_offset + idx + k];
                        else
                            local_valsB[k] = 0.0f;
                }
#pragma unroll num_items
                for (int k = 0; k < num_items; k++) {
                    if (BITS == 8 && dequant_stats != NULL)
                    // we do texture cache reads (__ldg) on dequant_stats which should be super fast
                    {
                        float valB = local_valsB[k];
                        float valA = local_valA[i];
                        if (valB != 0.0 && valA != 0.0)
                            local_valC[j + k] =
                                (float)local_valC[j + k] +
                                ((float)smem_dequant_stats[idx + k - local_idx_col_B_offset]) * DENORM * valB * valA;
                    } else
                        local_valC[j + k] = (float)local_valC[j + k] + (float)local_valsB[k] * (float)local_valA[i];
                }
            }
        }

        int idx_row_C = (colsB * local_row_idx);

#pragma unroll SPMM_ITEMS
        for (int j = 0; j < SPMM_ITEMS; j += num_items) {
            // int idx_col_C =  idx_col_B + (32*j) + warp_idx;
            int idx_col_C = idx_col_B + warp_idx * SPMM_ITEMS + j;
            int idx_val = idx_col_C + idx_row_C;

            if (idx_col_C + num_items < colsB) {

                // load outputs to do inplace addition
                reinterpret_cast<float4(&)[num_items / 4]>(local_valOut)[0] =
                    reinterpret_cast<float4*>(out)[idx_val / num_items];

#pragma unroll num_items
                for (int k = 0; k < num_items; k++)
                    local_valC[(j / num_items) + k] = (float)local_valC[(j / num_items) + k] + (float)local_valOut[k];

                reinterpret_cast<float4*>(out)[idx_val / num_items] =
                    reinterpret_cast<float4(&)[num_items]>(local_valC)[j / num_items];
            } else {
#pragma unroll num_items
                for (int k = 0; k < num_items; k++)
                    if (idx_col_C + k < colsB)
                        out[idx_val + k] = (float)out[idx_val + k] + (float)local_valC[j + k];
            }
        }

        idx_col_B += blockDim.x * SPMM_ITEMS;
        local_idx_col_B_offset += blockDim.x * SPMM_ITEMS;
    }
}

#define num_values_4bit 32

template <typename T, int THREADS, int BITS>
__global__ void kgemm_4bit_inference_naive(
    int M, int N, int K, T* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype, T* out,
    int lda, int ldb, int ldc, int blocksize
) {

    // per threadblock:
    // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
    // 4 warps -> 4 loads per iter
    // 1x32 * 32x4 -> 1x4 outputs per thread block
    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREADS / 32];

    const int warp_idx = threadIdx.x / 32;
    const int warp_lane = threadIdx.x % 32;
    const int row_B = (THREADS / 32) * blockIdx.x + warp_idx;
    const int offset_B = ldb * row_B;
    const int num_values_8bit = num_values_4bit / 2;
    float local_C = 0.0f;

    unsigned char local_B_4bit[num_values_8bit];
    T local_B[num_values_4bit / 4];
    T local_A[num_values_4bit / 4];
    __shared__ T quant_map[16];
    T local_absmax = T(0.0f);

    if (threadIdx.x < 16)
        quant_map[threadIdx.x] = T(__ldg(&datatype[threadIdx.x]));
    // for(int i = threadIdx.x; i < 16; i++)
    // quant_map[i] = T(__ldg(&datatype[i]));
    __syncthreads();

    // A: [1, K]
    // B: [N, K]
    for (int inner_idx = warp_lane * num_values_4bit; inner_idx < K; inner_idx += 32 * num_values_4bit) {
        const int inner_idx_halved = inner_idx / 2;

        // Since blocksize will always be a power-of-2, we avoid more expensive
        // division by the blocksize and instead use a shift operation.
        // This is equivalent to (i+threadId.x*NUM_PER_TH)/blocksize.
        const int absidx = ((2 * offset_B) + inner_idx) >> (31 - __clz(blocksize));

        local_absmax = __ldg(&(absmax[absidx]));

        if (row_B < M) {
            if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
                // this is the most important for performance considerations
                reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] =
                    reinterpret_cast<int4*>(B)[(offset_B + (inner_idx_halved)) / (num_values_8bit)];
            } else {
#pragma unroll
                for (int j = 0; j < (num_values_8bit); j++)
                    if ((inner_idx_halved) + j < (K / 2))
                        local_B_4bit[j] = B[offset_B + inner_idx_halved + j];
                    else
                        local_B_4bit[j] = 0b01110111;
            }
        } else {
#pragma unroll
            for (int j = 0; j < (num_values_8bit); j++)
                local_B_4bit[j] = 0b01110111;
        }

        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int k = 0; k < num_values_8bit / 4; k++) {
#if BNB_BF16_AVAILABLE
                local_B[k * 2] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4] * local_absmax;
                local_B[k * 2 + 1] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F] * local_absmax;
#else
                // bf16 multipliation not supported
                local_B[k * 2] =
                    T((float)quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4] * (float)local_absmax);
                local_B[k * 2 + 1] =
                    T((float)quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F] * (float)local_absmax);
#endif
            }

            if (inner_idx + (num_values_4bit / 4) + (i * num_values_4bit / 4) < K) {
                // this is also relatively important for performance
                if (BITS == 16) {
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] =
                        reinterpret_cast<int4*>(A)[inner_idx / (num_values_4bit / 4) + i];
                } else {
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] =
                        reinterpret_cast<int4*>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 0];
                    reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] =
                        reinterpret_cast<int4*>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 1];
                }

            } else
#pragma unroll
                for (int k = 0; k < num_values_4bit / 4; k++)
                    if (inner_idx + (i * num_values_4bit / 4) + k < K)
                        local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
                    else
                        local_A[k] = T(0.0f);

// accumulate in float; small performance hit for Ampere, but lower error for outputs
#pragma unroll
            for (int k = 0; k < num_values_4bit / 4; k++) {
#if BNB_BF16_AVAILABLE
                local_C += (float)(local_A[k] * local_B[k]);
#else
                // bf16 multipliation not supported
                local_C += ((float)local_A[k] * (float)local_B[k]);
#endif
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

    if (row_B < M && warp_lane == 0)
        out[row_B] = T(local_C);
}

template <typename T, int FUNC> __global__ void kfunc(T* A, T* B, T value, long n) {
    for (long i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (blockDim.x * gridDim.x)) {
        switch (FUNC) {
        case FILL:
            A[i] = (T)value;
            break;
        case ARANGE:
            A[i] = (T)i;
            break;
        case _MUL:
            A[i] = A[i] * B[i];
            break;
        }
    }
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template __global__ void kfunc<float, FILL>(float* A, float* B, float value, long n);
template __global__ void kfunc<unsigned char, FILL>(unsigned char* A, unsigned char* B, unsigned char value, long n);
template __global__ void kfunc<float, ARANGE>(float* A, float* B, float value, long n);
template __global__ void kfunc<float, _MUL>(float* A, float* B, float value, long n);

template __global__ void kgemm_4bit_inference_naive<half, 128, 16>(
    int M, int N, int K, half* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype, half* out,
    int lda, int ldb, int ldc, int blocksize
);
template __global__ void kgemm_4bit_inference_naive<__nv_bfloat16, 128, 16>(
    int M, int N, int K, __nv_bfloat16* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype,
    __nv_bfloat16* out, int lda, int ldb, int ldc, int blocksize
);
template __global__ void kgemm_4bit_inference_naive<float, 128, 32>(
    int M, int N, int K, float* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype,
    float* out, int lda, int ldb, int ldc, int blocksize
);

template __global__ void kspmm_coo_very_sparse_naive<half, 8, 16>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);
template __global__ void kspmm_coo_very_sparse_naive<half, 16, 16>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);
template __global__ void kspmm_coo_very_sparse_naive<half, 32, 16>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, half* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);
template __global__ void kspmm_coo_very_sparse_naive<signed char, 8, 8>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);
template __global__ void kspmm_coo_very_sparse_naive<signed char, 16, 8>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);
template __global__ void kspmm_coo_very_sparse_naive<signed char, 32, 8>(
    int* max_count, int* max_idx, int* offset_rowidx, int* rowidx, int* colidx, half* values, signed char* B, half* out,
    float* __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB
);

template __device__ unsigned char dQuantize<0>(float* smem_code, const float rand, float x);
template __device__ unsigned char dQuantize<1>(float* smem_code, const float rand, float x);

#define MAKE_PreconditionStatic8bit1State(oname, gtype)                                                                \
    template __global__ void kPreconditionOptimizerStatic8bit1State<gtype, oname>(                                     \
        gtype * p, gtype* __restrict__ const g, unsigned char* __restrict__ const state1, float* unorm,                \
        const float beta1, const float beta2, const float eps, const int step, float* __restrict__ const quantiles1,   \
        float* max1, float* new_max1, const float weight_decay, const float gnorm_scale, const int n                   \
    );

MAKE_PreconditionStatic8bit1State(MOMENTUM, half)
MAKE_PreconditionStatic8bit1State(MOMENTUM, float)
MAKE_PreconditionStatic8bit1State(RMSPROP, half)
MAKE_PreconditionStatic8bit1State(RMSPROP, float)
MAKE_PreconditionStatic8bit1State(LION, half)
MAKE_PreconditionStatic8bit1State(LION, float)
MAKE_PreconditionStatic8bit1State(ADAGRAD, half)
MAKE_PreconditionStatic8bit1State(ADAGRAD, float)

#define MAKE_optimizerStatic8bit1State(oname, gtype)                                                                   \
    template __global__ void kOptimizerStatic8bit1State<gtype, oname>(                                                 \
        gtype * p, gtype* const g, unsigned char* state1, const float* unorm, const float max_unorm,                   \
        const float param_norm, const float beta1, const float beta2, const float eps, const int step, const float lr, \
        float* __restrict__ const quantiles1, float* max1, float* new_max1, float weight_decay,                        \
        const float gnorm_scale, const int n                                                                           \
    );

MAKE_optimizerStatic8bit1State(MOMENTUM, half)
MAKE_optimizerStatic8bit1State(MOMENTUM, float)
MAKE_optimizerStatic8bit1State(RMSPROP, half)
MAKE_optimizerStatic8bit1State(RMSPROP, float)
MAKE_optimizerStatic8bit1State(LION, half)
MAKE_optimizerStatic8bit1State(LION, float)
MAKE_optimizerStatic8bit1State(ADAGRAD, half)
MAKE_optimizerStatic8bit1State(ADAGRAD, float)

#define MAKE_PreconditionStatic8bit2State(oname, gtype)                                                                \
    template __global__ void kPreconditionOptimizerStatic8bit2State<gtype, oname>(                                     \
        gtype * p, gtype* __restrict__ const g, unsigned char* __restrict__ const state1,                              \
        unsigned char* __restrict__ const state2, float* unorm, const float beta1, const float beta2, const float eps, \
        const int step, float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* max1,       \
        float* max2, float* new_max1, float* new_max2, const float gnorm_scale, const int n                            \
    );

MAKE_PreconditionStatic8bit2State(ADAM, half)
MAKE_PreconditionStatic8bit2State(ADAM, float)

#define MAKE_optimizerStatic8bit2State(oname, gtype)                                                                   \
    template __global__ void kOptimizerStatic8bit2State<gtype, oname>(                                                 \
        gtype * p, gtype* const g, unsigned char* state1, unsigned char* state2, const float* unorm,                   \
        const float max_unorm, const float param_norm, const float beta1, const float beta2, const float eps,          \
        const int step, const float lr, float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,    \
        float* max1, float* max2, float* new_max1, float* new_max2, float weight_decay, const float gnorm_scale,       \
        const int n                                                                                                    \
    );

MAKE_optimizerStatic8bit2State(ADAM, half)
MAKE_optimizerStatic8bit2State(ADAM, float)

template __global__ void
    kPercentileClipping<float, 2048, 4>(float* __restrict__ g, float* gnorm_vec, int step, const int n);
template __global__ void
    kPercentileClipping<half, 2048, 4>(half* __restrict__ g, float* gnorm_vec, int step, const int n);

template __global__ void kDequantizeBlockwise<half, 512, 64, 8, FP4>(
    float* code, unsigned char* A, float* absmax, half* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<half, 512, 64, 8, General8bit>(
    float* code, unsigned char* A, float* absmax, half* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<half, 512, 64, 8, NF4>(
    float* code, unsigned char* A, float* absmax, half* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, FP4>(
    float* code, unsigned char* A, float* absmax, float* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(
    float* code, unsigned char* A, float* absmax, float* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<float, 512, 64, 8, NF4>(
    float* code, unsigned char* A, float* absmax, float* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, FP4>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, General8bit>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, NF4>(
    float* code, unsigned char* A, float* absmax, __nv_bfloat16* out, const int blocksize, const int n
);
