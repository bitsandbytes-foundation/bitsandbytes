// kernels_unified.cu — EXAMPLE of merged CUDA/HIP kernel source
//
// This file demonstrates how kernels.cu and kernels.hip can be unified
// into a single source file. It shows representative kernels covering
// all categories of differences:
//
//   1. Shared code (identical on both platforms) — kQuantize, kQuantizeBlockwise
//   2. Platform-specific atomics — atomicMax (CUDA needs custom, HIP has native)
//   3. Warp-size-dependent kernels — kQuantizeBlockwiseSmall (replaces
//      kQuantizeBlockwise32 on CUDA and kQuantizeBlockwise64 on HIP)
//   4. Template instantiations — bnb_bfloat16 alias for __nv_bfloat16 / hip_bfloat16
//
// Key principles:
//   - Include "compat.cuh" for all platform abstractions
//   - Use bnb_cub:: instead of cub:: or hipcub::
//   - Use BNB_MAX_OP / BNB_SUM_OP instead of cub::Max() / hipcub::Max()
//   - Use bnb_bfloat16 instead of __nv_bfloat16 / hip_bfloat16
//   - Use #if BNB_HIP for truly divergent sections
//   - <<<grid, block>>> syntax works on both platforms (HIP supports it natively)
//
// This file compiles as:
//   - CUDA:  nvcc compiles it as .cu (default)
//   - HIP:   CMake sets LANGUAGE HIP on this .cu file, hipcc compiles it
//
// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "common.cuh" // merged common_unified.cuh in the real version
#include "compat.cuh"
#include "kernels.cuh" // merged kernel declarations
#include <common.h>    // DataType_t enum

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

// ============================================================================
// Lookup tables — identical on both platforms
// ============================================================================

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

// ============================================================================
// atomicMax for float — CUDA needs a custom CAS loop, HIP has native support
// ============================================================================

#if !BNB_HIP
// CUDA: no native atomicMax for float, use CAS loop
// source: https://stackoverflow.com/questions/17399119
__device__ float atomicMax(float* address, float val) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(reinterpret_cast<int*>(address), assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
#endif
// HIP: atomicMax for float is available natively in ROCm — no custom impl needed

// ============================================================================
// Device helper functions — identical on both platforms
// ============================================================================

__device__ __forceinline__ float dDequantizeFP4Tree(unsigned char val) {
    float sign = 1.0f - 2 * ((val & 0b1000) >> 3);
    return fp4_dequantization_lut[val & 0b111] * sign;
}

__device__ unsigned char dQuantizeFP4(float x) {
    int sign = x < 0 ? 0b1000 : 0b0000;
    x = fabsf(x);
    if (x > 0.29166667f)
        if (x > 0.583333f)
            if (x > 0.8333333f)
                return 0b0011 + sign;
            else
                return 0b0010 + sign;
        else if (x > 0.4166667f)
            return 0b101 + sign;
        else
            return 0b100 + sign;
    else if (x > 0.0859375f)
        if (x > 0.20833333f)
            return 0b0111 + sign;
        else
            return 0b0110 + sign;
    else if (x > 0.00260417f)
        return 0b0001 + sign;
    else
        return 0b0000 + sign;
}

__device__ __forceinline__ float dDequantizeNF4(unsigned char val) { return nf4_dequantization_lut[val & 0x0F]; }

__device__ unsigned char dQuantizeNF4(float x) {
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)
            if (x > 0.6427869200706482f)
                if (x > 0.8614784181118011f)
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f)
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f)
            if (x > 0.2920137718319893f)
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f)
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)
        if (x > -0.13791173323988914f)
            if (x > -0.045525018125772476f)
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f)
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f)
        if (x > -0.4599952697753906f)
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f)
        return 0b0001;
    else
        return 0b0000;
}

// (dQuantize<> helper omitted for brevity — same pattern, no platform diffs)
template <int STOCHASTIC> __device__ unsigned char dQuantize(float* smem_code, float rand, float x) {
    // Binary search in quantization code — identical on both platforms
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float val = smem_code[pivot];
    for (int i = 64; i > 0; i >>= 1) {
        if (x > val) {
            lower_pivot = pivot;
            pivot += i;
        } else {
            upper_pivot = pivot;
            pivot -= i;
        }
        val = smem_code[pivot];
    }

    if (upper_pivot == 255)
        upper_pivot = 254;

    if (STOCHASTIC) {
        if (rand >= (x - smem_code[lower_pivot]) / (smem_code[upper_pivot] - smem_code[lower_pivot]))
            return lower_pivot;
        else
            return upper_pivot;
    } else {
        if (fabsf(x - smem_code[lower_pivot]) < fabsf(x - smem_code[upper_pivot]))
            return lower_pivot;
        else
            return upper_pivot;
    }
}

// ============================================================================
// kQuantize — fully shared, zero #ifdefs needed
//
// Before (CUDA):  typedef cub::BlockLoad<...>
// Before (HIP):   typedef hipcub::BlockLoad<...>
// After (unified): typedef bnb_cub::BlockLoad<...>
// ============================================================================

__launch_bounds__(TH, 4) __global__
    void kQuantize(float* code, float* __restrict__ const A, unsigned char* out, const int n) {
    const int n_full = (NUM_BLOCK * (n / NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
    int valid_items = (blockIdx.x + 1 == gridDim.x) ? n - (blockIdx.x * NUM_BLOCK) : NUM_BLOCK;
    const int base_idx = (blockIdx.x * NUM_BLOCK);

    float vals[NUM];
    unsigned char qvals[NUM];

    //            vvvvvvvv  unified namespace alias — resolves to cub:: or hipcub::
    typedef bnb_cub::BlockLoad<float, TH, NUM, bnb_cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    typedef bnb_cub::BlockStore<unsigned char, TH, NUM, bnb_cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

    __shared__ typename LoadFloat::TempStorage loadf;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ float smem_code[256];

    if (threadIdx.x < 256)
        smem_code[threadIdx.x] = code[threadIdx.x];

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * NUM_BLOCK) {
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

// ============================================================================
// kQuantizeBlockwise — fully shared, uses BNB_MAX_OP
//
// The only change vs the original CUDA version:
//   cub::  →  bnb_cub::
//   CUB_REDUCTIONOP_MAX  →  BNB_MAX_OP
// ============================================================================

template <typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE>
__global__ void kQuantizeBlockwise(
    float* code, T* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,
    const int rand_offset, const int n
) {
    const int n_full = min(gridDim.x * BLOCK_SIZE, INT32_MAX);
    const int base_idx = blockIdx.x * BLOCK_SIZE;
    int valid_items = 0;

    T vals[NUM_PER_TH];
    float rand_vals[NUM_PER_TH];
    unsigned char qvals[(DATA_TYPE > 0) ? NUM_PER_TH / 2 : NUM_PER_TH];

    float local_abs_max = 0.0f;
    int local_rand_idx = 0;

    typedef bnb_cub::BlockLoad<T, BLOCK_SIZE / NUM_PER_TH, NUM_PER_TH, bnb_cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef bnb_cub::BlockStore<
        unsigned char, BLOCK_SIZE / NUM_PER_TH, (DATA_TYPE > 0) ? NUM_PER_TH / 2 : NUM_PER_TH,
        bnb_cub::BLOCK_STORE_WARP_TRANSPOSE>
        StoreChar;
    typedef bnb_cub::BlockReduce<float, BLOCK_SIZE / NUM_PER_TH> BlockReduce;
    typedef bnb_cub::BlockLoad<float, BLOCK_SIZE / NUM_PER_TH, NUM_PER_TH, bnb_cub::BLOCK_LOAD_WARP_TRANSPOSE>
        LoadFloat;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename LoadFloat::TempStorage loadf;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename BlockReduce::TempStorage reduce;
    __shared__ float smem_code[256];
    __shared__ float smem_absmax_value[1];

    if (DATA_TYPE == General8bit)
        for (int i = threadIdx.x; i < 256; i += blockDim.x)
            smem_code[i] = code[i];

    for (int64_t i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = min(BLOCK_SIZE, static_cast<int>(n - i));
        local_abs_max = -FLT_MAX;

        __syncthreads();
        LoadT(loadt).Load(&(A[i]), vals, valid_items, (T)0.0f);

#pragma unroll NUM_PER_TH
        for (int j = 0; j < NUM_PER_TH; j++)
            local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

        //                                         vvvvvvvvvv  unified reduction op
        local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, BNB_MAX_OP, valid_items);

        if (threadIdx.x == 0) {
            smem_absmax_value[0] = 1.0f / local_abs_max;
            absmax[i / BLOCK_SIZE] = local_abs_max;
        }
        __syncthreads();

        local_abs_max = smem_absmax_value[0];

        if (STOCHASTIC) {
            local_rand_idx = ((blockIdx.x * NUM_BLOCK) + (threadIdx.x * NUM) + rand_offset) % (1024 - 4);
            LoadFloat(loadf).Load(&rand[local_rand_idx], rand_vals, BLOCK_SIZE, 0);
        }

        switch (DATA_TYPE) {
        case General8bit:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                if (!STOCHASTIC)
                    qvals[j] = dQuantize<0>(smem_code, 0.0f, ((float)vals[j]) * local_abs_max);
                else
                    qvals[j] = dQuantize<1>(smem_code, rand_vals[j], ((float)vals[j]) * local_abs_max);
            }
            break;
        case FP4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH / 2; j++) {
                qvals[j] = dQuantizeFP4(((float)vals[2 * j]) * local_abs_max) << 4;
                qvals[j] |= dQuantizeFP4(((float)vals[2 * j + 1]) * local_abs_max);
            }
            break;
        case NF4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH / 2; j++) {
                qvals[j] = dQuantizeNF4(((float)vals[2 * j]) * local_abs_max) << 4;
                qvals[j] |= dQuantizeNF4(((float)vals[2 * j + 1]) * local_abs_max);
            }
            break;
        }

        __syncthreads();
        StoreChar(storec).Store(
            &(out[(DATA_TYPE > 0) ? i / 2 : i]), qvals, (DATA_TYPE > 0) ? (valid_items + 1) / 2 : valid_items
        );
    }
}

// ============================================================================
// kQuantizeBlockwiseSmall — unified warp-size-dependent kernel
//
// This replaces:
//   CUDA: kQuantizeBlockwise32  (32 threads, blocksize=32, WarpReduce<float,16>)
//   HIP:  kQuantizeBlockwise64  (64 threads, blocksize=64, WarpReduce<float,32>)
//
// Strategy: Use BNB_WARP_SIZE to derive all constants at compile time.
// On CUDA (warp=32): SMALL_BLOCK_SIZE=32, THREADS=32, THREADS_PER_BLOCK=16
// On HIP  (warp=64): SMALL_BLOCK_SIZE=64, THREADS=64, THREADS_PER_BLOCK=32
// On HIP  (warp=32): SMALL_BLOCK_SIZE=32, THREADS=32, THREADS_PER_BLOCK=16
//
// The algorithm is identical — only the numeric constants change.
// ============================================================================

template <typename T, int DATA_TYPE>
__global__ void kQuantizeBlockwiseSmall(
    float* code, T* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,
    const int rand_offset, const int n
) {
    // All constants derived from BNB_WARP_SIZE — no #ifdefs needed!
    constexpr int BLOCK_SIZE = BNB_WARP_SIZE; // 32 on CUDA, 32 or 64 on HIP
    constexpr int NUM_PER_TH = 2;
    constexpr int THREADS = BNB_WARP_SIZE;               // One full hardware warp
    constexpr int THREADS_PER_BLOCK = BNB_WARP_SIZE / 2; // Half-warp per quantization block

    const int base_idx = blockIdx.x * BLOCK_SIZE * 2; // 2 quantization blocks per thread block

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH / 2];
    float local_abs_max = 0.0f;

    const int block_id = threadIdx.x / THREADS_PER_BLOCK;
    const int local_thread_id = threadIdx.x % THREADS_PER_BLOCK;

    typedef bnb_cub::BlockLoad<T, THREADS, NUM_PER_TH, bnb_cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef bnb_cub::BlockStore<unsigned char, THREADS, NUM_PER_TH / 2, bnb_cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    // Logical warp of THREADS_PER_BLOCK: on warp32 HW this is a half-warp,
    // on warp64 HW this splits the single HW warp into two logical warps
    typedef bnb_cub::WarpReduce<float, THREADS_PER_BLOCK> WarpReduce;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename WarpReduce::TempStorage warp_reduce[2];
    __shared__ float smem_absmax_value[2];

    const int i = base_idx + block_id * BLOCK_SIZE;
    const bool block_valid = (i < n);

    __syncthreads();
    LoadT(loadt).Load(&(A[base_idx]), vals, min(BLOCK_SIZE * 2, n - base_idx), (T)0.0f);

    local_abs_max = -FLT_MAX;
#pragma unroll NUM_PER_TH
    for (int j = 0; j < NUM_PER_TH; j++)
        local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = WarpReduce(warp_reduce[block_id]).Reduce(local_abs_max, BNB_MAX_OP);

    if (local_thread_id == 0) {
        if (block_valid) {
            smem_absmax_value[block_id] = 1.0f / local_abs_max;
            absmax[blockIdx.x * 2 + block_id] = local_abs_max;
        } else {
            smem_absmax_value[block_id] = 0.0f;
        }
    }
    __syncthreads();

    local_abs_max = smem_absmax_value[block_id];

    switch (DATA_TYPE) {
    case FP4:
#pragma unroll NUM_PER_TH
        for (int j = 0; j < NUM_PER_TH / 2; j++) {
            qvals[j] = dQuantizeFP4(((float)vals[2 * j]) * local_abs_max) << 4;
            qvals[j] |= dQuantizeFP4(((float)vals[2 * j + 1]) * local_abs_max);
        }
        break;
    case NF4:
#pragma unroll NUM_PER_TH
        for (int j = 0; j < NUM_PER_TH / 2; j++) {
            qvals[j] = dQuantizeNF4(((float)vals[2 * j]) * local_abs_max) << 4;
            qvals[j] |= dQuantizeNF4(((float)vals[2 * j + 1]) * local_abs_max);
        }
        break;
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[base_idx / 2]), qvals, min((BLOCK_SIZE * 2 + 1) / 2, (n - base_idx + 1) / 2));
}

// ============================================================================
// kDequantizeBlockwise — fully shared
// ============================================================================

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

    typedef bnb_cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, bnb_cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef bnb_cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), bnb_cub::BLOCK_STORE_WARP_TRANSPOSE>
        StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    for (int i = base_idx; i < n_load; i += gridDim.x * TILE_SIZE) {
        if (DATA_TYPE > 0) {
            valid_items_load = min(TILE_SIZE, static_cast<int>((static_cast<int64_t>(n) + 1) / 2) - i);
            valid_items_store = min(TILE_SIZE * 2, n - i * 2);
        } else {
            valid_items_load = min(TILE_SIZE, n - i);
            valid_items_store = valid_items_load;
        }

        // blocksize is always power-of-2: use bitwise AND instead of division
        __syncthreads();
        LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load);

        switch (DATA_TYPE) {
        case General8bit:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                local_abs_max = absmax[(i + (threadIdx.x * NUM_PER_TH) + j) / blocksize];
                vals[j] = (T)(code[qvals[j]] * local_abs_max);
            }
            break;
        case FP4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                local_abs_max = absmax[((i * 2) + (threadIdx.x * NUM_PER_TH * 2) + (j * 2)) / blocksize];
                vals[j * 2] = (T)(dDequantizeFP4Tree(qvals[j] >> 4) * local_abs_max);
                vals[j * 2 + 1] = (T)(dDequantizeFP4Tree(qvals[j] & 0x0F) * local_abs_max);
            }
            break;
        case NF4:
#pragma unroll NUM_PER_TH
            for (int j = 0; j < NUM_PER_TH; j++) {
                local_abs_max = absmax[((i * 2) + (threadIdx.x * NUM_PER_TH * 2) + (j * 2)) / blocksize];
                vals[j * 2] = (T)(dDequantizeNF4(qvals[j] >> 4) * local_abs_max);
                vals[j * 2 + 1] = (T)(dDequantizeNF4(qvals[j] & 0x0F) * local_abs_max);
            }
            break;
        }

        __syncthreads();
        StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i * 2 : i]), vals, valid_items_store);
    }
}

// ============================================================================
// Template instantiations — bnb_bfloat16 replaces __nv_bfloat16 / hip_bfloat16
// ============================================================================

#define MAKE_kQuantizeBlockwise(dtype, block_size, num_per_th, stochastic, data_type_name)                             \
    template __global__ void kQuantizeBlockwise<dtype, block_size, num_per_th, stochastic, data_type_name>(            \
        float* code, dtype* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,   \
        const int rand_offset, const int n                                                                             \
    );

// half instantiations
MAKE_kQuantizeBlockwise(half, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(half, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(half, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(half, 2048, 4, 0, FP4)
// ... (remaining half/float instantiations identical to current)

// float instantiations
MAKE_kQuantizeBlockwise(float, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, General8bit)
// ... (remaining float instantiations)

// bnb_bfloat16 — resolves to __nv_bfloat16 on CUDA, hip_bfloat16 on HIP
MAKE_kQuantizeBlockwise(bnb_bfloat16, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 64, 2, 0, FP4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 4096, 4, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 2048, 4, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 1024, 4, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 512, 2, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 256, 2, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 128, 2, 0, NF4)
MAKE_kQuantizeBlockwise(bnb_bfloat16, 64, 2, 0, NF4)

// Unified small-blocksize kernel instantiations
#define MAKE_kQuantizeBlockwiseSmall(dtype, data_type_name)                                                            \
    template __global__ void kQuantizeBlockwiseSmall<dtype, data_type_name>(                                           \
        float* code, dtype* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,   \
        const int rand_offset, const int n                                                                             \
    );

MAKE_kQuantizeBlockwiseSmall(half, FP4) MAKE_kQuantizeBlockwiseSmall(float, FP4) MAKE_kQuantizeBlockwiseSmall(
    bnb_bfloat16, FP4
) MAKE_kQuantizeBlockwiseSmall(half, NF4) MAKE_kQuantizeBlockwiseSmall(float, NF4) MAKE_kQuantizeBlockwiseSmall(bnb_bfloat16, NF4)

    // Dequantize instantiations
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
template __global__ void kDequantizeBlockwise<bnb_bfloat16, 512, 64, 8, FP4>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<bnb_bfloat16, 512, 64, 8, General8bit>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, const int blocksize, const int n
);
template __global__ void kDequantizeBlockwise<bnb_bfloat16, 512, 64, 8, NF4>(
    float* code, unsigned char* A, float* absmax, bnb_bfloat16* out, const int blocksize, const int n
);
