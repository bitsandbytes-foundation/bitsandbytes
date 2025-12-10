#include "ops.cuh" // For Optimizer_t, CUDA_CHECK_RETURN
#include <cub/cub.cuh>
#include <float.h> // For FLT_MIN/FLT_MAX

// constants for 32bit optimizers
#define TH 1024
#define NUM_PER_THREAD 4

// 8bit blockwise constants
#define LANES 2
#define QUAD 3

// constants for host code
#define BLOCKSIZE_2STATE 256
#define NUM_2STATE 1
#define BLOCKSIZE_1STATE 256
#define NUM_1STATE 1

// from kernels.cu
// TODO move somewhere like common.cuh or cub_utils.cuh etc
#if CCCL_VERSION >= 2008002
#include <cuda/std/functional>
#define CUB_REDUCTIONOP_MAX                                                                                            \
    cuda::maximum<> {}
#else
#define CUB_REDUCTIONOP_MAX cub::Max()
#endif

// sign function for lion
// taken from https://stackoverflow.com/a/4609795, but not sure if there's a proper way to do this in CUDA
template <typename T> __device__ int sgn(T val) { return (T(0) < val) - (val < T(0)); }

/// 32bit, 2-state optimizers ///

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__launch_bounds__(BLOCK_SIZE / NUM_VALS, 1) __global__ void kPreconditionOptimizer32bit2State(
    T* g, T* p, float* state1, float* state2, float* unorm, const float beta1, const float beta2, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const int n
) {

    const int n_full = (BLOCK_SIZE * (n / BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
    const int base_idx = (blockIdx.x * blockDim.x * NUM_VALS);
    int valid_items = 0;

    T g_vals[NUM_VALS];

    float s1_vals[NUM_VALS];
    float s2_vals[NUM_VALS];

    const float correction1 = 1.0f / (1.0f - powf(beta1, step));
    const float correction2 = 1.0f / (1.0f - powf(beta2, step));

    typedef cub::BlockLoad<T, BLOCK_SIZE / NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
    typedef cub::BlockLoad<float, BLOCK_SIZE / NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    typedef cub::BlockReduce<float, BLOCK_SIZE / NUM_VALS> BlockReduce;

    __shared__ union {
        typename Load::TempStorage load;
        typename LoadFloat::TempStorage loadf;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

        __syncthreads();
        Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items, 0.0f);

#pragma unroll NUM_VALS
        for (unsigned int j = 0; j < NUM_VALS; j++)
            g_vals[j] = gnorm_scale * ((float)g_vals[j]);

#pragma unroll NUM_VALS
        for (unsigned int j = 0; j < NUM_VALS; j++) {
            switch (OPTIMIZER) {
            case ADAM:
                s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * ((float)g_vals[j]));
                s2_vals[j] = s2_vals[j] * beta2 + ((1.0f - beta2) * (((float)g_vals[j]) * ((float)g_vals[j])));
                s1_vals[j] *= correction1;
                s2_vals[j] *= correction2;
                s1_vals[j] = s1_vals[j] / (sqrtf(s2_vals[j]) + eps); // update
                s1_vals[j] *= s1_vals[j];                            // update l2 norm (update*update)
                break;
            case ADEMAMIX:
                break;
            }
        }

#pragma unroll NUM_VALS - 1
        for (unsigned int j = 1; j < NUM_VALS; j++)
            s1_vals[0] += s1_vals[j];

        __syncthreads();
        s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0]);

        if (threadIdx.x == 0)
            atomicAdd(&unorm[0], s1_vals[0]);

        __syncwarp();
    }
}

template <typename T, int OPTIMIZER>
__launch_bounds__(TH, 1) __global__ void kOptimizer32bit2State(
    T* g, T* p, float* state1, float* state2, float* unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float beta3, const float alpha, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const bool skip_zeros,
    const int n
) {

    const int n_full = ((TH * NUM_PER_THREAD) * (n / (TH * NUM_PER_THREAD))) +
                       (n % (TH * NUM_PER_THREAD) == 0 ? 0 : (TH * NUM_PER_THREAD));
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items = 0;
    float update_scale = 0.0f;
    T g_vals[NUM_PER_THREAD];
    T p_vals[NUM_PER_THREAD];

    float s1_vals[NUM_PER_THREAD];
    float s2_vals[NUM_PER_THREAD];

    // AdEMAMix has an additional state buffer, which we packed
    // into state1. We need thread-local storage here for these.
    // TODO: Mark with [[maybe_unused]] after upgrade to min compiler.
    float s3_vals[NUM_PER_THREAD];

    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2 = sqrtf(1.0f - powf(beta2, step));
    const float step_size = -lr * correction2 / correction1;

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

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * TH * NUM_PER_THREAD) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;

        __syncthreads();
        Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items);
        __syncthreads();
        Load(temp_storage.load).Load(&(p[i]), p_vals, valid_items);

        // Load additional state1 data for AdEMAMix
        // TODO: Make constexpr after updating min compiler
        if (OPTIMIZER == ADEMAMIX) {
            __syncthreads();
            LoadFloat(temp_storage.loadf).Load(&(state1[n + i]), s3_vals, valid_items);
        }

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD; j++)
            g_vals[j] = gnorm_scale * ((float)g_vals[j]);

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD; j++) {
            switch (OPTIMIZER) {
            case ADEMAMIX:
                // m1 update: m1 = beta1 * m1 + (1-beta1) * g
                s1_vals[j] = (s1_vals[j] * beta1) + ((1.0f - beta1) * (float)g_vals[j]);

                // m2 update: m2 = m2 * beta3 + (1-beta3) * g
                s3_vals[j] = (s3_vals[j] * beta3) + ((1.0f - beta3) * (float)g_vals[j]);

                // nu update: nu = beta2 * nu + (1-beta2) * g^2
                s2_vals[j] = (s2_vals[j] * beta2) + ((1.0f - beta2) * (float)g_vals[j] * (float)g_vals[j]);

                p_vals[j] = (float)p_vals[j] - lr * (((s1_vals[j] / correction1) + (alpha * s3_vals[j])) /
                                                     ((sqrtf(s2_vals[j]) / correction2) + eps));

                if (weight_decay > 0.0f)
                    p_vals[j] = ((float)p_vals[j]) * (1.0f - (lr * weight_decay));

                break;
            case ADAM:

                if (!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f))) {
                    s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * ((float)g_vals[j]));
                    s2_vals[j] = s2_vals[j] * beta2 + ((1.0f - beta2) * (((float)g_vals[j]) * ((float)g_vals[j])));
                    p_vals[j] = ((float)p_vals[j]) +
                                (update_scale * step_size * (s1_vals[j] / (sqrtf(s2_vals[j]) + (eps * correction2))));

                    if (weight_decay > 0.0f)
                        p_vals[j] = ((float)p_vals[j]) * (1.0f - (lr * weight_decay));
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

        if (OPTIMIZER == ADEMAMIX) {
            __syncthreads();
            StoreFloat(temp_storage.storef).Store(&(state1[n + i]), s3_vals, valid_items);
        }
    }
}

/// 32bit, 1-state optimizers ///
template <typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__launch_bounds__(BLOCK_SIZE / NUM_VALS, 1) __global__ void kPreconditionOptimizer32bit1State(
    T* g, T* p, float* state1, float* unorm, const float beta1, const float beta2, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const int n
) {

    const int n_full = (BLOCK_SIZE * (n / BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
    const int base_idx = (blockIdx.x * blockDim.x * NUM_VALS);
    int valid_items = 0;

    T g_vals[NUM_VALS];

    float s1_vals[NUM_VALS];

    typedef cub::BlockLoad<T, BLOCK_SIZE / NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
    typedef cub::BlockLoad<float, BLOCK_SIZE / NUM_VALS, NUM_VALS, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    typedef cub::BlockReduce<float, BLOCK_SIZE / NUM_VALS> BlockReduce;

    __shared__ union {
        typename Load::TempStorage load;
        typename LoadFloat::TempStorage loadf;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

        __syncthreads();
        Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);

#pragma unroll NUM_VALS
        for (unsigned int j = 0; j < NUM_VALS; j++)
            g_vals[j] = gnorm_scale * ((float)g_vals[j]);

#pragma unroll NUM_VALS
        for (unsigned int j = 0; j < NUM_VALS; j++) {
            switch (OPTIMIZER) {
            case MOMENTUM:
                if (step == 1)
                    s1_vals[j] = (float)g_vals[j]; // state update
                else
                    s1_vals[j] = s1_vals[j] * beta1 + ((float)g_vals[j]); // state update
                s1_vals[j] = s1_vals[j] * s1_vals[j];                     // update norm
                break;
            case LION:
                s1_vals[j] = s1_vals[j] * beta2 + ((1.0f - beta2) * (float)g_vals[j]); // state update
                break;
            case RMSPROP:
                s1_vals[j] =
                    s1_vals[j] * beta1 + ((1.0f - beta1) * ((float)g_vals[j]) * ((float)g_vals[j])); // state update
                s1_vals[j] = __fdividef((float)g_vals[j], sqrtf(s1_vals[j]) + eps);                  // update value
                s1_vals[j] = s1_vals[j] * s1_vals[j];                                                // update norm
                break;
            case ADAGRAD:
                s1_vals[j] = s1_vals[j] + ((float)g_vals[j]) * ((float)g_vals[j]);  // state update
                s1_vals[j] = __fdividef((float)g_vals[j], sqrtf(s1_vals[j]) + eps); // update value
                s1_vals[j] = s1_vals[j] * s1_vals[j];                               // update norm
                break;
            }
        }

#pragma unroll
        for (unsigned int j = 1; j < NUM_VALS; j++)
            s1_vals[0] += s1_vals[j];

        __syncthreads();
        s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0], valid_items);

        if (threadIdx.x == 0)
            atomicAdd(&unorm[0], s1_vals[0]);

        __syncwarp();
    }
}

template <typename T, int OPTIMIZER>
__launch_bounds__(TH, 1) __global__ void kOptimizer32bit1State(
    T* g, T* p, float* state1, float* unorm, const float max_unorm, const float param_norm, const float beta1,
    const float beta2, const float eps, const float weight_decay, const int step, const float lr,
    const float gnorm_scale, const bool skip_zeros, const int n
) {

    const int n_full = ((TH * NUM_PER_THREAD) * (n / (TH * NUM_PER_THREAD))) +
                       (n % (TH * NUM_PER_THREAD) == 0 ? 0 : (TH * NUM_PER_THREAD));
    const int base_idx = (blockIdx.x * blockDim.x * NUM_PER_THREAD);
    int valid_items = 0;
    float update_scale = 0.0f;

    if (max_unorm > 0.0f) {
        update_scale = max_unorm > 0.0f ? sqrtf(unorm[0]) : 1.0f;
        if (update_scale > max_unorm * param_norm + eps) {
            update_scale = (max_unorm * param_norm + eps) / update_scale;
        } else {
            update_scale = 1.0f;
        }
    } else {
        update_scale = 1.0f;
    }

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

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * TH * NUM_PER_THREAD) {
        valid_items = n - i >= (TH * NUM_PER_THREAD) ? (TH * NUM_PER_THREAD) : n - i;

        __syncthreads();
        Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items);
        __syncthreads();
        LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items);
        __syncthreads();
        Load(temp_storage.load).Load(&(p[i]), p_vals, valid_items);

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD; j++) {
            g_vals[j] = gnorm_scale * ((float)g_vals[j]);
            if (weight_decay > 0.0f)
                g_vals[j] = (float)g_vals[j] + (((float)p_vals[j]) * weight_decay);
        }

#pragma unroll 4
        for (unsigned int j = 0; j < NUM_PER_THREAD; j++) {
            if (!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f))) {
                switch (OPTIMIZER) {
                case MOMENTUM:
                    if (step == 1)
                        s1_vals[j] = (float)g_vals[j];
                    else
                        s1_vals[j] = s1_vals[j] * beta1 + ((float)g_vals[j]);

                    p_vals[j] = ((float)p_vals[j]) + update_scale * (-lr * (s1_vals[j]));
                    break;
                case LION:
                    p_vals[j] =
                        ((float)p_vals[j]) -
                        update_scale * (lr * sgn(((float)s1_vals[j]) * beta1 + ((1.0f - beta1) * ((float)g_vals[j]))));
                    s1_vals[j] = s1_vals[j] * beta2 + ((1.0f - beta2) * ((float)g_vals[j]));
                    break;
                case RMSPROP:
                    s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * ((float)g_vals[j]) * ((float)g_vals[j]));
                    p_vals[j] = ((float)p_vals[j]) -
                                update_scale * (lr * __fdividef((float)g_vals[j], sqrtf((float)s1_vals[j]) + eps));
                    break;
                case ADAGRAD:
                    s1_vals[j] = s1_vals[j] + ((float)g_vals[j]) * ((float)g_vals[j]);
                    p_vals[j] = ((float)p_vals[j]) - lr * __fdividef((float)g_vals[j], sqrtf((float)s1_vals[j]) + eps);
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

//// 8bit blockwise helper ///
template <int SIGNED>
__device__ __forceinline__ unsigned char
    quantize_2D(float* __restrict__ quadrants, float* __restrict__ const smem_code, float x) {
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
    for (int i = 64; i > 0; i >>= 1) {
        if (x > val) {
            lower_pivot = pivot;
            lower = val;
            pivot += i;
            // val = i == 64 ? quadrants[2] : smem_code[pivot];
            local_pivot += offset;
        } else {
            upper_pivot = pivot;
            upper = val;
            pivot -= i;
            // val = i == 64 ? quadrants[0] : smem_code[pivot];
            local_pivot -= offset;
        }
        val = i >= 64 ? quadrants[local_pivot] : smem_code[pivot];
        offset -= 1;
    }

    if (x > val) {
        midpoint = (upper + val) * 0.5f;
        if (x > midpoint)
            return upper_pivot;
        else
            return pivot;
    } else {
        midpoint = (lower + val) * 0.5f;
        if (x < midpoint)
            return lower_pivot;
        else
            return pivot;
    }
}

///////////// 8bit, 2state, blockwise optimizers /////////////

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3) __global__ void kOptimizerStatic8bit2StateBlockwise(
    T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2, const float beta1, const float beta2,
    const float beta3, const float alpha, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* absmax1, float* absmax2,
    float weight_decay, const float gnorm_scale, const bool skip_zeros, const int n
) {

    // const int n_full = n + (n%BLOCK_SIZE);
    const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = (blockIdx.x * BLOCK_SIZE);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];
    float s3_vals[N_PER_TH];

    // 2-5%
    const float correction1 = 1.0f - __powf(beta1, step);
    const float correction2 = sqrtf(1.0f - __powf(beta2, step));
    const float step_size = __fdividef(-lr * correction2, correction1);
    const int lane_id = threadIdx.x % LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float new_local_abs_max3 = -FLT_MAX;
    float quadrants1[QUAD];
    float quadrants2[QUAD];

    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    unsigned char c3s[N_PER_TH];

    T g_vals[N_PER_TH];
    T p_vals[N_PER_TH];
    typedef cub::BlockLoad<T, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[LANES][257];
    __shared__ float smem_quantiles2[LANES][257];
    typedef cub::BlockReduce<float, BLOCK_SIZE / N_PER_TH> BlockReduce1;
    typedef cub::BlockReduce<float, BLOCK_SIZE / N_PER_TH> BlockReduce2;
    typedef cub::BlockReduce<float, BLOCK_SIZE / N_PER_TH> BlockReduce3;
    __shared__ typename BlockReduce1::TempStorage reduce1;
    __shared__ typename BlockReduce2::TempStorage reduce2;
    __shared__ typename BlockReduce2::TempStorage reduce3;
    __shared__ float smem_exchange1[1];
    __shared__ float smem_exchange2[1];
    __shared__ float smem_exchange3[1]; // [[maybe_unused]]

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
#pragma unroll
    for (unsigned int j = 1; j < LANES; j++) {
        smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];
        smem_quantiles2[j][threadIdx.x] = smem_quantiles2[0][threadIdx.x];
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < QUAD; k++) {
        quadrants1[k] = smem_quantiles1[lane_id][(k * 256 / (QUAD + 1)) + (256 / (QUAD + 1) - 1)];
        quadrants2[k] = smem_quantiles2[lane_id][(k * 256 / (QUAD + 1)) + (256 / (QUAD + 1) - 1)];
    }

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        // loads: 0.23 -> 0.85/1.44
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(g[i]), g_vals, valid_items, (T)0.0f);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state1[i]), c1s, valid_items, 128);
        __syncthreads();
        LoadChar(temp_storage.loadc).Load(&(state2[i]), c2s, valid_items, 0);

        // AdEMAMix has an additional state packed into state1.
        if (OPTIMIZER == ADEMAMIX) {
            __syncthreads();
            LoadChar(temp_storage.loadc).Load(&(state1[n + i]), c3s, valid_items, 128);
        }

        new_local_abs_max1 = -FLT_MAX;
        new_local_abs_max2 = -FLT_MAX;
        new_local_abs_max3 = -FLT_MAX;

//  update: 2.48/1.57 -> 2.51/1.60
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            if (!isnan((float)g_vals[j]) && !isinf((float)g_vals[j])) {
                s2_vals[j] = smem_quantiles2[lane_id][c2s[j]] * absmax2[i / BLOCK_SIZE];
                g_val = g_vals[j];
                // float ratio = (g_val*g_val)/fmaxf(s2_vals[j], eps*eps);
                // g_val = ratio > 2.0f ? 2.0f*g_val/ratio : g_val;
                g_val *= gnorm_scale;

                s2_vals[j] = (s2_vals[j] * beta2) + (((1.0f - beta2) * g_val * g_val));

                s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * absmax1[i / BLOCK_SIZE];
                s1_vals[j] = (s1_vals[j] * beta1) + (((1.0f - beta1) * g_val));

                if (OPTIMIZER == ADEMAMIX) {
                    // The absmax for the third state is appended to absmax1
                    s3_vals[j] = smem_quantiles1[lane_id][c3s[j]] * absmax1[(n + i) / BLOCK_SIZE];
                    s3_vals[j] = (s3_vals[j] * beta3) + (((1.0f - beta3) * g_val));
                }
            } else {
                s1_vals[j] = 0.0f;
                s2_vals[j] = 0.0f;

                if (OPTIMIZER == ADEMAMIX) {
                    s3_vals[j] = 0.0f;
                }
            }

            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));

            if (OPTIMIZER == ADEMAMIX) {
                new_local_abs_max3 = fmaxf(new_local_abs_max3, fabsf(s3_vals[j]));
            }
        }

        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = BlockReduce1(reduce1).Reduce(new_local_abs_max1, CUB_REDUCTIONOP_MAX);
        new_local_abs_max2 = BlockReduce2(reduce2).Reduce(new_local_abs_max2, CUB_REDUCTIONOP_MAX);

        if (OPTIMIZER == ADEMAMIX) {
            new_local_abs_max3 = BlockReduce3(reduce3).Reduce(new_local_abs_max3, CUB_REDUCTIONOP_MAX);
        }

        if (threadIdx.x == 0) {
            smem_exchange1[0] = new_local_abs_max1;
            smem_exchange2[0] = new_local_abs_max2;

            if (OPTIMIZER == ADEMAMIX) {
                smem_exchange3[0] = new_local_abs_max3;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            absmax1[i / BLOCK_SIZE] = new_local_abs_max1;
            absmax2[i / BLOCK_SIZE] = new_local_abs_max2;

            if (OPTIMIZER == ADEMAMIX) {
                absmax1[(n + i) / BLOCK_SIZE] = new_local_abs_max3;
            }
        } else {
            new_local_abs_max1 = smem_exchange1[0];
            new_local_abs_max2 = smem_exchange2[0];

            if (OPTIMIZER == ADEMAMIX) {
                new_local_abs_max3 = smem_exchange3[0];
            }
        }

        __syncthreads();
        LoadT(temp_storage.loadh).Load(&(p[i]), p_vals, valid_items, (T)0.0f);
//  reduce: 2.67/1.69 -> 2.67/1.70
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            // if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
            if (!isnan((float)g_vals[j]) && !isinf((float)g_vals[j])) {
                if (OPTIMIZER == ADEMAMIX) {
                    p_vals[j] =
                        T((float)p_vals[j] - lr * (((s1_vals[j] / correction1) + (alpha * s3_vals[j])) /
                                                   ((sqrtf(s2_vals[j]) / correction2) + eps)));
                } else {
                    p_vals[j] =
                        (T)(((float)p_vals[j]) +
                            ((step_size * (__fdividef(s1_vals[j], (sqrtf(s2_vals[j]) + (correction2 * eps)))))));
                }

                if (weight_decay > 0.0f)
                    p_vals[j] = ((float)p_vals[j]) * (1.0f - (lr * weight_decay));
            }
        }

        //  store: 0.85/1.44 -> 2.48/1.57
        __syncthreads();
        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);

//  quantizaztion: 2.67/1.70  -> 3.4/3.3
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], __fdividef(s1_vals[j], new_local_abs_max1));
            c2s[j] = quantize_2D<0>(quadrants2, smem_quantiles2[lane_id], __fdividef(s2_vals[j], new_local_abs_max2));

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if (signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f)
                    c1s[j] += 1;
                else
                    c1s[j] -= 1;
            }

            if (OPTIMIZER == ADEMAMIX) {
                c3s[j] =
                    quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], __fdividef(s3_vals[j], new_local_abs_max3));

                if (signbit(smem_quantiles1[lane_id][c3s[j]]) != signbit(s3_vals[j])) {
                    c3s[j] += (s3_vals[j] > 0.0f) ? 1 : -1;
                }
            }
        }

        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state2[i]), c2s, valid_items);

        if (OPTIMIZER == ADEMAMIX) {
            __syncthreads();
            StoreChar(temp_storage.storec).Store(&(state1[n + i]), c3s, valid_items);
        }
    }
}

///////////// 8bit, 1state, blockwise optimizers /////////////
template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3) __global__ void kOptimizerStatic8bit1StateBlockwise(
    T* p, T* __restrict__ const g, unsigned char* state1, const float beta1, const float beta2, const float eps,
    const int step, const float lr, float* __restrict__ const quantiles1, float* absmax1, float weight_decay,
    const float gnorm_scale, const bool skip_zeros, const int n
) {

    // const int n_full = n + (n%BLOCK_SIZE);
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

    typedef cub::BlockLoad<T, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    typedef cub::BlockStore<unsigned char, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockStore<T, BLOCK_SIZE / N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ float smem_quantiles1[LANES][257];
    typedef cub::BlockReduce<float, BLOCK_SIZE / N_PER_TH> BlockReduce1;
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
#pragma unroll
    for (unsigned int j = 1; j < LANES; j++)
        smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];

    __syncthreads();

#pragma unroll
    for (int k = 0; k < QUAD; k++)
        quadrants1[k] = smem_quantiles1[lane_id][(k * 256 / (QUAD + 1)) + (256 / (QUAD + 1) - 1)];

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
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
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            if (!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f))) {
                if (weight_decay > 0.0f) {
                    switch (OPTIMIZER) {
                    case MOMENTUM:
                    case ADAGRAD:
                    case RMSPROP:
                        g_val += ((float)p_vals[j]) * weight_decay;
                        break;
                    case LION:
                        p_vals[j] = ((float)p_vals[j]) * (1.0f - lr * weight_decay);
                        break;
                    }
                }

                s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * absmax1[i / BLOCK_SIZE];

                switch (OPTIMIZER) {
                case MOMENTUM:
                    if (step == 1)
                        s1_vals[j] = g_val;
                    else
                        s1_vals[j] = (s1_vals[j] * beta1) + g_val;
                    break;
                case LION:
                    // here, using gvals[j] to store the gradient smoothed by beta1 for the following parameter update,
                    // before the momentum is updated by beta2
                    g_vals[j] = lr * sgn(((float)s1_vals[j]) * beta1 + ((1.0f - beta1) * g_val));
                    s1_vals[j] = s1_vals[j] * beta2 + ((1.0f - beta2) * g_val);
                    break;
                case RMSPROP:
                    s1_vals[j] = s1_vals[j] * beta1 + ((1.0f - beta1) * (g_val * g_val));
                    break;
                case ADAGRAD:
                    s1_vals[j] = s1_vals[j] + (g_val * g_val);
                    break;
                }
            }

            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
        }

        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = BlockReduce1(reduce1).Reduce(new_local_abs_max1, CUB_REDUCTIONOP_MAX);

        if (threadIdx.x == 0)
            smem_exchange1[0] = new_local_abs_max1;

        __syncthreads();

        if (threadIdx.x == 0)
            absmax1[i / BLOCK_SIZE] = new_local_abs_max1;
        else
            new_local_abs_max1 = smem_exchange1[0];

//  reduce: 2.67/1.69 -> 2.67/1.70
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            if (!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f))) {
                switch (OPTIMIZER) {
                case MOMENTUM:
                    p_vals[j] = ((float)p_vals[j]) - lr * (s1_vals[j]);
                    break;
                case LION:
                    p_vals[j] = ((float)p_vals[j]) - ((float)g_vals[j]);
                    break;
                case RMSPROP:
                    g_val = g_vals[j];
                    p_vals[j] = ((float)p_vals[j]) - lr * (__fdividef(g_val, sqrtf(s1_vals[j]) + eps));
                    break;
                case ADAGRAD:
                    g_val = g_vals[j];
                    p_vals[j] = ((float)p_vals[j]) - lr * (__fdividef(g_val, sqrtf(s1_vals[j]) + eps));
                    break;
                }
            }
        }

        //  store: 0.85/1.44 -> 2.48/1.57
        __syncthreads();
        StoreT(temp_storage.storeh).Store(&(p[i]), p_vals, valid_items);

//  quantizaztion: 2.67/1.70  -> 3.4/3.3
#pragma unroll N_PER_TH
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], __fdividef(s1_vals[j], new_local_abs_max1));

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if (signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f)
                    c1s[j] += 1;
                else
                    c1s[j] -= 1;
            }
        }

        __syncthreads();
        StoreChar(temp_storage.storec).Store(&(state1[i]), c1s, valid_items);
    }
}

/// launch code
template <typename T, int OPTIMIZER>
void optimizer32bit(
    T* g, T* p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm, const float beta1,
    const float beta2, const float beta3, const float alpha, const float eps, const float weight_decay, const int step,
    const float lr, const float gnorm_scale, bool skip_zeros, const int n
) {
    int num_blocks = n / 4096;
    num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
    switch (OPTIMIZER) {
    case ADAM:
    case ADEMAMIX:
        if (max_unorm > 0.0f) {
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(
                g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n
            );
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }
        kOptimizer32bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, beta3, alpha, eps, weight_decay, step, lr,
            gnorm_scale, skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
        if (max_unorm > 0.0f) {
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }

        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case LION:
        // in lion, the momentum update after the parameter update
        kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(
            g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale,
            skip_zeros, n
        );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        if (max_unorm > 0.0f) {
            CUDA_CHECK_RETURN(cudaMemset(unorm, 0, 1 * sizeof(float)));
            kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>
                <<<num_blocks, 512>>>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
        }
        break;
    }
}

template <typename T, int OPTIMIZER>
void optimizerStatic8bitBlockwise(
    T* p, T* g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3, float alpha,
    float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1, float* absmax2,
    float weight_decay, const float gnorm_scale, bool skip_zeros, int n
) {

    int num_blocks = 0;
    switch (OPTIMIZER) {
    case ADAM:
    case ADEMAMIX:
        num_blocks = n / BLOCKSIZE_2STATE;
        num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
        kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>
            <<<num_blocks, BLOCKSIZE_2STATE / NUM_2STATE>>>(
                p, g, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, quantiles1, quantiles2, absmax1,
                absmax2, weight_decay, gnorm_scale, skip_zeros, n
            );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
    case LION:
        num_blocks = n / BLOCKSIZE_1STATE;
        num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
        kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>
            <<<num_blocks, BLOCKSIZE_1STATE / NUM_1STATE>>>(
                p, g, state1, beta1, beta2, eps, step, lr, quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n
            );
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        break;
    }
}

// launch templates.
// TODO: we could just do the C API right here?

#define MAKE_optimizer32bit(name, gtype)                                                                               \
    template void optimizer32bit<gtype, name>(                                                                         \
        gtype * g, gtype * p, float* state1, float* state2, float* unorm, float max_unorm, float param_norm,           \
        const float beta1, const float beta2, const float beta3, const float alpha, const float eps,                   \
        const float weight_decay, const int step, const float lr, const float gnorm_scale, const bool skip_zeros,      \
        const int n                                                                                                    \
    );

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(ADAM, __nv_bfloat16)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(MOMENTUM, __nv_bfloat16)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(RMSPROP, __nv_bfloat16)
MAKE_optimizer32bit(LION, half)
MAKE_optimizer32bit(LION, float)
MAKE_optimizer32bit(LION, __nv_bfloat16)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)
MAKE_optimizer32bit(ADAGRAD, __nv_bfloat16)
MAKE_optimizer32bit(ADEMAMIX, half)
MAKE_optimizer32bit(ADEMAMIX, __nv_bfloat16)
MAKE_optimizer32bit(ADEMAMIX, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name)                                                           \
    template void optimizerStatic8bitBlockwise<gtype, optim_name>(                                                     \
        gtype * p, gtype * g, unsigned char* state1, unsigned char* state2, float beta1, float beta2, float beta3,     \
        float alpha, float eps, int step, float lr, float* quantiles1, float* quantiles2, float* absmax1,              \
        float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n                            \
    );

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, LION);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(half, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADEMAMIX);
MAKE_optimizerStatic8bitBlockwise(float, ADEMAMIX);
