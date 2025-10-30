#include <float.h>
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float nf4_dequant_lut[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

inline uchar quantize_nf4_scalar(float x) {
    if (x > 0.03979014977812767f) {
        if (x > 0.3893125355243683f) {
            if (x > 0.6427869200706482f) {
                if (x > 0.8614784181118011f) {
                    return 0b1111;
                } else {
                    return 0b1110;
                }
            } else {
                if (x > 0.5016634166240692f) {
                    return 0b1101;
                } else {
                    return 0b1100;
                }
            }
        } else {
            if (x > 0.2035212516784668f) {
                if (x > 0.2920137718319893f) {
                    return 0b1011;
                } else {
                    return 0b1010;
                }
            } else {
                if (x > 0.1202552504837513f) {
                    return 0b1001;
                } else {
                    return 0b1000;
                }
            }
        }
    } else {
        if (x > -0.33967943489551544f) {
            if (x > -0.13791173323988914f) {
                if (x > -0.045525018125772476f) {
                    return 0b0111;
                } else {
                    return 0b0110;
                }
            } else {
                if (x > -0.23460740596055984f) {
                    return 0b0101;
                } else {
                    return 0b0100;
                }
            }
        } else {
            if (x > -0.6106329262256622f) {
                if (x > -0.4599952697753906f) {
                    return 0b0011;
                } else {
                    return 0b0010;
                }
            } else {
                if (x > -0.8480964004993439f) {
                    return 0b0001;
                } else {
                    return 0b0000;
                }
            }
        }
    }
}

inline float dequantize_nf4_scalar(uchar code) {
    return nf4_dequant_lut[code & 0x0F];
}

template <typename T>
inline float load_value(const device T* src, uint index) {
    return static_cast<float>(src[index]);
}

template <>
inline float load_value<half>(const device half* src, uint index) {
    return static_cast<float>(src[index]);
}

template <typename T>
inline void store_value(device T* dst, uint index, float value) {
    dst[index] = static_cast<T>(value);
}

template <>
inline void store_value<half>(device half* dst, uint index, float value) {
    dst[index] = static_cast<half>(value);
}

struct BlockParams {
    uint n;
    uint blocksize;
    uint threads_per_group;
};

template <typename T>
inline void quantize_nf4_impl(
    const device T* input,
    device float* absmax,
    device uchar* output,
    constant BlockParams& params,
    threadgroup float* shared_vals,
    threadgroup float& shared_scale,
    threadgroup float& shared_absmax,
    uint tid,
    uint threads_per_group,
    uint block_idx) {

    const uint block_start = block_idx * params.blocksize;
    if (block_start >= params.n) {
        return;
    }

    const uint block_end = min(block_start + params.blocksize, params.n);
    const uint block_length = block_end - block_start;

    float local_max = 0.0f;
    for (uint idx = tid; idx < block_length; idx += threads_per_group) {
        const uint global_idx = block_start + idx;
        const float value = fabs(load_value(input, global_idx));
        local_max = fmax(local_max, value);
    }

    shared_vals[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_vals[tid] = fmax(shared_vals[tid], shared_vals[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        shared_absmax = fmax(shared_vals[0], 0.0f);
        shared_scale = shared_absmax > 0.0f ? 1.0f / shared_absmax : 0.0f;
        absmax[block_idx] = shared_absmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = shared_scale;
    const uint pair_stride = threads_per_group * 2;
    for (uint local_idx = tid * 2; local_idx < block_length; local_idx += pair_stride) {
        const uint global_idx = block_start + local_idx;
        const float v0 = load_value(input, global_idx) * scale;
        float v1 = 0.0f;
        if (local_idx + 1 < block_length) {
            v1 = load_value(input, global_idx + 1) * scale;
        }
        uchar packed = quantize_nf4_scalar(v0) << 4;
        packed |= quantize_nf4_scalar(v1);
        const uint pair_index = global_idx >> 1;
        output[pair_index] = packed;
    }
}

template <typename T>
inline void dequantize_nf4_impl(
    const device uchar* input,
    const device float* absmax,
    device T* output,
    constant BlockParams& params,
    uint tid,
    uint threads_per_group,
    uint block_idx) {

    const uint block_start = block_idx * params.blocksize;
    if (block_start >= params.n) {
        return;
    }

    const uint block_end = min(block_start + params.blocksize, params.n);
    const uint block_length = block_end - block_start;
    const float block_absmax = absmax[block_idx];

    const uint total_pairs = (params.n + 1) >> 1;
    const uint pair_start = block_start >> 1;

    for (uint local_pair = tid; local_pair < ((block_length + 1) >> 1); local_pair += threads_per_group) {
        const uint pair_index = pair_start + local_pair;
        if (pair_index >= total_pairs) {
            continue;
        }

        const uchar packed = input[pair_index];
        const float v0 = dequantize_nf4_scalar(packed >> 4) * block_absmax;
        const float v1 = dequantize_nf4_scalar(packed & 0x0F) * block_absmax;

        const uint elem0 = block_start + local_pair * 2;
        const uint elem1 = elem0 + 1;

        if (elem0 < params.n) {
            store_value(output, elem0, v0);
        }
        if (elem1 < params.n) {
            store_value(output, elem1, v1);
        }
    }
}

kernel void quantize_nf4_f16(
    const device half* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* output [[buffer(2)]],
    constant BlockParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]) {
    threadgroup float shared_vals[64];
    threadgroup float shared_scale;
    threadgroup float shared_absmax;
    quantize_nf4_impl<half>(
        input,
        absmax,
        output,
        params,
        shared_vals,
        shared_scale,
        shared_absmax,
        tid,
        tg_size.x,
        tg_pos.x);
}

kernel void quantize_nf4_f32(
    const device float* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* output [[buffer(2)]],
    constant BlockParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]) {
    threadgroup float shared_vals[64];
    threadgroup float shared_scale;
    threadgroup float shared_absmax;
    quantize_nf4_impl<float>(
        input,
        absmax,
        output,
        params,
        shared_vals,
        shared_scale,
        shared_absmax,
        tid,
        tg_size.x,
        tg_pos.x);
}

kernel void dequantize_nf4_f16(
    const device uchar* input [[buffer(0)]],
    const device float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant BlockParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]) {
    dequantize_nf4_impl<half>(input, absmax, output, params, tid, tg_size.x, tg_pos.x);
}

kernel void dequantize_nf4_f32(
    const device uchar* input [[buffer(0)]],
    const device float* absmax [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BlockParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]) {
    dequantize_nf4_impl<float>(input, absmax, output, params, tid, tg_size.x, tg_pos.x);
}
