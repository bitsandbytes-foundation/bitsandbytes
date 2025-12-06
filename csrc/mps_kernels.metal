#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

namespace {

constant float NF4_CODE[16] = {
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
};

constant float FP4_CODE[16] = {
    0.0, 0.0052, 0.6667, 1.0, 0.3333, 0.5, 0.1667, 0.25,
    0.0, -0.0052, -0.6667, -1.0, -0.3333, -0.5, -0.1667, -0.25
};

template <typename scalar_t>
inline uchar encode_value(float value, constant float* code_table) {
    float best = fabs(value - code_table[0]);
    uchar index = 0;
    for (uchar i = 1; i < 16; ++i) {
        float diff = fabs(value - code_table[i]);
        if (diff < best) {
            best = diff;
            index = i;
        }
    }
    return index;
}

template <typename scalar_t>
inline void quantize_block(
    device const scalar_t* input,
    device float* absmax,
    device uchar* packed,
    uint n,
    uint blocksize,
    uint block_index,
    constant float* code_table
) {
    uint start = block_index * blocksize;
    if (start >= n) {
        return;
    }

    uint end = min(start + blocksize, n);
    float max_val = 0.0f;
    for (uint i = start; i < end; ++i) {
        float current = fabs((float)input[i]);
        max_val = max(max_val, current);
    }

    absmax[block_index] = max_val;
    float inv = max_val > 0.0f ? 1.0f / max_val : 0.0f;

    uint out_byte = start >> 1;
    bool has_pending = false;
    uchar pending = 0;

    for (uint i = start; i < end; ++i) {
        float normalized = (max_val > 0.0f) ? clamp((float)input[i] * inv, -1.0f, 1.0f) : 0.0f;
        uchar q = encode_value<scalar_t>(normalized, code_table) & 0xF;

        if (!has_pending) {
            pending = q << 4;
            has_pending = true;
            if (i == end - 1) {
                packed[out_byte++] = pending;
                has_pending = false;
            }
        } else {
            packed[out_byte++] = pending | q;
            has_pending = false;
        }
    }
}

template <typename scalar_t>
inline void dequantize_block(
    device const uchar* packed,
    device const float* absmax,
    device scalar_t* output,
    uint n,
    uint blocksize,
    uint block_index,
    uint thread_idx,
    uint threadgroup_size,
    constant float* code_table,
    threadgroup float& shared_scale
) {
    uint block_start = block_index * blocksize;
    if (block_start >= n) {
        return;
    }
    uint block_end = min(block_start + blocksize, n);
    uint pairs_in_block = (block_end - block_start + 1) >> 1;

    if (thread_idx == 0) {
        shared_scale = absmax[block_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_scale;

    for (uint pair = thread_idx; pair < pairs_in_block; pair += threadgroup_size) {
        uint value_index0 = block_start + pair * 2;
        if (value_index0 >= block_end) {
            break;
        }

        uint byte_index0 = value_index0 >> 1;
        uchar byte_val0 = packed[byte_index0];
        bool upper0 = ((value_index0 & 1) == 0);
        uchar nibble0 = upper0 ? ((byte_val0 >> 4) & 0xF) : (byte_val0 & 0xF);
        float decoded0 = code_table[nibble0] * scale;
        output[value_index0] = scalar_t(decoded0);

        uint value_index1 = value_index0 + 1;
        if (value_index1 < block_end) {
            uint byte_index1 = value_index1 >> 1;
            uchar byte_val1 = (byte_index1 == byte_index0) ? byte_val0 : packed[byte_index1];
            bool upper1 = ((value_index1 & 1) == 0);
            uchar nibble1 = upper1 ? ((byte_val1 >> 4) & 0xF) : (byte_val1 & 0xF);
            float decoded1 = code_table[nibble1] * scale;
            output[value_index1] = scalar_t(decoded1);
        }
    }
}

}  // namespace

// Quantization kernels
kernel void quantize_4bit_fp16_fp4(
    device const half* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blocks) {
        return;
    }
    quantize_block(input, absmax, packed, n, blocksize, gid, FP4_CODE);
}

kernel void quantize_4bit_fp16_nf4(
    device const half* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blocks) {
        return;
    }
    quantize_block(input, absmax, packed, n, blocksize, gid, NF4_CODE);
}

kernel void quantize_4bit_fp32_fp4(
    device const float* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blocks) {
        return;
    }
    quantize_block(input, absmax, packed, n, blocksize, gid, FP4_CODE);
}

kernel void quantize_4bit_fp32_nf4(
    device const float* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blocks) {
        return;
    }
    quantize_block(input, absmax, packed, n, blocksize, gid, NF4_CODE);
}

// Dequantization kernels
kernel void dequantize_4bit_fp16_fp4(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    if (tgid >= blocks) {
        return;
    }
    threadgroup float shared_scale;
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE, shared_scale);
}

kernel void dequantize_4bit_fp16_nf4(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    if (tgid >= blocks) {
        return;
    }
    threadgroup float shared_scale;
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE, shared_scale);
}

kernel void dequantize_4bit_fp32_fp4(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    if (tgid >= blocks) {
        return;
    }
    threadgroup float shared_scale;
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE, shared_scale);
}

kernel void dequantize_4bit_fp32_nf4(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    if (tgid >= blocks) {
        return;
    }
    threadgroup float shared_scale;
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE, shared_scale);
}