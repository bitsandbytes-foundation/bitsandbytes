#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

namespace {

constant uint kQuantThreadsCapacity = 512;

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
    uint thread_idx,
    uint threadgroup_size,
    constant float* code_table,
    threadgroup float* shared_thread_max,
    threadgroup float& shared_scale,
    uint simd_lane_id,
    uint simd_group_id
) {
    uint start = block_index * blocksize;
    if (start >= n) {
        return;
    }

    uint end = min(start + blocksize, n);
    float local_max = 0.0f;
    for (uint i = start + thread_idx; i < end; i += threadgroup_size) {
        float current = fabs((float)input[i]);
        local_max = max(local_max, current);
    }
    
    // SIMD reduction
    local_max = simd_max(local_max);

    // Store SIMD group max to shared memory
    if (simd_lane_id == 0) {
        shared_thread_max[simd_group_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_idx == 0) {
        float max_val = 0.0f;
        uint num_simd_groups = (threadgroup_size + 31) / 32;
        for (uint i = 0; i < num_simd_groups; ++i) {
            max_val = max(max_val, shared_thread_max[i]);
        }
        shared_scale = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_val = shared_scale;
    absmax[block_index] = max_val;
    float inv = max_val > 0.0f ? 1.0f / max_val : 0.0f;

    uint pairs_in_block = (end - start + 1) >> 1;
    uint out_byte = block_index * ((blocksize + 1) >> 1);

    for (uint pair = thread_idx; pair < pairs_in_block; pair += threadgroup_size) {
        uint value_index0 = start + pair * 2;
        float normalized0 = (max_val > 0.0f) ? clamp((float)input[value_index0] * inv, -1.0f, 1.0f) : 0.0f;
        uchar nibble0 = encode_value<scalar_t>(normalized0, code_table) & 0xF;

        uint value_index1 = value_index0 + 1;
        uchar nibble1 = 0;
        if (value_index1 < end) {
            float normalized1 = (max_val > 0.0f) ? clamp((float)input[value_index1] * inv, -1.0f, 1.0f) : 0.0f;
            nibble1 = encode_value<scalar_t>(normalized1, code_table) & 0xF;
        }
        packed[out_byte + pair] = (nibble0 << 4) | nibble1;
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
    constant float* code_table
) {
    uint block_start = block_index * blocksize;
    if (block_start >= n) {
        return;
    }
    uint block_end;
    if (block_start + blocksize < n) {
        block_end = block_start + blocksize;
    } else {
        block_end = n;
    }
    uint pairs_in_block = (block_end - block_start + 1) >> 1;

    float scale = absmax[block_index];

    // Precompute scaled table in registers - avoids threadgroup bank conflicts
    // and constant memory is broadcast-optimized so initial loads are fast
    float scaled_table[16];
    for (uint i = 0; i < 16; i++) {
        scaled_table[i] = code_table[i] * scale;
    }

    for (uint pair = thread_idx; pair < pairs_in_block; pair += threadgroup_size) {
        uint value_index0 = block_start + pair * 2;
        if (value_index0 >= block_end) {
            break;
        }

        uint byte_index0 = value_index0 >> 1;
        uchar byte_val0 = packed[byte_index0];
        // High nibble -> even index, low nibble -> odd index (matches Python ref)
        uchar nibble0 = (byte_val0 >> 4) & 0xF;
        uchar nibble1 = byte_val0 & 0xF;
        float decoded0 = scaled_table[nibble0];
        float decoded1 = scaled_table[nibble1];
        // value_index0 is already the output index (block_start + pair*2)
        output[value_index0] = scalar_t(decoded0);
        
        // Bounds check for odd-length blocks
        if (value_index0 + 1 < block_end) {
            output[value_index0 + 1] = scalar_t(decoded1);
        }
    }
}

// template <typename scalar_t>
// inline void dequantize_block(
//     device const uchar* packed,
//     device const float* absmax,
//     device scalar_t* output,
//     uint n,
//     uint blocksize,
//     uint block_index,
//     uint thread_idx,
//     uint threadgroup_size,
//     constant float* code_table
// ) {
//     const uint block_start = block_index * blocksize;
//     if (block_start >= n) return;

//     const uint block_end = min(block_start + blocksize, n);
//     const uint num_values = block_end - block_start;

//     const float scale = absmax[block_index];

//     // Precompute scaled code table
//     float scaled_table[16];
//     for (uint i = 0; i < 16; ++i)
//         scaled_table[i] = code_table[i] * scale;

//     device const uchar* packed_ptr = packed + (block_start >> 1);
//     device scalar_t* output_ptr = output + block_start;

//     // Each thread processes multiple *bytes* at a stride
//     const uint bytes_in_block = (num_values + 1) >> 1;

//     for (uint byte_idx = thread_idx; byte_idx < bytes_in_block; byte_idx += threadgroup_size) {
//         uchar byte_val = packed_ptr[byte_idx];

//         // Decode upper and lower nibbles
//         uchar upper_nib = (byte_val >> 4) & 0xF;
//         uchar lower_nib = byte_val & 0xF;

//         // Compute global value index
//         uint val_idx = byte_idx << 1;  // byte_idx * 2

//         // Write both values if in bounds
//         if (val_idx < num_values) output_ptr[val_idx] = scalar_t(scaled_table[upper_nib]);
//         if (val_idx + 1 < num_values) output_ptr[val_idx + 1] = scalar_t(scaled_table[lower_nib]);
//     }
// }

// template <typename scalar_t>
// inline void dequantize_block(
//     device const uchar* packed,
//     device const float* absmax,
//     device scalar_t* output,
//     uint n,
//     uint blocksize,
//     uint block_index,
//     uint thread_idx,
//     uint threadgroup_size,
//     constant float* code_table
// ) {
//     const uint block_start = block_index * blocksize;
//     if (block_start >= n) return;

//     const uint block_end = min(block_start + blocksize, n);
//     const uint num_values = block_end - block_start;

//     const float scale = absmax[block_index];

//     // Precompute scaled code table
//     float scaled_table[16];
//     for (uint i = 0; i < 16; ++i)
//         scaled_table[i] = code_table[i] * scale;

//     device const uchar* packed_ptr = packed + (block_start >> 1);
//     device scalar_t* output_ptr = output + block_start;

//     // Each thread processes multiple uchar4 (4 bytes = 8 values)
//     const uint num_bytes = (num_values + 1) >> 1; // total bytes in block
//     const uint num_blocks = (num_bytes + 3) >> 2; // number of uchar4 blocks

//     for (uint block_idx = thread_idx; block_idx < num_blocks; block_idx += threadgroup_size) {
//         uint byte_offset = block_idx * 4; // starting byte in packed array
//         uchar4 b = uchar4(0); // default zero

//         // Load safely (handle tail)
//         if (byte_offset + 3 < num_bytes) {
//             b = *((device uchar4*)(packed_ptr + byte_offset));
//         } else {
//             // Tail case: read remaining bytes safely
//             uchar temp[4] = {0, 0, 0, 0};
//             for (uint i = 0; i < num_bytes - byte_offset; ++i) {
//                 temp[i] = packed_ptr[byte_offset + i];
//             }
//             b = uchar4(temp[0], temp[1], temp[2], temp[3]);
//         }

//         // Decode 8 nibbles into 8 values
//         uchar nibbles[8] = {
//             uchar((b.x >> 4) & 0xF), uchar(b.x & 0xF),
//             uchar((b.y >> 4) & 0xF), uchar(b.y & 0xF),
//             uchar((b.z >> 4) & 0xF), uchar(b.z & 0xF),
//             uchar((b.w >> 4) & 0xF), uchar(b.w & 0xF)
//         };

//         // Compute global value indices and write outputs
//         uint val_idx = byte_offset << 1; // byte_offset * 2
//         for (uint i = 0; i < 8; ++i) {
//             if (val_idx + i < num_values)
//                 output_ptr[val_idx + i] = scalar_t(scaled_table[nibbles[i]]);
//         }
//     }
// }

// template <typename scalar_t>
// inline void dequantize_block(
//     device const uchar* packed,
//     device const float* absmax,
//     device scalar_t* output,
//     uint n,
//     uint blocksize,
//     uint block_index,
//     uint thread_idx,
//     uint threadgroup_size,
//     constant float* code_table
// ) {
//     const uint block_start = block_index * blocksize;
//     if (block_start >= n) return;

//     const uint block_end = min(block_start + blocksize, n);
//     const uint num_values = block_end - block_start;

//     const float scale = absmax[block_index];

//     // Precompute scaled code table
//     float scaled_table[16];
//     for (uint i = 0; i < 16; ++i)
//         scaled_table[i] = code_table[i] * scale;

//     device const uchar* packed_ptr = packed + (block_start >> 1);
//     device scalar_t* output_ptr = output + block_start;

//     const uint num_bytes = (num_values + 1) >> 1;        // total bytes in block
//     const uint num_uchar4 = (num_bytes + 3) >> 2;        // total uchar4 blocks

//     // Each thread handles one or two uchar4 blocks
//     uint block_pos = thread_idx;
//     if (block_pos >= num_uchar4) return;

//     // Compute byte offset
//     uint byte_offset = block_pos * 4;
//     uchar4 b = uchar4(0, 0, 0, 0);

//     // Safe load
//     if (byte_offset + 3 < num_bytes) {
//         b = *((device uchar4*)(packed_ptr + byte_offset));
//     } else {
//         uchar temp[4] = {0, 0, 0, 0};
//         for (uint i = 0; i < num_bytes - byte_offset; ++i)
//             temp[i] = packed_ptr[byte_offset + i];
//         b = uchar4(temp[0], temp[1], temp[2], temp[3]);
//     }

//     // Decode 8 nibbles
//     uchar nibbles[8] = {
//         uchar((b.x >> 4) & 0xF), uchar(b.x & 0xF),
//         uchar((b.y >> 4) & 0xF), uchar(b.y & 0xF),
//         uchar((b.z >> 4) & 0xF), uchar(b.z & 0xF),
//         uchar((b.w >> 4) & 0xF), uchar(b.w & 0xF)
//     };

//     // Compute global value index
//     uint val_idx = byte_offset << 1; // byte_offset * 2

//     // Fully unrolled writes (branch-free for main values)
//     if (val_idx + 0 < num_values) output_ptr[val_idx + 0] = scalar_t(scaled_table[nibbles[0]]);
//     if (val_idx + 1 < num_values) output_ptr[val_idx + 1] = scalar_t(scaled_table[nibbles[1]]);
//     if (val_idx + 2 < num_values) output_ptr[val_idx + 2] = scalar_t(scaled_table[nibbles[2]]);
//     if (val_idx + 3 < num_values) output_ptr[val_idx + 3] = scalar_t(scaled_table[nibbles[3]]);
//     if (val_idx + 4 < num_values) output_ptr[val_idx + 4] = scalar_t(scaled_table[nibbles[4]]);
//     if (val_idx + 5 < num_values) output_ptr[val_idx + 5] = scalar_t(scaled_table[nibbles[5]]);
//     if (val_idx + 6 < num_values) output_ptr[val_idx + 6] = scalar_t(scaled_table[nibbles[6]]);
//     if (val_idx + 7 < num_values) output_ptr[val_idx + 7] = scalar_t(scaled_table[nibbles[7]]);
// }

}  // namespace

// Quantization kernels
kernel void quantize_4bit_fp16_fp4(
    device const half* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (tgid >= blocks || threadgroup_size > kQuantThreadsCapacity) {
        return;
    }
    threadgroup float shared_thread_max[kQuantThreadsCapacity];
    threadgroup float shared_scale;
    quantize_block(input, absmax, packed, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE, shared_thread_max, shared_scale, simd_lane_id, simd_group_id);
}

kernel void quantize_4bit_fp16_nf4(
    device const half* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (tgid >= blocks || threadgroup_size > kQuantThreadsCapacity) {
        return;
    }
    threadgroup float shared_thread_max[kQuantThreadsCapacity];
    threadgroup float shared_scale;
    quantize_block(input, absmax, packed, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE, shared_thread_max, shared_scale, simd_lane_id, simd_group_id);
}

kernel void quantize_4bit_fp32_fp4(
    device const float* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (tgid >= blocks || threadgroup_size > kQuantThreadsCapacity) {
        return;
    }
    threadgroup float shared_thread_max[kQuantThreadsCapacity];
    threadgroup float shared_scale;
    quantize_block(input, absmax, packed, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE, shared_thread_max, shared_scale, simd_lane_id, simd_group_id);
}

kernel void quantize_4bit_fp32_nf4(
    device const float* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uchar* packed [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& blocksize [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (tgid >= blocks || threadgroup_size > kQuantThreadsCapacity) {
        return;
    }
    threadgroup float shared_thread_max[kQuantThreadsCapacity];
    threadgroup float shared_scale;
    quantize_block(input, absmax, packed, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE, shared_thread_max, shared_scale, simd_lane_id, simd_group_id);
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
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE);
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
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE);
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
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, FP4_CODE);
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
    // if (tgid >= blocks) {
    //     return;
    // }
    dequantize_block(packed, absmax, output, n, blocksize, tgid, tid, threadgroup_size, NF4_CODE);
}