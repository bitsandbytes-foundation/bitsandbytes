#include <metal_stdlib>
using namespace metal;

// Hand-written blockwise quant/dequant kernels, written to match the CPU/default reference
// in bitsandbytes/backends/default/ops.py bit-for-bit.
//
// Shared reference conventions (see quantize_blockwise / quantize_4bit in default/ops.py):
//   - Per-block absmax = max(|A[i]|) over the block.
//   - FULL blocks (length == blocksize): stored absmax is the raw (unclamped) max, and
//     scaling is reciprocal-then-multiply: scaled = A * (1 / max(absmax, 1e-38)).
//   - The TAIL block (the last block when n % blocksize != 0, length < blocksize): stored
//     absmax is max clamped to 1e-38, and scaling is a DIRECT divide: scaled = A / absmax.
//     This asymmetry is in the reference; reproducing it is required for bit-exact absmax
//     and codes on partial-block inputs.
//   - scaled is clamped to [-1, 1] before the code lookup.
//   - Code lookup reproduces torch.bucketize(..., right=False): searchsorted-left, i.e. the
//     number of bounds strictly less than `scaled`.
//
// The metallib is compiled with -fno-fast-math (see CMakeLists.txt) so division is correctly
// rounded and no FMA contraction occurs -- this is what keeps bucket selection identical to
// the CPU oracle.

// searchsorted-left over `n_bounds` ascending bounds; returns an index in [0, n_bounds].
static inline uint searchsorted_left(float scaled, device const float* bounds, uint n_bounds) {
    uint lo = 0;
    uint hi = n_bounds;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (bounds[mid] < scaled) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// ---- 8-bit blockwise quantize: A (float32) -> out (uint8 codes) + absmax (float32) ----
kernel void quantize_blockwise(
    device const float* code [[buffer(0)]],  // 256-entry sorted code table
    device const float* A [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    device float* absmax [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& blocksize [[buffer(5)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * blocksize;
    if (start >= n) {
        return;
    }
    const uint end = min(start + blocksize, n);
    const bool is_tail = (end - start) < blocksize;

    float amax = 0.0f;
    for (uint i = start; i < end; ++i) {
        amax = fmax(amax, fabs(A[i]));
    }

    // Tail block stores clamped absmax and divides; full block stores raw and reciprocal-multiplies.
    const float stored = is_tail ? fmax(amax, 1e-38f) : amax;
    absmax[block_id] = stored;
    const float inv = 1.0f / fmax(amax, 1e-38f);

    for (uint i = start; i < end; ++i) {
        const float scaled = clamp(is_tail ? (A[i] / stored) : (A[i] * inv), -1.0f, 1.0f);
        // 255 midpoint bounds of the 256-entry code table, computed on the fly.
        uint lo = 0;
        uint hi = 255;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            const float bound = (code[mid] + code[mid + 1]) * 0.5f;
            if (bound < scaled) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = (uchar)lo;
    }
}

// ---- 8-bit blockwise dequantize: A (uint8 codes) + absmax -> out (float32) ----
// out[i] = code[A[i]] * absmax[i / blocksize]. The Python wrapper casts fp32 out to the
// requested dtype (matching the reference's trailing .to(dtype)).
kernel void dequantize_blockwise(
    device const float* code [[buffer(0)]],  // 256-entry code table
    device const uchar* A [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& blocksize [[buffer(5)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * blocksize;
    if (start >= n) {
        return;
    }
    const uint end = min(start + blocksize, n);
    const float am = absmax[block_id];
    for (uint i = start; i < end; ++i) {
        out[i] = code[A[i]] * am;
    }
}

// ---- 4-bit blockwise dequantize (NF4/FP4): packed A -> out (float32) ----
// Nibble layout matches the reference: high nibble -> even output index, low nibble -> odd.
//   out[j] = code4[nibble_j] * absmax[j / blocksize]
kernel void dequantize_4bit(
    device const float* code [[buffer(0)]],  // 16-entry 4-bit code (NF4 or FP4)
    device const uchar* A [[buffer(1)]],     // packed nibbles, ceil(n/2) bytes
    device const float* absmax [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& blocksize [[buffer(5)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * blocksize;
    if (start >= n) {
        return;
    }
    const uint end = min(start + blocksize, n);
    const float am = absmax[block_id];
    for (uint j = start; j < end; ++j) {
        const uint byte = j >> 1;
        const uchar nib = ((j & 1u) == 0u) ? (A[byte] >> 4) : (A[byte] & 0x0Fu);
        out[j] = code[nib] * am;
    }
}

// ---- Fused 4-bit gemv (NF4/FP4): out[n] = sum_k A[k] * dequant(B[n,k]) ----
// One threadgroup = one SIMD-group (32 threads) per output element n. Threads stride over
// the packed row in uint4 units (16 bytes = 32 elements), dequantize in registers, and
// accumulate the dot product in fp32; a simd_sum reduction produces out[n]. Packed B is
// never materialized as a dequantized tensor -- this is the Phase M2 bandwidth win over
// dequant + F.linear.
//
// Preconditions enforced by the Python router (fallback used otherwise):
//   - K % 32 == 0, so every packed row (K/2 bytes) is 16-byte aligned and uint4 loads are
//     valid for every n.
//   - blocksize is a power of two (>= 32); `bs_shift` = log2(blocksize). Because K % 32 == 0
//     and blocksize is a multiple of 32, a 32-element chunk never straddles an absmax block,
//     so absmax is loaded once per chunk.
//
// Numeric parity with the CPU oracle (dequantize to A.dtype, then F.linear): the dequantized
// weight code[nib] * absmax is computed in fp32 and then ROUNDED to the activation dtype T
// before the multiply, reproducing the reference's `.to(dtype)` on B_dq. A is read in its
// native dtype (upcast to fp32 is exact) and accumulation is fp32; only accumulation ORDER
// differs from the oracle, which is what the documented per-dtype tolerances absorb. The
// final sum is rounded to T on store, matching F.linear's output dtype.
//
// One kernel per activation dtype (fp32/fp16/bf16) so A and out bind in torch's own dtype:
// the Python wrapper then launches ZERO torch cast kernels per call.
template <typename T>
static inline void gemv_4bit_body(
    device const float* code,
    device const uchar* B,
    device const float* absmax,
    device const T* A,
    device T* out,
    uint K,
    uint bs_shift,
    uint n,
    uint lane) {
    const ulong row_base = (ulong)n * (ulong)K;  // flattened element index of B[n, 0]
    device const uint4* Brow = (device const uint4*)(B + (row_base >> 1));
    const uint chunks = K >> 5;  // 32 elements (16 packed bytes) per chunk

    // Four independent accumulators (one per uint word of the chunk) break the serial fma
    // dependency chain; the kernel is ALU/latency-bound, not memory-bound, so this matters.
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint c = lane; c < chunks; c += 32u) {
        const uint4 packed = Brow[c];
        const uint k0 = c << 5;
        // The whole chunk lives in one absmax block (see preconditions above).
        const float am = absmax[(row_base + k0) >> bs_shift];

#pragma unroll
        for (uint w = 0; w < 4; ++w) {
            const uint word = packed[w];
            const uint kw = k0 + (w << 3);
            float acc_hi = 0.0f;
            float acc_lo = 0.0f;
            // Little-endian: byte b of `word` is packed byte index (kw/2 + b), holding
            // elements kw + 2b (high nibble) and kw + 2b + 1 (low nibble).
#pragma unroll
            for (uint b = 0; b < 4; ++b) {
                const uint byte = (word >> (b << 3)) & 0xFFu;
                const uint k = kw + (b << 1);
                const float w_hi = (float)(T)(code[byte >> 4] * am);
                const float w_lo = (float)(T)(code[byte & 0x0Fu] * am);
                // Explicit fma: -fno-fast-math disables contraction, but a deliberate fused
                // multiply-add is both allowed and more accurate than mul-then-add.
                acc_hi = fma((float)A[k], w_hi, acc_hi);
                acc_lo = fma((float)A[k + 1], w_lo, acc_lo);
            }
            const float word_sum = acc_hi + acc_lo;
            if (w == 0) {
                acc0 += word_sum;
            } else if (w == 1) {
                acc1 += word_sum;
            } else if (w == 2) {
                acc2 += word_sum;
            } else {
                acc3 += word_sum;
            }
        }
    }

    const float total = simd_sum((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) {
        out[n] = (T)total;
    }
}

#define BNB_GEMV_4BIT_KERNEL(NAME, T)                                                                                \
    kernel void NAME(                                                                                                \
        device const float* code [[buffer(0)]],    /* 16-entry 4-bit code (NF4 or FP4) */                           \
        device const uchar* B [[buffer(1)]],       /* packed nibbles, N*K/2 bytes, row-major [N, K] */              \
        device const float* absmax [[buffer(2)]],  /* per-block scales over the flattened [N*K] index */            \
        device const T* A [[buffer(3)]],           /* activations, K elements of T */                               \
        device T* out [[buffer(4)]],               /* N elements of T */                                            \
        constant uint& K [[buffer(5)]],                                                                              \
        constant uint& bs_shift [[buffer(6)]], /* log2(blocksize) */                                                 \
        uint n [[threadgroup_position_in_grid]],                                                                     \
        uint lane [[thread_index_in_simdgroup]]) {                                                                   \
        gemv_4bit_body<T>(code, B, absmax, A, out, K, bs_shift, n, lane);                                            \
    }

BNB_GEMV_4BIT_KERNEL(gemv_4bit_fp32, float)
BNB_GEMV_4BIT_KERNEL(gemv_4bit_fp16, half)
BNB_GEMV_4BIT_KERNEL(gemv_4bit_bf16, bfloat)

// ---- Phase M3: chunked 4-bit dequant into the ACTIVATION dtype (gemm_4bit scratch) ----
// Fills the scratch B_dq consumed by MPSMatrixMultiplication in bnb_mps_gemm_4bit. One
// thread per 32-element chunk (16 packed bytes, one uint4 load), writing
// (T)(code[nib] * absmax) -- the same rounding as the reference's B_dq.to(dtype), so the
// GEMM multiplies exactly the weights the oracle multiplies. Preconditions match the fused
// gemv kernel (enforced by the Python router): K % 32 == 0 so rows are 16-byte aligned and
// total elements are a multiple of 32; blocksize is a power of two >= 32 (bs_shift =
// log2(blocksize)), so a chunk never straddles an absmax block.
template <typename T>
static inline void dequantize_4bit_chunked_body(
    device const float* code,
    device const uchar* B,
    device const float* absmax,
    device T* out,
    uint bs_shift,
    uint chunk) {
    const ulong base = (ulong)chunk << 5;  // first element index of this chunk
    device const uint4* p = (device const uint4*)(B + (base >> 1));
    const uint4 packed = *p;
    const float am = absmax[base >> bs_shift];

#pragma unroll
    for (uint w = 0; w < 4; ++w) {
        const uint word = packed[w];
        // Little-endian: byte b of `word` is packed byte (base/2 + w*4 + b), holding
        // elements base + w*8 + 2b (high nibble) and base + w*8 + 2b + 1 (low nibble).
#pragma unroll
        for (uint b = 0; b < 4; ++b) {
            const uint byte = (word >> (b << 3)) & 0xFFu;
            const ulong j = base + (ulong)((w << 3) | (b << 1));
            out[j] = (T)(code[byte >> 4] * am);
            out[j + 1] = (T)(code[byte & 0x0Fu] * am);
        }
    }
}

#define BNB_DEQUANT_4BIT_CHUNKED_KERNEL(NAME, T)                                                                     \
    kernel void NAME(                                                                                                \
        device const float* code [[buffer(0)]],   /* 16-entry 4-bit code (NF4 or FP4) */                            \
        device const uchar* B [[buffer(1)]],      /* packed nibbles, row-major [N, K], N*K/2 bytes */               \
        device const float* absmax [[buffer(2)]], /* per-block scales over the flattened [N*K] index */             \
        device T* out [[buffer(3)]],              /* N*K elements of T (the GEMM scratch) */                        \
        constant uint& bs_shift [[buffer(4)]],    /* log2(blocksize) */                                             \
        uint chunk [[thread_position_in_grid]]) {                                                                   \
        dequantize_4bit_chunked_body<T>(code, B, absmax, out, bs_shift, chunk);                                      \
    }

BNB_DEQUANT_4BIT_CHUNKED_KERNEL(dequantize_4bit_chunked_fp32, float)
BNB_DEQUANT_4BIT_CHUNKED_KERNEL(dequantize_4bit_chunked_fp16, half)
BNB_DEQUANT_4BIT_CHUNKED_KERNEL(dequantize_4bit_chunked_bf16, bfloat)

// ---- Phase M3: bias epilogue for gemm_4bit ----
// out[m, n] += bias[n], broadcast over rows, in the activation dtype (reproducing
// F.linear's bias add on the T-typed matmul result). 2-D grid: x = n (column), y = m (row).
#define BNB_GEMM_BIAS_ADD_KERNEL(NAME, T)                                                                            \
    kernel void NAME(                                                                                                \
        device T* out [[buffer(0)]],           /* [M, N] row-major */                                               \
        device const T* bias [[buffer(1)]],    /* [N] */                                                            \
        constant uint& N [[buffer(2)]],                                                                              \
        uint2 gid [[thread_position_in_grid]]) {                                                                     \
        const ulong idx = (ulong)gid.y * (ulong)N + (ulong)gid.x;                                                    \
        out[idx] = (T)(out[idx] + bias[gid.x]);                                                                      \
    }

BNB_GEMM_BIAS_ADD_KERNEL(gemm_bias_add_fp32, float)
BNB_GEMM_BIAS_ADD_KERNEL(gemm_bias_add_fp16, half)
BNB_GEMM_BIAS_ADD_KERNEL(gemm_bias_add_bf16, bfloat)

// ---- 4-bit blockwise quantize (NF4/FP4): A (float32) -> packed out + absmax ----
// `bounds` are the 15 midpoints of the SORTED 16-entry code; `order` maps the searchsorted
// index back to the stored 4-bit index (identity for NF4, the argsort remap for FP4).
// blocksize is even, so element pairs never cross block boundaries: each block packs its own
// bytes at output offset (start / 2), padding a final odd element's low nibble with 0 (as the
// reference does at the end of the whole array).
kernel void quantize_4bit(
    device const float* bounds [[buffer(0)]],  // 15 ascending midpoints
    device const uchar* order [[buffer(1)]],   // 16-entry remap
    device const float* A [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    device float* absmax [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& blocksize [[buffer(6)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * blocksize;
    if (start >= n) {
        return;
    }
    const uint end = min(start + blocksize, n);
    const bool is_tail = (end - start) < blocksize;

    float amax = 0.0f;
    for (uint i = start; i < end; ++i) {
        amax = fmax(amax, fabs(A[i]));
    }
    const float stored = is_tail ? fmax(amax, 1e-38f) : amax;
    absmax[block_id] = stored;
    const float inv = 1.0f / fmax(amax, 1e-38f);

    const uint len = end - start;
    const uint nbytes = (len + 1u) >> 1;
    const uint byte_base = start >> 1;
    for (uint k = 0; k < nbytes; ++k) {
        const uint hi_idx = start + 2u * k;
        const uint lo_idx = hi_idx + 1u;

        const float hs = clamp(is_tail ? (A[hi_idx] / stored) : (A[hi_idx] * inv), -1.0f, 1.0f);
        const uchar hi = order[searchsorted_left(hs, bounds, 15)];

        // For an odd-length tail block the final low nibble is padding: the reference pads
        // `scaled` with 0.0 and quantizes THAT (not a literal 0), so match it.
        const float ls = (lo_idx < end) ? clamp(is_tail ? (A[lo_idx] / stored) : (A[lo_idx] * inv), -1.0f, 1.0f) : 0.0f;
        const uchar lo = order[searchsorted_left(ls, bounds, 15)];
        out[byte_base + k] = (uchar)((hi << 4) | lo);
    }
}
