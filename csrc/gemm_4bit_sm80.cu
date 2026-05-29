// sm80+ MMA (mma.sync.aligned.m16n8k16) 4-bit GEMM kernel (bf16 and fp16).

#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "gemm_4bit_common.cuh"
#include "gemm_4bit_sm80.cuh"

[[maybe_unused]] static constexpr int MMA_M = 16;
[[maybe_unused]] static constexpr int MMA_N = 8;
[[maybe_unused]] static constexpr int MMA_K = 16;

static constexpr int NUM_WARPS = 8;
static constexpr int CTA_SIZE = NUM_WARPS * 32;

/// @brief In-place warp-level MMA: accum += A * B (bf16 or fp16, m16n8k16).
///        Called once per K chunk to accumulate the full matrix product.
///
/// Executes mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32.
/// Fragments are distributed across warp lanes per PTX spec.
///
/// @tparam T    Input dtype; selects bf16 or fp16 PTX instruction
/// @param accum In-place f32 accumulator (4 regs per lane)
/// @param a     A fragment: 16x16 operand (4 regs per lane)
/// @param b     B fragment: 16x8 operand (2 regs per lane)
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(FragC& accum, const uint32_t a[4], const uint32_t b[2]) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // clang-format off
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(accum.x[0]), "+f"(accum.x[1]), "+f"(accum.x[2]), "+f"(accum.x[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );
        // clang-format on
    } else {
        // clang-format off
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(accum.x[0]), "+f"(accum.x[1]), "+f"(accum.x[2]), "+f"(accum.x[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );
        // clang-format on
    }
}

// Smem stride = K_CHUNK + 8 elements (same for A and B).
// +8 padding: 16-byte row alignment, limits bank conflicts to 4-way for both
// K_CHUNK=64 (stride=72, gcd(36,32)=4) and K_CHUNK=128 (stride=136, gcd(68,32)=4).

// A: m16 x k16 from row-major [m][k] smem. ldmatrix.x4.
template <typename T>
__device__ __forceinline__ void
    load_A_frag(uint32_t frag[4], const T* smem_a, int m_off, int k_off, int lane, int stride) {
    const int mat_idx = lane / 8;
    const int row_in_mat = lane % 8;
    const int m_row = m_off + row_in_mat + (mat_idx & 1) * 8;
    const int k_col = k_off + (mat_idx >> 1) * 8;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[m_row * stride + k_col]));
    // clang-format off
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(addr)
    );
    // clang-format on
}

// B: k16 x n8 from row-major [n][k] smem (= col-major [k][n]). ldmatrix.x2.
template <typename T>
__device__ __forceinline__ void
    load_B_frag(uint32_t frag[2], const T* smem_b, int n_off, int k_off, int lane, int stride) {
    int n_row = lane % 8;
    int k_col = k_off + (lane / 8) * 8;
    if (lane >= 16) {
        n_row = (lane - 16) % 8;
        k_col = k_off + ((lane - 16) / 8) * 8;
    }
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[(n_off + n_row) * stride + k_col]));
    // clang-format off
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(addr)
    );
    // clang-format on
}

// Smem size per double-buffer in bytes (sizeof(T) == 2 for both bf16 and fp16).
template <typename T, int MT, int NT, int K_CHUNK = 64> static constexpr int smem_bytes_for() {
    constexpr int STRIDE = K_CHUNK + 8;
    return 2 * (MT + NT) * STRIDE * static_cast<int>(sizeof(T));
}

/// @brief Fused 4-bit dequantize + MMA GEMM for sm80+ (bf16 and fp16).
///        Computes C[M,N] = A[M,K] @ B[N,K]^T + bias.
///
/// Layout:
///   A: [M, K] row-major, T (activations)
///   B: [N, K/2] row-major, packed uint8 (2 nibbles per byte, weights)
///   C: [M, N] row-major, T (output)
///
/// MMA: `mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32`
///   A operand: row-major [M, K]
///   B operand: col-major [K, N] (B [N, K] row-major reinterpreted)
///
/// Double-buffered smem pipeline. Supports optional nested absmax and bias.
///
/// Smem per double-buffer = 2*(MT+NT)*(K_CHUNK+8)*sizeof(T) bytes:
///   KC=64:  32x 64  27KB   32x128  45KB   64x 64  36KB
///           64x128  54KB  128x 64  54KB  128x128  72KB
///   KC=128: 32x 64  51KB   32x128  85KB   32x256 153KB
///           64x 32  51KB   64x 64  68KB
///
/// @tparam T       Input/output dtype (`__nv_bfloat16` or `half`)
/// @tparam MT      M tile size (32, 64, or 128)
/// @tparam NT      N tile size (32, 64, 128, or 256)
/// @tparam K_CHUNK K elements per outer iteration (64 or 128)
template <typename T, int MT = 128, int NT = 128, int K_CHUNK = 64>
__global__ void __launch_bounds__(CTA_SIZE) gemm_4bit_sm80_m16n8k16(
    // clang-format off
    const T*       __restrict__ A,             // inputs [M, K]
    const uint8_t* __restrict__ B,             // packed 4-bit weights [N, K/2]
    const float*   __restrict__ absmax,        // fp32 absmax [N, K/blocksize] or
                                               //   [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* __restrict__ absmax_8bit,   // [N, K/blocksize] uint8 compressed absmax;
                                               //   nullptr = non-nested
    const float*   __restrict__ absmax_code,   // [256] codebook for 8bit absmax
    const float*   __restrict__ absmax_offset, // scalar; nullptr = non-nested
    T*             __restrict__ C,             // [M, N]
    const T*       __restrict__ bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,                       // problem shape
    int blocksize,                             // elements per quantization block
    int quant_type                             // 1 = FP4, 2 = NF4
    // clang-format on
) {
#if __CUDA_ARCH__ >= 800
    static_assert(MT == 32 || MT == 64 || MT == 128, "MT must be 32, 64, or 128");
    static_assert(NT == 32 || NT == 64 || NT == 128 || NT == 256, "NT must be 32, 64, 128, or 256");
    static_assert(K_CHUNK == 64 || K_CHUNK == 128, "K_CHUNK must be 64 or 128");

    // Trap on tile+arch combinations that the dispatcher can never reach.
    // if constexpr prunes the kernel body entirely at compile time; __trap()
    // provides a visible error if this assumption is ever violated at runtime.
    // clang-format off
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
    // HBM (sm90/sm100/sm103): MT=32 always returns K_CHUNK=128 with NT=64 or NT=256;
    // MT=64 never uses NT=32 (GDDR-only fallback).
    // sm80 is intentionally excluded: these four GDDR tiles are unused on sm80 itself
    // but must remain in the sm80 cubin so it can run correctly on sm86/sm89.
    if constexpr ((MT==32 && NT== 64 && K_CHUNK== 64) ||
                  (MT==32 && NT==128 && K_CHUNK== 64) ||
                  (MT==32 && NT==128 && K_CHUNK==128) ||
                  (MT==64 && NT== 32 && K_CHUNK==128)) { __trap(); return; }
#endif
#if __CUDA_ARCH__ == 900
    // sm90: only 64x64-64 is dispatched at MT=64 (NT forced to 64, K_CHUNK=128 never selected).
    if constexpr (MT==64 && !(NT==64 && K_CHUNK==64)) { __trap(); return; }
#endif
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
    // sm100/sm103 (B200/B300): only 64x64-128 is dispatched at MT=64.
    if constexpr (MT==64 && !(NT==64 && K_CHUNK==128)) { __trap(); return; }
#endif
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
    // GDDR (sm86/sm89/sm120/sm121): NT=256 only exists in the HBM MT=32 path.
    if constexpr (MT==32 && NT==256 && K_CHUNK==128) { __trap(); return; }
#endif
    // clang-format on

    const float absmax_offset_f = absmax_8bit ? __ldg(absmax_offset) : 0.0f;

    // K_CHUNK + 8 padding: 16-byte row alignment, limits bank conflicts to 4-way.
    constexpr int SMEM_A_STRIDE = K_CHUNK + 8;
    constexpr int SMEM_B_STRIDE = K_CHUNK + 8;

    using WL = MmaWarpLayout<MT, NT, NUM_WARPS, MMA_M, MMA_N>;
    constexpr int WARPS_M = WL::WARPS_M;
    constexpr int WARPS_N = WL::WARPS_N;
    constexpr int WARP_M = WL::WARP_M;
    constexpr int WARP_MMA_M = WL::WARP_MMA_M;
    constexpr int WARP_N = WL::WARP_N;
    constexpr int WARP_MMA_N = WL::WARP_MMA_N;

    static_assert(MT >= WARPS_M * MMA_M, "MT too small for warp layout");
    static_assert(NT >= WARPS_N * MMA_N, "NT too small for warp layout");

    // Packed bytes of B per thread per K-chunk:
    //   NT= 64, KC= 64 ->  8 bytes (uint2)
    //   NT=128, KC= 64 -> 16 bytes (uint4)
    //   NT= 32, KC=128 ->  8 bytes (uint2)
    //   NT= 64, KC=128 -> 16 bytes (uint4)
    //   NT=128, KC=128 -> 32 bytes (2x uint4)
    //   NT=256, KC=128 -> 64 bytes (4x uint4)
    constexpr int B_BYTES = NT * (K_CHUNK / 2) / CTA_SIZE;
    static_assert(B_BYTES == 8 || B_BYTES == 16 || B_BYTES == 32 || B_BYTES == 64, "unexpected B bytes per thread");

    extern __shared__ char smem_raw[];
    constexpr int buf_offset = (MT * SMEM_A_STRIDE + NT * SMEM_B_STRIDE) * sizeof(T);
    auto smem_a_buf = [&](int buf) -> T* { return reinterpret_cast<T*>(smem_raw + buf * buf_offset); };
    auto smem_b_buf = [&](int buf) -> T* {
        return reinterpret_cast<T*>(smem_raw + buf * buf_offset + MT * SMEM_A_STRIDE * sizeof(T));
    };

    const int bm = blockIdx.x * MT;
    const int bn = blockIdx.y * NT;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    // LUT in fp32: centroid * scale in fp32 avoids double rounding to output dtype.
    const float* lut = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
    const float my_lut_f32 = (lane_id < 16) ? lut[lane_id] : 0.0f;

    const int k_iters = K / K_CHUNK;
    const int blocksize_log2 = __ffs(blocksize) - 1;
    const int blocks_per_row = K >> blocksize_log2;

    FragC accum[WARP_MMA_M][WARP_MMA_N];
#pragma unroll
    for (int wm = 0; wm < WARP_MMA_M; wm++)
#pragma unroll
        for (int wn = 0; wn < WARP_MMA_N; wn++) {
            accum[wm][wn].x[0] = 0.f;
            accum[wm][wn].x[1] = 0.f;
            accum[wm][wn].x[2] = 0.f;
            accum[wm][wn].x[3] = 0.f;
        }

    // Tile loading: A (direct copy from global) + B (load packed 4-bit + dequant)
    auto load_tile = [&](int k_iter, int buf) {
        const int k_base = k_iter * K_CHUNK;
        T* __restrict__ sa = smem_a_buf(buf);
        T* __restrict__ sb = smem_b_buf(buf);

        // Load A: each thread loads vecs_per_thread uint4 (8 T elements each).
        constexpr int vecs_per_row = K_CHUNK / 8;
        constexpr int vecs_per_thread = MT * vecs_per_row / CTA_SIZE;
#pragma unroll
        for (int v = 0; v < vecs_per_thread; v++) {
            const int vec_idx = threadIdx.x * vecs_per_thread + v;
            const int row = vec_idx / vecs_per_row;
            const int col = (vec_idx % vecs_per_row) * 8;
            const int g_row = bm + row;
            uint4 val = {0u, 0u, 0u, 0u};
            if (g_row < M) {
                // clang-format off
                asm volatile(
                    "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                    : "l"(&A[g_row * K + k_base + col])
                );
                // clang-format on
            }
            *reinterpret_cast<uint4*>(&sa[row * SMEM_A_STRIDE + col]) = val;
        }

        // Pre-fetch nested_idx_b before scale_f computation; the A-tile loads above
        // provide enough in-flight latency to hide the absmax_8bit read.
        const int byte_start_b = threadIdx.x * B_BYTES;
        const int n_local_b = byte_start_b / (K_CHUNK / 2);
        const int k_byte0_b = byte_start_b % (K_CHUNK / 2);
        const int n_global_b = bn + n_local_b;
        const int k_elem0_b = k_byte0_b * 2;
        const int blk_idx_b = n_global_b * blocks_per_row + ((k_base + k_elem0_b) >> blocksize_log2);
        uint8_t nested_idx_b = 0;
        if (absmax_8bit && n_global_b < N)
            nested_idx_b = __ldg(&absmax_8bit[blk_idx_b]);

        // Load + dequant B
        {
            const int n_local = n_local_b;
            const int k_byte0 = k_byte0_b;
            const int n_global = n_global_b;
            const int k_elem0 = k_elem0_b;
            // (k_base + k_elem0) selects the correct absmax block for any blocksize:
            // for blocksize >= K_CHUNK the index collapses to k_iter; for smaller
            // blocksizes each thread's k_elem0 selects the right sub-chunk block.
            const int blk_idx = blk_idx_b;

            float scale_f = 0.0f;
            if (n_global < N) {
                if (absmax_8bit) {
                    // nested_idx_b was pre-fetched above; __ldg keeps the 1KB codebook in read-only L1.
                    scale_f = __ldg(&absmax_code[nested_idx_b]) * __ldg(&absmax[blk_idx >> 8]) + absmax_offset_f;
                } else {
                    scale_f = __ldg(&absmax[blk_idx]);
                }
            }

            // fp32 centroid * scale avoids double rounding; hi nibble (>>4) = lower K index.
            auto dequant_byte = [&](uint8_t byte, int smem_off) {
                const float hi = __shfl_sync(0xffffffff, my_lut_f32, byte >> 4);
                const float lo = __shfl_sync(0xffffffff, my_lut_f32, byte & 0x0f);
                const auto dq = make_vec2<T>(hi * scale_f, lo * scale_f);
                *reinterpret_cast<uint32_t*>(&sb[n_local * SMEM_B_STRIDE + smem_off]) =
                    *reinterpret_cast<const uint32_t*>(&dq);
            };

            if constexpr (B_BYTES == 64) {
                // NT=256, KC=128: each thread covers one N-column x K_CHUNK K-elements.
                // scale_f for the first block is fetched above. The loop refreshes it at
                // each absmax block boundary using (j*2 & (blocksize-1)) == 0, which is
                // a single bitmask AND per unrolled step since blocksize is always a power of 2.
                uint4 p0 = {0, 0, 0, 0}, p1 = {0, 0, 0, 0}, p2 = {0, 0, 0, 0}, p3 = {0, 0, 0, 0};
                const uint8_t* bptr = &B[n_global * (K / 2) + k_base / 2 + k_byte0];
                if (n_global < N) {
                    // clang-format off
                    asm volatile(
                        "ld.global.cs.v4.u32 {%0,%1,%2,%3},   [%16];\n"
                        "ld.global.cs.v4.u32 {%4,%5,%6,%7},   [%16+16];\n"
                        "ld.global.cs.v4.u32 {%8,%9,%10,%11}, [%16+32];\n"
                        "ld.global.cs.v4.u32 {%12,%13,%14,%15},[%16+48];\n"
                        : "=r"(p0.x), "=r"(p0.y), "=r"(p0.z), "=r"(p0.w),
                          "=r"(p1.x), "=r"(p1.y), "=r"(p1.z), "=r"(p1.w),
                          "=r"(p2.x), "=r"(p2.y), "=r"(p2.z), "=r"(p2.w),
                          "=r"(p3.x), "=r"(p3.y), "=r"(p3.z), "=r"(p3.w)
                        : "l"(bptr)
                    );
                    // clang-format on
                }
                uint8_t bytes[64];
                memcpy(bytes, &p0, 16);
                memcpy(bytes + 16, &p1, 16);
                memcpy(bytes + 32, &p2, 16);
                memcpy(bytes + 48, &p3, 16);
#pragma unroll
                for (int j = 0; j < B_BYTES; j++) {
                    if (j > 0 && ((j * 2) & (blocksize - 1)) == 0 && n_global < N) {
                        const int blk_idx = n_global * blocks_per_row + ((k_base + k_elem0 + j * 2) >> blocksize_log2);
                        if (absmax_8bit) {
                            scale_f = __ldg(&absmax_code[__ldg(&absmax_8bit[blk_idx])]) * __ldg(&absmax[blk_idx >> 8]) +
                                      absmax_offset_f;
                        } else {
                            scale_f = __ldg(&absmax[blk_idx]);
                        }
                    }
                    dequant_byte(bytes[j], k_elem0 + j * 2);
                }
            } else if constexpr (B_BYTES == 32) {
                // NT=128, KC=128: same block boundary refresh as B_BYTES=64.
                // Needed when blocksize < K_CHUNK (e.g. blocksize=32 with KC=128).
                uint4 p0 = {0, 0, 0, 0}, p1 = {0, 0, 0, 0};
                const uint8_t* bptr = &B[n_global * (K / 2) + k_base / 2 + k_byte0];
                if (n_global < N) {
                    // clang-format off
                    asm volatile(
                        "ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%8];\n"
                        "ld.global.cs.v4.u32 {%4,%5,%6,%7}, [%8+16];\n"
                        : "=r"(p0.x), "=r"(p0.y), "=r"(p0.z), "=r"(p0.w),
                          "=r"(p1.x), "=r"(p1.y), "=r"(p1.z), "=r"(p1.w)
                        : "l"(bptr)
                    );
                    // clang-format on
                }
                uint8_t bytes[32];
                memcpy(bytes, &p0, 16);
                memcpy(bytes + 16, &p1, 16);
#pragma unroll
                for (int j = 0; j < B_BYTES; j++) {
                    if (j > 0 && ((j * 2) & (blocksize - 1)) == 0 && n_global < N) {
                        const int blk_idx = n_global * blocks_per_row + ((k_base + k_elem0 + j * 2) >> blocksize_log2);
                        if (absmax_8bit) {
                            scale_f = __ldg(&absmax_code[__ldg(&absmax_8bit[blk_idx])]) * __ldg(&absmax[blk_idx >> 8]) +
                                      absmax_offset_f;
                        } else {
                            scale_f = __ldg(&absmax[blk_idx]);
                        }
                    }
                    dequant_byte(bytes[j], k_elem0 + j * 2);
                }
            } else if constexpr (B_BYTES == 16) {
                uint4 packed4 = {0u, 0u, 0u, 0u};
                if (n_global < N) {
                    // clang-format off
                    asm volatile(
                        "ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(packed4.x), "=r"(packed4.y),
                          "=r"(packed4.z), "=r"(packed4.w)
                        : "l"(&B[n_global * (K / 2) + k_base / 2 + k_byte0])
                    );
                    // clang-format on
                }
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed4);
#pragma unroll
                for (int j = 0; j < 16; j++)
                    dequant_byte(bytes[j], k_elem0 + j * 2);
            } else if constexpr (B_BYTES == 8) {
                uint2 packed2 = {0u, 0u};
                if (n_global < N) {
                    // clang-format off
                    asm volatile(
                        "ld.global.cs.v2.u32 {%0,%1}, [%2];\n"
                        : "=r"(packed2.x), "=r"(packed2.y)
                        : "l"(&B[n_global * (K / 2) + k_base / 2 + k_byte0])
                    );
                    // clang-format on
                }
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed2);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    dequant_byte(bytes[j], k_elem0 + j * 2);
            }
        }
    };

    // Compute MMA on one buffer
    auto compute = [&](int buf) {
        const T* sa = smem_a_buf(buf);
        const T* sb = smem_b_buf(buf);
        const int wm_off = warp_m * WARP_M;
        const int wn_off = warp_n * WARP_N;

        for (int kk = 0; kk < K_CHUNK / MMA_K; kk++) {
            const int k_off = kk * MMA_K;
            uint32_t a_frag[WARP_MMA_M][4];
#pragma unroll
            for (int wm = 0; wm < WARP_MMA_M; wm++)
                load_A_frag<T>(a_frag[wm], sa, wm_off + wm * MMA_M, k_off, lane_id, SMEM_A_STRIDE);
#pragma unroll
            for (int wn = 0; wn < WARP_MMA_N; wn++) {
                uint32_t b_frag[2];
                load_B_frag<T>(b_frag, sb, wn_off + wn * MMA_N, k_off, lane_id, SMEM_B_STRIDE);
#pragma unroll
                for (int wm = 0; wm < WARP_MMA_M; wm++)
                    mma_m16n8k16<T>(accum[wm][wn], a_frag[wm], b_frag);
            }
        }
    };

    // Main loop: double buffered
    load_tile(0, 0);
    __syncthreads();

    for (int k_iter = 0; k_iter < k_iters; k_iter++) {
        const int cur_buf = k_iter % 2;
        const int next_buf = 1 - cur_buf;
        if (k_iter + 1 < k_iters)
            load_tile(k_iter + 1, next_buf);
        compute(cur_buf);
        __syncthreads();
    }

    mma_store_accum<T, WARP_MMA_M, WARP_MMA_N, MMA_M, MMA_N>(
        C, accum, bm, bn, warp_m * WARP_M, warp_n * WARP_N, M, N, lane_id, bias
    );
#endif // __CUDA_ARCH__ >= 800
}

template <typename T, int MT, int NT, int KC>
static void launch_tile(
    // clang-format off
    const T* A,
    const uint8_t* B,
    const float* absmax,
    const uint8_t* absmax_8bit,
    const float* absmax_code,
    const float* absmax_offset,
    T* C,
    const T* bias,
    int M, int N, int K,
    int blocksize,
    int quant_type,
    GpuProps gpu,
    cudaStream_t stream
    // clang-format on
) {
    constexpr int smem = smem_bytes_for<T, MT, NT, KC>();
    static bool cfg[16] = {};
    if (gpu.device_index >= 16 || !cfg[gpu.device_index]) {
        cudaFuncSetAttribute(gemm_4bit_sm80_m16n8k16<T, MT, NT, KC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (gpu.device_index < 16)
            cfg[gpu.device_index] = true;
    }
    dim3 grid((M + MT - 1) / MT, (N + NT - 1) / NT);
    gemm_4bit_sm80_m16n8k16<T, MT, NT, KC><<<grid, dim3(CTA_SIZE), smem, stream>>>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type
    );
}

/// @brief Auto-dispatch launcher for the sm80+ MMA kernel. Selects MT/NT/KC tile
///        based on GPU arch, SM count, and shape.
/// @tparam T Input/output dtype (`__nv_bfloat16` or `half`)
template <typename T>
void launch_gemm_4bit_sm80_m16n8k16(
    // clang-format off
    const T* A,
    const uint8_t* B,
    const float* absmax,
    const uint8_t* absmax_8bit,
    const float* absmax_code,
    const float* absmax_offset,
    T* C,
    const T* bias,
    int M, int N, int K,
    int blocksize,
    int quant_type,
    GpuProps gpu,
    cudaStream_t stream
    // clang-format on
) {
    const int num_sms = gpu.num_sms;
    const int cc_major = gpu.cc_major;
    const int cc_minor = gpu.cc_minor;

    const bool hbm_arch = (cc_major == 8 && cc_minor == 0) || cc_major == 9 || cc_major == 10;
    const bool sm86 = cc_major == 8 && cc_minor == 6;
    // A10-class sm86 cards run ~600 GB/s; RTX 3090 (82 SMs) and A40 (84 SMs) are above
    // the 72-SM threshold and have higher bandwidth (behavior not validated here).
    const bool sm86_low_bw = sm86 && num_sms <= 72;
    // RTX Pro 6000 (sm120, 188 SMs) is the only validated high-SM sm120 card.
    const bool high_sm_sm120 = cc_major == 12 && num_sms >= 150;

    const int m_blocks_32 = (M + 31) / 32;
    const int m_blocks_64 = (M + 63) / 64;
    const int m_blocks_128 = (M + 127) / 128;
    const int n_blocks_64 = (N + 63) / 64;
    const int n_blocks_128 = (N + 127) / 128;

    const int blocks_32x64 = m_blocks_32 * n_blocks_64;
    const int blocks_64x64 = m_blocks_64 * n_blocks_64;
    const int blocks_128x128 = m_blocks_128 * n_blocks_128;

    // MT=128 wastes M-rows when M % 128 is in [1, 64]: the last tile is <=50% full.
    const bool mt128_row_waste = (M % 128 != 0) && (M % 128 <= 64);
    // MT=128 is adequate at >=3/4 wave. At M=96 on A10 (N=8192: 0.89 waves),
    // MT=128 compute efficiency beats MT=32 despite modest under-subscription.
    const bool mt128_adequate = blocks_128x128 * 4 >= num_sms * 3;
    // Suppress MT=32 when M just crossed a 64-row boundary and the extra MT=32 block is
    // nearly empty. Once blocks_64x64 >= num_sms (1 wave with 64x64 tiles), MT=64 fills cleanly.
    const bool mt32_boundary_waste = (m_blocks_32 > m_blocks_64) && (blocks_64x64 >= num_sms);

    const bool use_mt32 =
        (M < 48 && !mt32_boundary_waste) || (M <= 128 && !(mt32_boundary_waste && mt128_adequate) &&
                                             blocks_32x64 >= num_sms * 2 && m_blocks_32 * 32 < m_blocks_64 * 64);

    const bool use_mt64 = !use_mt32 && (mt128_row_waste || !mt128_adequate) && blocks_64x64 > blocks_128x128;

    struct Tile {
        int mt, nt, kc;
    };

    auto select_tile = [&]() -> Tile {
        int mt = use_mt32 ? 32 : (use_mt64 ? 64 : 128);

        if (mt == 64) {
            // sm86 low-BW, tall-K (K>=N) at >=0.5 wave: MT=128 amortizes B-matrix bandwidth
            // more efficiently than MT=64.
            // Calibrated on A10: 128x128 beats 64x128 at M=97-128 for tall-K.
            if (sm86_low_bw && K >= N && !mt128_row_waste && blocks_128x128 * 2 >= num_sms)
                mt = 128;
            // High-SM sm120 below 1 wave: MT=64 is severely under-subscribed.
            // MT=32 doubles M-block count and recovers occupancy.
            // Calibrated on RTX Pro 6000: N=8192,K=8192 M=63-64.
            else if (high_sm_sm120 && blocks_64x64 < num_sms)
                mt = 32;
            // HBM narrow-N below 0.5 wave.
            // Calibrated on A100: N=512,K=4096 M=48-384.
            else if (hbm_arch && blocks_64x64 < num_sms / 2)
                mt = 32;
            // Short-K small-weight (K*2<N, weight<2MB): K-loop too short to amortize MT=64.
            // MT=32 fills >=1 wave and wins when more M-blocks are in flight.
            else if (K * 2 < N && (long long)N * K < 4LL * 1024 * 1024 && blocks_32x64 >= num_sms)
                mt = 32;
            // GDDR short-K at 1/4-3/4 wave: 32x64-128 outperforms MT=64 because the shorter
            // K-loop makes KC=128 ILP more valuable than a wider output tile.
            // num_sms >= 60 excludes L4 (58 SMs) where KC=64 competes.
            else if (!hbm_arch && K * 2 < N && num_sms >= 60 && blocks_64x64 >= num_sms / 4 &&
                     blocks_64x64 < num_sms * 3 / 4)
                return {32, 64, 128};
        }

        if (mt == 32) {
            if (hbm_arch) {
                // HBM MT=32: always KC=128 (longer chunks hide HBM latency).
                // NT=256 in the 3/4-to-1-wave window; NT=64's 4x block advantage
                // takes over above 1 wave.
                // Calibrated on A100/H100/H200: N=36864 (>1 wave), 32x64 beats 32x256.
                const int n_blocks_256 = (N + 255) / 256;
                const int blocks_32x256 = m_blocks_32 * n_blocks_256;
                const int nt = (blocks_32x256 * 4 >= num_sms * 3 && blocks_32x256 <= num_sms) ? 256 : 64;
                return {32, nt, 128};
            }

            // GDDR MT=32: KC and NT driven by occupancy and register pressure.
            if (m_blocks_32 >= 2 && blocks_32x64 > num_sms) {
                // M>=33, >1 wave: register pressure at this occupancy favors KC=64.
                // NT=128 in the 1-2 wave window on sm86 or short-K (K*3<=N).
                if (m_blocks_32 >= 3) {
                    const int blocks_32x128 = m_blocks_32 * n_blocks_128;
                    if (blocks_32x128 > num_sms && blocks_32x128 <= num_sms * 2 && (sm86 || K * 3 <= N))
                        return {32, 128, 64};
                }
                return {32, 64, 64};
            }
            if (m_blocks_32 == 1) {
                // M<33, just above 1 wave (1.0-1.2x): KC=64 wins in this narrow window.
                if (blocks_32x64 > num_sms && blocks_32x64 < num_sms + num_sms / 5)
                    return {32, 64, 64};
                // M<33, >3 waves on sm86: NT=128 + KC=64.
                // Calibrated on A10: N=14336-36864 M=5-32; not validated on sm89/sm120.
                if (sm86 && blocks_32x64 > num_sms * 3)
                    return {32, 128, 64};
            }
            // 64x32-128: NT=32 gives 2x more N-blocks than NT=64 when occupancy-limited.
            // Excluded for high-SM sm120 (no calibrated wins on RTX Pro 6000).
            if (!high_sm_sm120) {
                const int blocks_64x32 = m_blocks_64 * ((N + 31) / 32);
                if (blocks_64x32 <= num_sms)
                    return {64, 32, 128};
            }
            // KC=128 fallback: NT=128 when >=2/3 wave, NT=64 otherwise.
            const int nt = (m_blocks_32 * n_blocks_128 * 3 >= num_sms * 2) ? 128 : 64;
            return {32, nt, 128};
        }

        if (mt == 64) {
            // sm90/sm100: no calibrated wins for 64x128; default to NT=64.
            // A100 and GDDR retain NT=128 when well-subscribed.
            int nt;
            if (cc_major == 9 || cc_major == 10)
                nt = 64;
            else if (m_blocks_64 * n_blocks_128 < num_sms)
                nt = 64;
            else
                nt = 128;
            // KC=128 for sm100 (Blackwell) or GDDR tall-K (K>=N).
            // sm80: no calibrated wins for KC=128 at MT=64.
            if (nt == 64 && (cc_major == 10 || (!hbm_arch && K >= N)))
                return {64, 64, 128};
            return {64, nt, 64};
        }

        // MT=128: KC=128 not dispatched (needs M>768 benchmarks to calibrate HBM crossover).
        // NT=128 threshold is halved on sm86 low-BW vs other arches.
        // Calibrated on A10: N=8192,K=2048 M=127, NT=128 wins 1.49x vs NT=64.
        const int nt128_min_wave = sm86_low_bw ? num_sms / 2 : num_sms;
        const int nt = (blocks_128x128 >= nt128_min_wave) ? 128 : 64;
        return {128, nt, 64};
    };

    auto [mt, nt, kc] = select_tile();

    // KC=128 requires K%128==0 (k_iters = K/K_CHUNK truncates the remainder).
    // Remap to the best valid KC=64 tile for this arch without changing any __trap() guards.
    if (kc == 128 && K % 128 != 0) {
        kc = 64;
        if (hbm_arch) {
            // MT=32 KC=64 tiles are trapped on all HBM arches; sm100 also traps MT=64 KC=64.
            nt = 64;
            mt = (cc_major == 10) ? 128 : 64;
        } else if (nt == 32) {
            nt = 64; // no 64x32-64 tile
        }
    }

    // clang-format off
#define LAUNCH_SM80(MT, NT, KC) \
    launch_tile<T, MT, NT, KC>(A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, gpu, stream)

    if (kc == 64) {
        if      (mt ==  32 && nt ==  64) LAUNCH_SM80( 32,  64,  64);
        else if (mt ==  32 && nt == 128) LAUNCH_SM80( 32, 128,  64);
        else if (mt ==  64 && nt ==  64) LAUNCH_SM80( 64,  64,  64);
        else if (mt ==  64 && nt == 128) LAUNCH_SM80( 64, 128,  64);
        else if (mt == 128 && nt ==  64) LAUNCH_SM80(128,  64,  64);
        else if (mt == 128 && nt == 128) LAUNCH_SM80(128, 128,  64);
    } else {
        if      (mt ==  32 && nt ==  64) LAUNCH_SM80( 32,  64, 128);
        else if (mt ==  32 && nt == 128) LAUNCH_SM80( 32, 128, 128);
        else if (mt ==  32 && nt == 256) LAUNCH_SM80( 32, 256, 128);
        else if (mt ==  64 && nt ==  32) LAUNCH_SM80( 64,  32, 128);
        else if (mt ==  64 && nt ==  64) LAUNCH_SM80( 64,  64, 128);
        // else if (mt ==  64 && nt == 128) LAUNCH_SM80( 64, 128, 128);  // unreachable: KC=128 at MT=64 requires NT=64
        // else if (mt == 128 && nt ==  64) LAUNCH_SM80(128,  64, 128);  // unreachable: MT=128 always dispatches KC=64
        // else if (mt == 128 && nt == 128) LAUNCH_SM80(128, 128, 128);  // unreachable: same
    }
#undef LAUNCH_SM80
    // clang-format on
}

// Explicit instantiations
template void launch_gemm_4bit_sm80_m16n8k16<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, __nv_bfloat16*,
    const __nv_bfloat16*, int, int, int, int, int, GpuProps, cudaStream_t
);
template void launch_gemm_4bit_sm80_m16n8k16<half>(
    const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int, int,
    int, int, int, GpuProps, cudaStream_t
);
