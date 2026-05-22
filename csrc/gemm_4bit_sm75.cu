// sm75 MMA (mma.sync.aligned.m16n8k8) 4-bit GEMM kernel. fp16 only.

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

#include "gemm_4bit_common.cuh"
#include "gemm_4bit_sm75.cuh"

static constexpr int K_CHUNK = 64;
[[maybe_unused]] static constexpr int MMA_M = 16;
[[maybe_unused]] static constexpr int MMA_N = 8;
[[maybe_unused]] static constexpr int MMA_K = 8;

static constexpr int NUM_WARPS = 8;
static constexpr int CTA_SIZE = NUM_WARPS * 32;

// K_CHUNK + 8 stride: 16-byte row alignment, limits bank conflicts to 4-way for K_CHUNK=64.
static constexpr int SMEM_A_STRIDE = K_CHUNK + 8; // 72 half elements
static constexpr int SMEM_B_STRIDE = K_CHUNK + 8;

/// @brief In-place warp-level MMA: accum += A * B (fp16, m16n8k8).
///        Called once per K chunk to accumulate the full matrix product.
///
/// Executes mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32.
/// Fragments are distributed across warp lanes per PTX spec.
///
/// @param accum In-place f32 accumulator (4 regs per lane)
/// @param a     A fragment: 16x8 fp16 operand (2 regs per lane)
/// @param b     B fragment: 8x8 fp16 operand (1 reg per lane)
__device__ __forceinline__ void mma_m16n8k8(FragC& accum, const uint32_t a[2], const uint32_t b[1]) {
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(accum.x[0]), "+f"(accum.x[1]), "+f"(accum.x[2]), "+f"(accum.x[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0])
    );
    // clang-format on
}

// A: m16 x k8 from row-major smem. ldmatrix.x2 loads two stacked m8xk8 halves.
// Thread t provides the address for row m_off + (t % 16) at column k_off.
// Lanes 16-31 mirror 0-15 (both sets address the same 16 rows for .x2).
__device__ __forceinline__ void load_A_frag(uint32_t frag[2], const half* smem_a, int m_off, int k_off, int lane) {
    const int m_row = m_off + (lane % 16);
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[m_row * SMEM_A_STRIDE + k_off]));
    // clang-format off
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(addr)
    );
    // clang-format on
}

// B: k8 x n8 from row-major smem (= col-major [k][n]). ldmatrix.x1 loads one m8xk8.
// Thread t addresses row n_off + (t % 8) at column k_off.
__device__ __forceinline__ void load_B_frag(uint32_t frag[1], const half* smem_b, int n_off, int k_off, int lane) {
    const int n_row = n_off + (lane % 8);
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[n_row * SMEM_B_STRIDE + k_off]));
    // clang-format off
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(frag[0])
        : "r"(addr)
    );
    // clang-format on
}

template <typename T, int MT, int NT> static constexpr int smem_bytes_for() {
    return 2 * (MT * SMEM_A_STRIDE + NT * SMEM_B_STRIDE) * static_cast<int>(sizeof(T));
}

/// @brief Fused 4-bit dequantize + MMA GEMM for sm75 (fp16 only).
///        Computes C[M,N] = A[M,K] @ B[N,K]^T + bias.
///
/// Layout:
///   A: [M, K] row-major, half (activations)
///   B: [N, K/2] row-major, packed uint8 (2 nibbles per byte, weights)
///   C: [M, N] row-major, half (output)
///
/// MMA: `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
///   A operand: row-major [M, K]
///   B operand: col-major [K, N] (B [N, K] row-major reinterpreted)
///
/// Double-buffered smem pipeline. Supports optional nested absmax and bias.
///
/// Smem per double-buffer (K_CHUNK=64, half=2 bytes):
///   MT=32, NT=64: 27 KB   MT=32, NT=128: 45 KB
///   MT=64, NT=64: 36 KB   MT=64, NT=128: 54 KB
///
/// @tparam T  Input/output dtype (must be `half`; static_assert enforces)
/// @tparam MT M tile size (32 or 64)
/// @tparam NT N tile size (64 or 128)
template <typename T, int MT = 32, int NT = 64>
__global__ void __launch_bounds__(CTA_SIZE) gemm_4bit_sm75_m16n8k8(
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
#if __CUDA_ARCH__ == 750
    static_assert(std::is_same_v<T, half>, "sm75 MMA requires fp16 (half)");
    static_assert(MT == 32 || MT == 64, "MT must be 32 or 64");
    static_assert(NT == 64 || NT == 128, "NT must be 64 or 128");

    using WL = MmaWarpLayout<MT, NT, NUM_WARPS, MMA_M, MMA_N>;
    constexpr int WARPS_M = WL::WARPS_M;
    constexpr int WARPS_N = WL::WARPS_N;
    constexpr int WARP_M = WL::WARP_M;
    constexpr int WARP_MMA_M = WL::WARP_MMA_M;
    constexpr int WARP_N = WL::WARP_N;
    constexpr int WARP_MMA_N = WL::WARP_MMA_N;

    static_assert(MT >= WARPS_M * MMA_M, "MT too small for warp layout");
    static_assert(NT >= WARPS_N * MMA_N, "NT too small for warp layout");

    // Packed bytes of B per thread per K-chunk (NT=64: 8 bytes; NT=128: 16 bytes).
    constexpr int B_BYTES = NT * (K_CHUNK / 2) / CTA_SIZE;
    static_assert(B_BYTES == 8 || B_BYTES == 16, "unexpected B bytes per thread");

    extern __shared__ char smem_raw[];
    constexpr int buf_offset = (MT * SMEM_A_STRIDE + NT * SMEM_B_STRIDE) * sizeof(T);
    auto smem_a_buf = [&](int buf) -> half* { return reinterpret_cast<half*>(smem_raw + buf * buf_offset); };
    auto smem_b_buf = [&](int buf) -> half* {
        return reinterpret_cast<half*>(smem_raw + buf * buf_offset + MT * SMEM_A_STRIDE * sizeof(T));
    };

    const float absmax_offset_f = absmax_8bit ? __ldg(absmax_offset) : 0.0f;

    const int bm = blockIdx.x * MT;
    const int bn = blockIdx.y * NT;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

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

    // Tile loading: A (direct copy from global) + B (load packed 4-bit + dequant to half2)
    auto load_tile = [&](int k_iter, int buf) {
        const int k_base = k_iter * K_CHUNK;
        half* __restrict__ sa = smem_a_buf(buf);
        half* __restrict__ sb = smem_b_buf(buf);

        // Load A: (MT/32) x uint4 per thread, .ca
        constexpr int vecs_per_thread = MT / 32;
#pragma unroll
        for (int v = 0; v < vecs_per_thread; v++) {
            const int vec_idx = threadIdx.x * vecs_per_thread + v;
            const int row = vec_idx / 8;
            const int col = (vec_idx % 8) * 8;
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

        // Load + dequant B
        {
            const int byte_start = threadIdx.x * B_BYTES;
            const int n_local = byte_start / (K_CHUNK / 2);
            const int k_byte0 = byte_start % (K_CHUNK / 2);
            const int n_global = bn + n_local;
            const int k_elem0 = k_byte0 * 2;

            float scale_f = 0.0f;
            if (n_global < N) {
                const int blk_idx = n_global * blocks_per_row + ((k_base + k_elem0) >> blocksize_log2);
                if (absmax_8bit) {
                    // absmax_8bit[blk_idx] indexes absmax_code (256-entry codebook).
                    // absmax[blk_idx >> 8] is the fp32 state2 scale (one per 256 blocks).
                    scale_f = __ldg(&absmax_code[__ldg(&absmax_8bit[blk_idx])]) * __ldg(&absmax[blk_idx >> 8]) +
                              absmax_offset_f;
                } else {
                    scale_f = __ldg(&absmax[blk_idx]);
                }
            }

            // Centroid * scale in fp32 avoids double rounding to half.
            auto dequant_byte = [&](uint8_t byte, int smem_off) {
                const float hi = __shfl_sync(0xffffffff, my_lut_f32, byte >> 4);
                const float lo = __shfl_sync(0xffffffff, my_lut_f32, byte & 0x0f);
                const auto dq = make_vec2<half>(hi * scale_f, lo * scale_f);
                *reinterpret_cast<uint32_t*>(&sb[n_local * SMEM_B_STRIDE + smem_off]) =
                    *reinterpret_cast<const uint32_t*>(&dq);
            };

            if constexpr (B_BYTES == 16) {
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
            } else {
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

    // Compute MMA on one buffer. K_CHUNK/MMA_K = 8 iterations.
    auto compute = [&](int buf) {
        const half* sa = smem_a_buf(buf);
        const half* sb = smem_b_buf(buf);
        const int wm_off = warp_m * WARP_M;
        const int wn_off = warp_n * WARP_N;

        for (int kk = 0; kk < K_CHUNK / MMA_K; kk++) {
            const int k_off = kk * MMA_K;

            uint32_t a_frag[WARP_MMA_M][2];
#pragma unroll
            for (int wm = 0; wm < WARP_MMA_M; wm++)
                load_A_frag(a_frag[wm], sa, wm_off + wm * MMA_M, k_off, lane_id);

#pragma unroll
            for (int wn = 0; wn < WARP_MMA_N; wn++) {
                uint32_t b_frag[1];
                load_B_frag(b_frag, sb, wn_off + wn * MMA_N, k_off, lane_id);
#pragma unroll
                for (int wm = 0; wm < WARP_MMA_M; wm++)
                    mma_m16n8k8(accum[wm][wn], a_frag[wm], b_frag);
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
#endif // __CUDA_ARCH__ == 750
}

template <typename T, int MT, int NT>
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
    constexpr int smem = smem_bytes_for<T, MT, NT>();
    static bool cfg[16] = {};
    if (gpu.device_index >= 16 || !cfg[gpu.device_index]) {
        cudaFuncSetAttribute(gemm_4bit_sm75_m16n8k8<T, MT, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (gpu.device_index < 16)
            cfg[gpu.device_index] = true;
    }
    dim3 grid((M + MT - 1) / MT, (N + NT - 1) / NT);
    gemm_4bit_sm75_m16n8k8<T, MT, NT><<<grid, dim3(CTA_SIZE), smem, stream>>>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type
    );
}

/// @brief Auto-dispatch launcher for the sm75 MMA kernel. Selects MT/NT tile
///        based on GPU SM count and shape.
/// @tparam T Input/output dtype (must be `half`)
template <typename T>
void launch_gemm_4bit_sm75_m16n8k8(
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
    static_assert(std::is_same_v<T, half>, "sm75 MMA requires fp16 (half)");

    const int num_sms = gpu.num_sms;
    const int m_blocks_32 = (M + 31) / 32;
    const int m_blocks_64 = (M + 63) / 64;
    const int n_blocks_128 = (N + 127) / 128;
    const int n_blocks_64 = (N + 63) / 64;
    const int blocks_32x64 = m_blocks_32 * n_blocks_64;

    // Suppress MT=32 when M just crossed a 64-row boundary and the extra
    // MT=32 block would be nearly empty -- but only once 64x128 reaches 0.25 wave.
    // Below 0.25 wave, MT=32 still wins by keeping more blocks in flight.
    const bool mt32_boundary_waste = (m_blocks_32 > m_blocks_64) && (m_blocks_64 * n_blocks_128 >= num_sms / 4);

    const bool use_mt32 = (M < 48 && !mt32_boundary_waste) ||
                          (M <= 128 && blocks_32x64 >= num_sms * 2 && m_blocks_32 * 32 < m_blocks_64 * 64 &&
                           m_blocks_64 * n_blocks_128 < num_sms * 3) ||
                          (m_blocks_64 * n_blocks_128 < num_sms / 4); // 64x128 < 0.25 wave

    int m_blocks = use_mt32 ? m_blocks_32 : m_blocks_64;
    int mt = use_mt32 ? 32 : 64;
    int nt;

    if (mt == 32) {
        // NT=128 only at very high occupancy (>=5 waves); NT=64 otherwise gives
        // 2x more blocks and wins at normal occupancy on sm75.
        const bool use_nt64 = (m_blocks * n_blocks_128 < num_sms * 5) && (n_blocks_64 > n_blocks_128);
        nt = use_nt64 ? 64 : 128;
    } else {
        // Fall back to MT=32 when 64x128 is severely undersubscribed (< 0.25 wave).
        if (m_blocks * n_blocks_128 < num_sms / 4 && n_blocks_64 > n_blocks_128) {
            m_blocks = m_blocks_32;
            mt = 32;
            const bool use_nt64 = (m_blocks * n_blocks_128 < num_sms * 5) && (n_blocks_64 > n_blocks_128);
            nt = use_nt64 ? 64 : 128;
        } else {
            // 64x128 otherwise, except 64x64 when NT=128 is below 0.5 wave.
            nt = (m_blocks * n_blocks_128 < num_sms / 2) ? 64 : 128;
        }
    }

    // clang-format off
#define LAUNCH_SM75(MT, NT) \
    launch_tile<T, MT, NT>(A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, gpu, stream)

    if      (mt == 32 && nt ==  64) LAUNCH_SM75(32,  64);
    else if (mt == 32 && nt == 128) LAUNCH_SM75(32, 128);
    else if (mt == 64 && nt ==  64) LAUNCH_SM75(64,  64);
    else if (mt == 64 && nt == 128) LAUNCH_SM75(64, 128);
#undef LAUNCH_SM75
    // clang-format on
}

// Explicit instantiation
template void launch_gemm_4bit_sm75_m16n8k8<half>(
    const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int, int,
    int, int, int, GpuProps, cudaStream_t
);
