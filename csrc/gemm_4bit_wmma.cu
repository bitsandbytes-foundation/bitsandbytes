// AMD WMMA 4-bit GEMM kernel for RDNA3 (gfx110x) and RDNA4 (gfx120x), wave32.
// Computes C[M,N] = A[M,K] @ B[N,K]^T + bias with fused 4-bit dequantization.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "common.cuh"
#include "gemm_4bit_common.cuh"
#include "gemm_4bit_wmma.cuh"

// RDNA3.5 (gfx115x) shares the gfx11 WMMA encoding and half-wave behaviour
//, so it runs the gfx11 kernels.
#define BNB_WMMA_RDNA3 (IS_RDNA3 || IS_RDNA3_5)
#define BNB_WMMA_RDNA4 IS_RDNA4

// WMMA generation the current device pass compiles kernel bodies for (0 = none,
// including the host pass). Kernels are templated on the generation so the host
// can size tiles for the device it runs on.
#if BNB_WMMA_RDNA3
#define BNB_WMMA_DEVICE_ARCH 11
#elif BNB_WMMA_RDNA4
#define BNB_WMMA_DEVICE_ARCH 12
#else
#define BNB_WMMA_DEVICE_ARCH 0
#endif

// Which kernel generations a compilation pass instantiates: the host pass needs every
// launch stub, a device pass only the kernels of its own generation, so no target's code
// object carries empty kernels for the other one (non-WMMA targets carry none).
#ifdef __HIP_DEVICE_COMPILE__
#define BNB_WMMA_EMIT_11 (BNB_WMMA_DEVICE_ARCH == 11)
#define BNB_WMMA_EMIT_12 (BNB_WMMA_DEVICE_ARCH == 12)
#else
#define BNB_WMMA_EMIT_11 1
#define BNB_WMMA_EMIT_12 1
#endif

namespace {

// K elements per warp per 64-K unit: each half-wave takes 32 contiguous K.
// A main-loop iteration consumes KCH units, i.e. 64*KCH K per warp.
constexpr int WMMA_KC = 64;

// Native vector types: HIP's uint4 is a class, and copying it through private
// arrays lowers to memcpy that stays in scratch memory.
using wmma_v2u = uint32_t __attribute__((ext_vector_type(2)));
using wmma_v4u = uint32_t __attribute__((ext_vector_type(4)));
using wmma_v8f = float __attribute__((ext_vector_type(8)));
using wmma_v8u = uint32_t __attribute__((ext_vector_type(8)));
using wmma_v8h = _Float16 __attribute__((ext_vector_type(8)));
using wmma_v8s = short __attribute__((ext_vector_type(8)));
using wmma_v16h = _Float16 __attribute__((ext_vector_type(16)));
using wmma_v16s = short __attribute__((ext_vector_type(16)));

// Output rows per WMMA that carry real data: gfx11 runs two 8-row half-wave
// products (see wmma_16x16x16), gfx12 one full 16x16x16 product.
constexpr int wmma_rows(int arch) { return arch == 12 ? 16 : 8; }

// Tuning targets returned by gemm_4bit_wmma_arch(): 11 (gfx110x), 115 (gfx115x, RDNA3.5)
// and 12 (gfx120x). gfx115x runs the gfx11 kernels with its own tile rules.
constexpr int WMMA_TARGET_GFX115X = 115;
// gfx1103 (Radeon 760M/780M iGPU): gfx110x kernels and tile rules, own fp32 dispatch (see
// csrc/gemm_4bit.cu and the Python cutover).
constexpr int WMMA_TARGET_GFX1103 = 113;

constexpr int wmma_gen(int target) {
    return target == WMMA_TARGET_GFX115X || target == WMMA_TARGET_GFX1103 ? 11 : target;
}

// Per-lane operand fragment: 16 (gfx11) or 8 (gfx12) 16-bit values.
template <int ARCH> using wmma_frag = std::conditional_t<ARCH == 12, wmma_v4u, wmma_v8u>;

/// @brief One v_wmma_f32_16x16x16_{f16,bf16} on 16-bit operand bits (bf16 for
/// T = bnb_bfloat16 and for the split fp32 path).
///
/// gfx11 executes a wave32 WMMA as two independent 8x16x16 products: lanes 0-15
/// produce the even output rows from their own B copy and the A rows held by the
/// even lanes, lanes 16-31 produce the odd rows from their own B copy and the odd
/// lanes' A rows. The ISA asks for replicated operands, but because each half only
/// reads its own copy, the kernel feeds the two halves different K slices and sums
/// the halves once at the end. Every weight nibble is decoded exactly once.
///
/// gfx12 has no replication: lane L holds A row L%16 (B column L%16) at K
/// 8*(L/16) + e for its 8 elements e, and accumulator element j holds row
/// j + 8*(L/16), column L%16.
/// Since only the pairing of A and B K indices matters, each half-wave again
/// supplies its own K slice, now summed inside the WMMA.
template <int ARCH, typename T>
__device__ __forceinline__ wmma_v8f wmma_16x16x16(wmma_frag<ARCH> a, wmma_frag<ARCH> b, wmma_v8f c) {
    if constexpr (ARCH == 11) {
#if BNB_WMMA_RDNA3
        if constexpr (std::is_same_v<T, half>) {
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                __builtin_bit_cast(wmma_v16h, a), __builtin_bit_cast(wmma_v16h, b), c
            );
        } else {
            return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                __builtin_bit_cast(wmma_v16s, a), __builtin_bit_cast(wmma_v16s, b), c
            );
        }
#endif
    } else {
#if BNB_WMMA_RDNA4
        if constexpr (std::is_same_v<T, half>) {
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                __builtin_bit_cast(wmma_v8h, a), __builtin_bit_cast(wmma_v8h, b), c
            );
        } else {
            return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                __builtin_bit_cast(wmma_v8s, a), __builtin_bit_cast(wmma_v8s, b), c
            );
        }
#endif
    }
    return c;
}

// fp32 operands are split into this many bf16 parts (~24 mantissa bits).
constexpr int WMMA_F32_PARTS = 3;

// Split an fp32 value into P bf16 parts (bit patterns in the low 16 bits) whose sum
// is x to ~2^-(8P) relative: every part but the last truncates, so each remainder
// is exact in fp32; the last rounds to nearest even. Inf/NaN keep the lower parts at
// zero so they propagate through the leading part alone.
template <int P> __device__ __forceinline__ void split_bf16(float x, uint32_t (&parts)[P]) {
    const bool finite = (__builtin_bit_cast(uint32_t, x) & 0x7f800000u) != 0x7f800000u;
    float r = x;
#pragma unroll
    for (int p = 0; p < P - 1; p++) {
        const uint32_t bits = __builtin_bit_cast(uint32_t, r);
        parts[p] = bits >> 16;
        r = r - __builtin_bit_cast(float, bits & 0xffff0000u);
    }
    const uint32_t bits = __builtin_bit_cast(uint32_t, r);
    parts[P - 1] = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    if (!finite) {
#pragma unroll
        for (int p = 1; p < P; p++)
            parts[p] = 0u;
    }
}

// Add the value held by the lane 16 positions away (the other half-wave).
__device__ __forceinline__ float half_wave_sum(float v) {
#if BNB_WMMA_RDNA3
    const uint32_t other = __builtin_amdgcn_permlanex16(
        __builtin_bit_cast(uint32_t, v), __builtin_bit_cast(uint32_t, v), 0x76543210u, 0xfedcba98u, false, false
    );
    return v + __builtin_bit_cast(float, other);
#else
    return v;
#endif
}

// LDS bytes for the A stages (two stages of `planes` 16-bit tiles: 1, or WMMA_F32_PARTS = 3
// for the fp32 hi/mid/lo split) or, reusing the same space, the split-K reduction tile,
// whichever is larger. The staged A rows are padded by 8 elements: keeps 16-byte
// alignment and spreads rows across LDS banks.
constexpr int wmma_smem_bytes(int mt, int nt, int ksplit, int kch, int planes) {
    const int a_bytes = 2 * planes * ksplit * mt * (WMMA_KC * kch + 8) * 2;
    const int red_bytes = ksplit > 1 ? ksplit * mt * nt * 4 : 0;
    return a_bytes > red_bytes ? a_bytes : red_bytes;
}

} // namespace

/// @brief Fused 4-bit dequantize + WMMA GEMM for RDNA3 and RDNA4 (bf16, fp16 and fp32).
///
/// fp32 has no WMMA operand type, so A and the centroids are split into three bf16
/// parts (hi + mid + lo, ~24 mantissa bits) and each tile accumulates the six
/// products whose part indices sum to < 3 in fp32 -- six WMMAs per 16-bit one,
/// with fp32-level accuracy.
///
/// Block: KSPLIT x WARPS_M x WARPS_N warps; block tile MT = WARPS_M*R*WM rows x
/// NT = WARPS_N*16*WN columns, R = wmma_rows(ARCH). Each warp owns an (R*WM) x
/// (16*WN) sub-tile; the KSPLIT warp groups split K in interleaved KC = 64*KCH
/// chunks and are reduced through LDS in a fixed order (deterministic). A gfx11
/// WMMA covers 8 real rows x 16 columns x 32 K (16 K per half-wave), a gfx12 WMMA
/// 16 rows x 16 columns x 16 K (8 K per half-wave).
///
/// Per iteration the block stages the A tile of its KSPLIT current chunks in LDS
/// (double-buffered, shared by all warps of a K group). Each lane (half h =
/// lane/16, c = lane%16) streams its own B columns straight from global memory,
/// 32 K per 64-K unit u = 0..KCH-1 of the chunk starting at k0:
///   B: column n; gfx11: K [k0 + 32*KCH*h + 32*u, +32), one 16-byte load per unit
///      (each half-wave is its own product, so it may use its own absmax);
///      gfx12: K [k0 + 64*u + 32*h, +32), one 16-byte load, or for blocksize 32
///      K [k0 + 64*u + 16*h, +16) and [k0 + 64*u + 32 + 16*h, +16), two 8-byte
///      loads (both halves feed every WMMA, so their K must share an absmax).
///      Decoded through a 256-entry LDS LUT (byte -> packed pair of centroids).
///   A: row c/2 (gfx11) or c (gfx12) of each R-row sub-tile, same K range, from LDS.
///   Scale: absmax for that column/K range, applied to the fp32 accumulator after
///      the chunk's WMMAs (centroids enter the WMMA unscaled).
/// The next iteration's A and B loads are issued before the current chunk is
/// consumed; A is issued first so storing it to LDS never waits on the B loads.
///
/// Requires K % (64*KCH) == 0 and blocksize >= 32.
///
/// @tparam ARCH    WMMA generation: 11 (gfx110x) or 12 (gfx120x)
/// @tparam T       Input/output dtype (`bnb_bfloat16`, `half`, or `float`)
/// @tparam WM      R-row M sub-tiles per warp
/// @tparam WN      16-column N sub-tiles per warp
/// @tparam WARPS_M Warps along M
/// @tparam WARPS_N Warps along N
/// @tparam KSPLIT  Warp groups along K
/// @tparam KCH     64-K units per main-loop iteration
/// @tparam PRESCALE gfx11 fp16: round scaled B operands to half and accumulate across groups.
/// @tparam GS      Consecutive WMMAs (per chain) sharing one absmax: each group's WMMAs
///                 chain into one fp32 tile that is scaled once.
template <
    int ARCH, typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH, int GS, bool PRESCALE = false>
__device__ __forceinline__ void gemm_4bit_wmma_body(
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
    int quant_type,                            // 1 = FP4, 2 = NF4
    bool row_tiles_first                       // block order, see below
    // clang-format on
) {
    constexpr bool GFX12 = ARCH == 12;
    constexpr int R = wmma_rows(ARCH);
    constexpr int MT = WARPS_M * R * WM;
    constexpr int NT = WARPS_N * 16 * WN;
    constexpr int NUM_THREADS = KSPLIT * WARPS_M * WARPS_N * 32;
    constexpr int KC = WMMA_KC * KCH; // K per warp per iteration
    constexpr int A_STRIDE = KC + 8;  // LDS row stride (elements)
    constexpr bool F32 = std::is_same_v<T, float>;
    static_assert(!PRESCALE || (ARCH == 11 && std::is_same_v<T, half>));
    constexpr int PLANES = F32 ? WMMA_F32_PARTS : 1; // fp32: one bf16 tile per split part
    constexpr int A_PLANE = KSPLIT * MT * A_STRIDE;  // elements per A tile
    constexpr int A_STAGE = PLANES * A_PLANE;        // elements per A stage

    // byte -> {centroid[hi nibble] (lower K), centroid[lo nibble] (higher K)} as
    // 16-bit operand bits; fp32 keeps one table per bf16 split part.
    __shared__ uint32_t lut[PLANES][256];
    // Codebook for the nested (8-bit) absmax, so the per-chunk scale needs no
    // dependent global load.
    __shared__ float code_lds[256];
    // A stages [2][KSPLIT][MT][A_STRIDE]; reused for the split-K reduction.
    __shared__ __attribute__((aligned(16))) uint8_t smem[wmma_smem_bytes(MT, NT, KSPLIT, KCH, PLANES)];
    uint16_t* a_lds = reinterpret_cast<uint16_t*>(smem);

    const bool nested = absmax_8bit != nullptr;
    {
        const float* centroids = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
        for (int i = threadIdx.x; i < 256; i += NUM_THREADS) {
            if constexpr (F32) {
                uint32_t c0[PLANES], c1[PLANES];
                split_bf16<PLANES>(centroids[i >> 4], c0);
                split_bf16<PLANES>(centroids[i & 0x0f], c1);
#pragma unroll
                for (int p = 0; p < PLANES; p++)
                    lut[p][i] = c0[p] | (c1[p] << 16);
            } else {
                const T lo = static_cast<T>(centroids[i >> 4]);
                const T hi = static_cast<T>(centroids[i & 0x0f]);
                lut[0][i] = (uint32_t)(*reinterpret_cast<const uint16_t*>(&lo)) |
                            ((uint32_t)(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
            }
            if (nested)
                code_lds[i] = __ldg(&absmax_code[i]);
        }
    }

    const float absmax_offset_f = nested ? __ldg(absmax_offset) : 0.0f;
    const int blocksize_log2 = __ffs(blocksize) - 1;
    const int blocks_per_row = K >> blocksize_log2;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_n = warp % WARPS_N;
    const int warp_m = (warp / WARPS_N) % WARPS_M;
    const int warp_k = warp / (WARPS_N * WARPS_M);
    const int half_id = lane >> 4;
    const int col = lane & 15;

    // Block order. Walking N first makes every row tile stream all of B from DRAM again
    // once B no longer fits in the last-level cache. With row_tiles_first, the row tiles of
    // the same N columns get consecutive block ids, so they are resident together and share
    // each B chunk through L2 (see wmma_row_tiles_first).
    int bx = blockIdx.x, by = blockIdx.y;
    if (row_tiles_first) {
        const int lin = blockIdx.x + gridDim.x * blockIdx.y;
        bx = lin / gridDim.y;
        by = lin % gridDim.y;
    }
    const int bn = bx * NT;
    const int bm = by * MT;

    // Per-lane weight rows (clamped so tail lanes read valid memory; never stored).
    const uint8_t* b_row[WN];
    int blk_row[WN];
#pragma unroll
    for (int wn = 0; wn < WN; wn++) {
        const int n = min(bn + (warp_n * WN + wn) * 16 + col, N - 1);
        b_row[wn] = B + (size_t)n * (K / 2);
        blk_row[wn] = n * blocks_per_row;
    }

    const int num_chunks = K / KC;
    const int num_iters = (num_chunks + KSPLIT - 1) / KSPLIT;

    // Blocks start at different K chunks so that concurrently resident blocks
    // stream different K offsets of their rows. With power-of-two row strides
    // (K/2 bytes), lock-stepped blocks otherwise camp on a few DRAM channels.
    const int chunk_rot = bx % num_chunks;
    // First K element of warp group g's chunk at iteration i. Past-the-end chunks
    // (last iteration when KSPLIT does not divide num_chunks) alias a valid chunk:
    // they are loaded but never consumed, which keeps every load unconditional.
    auto chunk_valid = [&](int i, int g) { return i * KSPLIT + g < num_chunks; };
    auto chunk_k0 = [&](int i, int g) {
        int ch = i * KSPLIT + g + chunk_rot;
        while (ch >= num_chunks)
            ch -= num_chunks;
        return ch * KC;
    };

    // Cooperative A tile copy: KSPLIT x MT rows x KC/VEC_ELEMS 16-byte vectors per
    // stage (fp32 vectors are split into the hi/lo tiles on the way to LDS). Rows
    // past M are clamped to a valid row: WMMA output rows only depend on their own A
    // row, and those rows are never stored.
    constexpr int VEC_ELEMS = 16 / (int)sizeof(T);
    constexpr int ROW_VECS = KC / VEC_ELEMS;
    constexpr int A_VECS = KSPLIT * MT * ROW_VECS;
    constexpr int A_VECS_PER_THREAD = (A_VECS + NUM_THREADS - 1) / NUM_THREADS;
    wmma_v4u a_stage[A_VECS_PER_THREAD];
    auto issue_a = [&](int i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < A_VECS_PER_THREAD; j++) {
            const int v = threadIdx.x + j * NUM_THREADS;
            if (A_VECS % NUM_THREADS == 0 || v < A_VECS) {
                const int part = v % ROW_VECS;
                const int row = (v / ROW_VECS) % MT;
                const int g = (v / ROW_VECS) / MT;
                a_stage[j] = *reinterpret_cast<const wmma_v4u*>(
                    A + (size_t)min(bm + row, M - 1) * K + chunk_k0(i, g) + part * VEC_ELEMS
                );
            }
        }
    };
    auto store_a = [&](int buf) {
#pragma clang loop unroll(full)
        for (int j = 0; j < A_VECS_PER_THREAD; j++) {
            const int v = threadIdx.x + j * NUM_THREADS;
            if (A_VECS % NUM_THREADS == 0 || v < A_VECS) {
                const int part = v % ROW_VECS;
                const int rg = v / ROW_VECS; // g * MT + row
                uint16_t* dst = &a_lds[buf * A_STAGE + rg * A_STRIDE + part * VEC_ELEMS];
                if constexpr (F32) {
                    uint32_t parts[4][PLANES];
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        // Copy the element out first: __builtin_bit_cast on a vector element
                        // lvalue reads element 0 with this clang.
                        const uint32_t bits = a_stage[j][e];
                        split_bf16<PLANES>(__builtin_bit_cast(float, bits), parts[e]);
                    }
#pragma unroll
                    for (int p = 0; p < PLANES; p++)
                        *reinterpret_cast<wmma_v2u*>(dst + p * A_PLANE) =
                            wmma_v2u{parts[0][p] | (parts[1][p] << 16), parts[2][p] | (parts[3][p] << 16)};
                } else {
                    *reinterpret_cast<wmma_v4u*>(dst) = a_stage[j];
                }
            }
        }
    };

    // Each lane decodes 32 K (16 bytes) of its column per 64-K unit u, as SUBS
    // WMMAs of 16-bit words (FRAG_WORDS per operand fragment):
    //   gfx11: half h takes K [32*KCH*h + 32*u, +32), 2 WMMAs of 16 K; the halves
    //          are separate products, so each half may use its own absmax.
    //   gfx12: both halves feed every WMMA, so their K must share an absmax. Half h
    //          takes K [64*u + 32*h, +32) when blocksize >= 64 (one 16-byte load),
    //          or K [64*u + 16*h, +16) and [64*u + 32 + 16*h, +16) when blocksize
    //          is 32 (two 8-byte loads; WMMAs 0-1 then read the first 32-K block,
    //          WMMAs 2-3 the second). 4 WMMAs of 8 K per lane.
    using Frag = wmma_frag<ARCH>;
    constexpr int SUBS = GFX12 ? 4 : 2;
    constexpr int FRAG_WORDS = 16 / SUBS;
    constexpr bool SPLIT32 = GFX12 && GS < SUBS; // gfx12, blocksize 32
    constexpr int NG = KCH * SUBS / GS;          // scale groups per iteration
    // WMMAs per group within one unit. gfx12 fp32 (split bf16) decodes one fragment at a
    // time to save registers; other paths decode the whole group.
    constexpr int GROUP_SUBS = ARCH == 12 && F32 ? 1 : (GS < SUBS ? GS : SUBS);
    constexpr int UNIT_K = GFX12 ? 64 : 32; // K between a lane's consecutive units
    // First K of the lane's slice in unit 0 (relative to the chunk start).
    const int k_lane = GFX12 ? (SPLIT32 ? 16 : 32) * half_id : 32 * KCH * half_id;
    // K offset of the lane's WMMA t (flat index u * SUBS + s) from k_lane.
    auto k_sub = [](int t) {
        const int s = t % SUBS;
        return UNIT_K * (t / SUBS) + (SPLIT32 && s >= 2 ? 16 : 0) + s * (32 / SUBS);
    };

    struct BStage {
        wmma_v4u b[WN][KCH];
        float scale[WN][NG];   // absmax, or the fp32 state2 absmax when nested
        uint32_t qidx[WN][NG]; // nested: 8-bit absmax code index
    };

    auto issue_b = [&](int i, BStage& st) {
        const int k_half = chunk_k0(i, warp_k) + k_lane;
#pragma unroll
        for (int wn = 0; wn < WN; wn++) {
#pragma unroll
            for (int u = 0; u < KCH; u++) {
                const uint8_t* p = b_row[wn] + (k_half + u * UNIT_K) / 2;
                if constexpr (SPLIT32) {
                    const wmma_v2u lo = *reinterpret_cast<const wmma_v2u*>(p);
                    const wmma_v2u hi = *reinterpret_cast<const wmma_v2u*>(p + 16);
                    st.b[wn][u] = wmma_v4u{lo.x, lo.y, hi.x, hi.y};
                } else {
                    st.b[wn][u] = *reinterpret_cast<const wmma_v4u*>(p);
                }
            }
#pragma unroll
            for (int g = 0; g < NG; g++) {
                const int blk = blk_row[wn] + ((k_half + k_sub(g * GS)) >> blocksize_log2);
                st.qidx[wn][g] = nested ? absmax_8bit[blk] : 0u;
                st.scale[wn][g] = __ldg(&absmax[nested ? blk >> 8 : blk]);
            }
        }
    };

    wmma_v8f acc[WM][WN];
#pragma unroll
    for (int wm = 0; wm < WM; wm++)
#pragma unroll
        for (int wn = 0; wn < WN; wn++)
            acc[wm][wn] = wmma_v8f{};

    auto compute = [&](int buf, const BStage& st) {
        const uint16_t* a_base =
            a_lds + buf * A_STAGE + (warp_k * MT + warp_m * R * WM + (GFX12 ? col : col >> 1)) * A_STRIDE + k_lane;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            float scale[WN];
#pragma unroll
            for (int wn = 0; wn < WN; wn++)
                scale[wn] = nested ? code_lds[st.qidx[wn][g]] * st.scale[wn][g] + absmax_offset_f : st.scale[wn][g];

            // Issue order keeps WM*WN independent WMMA chains in flight; the group's
            // GS WMMAs per chain accumulate unscaled and are scaled once. A fragments
            // stream from LDS one R-row sub-tile at a time.
            wmma_v8f tmp[WM][WN];
#pragma unroll
            for (int t0 = g * GS; t0 < (g + 1) * GS; t0 += GROUP_SUBS) {
                const int u = t0 / SUBS;
                Frag b_frag[PLANES][WN][GROUP_SUBS];
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
#pragma unroll
                    for (int s = 0; s < GROUP_SUBS; s++)
#pragma unroll
                        for (int i = 0; i < FRAG_WORDS; i++) {
                            const int b = (t0 % SUBS + s) * FRAG_WORDS + i; // byte within the 16-byte load
                            const int byte = (st.b[wn][u][b / 4] >> (8 * (b % 4))) & 0xff;
#pragma unroll
                            for (int p = 0; p < PLANES; p++) {
                                uint32_t pair = lut[p][byte];
                                if constexpr (PRESCALE) {
                                    // Shared across WM row tiles; avoids their temporary scaled sums.
                                    const half2 v = __builtin_bit_cast(half2, pair);
                                    const half2 factor = __float2half2_rn(scale[wn]);
                                    pair = __builtin_bit_cast(uint32_t, __hmul2(v, factor));
                                }
                                b_frag[p][wn][s][i] = pair;
                            }
                        }
#pragma unroll
                for (int s = 0; s < GROUP_SUBS; s++)
#pragma unroll
                    for (int wm = 0; wm < WM; wm++) {
                        const uint16_t* ap = a_base + wm * R * A_STRIDE + k_sub(t0 + s);
                        const Frag a_frag = *reinterpret_cast<const Frag*>(ap);
                        const bool first = t0 == g * GS && s == 0;
#pragma unroll
                        for (int wn = 0; wn < WN; wn++) {
                            wmma_v8f c = PRESCALE ? acc[wm][wn] : (first ? wmma_v8f{} : tmp[wm][wn]);
                            if constexpr (F32) {
                                // Products of split parts whose indices sum to < 3, smallest
                                // first; the leading hi*hi product comes last (below).
                                const Frag a_mid = *reinterpret_cast<const Frag*>(ap + A_PLANE);
                                const Frag a_lo = *reinterpret_cast<const Frag*>(ap + 2 * A_PLANE);
                                c = wmma_16x16x16<ARCH, T>(a_lo, b_frag[0][wn][s], c);
                                c = wmma_16x16x16<ARCH, T>(a_frag, b_frag[2][wn][s], c);
                                c = wmma_16x16x16<ARCH, T>(a_mid, b_frag[1][wn][s], c);
                                c = wmma_16x16x16<ARCH, T>(a_mid, b_frag[0][wn][s], c);
                                c = wmma_16x16x16<ARCH, T>(a_frag, b_frag[1][wn][s], c);
                            }
                            if constexpr (PRESCALE)
                                acc[wm][wn] = wmma_16x16x16<ARCH, T>(a_frag, b_frag[0][wn][s], c);
                            else
                                tmp[wm][wn] = wmma_16x16x16<ARCH, T>(a_frag, b_frag[0][wn][s], c);
                        }
                    }
            }
#pragma unroll
            for (int wm = 0; wm < WM; wm++)
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
                    if constexpr (!PRESCALE)
                        acc[wm][wn] += tmp[wm][wn] * scale[wn];
        }
    };

    BStage b_cur, b_nxt;
    issue_a(0);
    issue_b(0, b_cur);
    store_a(0);
    __syncthreads();

    for (int it = 0; it < num_iters; it++) {
        const bool has_next = it + 1 < num_iters;
        if (has_next) {
            issue_a(it + 1);
            issue_b(it + 1, b_nxt);
        }
        if (chunk_valid(it, warp_k))
            compute(it & 1, b_cur);
        if (has_next)
            store_a((it + 1) & 1);
        __syncthreads();
        b_cur = b_nxt;
    }

    // gfx11: combine the two half-wave K slices, so every lane holds full-K sums
    // for rows j (element j) and column `col`; the upper half-wave stores nothing.
    // gfx12: element j of lane (h, c) already holds the full sum for row j + 8h.
    if constexpr (!GFX12) {
#pragma unroll
        for (int wm = 0; wm < WM; wm++)
#pragma unroll
            for (int wn = 0; wn < WN; wn++)
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[wm][wn][j] = half_wave_sum(acc[wm][wn][j]);
    }
    const bool stores = GFX12 || half_id == 0;
    const int row_lane = GFX12 ? 8 * half_id : 0; // row of element 0 within a sub-tile

    const int m_warp = warp_m * R * WM + row_lane; // lane's first row within the block tile
    const int n_warp = warp_n * 16 * WN;
    if constexpr (KSPLIT == 1) {
        if (stores) {
#pragma unroll
            for (int wn = 0; wn < WN; wn++) {
                const int n = bn + n_warp + wn * 16 + col;
                if (n >= N)
                    continue;
                const float bias_f = bias ? static_cast<float>(bias[n]) : 0.0f;
#pragma unroll
                for (int wm = 0; wm < WM; wm++)
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int m = bm + m_warp + wm * R + j;
                        if (m < M)
                            C[(size_t)m * N + n] = static_cast<T>(acc[wm][wn][j] + bias_f);
                    }
            }
        }
    } else {
        // Partial tiles [KSPLIT][MT][NT] in LDS (the A stages are no longer needed;
        // the loop's trailing barrier ordered all reads before these writes).
        float* red = reinterpret_cast<float*>(smem);
        if (stores) {
#pragma unroll
            for (int wm = 0; wm < WM; wm++)
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
#pragma unroll
                    for (int j = 0; j < 8; j++)
                        red[(warp_k * MT + m_warp + wm * R + j) * NT + n_warp + wn * 16 + col] = acc[wm][wn][j];
        }
        __syncthreads();

        for (int o = threadIdx.x; o < MT * NT; o += NUM_THREADS) {
            const int m = bm + o / NT;
            const int n = bn + o % NT;
            if (m >= M || n >= N)
                continue;
            float sum = 0.0f;
#pragma unroll
            for (int g = 0; g < KSPLIT; g++)
                sum += red[g * MT * NT + o];
            if (bias)
                sum += static_cast<float>(bias[n]);
            C[(size_t)m * N + n] = static_cast<T>(sum);
        }
    }
}

/// Kernel entry: the body is only emitted in the device pass of the matching WMMA
/// generation (see gemm_4bit_wmma_body for the parameters).
template <
    int ARCH, typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH, int GS, bool PRESCALE = false>
__global__ void __launch_bounds__(KSPLIT* WARPS_M* WARPS_N * 32) gemm_4bit_wmma(
    const T* __restrict__ A, const uint8_t* __restrict__ B, const float* __restrict__ absmax,
    const uint8_t* __restrict__ absmax_8bit, const float* __restrict__ absmax_code,
    const float* __restrict__ absmax_offset, T* __restrict__ C, const T* __restrict__ bias, int M, int N, int K,
    int blocksize, int quant_type, bool row_tiles_first
) {
#if BNB_WMMA_DEVICE_ARCH
    if constexpr (ARCH == BNB_WMMA_DEVICE_ARCH)
        gemm_4bit_wmma_body<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH, GS, PRESCALE>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type,
            row_tiles_first
        );
#endif
}

namespace {

// WMMA tuning target for a gcnArchName ("gfx1201" or "gfx1201:xnack-"): 11, 113, 115, 12, or 0.
// The chips must be ones whose device pass emits the kernel body (IS_RDNA3 / IS_RDNA4 in
// common.cuh); any other device would launch an empty kernel. Measured: gfx1100, gfx1103,
// gfx1151 and gfx1201. gfx1101/gfx1102, gfx1150/gfx1152/gfx1153 and gfx1200 share the WMMA
// instructions of their family and use its rules (gfx1100, gfx1151, gfx1201) unmeasured.
int wmma_arch_of(const char* name) {
    static constexpr struct {
        const char* chip;
        int arch;
    } chips[] = {
        {"gfx1100", 11},
        {"gfx1101", 11},
        {"gfx1102", 11},
        {"gfx1103", WMMA_TARGET_GFX1103},
        {"gfx1150", WMMA_TARGET_GFX115X},
        {"gfx1151", WMMA_TARGET_GFX115X},
        {"gfx1152", WMMA_TARGET_GFX115X},
        {"gfx1153", WMMA_TARGET_GFX115X},
        {"gfx1200", 12},
        {"gfx1201", 12}
    };

    for (const auto& c : chips) {
        const size_t n = std::strlen(c.chip);
        if (std::strncmp(name, c.chip, n) == 0 && (name[n] == '\0' || name[n] == ':'))
            return c.arch;
    }
    return 0;
}

// Cached per device (devices 0-15; later ones are queried on every call).
int wmma_device_arch() {
    static std::atomic<int> cache[16] = {}; // 0 = unknown, else arch + 1
    int dev = 0;
    (void)hipGetDevice(&dev);
    if (dev < 16) {
        const int cached = cache[dev].load(std::memory_order_relaxed);
        if (cached != 0)
            return cached - 1;
    }
    hipDeviceProp_t prop;
    const int arch = hipGetDeviceProperties(&prop, dev) == hipSuccess ? wmma_arch_of(prop.gcnArchName) : 0;
    if (dev < 16)
        cache[dev].store(arch + 1, std::memory_order_relaxed);
    return arch;
}

/// Whether blocks walk the row tiles of the same N columns first, so they share B in L2
/// (see gemm_4bit_wmma_body). Only matters with two or more row tiles. fp32 on gfx110x
/// stays N-first. The tile rules are fitted with this order.
bool wmma_row_tiles_first(int target, bool fp32) {
    if (target == 11 || target == WMMA_TARGET_GFX1103)
        return !fp32;
    return target == 12 || target == WMMA_TARGET_GFX115X;
}

/// Launch one tile configuration. KCH_MAX is lowered to 1 when K is not a multiple
/// of 128; one absmax covers each half-wave's per-iteration K slice when
/// blocksize >= 32 * KCH.
template <
    int ARCH, typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH_MAX, bool PRESCALE = false>
void launch_wmma_tile_impl(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bool row_tiles_first, bnb_stream_t stream
) {
    constexpr int MT = WARPS_M * wmma_rows(ARCH) * WM;
    constexpr int NT = WARPS_N * 16 * WN;
    constexpr int THREADS = KSPLIT * WARPS_M * WARPS_N * 32;
    static_assert(
        wmma_smem_bytes(MT, NT, KSPLIT, KCH_MAX, std::is_same_v<T, float> ? WMMA_F32_PARTS : 1) <= 48 * 1024,
        "tile exceeds the LDS budget"
    );
    const dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT);
    auto launch = [&](auto kernel) {
        kernel<<<grid, THREADS, 0, stream>>>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type,
            row_tiles_first
        );
    };
    // WMMAs per absmax group: gfx11 2 per 32-K slice of a lane,
    // gfx12 4 per 64-K unit (2 when blocksize is 32).
    if constexpr (KCH_MAX == 2) {
        if (K % 128 == 0) {
            if (blocksize >= 64)
                launch(gemm_4bit_wmma<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 2, 4, PRESCALE>);
            else
                launch(gemm_4bit_wmma<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 2, 2, PRESCALE>);
            return;
        }
    }
    if constexpr (ARCH == 12) {
        if (blocksize >= 64) {
            launch(gemm_4bit_wmma<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 1, 4, PRESCALE>);
            return;
        }
    }
    launch(gemm_4bit_wmma<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 1, 2, PRESCALE>);
}

template <int ARCH, typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH_MAX>
void launch_wmma_tile(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bool row_tiles_first, bnb_stream_t stream
) {
    // Prescaling shares B multiplies across row tiles. Short-K-chunk tiles and large-M
    // contraction layers keep the accumulator-scaled kernel, which every tile has.
    if constexpr (ARCH == 11 && std::is_same_v<T, half>) {
        bool prescale = M >= 96 && WM >= 2 && KCH_MAX == 2 && (M <= 128 || K <= N);
        // Nested reconstruction can exceed 65504 even for finite fp16 source weights.
        // Rounding that scale to half would contaminate otherwise finite products.
        prescale = prescale && absmax_8bit == nullptr;
        if (prescale) {
            launch_wmma_tile_impl<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX, true>(
                A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type,
                row_tiles_first, stream
            );
            return;
        }
    }
    launch_wmma_tile_impl<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX, false>(
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, row_tiles_first,
        stream
    );
}

/// @brief Tile selection for one WMMA generation (parameters as launch_gemm_4bit_wmma).
///
/// gfx11: calibrated on gfx1100 (W7900) at pinned clocks (perf level high) over 40 LLM
/// linear-layer shapes with row-tiles-first block order and fp16 prescaling: up to three
/// leaves on N or K/N per M bucket, fp16 total
/// time within 0.3-4.4% of the best swept tile per shape (fp32 within 1.5%). Sweeps at default (auto) clocks
/// are bimodal by ~10% and picked wrong small-M tiles. Narrow N keeps 16-column warp tiles and deeper
/// split-K so enough warps stream B; wide N uses 32-column warp tiles and 128-K
/// iterations.
template <int TARGET, typename T>
void launch_gemm_4bit_wmma_arch(
    // clang-format off
    const T*       A,             // inputs [M, K]
    const uint8_t* B,             // packed 4-bit weights [N, K/2]
    const float*   absmax,        // fp32 absmax [N*K/blocksize] or [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* absmax_8bit,   // [N*K/blocksize] uint8 compressed absmax; nullptr = non-nested
    const float*   absmax_code,   // [256] codebook for 8bit absmax
    const float*   absmax_offset, // scalar; nullptr = non-nested
    T*             C,             // output [M, N]
    const T*       bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,          // problem shape
    int blocksize,                // elements per quantization block
    int quant_type,               // 1 = FP4, 2 = NF4
    bnb_stream_t stream           // HIP stream
    // clang-format on
) {
    constexpr int ARCH = wmma_gen(TARGET);
    const bool row_tiles_first = wmma_row_tiles_first(TARGET, std::is_same_v<T, float>);
    // clang-format off
#define LAUNCH_WMMA(WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX)                                           \
    launch_wmma_tile<ARCH, T, WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX>(                                \
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type,  \
        row_tiles_first, stream)

    if constexpr (ARCH == 12) {
        // gfx12 (16-row WMMA): calibrated on gfx1201 (R9700) at pinned clocks over the same
        // 40 shapes, within ~2-6% (bf16/fp16) and ~0-2% (fp32) geomean of the best swept tile.
        // One 16-row tile covers M <= 16, where the kernel is bandwidth-bound; larger M adds
        // warps along N and rows per warp rather than warps along M.
        if constexpr (std::is_same_v<T, float>) {
            // Refitted at pinned clocks on the one-fragment fp32 kernel: total time within
            // 0.0-1.4% of the best swept tile per shape.
            if (M <= 16) {
                if (N < 3840)      LAUNCH_WMMA(1, 1, 1, 2, 2, 1);
                else if (N < 4608) LAUNCH_WMMA(1, 1, 1, 4, 2, 1);
                else               LAUNCH_WMMA(1, 2, 1, 2, 2, 1);
            } else if (M <= 48) {
                if (N < 8192) LAUNCH_WMMA(1, 1, 1, 4, 2, 1);
                else          LAUNCH_WMMA(2, 2, 1, 4, 1, 1);
            } else if (M <= 96) {
                if (N < 3840) LAUNCH_WMMA(2, 1, 1, 4, 1, 1);
                else          LAUNCH_WMMA(2, 2, 1, 4, 1, 1);
            } else if (M <= 192) {
                if (N < 2048) LAUNCH_WMMA(2, 1, 1, 4, 1, 1);
                else          LAUNCH_WMMA(2, 2, 1, 4, 1, 1);
            } else {
                LAUNCH_WMMA(2, 2, 1, 4, 1, 1);
            }
        } else if (M <= 16) {
            // fp16/bf16 refitted at pinned clocks (perf level high) on R9700: total time within
            // 0.1-3.6% of the best swept tile per shape (the auto-clock rules were 9% off at
            // M <= 16 and 12.5% at M = 256).
            if (N < 3840) LAUNCH_WMMA(1, 1, 1, 2, 8, 1);
            else          LAUNCH_WMMA(1, 2, 1, 2, 4, 2);
        } else if (M <= 32) {
            if (N < 2048)      LAUNCH_WMMA(1, 1, 1, 2, 8, 1);
            else if (N < 8192) LAUNCH_WMMA(2, 1, 1, 4, 4, 1);
            else               LAUNCH_WMMA(2, 2, 1, 4, 2, 2);
        } else if (M <= 96) {
            if (N < 3840) LAUNCH_WMMA(2, 1, 1, 4, 4, 1);
            else          LAUNCH_WMMA(2, 2, 1, 4, 2, 2);
        } else if (M <= 128) {
            if (N < 3072)      LAUNCH_WMMA(2, 1, 1, 4, 4, 1);
            else if (N < 8192) LAUNCH_WMMA(2, 2, 1, 4, 2, 2);
            else               LAUNCH_WMMA(4, 2, 1, 4, 1, 1);
        } else if (M <= 160) {
            if (N < 3072) LAUNCH_WMMA(2, 1, 1, 4, 2, 1);
            else          LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
        } else {
            if (N < 2816)      LAUNCH_WMMA(2, 2, 1, 4, 2, 2);
            else if (N < 3840) LAUNCH_WMMA(4, 1, 1, 4, 1, 1);
            else               LAUNCH_WMMA(4, 2, 1, 4, 1, 1);
        }
        return;
    } else
    if constexpr (TARGET == WMMA_TARGET_GFX115X) {
        // gfx115x (gfx11 kernels): calibrated on gfx1151 (Radeon 8060S, LPDDR5X) at default
        // clocks over the 40 shapes with row-tiles-first; total time within 1-6% (fp16) and
        // 0.4-2% (fp32) of the best swept tile per shape (the gfx1100 rules were 15-17% off at
        // M <= 16 and 33% at M = 128). The 8-row half-wave tiles and 40 CUs favour fewer warps
        // along N at small M than on gfx1100.
        if constexpr (std::is_same_v<T, float>) {
            if (M <= 8) {
                if (N < 2048)      LAUNCH_WMMA(1, 1, 1, 2, 4, 1);
                else if (N < 3072) LAUNCH_WMMA(1, 1, 1, 4, 4, 1);
                else               LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
            } else if (M <= 16) {
                // M > 8 refitted from a pinned-clock sweep at M = 12, 24, 48 and 96: the previous
                // buckets (auto-clock sweep, no data at M = 24/48) were 1.13-1.49x off the best tile.
                if (K < 4608)      LAUNCH_WMMA(2, 2, 1, 2, 2, 1);
                else if (K < 8192) LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
                else               LAUNCH_WMMA(2, 1, 1, 4, 1, 2);
            } else if (M <= 32) {
                if (K < 3840)       LAUNCH_WMMA(1, 1, 1, 4, 2, 1);
                else if (K < 17408) LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
                else                LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
            } else if (M <= 64) {
                if (N < 3840)       LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
                else if (N < 10240) LAUNCH_WMMA(2, 2, 1, 2, 2, 1);
                else                LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
            } else {
                if (N < 2048)      LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
                else if (N < 3072) LAUNCH_WMMA(2, 2, 1, 2, 1, 1);
                else               LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
            }
        } else if (M <= 8) {
            if (N < 3072)       LAUNCH_WMMA(1, 1, 1, 1, 4, 1);
            else if (N < 28672) LAUNCH_WMMA(1, 1, 1, 1, 8, 2);
            else                LAUNCH_WMMA(1, 1, 1, 1, 2, 2);
        } else if (M <= 16) {
            if (N < 3072)       LAUNCH_WMMA(2, 1, 1, 2, 4, 1);
            else if (N < 25600) LAUNCH_WMMA(2, 2, 1, 1, 4, 2);
            else                LAUNCH_WMMA(2, 1, 1, 2, 2, 2);
        } else if (M <= 24) {
            // M > 16 refitted from a pinned-clock sweep at M = 24, 48, 96, 192 and 384 (the
            // auto-clock buckets were 1.10-1.12x off the best tile at M = 24-96). The bucket tops
            // M = 25-32 and 49-64 keep the gfx1100 tiles, which measured 2-6% faster there.
            if (K < 8192)       LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
            else if (K < 10240) LAUNCH_WMMA(4, 1, 1, 2, 1, 2);
            else                LAUNCH_WMMA(1, 2, 1, 1, 8, 2);
        } else if (M <= 32) {
            if (N < 3072)      LAUNCH_WMMA(2, 1, 1, 2, 8, 1);
            else if (N < 5120) LAUNCH_WMMA(2, 2, 1, 2, 4, 2);
            else               LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        } else if (M <= 48) {
            if (N < 2816)      LAUNCH_WMMA(2, 1, 1, 4, 2, 2);
            else if (N < 3840) LAUNCH_WMMA(2, 1, 1, 4, 1, 2);
            else               LAUNCH_WMMA(2, 1, 1, 4, 4, 2);
        } else if (M <= 64) {
            if (N < 3072)      LAUNCH_WMMA(4, 1, 1, 2, 4, 1);
            else if (N < 5120) LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
            else               LAUNCH_WMMA(4, 2, 1, 4, 2, 2);
        } else if (M <= 128) {
            if (K < 8192) LAUNCH_WMMA(4, 2, 1, 4, 2, 2);
            else          LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        } else if (M <= 256) {
            if (5 * (int64_t)K < 6 * (int64_t)N) LAUNCH_WMMA(4, 2, 1, 4, 2, 2);
            else                                  LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        } else {
            if (K < 21504) LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
            else           LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        }
        return;
    } else
    if constexpr (std::is_same_v<T, float>) {
        // fp32 (three LDS A tiles per stage), fitted separately: total time over the 40 shapes
        // within 0.1-2.8% of the best swept tile per shape.
        if (M <= 8) {
            if (N < 3840)      LAUNCH_WMMA(1, 1, 1, 1, 4, 1);
            else if (N < 5120) LAUNCH_WMMA(1, 1, 1, 2, 2, 2);
            else               LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
        } else if (M <= 16) {
            if (N < 3840)       LAUNCH_WMMA(1, 1, 1, 2, 4, 1);
            else if (N < 5120)  LAUNCH_WMMA(2, 1, 1, 2, 2, 1);
            else if (N < 15360) LAUNCH_WMMA(2, 1, 1, 4, 1, 2);
            else                LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
        } else if (M <= 24) {
            if (N < 3072) LAUNCH_WMMA(1, 1, 1, 2, 4, 1);
            else          LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
        } else if (M <= 48) {
            if (N < 2560)      LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
            else if (N < 5376) LAUNCH_WMMA(2, 2, 1, 2, 2, 1);
            else               LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
        } else {
            LAUNCH_WMMA(4, 1, 1, 4, 1, 1);
        }
        return;
    } else
    if (M <= 8) {
        // Leaves on the aspect ratio K/N: very wide layers (K/N < 2/7) use two warps along N,
        // very tall ones (K >= 5N) a shorter split-K with 128-K iterations.
        if (7 * (int64_t)K < 2 * (int64_t)N) LAUNCH_WMMA(1, 1, 1, 2, 8, 1);
        else if (K >= 5 * N)                  LAUNCH_WMMA(1, 1, 1, 2, 4, 2);
        else                                  LAUNCH_WMMA(1, 1, 1, 1, 8, 1);
    } else if (M <= 16) {
        if (N < 5120) LAUNCH_WMMA(2, 1, 1, 2, 8, 1);
        else          LAUNCH_WMMA(2, 2, 1, 2, 4, 2);
    } else if (M <= 24) {
        if (N < 3072)      LAUNCH_WMMA(1, 2, 1, 1, 8, 1);
        else if (N < 5120) LAUNCH_WMMA(2, 2, 1, 2, 4, 2);
        else               LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
    } else if (M <= 32) {
        if (N < 3072)      LAUNCH_WMMA(2, 1, 1, 2, 8, 1);
        else if (N < 5120) LAUNCH_WMMA(2, 2, 1, 2, 4, 2);
        else               LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
    } else if (M <= 48) {
        if (N < 3840)      LAUNCH_WMMA(2, 1, 1, 2, 4, 1);
        else if (N < 8192) LAUNCH_WMMA(2, 2, 1, 2, 2, 2);
        else               LAUNCH_WMMA(2, 2, 1, 4, 1, 2);
    } else if (M <= 64) {
        if (N < 3072)       LAUNCH_WMMA(4, 1, 1, 2, 4, 1);
        else if (N < 10240) LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        else                LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
    } else if (M <= 96) {
        if (N < 3072)      LAUNCH_WMMA(4, 1, 1, 2, 4, 1);
        else if (N < 8192) LAUNCH_WMMA(4, 2, 1, 4, 2, 2);
        else               LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
    } else {
        if (N < 2816)      LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
        else if (N < 3072) LAUNCH_WMMA(4, 2, 1, 4, 2, 1);
        else               LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
    }
#undef LAUNCH_WMMA
    // clang-format on
}

} // namespace

int gemm_4bit_wmma_arch() { return wmma_device_arch(); }

/// @brief Host launcher for the RDNA3/RDNA4 WMMA 4-bit GEMM kernel.
/// @return false (nothing launched) if `arch` is not a WMMA generation this build has kernels for.
/// @tparam T Input/output dtype (`bnb_bfloat16`, `half` or `float`)
template <typename T>
bool launch_gemm_4bit_wmma(
    // clang-format off
    int            arch,          // gemm_4bit_wmma_arch() of the current device
    const T*       A,             // inputs [M, K]
    const uint8_t* B,             // packed 4-bit weights [N, K/2]
    const float*   absmax,        // fp32 absmax [N*K/blocksize] or [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* absmax_8bit,   // [N*K/blocksize] uint8 compressed absmax; nullptr = non-nested
    const float*   absmax_code,   // [256] codebook for 8bit absmax
    const float*   absmax_offset, // scalar; nullptr = non-nested
    T*             C,             // output [M, N]
    const T*       bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,          // problem shape
    int blocksize,                // elements per quantization block
    int quant_type,               // 1 = FP4, 2 = NF4
    bnb_stream_t stream           // HIP stream
    // clang-format on
) {
#if BNB_WMMA_EMIT_12
    if (arch == 12) {
        launch_gemm_4bit_wmma_arch<12, T>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, stream
        );
        return true;
    }
#endif
#if BNB_WMMA_EMIT_11
    if (arch == 11) {
        launch_gemm_4bit_wmma_arch<11, T>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, stream
        );
        return true;
    }
    if (arch == WMMA_TARGET_GFX1103) {
        launch_gemm_4bit_wmma_arch<WMMA_TARGET_GFX1103, T>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, stream
        );
        return true;
    }
    if (arch == WMMA_TARGET_GFX115X) {
        launch_gemm_4bit_wmma_arch<WMMA_TARGET_GFX115X, T>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, stream
        );
        return true;
    }
#endif
    return false;
}

template bool launch_gemm_4bit_wmma<bnb_bfloat16>(
    int, const bnb_bfloat16*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, bnb_bfloat16*,
    const bnb_bfloat16*, int, int, int, int, int, bnb_stream_t
);
template bool launch_gemm_4bit_wmma<half>(
    int, const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int,
    int, int, int, int, bnb_stream_t
);
template bool launch_gemm_4bit_wmma<float>(
    int, const float*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, float*, const float*,
    int, int, int, int, int, bnb_stream_t
);
