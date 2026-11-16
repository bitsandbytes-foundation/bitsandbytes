#pragma once
// AMD WMMA 4-bit GEMM tuning rules per target: SIMT or WMMA, and the WMMA tile.

#include <climits>
#include <cstdint>

// Tuning targets returned by gemm_4bit_wmma_arch(). gfx1103 and gfx115x run the gfx11 kernels; gfx1103 uses the
// gfx110x tile rules with its own fp32 SIMT/WMMA cutoffs, gfx115x has its own rules.
inline constexpr int WMMA_TARGET_GFX11 = 11;    // gfx1100-gfx1102
inline constexpr int WMMA_TARGET_GFX1103 = 113; // gfx1103 (Radeon 760M/780M iGPU)
inline constexpr int WMMA_TARGET_GFX115X = 115; // gfx1150-gfx1153
inline constexpr int WMMA_TARGET_GFX12 = 12;    // gfx1200-gfx1201

// ---------------------------------------------------------------------------------------------------------------
// SIMT or WMMA (csrc/gemm_4bit.cu), once Python has chosen the fused kernel over dequant+F.linear.

/// fp32 cutoffs: the first row with M <= max_m decides, WMMA iff N >= min_n; past the last row WMMA always.
/// SIMT keeps narrow N at small M, where the split-bf16 WMMA tiles leave too few blocks.
struct WmmaFp32Cutoff {
    int max_m, min_n;
};

// clang-format off
/// gfx11: SIMT through M = 4, then WMMA by N; N < 1024 stays on SIMT through M = 64.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx11[] = {{4, INT_MAX}, {5, 3072}, {7, 2816}, {8, 2048}, {12, 2816}, {64, 1024}};
/// gfx1103: SIMT through M = 2, WMMA from M = 3 (N >= 3840 through M = 4).
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx1103[] = {{2, INT_MAX}, {4, 3840}};
/// gfx115x: SIMT through M = 3, then WMMA by N through M = 7.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx115[] = {{3, INT_MAX}, {5, 2560}, {7, 1024}};
/// gfx12: SIMT through M = 4, then WMMA by N through M = 31.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx12[] = {{4, INT_MAX}, {5, 8192}, {6, 4096}, {9, 2048}, {15, 1408}, {31, 1024}};
// clang-format on

template <int R> constexpr bool wmma_fp32_pick(const WmmaFp32Cutoff (&cutoffs)[R], int M, int N) {
    for (const WmmaFp32Cutoff& c : cutoffs)
        if (M <= c.max_m)
            return N >= c.min_n;
    return true;
}

/// Whether the WMMA kernel (rather than SIMT) runs on a WMMA target (target != 0). bf16/fp16: WMMA at every M
/// on the gfx11 kernels, from M = 2 on gfx12 (a gfx12 WMMA always computes 16 rows); fp32: the target's cutoffs.
constexpr bool wmma_preferred(int target, bool fp32, int M, int N) {
    if (!fp32)
        return target != WMMA_TARGET_GFX12 || M >= 2;
    switch (target) {
    case WMMA_TARGET_GFX12:
        return wmma_fp32_pick(wmma_fp32_cutoffs_gfx12, M, N);
    case WMMA_TARGET_GFX115X:
        return wmma_fp32_pick(wmma_fp32_cutoffs_gfx115, M, N);
    case WMMA_TARGET_GFX1103:
        return wmma_fp32_pick(wmma_fp32_cutoffs_gfx1103, M, N);
    default:
        return wmma_fp32_pick(wmma_fp32_cutoffs_gfx11, M, N);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// WMMA tile per problem (csrc/gemm_4bit_wmma.cu).

/// One tile configuration: the template parameters of the gemm_4bit_wmma kernel.
struct WmmaTile {
    int wm, wn;           // WMMA tiles per warp along M and N (8x16 on gfx11, 16x16 on gfx12)
    int warps_m, warps_n; // warps per block along M and N
    int ksplit;           // warp groups splitting K, reduced through LDS
    int kch;              // 64-K units per main-loop iteration (lowered to 1 when K % 128 != 0)
};

constexpr bool operator==(const WmmaTile& a, const WmmaTile& b) {
    return a.wm == b.wm && a.wn == b.wn && a.warps_m == b.warps_m && a.warps_n == b.warps_n && a.ksplit == b.ksplit &&
           a.kch == b.kch;
}

/// Leaf condition: N < lt, K < lt, or K / N < lt / den.
enum class WmmaOn : uint8_t { NONE, N, K, K_OVER_N };

struct WmmaLeaf {
    WmmaOn on;
    int lt;
    int den;
    WmmaTile tile;
};

constexpr WmmaLeaf n_lt(int n, WmmaTile tile) { return {WmmaOn::N, n, 1, tile}; }

constexpr WmmaLeaf k_lt(int k, WmmaTile tile) { return {WmmaOn::K, k, 1, tile}; }

constexpr WmmaLeaf k_over_n_lt(int num, int den, WmmaTile tile) { return {WmmaOn::K_OVER_N, num, den, tile}; }

/// One M bucket, used when M <= max_m: the first matching leaf picks the tile, else `otherwise`. A table's
/// last bucket has max_m = INT_MAX.
struct WmmaRule {
    int max_m;
    WmmaLeaf leaves[3];
    WmmaTile otherwise;
};

constexpr bool wmma_leaf_matches(const WmmaLeaf& leaf, int N, int K) {
    switch (leaf.on) {
    case WmmaOn::N:
        return N < leaf.lt;
    case WmmaOn::K:
        return K < leaf.lt;
    case WmmaOn::K_OVER_N:
        return (int64_t)K * leaf.den < (int64_t)leaf.lt * N;
    default:
        return false;
    }
}

/// The tile a rule table picks for (M, N, K).
template <int R> constexpr WmmaTile wmma_pick(const WmmaRule (&rules)[R], int M, int N, int K) {
    for (const WmmaRule& rule : rules) {
        if (M > rule.max_m)
            continue;
        for (const WmmaLeaf& leaf : rule.leaves)
            if (wmma_leaf_matches(leaf, N, K))
                return leaf.tile;
        return rule.otherwise;
    }
    return rules[R - 1].otherwise;
}

// clang-format off

/// gfx11 bf16/fp16. Narrow N keeps 16-column warp tiles and deeper split-K so enough warps stream B; wide N
/// uses 32-column warp tiles and 128-K iterations. At M <= 8, very wide layers (K/N < 2/7) use two warps along
/// N and very tall ones (K >= 5N) a shorter split-K with 128-K iterations.
inline constexpr WmmaRule wmma_rules_gfx11_f16[] = {
    {8,       {k_over_n_lt(2, 7, {1, 1, 1, 2, 8, 1}), k_over_n_lt(5, 1, {1, 1, 1, 1, 8, 1})},       {1, 1, 1, 2, 4, 2}},
    {16,      {n_lt(5120, {2, 1, 1, 2, 8, 1})},                                                     {2, 2, 1, 2, 4, 2}},
    {24,      {n_lt(3072, {1, 2, 1, 1, 8, 1}), n_lt(5120, {2, 2, 1, 2, 4, 2})},                     {4, 1, 1, 4, 2, 2}},
    {32,      {n_lt(3072, {2, 1, 1, 2, 8, 1}), n_lt(5120, {2, 2, 1, 2, 4, 2})},                     {4, 1, 1, 4, 2, 2}},
    {48,      {n_lt(3840, {2, 1, 1, 2, 4, 1}), n_lt(8192, {2, 2, 1, 2, 2, 2})},                     {2, 2, 1, 4, 1, 2}},
    {64,      {n_lt(3072, {4, 1, 1, 2, 4, 1}), n_lt(10240, {4, 1, 1, 4, 2, 2})},                    {4, 2, 1, 4, 1, 2}},
    {96,      {n_lt(3072, {4, 1, 1, 2, 4, 1}), n_lt(8192, {4, 2, 1, 4, 2, 2})},                     {4, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2816, {4, 1, 1, 4, 2, 2}), n_lt(3072, {4, 2, 1, 4, 2, 1})},                     {4, 2, 1, 4, 1, 2}},
};

/// gfx11 fp32 (three LDS A tiles per stage).
inline constexpr WmmaRule wmma_rules_gfx11_f32[] = {
    {8,       {n_lt(3840, {1, 1, 1, 1, 4, 1}), n_lt(5120, {1, 1, 1, 2, 2, 2})},                           {1, 2, 1, 2, 2, 2}},
    {16,      {n_lt(3840, {1, 1, 1, 2, 4, 1}), n_lt(5120, {2, 1, 1, 2, 2, 1}), n_lt(15360, {2, 1, 1, 4, 1, 2})}, {2, 2, 1, 4, 1, 2}},
    {24,      {n_lt(3072, {1, 1, 1, 2, 4, 1})},                                                           {1, 2, 1, 2, 2, 2}},
    {48,      {n_lt(2560, {1, 2, 1, 2, 2, 2}), n_lt(5376, {2, 2, 1, 2, 2, 1})},                           {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {},                                                                                          {4, 1, 1, 4, 1, 1}},
};

/// gfx115x bf16/fp16: fewer warps along N at small M than gfx1100.
inline constexpr WmmaRule wmma_rules_gfx115_f16[] = {
    {8,       {n_lt(3072, {1, 1, 1, 1, 4, 1}), n_lt(28672, {1, 1, 1, 1, 8, 2})},       {1, 1, 1, 1, 2, 2}},
    {16,      {n_lt(3072, {2, 1, 1, 2, 4, 1}), n_lt(25600, {2, 2, 1, 1, 4, 2})},       {2, 1, 1, 2, 2, 2}},
    {24,      {k_lt(8192, {4, 1, 1, 4, 2, 2}), k_lt(10240, {4, 1, 1, 2, 1, 2})},       {1, 2, 1, 1, 8, 2}},
    {32,      {n_lt(3072, {2, 1, 1, 2, 8, 1}), n_lt(5120, {2, 2, 1, 2, 4, 2})},        {4, 1, 1, 4, 2, 2}},
    {48,      {n_lt(2816, {2, 1, 1, 4, 2, 2}), n_lt(3840, {2, 1, 1, 4, 1, 2})},        {2, 1, 1, 4, 4, 2}},
    {64,      {n_lt(3072, {4, 1, 1, 2, 4, 1}), n_lt(5120, {4, 1, 1, 4, 2, 2})},        {4, 2, 1, 4, 2, 2}},
    {128,     {k_lt(8192, {4, 2, 1, 4, 2, 2})},                                        {4, 1, 1, 4, 2, 2}},
    {256,     {k_over_n_lt(6, 5, {4, 2, 1, 4, 2, 2})},                                 {4, 1, 1, 4, 2, 2}},
    {INT_MAX, {k_lt(21504, {4, 2, 1, 4, 1, 2})},                                       {4, 1, 1, 4, 2, 2}},
};

/// gfx115x fp32.
inline constexpr WmmaRule wmma_rules_gfx115_f32[] = {
    {8,       {n_lt(2048, {1, 1, 1, 2, 4, 1}), n_lt(3072, {1, 1, 1, 4, 4, 1})},        {1, 2, 1, 2, 2, 2}},
    {16,      {k_lt(4608, {2, 2, 1, 2, 2, 1}), k_lt(8192, {2, 2, 1, 4, 1, 2})},        {2, 1, 1, 4, 1, 2}},
    {32,      {k_lt(3840, {1, 1, 1, 4, 2, 1}), k_lt(17408, {2, 2, 1, 4, 1, 2})},       {1, 2, 1, 2, 2, 2}},
    {64,      {n_lt(3840, {2, 2, 1, 4, 1, 2}), n_lt(10240, {2, 2, 1, 2, 2, 1})},       {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2048, {2, 2, 1, 4, 1, 2}), n_lt(3072, {2, 2, 1, 2, 1, 1})},        {2, 2, 1, 4, 1, 2}},
};

/// gfx12 bf16/fp16. One 16-row tile covers M <= 16; larger M adds warps along N and rows per warp rather than
/// warps along M.
inline constexpr WmmaRule wmma_rules_gfx12_f16[] = {
    {16,      {n_lt(3840, {1, 1, 1, 2, 8, 1})},                                        {1, 2, 1, 2, 4, 2}},
    {32,      {n_lt(2048, {1, 1, 1, 2, 8, 1}), n_lt(8192, {2, 1, 1, 4, 4, 1})},        {2, 2, 1, 4, 2, 2}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 4, 1})},                                        {2, 2, 1, 4, 2, 2}},
    {128,     {n_lt(3072, {2, 1, 1, 4, 4, 1}), n_lt(8192, {2, 2, 1, 4, 2, 2})},        {4, 2, 1, 4, 1, 1}},
    {160,     {n_lt(3072, {2, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2816, {2, 2, 1, 4, 2, 2}), n_lt(3840, {4, 1, 1, 4, 1, 1})},        {4, 2, 1, 4, 1, 1}},
};

/// gfx12 fp32 (one-fragment split-bf16 kernel).
inline constexpr WmmaRule wmma_rules_gfx12_f32[] = {
    {16,      {n_lt(3840, {1, 1, 1, 2, 2, 1}), n_lt(4608, {1, 1, 1, 4, 2, 1})},        {1, 2, 1, 2, 2, 1}},
    {48,      {n_lt(8192, {1, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {192,     {n_lt(2048, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {INT_MAX, {},                                                                       {2, 2, 1, 4, 1, 1}},
};

// clang-format on
