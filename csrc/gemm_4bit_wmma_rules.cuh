#pragma once
// AMD WMMA 4-bit GEMM tuning rules per gfx target: SIMT or WMMA, and the WMMA tile. A target without its own
// specialization below uses its family main's (wmma_family_main).

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

/// The supported chips (the device pass of each emits the kernel body, see BNB_WMMA_DEVICE_ARCH).
inline constexpr int WMMA_TARGETS[] = {1100, 1101, 1102, 1103, 1150, 1151, 1152, 1153, 1200, 1201};

/// WMMA generation (kernel family): gfx120x runs the gfx12 kernels, gfx110x and gfx115x the gfx11 kernels.
constexpr int wmma_gen(int target) { return target >= 1200 ? 12 : 11; }

/// The main target of a chip's family, whose tunings it uses where it has none of its own.
constexpr int wmma_family_main(int target) {
    if (target >= 1200)
        return 1201;
    if (target >= 1150)
        return 1151;
    return 1100;
}

// ---------------------------------------------------------------------------------------------------------------
// SIMT or WMMA (csrc/gemm_4bit.cu), once Python has chosen the fused kernel over dequant+F.linear.

/// fp32 cutoffs: the first row with M <= max_m decides, WMMA iff N >= min_n; past the last row WMMA always.
/// SIMT keeps narrow N at small M, where the split-bf16 WMMA tiles leave too few blocks.
struct WmmaFp32Cutoff {
    int max_m, min_n;
};

// clang-format off
/// gfx1100: SIMT through M = 4, then WMMA by N; N < 1024 stays on SIMT through M = 64.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx1100[] = {{4, INT_MAX}, {5, 3072}, {7, 2816}, {8, 2048}, {12, 2816}, {64, 1024}};
/// gfx1103: SIMT through M = 2, WMMA from M = 3 (N >= 3840 through M = 4).
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx1103[] = {{2, INT_MAX}, {4, 3840}};
/// gfx1151: SIMT through M = 3, then WMMA by N through M = 7.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx1151[] = {{3, INT_MAX}, {5, 2560}, {7, 1024}};
/// gfx1201: SIMT through M = 4, then WMMA by N through M = 31.
inline constexpr WmmaFp32Cutoff wmma_fp32_cutoffs_gfx1201[] = {{4, INT_MAX}, {5, 8192}, {6, 4096}, {9, 2048}, {15, 1408}, {31, 1024}};
// clang-format on

/// The fp32 cutoffs of a target: its own where it has them, else its family main's.
template <int TARGET> struct WmmaFp32Cutoffs : WmmaFp32Cutoffs<wmma_family_main(TARGET)> {};

template <> struct WmmaFp32Cutoffs<1100> {
    static constexpr const auto& table = wmma_fp32_cutoffs_gfx1100;
};

template <> struct WmmaFp32Cutoffs<1103> {
    static constexpr const auto& table = wmma_fp32_cutoffs_gfx1103;
};

template <> struct WmmaFp32Cutoffs<1151> {
    static constexpr const auto& table = wmma_fp32_cutoffs_gfx1151;
};

template <> struct WmmaFp32Cutoffs<1201> {
    static constexpr const auto& table = wmma_fp32_cutoffs_gfx1201;
};

template <int R> constexpr bool wmma_fp32_pick(const WmmaFp32Cutoff (&cutoffs)[R], int M, int N) {
    for (const WmmaFp32Cutoff& c : cutoffs)
        if (M <= c.max_m)
            return N >= c.min_n;
    return true;
}

/// Whether the WMMA kernel (rather than SIMT) runs on a WMMA target (target != 0). bf16/fp16: WMMA at every M
/// on the gfx11 kernels, from M = 2 on gfx12 (a gfx12 WMMA always computes 16 rows); fp32: the target's cutoffs.
template <int TARGET> constexpr bool wmma_preferred_for(bool fp32, int M, int N) {
    if (!fp32)
        return wmma_gen(TARGET) != 12 || M >= 2;
    return wmma_fp32_pick(WmmaFp32Cutoffs<TARGET>::table, M, N);
}

template <size_t... I>
constexpr bool wmma_preferred_of(int target, bool fp32, int M, int N, std::index_sequence<I...>) {
    bool r = false;
    (void)((target == WMMA_TARGETS[I] && (r = wmma_preferred_for<WMMA_TARGETS[I]>(fp32, M, N), true)) || ...);
    return r;
}

constexpr bool wmma_preferred(int target, bool fp32, int M, int N) {
    return wmma_preferred_of(target, fp32, M, N, std::make_index_sequence<std::size(WMMA_TARGETS)>{});
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

/// gfx1100 bf16/fp16. Narrow N keeps 16-column warp tiles and deeper split-K so enough warps stream B; wide N
/// uses 32-column warp tiles and 128-K iterations. At M <= 8, very wide layers (K/N < 2/7) use two warps along
/// N and very tall ones (K >= 5N) a shorter split-K with 128-K iterations.
inline constexpr WmmaRule wmma_rules_gfx1100_f16[] = {
    {8,       {k_over_n_lt(2, 7, {1, 1, 1, 2, 8, 1}), k_over_n_lt(5, 1, {1, 1, 1, 1, 8, 1})},       {1, 1, 1, 2, 4, 2}},
    {16,      {n_lt(5120, {2, 1, 1, 2, 8, 1})},                                                     {2, 2, 1, 2, 4, 2}},
    {24,      {n_lt(3072, {1, 2, 1, 1, 8, 1}), n_lt(5120, {2, 2, 1, 2, 4, 2})},                     {4, 1, 1, 4, 2, 2}},
    {32,      {n_lt(3072, {2, 1, 1, 2, 8, 1}), n_lt(5120, {2, 2, 1, 2, 4, 2})},                     {4, 1, 1, 4, 2, 2}},
    {48,      {n_lt(3840, {2, 1, 1, 2, 4, 1}), n_lt(8192, {2, 2, 1, 2, 2, 2})},                     {2, 2, 1, 4, 1, 2}},
    {64,      {n_lt(3072, {4, 1, 1, 2, 4, 1}), n_lt(10240, {4, 1, 1, 4, 2, 2})},                    {4, 2, 1, 4, 1, 2}},
    {96,      {n_lt(3072, {4, 1, 1, 2, 4, 1}), n_lt(8192, {4, 2, 1, 4, 2, 2})},                     {4, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2816, {4, 1, 1, 4, 2, 2}), n_lt(3072, {4, 2, 1, 4, 2, 1})},                     {4, 2, 1, 4, 1, 2}},
};

/// gfx1100 fp32 (three LDS A tiles per stage).
inline constexpr WmmaRule wmma_rules_gfx1100_f32[] = {
    {8,       {n_lt(3840, {1, 1, 1, 1, 4, 1}), n_lt(5120, {1, 1, 1, 2, 2, 2})},                           {1, 2, 1, 2, 2, 2}},
    {16,      {n_lt(3840, {1, 1, 1, 2, 4, 1}), n_lt(5120, {2, 1, 1, 2, 2, 1}), n_lt(15360, {2, 1, 1, 4, 1, 2})}, {2, 2, 1, 4, 1, 2}},
    {24,      {n_lt(3072, {1, 1, 1, 2, 4, 1})},                                                           {1, 2, 1, 2, 2, 2}},
    {48,      {n_lt(2560, {1, 2, 1, 2, 2, 2}), n_lt(5376, {2, 2, 1, 2, 2, 1})},                           {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {},                                                                                          {4, 1, 1, 4, 1, 1}},
};

/// gfx1151 bf16/fp16: fewer warps along N at small M than gfx1100.
inline constexpr WmmaRule wmma_rules_gfx1151_f16[] = {
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

/// gfx1151 fp32.
inline constexpr WmmaRule wmma_rules_gfx1151_f32[] = {
    {8,       {n_lt(2048, {1, 1, 1, 2, 4, 1}), n_lt(3072, {1, 1, 1, 4, 4, 1})},        {1, 2, 1, 2, 2, 2}},
    {16,      {k_lt(4608, {2, 2, 1, 2, 2, 1}), k_lt(8192, {2, 2, 1, 4, 1, 2})},        {2, 1, 1, 4, 1, 2}},
    {32,      {k_lt(3840, {1, 1, 1, 4, 2, 1}), k_lt(17408, {2, 2, 1, 4, 1, 2})},       {1, 2, 1, 2, 2, 2}},
    {64,      {n_lt(3840, {2, 2, 1, 4, 1, 2}), n_lt(10240, {2, 2, 1, 2, 2, 1})},       {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2048, {2, 2, 1, 4, 1, 2}), n_lt(3072, {2, 2, 1, 2, 1, 1})},        {2, 2, 1, 4, 1, 2}},
};

/// gfx1200 bf16/fp16: own tiles at M <= 32 and M = 97-128, the gfx1201 tiles elsewhere. At M <= 16 the
/// split-K-8 tile covers N < 10240 (gfx1201: N < 3840).
inline constexpr WmmaRule wmma_rules_gfx1200_f16[] = {
    {16,      {n_lt(10240, {1, 1, 1, 2, 8, 1})},                                       {1, 1, 1, 4, 4, 2}},
    {32,      {n_lt(3840, {2, 1, 1, 4, 4, 1}), n_lt(8192, {2, 2, 1, 4, 2, 2})},        {2, 1, 1, 4, 2, 2}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 4, 1})},                                        {2, 2, 1, 4, 2, 2}},
    {128,     {n_lt(2816, {2, 2, 1, 4, 2, 2}), n_lt(8192, {4, 2, 1, 4, 1, 1})},        {2, 2, 1, 4, 1, 2}},
    {160,     {n_lt(3072, {2, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2816, {2, 2, 1, 4, 2, 2}), n_lt(3840, {4, 1, 1, 4, 1, 1})},        {4, 2, 1, 4, 1, 1}},
};

/// gfx1200 fp32: the gfx1201 table plus its own M = 17-32 bucket.
inline constexpr WmmaRule wmma_rules_gfx1200_f32[] = {
    {16,      {n_lt(3840, {1, 1, 1, 2, 2, 1}), n_lt(4608, {1, 1, 1, 4, 2, 1})},        {1, 2, 1, 2, 2, 1}},
    {32,      {n_lt(3840, {1, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {48,      {n_lt(8192, {1, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {192,     {n_lt(2048, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {INT_MAX, {},                                                                       {2, 2, 1, 4, 1, 1}},
};

/// gfx1201 bf16/fp16. One 16-row tile covers M <= 16; larger M adds warps along N and rows per warp rather than
/// warps along M.
inline constexpr WmmaRule wmma_rules_gfx1201_f16[] = {
    {16,      {n_lt(3840, {1, 1, 1, 2, 8, 1})},                                        {1, 2, 1, 2, 4, 2}},
    {32,      {n_lt(2048, {1, 1, 1, 2, 8, 1}), n_lt(8192, {2, 1, 1, 4, 4, 1})},        {2, 2, 1, 4, 2, 2}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 4, 1})},                                        {2, 2, 1, 4, 2, 2}},
    {128,     {n_lt(3072, {2, 1, 1, 4, 4, 1}), n_lt(8192, {2, 2, 1, 4, 2, 2})},        {4, 2, 1, 4, 1, 1}},
    {160,     {n_lt(3072, {2, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 2}},
    {INT_MAX, {n_lt(2816, {2, 2, 1, 4, 2, 2}), n_lt(3840, {4, 1, 1, 4, 1, 1})},        {4, 2, 1, 4, 1, 1}},
};

/// gfx1201 fp32 (one-fragment split-bf16 kernel).
inline constexpr WmmaRule wmma_rules_gfx1201_f32[] = {
    {16,      {n_lt(3840, {1, 1, 1, 2, 2, 1}), n_lt(4608, {1, 1, 1, 4, 2, 1})},        {1, 2, 1, 2, 2, 1}},
    {48,      {n_lt(8192, {1, 1, 1, 4, 2, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {96,      {n_lt(3840, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {192,     {n_lt(2048, {2, 1, 1, 4, 1, 1})},                                        {2, 2, 1, 4, 1, 1}},
    {INT_MAX, {},                                                                       {2, 2, 1, 4, 1, 1}},
};

// clang-format on

/// The tile rule tables of a target: its own where it has them, else its family main's (gfx1103 uses gfx1100's).
/// `owner` is the target whose tables they are, so targets sharing tables share one launcher instantiation.
template <int TARGET> struct WmmaTileRules : WmmaTileRules<wmma_family_main(TARGET)> {};

template <> struct WmmaTileRules<1100> {
    static constexpr int owner = 1100;
    static constexpr const auto& f16 = wmma_rules_gfx1100_f16;
    static constexpr const auto& f32 = wmma_rules_gfx1100_f32;
};

template <> struct WmmaTileRules<1151> {
    static constexpr int owner = 1151;
    static constexpr const auto& f16 = wmma_rules_gfx1151_f16;
    static constexpr const auto& f32 = wmma_rules_gfx1151_f32;
};

template <> struct WmmaTileRules<1200> {
    static constexpr int owner = 1200;
    static constexpr const auto& f16 = wmma_rules_gfx1200_f16;
    static constexpr const auto& f32 = wmma_rules_gfx1200_f32;
};

template <> struct WmmaTileRules<1201> {
    static constexpr int owner = 1201;
    static constexpr const auto& f16 = wmma_rules_gfx1201_f16;
    static constexpr const auto& f32 = wmma_rules_gfx1201_f32;
};
