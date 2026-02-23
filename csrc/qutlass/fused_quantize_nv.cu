/*
 * Modified from QuTLASS (https://github.com/IST-DASLab/qutlass)
 * Original copyright (C) 2025 Roberto L. Castro. Apache License 2.0.
 *
 * bitsandbytes vendored version: torch dependencies removed,
 * only NVFP4 RotationSize=16 variants retained (AbsMax + Quest).
 */

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass_extensions/gemm/device/gemm_quant.h"

namespace bitsandbytes {

using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementGemmOutput = cutlass::bfloat16_t;
using ElementOutput = cutlass::float_e2m1_t;

using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

template <
    typename ShapeMMAThreadBlock, typename ShapeMMAWarp, typename InstructionShape, bool Quest = false,
    int RotationSize = 16>
using Gemm_ = cutlass::gemm::device::GemmQuantNv<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementGemmOutput, LayoutOutput, ElementOutput,
    LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ShapeMMAThreadBlock,
    ShapeMMAWarp, InstructionShape, Quest, RotationSize>;

template <typename Gemm> struct GemmRunner {
    bool
        run(const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int32_t M, int32_t N,
            int32_t K, cudaStream_t stream) {
        using GemmCoord = cutlass::gemm::GemmCoord;
        Gemm gemmOp;

        typename Gemm::Arguments arguments{
            {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
            {(cutlass::bfloat16_t*)A, K},
            {(cutlass::bfloat16_t*)B, N},
            {(cutlass::float_e2m1_t*)D, N},
            {(cutlass::float_e2m1_t*)D, N},
            {(cutlass::float_ue4m3_t*)D_sf, M},
            const_cast<float*>(global_scale),
            cutlass::bfloat16_t(0)
        };

        auto status = gemmOp.initialize(arguments, nullptr, stream);
        if (status != cutlass::Status::kSuccess)
            return false;

        status = gemmOp(arguments, nullptr, stream);
        return status == cutlass::Status::kSuccess;
    }
};

// RotationSize=16, Quest=false (AbsMax)
using TileShape16 = cutlass::gemm::GemmShape<128, 32, 32>;
using WarpShape16 = cutlass::gemm::GemmShape<32, 32, 32>;
using MmaShape16 = cutlass::gemm::GemmShape<16, 8, 16>;

using GemmAbsMax16 = Gemm_<TileShape16, WarpShape16, MmaShape16, false, 16>;
using GemmQuest16 = Gemm_<TileShape16, WarpShape16, MmaShape16, true, 16>;

} // namespace bitsandbytes

extern "C" {

void cfused_quantize_nvfp4_absmax(
    const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int M, int N, int K,
    cudaStream_t stream
) {
    bitsandbytes::GemmRunner<bitsandbytes::GemmAbsMax16> runner;
    runner.run(A, B, D, D_sf, global_scale, M, N, K, stream);
}

void cfused_quantize_nvfp4_quest(
    const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int M, int N, int K,
    cudaStream_t stream
) {
    bitsandbytes::GemmRunner<bitsandbytes::GemmQuest16> runner;
    runner.run(A, B, D, D_sf, global_scale, M, N, K, stream);
}

} // extern "C"
