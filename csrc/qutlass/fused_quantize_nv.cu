/*
 * Modified from QuTLASS (https://github.com/IST-DASLab/qutlass)
 * Original copyright (C) 2025 Roberto L. Castro. Apache License 2.0.
 *
 * bitsandbytes vendored version: torch dependencies removed,
 * only NVFP4 RotationSize=16 variants retained (AbsMax + Quest).
 *
 * The runner is split into init() and run() so that run() only contains
 * the kernel launch (no cudaFuncSetAttribute), making it CUDA-graph-safe.
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

// Persistent runner: init() called once (sets cudaFuncSetAttribute),
// run() called per-invocation (kernel launch only, graph-safe).
template <typename Gemm> struct PersistentRunner {
    using GemmKernel = typename Gemm::GemmKernel;
    using ThreadblockSwizzle = typename Gemm::ThreadblockSwizzle;
    using ThreadblockShape = typename Gemm::ThreadblockShape;

    Gemm gemmOp;
    int smem_size;
    bool initialized = false;

    // Call once. NOT graph-safe (calls cudaFuncSetAttribute).
    bool init() {
        smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        // Set shared memory attribute once (NOT graph-safe)
        if (smem_size >= (48 << 10)) {
            cudaError_t result = cudaFuncSetAttribute(
                cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
            );
            if (result != cudaSuccess)
                return false;
        }

        initialized = true;
        return true;
    }

    // Call per invocation. Graph-safe: only host math + kernel launch.
    bool
        run(const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int32_t M, int32_t N,
            int32_t K, cudaStream_t stream) {
        using GemmCoord = cutlass::gemm::GemmCoord;

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

        // initialize() fills params_ struct (host-side only, no CUDA API calls)
        auto status = gemmOp.initialize(arguments, nullptr, stream);
        if (status != cutlass::Status::kSuccess)
            return false;

        // Compute grid/block for this problem size (host math only)
        ThreadblockSwizzle swizzle;
        dim3 grid = swizzle.get_grid_shape(gemmOp.params_.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);

        // Kernel launch (graph-safe)
        cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(gemmOp.params_);
        return cudaGetLastError() == cudaSuccess;
    }
};

// RotationSize=16, Quest=false (AbsMax)
using TileShape16 = cutlass::gemm::GemmShape<128, 32, 32>;
using WarpShape16 = cutlass::gemm::GemmShape<32, 32, 32>;
using MmaShape16 = cutlass::gemm::GemmShape<16, 8, 16>;

using GemmAbsMax16 = Gemm_<TileShape16, WarpShape16, MmaShape16, false, 16>;
using GemmQuest16 = Gemm_<TileShape16, WarpShape16, MmaShape16, true, 16>;

// Singleton runners — initialized lazily on first call
static PersistentRunner<GemmAbsMax16> g_absmax_runner;
static PersistentRunner<GemmQuest16> g_quest_runner;

} // namespace bitsandbytes

extern "C" {

void cfused_quantize_nvfp4_absmax(
    const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int M, int N, int K,
    cudaStream_t stream
) {
    auto& runner = bitsandbytes::g_absmax_runner;
    if (!runner.initialized)
        runner.init();
    runner.run(A, B, D, D_sf, global_scale, M, N, K, stream);
}

void cfused_quantize_nvfp4_quest(
    const void* A, const void* B, void* D, void* D_sf, const float* global_scale, int M, int N, int K,
    cudaStream_t stream
) {
    auto& runner = bitsandbytes::g_quest_runner;
    if (!runner.initialized)
        runner.init();
    runner.run(A, B, D, D_sf, global_scale, M, N, K, stream);
}

} // extern "C"
