/*
 * Shared CUTLASS type definitions for SM_120 block-scaled FP4 GEMM.
 *
 * Both the dense and batched MoE kernels use these common type aliases
 * to ensure they instantiate identical GemmKernel types, which is required
 * for proper CUDA device kernel registration across translation units.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// =========================================================================
// FpGemm: CUTLASS GEMM template for block-scaled FP4 operations (SM_120)
// =========================================================================
template <
    typename MmaTileShape, typename ClusterShape, typename PerSmTileShape_MNK, typename ArchTag, typename ElementA,
    typename LayoutATag, int AlignmentA, typename ElementB, typename LayoutBTag, int AlignmentB>
struct FpGemm {
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Scale factor config and layout types (useful for both dense and MoE)
    using ElementSF    = cutlass::float_ue4m3_t;
    using StrideA      = typename GemmKernel::StrideA;
    using StrideB      = typename GemmKernel::StrideB;
    using StrideC      = typename GemmKernel::StrideC;
    using StrideD      = typename GemmKernel::StrideD;
    using LayoutSFA    = typename GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB    = typename GemmKernel::CollectiveMainloop::LayoutSFB;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// Standard type instantiation for SM_120 block-scaled FP4
using ElementA_SM120 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB_SM120 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA_SM120  = cutlass::layout::RowMajor;
using LayoutB_SM120  = cutlass::layout::ColumnMajor;
static constexpr int AlignA_SM120 = 32;
static constexpr int AlignB_SM120 = 32;
using ArchTag_SM120  = cutlass::arch::Sm120;
using ClusterShape_SM120 = Shape<_1, _1, _1>;

// Small tile (128x128x128) — for M < 512
using FpGemmSmall = FpGemm<
    Shape<_128, _128, _128>, ClusterShape_SM120, Shape<_128, _128, _128>,
    ArchTag_SM120, ElementA_SM120, LayoutA_SM120, AlignA_SM120,
    ElementB_SM120, LayoutB_SM120, AlignB_SM120>;

// Large tile (256x128x128) — for M >= 512
using FpGemmLarge = FpGemm<
    Shape<_256, _128, _128>, ClusterShape_SM120, Shape<_256, _128, _128>,
    ArchTag_SM120, ElementA_SM120, LayoutA_SM120, AlignA_SM120,
    ElementB_SM120, LayoutB_SM120, AlignB_SM120>;

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED
