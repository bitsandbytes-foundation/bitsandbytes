/*
 * Modified by Roberto L. Castro (Roberto.LopezCastro@ist.ac.at).
 */

/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

  The shared memory resource is time-sliced across warps.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include <cmath>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {
////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
static uint32_t fp32_vec_to_e2m1(float* array) {
    uint32_t val;
    asm volatile("{\n"
                 ".reg .b8 byte0;\n"
                 ".reg .b8 byte1;\n"
                 ".reg .b8 byte2;\n"
                 ".reg .b8 byte3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
                 "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                 "}"
                 : "=r"(val)
                 : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]),
                   "f"(array[6]), "f"(array[7]));
    return val;
}

CUTLASS_HOST_DEVICE
static uint8_t f32_to_e4m3_hi(float v) {
    uint16_t packed;
    // 0.0f → lower 8 bits, v → upper 8 bits
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(packed) : "f"(0.0f), "f"(v));
    return uint8_t(packed >> 8);
}

CUTLASS_HOST_DEVICE
static float e4m3_to_f32(uint8_t hi) {
    uint16_t packed = uint16_t(hi) << 8;
    uint32_t fp16x2;

    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(fp16x2) : "h"(packed));

    uint16_t fp16_hi = static_cast<uint16_t>(fp16x2 >> 16);

    float out;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(fp16_hi));
    return out;
}

// Fast reciprocal.
CUTLASS_HOST_DEVICE
static float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

/// Epilogue operator
template <
    typename Shape_,                       ///< Shape of threadblock tile (concept: GemmShape)
    typename WarpMmaOperator_,             ///< Warp-level MMA operator (concept:
                                           ///< gemm::warp::MmaTensorOp)
    int PartitionsK,                       ///< Number of partitions of the K dimension
    typename OutputTileIterator_,          ///< Tile iterator reading and writing
                                           ///< output tensors
    typename AccumulatorFragmentIterator_, ///< Fragment iterator
                                           ///< selecting accumulators
    typename WarpTileIterator_,            ///< Warp-scoped tile iterator writing
                                           ///< accumulators to SMEM
    typename SharedLoadIterator_,          ///< Threadblock-scoped tile iterator
                                           ///< loading from SMEM
    typename OutputOp_,                    ///< Output operator
    typename Padding_,                     ///< Padding added to SMEM allocation to avoid
                                           ///< bank conflicts (concept: MatrixShape)
    int FragmentsPerPartition = 1,         ///< Used to coarsten the epilogue granularity
    bool is_quartet = true, int RotationSize = 32,
    int IterationsUnroll = ///< Used to reduce binary size when epilogue
                           ///< op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class EpilogueQuantMx
    : public EpilogueBase<
          Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
          Padding_, FragmentsPerPartition>,
      public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_> {
  public:
    using Base = EpilogueBase<
        Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
        Padding_, FragmentsPerPartition>;

    using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;
    using Layout = layout::RowMajor;
    using LongIndex = typename Layout::LongIndex;

    /// Number of warps per block
    using WarpCount = typename Base::WarpCount;

    /// Number of threads per block
    static int const kBlockThreads = 32 * WarpCount::kCount;

    /// Per-thread accumulator tile type
    using AccumulatorTile = typename Base::AccumulatorTile;

    /// Numerical accumulation element type
    using ElementAccumulator = typename WarpMmaOperator::ElementC;

    /// Fragment type used by the accumulator tile's fragment iterator
    using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Vector type used by the global output iterator
    using OutputAccessType = Array<ElementOutput, OutputTileIterator::kElementsPerAccess>;

    using OutputGemmAccessType = Array<cutlass::bfloat16_t,
                                       OutputTileIterator::kElementsPerAccess>; // TODO: float
    using OutputAccessType2 = Array<cutlass::float_e2m1_t,
                                    OutputTileIterator::kElementsPerAccess>; // TODO: bfloat16_t

    /// Vector type used by the shared output iterator
    using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

    static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

    static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  public:
    static_assert(
        SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
        "Mismatch between shared load iterator and output tile iterator."
    );

    static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), "Divisibility");

    static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

  public:
    /// Aspect for when epilogue source is needed
    struct SourceAspectNeeded {
        OutputTileIterator source_iterator;

        typename OutputTileIterator::Fragment source_fragment;

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        static void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
            typename OutputTileIterator::Fragment const& source_fragment
        ) {

            OutputAccessType* output_frag_ptr = reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

            OutputGemmAccessType const* source_frag_ptr =
                reinterpret_cast<OutputGemmAccessType const*>(&source_fragment);

            int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
            }
        }

        /// Constructor
        CUTLASS_DEVICE
        SourceAspectNeeded(OutputTileIterator source_iterator) : source_iterator(source_iterator) {
            source_fragment.clear();
        }

        // Load addend source fragment from global memory
        CUTLASS_DEVICE
        void load() {
            source_iterator.load(source_fragment);
            ++source_iterator;
        }

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment
        ) {
            apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
        }
    };

  private:
    /// Loads fragment from shared memory aligned with output tensor
    SharedLoadIterator shared_load_iterator_;

    /// Thread index in the threadblock
    int thread_idx;

    /// Warp index in the threadblock
    int warp_idx;

  public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueQuantMx(
        typename Base::SharedStorage& shared_storage, ///< Shared storage object
        int thread_idx,                               ///< ID of a thread within the threadblock
        int warp_idx,                                 ///< ID of warp within threadblock
        int lane_idx
    ) ///< Id of thread within warp
        : Base(shared_storage, thread_idx, warp_idx, lane_idx), BaseStreamK(thread_idx),
          shared_load_iterator_(shared_storage.reference(), thread_idx), thread_idx(thread_idx), warp_idx(warp_idx) {}

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements two alternative codepaths, depending on whether the output op
    /// requires addend data to be loaded.
    CUTLASS_DEVICE
    void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator,      ///< Tile iterator for addend source
        cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size
    ) {
        static_assert(
            RotationSize == 32 || RotationSize == 64 || RotationSize == 128, "RotationSize must be 32/64/128"
        );
        operator()(
            output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator), D, D_sf, problem_m_size
        );
    }

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements a single codepath, regardless of whether the output op requires
    /// addend data to be loaded
    CUTLASS_DEVICE
    void unified(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator
    ) ///< Tile iterator for addend source
    {
        if (!output_op.is_source_needed()) {
            source_iterator.clear_mask();
            __syncthreads(); // Dummy (CUDA 11.0)
        }

        operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
    }

    template <class Seq> struct acc2smem;

    template <size_t... Seq> struct acc2smem<cutlass::index_sequence<Seq...>> {
        template <int Advance>
        CUTLASS_DEVICE static void
            helper(AccumulatorFragmentIterator accum_fragment_iterator, WarpTileIterator& warp_tile_iterator) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Advance; i++) {
                ++accum_fragment_iterator;
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;
            warp_tile_iterator.store(accum_fragment);
        }

        CUTLASS_DEVICE
        static void
            push(size_t pos, AccumulatorFragmentIterator const& iterator_begin, WarpTileIterator& warp_tile_iterator) {
            int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
        }
    };

    /// Streams the result to global memory
    template <typename SourceAspect>
    CUTLASS_DEVICE void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size
    ) {
        static_assert(
            RotationSize == 32 || RotationSize == 64 || RotationSize == 128, "RotationSize must be 32/64/128"
        );
        EpilogueOpImpl<RotationSize, EpilogueQuantMx>::run(
            *this, output_op, destination_iterator, accumulators, source, D, D_sf, problem_m_size
        );
    }

  private:
    template <int RS, typename Epilogue> struct EpilogueOpImpl;

    template <typename Epilogue> struct EpilogueOpImpl<32, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_32(std::forward<Args>(args)...);
        }
    };

    template <typename Epilogue> struct EpilogueOpImpl<64, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_64(std::forward<Args>(args)...);
        }
    };

    template <typename Epilogue> struct EpilogueOpImpl<128, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_128(std::forward<Args>(args)...);
        }
    };

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_32(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[32];
            uint32_t result_reg[4];

            int row = iter * (32 / 4) + ((threadIdx.x % 32) / 4) +
                      (threadIdx.x / 32) * (32 / 4) * OutputTileIterator::kIterations + blockIdx.x * blockDim.x;

            float4* result_ptr = ((float4*)D + row);      // 4=32/8
            uint8_t* x_e8m0_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if ((threadIdx.x % 4) == 0 && row < problem_m_size) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() +
                     (threadIdx.x / 4) * 10); // + iter*(blockDim.x/4)*32); 40=32+8
                                              // padding of 32 elements? check bank conflicts
                                              // 10=40/4
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 / 32;
                    float var = c_sum2 / 32 - c_mean * c_mean;
                    float scale = 1.0;
                    if (var >= 0) {
                        scale = std::sqrt(var) * (2.92247856 / 6.) + 1e-8;
                    }

                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    float abs_max = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float scale = abs_max + 1e-8f;
                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                            mat_c[w * 8 + z] *= 3;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float4*)result_ptr) = *((float4*)result_reg);
            }
        }
    }

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_64(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[32];
            uint32_t result_reg[4];

            int row = iter * (32 / 4) * 2 + ((threadIdx.x % 32) / 4) * 2 + (threadIdx.x % 32) % 2 +
                      (threadIdx.x / 32) * (32 / 4) * 2 * OutputTileIterator::kIterations + blockIdx.x * blockDim.x * 2;

            float4* result_ptr = ((float4*)D + row);      // 4=32/8
            uint8_t* x_e8m0_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if ((threadIdx.x % 4) < 2 && row < problem_m_size * 2) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() + (threadIdx.x / 4) * 18 +
                     (threadIdx.x % 2) * 8); // + iter*(blockDim.x/4)*32); 40=32+8
                                             // padding of 32 elements? check bank conflicts
                                             // 10=40/4
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 / 32;
                    float var = c_sum2 / 32 - c_mean * c_mean;
                    float scale = 1.0;
                    if (var >= 0) {
                        scale = std::sqrt(var) * (2.92247856 / 6.) + 1e-8;
                    }

                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    float abs_max = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float scale = abs_max + 1e-8f;
                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                            mat_c[w * 8 + z] *= 3;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float4*)result_ptr) = *((float4*)result_reg);
            }
        }
    }

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_128(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[32];
            uint32_t result_reg[4];

            int row = iter * (32 / 4) + ((threadIdx.x % 32) / 4) +
                      (threadIdx.x / 32) * (32 / 4) * OutputTileIterator::kIterations + blockIdx.x * blockDim.x;

            float4* result_ptr = ((float4*)D + row * 4 + (threadIdx.x % 32) % 4);      // 4=32/8
            uint8_t* x_e8m0_ptr = ((uint8_t*)D_sf + row * 4 + (threadIdx.x % 32) % 4); // 4=32/8

            if (row < problem_m_size) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() + (threadIdx.x / 4) * 34 +
                     (threadIdx.x % 4) * 8); // + iter*(blockDim.x/4)*32); 40=32+8
                                             // padding of 32 elements? check bank conflicts
                                             // 10=40/4
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 / 32;
                    float var = c_sum2 / 32 - c_mean * c_mean;
                    float scale = 1.0;
                    if (var >= 0) {
                        scale = std::sqrt(var) * (2.92247856 / 6.) + 1e-8;
                    }

                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    float abs_max = 0.f;

#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float scale = abs_max + 1e-8f;
                    reinterpret_cast<uint32_t&>(scale) =
                        (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                    x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] /= scale;
                            mat_c[w * 8 + z] *= 3;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float4*)result_ptr) = *((float4*)result_reg);
            }
        }
    }
};

template <
    typename Shape_,                       ///< Shape of threadblock tile (concept: GemmShape)
    typename WarpMmaOperator_,             ///< Warp-level MMA operator (concept:
                                           ///< gemm::warp::MmaTensorOp)
    int PartitionsK,                       ///< Number of partitions of the K dimension
    typename OutputTileIterator_,          ///< Tile iterator reading and writing
                                           ///< output tensors
    typename AccumulatorFragmentIterator_, ///< Fragment iterator
                                           ///< selecting accumulators
    typename WarpTileIterator_,            ///< Warp-scoped tile iterator writing
                                           ///< accumulators to SMEM
    typename SharedLoadIterator_,          ///< Threadblock-scoped tile iterator
                                           ///< loading from SMEM
    typename OutputOp_,                    ///< Output operator
    typename Padding_,                     ///< Padding added to SMEM allocation to avoid
                                           ///< bank conflicts (concept: MatrixShape)
    int FragmentsPerPartition = 1,         ///< Used to coarsten the epilogue granularity
    int IterationsUnroll =                 ///< Used to reduce binary size when epilogue
                                           ///< op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class EpilogueQuantMxMask
    : public EpilogueBase<
          Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
          Padding_, FragmentsPerPartition>,
      public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_> {
  public:
    using Base = EpilogueBase<
        Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
        Padding_, FragmentsPerPartition>;

    using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;
    using Layout = layout::RowMajor;
    using LongIndex = typename Layout::LongIndex;

    /// Number of warps per block
    using WarpCount = typename Base::WarpCount;

    /// Number of threads per block
    static int const kBlockThreads = 32 * WarpCount::kCount;

    /// Per-thread accumulator tile type
    using AccumulatorTile = typename Base::AccumulatorTile;

    /// Numerical accumulation element type
    using ElementAccumulator = typename WarpMmaOperator::ElementC;

    /// Fragment type used by the accumulator tile's fragment iterator
    using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Vector type used by the global output iterator
    using OutputAccessType = Array<ElementOutput, OutputTileIterator::kElementsPerAccess>;

    using OutputGemmAccessType = Array<cutlass::bfloat16_t,
                                       OutputTileIterator::kElementsPerAccess>; // FIXME: float
    using OutputAccessType2 = Array<cutlass::float_e2m1_t,
                                    OutputTileIterator::kElementsPerAccess>; // FIXME: bfloat16_t

    /// Vector type used by the shared output iterator
    using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

    static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

    static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  public:
    static_assert(
        SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
        "Mismatch between shared load iterator and output tile iterator."
    );

    static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), "Divisibility");

    static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

  public:
    /// Aspect for when epilogue source is needed
    struct SourceAspectNeeded {
        OutputTileIterator source_iterator;

        typename OutputTileIterator::Fragment source_fragment;

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        static void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
            typename OutputTileIterator::Fragment const& source_fragment
        ) {

            OutputAccessType* output_frag_ptr = reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

            OutputGemmAccessType const* source_frag_ptr =
                reinterpret_cast<OutputGemmAccessType const*>(&source_fragment);

            int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
            }
        }

        /// Constructor
        CUTLASS_DEVICE
        SourceAspectNeeded(OutputTileIterator source_iterator) : source_iterator(source_iterator) {
            source_fragment.clear();
        }

        // Load addend source fragment from global memory
        CUTLASS_DEVICE
        void load() {
            source_iterator.load(source_fragment);
            ++source_iterator;
        }

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment
        ) {
            apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
        }
    };

  private:
    /// Loads fragment from shared memory aligned with output tensor
    SharedLoadIterator shared_load_iterator_;

    /// Thread index in the threadblock
    int thread_idx;

    /// Warp index in the threadblock
    int warp_idx;

  public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueQuantMxMask(
        typename Base::SharedStorage& shared_storage, ///< Shared storage object
        int thread_idx,                               ///< ID of a thread within the threadblock
        int warp_idx,                                 ///< ID of warp within threadblock
        int lane_idx
    ) ///< Id of thread within warp
        : Base(shared_storage, thread_idx, warp_idx, lane_idx), BaseStreamK(thread_idx),
          shared_load_iterator_(shared_storage.reference(), thread_idx), thread_idx(thread_idx), warp_idx(warp_idx) {}

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements two alternative codepaths, depending on whether the output op
    /// requires addend data to be loaded.
    CUTLASS_DEVICE
    void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator,      ///< Tile iterator for addend source
        cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size, uint8_t* D_mask
    ) {
        operator()(
            output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator), D, D_sf, problem_m_size,
            D_mask
        );
    }

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements a single codepath, regardless of whether the output op requires
    /// addend data to be loaded
    CUTLASS_DEVICE
    void unified(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator
    ) ///< Tile iterator for addend source
    {
        if (!output_op.is_source_needed()) {
            source_iterator.clear_mask();
            __syncthreads(); // Dummy (CUDA 11.0)
        }

        operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
    }

    template <class Seq> struct acc2smem;

    template <size_t... Seq> struct acc2smem<cutlass::index_sequence<Seq...>> {
        template <int Advance>
        CUTLASS_DEVICE static void
            helper(AccumulatorFragmentIterator accum_fragment_iterator, WarpTileIterator& warp_tile_iterator) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Advance; i++) {
                ++accum_fragment_iterator;
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;
            warp_tile_iterator.store(accum_fragment);
        }

        CUTLASS_DEVICE
        static void
            push(size_t pos, AccumulatorFragmentIterator const& iterator_begin, WarpTileIterator& warp_tile_iterator) {
            int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
        }
    };

    /// Streams the result to global memory
    template <typename SourceAspect>
    CUTLASS_DEVICE void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue8m0_t* D_sf, int problem_m_size, uint8_t* D_mask
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[32];
            uint32_t result_reg[4];
            uint8_t mask[4] = {0, 0, 0, 0};

            int row = iter * (32 / 4) + ((threadIdx.x % 32) / 4) +
                      (threadIdx.x / 32) * (32 / 4) * OutputTileIterator::kIterations + blockIdx.x * blockDim.x;

            float4* result_ptr = ((float4*)D + row);      // 4=32/8
            uint8_t* x_e8m0_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if ((threadIdx.x % 4) == 0 && row < problem_m_size) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() +
                     (threadIdx.x / 4) * 10); // + iter*(blockDim.x/4)*32); 40=32+8
                                              // padding of 32 elements? check bank conflicts
                                              // 10=40/4
                float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    float c_val = mat_c[i];
                    c_sum1 += c_val;
                    c_sum2 += c_val * c_val;
                }

                float c_mean = c_sum1 / 32;
                float var = c_sum2 / 32 - c_mean * c_mean;
                float scale = 1.0;
                if (var >= 0) {
                    scale = std::sqrt(var) * (2.92247856 / 6.) + 1e-8;
                }

                reinterpret_cast<uint32_t&>(scale) = (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;

                x_e8m0_ptr[0] = reinterpret_cast<uint32_t&>(scale) >> 23;

#pragma unroll
                for (int w = 0; w < 4; w++) {
                    for (int z = 0; z < 8; z++) {
                        mat_c[w * 8 + z] /= scale;
                    }
                    result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                }

                *((float4*)result_ptr) = *((float4*)result_reg);

#pragma unroll
                for (int i = 0; i < 32; i++) {
                    float c_val = mat_c[i];
                    float abs_val = fabsf(c_val);
                    if (abs_val < 6.f) {
                        // mask[i/8] |= 1 << (i%8);
                        mask[i >> 3] |= (1u << (i & 7));
                    }
                }
                //*((float*) clip_mask_ptr) = *((float*)mask);

                uint32_t mask32 = (uint32_t)mask[0] | ((uint32_t)mask[1] << 8) | ((uint32_t)mask[2] << 16) |
                                  ((uint32_t)mask[3] << 24);

                reinterpret_cast<uint32_t*>(D_mask)[row] = mask32;
            }

            /* if (kPartitionsK > 1) {
              plus<typename SharedLoadIterator::Fragment> add_fragments;

              CUTLASS_PRAGMA_UNROLL
              for (int i = 1; i < kPartitionsK; ++i) {
                shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
                shared_load_iterator_.load(aligned_accum_fragment[i]);
                aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0],
                                                          aligned_accum_fragment[i]);
              }

              shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) *
                                                       kSmemPointerOffset);
            } */

            //
            // Compute the output result
            //
            // if(iter!=0){
            /* typename OutputTileIterator::Fragment output_fragment;
            source.apply_output_operator(output_fragment, output_op,
                                        aligned_accum_fragment[0]); */

            //
            // Store the final result
            //

            // destination_iterator.store(output_fragment);
            //}
            //++destination_iterator;
        }
    }
};

/// Epilogue operator
template <
    typename Shape_,                       ///< Shape of threadblock tile (concept: GemmShape)
    typename WarpMmaOperator_,             ///< Warp-level MMA operator (concept:
                                           ///< gemm::warp::MmaTensorOp)
    int PartitionsK,                       ///< Number of partitions of the K dimension
    typename OutputTileIterator_,          ///< Tile iterator reading and writing
                                           ///< output tensors
    typename AccumulatorFragmentIterator_, ///< Fragment iterator
                                           ///< selecting accumulators
    typename WarpTileIterator_,            ///< Warp-scoped tile iterator writing
                                           ///< accumulators to SMEM
    typename SharedLoadIterator_,          ///< Threadblock-scoped tile iterator
                                           ///< loading from SMEM
    typename OutputOp_,                    ///< Output operator
    typename Padding_,                     ///< Padding added to SMEM allocation to avoid
                                           ///< bank conflicts (concept: MatrixShape)
    int FragmentsPerPartition = 1,         ///< Used to coarsten the epilogue granularity
    bool is_quartet = true, int RotationSize = 16,
    int IterationsUnroll = ///< Used to reduce binary size when epilogue
                           ///< op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class EpilogueQuantNv
    : public EpilogueBase<
          Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
          Padding_, FragmentsPerPartition>,
      public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_> {
  public:
    using Base = EpilogueBase<
        Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_,
        Padding_, FragmentsPerPartition>;

    using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;
    using Layout = layout::RowMajor;
    using LongIndex = typename Layout::LongIndex;

    /// Number of warps per block
    using WarpCount = typename Base::WarpCount;

    /// Number of threads per block
    static int const kBlockThreads = 32 * WarpCount::kCount;

    /// Per-thread accumulator tile type
    using AccumulatorTile = typename Base::AccumulatorTile;

    /// Numerical accumulation element type
    using ElementAccumulator = typename WarpMmaOperator::ElementC;

    /// Fragment type used by the accumulator tile's fragment iterator
    using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Vector type used by the global output iterator
    using OutputAccessType = Array<ElementOutput, OutputTileIterator::kElementsPerAccess>;

    using OutputGemmAccessType = Array<cutlass::bfloat16_t,
                                       OutputTileIterator::kElementsPerAccess>; // TODO: float
    using OutputAccessType2 = Array<cutlass::float_e2m1_t,
                                    OutputTileIterator::kElementsPerAccess>; // TODO: bfloat16_t

    /// Vector type used by the shared output iterator
    using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

    static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

    static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  public:
    static_assert(
        SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
        "Mismatch between shared load iterator and output tile iterator."
    );

    static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), "Divisibility");

    static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

  public:
    /// Aspect for when epilogue source is needed
    struct SourceAspectNeeded {
        OutputTileIterator source_iterator;

        typename OutputTileIterator::Fragment source_fragment;

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        static void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
            typename OutputTileIterator::Fragment const& source_fragment
        ) {

            OutputAccessType* output_frag_ptr = reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

            OutputGemmAccessType const* source_frag_ptr =
                reinterpret_cast<OutputGemmAccessType const*>(&source_fragment);

            int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
            }
        }

        /// Constructor
        CUTLASS_DEVICE
        SourceAspectNeeded(OutputTileIterator source_iterator) : source_iterator(source_iterator) {
            source_fragment.clear();
        }

        // Load addend source fragment from global memory
        CUTLASS_DEVICE
        void load() {
            source_iterator.load(source_fragment);
            ++source_iterator;
        }

        /// Invoke the output functor over each vector of output
        CUTLASS_DEVICE
        void apply_output_operator(
            typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment
        ) {
            apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
        }
    };

  private:
    /// Loads fragment from shared memory aligned with output tensor
    SharedLoadIterator shared_load_iterator_;

    /// Thread index in the threadblock
    int thread_idx;

    /// Warp index in the threadblock
    int warp_idx;

  public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueQuantNv(
        typename Base::SharedStorage& shared_storage, ///< Shared storage object
        int thread_idx,                               ///< ID of a thread within the threadblock
        int warp_idx,                                 ///< ID of warp within threadblock
        int lane_idx
    ) ///< Id of thread within warp
        : Base(shared_storage, thread_idx, warp_idx, lane_idx), BaseStreamK(thread_idx),
          shared_load_iterator_(shared_storage.reference(), thread_idx), thread_idx(thread_idx), warp_idx(warp_idx) {}

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements two alternative codepaths, depending on whether the output op
    /// requires addend data to be loaded.
    CUTLASS_DEVICE
    void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator,      ///< Tile iterator for addend source
        cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale, int problem_m_size
    ) {
        operator()(
            output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator), D, D_sf, global_scale,
            problem_m_size
        );
    }

    /// Perform the epilogue computations and stream the result to global memory.
    /// Implements a single codepath, regardless of whether the output op requires
    /// addend data to be loaded
    CUTLASS_DEVICE
    void unified(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        OutputTileIterator source_iterator
    ) ///< Tile iterator for addend source
    {
        if (!output_op.is_source_needed()) {
            source_iterator.clear_mask();
            __syncthreads(); // Dummy (CUDA 11.0)
        }

        operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
    }

    template <class Seq> struct acc2smem;

    template <size_t... Seq> struct acc2smem<cutlass::index_sequence<Seq...>> {
        template <int Advance>
        CUTLASS_DEVICE static void
            helper(AccumulatorFragmentIterator accum_fragment_iterator, WarpTileIterator& warp_tile_iterator) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Advance; i++) {
                ++accum_fragment_iterator;
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;
            warp_tile_iterator.store(accum_fragment);
        }

        CUTLASS_DEVICE
        static void
            push(size_t pos, AccumulatorFragmentIterator const& iterator_begin, WarpTileIterator& warp_tile_iterator) {
            int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
        }
    };

    /// Streams the result to global memory
    template <typename SourceAspect>
    CUTLASS_DEVICE void operator()(
        OutputOp const& output_op,               ///< Output operator
        OutputTileIterator destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const& accumulators,     ///< Complete warp-level accumulator tile
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale,
        int problem_m_size
    ) {
        static_assert(
            RotationSize == 16 || RotationSize == 32 || RotationSize == 64 || RotationSize == 128,
            "RotationSize must be 16/32/64/128"
        );
        EpilogueOpImpl<RotationSize, EpilogueQuantNv>::run(
            *this, output_op, destination_iterator, accumulators, source, D, D_sf, global_scale, problem_m_size
        );
    }

  private:
    template <int RS, typename Epilogue> struct EpilogueOpImpl;

    template <typename Epilogue> struct EpilogueOpImpl<16, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_16(std::forward<Args>(args)...);
        }
    };

    template <typename Epilogue> struct EpilogueOpImpl<32, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_32(std::forward<Args>(args)...);
        }
    };

    template <typename Epilogue> struct EpilogueOpImpl<64, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_64(std::forward<Args>(args)...);
        }
    };

    template <typename Epilogue> struct EpilogueOpImpl<128, Epilogue> {
        template <typename... Args> CUTLASS_DEVICE static void run(Epilogue& self, Args&&... args) {
            self.template op_128(std::forward<Args>(args)...);
        }
    };

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_16(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale,
        int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[16];
            uint32_t result_reg[4];

            int row = iter * (32 / 4) + ((threadIdx.x % 32) / 4) +
                      (threadIdx.x / 32) * (32 / 4) * OutputTileIterator::kIterations + blockIdx.x * blockDim.x;

            float2* result_ptr = ((float2*)D + row);      // 4=32/8
            uint8_t* x_e4m3_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if ((threadIdx.x % 4) == 0 && row < problem_m_size) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() +
                     (threadIdx.x / 4) * 10); // + iter*(blockDim.x/4)*32); 40=32+8
                                              // padding of 32 elements? check bank conflicts
                                              // 10=40/4
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 * reciprocal_approximate_ftz(16.0);
                    float scale =
                        std::sqrt(c_sum2 * reciprocal_approximate_ftz(16.0) - c_mean * c_mean) * (2.92247856 / 6.) +
                        1e-8;

                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(scale);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    float scale_q = e4m3_to_f32(fp8SFVal);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale = (scale_q > 0.f) ? reciprocal_approximate_ftz(scale_q) : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    /*
                    # based on:
                    https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L102

                    vec_max = torch.max(torch.abs(x), dim=-1,
                                keepdim=True)[0].to(torch.float32)
                    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
                    scale = torch.clamp(scale, max=448, min=-448)
                    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
                    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

                    scaled_x = x.to(torch.float32) * output_scale
                    */

                    float abs_max = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float global_scale_val = *global_scale;

                    float SFValue = global_scale_val * (abs_max * reciprocal_approximate_ftz(6.0));
                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    SFValue = float(tmp);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale =
                        SFValue != 0
                            ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(global_scale_val))
                            : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float2*)result_ptr) = *((float2*)result_reg);
            }
        }
    }

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_32(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale,
        int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[16];
            uint32_t result_reg[4];

            int row = iter * (32 / 4) * 2 + ((threadIdx.x % 32) / 4) * 2 + (threadIdx.x % 32) % 2 +
                      (threadIdx.x / 32) * (32 / 4) * 2 * OutputTileIterator::kIterations + blockIdx.x * blockDim.x * 2;

            float2* result_ptr = ((float2*)D + row);      // 4=32/8
            uint8_t* x_e4m3_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if ((threadIdx.x % 4) < 2 && row < problem_m_size * 2) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() + (threadIdx.x / 4) * 10 +
                     (threadIdx.x % 2) * 4); // + iter*(blockDim.x/4)*32); 40=32+8
                                             // padding of 32 elements? check bank conflicts
                                             // 10=40/4
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 * reciprocal_approximate_ftz(16.0);
                    float scale =
                        std::sqrt(c_sum2 * reciprocal_approximate_ftz(16.0) - c_mean * c_mean) * (2.92247856 / 6.) +
                        1e-8;

                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(scale);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    float scale_q = e4m3_to_f32(fp8SFVal);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale = (scale_q > 0.f) ? reciprocal_approximate_ftz(scale_q) : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    /*
                    # based on:
                    https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L102

                    vec_max = torch.max(torch.abs(x), dim=-1,
                                keepdim=True)[0].to(torch.float32)
                    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
                    scale = torch.clamp(scale, max=448, min=-448)
                    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
                    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

                    scaled_x = x.to(torch.float32) * output_scale
                    */

                    float abs_max = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float global_scale_val = *global_scale;

                    float SFValue = global_scale_val * (abs_max * reciprocal_approximate_ftz(6.0));
                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    SFValue = float(tmp);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale =
                        SFValue != 0
                            ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(global_scale_val))
                            : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float2*)result_ptr) = *((float2*)result_reg);
            }
        }
    }

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_64(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale,
        int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[16];
            uint32_t result_reg[4]; // FIXME: 2?

            int row = iter * (32 / 4) * 4 + ((threadIdx.x % 32) / 4) * 4 + (threadIdx.x % 32) % 4 +
                      (threadIdx.x / 32) * (32 / 4) * 4 * OutputTileIterator::kIterations + blockIdx.x * blockDim.x * 4;

            float2* result_ptr = ((float2*)D + row);      // 4=32/8
            uint8_t* x_e4m3_ptr = ((uint8_t*)D_sf + row); // 4=32/8

            if (row < problem_m_size * 4) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() + (threadIdx.x / 4) * 18 +
                     (threadIdx.x % 4) * 4); // + iter*(blockDim.x/4)*32); 40=32+8
                                             // padding of 32 elements? check bank conflicts
                                             // 10=40/4
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

                if constexpr (is_quartet) {
                    float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        c_sum1 += c_val;
                        c_sum2 += c_val * c_val;
                    }

                    float c_mean = c_sum1 * reciprocal_approximate_ftz(16.0);
                    float scale =
                        std::sqrt(c_sum2 * reciprocal_approximate_ftz(16.0) - c_mean * c_mean) * (2.92247856 / 6.) +
                        1e-8;

                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(scale);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    float scale_q = e4m3_to_f32(fp8SFVal);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale = (scale_q > 0.f) ? reciprocal_approximate_ftz(scale_q) : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                } else {
                    /*
                    # based on:
                    https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L102

                    vec_max = torch.max(torch.abs(x), dim=-1,
                                keepdim=True)[0].to(torch.float32)
                    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
                    scale = torch.clamp(scale, max=448, min=-448)
                    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
                    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

                    scaled_x = x.to(torch.float32) * output_scale
                    */

                    float abs_max = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float c_val = mat_c[i];
                        float abs_val = std::abs(c_val);
                        if (abs_val > abs_max)
                            abs_max = abs_val;
                    }

                    float global_scale_val = *global_scale;

                    float SFValue = global_scale_val * (abs_max * reciprocal_approximate_ftz(6.0));
                    uint8_t fp8SFVal;
                    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
                    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                    SFValue = float(tmp);

                    *x_e4m3_ptr = fp8SFVal;

                    float outputScale =
                        SFValue != 0
                            ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(global_scale_val))
                            : 0.0f;

#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        for (int z = 0; z < 8; z++) {
                            mat_c[w * 8 + z] *= outputScale;
                        }
                        result_reg[w] = fp32_vec_to_e2m1((float*)mat_c + w * 8);
                    }
                }

                *((float2*)result_ptr) = *((float2*)result_reg);
            }
        }
    }

    template <typename SourceAspect>
    CUTLASS_DEVICE void op_128(
        OutputOp const& output_op, OutputTileIterator destination_iterator, AccumulatorTile const& accumulators,
        SourceAspect source, cutlass::float_e2m1_t* D, cutlass::float_ue4m3_t* D_sf, ElementAccumulator* global_scale,
        int problem_m_size
    ) {
        // Iterator over warp-level accumulator fragment
        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source.load();
            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
                iter, accum_fragment_iterator, this->warp_tile_iterator_
            );

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator_.load(aligned_accum_fragment[0]);

            float mat_c[32];
            uint32_t result_reg[4];
            uint8_t out_s[2];

            int row = iter * (32 / 4) * 4 + ((threadIdx.x % 32) / 4) * 4 + (threadIdx.x % 32) % 4 +
                      (threadIdx.x / 32) * (32 / 4) * 4 * OutputTileIterator::kIterations + blockIdx.x * blockDim.x * 4;

            float4* result_ptr = ((float4*)D + row);        // 4=32/8
            uint16_t* x_e4m3_ptr = ((uint16_t*)D_sf + row); // 4=32/8

            if (row < problem_m_size * 4) {
                float4* raw =
                    ((float4*)this->shared_storage_.reference().data() + (threadIdx.x / 4) * 34 +
                     (threadIdx.x % 4) * 8); // + iter*(blockDim.x/4)*32); 40=32+8
                                             // padding of 32 elements? check bank conflicts
                                             // 10=40/4
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *((float4*)mat_c + i) = *((float4*)raw + i);
                }

#pragma unroll
                for (int nvs = 0; nvs < 2; ++nvs) {
                    if constexpr (is_quartet) {
                        float c_sum1 = 0.f, c_sum2 = 0.f;

#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            float c_val = mat_c[i + nvs * 16];
                            c_sum1 += c_val;
                            c_sum2 += c_val * c_val;
                        }

                        float c_mean = c_sum1 * reciprocal_approximate_ftz(16.0);
                        float scale =
                            std::sqrt(c_sum2 * reciprocal_approximate_ftz(16.0) - c_mean * c_mean) * (2.92247856 / 6.) +
                            1e-8;

                        uint8_t fp8SFVal;
                        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(scale);
                        reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                        float scale_q = e4m3_to_f32(fp8SFVal);

                        out_s[nvs] = fp8SFVal;

                        float outputScale = (scale_q > 0.f) ? reciprocal_approximate_ftz(scale_q) : 0.0f;

#pragma unroll
                        for (int w = 0; w < 2; w++) {
                            for (int z = 0; z < 8; z++) {
                                mat_c[w * 8 + z + nvs * 16] *= outputScale;
                            }
                            result_reg[w + nvs * 2] = fp32_vec_to_e2m1((float*)mat_c + w * 8 + nvs * 16);
                        }
                    } else {
                        /*
                        # based on:
                        https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L102

                        vec_max = torch.max(torch.abs(x), dim=-1,
                                    keepdim=True)[0].to(torch.float32)
                        scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
                        scale = torch.clamp(scale, max=448, min=-448)
                        scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
                        output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

                        scaled_x = x.to(torch.float32) * output_scale
                        */

                        float abs_max = 0.f;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            float c_val = mat_c[i + nvs * 16];
                            float abs_val = std::abs(c_val);
                            if (abs_val > abs_max)
                                abs_max = abs_val;
                        }

                        float global_scale_val = *global_scale;

                        float SFValue = global_scale_val * (abs_max * reciprocal_approximate_ftz(6.0));
                        uint8_t fp8SFVal;
                        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
                        reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                        SFValue = float(tmp);

                        out_s[nvs] = fp8SFVal;

                        float outputScale =
                            SFValue != 0
                                ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(global_scale_val))
                                : 0.0f;

#pragma unroll
                        for (int w = 0; w < 2; w++) {
                            for (int z = 0; z < 8; z++) {
                                mat_c[w * 8 + z + nvs * 16] *= outputScale;
                            }
                            result_reg[w + nvs * 2] = fp32_vec_to_e2m1((float*)mat_c + w * 8 + nvs * 16);
                        }
                    }
                }

                *((uint16_t*)x_e4m3_ptr) = *((uint16_t*)out_s);
                *((float4*)result_ptr) = *((float4*)result_reg);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
