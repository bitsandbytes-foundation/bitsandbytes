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
#pragma once

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_quant.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {
////////////////////////////////////////////////////////////////////////////////
template <
    typename Shape_, typename WarpMmaTensorOp_, int PartitionsK, typename OutputOp_, int ElementsPerAccess,
    bool ScatterD = false, typename PermuteDLayout = layout::NoPermute, bool is_quartet = true, int RotationSize = 32>
struct DefaultEpilogueTensorOpQuantMx
    : public DefaultEpilogueTensorOp<
          Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout> {
    using OutputOp = OutputOp_;
    using DefaultEpilogueTensorOp = DefaultEpilogueTensorOp<
        Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout>;

    using Epilogue = cutlass::epilogue::threadblock::EpilogueQuantMx<
        typename DefaultEpilogueTensorOp::Shape, typename DefaultEpilogueTensorOp::WarpMmaTensorOp,
        DefaultEpilogueTensorOp::kPartitionsK, typename DefaultEpilogueTensorOp::OutputTileIterator,
        typename DefaultEpilogueTensorOp::AccumulatorFragmentIterator,
        typename DefaultEpilogueTensorOp::WarpTileIterator, typename DefaultEpilogueTensorOp::SharedLoadIterator,
        OutputOp, typename DefaultEpilogueTensorOp::Padding, DefaultEpilogueTensorOp::kFragmentsPerIteration,
        is_quartet, RotationSize>;
};

template <
    typename Shape_, typename WarpMmaTensorOp_, int PartitionsK, typename OutputOp_, int ElementsPerAccess,
    bool ScatterD = false, typename PermuteDLayout = layout::NoPermute>
struct DefaultEpilogueTensorOpQuantMxMask
    : public DefaultEpilogueTensorOp<
          Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout> {
    using OutputOp = OutputOp_;
    using DefaultEpilogueTensorOp = DefaultEpilogueTensorOp<
        Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout>;

    using Epilogue = cutlass::epilogue::threadblock::EpilogueQuantMxMask<
        typename DefaultEpilogueTensorOp::Shape, typename DefaultEpilogueTensorOp::WarpMmaTensorOp,
        DefaultEpilogueTensorOp::kPartitionsK, typename DefaultEpilogueTensorOp::OutputTileIterator,
        typename DefaultEpilogueTensorOp::AccumulatorFragmentIterator,
        typename DefaultEpilogueTensorOp::WarpTileIterator, typename DefaultEpilogueTensorOp::SharedLoadIterator,
        OutputOp, typename DefaultEpilogueTensorOp::Padding, DefaultEpilogueTensorOp::kFragmentsPerIteration>;
};

template <
    typename Shape_, typename WarpMmaTensorOp_, int PartitionsK, typename OutputOp_, int ElementsPerAccess,
    bool ScatterD = false, typename PermuteDLayout = layout::NoPermute, bool is_quartet = true, int RotationSize = 16>
struct DefaultEpilogueTensorOpQuantNv
    : public DefaultEpilogueTensorOp<
          Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout> {
    using OutputOp = OutputOp_;
    using DefaultEpilogueTensorOp = DefaultEpilogueTensorOp<
        Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_, ElementsPerAccess, ScatterD, PermuteDLayout>;

    using Epilogue = cutlass::epilogue::threadblock::EpilogueQuantNv<
        typename DefaultEpilogueTensorOp::Shape, typename DefaultEpilogueTensorOp::WarpMmaTensorOp,
        DefaultEpilogueTensorOp::kPartitionsK, typename DefaultEpilogueTensorOp::OutputTileIterator,
        typename DefaultEpilogueTensorOp::AccumulatorFragmentIterator,
        typename DefaultEpilogueTensorOp::WarpTileIterator, typename DefaultEpilogueTensorOp::SharedLoadIterator,
        OutputOp, typename DefaultEpilogueTensorOp::Padding, DefaultEpilogueTensorOp::kFragmentsPerIteration,
        is_quartet, RotationSize>; // TODO: remove/add?
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
