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
  \brief Functor performing linear combination operations used by dequantize
  epilogues.
*/
#pragma once

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

struct MyScaleType {
  enum Kind {
    Quantize,
  };
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_ = cutlass::bfloat16_t, //TODO: float
          MyScaleType::Kind Scale = MyScaleType::Quantize,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename ElementSource_ = cutlass::bfloat16_t> //TODO: float
class LinearCombinationQuantMx {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const MyScaleType::Kind kScale = MyScaleType::Quantize;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute beta;

    CUTLASS_HOST_DEVICE
    Params() : beta(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta) : beta(beta) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute beta_ = ElementCompute(0);

 public:
  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationQuantMx(Params const &params) { beta_ = params.beta; }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source) const {
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    FragmentOutput result;
    uint32_t *result_ptr = reinterpret_cast<uint32_t *>(&result);

    const cutlass::bfloat16_t *acc_ptr =
        reinterpret_cast<const cutlass::bfloat16_t *>(&converted_accumulator);

   return result;
  }
};

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_ = cutlass::bfloat16_t, //FIXME: float
          MyScaleType::Kind Scale = MyScaleType::Quantize,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, //TODO: change?
          typename ElementSource_ = cutlass::bfloat16_t> //FIXME: float
class LinearCombinationQuantMxMask {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const MyScaleType::Kind kScale = MyScaleType::Quantize;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute beta;

    CUTLASS_HOST_DEVICE
    Params() : beta(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta) : beta(beta) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute beta_ = ElementCompute(0);

 public:
  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationQuantMxMask(Params const &params) { beta_ = params.beta; }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source) const {
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    FragmentOutput result;
    uint32_t *result_ptr = reinterpret_cast<uint32_t *>(&result);

    const cutlass::bfloat16_t *acc_ptr =
        reinterpret_cast<const cutlass::bfloat16_t *>(&converted_accumulator);

   return result;
  }
};

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_ = cutlass::bfloat16_t, //TODO: float
          MyScaleType::Kind Scale = MyScaleType::Quantize,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename ElementSource_ = cutlass::bfloat16_t> //TODO: float
class LinearCombinationQuantNv {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const MyScaleType::Kind kScale = MyScaleType::Quantize;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute beta;

    CUTLASS_HOST_DEVICE
    Params() : beta(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta) : beta(beta) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute beta_ = ElementCompute(0);

 public:
  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationQuantNv(Params const &params) { beta_ = params.beta; }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source) const {
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    FragmentOutput result;
    uint32_t *result_ptr = reinterpret_cast<uint32_t *>(&result);

    const cutlass::bfloat16_t *acc_ptr =
        reinterpret_cast<const cutlass::bfloat16_t *>(&converted_accumulator);

   return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
