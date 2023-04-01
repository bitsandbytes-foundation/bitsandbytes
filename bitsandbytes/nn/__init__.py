# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .modules import Int8Params, Linear8bitLt, StableEmbedding, OutlierAwareLinear, Fake4bitLinear, LinearFP8, LinearInt8, Linear8bitLtThresh, LinearInt8Cast, Linear8bitLt2, Linear8bitLtMixed, LinearFP8Global, LinearFP4, LinearFP8Mixed
from .triton_based_modules import SwitchBackLinear, SwitchBackLinearGlobal, SwitchBackLinearVectorized, StandardLinear
