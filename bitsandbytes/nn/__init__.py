# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ..backends import backends
from .modules import (
    Embedding,
    Int8Params,
    Linear4bit,
    Linear8bitLt,
    LinearFP4,
    LinearNF4,
    OutlierAwareLinear,
    Params4bit,
    StableEmbedding,
    SwitchBackLinearBnb,
)

# CPU and XPU backend do not need triton, and XPU so not support triton for now.
if "xpu" not in backends.keys() and len(backends.keys()) > 1:
    from .triton_based_modules import (
        StandardLinear,
        SwitchBackLinear,
        SwitchBackLinearGlobal,
        SwitchBackLinearVectorwise,
    )
