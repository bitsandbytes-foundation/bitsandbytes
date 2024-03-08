# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import cuda_setup, research, utils
from .autograd._functions import (
    MatmulLtState,
    bmm_cublas,
    matmul,
    matmul_4bit,
    matmul_cublas,
    mm_cublas,
)
from .cextension import COMPILED_WITH_CUDA
from .nn import modules

if COMPILED_WITH_CUDA:
    from .optim import adam

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.43.0"

PACKAGE_GITHUB_URL = "https://github.com/TimDettmers/bitsandbytes"
