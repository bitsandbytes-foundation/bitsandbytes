# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from . import research, utils
from .autograd._functions import (
    MatmulLtState,
    bmm_cublas,
    matmul,
    matmul_4bit,
    matmul_cublas,
    mm_cublas,
)
from .cextension import lib
from .nn import modules
from .backends import register_backend

from .backends.cpu import CPUBackend
register_backend("cpu", CPUBackend)

if lib and lib.compiled_with_cuda:
    from .backends.cuda import CUDABackend
    from .optim import adam

    register_backend("cuda", CUDABackend())

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.44.0.dev"
