# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from . import _ops, research, utils
from .autograd._functions import (
    MatmulLtState,
    matmul,
    matmul_4bit,
)
from .backends.cpu import ops as cpu_ops
from .backends.cuda import ops as cuda_ops  ## TODO: We would guard this for CUDA only
from .backends.default import ops as default_ops
from .nn import modules
from .optim import adam

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.45.4.dev0"
