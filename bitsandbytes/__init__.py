# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from . import _ops, research, utils
from .autograd._functions import (
    MatmulLtState,
    matmul,
    matmul_4bit,
)
from .backends.cpu import ops as cpu_ops
from .backends.default import ops as default_ops
from .nn import modules
from .optim import adam

# This is a signal for integrations with transformers/diffusers.
# Eventually, we will remove this and check based on release version.
features = {"multi-backend"}
supported_torch_devices = {
    "cuda",
    "cpu",
    # "mps",
    # "xpu",
    # "hpu",
    # "npu",
}

if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.45.4"
