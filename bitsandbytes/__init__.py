# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Import the dynamically generated version from _version.py  (see setup.py)
from ._version import __version__  # isort: skip # type: ignore

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
from .backends import backends, register_backend
from .backends.cpu import CPUBackend
from .backends.npu import NPUBackend
from .cextension import lib

features = {"multi_backend"}
supported_torch_devices = {
    "cuda",  # includes ROCm
    "npu",  # Ascend NPU
    "xpu",  # Intel GPU
    "cpu",
}

# Always register the CPU backend.
register_backend("cpu", CPUBackend())

# Register either CUDA or ROCm backend, if available.
# Only one of these backends can be used at a time, since the torch.device semantics are
# the same for both torch+rocm and torch+cuda (e.g. device name is "cuda")
if torch.cuda.is_available():
    # TODO: Consider deferring loading of cextension - should backend class implement that?

    if torch.version.cuda:
        from .backends.cuda import CUDABackend

        register_backend("cuda", CUDABackend())
    elif torch.version.hip:
        from .backends.rocm import ROCmBackend

        register_backend("cuda", ROCmBackend())

# Register MPS backend, if available.
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    from .backends.mps import MPSBackend

    register_backend("mps", MPSBackend())

# Register Intel XPU backend, if available.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .backends.xpu import XPUBackend

    register_backend("xpu", XPUBackend())

# Register Ascend NPU backend, if available.
if hasattr(torch, "npu") and torch.npu.is_available():
    register_backend("npu", NPUBackend())


# import module after decided backends
if backends:
    from .nn import modules

# TODO: Other potential backends:
# XLA - Google TPU / PJRT runtime
# HPU - Habana / Intel Gaudi
# IPU - Graphcore
# Note that we may not map 1:1 with a device type, e.g. SYCL, XLA
# In this case, it will be up to each backend to dispatch as needed

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}
