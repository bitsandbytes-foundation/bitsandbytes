# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import sys

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
# Eventually we may remove this but it is currently required for compatibility.
features = {"multi_backend"}
supported_torch_devices = {
    "cpu",
    "cuda",  # NVIDIA/AMD GPU
    "xpu",  # Intel GPU
    "hpu",  # Intel Gaudi
    "npu",  # Ascend NPU
    "mps",  # Apple Silicon
}

if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops

if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .backends.xpu import ops as xpu_ops

if importlib.util.find_spec("habana_frameworks") and importlib.util.find_spec("habana_frameworks.torch"):
    # In case not automatically imported
    import habana_frameworks.torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .backends.hpu import ops as hpu_ops


def _import_backends():
    """
    Discover and autoload all available backends installed as separate packages.
    Packages with an entrypoint for "bitsandbytes.backends" will be loaded.
    Inspired by PyTorch implementation: https://pytorch.org/tutorials/prototype/python_extension_autoload.html
    """
    from importlib.metadata import entry_points

    extensions = entry_points(group="bitsandbytes.backends")

    for ext in extensions:
        try:
            entry = ext.load()
            entry()
        except Exception as e:
            raise RuntimeError(f"bitsandbytes: failed to load backend {ext.name}: {e}") from e


_import_backends()

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.49.0"
