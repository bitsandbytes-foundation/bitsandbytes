import functools
from io import BytesIO
from itertools import product
import os
import random
from typing import Any

import torch

from bitsandbytes.cextension import HIP_ENVIRONMENT

test_dims_rng = random.Random(42)


TRUE_FALSE = (True, False)
BOOLEAN_TRIPLES = list(product(TRUE_FALSE, repeat=3))  # all combinations of (bool, bool, bool)
BOOLEAN_TUPLES = list(product(TRUE_FALSE, repeat=2))  # all combinations of (bool, bool)


@functools.cache
def get_available_devices():
    if "BNB_TEST_DEVICE" in os.environ:
        # If the environment variable is set, use it directly.
        return [os.environ["BNB_TEST_DEVICE"]]

    devices = [] if HIP_ENVIRONMENT else ["cpu"]

    if hasattr(torch, "accelerator"):
        # PyTorch 2.6+ - determine accelerator using agnostic API.
        if torch.accelerator.is_available():
            devices += [str(torch.accelerator.current_accelerator())]
    else:
        if torch.cuda.is_available():
            devices += ["cuda"]

        if torch.backends.mps.is_available():
            devices += ["mps"]

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices += ["xpu"]

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        custom_backend_module = getattr(torch, custom_backend_name, None)
        custom_backend_is_available_fn = getattr(custom_backend_module, "is_available", None)

        if custom_backend_is_available_fn and custom_backend_module.is_available():
            devices += [custom_backend_name]

    return devices


def torch_save_to_buffer(obj):
    buffer = BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return buffer


def torch_load_from_buffer(buffer):
    buffer.seek(0)
    obj = torch.load(buffer, weights_only=False)
    buffer.seek(0)
    return obj


def get_test_dims(min: int, max: int, *, n: int) -> list[int]:
    return [test_dims_rng.randint(min, max) for _ in range(n)]


def format_with_label(label: str, value: Any) -> str:
    if isinstance(value, bool):
        formatted = "T" if value else "F"
    elif isinstance(value, (list, tuple)) and all(isinstance(v, bool) for v in value):
        formatted = "".join("T" if b else "F" for b in value)
    elif isinstance(value, torch.dtype):
        formatted = describe_dtype(value)
    else:
        formatted = str(value)
    return f"{label}={formatted}"


def id_formatter(label: str):
    """
    Return a function that formats the value given to it with the given label.
    """
    return lambda value: format_with_label(label, value)


DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.bool: "bool",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
}


def describe_dtype(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype) or str(dtype).rpartition(".")[2]


def is_supported_on_hpu(
    quant_type: str = "nf4", dtype: torch.dtype = torch.bfloat16, quant_storage: torch.dtype = torch.uint8
) -> bool:
    """
    Check if the given quant_type, dtype and quant_storage are supported on HPU.
    """
    if quant_type == "fp4" or dtype == torch.float16 or quant_storage not in (torch.uint8, torch.bfloat16):
        return False
    return True
