import pytest
import torch
from bitsandbytes.utils import get_cuda_devices

def get_gpu_devices():
    """
    Returns a list of all GPU devices supported by Torch in the current environment (i.e. devices that Torch was built with
    support for and are present in the current environment).
    """
    ret = []
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        ret.append("mps")
    if torch.cuda.is_available():
        ret += get_cuda_devices()
    return ret

def skip_if_no_gpu():
    return pytest.mark.skipif(not get_gpu_devices() or not torch.cuda.is_available(), reason="No GPU device found by Torch")

def skip_if_no_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA-compatible device found by Torch")
