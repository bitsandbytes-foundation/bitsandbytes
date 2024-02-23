import ctypes
from typing import Optional

import torch


def pre_call(device):
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    return prev_device


def post_call(prev_device):
    torch.cuda.set_device(prev_device)


def get_ptr(A: Optional[torch.Tensor]) -> Optional[ctypes.c_void_p]:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    if A is None:
        return None
    else:
        return ctypes.c_void_p(A.data.data_ptr())


def is_on_gpu(tensors):
    on_gpu = True
    gpu_ids = set()
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        is_paged = getattr(t, "is_paged", False)
        on_gpu &= t.device.type == "cuda" or is_paged
        if not is_paged:
            gpu_ids.add(t.device.index)
    if not on_gpu:
        raise TypeError(
            f"All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}"
        )
    if len(gpu_ids) > 1:
        raise TypeError(
            f"Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}"
        )
    return on_gpu
