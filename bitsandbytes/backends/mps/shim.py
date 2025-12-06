import ctypes as ct
from dataclasses import dataclass
from typing import Callable

import torch


class _BNBMPSTensor(ct.Structure):
    _fields_ = [
        ("storage", ct.c_void_p),
        ("byte_offset", ct.c_size_t),
        ("nbytes", ct.c_size_t),
    ]


@dataclass(slots=True)
class MPSTensorShim:
    """
    Lightweight wrapper that keeps a Tensor alive while exposing its Metal storage.

    PyTorch stores an ``id<MTLBuffer>`` inside the tensor's untyped storage data
    pointer on MPS.  We capture that pointer once and forward the storage offset
    so native kernels can bind the correct buffer without any host copies.
    """

    tensor: torch.Tensor
    struct: _BNBMPSTensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "MPSTensorShim":
        if hasattr(tensor, "untyped_storage"):
            storage = tensor.untyped_storage()
        else:
            storage = tensor.storage()

        storage_ptr = storage.data_ptr()
        byte_offset = tensor.storage_offset() * tensor.element_size()
        nbytes = tensor.nbytes

        struct = _BNBMPSTensor(
            ct.c_void_p(storage_ptr),
            ct.c_size_t(byte_offset),
            ct.c_size_t(nbytes),
        )
        return cls(tensor=tensor, struct=struct)


# def configure_mps_blockwise_kernel(fn: Callable[[object], None]) -> None:
#     """
#     Ensure ctypes knows the function expects our tensor shim structs by value.
#     """

#     try:
#         argtypes = getattr(fn, "argtypes")
#     except AttributeError:
#         argtypes = None

#     desired = [_BNBMPSTensor, _BNBMPSTensor, _BNBMPSTensor, ct.c_int32, ct.c_int32]
#     if argtypes != desired:
#         fn.argtypes = desired
#     if getattr(fn, "restype", None) is not None:
#         fn.restype = None

