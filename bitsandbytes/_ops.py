import ctypes as ct
from math import prod
from typing import Optional

import torch

from .cextension import lib
from .functional import CUBLAS_Context, _cuda_device_of, _get_tensor_stream, get_ptr, is_on_gpu

_IS_TORCH_GTE_24 = False

if hasattr(torch.library, "register_fake"):
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl

# Define op
# TODO: mutable output arg as alias of return can be challenging;
#       consider a separate op without aliased return:
#           int8_linear_matmul_out(
#               Tensor A, Tensor B, Tensor out, ScalarType dtype=int32
#           ) -> ()
#           return () instead of `None` for compatibility, see here: https://github.com/pytorch/pytorch/issues/125044
torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B, Tensor(a!)? out=None, ScalarType dtype=int32) -> Tensor(a!)",
)


# Fake/abstract op
@register_fake("bitsandbytes::int8_linear_matmul")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    shapeC = (*A.shape[:-1], B.shape[0])
    if out is None:
        return torch.empty(shapeC, device=A.device, dtype=dtype)
    return out


# CPU implementation
@register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


# MPS impl
@register_kernel("bitsandbytes::int8_linear_matmul", "mps")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    pass


# XPU impl
@register_kernel("bitsandbytes::int8_linear_matmul", "xpu")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    pass


# Ascend NPU impl
@register_kernel("bitsandbytes::int8_linear_matmul", "npu")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    pass


# CUDA/ROCm impl
@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    A, B = B, A

    shapeA = A.shape
    shapeB = B.shape

    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert A.ndim == 2, "Only two dimensional matrices are supported for argument B"
    assert B.ndim in [2, 3], "Only two or three dimensional matrices are supported for argument A"
    assert prod(shapeB) > 0, f"Input tensor dimensions need to be > 0: {shapeB}"
    assert out is None or out.dtype == dtype

    shapeC = (*shapeB[:-1], shapeA[0])

    k, m = shapeA
    n = prod(shapeB[:-1])
    lda = shapeA[-1]  # Weights (outputs, inputs)
    ldb = shapeB[-1]  # Activations (batch, tokens, inputs)
    ldc = shapeC[-1]  # Output (batch, tokens, outputs)

    assert (
        lda == ldb
    ), f"int8_linear_matmul only supports B^T @ A. Inner dimensions do not match: B @ A = {shapeB} @ {shapeA}"

    # cuBLASLt does not support int8 matmul with inner dimensions that are not divisible by 4.
    # We'll fall back to a slower fp32 calculation in this circumstance.
    # Fortunately, this should not be very common.
    if lda % 4 != 0:
        result = torch.matmul(B.float(), A.float().t()).to(torch.int32)
        if out is not None:
            result = out.copy_(result)
        return result

    if out is None:
        out = torch.empty(shapeC, device=A.device, dtype=dtype)

    is_on_gpu([A, B, out])

    with _cuda_device_of(A):
        ctx = CUBLAS_Context.get_instance().get_context(A.device)
        ptrA = get_ptr(A)
        ptrB = get_ptr(B)
        ptrC = get_ptr(out)
        ptrRowScale = None
        m = ct.c_int32(m)
        n = ct.c_int32(n)
        k = ct.c_int32(k)
        lda = ct.c_int32(lda)
        ldb = ct.c_int32(ldb)
        ldc = ct.c_int32(ldc)
        stream = _get_tensor_stream(A)

        if dtype == torch.int32:
            has_error = lib.cigemmlt_32(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)
        else:
            has_error = lib.cigemmlt_8(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)

    if has_error == 100:  # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
        raise NotImplementedError("int8_linear_matmul not implemented!")

    if has_error:
        raise RuntimeError(
            f"cublasLt ran into an error!\n"
            f"\t{shapeA=}, {shapeB=}, {shapeC=}\n"
            f"\t{(lda, ldb, ldc)=}\n"
            f"\t{(m, n, k)=}"
        )

    return out
