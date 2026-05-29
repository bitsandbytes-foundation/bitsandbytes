from collections.abc import Sequence
import ctypes as ct
import logging
from typing import Optional
from warnings import warn

from packaging import version
import torch

from bitsandbytes.functional import _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ...cextension import ErrorHandlerMockBNBNativeLibrary, lib
from ..default.ops import _gemm_4bit_default_impl
from ..utils import _get_4bit_code, triton_available

logger = logging.getLogger(__name__)

# _int_mm is available in torch starting from 2.9 version
if version.parse(torch.__version__).release >= version.parse("2.9").release:

    @register_kernel("bitsandbytes::int8_linear_matmul", "xpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    # XPU SYCL kernels only support contiguous tensors.
    A = A.contiguous()
    args = (
        None,
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_int(blocksize),
        ct.c_int(out.numel()),
        _get_tensor_stream(A),
    )
    if dtype == torch.bfloat16:
        if quant_type == "fp4":
            lib.cdequantize_blockwise_bf16_fp4(*args)
        else:
            lib.cdequantize_blockwise_bf16_nf4(*args)
    elif dtype == torch.float16:
        if quant_type == "fp4":
            lib.cdequantize_blockwise_fp16_fp4(*args)
        else:
            lib.cdequantize_blockwise_fp16_nf4(*args)
    elif dtype == torch.float32:
        if quant_type == "fp4":
            lib.cdequantize_blockwise_fp32_fp4(*args)
        else:
            lib.cdequantize_blockwise_fp32_nf4(*args)


def _dequantize_blockwise_impl(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    # XPU SYCL kernels only support contiguous tensors.
    A = A.contiguous()
    args = (
        get_ptr(code),
        get_ptr(A),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_int(blocksize),
        ct.c_int(A.numel()),
        _get_tensor_stream(A),
    )
    if dtype == torch.float16:
        lib.cdequantize_blockwise_fp16(*args)
    elif dtype == torch.bfloat16:
        lib.cdequantize_blockwise_bf16(*args)
    elif dtype == torch.float32:
        lib.cdequantize_blockwise_fp32(*args)


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    m = ct.c_int32(1)
    n = ct.c_int32(shapeB[0])
    k = ct.c_int32(shapeB[1])

    lda = m
    ldb = ct.c_int32((A.shape[-1] + 1) // 2)
    ldc = m

    stream = _get_tensor_stream(A)
    if A.dtype == torch.float16:
        lib.cgemv_4bit_inference_fp16(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )
    elif A.dtype == torch.bfloat16:
        lib.cgemv_4bit_inference_bf16(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )
    elif A.dtype == torch.float32:
        lib.cgemv_4bit_inference_fp32(
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(blocksize),
            stream,
        )


@register_kernel("bitsandbytes::gemm_4bit", "xpu")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    bias: Optional[torch.Tensor] = None,
    absmax_8bit: Optional[torch.Tensor] = None,
    absmax_code: Optional[torch.Tensor] = None,
    absmax_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    K = A.shape[-1]
    M = A.numel() // K

    if M == 1:
        if K % blocksize == 0:
            if absmax_8bit is not None:
                absmax = (
                    torch.ops.bitsandbytes.dequantize_blockwise.default(
                        absmax_8bit, absmax, absmax_code, 256, torch.float32
                    )
                    + absmax_offset
                )

            code = _get_4bit_code(quant_type, A.device)
            out = torch.ops.bitsandbytes.gemv_4bit.default(A, B, shapeB, absmax, code, blocksize)

            if bias is not None:
                out = out + bias
            return out

        warn(
            f"inner dimension ({K}) is not aligned for fast kernel "
            f"with blocksize={blocksize}, falling back to slower implementation.",
            UserWarning,
        )

    return _gemm_4bit_default_impl(
        A,
        B,
        shapeB,
        absmax,
        blocksize,
        quant_type,
        bias,
        absmax_8bit=absmax_8bit,
        absmax_code=absmax_code,
        absmax_offset=absmax_offset,
    )


# SYCL should be faster for xpu, so at first checking if it is available.
if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):
    logger.info("Register sycl bitsandbytes kernels for XPU")

    # TODO: Remove the triton register when quantization sycl kernel is ready.
    if triton_available:
        from ..triton import ops as triton_ops

        register_kernel("bitsandbytes::quantize_blockwise", "xpu")(triton_ops.quantize_blockwise)
        register_kernel("bitsandbytes::quantize_4bit", "xpu")(triton_ops.quantize_4bit)
        register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "xpu")(
            triton_ops.optimizer_update_8bit_blockwise
        )
        register_kernel("bitsandbytes::optimizer_update_32bit", "xpu")(triton_ops.optimizer_update_32bit)

    @register_kernel("bitsandbytes::dequantize_4bit", "xpu")
    def _(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        out = torch.empty(shape, dtype=dtype, device=A.device)
        _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
        return out

    @register_kernel("bitsandbytes::dequantize_blockwise", "xpu")
    def _(
        A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
    ) -> torch.Tensor:
        out = torch.empty_like(A, dtype=dtype)
        _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)
        return out

    @register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")
    def _(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
        out: torch.Tensor,
    ) -> None:
        if out.dtype != dtype:
            raise ValueError(f"Expected out.dtype == {dtype}, got {out.dtype}")
        if out.shape != A.shape:
            raise ValueError(f"Expected out.shape == {A.shape}, got {out.shape}")
        _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)

    @register_kernel("bitsandbytes::gemv_4bit", "xpu")
    def _(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB: Sequence[int],
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
    ) -> torch.Tensor:
        shape = (*A.shape[:-1], shapeB[0])
        out = torch.empty(shape, device=A.device, dtype=A.dtype)
        _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)
        return out

    @register_kernel("bitsandbytes::gemv_4bit.out", "xpu")
    def _(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB: Sequence[int],
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        out: torch.Tensor,
    ) -> None:
        expected_shape = (*A.shape[:-1], shapeB[0])
        if out.shape != expected_shape:
            raise ValueError(f"Expected out.shape == {expected_shape}, got {out.shape}")
        if out.dtype != A.dtype:
            raise ValueError(f"Expected out.dtype == {A.dtype}, got {out.dtype}")
        _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)

elif triton_available:
    logger.info("Register triton bitsandbytes kernels for XPU")
    from ..triton import ops as triton_ops

    register_kernel("bitsandbytes::quantize_blockwise", "xpu")(triton_ops.quantize_blockwise)
    register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")(triton_ops.dequantize_blockwise_inplace)
    register_kernel("bitsandbytes::dequantize_blockwise", "xpu")(triton_ops.dequantize_blockwise)
    register_kernel("bitsandbytes::quantize_4bit", "xpu")(triton_ops.quantize_4bit)
    register_kernel("bitsandbytes::dequantize_4bit.out", "xpu")(triton_ops.dequantize_4bit_inplace)
    register_kernel("bitsandbytes::dequantize_4bit", "xpu")(triton_ops.dequantize_4bit)
    register_kernel("bitsandbytes::gemv_4bit", "xpu")(triton_ops.gemv_4bit)
    register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "xpu")(triton_ops.optimizer_update_8bit_blockwise)
    register_kernel("bitsandbytes::optimizer_update_32bit", "xpu")(triton_ops.optimizer_update_32bit)
else:
    logger.warning("Register pytorch bitsandbytes kernels for XPU because no native library or triton packages found.")
