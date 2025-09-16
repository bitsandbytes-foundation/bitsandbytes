from collections.abc import Sequence
import ctypes as ct
import logging

from packaging import version
import torch

from bitsandbytes.functional import _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ...cextension import ErrorHandlerMockBNBNativeLibrary, lib
from ..utils import triton_available

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
        torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
        torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
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
        torch._check(
            out.shape == (*A.shape[:-1], shapeB[0]),
            lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
        )
        torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
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
