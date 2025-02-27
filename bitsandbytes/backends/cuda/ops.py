import ctypes as ct
from math import prod
from typing import Optional, Sequence, Tuple

import torch

from bitsandbytes.functional import CUBLAS_Context, _cuda_device_of, _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ...cextension import lib


@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    A, B = B, A

    shapeA = A.shape
    shapeB = B.shape

    torch._check(A.dtype == torch.int8, lambda: "B must be int8")
    torch._check(B.dtype == torch.int8, lambda: "A must be int8")
    torch._check(A.ndim == 2, lambda: "Only two dimensional matrices are supported for argument B")
    torch._check(B.ndim in [2, 3], lambda: "Only two or three dimensional matrices are supported for argument A")
    torch._check(prod(shapeB) > 0, lambda: f"Input tensor dimensions need to be > 0: {shapeB}")
    torch._check(out is None or out.dtype == dtype)

    shapeC = (*shapeB[:-1], shapeA[0])

    k, m = shapeA
    n = prod(shapeB[:-1])
    lda = shapeA[-1]  # Weights (outputs, inputs)
    ldb = shapeB[-1]  # Activations (batch, tokens, inputs)
    ldc = shapeC[-1]  # Output (batch, tokens, outputs)

    torch._check(
        lda == ldb,
        lambda: f"int8_linear_matmul only supports B^T @ A. Inner dimensions do not match: B @ A = {shapeB} @ {shapeA}",
    )

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

    if has_error:
        if has_error == 100:
            # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
            # TODO: Warn and implement a fallback to fp32 compute?
            raise NotImplementedError("int8_linear_matmul not implemented!")
        else:
            raise RuntimeError(
                f"cublasLt ran into an error!\n"
                f"\t{shapeA=}, {shapeB=}, {shapeC=}\n"
                f"\t{(lda, ldb, ldc)=}\n"
                f"\t{(m, n, k)=}"
            )

    return out


@register_kernel("bitsandbytes::int8_mm_dequant", "cuda")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype=torch.float16,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

    if out is None:
        # TODO: deprecate out arg
        # Note: cuda kernel only currently supports fp16 output.
        # We'll later cast to desired dtype if needed.
        out = torch.empty_like(A, dtype=torch.float16)

    ptrA = get_ptr(A)
    ptrOut = get_ptr(out)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    numRows = ct.c_int32(prod(A.shape[:-1]))
    numCols = ct.c_int32(A.shape[-1])

    # Note: fused bias in the kernel is only supported for fp16
    # TODO(matthewdouglas): Consider supporting bf16 fused bias
    ptrBias = get_ptr(bias) if bias is not None and bias.dtype == torch.float16 else None

    with _cuda_device_of(A):
        lib.cdequant_mm_int32_fp16(
            ptrA, ptrRowStats, ptrColStats, ptrOut, ptrBias, numRows, numCols, _get_tensor_stream(A)
        )

    # Add bias separately if not fused in kernel
    if bias is not None and bias.dtype != torch.float16:
        out.add_(bias)

    return out.to(dtype)


@register_kernel("bitsandbytes::int8_vectorwise_quant", "cuda")
def _(A: torch.Tensor, threshold=0.0):
    torch._check(A.dtype == torch.float16, lambda: f"A must be float16, got {A.dtype}")
    torch._check(threshold >= 0.0, lambda: "threshold must be non-negative")

    rows = prod(A.shape[:-1])
    cols = A.shape[-1]

    row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)

    outlier_cols = None

    if threshold > 0.0:
        # TODO we could improve perf of this
        outliers = A.abs() >= threshold

        if outliers.any():
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)

    with _cuda_device_of(A):
        lib.cint8_vector_quant(
            get_ptr(A),
            get_ptr(out_row),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            _get_tensor_stream(A),
        )

    # Zero out values from outlier columns across all rows.
    # The kernel will handle this for outliers themselves, so we can optimize for rows=1.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    return out_row, row_stats, outlier_cols


@register_kernel("bitsandbytes::int8_double_quant", "cuda")
def _(
    A: torch.Tensor,
    col_stats: Optional[torch.Tensor] = None,
    row_stats: Optional[torch.Tensor] = None,
    out_col: Optional[torch.Tensor] = None,
    out_row: Optional[torch.Tensor] = None,
    threshold=0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # TODO: Optimize/write CUDA kernel for this?

    # Use CUDA kernel for rowwise and COO tensor
    quant_row, row_stats, outlier_cols = torch.ops.bitsandbytes.int8_vectorwise_quant(A, threshold=threshold)

    # PyTorch impl for colwise
    col_stats, outlier_mask = _get_col_absmax(A, threshold=threshold)
    if threshold > 0.0 and outlier_mask is not None:
        A = A.masked_fill(outlier_mask, 0.0)
    quant_col = torch.round(A.mul(127.0) / col_stats.unsqueeze(0)).to(torch.int8)

    if out_row is not None:
        quant_row = out_row.copy_(quant_row)
    if out_col is not None:
        quant_col = out_col.copy_(quant_col)

    return quant_row, quant_col, row_stats, col_stats.flatten().float(), outlier_cols


def _get_col_absmax(
    A: torch.Tensor,
    threshold=0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    torch._check(A.is_floating_point())

    outlier_mask = None

    absA = A.abs().view(-1, A.shape[-1])

    if threshold > 0.0:
        # Filter outliers from stats when enabled
        outlier_mask = absA >= threshold
        absA.masked_fill_(outlier_mask, 0.0)

    # shape [cols]; unsqueeze(0) gives [1,cols]
    col_stats = absA.amax(dim=0, keepdim=False).float()

    return col_stats, outlier_mask


@register_kernel("bitsandbytes::quantize_blockwise", "cuda")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    with _cuda_device_of(A):
        args = (
            get_ptr(code),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int32(blocksize),
            ct.c_int(A.numel()),
        )

        if A.dtype == torch.float16:
            lib.cquantize_blockwise_fp16(*args)
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_bf16(*args)
        elif A.dtype == torch.float32:
            lib.cquantize_blockwise_fp32(*args)
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "cuda")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(
        dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"Blockwise dequantization only supports 16bit/32bit floating types, got {dtype}",
    )

    out = torch.empty_like(A, dtype=dtype)

    with _cuda_device_of(A):
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

    return out


@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(quant_type in ["fp4", "nf4"])
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)

    with _cuda_device_of(A):
        args = (
            None,
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int32(blocksize),
            ct.c_int(n),
        )

        if A.dtype == torch.bfloat16:
            if quant_type == "fp4":
                lib.cquantize_blockwise_bf16_fp4(*args)
            else:
                lib.cquantize_blockwise_bf16_nf4(*args)
        elif A.dtype == torch.float16:
            if quant_type == "fp4":
                lib.cquantize_blockwise_fp16_fp4(*args)
            else:
                lib.cquantize_blockwise_fp16_nf4(*args)
        elif A.dtype == torch.float32:
            if quant_type == "fp4":
                lib.cquantize_blockwise_fp32_fp4(*args)
            else:
                lib.cquantize_blockwise_fp32_nf4(*args)

    return out, absmax


@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _(
    A: torch.Tensor, absmax: torch.Tensor, blocksize: int, quant_type: str, shape: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(quant_type in ["fp4", "nf4"])
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    out = torch.empty(shape, dtype=dtype, device=A.device)
    n = out.numel()

    stream = _get_tensor_stream(A)

    with _cuda_device_of(A):
        args = (
            None,
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(blocksize),
            ct.c_int(n),
            stream,
        )

        if out.dtype == torch.bfloat16:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_bf16_fp4(*args)
            else:
                lib.cdequantize_blockwise_bf16_nf4(*args)
        elif out.dtype == torch.float16:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_fp16_fp4(*args)
            else:
                lib.cdequantize_blockwise_fp16_nf4(*args)
        elif out.dtype == torch.float32:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_fp32_fp4(*args)
            else:
                lib.cdequantize_blockwise_fp32_nf4(*args)

    return out


@register_kernel("bitsandbytes::gemv_4bit", "cuda")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1), lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    torch._check(absmax.dtype == torch.float32, lambda: f"absmax must be float32, got {absmax.dtype}")
    torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    shape = (*A.shape[:-1], shapeB[0])
    out = torch.empty(shape, device=A.device, dtype=A.dtype)

    m = ct.c_int32(shapeB[0])
    n = ct.c_int32(1)
    k = ct.c_int32(shapeB[1])

    lda = m
    ldb = ct.c_int32((A.shape[-1] + 1) // 2)
    ldc = m

    stream = _get_tensor_stream(A)

    with _cuda_device_of(A):
        if A.dtype == torch.float16:
            lib.cgemm_4bit_inference_naive_fp16(
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
            lib.cgemm_4bit_inference_naive_bf16(
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
            lib.cgemm_4bit_inference_naive_fp32(
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

    return out
