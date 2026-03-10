from collections.abc import Sequence
import ctypes as ct
from math import prod
from typing import Optional

import torch

from bitsandbytes.functional import CUBLAS_Context, _cuda_device_of, _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ...cextension import ROCM_WARP_SIZE_64, lib


@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _(A: torch.Tensor, B: torch.Tensor):
    out = torch.empty((*A.shape[:-1], B.shape[0]), device=A.device, dtype=torch.int32)
    return _int8_linear_matmul_impl(A, B, out)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    A, B = B, A

    shapeA = A.shape
    shapeB = B.shape

    torch._check(A.dtype == torch.int8, lambda: "B must be int8")
    torch._check(B.dtype == torch.int8, lambda: "A must be int8")
    torch._check(A.ndim == 2, lambda: "Only two dimensional matrices are supported for argument B")
    torch._check(B.ndim in [2, 3], lambda: "Only two or three dimensional matrices are supported for argument A")
    torch._check(prod(shapeB) > 0, lambda: f"Input tensor dimensions need to be > 0: {shapeB}")
    torch._check(out.dtype == torch.int32)

    shapeC = (*shapeB[:-1], shapeA[0])
    torch._check(out.shape == shapeC, lambda: f"Output shape {out.shape} does not match expected shape {shapeC}")

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
        return out.copy_(result)

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

        has_error = lib.cigemmlt_32(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)

    if has_error:
        if has_error == 100:
            # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
            # TODO: Warn and implement a fallback to fp32 compute?
            raise NotImplementedError("int8_linear_matmul not implemented!")
        else:
            raise RuntimeError(
                f"cublasLt ran into an error!\n\t{shapeA=}, {shapeB=}, {shapeC=}\n\t{(lda, ldb, ldc)=}\n\t{(m, n, k)=}"
            )

    return out


@register_kernel("bitsandbytes::int8_mm_dequant", "cuda")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

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

    return out.to(dtype or torch.float16)


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
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

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
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Use CUDA kernel for rowwise and COO tensor
    quant_row, row_stats, outlier_cols = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        A,
        threshold=threshold,
    )

    # PyTorch impl for colwise
    col_stats, outlier_mask = _get_col_absmax(A, threshold=threshold)
    if threshold > 0.0 and outlier_mask is not None:
        A = A.masked_fill(outlier_mask, 0.0)
    quant_col = torch.round(A.mul(127.0) / col_stats.unsqueeze(0)).to(torch.int8)

    return quant_row, quant_col, row_stats, col_stats.flatten().float(), outlier_cols


def _get_col_absmax(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    A = A.contiguous()
    torch._check_is_size(blocksize)

    if ROCM_WARP_SIZE_64:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    else:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])

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
    out = torch.empty_like(A, dtype=dtype)
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_blockwise.out", "cuda")
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


def _dequantize_blockwise_impl(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    A = A.contiguous()
    if ROCM_WARP_SIZE_64:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    else:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])

    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(
        dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"Blockwise dequantization only supports 16bit/32bit floating types, got {dtype}",
    )

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


@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    A = A.contiguous()
    if ROCM_WARP_SIZE_64:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    else:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])

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
            ct.c_int32(n),
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


@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    A = A.contiguous()
    if ROCM_WARP_SIZE_64:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    else:
        torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64, 32])

    torch._check(quant_type in ["fp4", "nf4"])
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    with _cuda_device_of(A):
        args = (
            None,
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(blocksize),
            ct.c_int32(out.numel()),
            _get_tensor_stream(A),
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


@register_kernel("bitsandbytes::gemv_4bit", "cuda")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    shape = (*A.shape[:-1], shapeB[0])
    out = torch.empty(shape, device=A.device, dtype=A.dtype)
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)
    return out


@register_kernel("bitsandbytes::gemv_4bit.out", "cuda")
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


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)

    # Note: these checks are not strictly necessary, and cost more than they are worth, so they are commented out for now.
    # torch._check(
    #     A.numel() == A.size(-1),
    #     lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}",
    # )
    # torch._check(
    #     A.dtype in [torch.float16, torch.bfloat16, torch.float32],
    #     lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    # )
    # torch._check(
    #     B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
    #     lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    # )
    # torch._check(absmax.dtype == torch.float32, lambda: f"absmax must be float32, got {absmax.dtype}")
    # torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

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


"""C FUNCTIONS FOR OPTIMIZERS"""
str2optimizer32bit = {
    "adam": (
        lib.cadam32bit_grad_fp32,
        lib.cadam32bit_grad_fp16,
        lib.cadam32bit_grad_bf16,
    ),
    "momentum": (
        lib.cmomentum32bit_grad_32,
        lib.cmomentum32bit_grad_16,
    ),
    "rmsprop": (
        lib.crmsprop32bit_grad_32,
        lib.crmsprop32bit_grad_16,
    ),
    "lion": (
        lib.clion32bit_grad_fp32,
        lib.clion32bit_grad_fp16,
        lib.clion32bit_grad_bf16,
    ),
    "adagrad": (
        lib.cadagrad32bit_grad_32,
        lib.cadagrad32bit_grad_16,
    ),
    "lamb": (
        lib.cadam32bit_grad_fp32,
        lib.cadam32bit_grad_fp16,
        lib.cadam32bit_grad_bf16,
    ),
    "ademamix": (
        lib.cademamix32bit_grad_fp32,
        lib.cademamix32bit_grad_fp16,
        lib.cademamix32bit_grad_bf16,
    ),
    "lars": (
        lib.cmomentum32bit_grad_32,
        lib.cmomentum32bit_grad_16,
    ),
}

str2optimizer8bit_blockwise = {
    "adam": (
        lib.cadam_8bit_blockwise_grad_fp32,
        lib.cadam_8bit_blockwise_grad_fp16,
        lib.cadam_8bit_blockwise_grad_bf16,
    ),
    "momentum": (
        lib.cmomentum_8bit_blockwise_grad_fp32,
        lib.cmomentum_8bit_blockwise_grad_fp16,
        lib.cmomentum_8bit_blockwise_grad_bf16,
    ),
    "rmsprop": (
        lib.crmsprop_8bit_blockwise_grad_fp32,
        lib.crmsprop_8bit_blockwise_grad_fp16,
        lib.crmsprop_8bit_blockwise_grad_bf16,
    ),
    "lion": (
        lib.clion_8bit_blockwise_grad_fp32,
        lib.clion_8bit_blockwise_grad_fp16,
        lib.clion_8bit_blockwise_grad_bf16,
    ),
    "adagrad": (
        lib.cadagrad_8bit_blockwise_grad_fp32,
        lib.cadagrad_8bit_blockwise_grad_fp16,
        lib.cadagrad_8bit_blockwise_grad_bf16,
    ),
    "ademamix": (
        lib.cademamix_8bit_blockwise_grad_fp32,
        lib.cademamix_8bit_blockwise_grad_fp16,
        lib.cademamix_8bit_blockwise_grad_bf16,
    ),
}


def _optimizer_update_32bit_impl(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    unorm_vec: Optional[torch.Tensor],
    max_unorm: float,
    param_norm: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    step: int,
    lr: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    optim_fns = str2optimizer32bit.get(optimizer_name, None)
    if optim_fns is None:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {list(str2optimizer32bit.keys())}"
        )
    if g.dtype == torch.float32:
        optim_func = optim_fns[0]
    elif g.dtype == torch.float16:
        optim_func = optim_fns[1]
    elif g.dtype == torch.bfloat16 and len(optim_fns) == 3:
        optim_func = optim_fns[2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )

    with _cuda_device_of(g):
        optim_func(
            get_ptr(g),
            get_ptr(p),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(beta3),
            ct.c_float(alpha),
            ct.c_float(eps),
            ct.c_float(weight_decay),
            ct.c_int32(step),
            ct.c_float(lr),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )


def _optimizer_update_8bit_blockwise_impl(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: torch.Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: torch.Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    # torch._check(
    #     g.numel() == p.numel(),
    #     lambda: f"g and p must have the same number of elements, got {g.numel()} and {p.numel()}",
    # )
    # compute_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # torch._check(
    #     g.dtype in compute_dtypes,
    #     lambda: f"g must be bfloat16, float16, or float32, got {g.dtype}",
    # )
    # torch._check(
    #     g.dtype == p.dtype,
    #     lambda: f"Expected all tensors to have the same dtype, got g.dtype={g.dtype}, p.dtype={p.dtype}",
    # )
    # torch._check(
    #     state1.dtype == torch.uint8,
    #     lambda: f"state1 must be uint8, got {state1.dtype}",
    # )
    # torch._check(
    #     qmap1.dtype == absmax1.dtype == torch.float32,
    #     lambda: f"Expected qmap1 and absmax1 to be float32, got qmap1.dtype={qmap1.dtype}, absmax1.dtype={absmax1.dtype}",
    # )
    # if state2 is not None:
    #     torch._check(
    #         state2.dtype == torch.uint8,
    #         lambda: f"state2 must be uint8, got {state2.dtype}",
    #     )
    #     torch._check(
    #         qmap2.dtype == absmax2.dtype == torch.float32,
    #         lambda: f"Expected qmap2 and absmax2 to be float32, got qmap2.dtype={qmap2.dtype}, absmax2.dtype={absmax2.dtype}",
    #     )
    optimizer_fns = str2optimizer8bit_blockwise.get(optimizer_name)
    if optimizer_fns is None:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {list(str2optimizer8bit_blockwise.keys())}"
        )

    if g.dtype == torch.float32:
        optimizer_fn = optimizer_fns[0]
    elif g.dtype == torch.float16:
        optimizer_fn = optimizer_fns[1]
    elif g.dtype == torch.bfloat16:
        optimizer_fn = optimizer_fns[2]
    else:
        raise ValueError(
            f"Unsupported gradient dtype: {g.dtype}. Supported dtypes: torch.float32, torch.float16, torch.bfloat16"
        )

    with _cuda_device_of(g):
        optimizer_fn(
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(beta3),
            ct.c_float(alpha),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(absmax1),
            get_ptr(absmax2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )


register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "cuda")(_optimizer_update_8bit_blockwise_impl)
register_kernel("bitsandbytes::optimizer_update_32bit", "cuda")(_optimizer_update_32bit_impl)


# NVFP4 dequantization
@register_kernel("bitsandbytes::dequantize_nvfp4", "cuda")
def _(
    packed: torch.Tensor, block_scales: torch.Tensor, tensor_scale: float, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    packed = packed.contiguous()
    block_scales = block_scales.contiguous()
    output = torch.zeros(numel, dtype=dtype, device=packed.device)

    with _cuda_device_of(packed):
        if dtype == torch.float16:
            lib.cdequantize_nvfp4_fp16(
                get_ptr(packed),
                get_ptr(block_scales),
                ct.c_float(tensor_scale),
                get_ptr(output),
                ct.c_int(numel),
                ct.c_void_p(0),
            )
        elif dtype == torch.bfloat16:
            lib.cdequantize_nvfp4_bf16(
                get_ptr(packed),
                get_ptr(block_scales),
                ct.c_float(tensor_scale),
                get_ptr(output),
                ct.c_int(numel),
                ct.c_void_p(0),
            )
        else:
            lib.cdequantize_nvfp4_fp32(
                get_ptr(packed),
                get_ptr(block_scales),
                ct.c_float(tensor_scale),
                get_ptr(output),
                ct.c_int(numel),
                ct.c_void_p(0),
            )

    return output




# CUTLASS-based fused quantize for NVFP4 (SM_120+)
# Uses QuTLASS GEMM-as-quantize approach with always-on randomized Hadamard
# rotation. The 16x16 rotation matrix is generated once per device and cached.
_rotation_matrices: dict[torch.device, torch.Tensor] = {}

# Fixed seed for reproducible rotation across weight quantization and inference.
_ROTATION_SEED = 42


def _get_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Get cached 16x16 randomized Hadamard matrix for fused quantize.

    Builds R = H * D where H is the 16x16 normalized Hadamard matrix and D is
    a diagonal sign-flip matrix (±1 per column) from a fixed seed. The CUTLASS
    GEMM computes ``A @ R`` (no transpose), so dequant must apply ``@ R^T``.
    The same matrix must be used for both weight and activation quantization.
    """
    if device not in _rotation_matrices:
        # Build normalized 16x16 Hadamard via Sylvester construction
        h = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(4):  # 2^4 = 16
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        h /= 4.0  # normalize by 1/sqrt(16)

        # Apply random sign flips per column (H @ D)
        gen = torch.Generator().manual_seed(_ROTATION_SEED)
        signs = torch.randint(0, 2, (16,), generator=gen) * 2 - 1  # ±1
        h = h * signs.float()

        _rotation_matrices[device] = h.to(dtype=torch.bfloat16, device=device)
    return _rotation_matrices[device]


def _fused_quantize_nvfp4_raw(
    A_flat: torch.Tensor,
    rotation: torch.Tensor,
    packed_out: torch.Tensor,
    scales_out: torch.Tensor,
    global_scale: torch.Tensor,
    M: int,
) -> None:
    """Raw CUTLASS fused quantize — zero allocations, CUDA-graph-safe.

    All buffers must be pre-allocated and pre-filled by the caller. Input A
    must already be padded so that M is a multiple of 128. The global_scale
    buffer must contain ``1.0 / tensor_scale``.

    This is the innermost call used by both the convenience wrapper and
    CUDA graph capture paths.
    """
    lib.cfused_quantize_nvfp4_absmax(
        get_ptr(A_flat),
        get_ptr(rotation),
        get_ptr(packed_out),
        get_ptr(scales_out),
        get_ptr(global_scale),
        ct.c_int(M),
        ct.c_int(16),
        ct.c_int(16),
        _get_tensor_stream(A_flat),
    )


def _fused_quantize_nvfp4_impl(
    A: torch.Tensor,
    tensor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience wrapper that allocates outputs. Not graph-safe."""
    A = A.contiguous()
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    torch._check(
        A.dtype == torch.bfloat16,
        lambda: f"CUTLASS fused quantize requires bfloat16, got {A.dtype}",
    )

    K = 16
    orig_M = n // K
    padded_M = ((orig_M + 127) // 128) * 128

    if padded_M != orig_M:
        A_2d = A.view(orig_M, K)
        A_2d = torch.nn.functional.pad(A_2d, (0, 0, 0, padded_M - orig_M))
        A_flat = A_2d.reshape(-1)
    else:
        A_flat = A

    global_scale = torch.tensor(
        [1.0 / tensor_scale if tensor_scale > 0 else 0.0],
        dtype=torch.float32,
        device=A.device,
    )
    packed_padded = torch.zeros(padded_M * K // 2, dtype=torch.uint8, device=A.device)
    scales_padded = torch.zeros(padded_M, dtype=torch.uint8, device=A.device)

    _fused_quantize_nvfp4_raw(
        A_flat,
        _get_rotation_matrix(A.device),
        packed_padded,
        scales_padded,
        global_scale,
        padded_M,
    )

    packed = packed_padded[: orig_M * K // 2] if padded_M != orig_M else packed_padded
    block_scales = scales_padded[:orig_M] if padded_M != orig_M else scales_padded

    ts_out = torch.tensor([tensor_scale], dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


@register_kernel("bitsandbytes::cutlass_fused_quantize_nvfp4", "cuda")
def _(
    A: torch.Tensor,
    tensor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUTLASS-based fused quantize with randomized Hadamard rotation."""
    return _fused_quantize_nvfp4_impl(A, tensor_scale)


@register_kernel("bitsandbytes::cutlass_fused_quantize_nvfp4_raw", "cuda")
def _(
    A: torch.Tensor,
    global_scale_dev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side quantize: global_scale is a pre-computed device tensor.

    Returns (packed_data, block_scales_rowmajor) — no swizzling, no QuantState.
    The global_scale_dev tensor should contain 1.0/tensor_scale as a float32
    scalar on the GPU (0-dim or 1-element tensor).
    """
    A = A.contiguous()
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    torch._check(
        A.dtype == torch.bfloat16,
        lambda: f"CUTLASS fused quantize requires bfloat16, got {A.dtype}",
    )

    K = 16
    orig_M = n // K
    padded_M = ((orig_M + 127) // 128) * 128

    if padded_M != orig_M:
        A_2d = A.view(orig_M, K)
        A_2d = torch.nn.functional.pad(A_2d, (0, 0, 0, padded_M - orig_M))
        A_flat = A_2d.reshape(-1)
    else:
        A_flat = A

    packed_padded = torch.zeros(padded_M * K // 2, dtype=torch.uint8, device=A.device)
    scales_padded = torch.zeros(padded_M, dtype=torch.uint8, device=A.device)

    _fused_quantize_nvfp4_raw(
        A_flat,
        _get_rotation_matrix(A.device),
        packed_padded,
        scales_padded,
        global_scale_dev.to(dtype=torch.float32).contiguous(),
        padded_M,
    )

    packed = packed_padded[: orig_M * K // 2] if padded_M != orig_M else packed_padded
    block_scales = scales_padded[:orig_M] if padded_M != orig_M else scales_padded

    return packed, block_scales


# Scale reordering for CUTLASS block-scaled GEMM
@register_kernel("bitsandbytes::scale_to_blocked", "cuda")
def _(scales: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reorder flat row-major scales to CUTLASS block-scaled layout.

    Called once at quantization time to pre-compute the swizzled scales
    that CUTLASS needs. The result is stored in NVFP4QuantState.
    """
    n_row_blocks = (H + 127) // 128
    n_col_blocks = (W + 3) // 4
    out_size = n_row_blocks * n_col_blocks * 128 * 4
    out = torch.empty(out_size, dtype=torch.uint8, device=scales.device)
    with _cuda_device_of(scales):
        lib.cscale_to_blocked(
            get_ptr(scales),
            get_ptr(out),
            ct.c_int(H),
            ct.c_int(W),
            _get_tensor_stream(scales),
        )
    return out


@register_kernel("bitsandbytes::scale_from_blocked", "cuda")
def _(blocked_scales: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reverse CUTLASS block-scaled layout back to flat row-major scales."""
    out = torch.empty(H * W, dtype=torch.uint8, device=blocked_scales.device)
    with _cuda_device_of(blocked_scales):
        lib.cscale_from_blocked(
            get_ptr(blocked_scales),
            get_ptr(out),
            ct.c_int(H),
            ct.c_int(W),
            _get_tensor_stream(blocked_scales),
        )
    return out


@register_kernel("bitsandbytes::scale_to_blocked_batched", "cuda")
def _(
    scales_rowmajor: torch.Tensor,
    expert_row_offsets: torch.Tensor,
    expert_M: torch.Tensor,
    expert_out_offsets: torch.Tensor,
    W: int,
    num_experts: int,
    max_row_blocks: int,
    total_out_bytes: int,
) -> torch.Tensor:
    """Batched scale swizzle: row-major → per-expert CUTLASS block-scaled layout.

    Input: concatenated row-major scales from quantize_nvfp4_raw.
    Output: contiguous buffer with independently swizzled per-expert blocks.
    """
    out = torch.zeros(total_out_bytes, dtype=torch.uint8, device=scales_rowmajor.device)
    with _cuda_device_of(scales_rowmajor):
        lib.cscale_to_blocked_batched(
            get_ptr(scales_rowmajor),
            get_ptr(out),
            get_ptr(expert_row_offsets),
            get_ptr(expert_M),
            get_ptr(expert_out_offsets),
            ct.c_int(W),
            ct.c_int(num_experts),
            ct.c_int(max_row_blocks),
            _get_tensor_stream(scales_rowmajor),
        )
    return out


@register_kernel("bitsandbytes::moe_scatter_nvfp4", "cuda")
def _(
    packed_concat: torch.Tensor,
    expert_offsets: torch.Tensor,
    max_M: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    """Scatter concatenated FP4 data to padded per-expert batched layout."""
    row_bytes = K // 2
    out = torch.empty(
        num_experts * max_M * row_bytes, dtype=torch.uint8, device=packed_concat.device,
    )
    with _cuda_device_of(packed_concat):
        lib.cmoe_scatter_nvfp4(
            get_ptr(packed_concat),
            get_ptr(out),
            get_ptr(expert_offsets),
            ct.c_int(max_M),
            ct.c_int(K),
            ct.c_int(num_experts),
            _get_tensor_stream(packed_concat),
        )
    return out


@register_kernel("bitsandbytes::moe_gather_bf16", "cuda")
def _(
    D_batched: torch.Tensor,
    expert_offsets: torch.Tensor,
    max_M: int,
    N: int,
    num_experts: int,
    total_tokens: int,
) -> torch.Tensor:
    """Gather BF16 results from padded per-expert layout to concatenated output."""
    out = torch.empty(
        total_tokens * N, dtype=torch.bfloat16, device=D_batched.device,
    )
    with _cuda_device_of(D_batched):
        lib.cmoe_gather_bf16(
            get_ptr(D_batched),
            get_ptr(out),
            get_ptr(expert_offsets),
            ct.c_int(max_M),
            ct.c_int(N),
            ct.c_int(num_experts),
            _get_tensor_stream(D_batched),
        )
    return out


# Hand-written NVFP4 GEMM (SM_120+)
#
# Uses mma.sync.aligned.block_scale instructions for small-M decode.
# Expects flat (non-swizzled) row-major scales.
# Uses automatic split-K when tile count is low relative to SM count.
# BF16 output variant needs FP32 workspace for split-K accumulation.
def _gemm_nvfp4_hw_bf16_raw(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    D_out: torch.Tensor,
    workspace: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> None:
    """Raw hand-written NVFP4 GEMM (BF16 output) — zero allocations, CUDA-graph-safe.

    All buffers must be pre-allocated. D_out must be BF16 of shape (M, N).
    workspace must be FP32 of shape (M, N) — used for split-K accumulation.
    Scales are flat row-major (not swizzled).
    """
    lib.cgemm_nvfp4_bf16(
        get_ptr(A_packed),
        get_ptr(B_packed),
        get_ptr(A_scales),
        get_ptr(B_scales),
        get_ptr(D_out),
        get_ptr(workspace),
        ct.c_int(M),
        ct.c_int(N),
        ct.c_int(K),
        _get_tensor_stream(A_packed),
    )


# NVFP4 GEMM (CUTLASS-based)
#
# Expects pre-swizzled scales in CUTLASS block-scaled layout (computed at
# quantization time by scale_to_blocked). Tensor scales are folded into
# the CUTLASS epilogue alpha. Output is BF16, converted to FP32 for
# API compatibility.
def _gemm_nvfp4_raw(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    D_out: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: torch.Tensor,
) -> None:
    """Raw NVFP4 GEMM — zero allocations, CUDA-graph-safe.

    All buffers must be pre-allocated by the caller. The alpha buffer must
    contain ``A_tensor_scale * B_tensor_scale``.

    Dispatches to SM_100 (B200/B100) or SM_120 (RTX 5090) kernel based on
    the current GPU's compute capability.
    """
    major, _ = torch.cuda.get_device_capability(A_packed.device)
    if major == 10 and hasattr(lib, "cgemm_nvfp4_cutlass_sm100"):
        lib.cgemm_nvfp4_cutlass_sm100(
            get_ptr(A_packed),
            get_ptr(B_packed),
            get_ptr(A_scales),
            get_ptr(B_scales),
            get_ptr(D_out),
            ct.c_int(M),
            ct.c_int(N),
            ct.c_int(K),
            get_ptr(alpha),
            _get_tensor_stream(A_packed),
        )
    else:
        lib.cgemm_nvfp4_cutlass(
            get_ptr(A_packed),
            get_ptr(B_packed),
            get_ptr(A_scales),
            get_ptr(B_scales),
            get_ptr(D_out),
            ct.c_int(M),
            ct.c_int(N),
            ct.c_int(K),
            get_ptr(alpha),
            _get_tensor_stream(A_packed),
        )


def _gemm_nvfp4_impl(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Convenience wrapper that allocates outputs. Not graph-safe."""
    with _cuda_device_of(A_packed):
        alpha = torch.tensor([A_tensor_scale * B_tensor_scale], dtype=torch.float32, device=A_packed.device)
        D_out = torch.empty(M, N, dtype=torch.bfloat16, device=A_packed.device)
        _gemm_nvfp4_raw(A_packed, B_packed, A_scales, B_scales, D_out, M, N, K, alpha)
    return D_out.float()


@register_kernel("bitsandbytes::gemm_nvfp4", "cuda")
def _(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """NVFP4 GEMM: A @ B^T with block-scaled FP4 inputs."""
    return _gemm_nvfp4_impl(
        A_packed,
        B_packed,
        A_scales,
        B_scales,
        A_tensor_scale,
        B_tensor_scale,
        M,
        N,
        K,
    )


# Grouped NVFP4 GEMM for MoE inference (SM_120+)
#
# Fuses all expert GEMMs into a single kernel launch using expert-offset
# work decomposition with binary search. Uses swizzled (block-scaled) scales.
# CUDA-graph-safe: no dynamic allocations.
def _gemm_nvfp4_grouped_raw(
    A_concat: torch.Tensor,
    B_all: torch.Tensor,
    SFA_concat: torch.Tensor,
    SFB_all: torch.Tensor,
    D_concat: torch.Tensor,
    expert_offsets: torch.Tensor,
    cumul_m_tiles: torch.Tensor,
    N: int,
    K: int,
    num_experts: int,
    total_tiles: int,
) -> None:
    """Raw grouped NVFP4 GEMM (BF16 output) — zero allocations, CUDA-graph-safe.

    All buffers must be pre-allocated. D_concat must be BF16 of shape (total_tokens, N).
    expert_offsets and cumul_m_tiles must be int32 on the same device.
    """
    lib.cgemm_nvfp4_grouped_bf16(
        get_ptr(A_concat),
        get_ptr(B_all),
        get_ptr(SFA_concat),
        get_ptr(SFB_all),
        get_ptr(D_concat),
        get_ptr(expert_offsets),
        get_ptr(cumul_m_tiles),
        ct.c_int(N),
        ct.c_int(K),
        ct.c_int(num_experts),
        ct.c_int(total_tiles),
        _get_tensor_stream(A_concat),
    )


@register_kernel("bitsandbytes::gemm_nvfp4_grouped", "cuda")
def _(
    A_concat: torch.Tensor,
    B_all: torch.Tensor,
    SFA_rowmajor: torch.Tensor,
    SFB_all: torch.Tensor,
    expert_offsets: torch.Tensor,
    cumul_m_tiles: torch.Tensor,
    A_tensor_scale: float,
    B_tensor_scale: float,
    N: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    """Grouped NVFP4 GEMM for MoE: fuse all expert GEMMs into one launch.

    SFA_rowmajor: row-major activation scales (NOT swizzled).
    SFB_all: per-expert swizzled weight scales (each expert independently swizzled
             by quantize_nvfp4, then concatenated).
    """
    # SM_120 (consumer Blackwell): use hand-written grouped kernel
    # SM_120 expects globally-swizzled SFA, so swizzle the row-major input
    total_tokens = A_concat.numel() // (K // 2)
    scale_W = K // 16
    SFA_blocked = torch.ops.bitsandbytes.scale_to_blocked(SFA_rowmajor, total_tokens, scale_W)

    num_n_tiles = (N + 127) // 128

    with _cuda_device_of(A_concat):
        D_concat = torch.empty(total_tokens, N, dtype=torch.bfloat16, device=A_concat.device)
        total_tiles = cumul_m_tiles[-1].item() * num_n_tiles

        _gemm_nvfp4_grouped_raw(
            A_concat, B_all, SFA_blocked, SFB_all, D_concat,
            expert_offsets, cumul_m_tiles, N, K, num_experts, total_tiles,
        )

    # Apply tensor scales (SM_120 kernel has no alpha epilogue)
    D_concat *= A_tensor_scale * B_tensor_scale
    return D_concat


# =========================================================================
# Batched NVFP4 GEMM for MoE inference (SM_100 datacenter Blackwell)
# =========================================================================

# Cached state for batched SM_100 MoE GEMM
_moe_batched_restype_set = False
_moe_batched_sm100_cache: Optional[dict] = None


def _ensure_moe_batched_restype():
    global _moe_batched_restype_set
    if not _moe_batched_restype_set:
        lib.cgemm_nvfp4_moe_sm100_sfa_size.restype = ct.c_size_t
        lib.cgemm_nvfp4_moe_sm100_sfb_size.restype = ct.c_size_t
        lib.cgemm_nvfp4_moe_sm100_sfa_size_per_expert.restype = ct.c_size_t
        lib.cgemm_nvfp4_moe_sm100_sfb_size_per_expert.restype = ct.c_size_t
        lib.cgemm_nvfp4_moe_sm100_workspace_size.restype = ct.c_size_t
        lib.cgemm_nvfp4_moe_sm100_init.restype = ct.c_int
        lib.cgemm_nvfp4_moe_sm100_run.restype = ct.c_int
        _moe_batched_restype_set = True


def _batched_moe_sm100_init_if_needed(
    A_batched: torch.Tensor,
    B_all: torch.Tensor,
    SFA_batched: torch.Tensor,
    SFB_all: torch.Tensor,
    D_out: torch.Tensor,
    alpha: torch.Tensor,
    max_M: int,
    N: int,
    K: int,
    num_experts: int,
    stream: int,
) -> None:
    """Call cgemm_nvfp4_moe_sm100_init if the configuration changed, else skip."""
    global _moe_batched_sm100_cache
    _ensure_moe_batched_restype()

    cache_key = (
        N, K, max_M, num_experts,
        A_batched.data_ptr(), B_all.data_ptr(),
        SFA_batched.data_ptr(), SFB_all.data_ptr(),
        D_out.data_ptr(), alpha.data_ptr(),
    )

    if (_moe_batched_sm100_cache is not None
            and _moe_batched_sm100_cache["key"] == cache_key):
        return

    ws_size = lib.cgemm_nvfp4_moe_sm100_workspace_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
    )
    workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device=A_batched.device)

    ret = lib.cgemm_nvfp4_moe_sm100_init(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
        get_ptr(A_batched), get_ptr(B_all),
        get_ptr(SFA_batched), get_ptr(SFB_all),
        get_ptr(D_out), get_ptr(alpha),
        get_ptr(workspace), ct.c_size_t(ws_size), stream,
    )
    if ret != 0:
        raise RuntimeError(f"cgemm_nvfp4_moe_sm100_init failed with code {ret}")

    _moe_batched_sm100_cache = {
        "key": cache_key,
        "workspace": workspace,  # prevent GC
    }


def _gemm_nvfp4_batched_moe_sm100_raw(
    A_batched: torch.Tensor,
    B_all: torch.Tensor,
    SFA_batched: torch.Tensor,
    SFB_all: torch.Tensor,
    D_out: torch.Tensor,
    alpha: torch.Tensor,
    max_M: int,
    N: int,
    K: int,
    num_experts: int,
) -> None:
    """Raw batched MoE NVFP4 GEMM — init-if-needed then run.

    All buffers must be pre-allocated. D_out must be BF16 of shape (num_experts * max_M, N).
    alpha must be a float32 device tensor of shape (1,) containing A_scale * B_scale.
    """
    stream = _get_tensor_stream(A_batched)
    _batched_moe_sm100_init_if_needed(
        A_batched, B_all, SFA_batched, SFB_all, D_out, alpha,
        max_M, N, K, num_experts, stream,
    )
    ret = lib.cgemm_nvfp4_moe_sm100_run(stream)
    if ret != 0:
        raise RuntimeError(f"cgemm_nvfp4_moe_sm100_run failed with code {ret}")


@register_kernel("bitsandbytes::gemm_nvfp4_moe", "cuda")
def _(
    A_batched: torch.Tensor,
    B_batched: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    alpha: torch.Tensor,
    max_M: int,
    N: int,
    K: int,
    num_experts: int,
) -> torch.Tensor:
    with _cuda_device_of(A_batched):
        D_out = torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=A_batched.device)
        _gemm_nvfp4_batched_moe_sm100_raw(
            A_batched, B_batched, SFA, SFB, D_out, alpha,
            max_M, N, K, num_experts,
        )
    return D_out.view(num_experts, max_M, N)


@register_kernel("bitsandbytes::moe_weighted_gather_bf16", "cuda")
def _(
    D_batched: torch.Tensor,
    output_bf16: torch.Tensor,
    workspace_fp32: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    max_M: int,
    N: int,
) -> torch.Tensor:
    """Fused gather + weight + FP32 accumulate + BF16 convert.

    Internally launches: memset(workspace) -> atomicAdd gather -> FP32->BF16 convert.
    All three operations on the same stream, capturable in a CUDA graph.
    """
    total_assignments = token_ids.shape[0]
    with _cuda_device_of(D_batched):
        lib.cmoe_weighted_gather_bf16(
            get_ptr(D_batched),
            get_ptr(output_bf16),
            get_ptr(workspace_fp32),
            get_ptr(token_ids),
            get_ptr(expert_ids),
            get_ptr(slot_ids),
            get_ptr(weights),
            ct.c_int(total_assignments),
            ct.c_int(num_tokens),
            ct.c_int(max_M),
            ct.c_int(N),
            _get_tensor_stream(D_batched),
        )
    return output_bf16


# K-bit blockwise quantization (K=2..5, blocksize=32)

_KBIT_DTYPE_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@register_kernel("bitsandbytes::quantize_kbit", "cuda")
def _(A: torch.Tensor, codebook: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in _KBIT_DTYPE_SUFFIX,
        lambda: f"quantize_kbit only supports float16/bfloat16/float32, got {A.dtype}",
    )
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(codebook.numel() == (1 << k), lambda: f"codebook must have {1 << k} entries for k={k}")

    n = A.numel()
    num_blocks = -(n // -32)
    packed = torch.zeros(num_blocks * k + k, device=A.device, dtype=torch.int32)
    absmax = torch.zeros(num_blocks + 1, device=A.device, dtype=torch.uint8)

    with _cuda_device_of(A):
        tname = _KBIT_DTYPE_SUFFIX[A.dtype]
        fn = getattr(lib, f"cquantize_kbit_{tname}_k{k}")
        fn(
            get_ptr(codebook),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(packed),
            ct.c_int(n),
            _get_tensor_stream(A),
        )

    return packed, absmax


_KBIT_ABSMAX_SUFFIX = {
    torch.uint8: "u8abs",
    torch.float16: "fp16abs",
    torch.float32: "fp32abs",
}


def _dequantize_kbit_impl(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        dtype in _KBIT_DTYPE_SUFFIX,
        lambda: f"dequantize_kbit only supports float16/bfloat16/float32, got {dtype}",
    )
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(
        absmax.dtype in (torch.float32, torch.float16, torch.uint8),
        lambda: f"absmax must be float32, float16, or uint8 (E4M4), got {absmax.dtype}",
    )

    # If fp32 absmax, encode to E4M4 first
    if absmax.dtype == torch.float32:
        from bitsandbytes.functional import encode_absmax_e4m4

        absmax = encode_absmax_e4m4(absmax)

    tname = _KBIT_DTYPE_SUFFIX[dtype]
    aname = _KBIT_ABSMAX_SUFFIX[absmax.dtype]

    with _cuda_device_of(packed):
        fn = getattr(lib, f"cdequantize_kbit_{tname}_{aname}_k{k}")
        fn(
            get_ptr(packed),
            get_ptr(codebook),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(n),
            _get_tensor_stream(packed),
        )


@register_kernel("bitsandbytes::dequantize_kbit", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_blocks = -(n // -32)
    out = torch.empty(num_blocks * 32, device=packed.device, dtype=dtype)
    _dequantize_kbit_impl(packed, codebook, absmax, k, n, dtype, out)
    return out


@register_kernel("bitsandbytes::dequantize_kbit_", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    _dequantize_kbit_impl(packed, codebook, absmax, k, n, dtype, out)
    return out


def _dequantize_kbit_tiled_impl(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        dtype in _KBIT_DTYPE_SUFFIX,
        lambda: f"dequantize_kbit_tiled only supports float16/bfloat16/float32, got {dtype}",
    )
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(
        absmax.dtype in (torch.float32, torch.float16, torch.uint8),
        lambda: f"absmax must be float32, float16, or uint8 (E4M4), got {absmax.dtype}",
    )

    if absmax.dtype == torch.float32:
        from bitsandbytes.functional import encode_absmax_e4m4

        absmax = encode_absmax_e4m4(absmax)

    tname = _KBIT_DTYPE_SUFFIX[dtype]
    aname = _KBIT_ABSMAX_SUFFIX[absmax.dtype]

    with _cuda_device_of(packed):
        fn = getattr(lib, f"cdequantize_kbit_tiled_{tname}_{aname}_k{k}")
        fn(
            get_ptr(packed),
            get_ptr(codebook),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(packed),
        )


@register_kernel("bitsandbytes::dequantize_kbit_tiled", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = N * K_dim
    num_blocks = -(n // -32)
    out = torch.empty(num_blocks * 32, device=packed.device, dtype=dtype)
    _dequantize_kbit_tiled_impl(packed, codebook, absmax, k, K_dim, N, dtype, out)
    return out


@register_kernel("bitsandbytes::dequantize_kbit_tiled_", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    _dequantize_kbit_tiled_impl(packed, codebook, absmax, k, K_dim, N, dtype, out)
    return out


_VQ_DTYPE_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@register_kernel("bitsandbytes::quantize_vq", "cuda")
def _(A: torch.Tensor, codebook: torch.Tensor, p: int, index_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    from bitsandbytes._ops import _vq_traits

    torch._check(
        A.dtype in _VQ_DTYPE_SUFFIX,
        lambda: f"quantize_vq only supports float16/bfloat16/float32, got {A.dtype}",
    )
    torch._check(codebook.dtype == torch.float16, lambda: f"codebook must be float16, got {codebook.dtype}")

    traits = _vq_traits(p, index_bits)
    n = A.numel()
    num_blocks = -(n // -traits["BS"])
    packed = torch.zeros(num_blocks * traits["WORDS"], device=A.device, dtype=torch.int32)
    absmax = torch.zeros(num_blocks, device=A.device, dtype=torch.uint8)

    with _cuda_device_of(A):
        tname = _VQ_DTYPE_SUFFIX[A.dtype]
        fn = getattr(lib, f"cquantize_vq_{tname}_p{p}b{index_bits}")
        fn(
            get_ptr(codebook),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(packed),
            ct.c_int(n),
            _get_tensor_stream(A),
        )

    return packed, absmax


def _dequantize_vq_impl(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    p: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> None:
    torch._check(
        dtype in _VQ_DTYPE_SUFFIX,
        lambda: f"dequantize_vq only supports float16/bfloat16/float32, got {dtype}",
    )
    torch._check(codebook.dtype == torch.float16, lambda: f"codebook must be float16, got {codebook.dtype}")

    # If fp32 absmax, encode to E4M4 first
    if absmax.dtype == torch.float32:
        from bitsandbytes.functional import encode_absmax_e4m4

        absmax = encode_absmax_e4m4(absmax)

    tname = _VQ_DTYPE_SUFFIX[dtype]
    aname = _KBIT_ABSMAX_SUFFIX[absmax.dtype]

    with _cuda_device_of(packed):
        fn = getattr(lib, f"cdequantize_vq_{tname}_{aname}_p{p}b{index_bits}")
        fn(
            get_ptr(packed),
            get_ptr(codebook),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(n),
            _get_tensor_stream(packed),
        )


@register_kernel("bitsandbytes::dequantize_vq", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    p: int,
    n: int,
    dtype: torch.dtype,
    index_bits: int = 8,
) -> torch.Tensor:
    BS = 48 if p == 3 else 32
    num_blocks = -(n // -BS)
    out = torch.empty(num_blocks * BS, device=packed.device, dtype=dtype)
    _dequantize_vq_impl(packed, codebook, absmax, p, n, dtype, out, index_bits)
    return out


@register_kernel("bitsandbytes::dequantize_vq_", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    p: int,
    n: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    _dequantize_vq_impl(packed, codebook, absmax, p, n, dtype, out, index_bits)
    return out


def _dequantize_vq_tiled_impl(
    packed_tiled: torch.Tensor,
    codebook: torch.Tensor,
    absmax_tiled: torch.Tensor,
    p: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> None:
    torch._check(codebook.dtype == torch.float16, lambda: f"codebook must be float16, got {codebook.dtype}")

    if dtype in (torch.float16,):
        tname = "fp16"
    elif dtype == torch.bfloat16:
        tname = "bf16"
    else:
        raise ValueError(f"dequantize_vq_tiled only supports float16/bfloat16, got {dtype}")

    if absmax_tiled.dtype == torch.uint8:
        aname = "u8abs"
    elif absmax_tiled.dtype == torch.float32:
        aname = "fp32abs"
    else:
        raise ValueError(f"absmax must be uint8 or float32, got {absmax_tiled.dtype}")

    with _cuda_device_of(packed_tiled):
        fn = getattr(lib, f"cdequantize_vq_tiled_{tname}_{aname}_p{p}b{index_bits}")
        fn(
            get_ptr(packed_tiled),
            get_ptr(codebook),
            get_ptr(absmax_tiled),
            get_ptr(out),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(packed_tiled),
        )


@register_kernel("bitsandbytes::dequantize_vq_tiled", "cuda")
def _(
    packed_tiled: torch.Tensor,
    codebook: torch.Tensor,
    absmax_tiled: torch.Tensor,
    p: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    index_bits: int = 8,
) -> torch.Tensor:
    out = torch.empty(N * K_dim, device=packed_tiled.device, dtype=dtype)
    _dequantize_vq_tiled_impl(packed_tiled, codebook, absmax_tiled, p, K_dim, N, dtype, out, index_bits)
    return out


@register_kernel("bitsandbytes::dequantize_vq_tiled_", "cuda")
def _(
    packed_tiled: torch.Tensor,
    codebook: torch.Tensor,
    absmax_tiled: torch.Tensor,
    p: int,
    K_dim: int,
    N: int,
    dtype: torch.dtype,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    _dequantize_vq_tiled_impl(packed_tiled, codebook, absmax_tiled, p, K_dim, N, dtype, out, index_bits)
    return out


def _vq_scalar_gemv_impl(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    out: torch.Tensor,
    tiled: bool = False,
    index_bits: int = 8,
) -> None:
    M = A.shape[0]
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    tiled_str = "_tiled" if tiled else ""

    with _cuda_device_of(A):
        fn = getattr(lib, f"cvq_scalar_gemv{tiled_str}_{dtype_suffix}_p{p}b{index_bits}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(out),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(A),
        )


@register_kernel("bitsandbytes::vq_scalar_gemv", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_scalar_gemv supports float16 and bfloat16, got {A.dtype}",
    )
    M = A.shape[0]
    out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    _vq_scalar_gemv_impl(A, B_packed, B_absmax, codebook, K_dim, N, p, out=out, index_bits=index_bits)
    return out


@register_kernel("bitsandbytes::vq_scalar_gemv.out", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> None:
    _vq_scalar_gemv_impl(A, B_packed, B_absmax, codebook, K_dim, N, p, out=out, index_bits=index_bits)


@register_kernel("bitsandbytes::vq_scalar_gemv_tiled", "cuda")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_scalar_gemv_tiled supports float16 and bfloat16, got {A.dtype}",
    )
    M = A.shape[0]
    out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    _vq_scalar_gemv_impl(A, B_packed_tiled, B_absmax_tiled, codebook, K_dim, N, p, out=out, tiled=True, index_bits=index_bits)
    return out


@register_kernel("bitsandbytes::vq_scalar_gemv_tiled_", "cuda")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_scalar_gemv_tiled_ supports float16 and bfloat16, got {A.dtype}",
    )
    M = A.shape[0]
    _vq_scalar_gemv_impl(A, B_packed_tiled, B_absmax_tiled, codebook, K_dim, N, p, out=out, tiled=True, index_bits=index_bits)
    return out


@register_kernel("bitsandbytes::repack_kbit", "cuda")
def _(
    packed_flat: torch.Tensor,
    absmax_flat: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(packed_flat.dtype == torch.int32, lambda: f"packed_flat must be int32, got {packed_flat.dtype}")
    torch._check(
        absmax_flat.dtype == torch.uint8, lambda: f"absmax_flat must be uint8 (E4M4), got {absmax_flat.dtype}"
    )

    TILE_K, TILE_N, BLOCKSIZE = 64, 128, 32
    torch._check(N % TILE_N == 0, lambda: f"N ({N}) must be divisible by {TILE_N}")
    torch._check(K_dim % BLOCKSIZE == 0, lambda: f"K_dim ({K_dim}) must be divisible by {BLOCKSIZE}")

    K_dim_padded = ((K_dim + TILE_K - 1) // TILE_K) * TILE_K
    k_tiles = K_dim_padded // TILE_K
    n_tiles = N // TILE_N
    k_blocks_per_tile = TILE_K // BLOCKSIZE
    total_words = k_tiles * n_tiles * TILE_N * k_blocks_per_tile * k
    total_absmax = k_tiles * n_tiles * TILE_N * k_blocks_per_tile

    # Zero-fill for padding regions (when K_dim is not multiple of TILE_K)
    packed_tiled = torch.zeros(total_words, device=packed_flat.device, dtype=torch.int32)
    absmax_tiled = torch.zeros(total_absmax, device=packed_flat.device, dtype=torch.uint8)

    with _cuda_device_of(packed_flat):
        fn = getattr(lib, f"crepack_kbit_k{k}")
        fn(
            get_ptr(packed_flat),
            get_ptr(absmax_flat),
            get_ptr(packed_tiled),
            get_ptr(absmax_tiled),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(packed_flat),
        )

    return packed_tiled, absmax_tiled


@register_kernel("bitsandbytes::repack_vq", "cuda")
def _(
    packed_flat: torch.Tensor,
    absmax_flat: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    index_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    from bitsandbytes._ops import _vq_traits

    torch._check(packed_flat.dtype == torch.int32, lambda: f"packed_flat must be int32, got {packed_flat.dtype}")
    torch._check(
        absmax_flat.dtype == torch.uint8, lambda: f"absmax_flat must be uint8 (E4M4), got {absmax_flat.dtype}"
    )

    traits = _vq_traits(p, index_bits)
    BS = traits["BS"]
    TILE_K = traits["TILE_K"]
    TILE_N = traits["TILE_N"]
    WORDS = traits["WORDS"]
    KB_PER_TILE = traits["KB_PER_TILE"]
    torch._check(N % TILE_N == 0, lambda: f"N ({N}) must be divisible by {TILE_N}")
    torch._check(K_dim % BS == 0, lambda: f"K_dim ({K_dim}) must be divisible by {BS}")

    K_dim_padded = ((K_dim + TILE_K - 1) // TILE_K) * TILE_K
    k_tiles = K_dim_padded // TILE_K
    n_tiles = N // TILE_N
    total_words = k_tiles * n_tiles * TILE_N * KB_PER_TILE * WORDS
    total_absmax = k_tiles * n_tiles * TILE_N * KB_PER_TILE

    # Zero-fill for padding regions (when K_dim is not multiple of TILE_K)
    packed_tiled = torch.zeros(total_words, device=packed_flat.device, dtype=torch.int32)
    absmax_tiled = torch.zeros(total_absmax, device=packed_flat.device, dtype=torch.uint8)

    with _cuda_device_of(packed_flat):
        fn = getattr(lib, f"crepack_vq_p{p}b{index_bits}")
        fn(
            get_ptr(packed_flat),
            get_ptr(absmax_flat),
            get_ptr(packed_tiled),
            get_ptr(absmax_tiled),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(packed_flat),
        )

    return packed_tiled, absmax_tiled


@register_kernel("bitsandbytes::hadamard_rotate_", "cuda")
def _(data: torch.Tensor, block_size: int, signs: Optional[torch.Tensor]) -> torch.Tensor:
    torch._check(
        block_size in (32, 64, 128, 256),
        lambda: f"block_size must be 32, 64, 128, or 256, got {block_size}",
    )
    torch._check(
        data.dtype in (torch.float16, torch.bfloat16),
        lambda: f"hadamard_rotate only supports float16/bfloat16, got {data.dtype}",
    )

    tname = _KBIT_DTYPE_SUFFIX[data.dtype]
    signs_ptr = get_ptr(signs) if signs is not None else None
    with _cuda_device_of(data):
        fn = getattr(lib, f"chadamard_rotate_{tname}")
        fn(
            get_ptr(data),
            ct.c_int(data.numel()),
            ct.c_int(block_size),
            signs_ptr,
            _get_tensor_stream(data),
        )

    return data


@register_kernel("bitsandbytes::hadamard_rotate_full_", "cuda")
def _(data: torch.Tensor, dim: int, signs: Optional[torch.Tensor]) -> torch.Tensor:
    supported_dims = (512, 1024, 2048, 4096, 8192)
    torch._check(
        dim in supported_dims,
        lambda: f"dim must be one of {supported_dims}, got {dim}",
    )
    torch._check(
        data.dtype in (torch.float16, torch.bfloat16),
        lambda: f"hadamard_rotate_full only supports float16/bfloat16, got {data.dtype}",
    )

    num_rows = data.numel() // dim
    tname = _KBIT_DTYPE_SUFFIX[data.dtype]
    signs_ptr = get_ptr(signs) if signs is not None else None
    with _cuda_device_of(data):
        fn = getattr(lib, f"chadamard_rotate_full_{tname}")
        fn(
            get_ptr(data),
            ct.c_int(num_rows),
            ct.c_int(dim),
            signs_ptr,
            _get_tensor_stream(data),
        )

    return data


class _WorkspaceCache:
    """Per-device cache for split-K workspace buffers (C_workspace + tile_counters).

    Avoids repeated torch.zeros allocations in the default (non-workspace) path.
    Buffers are allocated at the max size seen per device and reused via views.
    The _impl functions call .zero_() on the views, so only used elements are zeroed.

    Memory cost is modest: at M=16 with N=5120, C_workspace is 320 KB (float32)
    and tile_counters is <1 KB.  For MoE with 8 experts × max_M=16, C_workspace
    is ~2.5 MB.  Buffers are never freed until process exit.

    Not thread-safe — assumes single-threaded inference (typical for LLM serving).
    """

    def __init__(self):
        # {device_index: (flat_ws_tensor, flat_tc_tensor)}
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, device: torch.device, ws_numel: int, tc_numel: int):
        """Return (C_workspace_flat, tile_counters_flat) views of cached buffers.

        Grows the cache if needed, never shrinks.
        """
        idx = device.index if device.index is not None else 0
        if idx in self._cache:
            ws_buf, tc_buf = self._cache[idx]
            if ws_buf.numel() >= ws_numel and tc_buf.numel() >= tc_numel:
                return ws_buf[:ws_numel], tc_buf[:tc_numel]

        # Allocate with 2x headroom to reduce re-allocations
        ws_buf = torch.empty(max(ws_numel * 2, 1), device=device, dtype=torch.float32)
        tc_buf = torch.empty(max(tc_numel * 2, 1024), device=device, dtype=torch.int32)
        self._cache[idx] = (ws_buf, tc_buf)
        return ws_buf[:ws_numel], tc_buf[:tc_numel]

    def clear(self):
        """Free all cached buffers."""
        self._cache.clear()


_workspace_cache = _WorkspaceCache()


def _kbit_gemm_prod_check(A, B_packed, B_absmax, codebook, N, k, k_chunks):
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_gemm_prod supports float16 and bfloat16, got {A.dtype}",
    )
    torch._check(B_packed.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed.dtype}")
    torch._check(
        B_absmax.dtype in (torch.uint8, torch.float16),
        lambda: f"B_absmax must be uint8 (E4M4) or float16, got {B_absmax.dtype}",
    )
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(N % 64 == 0, lambda: f"N ({N}) must be divisible by 64")
    torch._check(k_chunks >= 1, lambda: f"k_chunks must be >= 1, got {k_chunks}")


def _kbit_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, k, k_chunks, C, C_workspace, tile_counters):
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax.dtype == torch.float16 else ""

    # Zero workspace and counters (required by atomicAdd accumulation)
    C_workspace.zero_()
    tile_counters.zero_()

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_gemm_prod_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            ct.c_int(A.shape[0]),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(k_chunks),
            _get_tensor_stream(A),
        )


@register_kernel("bitsandbytes::kbit_gemm_prod", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    k_chunks: int,
) -> torch.Tensor:
    _kbit_gemm_prod_check(A, B_packed, B_absmax, codebook, N, k, k_chunks)

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)

    TILE_M = 16
    TILE_N = 64  # worst case (most tiles)
    m_tiles = (M + TILE_M - 1) // TILE_M
    n_tiles = N // TILE_N

    ws_flat, tc_flat = _workspace_cache.get(A.device, M * N, m_tiles * n_tiles)
    C_workspace = ws_flat.view(M, N)
    tile_counters = tc_flat

    _kbit_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, k, k_chunks, C, C_workspace, tile_counters)
    return C


@register_kernel("bitsandbytes::kbit_gemm_prod_", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    k_chunks: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    _kbit_gemm_prod_check(A, B_packed, B_absmax, codebook, N, k, k_chunks)
    _kbit_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, k, k_chunks, out, C_workspace, tile_counters)
    return out


def _vq_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, p, k_chunks, C, C_workspace, tile_counters, index_bits=8):
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"

    # Zero workspace and counters (required by atomicAdd accumulation)
    C_workspace.zero_()
    tile_counters.zero_()

    with _cuda_device_of(A):
        fn = getattr(lib, f"cvq_gemm_prod_{dtype_suffix}_p{p}b{index_bits}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            ct.c_int(A.shape[0]),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(k_chunks),
            _get_tensor_stream(A),
        )


@register_kernel("bitsandbytes::vq_gemm_prod", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    k_chunks: int,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_gemm_prod supports float16 and bfloat16, got {A.dtype}",
    )

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)

    TILE_M = 16
    TILE_N = 64  # worst case (most tiles)
    m_tiles = (M + TILE_M - 1) // TILE_M
    n_tiles = N // TILE_N

    ws_flat, tc_flat = _workspace_cache.get(A.device, M * N, m_tiles * n_tiles)
    C_workspace = ws_flat.view(M, N)
    tile_counters = tc_flat

    _vq_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, p, k_chunks, C, C_workspace, tile_counters, index_bits)
    return C


@register_kernel("bitsandbytes::vq_gemm_prod_", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    k_chunks: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_gemm_prod_ supports float16 and bfloat16, got {A.dtype}",
    )
    _vq_gemm_prod_impl(A, B_packed, B_absmax, codebook, K_dim, N, p, k_chunks, out, C_workspace, tile_counters, index_bits)
    return out


def _kbit_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, k):
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_grouped_gemm supports float16 and bfloat16, got {A_concat.dtype}",
    )
    torch._check(B_packed_all.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed_all.dtype}")
    torch._check(
        B_absmax_all.dtype in (torch.uint8, torch.float16),
        lambda: f"B_absmax must be uint8 (E4M4) or float16, got {B_absmax_all.dtype}",
    )
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(
        expert_offsets.dtype == torch.int32, lambda: f"expert_offsets must be int32, got {expert_offsets.dtype}"
    )
    torch._check(N % 64 == 0, lambda: f"N ({N}) must be divisible by 64")


def _kbit_grouped_gemm_impl(
    A_concat,
    B_packed_all,
    B_absmax_all,
    codebook,
    expert_offsets,
    K_dim,
    N,
    k,
    num_experts,
    max_M,
    C_concat,
    C_workspace,
    tile_counters,
):
    dtype_suffix = "fp16" if A_concat.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax_all.dtype == torch.float16 else ""

    # Zero workspace and counters (required by atomicAdd accumulation)
    C_workspace.zero_()
    tile_counters.zero_()

    with _cuda_device_of(A_concat):
        fn = getattr(lib, f"ckbit_grouped_gemm_prod_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A_concat),
            get_ptr(B_packed_all),
            get_ptr(B_absmax_all),
            get_ptr(codebook),
            get_ptr(C_concat),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            get_ptr(expert_offsets),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(num_experts),
            ct.c_int(max_M),
            _get_tensor_stream(A_concat),
        )


@register_kernel("bitsandbytes::kbit_grouped_gemm", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    num_experts: int,
    max_M: int,
) -> torch.Tensor:
    _kbit_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, k)

    total_M = A_concat.shape[0]
    C_concat = torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)

    m_blocks = 1
    if max_M > 48:
        m_blocks = 4
    elif max_M > 32:
        m_blocks = 3
    elif max_M > 16:
        m_blocks = 2
    tile_n = 64 if (m_blocks == 1 and N % 64 == 0) else 128
    n_tiles = N // tile_n
    m_tiles = (max_M + m_blocks * 16 - 1) // (m_blocks * 16)
    mn_tiles = num_experts * m_tiles * n_tiles

    ws_flat, tc_flat = _workspace_cache.get(A_concat.device, total_M * N, mn_tiles)
    C_workspace = ws_flat.view(total_M, N)
    tile_counters = tc_flat

    _kbit_grouped_gemm_impl(
        A_concat,
        B_packed_all,
        B_absmax_all,
        codebook,
        expert_offsets,
        K_dim,
        N,
        k,
        num_experts,
        max_M,
        C_concat,
        C_workspace,
        tile_counters,
    )
    return C_concat


@register_kernel("bitsandbytes::kbit_grouped_gemm_", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    _kbit_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, k)
    _kbit_grouped_gemm_impl(
        A_concat,
        B_packed_all,
        B_absmax_all,
        codebook,
        expert_offsets,
        K_dim,
        N,
        k,
        num_experts,
        max_M,
        out,
        C_workspace,
        tile_counters,
    )
    return out


# VQ Grouped GEMM — fused VQ codebook MoE GEMM

def _vq_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, p, index_bits=8):
    torch._check(p in (2, 3, 4), lambda: f"VQ grouped GEMM supports p=2,3,4, got {p}")
    torch._check(index_bits in (8, 10), lambda: f"VQ grouped GEMM supports index_bits=8,10, got {index_bits}")
    torch._check(
        (p, index_bits) in ((2, 8), (2, 10), (3, 8), (3, 10), (4, 8)),
        lambda: f"Unsupported VQ config (p={p}, index_bits={index_bits})",
    )
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_grouped_gemm supports float16 and bfloat16, got {A_concat.dtype}",
    )
    torch._check(B_packed_all.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed_all.dtype}")
    torch._check(B_absmax_all.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax_all.dtype}")
    torch._check(codebook.dtype == torch.float16, lambda: f"codebook must be float16, got {codebook.dtype}")
    torch._check(
        expert_offsets.dtype == torch.int32, lambda: f"expert_offsets must be int32, got {expert_offsets.dtype}"
    )
    torch._check(N % 64 == 0, lambda: f"N ({N}) must be divisible by 64")


def _vq_grouped_gemm_impl(
    A_concat,
    B_packed_all,
    B_absmax_all,
    codebook,
    expert_offsets,
    K_dim,
    N,
    p,
    num_experts,
    max_M,
    C_concat,
    C_workspace,
    tile_counters,
    index_bits=8,
):
    dtype_suffix = "fp16" if A_concat.dtype == torch.float16 else "bf16"

    # Zero workspace and counters (required by atomicAdd accumulation)
    C_workspace.zero_()
    tile_counters.zero_()

    with _cuda_device_of(A_concat):
        fn = getattr(lib, f"cvq_grouped_gemm_prod_{dtype_suffix}_p{p}b{index_bits}")
        fn(
            get_ptr(A_concat),
            get_ptr(B_packed_all),
            get_ptr(B_absmax_all),
            get_ptr(codebook),
            get_ptr(C_concat),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            get_ptr(expert_offsets),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(num_experts),
            ct.c_int(max_M),
            _get_tensor_stream(A_concat),
        )


@register_kernel("bitsandbytes::vq_grouped_gemm", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    index_bits: int = 8,
) -> torch.Tensor:
    _vq_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, p, index_bits)

    total_M = A_concat.shape[0]
    C_concat = torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)

    m_blocks = 1
    if max_M > 48:
        m_blocks = 4
    elif max_M > 32:
        m_blocks = 3
    elif max_M > 16:
        m_blocks = 2
    tile_n = 64 if (m_blocks == 1 and N % 64 == 0) else 128
    n_tiles = N // tile_n
    m_tiles = (max_M + m_blocks * 16 - 1) // (m_blocks * 16)
    mn_tiles = num_experts * m_tiles * n_tiles

    ws_flat, tc_flat = _workspace_cache.get(A_concat.device, total_M * N, mn_tiles)
    C_workspace = ws_flat.view(total_M, N)
    tile_counters = tc_flat

    _vq_grouped_gemm_impl(
        A_concat,
        B_packed_all,
        B_absmax_all,
        codebook,
        expert_offsets,
        K_dim,
        N,
        p,
        num_experts,
        max_M,
        C_concat,
        C_workspace,
        tile_counters,
        index_bits,
    )
    return C_concat


@register_kernel("bitsandbytes::vq_grouped_gemm_", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    _vq_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, p, index_bits)
    _vq_grouped_gemm_impl(
        A_concat,
        B_packed_all,
        B_absmax_all,
        codebook,
        expert_offsets,
        K_dim,
        N,
        p,
        num_experts,
        max_M,
        out,
        C_workspace,
        tile_counters,
        index_bits,
    )
    return out


# VQ Grouped Scalar GEMV — fused VQ codebook MoE scalar GEMV (M=1-4)

def _vq_grouped_scalar_gemv_impl(
    A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets,
    K_dim, N, p, num_experts, max_M, C_concat, index_bits=8,
):
    dtype_suffix = "fp16" if A_concat.dtype == torch.float16 else "bf16"

    with _cuda_device_of(A_concat):
        fn = getattr(lib, f"cvq_grouped_scalar_gemv_{dtype_suffix}_p{p}b{index_bits}")
        fn(
            get_ptr(A_concat),
            get_ptr(B_packed_all),
            get_ptr(B_absmax_all),
            get_ptr(codebook),
            get_ptr(C_concat),
            get_ptr(expert_offsets),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(num_experts),
            ct.c_int(max_M),
            _get_tensor_stream(A_concat),
        )


@register_kernel("bitsandbytes::vq_grouped_scalar_gemv", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    index_bits: int = 8,
) -> torch.Tensor:
    _vq_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, p, index_bits)
    torch._check(max_M <= 4, lambda: f"vq_grouped_scalar_gemv supports max_M<=4, got {max_M}")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_grouped_scalar_gemv supports float16 and bfloat16, got {A_concat.dtype}",
    )

    total_M = A_concat.shape[0]
    C_concat = torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)

    _vq_grouped_scalar_gemv_impl(
        A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets,
        K_dim, N, p, num_experts, max_M, C_concat, index_bits,
    )
    return C_concat


@register_kernel("bitsandbytes::vq_grouped_scalar_gemv_", "cuda")
def _(
    A_concat: torch.Tensor,
    B_packed_all: torch.Tensor,
    B_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    expert_offsets: torch.Tensor,
    K_dim: int,
    N: int,
    p: int,
    num_experts: int,
    max_M: int,
    out: torch.Tensor,
    index_bits: int = 8,
) -> torch.Tensor:
    _vq_grouped_gemm_check(A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets, N, p, index_bits)
    torch._check(max_M <= 4, lambda: f"vq_grouped_scalar_gemv_ supports max_M<=4, got {max_M}")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"vq_grouped_scalar_gemv_ supports float16 and bfloat16, got {A_concat.dtype}",
    )

    _vq_grouped_scalar_gemv_impl(
        A_concat, B_packed_all, B_absmax_all, codebook, expert_offsets,
        K_dim, N, p, num_experts, max_M, out, index_bits,
    )
    return out


def _kbit_scalar_gemv_impl(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
) -> None:
    M = A.shape[0]
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax.dtype == torch.float16 else ""

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_scalar_gemv_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(out),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(A),
        )


@register_kernel("bitsandbytes::kbit_scalar_gemv", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_scalar_gemv supports float16 and bfloat16, got {A.dtype}",
    )

    M = A.shape[0]
    out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    _kbit_scalar_gemv_impl(A, B_packed, B_absmax, codebook, K_dim, N, k, out=out)
    return out


@register_kernel("bitsandbytes::kbit_scalar_gemv.out", "cuda")
def _(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_absmax: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
) -> None:
    _kbit_scalar_gemv_impl(A, B_packed, B_absmax, codebook, K_dim, N, k, out=out)


@register_kernel("bitsandbytes::kbit_scalar_gemv_tiled", "cuda")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_scalar_gemv_tiled supports float16 and bfloat16, got {A.dtype}",
    )

    M = A.shape[0]
    out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax_tiled.dtype == torch.float16 else ""

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_scalar_gemv_tiled_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed_tiled),
            get_ptr(B_absmax_tiled),
            get_ptr(codebook),
            get_ptr(out),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(A),
        )
    return out


@register_kernel("bitsandbytes::kbit_scalar_gemv_tiled_", "cuda")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_scalar_gemv_tiled_ supports float16 and bfloat16, got {A.dtype}",
    )

    M = A.shape[0]
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax_tiled.dtype == torch.float16 else ""

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_scalar_gemv_tiled_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed_tiled),
            get_ptr(B_absmax_tiled),
            get_ptr(codebook),
            get_ptr(out),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(A),
        )
    return out


@register_kernel("bitsandbytes::kbit_scalar_gemv_v2_", "cuda")
def _(
    A: torch.Tensor,
    B_packed_tiled: torch.Tensor,
    B_absmax_tiled: torch.Tensor,
    codebook: torch.Tensor,
    K_dim: int,
    N: int,
    k: int,
    out: torch.Tensor,
    C_workspace: torch.Tensor,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_scalar_gemv_v2_ supports float16 and bfloat16, got {A.dtype}",
    )

    M = A.shape[0]
    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"
    abs_suffix = "_fp16abs" if B_absmax_tiled.dtype == torch.float16 else ""

    # Zero workspace and counters (required by atomicAdd accumulation)
    C_workspace.zero_()
    tile_counters.zero_()

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_scalar_gemv_v2_{dtype_suffix}{abs_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed_tiled),
            get_ptr(B_absmax_tiled),
            get_ptr(codebook),
            get_ptr(out),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            _get_tensor_stream(A),
        )
    return out
