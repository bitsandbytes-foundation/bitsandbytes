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


# NVFP4 quantization
@register_kernel("bitsandbytes::quantize_nvfp4", "cuda")
def _(A: torch.Tensor, tensor_scale: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = A.contiguous()
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"NVFP4 quantization requires float16/bfloat16/float32, got {A.dtype}",
    )

    if tensor_scale is None:
        tensor_scale = A.abs().max().item()

    packed = torch.zeros(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.zeros(n // 16, dtype=torch.uint8, device=A.device)

    with _cuda_device_of(A):
        if A.dtype == torch.float16:
            lib.cquantize_nvfp4_fp16(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )
        elif A.dtype == torch.bfloat16:
            lib.cquantize_nvfp4_bf16(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )
        else:
            lib.cquantize_nvfp4_fp32(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )

    ts_out = torch.tensor([tensor_scale], dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


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


# NVFP4 Hadamard rotation (in-place)
@register_kernel("bitsandbytes::hadamard_rotate_nvfp4", "cuda")
def _(A: torch.Tensor) -> None:
    A_contig = A.contiguous()
    n = A_contig.numel()
    torch._check(n % 16 == 0, lambda: f"Hadamard rotation requires numel divisible by 16, got {n}")

    with _cuda_device_of(A_contig):
        if A_contig.dtype == torch.float16:
            lib.chadamard_rotate16_fp16(get_ptr(A_contig), ct.c_int(n))
        elif A_contig.dtype == torch.bfloat16:
            lib.chadamard_rotate16_bf16(get_ptr(A_contig), ct.c_int(n))
        else:
            lib.chadamard_rotate16_fp32(get_ptr(A_contig), ct.c_int(n))

    if not A.is_contiguous():
        A.copy_(A_contig)


# Fused Hadamard rotation + NVFP4 quantize
@register_kernel("bitsandbytes::fused_hadamard_quantize_nvfp4", "cuda")
def _(A: torch.Tensor, tensor_scale: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = A.contiguous()
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")

    if tensor_scale is None:
        # Compute scale on rotated data
        A_copy = A.clone()
        torch.ops.bitsandbytes.hadamard_rotate_nvfp4(A_copy)
        tensor_scale = A_copy.abs().max().item()

    packed = torch.zeros(n // 2, dtype=torch.uint8, device=A.device)
    block_scales = torch.zeros(n // 16, dtype=torch.uint8, device=A.device)

    with _cuda_device_of(A):
        if A.dtype == torch.float16:
            lib.cfused_hadamard_quantize_nvfp4_fp16(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )
        elif A.dtype == torch.bfloat16:
            lib.cfused_hadamard_quantize_nvfp4_bf16(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )
        else:
            lib.cfused_hadamard_quantize_nvfp4_fp32(
                get_ptr(A), get_ptr(packed), get_ptr(block_scales), ct.c_float(tensor_scale), ct.c_int(n)
            )

    ts_out = torch.tensor([tensor_scale], dtype=torch.float32, device=A.device)
    return packed, block_scales, ts_out


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


def _fused_quantize_nvfp4_impl(
    A: torch.Tensor,
    tensor_scale: float,
    packed_out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
    global_scale_buf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core CUTLASS fused quantize implementation.

    When output buffers are provided, no allocations occur — safe for CUDA
    graph capture. When None, buffers are allocated (convenient but not
    graph-safe).

    Args:
        A: BF16 input, numel must be divisible by 16.
        tensor_scale: Global tensor scale.
        packed_out: Pre-allocated uint8 output (padded_M * 8 bytes). None to allocate.
        scales_out: Pre-allocated uint8 scales (padded_M bytes). None to allocate.
        global_scale_buf: Pre-allocated float32 scalar buffer. None to allocate.
    """
    A = A.contiguous()
    n = A.numel()
    torch._check(n % 16 == 0, lambda: f"NVFP4 requires numel divisible by 16, got {n}")
    torch._check(
        A.dtype == torch.bfloat16,
        lambda: f"CUTLASS fused quantize requires bfloat16, got {A.dtype}",
    )

    K = 16
    N = 16
    orig_M = n // K
    padded_M = ((orig_M + 127) // 128) * 128

    # Pad input if needed
    if padded_M != orig_M:
        A_2d = A.view(orig_M, K)
        pad_rows = padded_M - orig_M
        A_2d = torch.nn.functional.pad(A_2d, (0, 0, 0, pad_rows))
        A_flat = A_2d.reshape(-1)
    else:
        A_flat = A

    # Use pre-allocated buffers or allocate new ones
    if global_scale_buf is not None:
        global_scale_buf.fill_(1.0 / tensor_scale if tensor_scale > 0 else 0.0)
        global_scale = global_scale_buf
    else:
        global_scale = torch.tensor(
            [1.0 / tensor_scale if tensor_scale > 0 else 0.0],
            dtype=torch.float32,
            device=A.device,
        )

    packed_padded = (
        packed_out if packed_out is not None else torch.zeros(padded_M * K // 2, dtype=torch.uint8, device=A.device)
    )
    scales_padded = scales_out if scales_out is not None else torch.zeros(padded_M, dtype=torch.uint8, device=A.device)

    B = _get_rotation_matrix(A.device)

    with _cuda_device_of(A):
        lib.cfused_quantize_nvfp4_absmax(
            get_ptr(A_flat),
            get_ptr(B),
            get_ptr(packed_padded),
            get_ptr(scales_padded),
            get_ptr(global_scale),
            ct.c_int(padded_M),
            ct.c_int(N),
            ct.c_int(K),
            _get_tensor_stream(A),
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


# NVFP4 GEMM (CUTLASS-based)
#
# Expects pre-swizzled scales in CUTLASS block-scaled layout (computed at
# quantization time by scale_to_blocked). Tensor scales are folded into
# the CUTLASS epilogue alpha. Output is BF16, converted to FP32 for
# API compatibility.
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
    D_out: Optional[torch.Tensor] = None,
    alpha_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Core NVFP4 GEMM implementation.

    When D_out and alpha_buf are provided, no allocations occur — safe for
    CUDA graph capture. When None, buffers are allocated.

    Args:
        D_out: Pre-allocated BF16 output (M, N). None to allocate.
        alpha_buf: Pre-allocated float32 scalar buffer. None to allocate.
    """
    with _cuda_device_of(A_packed):
        if alpha_buf is not None:
            alpha_buf.fill_(A_tensor_scale * B_tensor_scale)
            alpha = alpha_buf
        else:
            alpha = torch.tensor([A_tensor_scale * B_tensor_scale], dtype=torch.float32, device=A_packed.device)

        if D_out is None:
            D_out = torch.empty(M, N, dtype=torch.bfloat16, device=A_packed.device)

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
