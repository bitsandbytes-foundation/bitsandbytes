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
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {list(str2optimizer8bit_blockwise.keys())}"
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
    absmax = torch.zeros(num_blocks + 1, device=A.device, dtype=torch.float32)

    with _cuda_device_of(A):
        tname = _KBIT_DTYPE_SUFFIX[A.dtype]
        fn = getattr(lib, f"cquantize_kbit_{tname}_k{k}")
        fn(
            get_ptr(codebook),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(packed),
            ct.c_int(n),
        )

    return packed, absmax


_KBIT_ABSMAX_SUFFIX = {
    torch.uint8: "u8abs",
    torch.float16: "fp16abs",
}


@register_kernel("bitsandbytes::dequantize_kbit", "cuda")
def _(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    absmax: torch.Tensor,
    k: int,
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
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

    num_blocks = -(n // -32)
    out = torch.empty(num_blocks * 32, device=packed.device, dtype=dtype)

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
    torch._check(absmax_flat.dtype == torch.float32, lambda: f"absmax_flat must be float32, got {absmax_flat.dtype}")

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
        )

    return packed_tiled, absmax_tiled


@register_kernel("bitsandbytes::kbit_gemm", "cuda")
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
    torch._check(A.dtype == torch.float16, lambda: f"kbit_gemm currently supports float16 only, got {A.dtype}")
    torch._check(B_packed.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed.dtype}")
    torch._check(B_absmax.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=torch.float16)

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_gemm_fp16_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
        )

    return C


@register_kernel("bitsandbytes::kbit_gemm_pipelined", "cuda")
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
    torch._check(A.dtype == torch.float16, lambda: f"kbit_gemm_pipelined supports float16 only, got {A.dtype}")
    torch._check(B_packed.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed.dtype}")
    torch._check(B_absmax.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=torch.float16)

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_gemm_pipelined_fp16_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
        )

    return C


@register_kernel("bitsandbytes::kbit_gemm_splitk", "cuda")
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
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(A.dtype == torch.float16, lambda: f"kbit_gemm_splitk supports float16 only, got {A.dtype}")
    torch._check(B_packed.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed.dtype}")
    torch._check(B_absmax.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")
    torch._check(k_chunks >= 1, lambda: f"k_chunks must be >= 1, got {k_chunks}")

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=torch.float16)

    TILE_M = 16
    TILE_N = 128
    m_tiles = (M + TILE_M - 1) // TILE_M
    n_tiles = N // TILE_N

    # Allocate workspace and tile counters for split-K (k_chunks > 1)
    if k_chunks > 1:
        C_workspace = torch.zeros(M, N, device=A.device, dtype=torch.float32)
        tile_counters = torch.zeros(m_tiles * n_tiles, device=A.device, dtype=torch.int32)
    else:
        C_workspace = torch.empty(0, device=A.device, dtype=torch.float32)
        tile_counters = torch.empty(0, device=A.device, dtype=torch.int32)

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_gemm_splitk_fp16_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(k_chunks),
        )

    return C


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
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_gemm_prod supports float16 and bfloat16, got {A.dtype}",
    )
    torch._check(B_packed.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed.dtype}")
    torch._check(B_absmax.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")
    torch._check(k_chunks >= 1, lambda: f"k_chunks must be >= 1, got {k_chunks}")

    M = A.shape[0]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)

    # The persistent kernel auto-selects k_splits internally. When
    # k_splits > 1, it needs a zeroed fp32 workspace and tile counters.
    # Always allocate these since the C++ decides at runtime.
    TILE_M = 16
    TILE_N = 128
    m_tiles = (M + TILE_M - 1) // TILE_M
    n_tiles = N // TILE_N

    C_workspace = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    tile_counters = torch.zeros(m_tiles * n_tiles, device=A.device, dtype=torch.int32)

    dtype_suffix = "fp16" if A.dtype == torch.float16 else "bf16"

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_gemm_prod_{dtype_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(C),
            get_ptr(C_workspace),
            get_ptr(tile_counters),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
            ct.c_int(k_chunks),
        )

    return C


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
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_grouped_gemm supports float16 and bfloat16, got {A_concat.dtype}",
    )
    torch._check(B_packed_all.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed_all.dtype}")
    torch._check(B_absmax_all.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax_all.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(expert_offsets.dtype == torch.int32, lambda: f"expert_offsets must be int32, got {expert_offsets.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")

    total_M = A_concat.shape[0]
    C_concat = torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)

    dtype_suffix = "fp16" if A_concat.dtype == torch.float16 else "bf16"

    with _cuda_device_of(A_concat):
        fn = getattr(lib, f"ckbit_grouped_gemm_prod_{dtype_suffix}_k{k}")
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
        )

    return C_concat


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

    with _cuda_device_of(A):
        fn = getattr(lib, f"ckbit_scalar_gemv_{dtype_suffix}_k{k}")
        fn(
            get_ptr(A),
            get_ptr(B_packed),
            get_ptr(B_absmax),
            get_ptr(codebook),
            get_ptr(out),
            ct.c_int(M),
            ct.c_int(K_dim),
            ct.c_int(N),
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


@register_kernel("bitsandbytes::kbit_grouped_scalar_gemv", "cuda")
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
) -> torch.Tensor:
    torch._check(k >= 2 and k <= 5, lambda: f"k must be 2-5, got {k}")
    torch._check(
        A_concat.dtype in (torch.float16, torch.bfloat16),
        lambda: f"kbit_grouped_scalar_gemv supports float16 and bfloat16, got {A_concat.dtype}",
    )
    torch._check(B_packed_all.dtype == torch.int32, lambda: f"B_packed must be int32, got {B_packed_all.dtype}")
    torch._check(B_absmax_all.dtype == torch.uint8, lambda: f"B_absmax must be uint8 (E4M4), got {B_absmax_all.dtype}")
    torch._check(codebook.dtype == torch.float32, lambda: f"codebook must be float32, got {codebook.dtype}")
    torch._check(expert_offsets.dtype == torch.int32, lambda: f"expert_offsets must be int32, got {expert_offsets.dtype}")
    torch._check(N % 128 == 0, lambda: f"N ({N}) must be divisible by 128")

    total_M = A_concat.shape[0]
    C_concat = torch.empty(total_M, N, device=A_concat.device, dtype=A_concat.dtype)

    dtype_suffix = "fp16" if A_concat.dtype == torch.float16 else "bf16"

    with _cuda_device_of(A_concat):
        fn = getattr(lib, f"ckbit_grouped_scalar_gemv_{dtype_suffix}_k{k}")
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
        )

    return C_concat


# ============================================================================
# Training Kernels: SwiGLU, RMSNorm, RoPE
# ============================================================================


@register_kernel("bitsandbytes::swiglu_forward", "cuda")
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    torch._check(gate.shape == up.shape, lambda: "gate and up must have same shape")
    torch._check(gate.is_contiguous(), lambda: "gate must be contiguous")
    torch._check(up.is_contiguous(), lambda: "up must be contiguous")
    torch._check(
        gate.dtype in (torch.float16, torch.bfloat16),
        lambda: f"swiglu supports float16/bfloat16, got {gate.dtype}",
    )

    out = torch.empty_like(gate)
    n = gate.numel()
    dtype_suffix = "fp16" if gate.dtype == torch.float16 else "bf16"

    with _cuda_device_of(gate):
        fn = getattr(lib, f"cswiglu_forward_{dtype_suffix}_c")
        fn(get_ptr(gate), get_ptr(up), get_ptr(out), ct.c_int(n))

    return out


@register_kernel("bitsandbytes::swiglu_backward", "cuda")
def _(grad_h: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(grad_h.is_contiguous(), lambda: "grad_h must be contiguous")
    torch._check(gate.is_contiguous(), lambda: "gate must be contiguous")
    torch._check(up.is_contiguous(), lambda: "up must be contiguous")

    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n = gate.numel()
    dtype_suffix = "fp16" if gate.dtype == torch.float16 else "bf16"

    with _cuda_device_of(gate):
        fn = getattr(lib, f"cswiglu_backward_{dtype_suffix}_c")
        fn(get_ptr(grad_h), get_ptr(gate), get_ptr(up), get_ptr(grad_gate), get_ptr(grad_up), ct.c_int(n))

    return grad_gate, grad_up


@register_kernel("bitsandbytes::rmsnorm_forward", "cuda")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    add_unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(x.dim() == 2, lambda: "x must be 2D [rows, cols]")
    torch._check(x.is_contiguous(), lambda: "x must be contiguous")
    torch._check(
        x.dtype in (torch.float16, torch.bfloat16),
        lambda: f"rmsnorm supports float16/bfloat16, got {x.dtype}",
    )

    rows, cols = x.shape
    out = torch.empty_like(x)
    rrms = torch.empty(rows, device=x.device, dtype=torch.float32)
    dtype_suffix = "fp16" if x.dtype == torch.float16 else "bf16"

    with _cuda_device_of(x):
        fn = getattr(lib, f"crmsnorm_forward_{dtype_suffix}_c")
        fn(
            get_ptr(x),
            get_ptr(w),
            get_ptr(out),
            get_ptr(rrms),
            ct.c_int(rows),
            ct.c_int(cols),
            ct.c_float(eps),
            ct.c_bool(add_unit_offset),
        )

    return out, rrms


@register_kernel("bitsandbytes::rmsnorm_backward", "cuda")
def _(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rrms: torch.Tensor,
    add_unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(grad_out.is_contiguous(), lambda: "grad_out must be contiguous")
    torch._check(x.is_contiguous(), lambda: "x must be contiguous")

    rows, cols = x.shape
    grad_x = torch.empty_like(x)
    grad_w = torch.zeros(cols, device=x.device, dtype=torch.float32)
    dtype_suffix = "fp16" if x.dtype == torch.float16 else "bf16"

    with _cuda_device_of(x):
        fn = getattr(lib, f"crmsnorm_backward_{dtype_suffix}_c")
        fn(
            get_ptr(grad_out),
            get_ptr(x),
            get_ptr(w),
            get_ptr(rrms),
            get_ptr(grad_x),
            get_ptr(grad_w),
            ct.c_int(rows),
            ct.c_int(cols),
            ct.c_bool(add_unit_offset),
        )

    return grad_x, grad_w


@register_kernel("bitsandbytes::rope_forward", "cuda")
def _(
    q: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    n_heads: int,
) -> None:
    torch._check(q.is_contiguous(), lambda: "q must be contiguous")
    torch._check(
        q.dtype in (torch.float16, torch.bfloat16),
        lambda: f"rope supports float16/bfloat16, got {q.dtype}",
    )

    total_tokens = q.shape[0]
    head_dim = q.shape[-1]
    dtype_suffix = "fp16" if q.dtype == torch.float16 else "bf16"

    with _cuda_device_of(q):
        fn = getattr(lib, f"crope_forward_{dtype_suffix}_c")
        fn(
            get_ptr(q),
            get_ptr(cos_cache),
            get_ptr(sin_cache),
            ct.c_int(total_tokens),
            ct.c_int(n_heads),
            ct.c_int(head_dim),
        )


@register_kernel("bitsandbytes::cross_entropy_forward", "cuda")
def _(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(logits.dim() == 2, lambda: "logits must be 2D [N, V]")
    torch._check(logits.is_contiguous(), lambda: "logits must be contiguous")
    torch._check(
        logits.dtype in (torch.float16, torch.bfloat16),
        lambda: f"cross_entropy supports float16/bfloat16, got {logits.dtype}",
    )

    N, V = logits.shape
    losses = torch.empty(N, device=logits.device, dtype=torch.float32)
    logsumexp = torch.empty(N, device=logits.device, dtype=torch.float32)
    dtype_suffix = "fp16" if logits.dtype == torch.float16 else "bf16"

    with _cuda_device_of(logits):
        fn = getattr(lib, f"ccross_entropy_forward_{dtype_suffix}_c")
        fn(
            get_ptr(logits),
            get_ptr(labels),
            get_ptr(losses),
            get_ptr(logsumexp),
            ct.c_int(N),
            ct.c_int(V),
            ct.c_int(ignore_index),
        )

    return losses, logsumexp


@register_kernel("bitsandbytes::cross_entropy_backward", "cuda")
def _(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    torch._check(logits.is_contiguous(), lambda: "logits must be contiguous")

    N, V = logits.shape
    grad_logits = torch.empty_like(logits)
    dtype_suffix = "fp16" if logits.dtype == torch.float16 else "bf16"

    with _cuda_device_of(logits):
        fn = getattr(lib, f"ccross_entropy_backward_{dtype_suffix}_c")
        fn(
            get_ptr(logits),
            get_ptr(labels),
            get_ptr(grad_output),
            get_ptr(logsumexp),
            get_ptr(grad_logits),
            ct.c_int(N),
            ct.c_int(V),
            ct.c_int(ignore_index),
        )

    return grad_logits
