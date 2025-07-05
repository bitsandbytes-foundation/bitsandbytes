from collections.abc import Sequence
import ctypes as ct
from math import prod
from typing import Optional

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib


@register_kernel("bitsandbytes::int8_linear_matmul", "mps")
def _(A: torch.Tensor, B: torch.Tensor):
    out = torch.empty((*A.shape[:-1], B.shape[0]), device=A.device, dtype=torch.int32)
    return _int8_linear_matmul_impl(A, B, out)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "mps")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    """MPS implementation of int8 linear matrix multiplication."""
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

    # Use MPS native implementation when available, otherwise fall back to CPU
    try:
        # Try MPS native implementation first
        lib.gemm_4bit_inference_naive_mps(
            ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
            get_ptr(A), get_ptr(B), get_ptr(out),
            ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc)
        )
    except (AttributeError, RuntimeError):
        # Fall back to CPU implementation if MPS function not available
        A_cpu = A.to("cpu")
        B_cpu = B.to("cpu")
        out_cpu = torch.empty_like(out, device="cpu")
        
        lib.cgemm_4bit_inference_naive_fp16(
            ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
            get_ptr(A_cpu), get_ptr(B_cpu), get_ptr(out_cpu),
            ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc)
        )
        
        out.copy_(out_cpu.to(out.device))
    return out


@register_kernel("bitsandbytes::quantize_blockwise", "mps")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MPS implementation of blockwise quantization."""
    torch._check_is_size(blocksize)

    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    # Use MPS native implementation when available, otherwise fall back to CPU
    if A.dtype == torch.float32:
        try:
            # Try MPS native implementation first
            lib.quantize_blockwise_mps(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
        except (AttributeError, RuntimeError):
            # Fall back to CPU implementation if MPS function not available
            A_cpu = A.to("cpu")
            absmax_cpu = absmax.to("cpu")
            out_cpu = out.to("cpu")
            
            lib.cquantize_blockwise_cpu_fp32(
                get_ptr(code),
                get_ptr(A_cpu),
                get_ptr(absmax_cpu),
                get_ptr(out_cpu),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
            
            absmax.copy_(absmax_cpu.to(absmax.device))
            out.copy_(out_cpu.to(out.device))
    else:
        # Fall back to Python implementation for other dtypes
        rem = n % blocksize
        has_rem = rem > 0
        
        reshaped = A_cpu.view(-1, blocksize) if not has_rem else A_cpu.view(-1)[:n - rem].view(-1, blocksize)
        absmax_cpu[:len(reshaped)] = reshaped.abs().max(dim=1)[0]
        
        if has_rem:
            absmax_cpu[-1] = A_cpu.view(-1)[n - rem:].abs().max()

        # Normalize and quantize
        for i in range(len(reshaped)):
            if absmax[i] > 0:
                normalized = A.view(-1)[i * blocksize:(i + 1) * blocksize] / absmax[i]
                # Simple quantization to 8-bit
                quantized = (normalized * 127).round().clamp(-128, 127) + 128
                out.view(-1)[i * blocksize:(i + 1) * blocksize] = quantized.to(torch.uint8)

        if has_rem:
            if absmax[-1] > 0:
                normalized = A.view(-1)[n - rem:] / absmax[-1]
                quantized = (normalized * 127).round().clamp(-128, 127) + 128
                out.view(-1)[n - rem:] = quantized.to(torch.uint8)
    
    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "mps")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int) -> torch.Tensor:
    """MPS implementation of blockwise dequantization."""
    torch._check_is_size(blocksize)

    n = A.numel()
    out = torch.empty_like(A, dtype=torch.float32)

    # Use MPS native implementation when available, otherwise fall back to CPU
    if A.dtype == torch.uint8:
        try:
            # Try MPS native implementation first
            lib.dequantize_blockwise_mps(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
        except (AttributeError, RuntimeError):
            # Fall back to CPU implementation if MPS function not available
            A_cpu = A.to("cpu")
            absmax_cpu = absmax.to("cpu")
            out_cpu = out.to("cpu")
            
            lib.cdequantize_blockwise_cpu_fp32(
                get_ptr(code),
                get_ptr(A_cpu),
                get_ptr(absmax_cpu),
                get_ptr(out_cpu),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
            
            out.copy_(out_cpu.to(out.device))
    else:
        # Fall back to Python implementation
        rem = n % blocksize
        has_rem = rem > 0
        
        reshaped = A_cpu.view(-1, blocksize) if not has_rem else A_cpu.view(-1)[:n - rem].view(-1, blocksize)
        
        for i in range(len(reshaped)):
            # Dequantize: convert back from uint8 to float
            dequantized = (reshaped[i].float() - 128) / 127.0 * absmax_cpu[i]
            out_cpu.view(-1)[i * blocksize:(i + 1) * blocksize] = dequantized

        if has_rem:
            dequantized = (A_cpu.view(-1)[n - rem:].float() - 128) / 127.0 * absmax_cpu[-1]
            out_cpu.view(-1)[n - rem:] = dequantized

    out.copy_(out_cpu.to(out.device))
    return out