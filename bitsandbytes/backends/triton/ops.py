from collections.abc import Sequence

import torch

from .utils import _FP4_QUANT_TABLE, _NF4_QUANT_TABLE

try:
    from . import triton_kernels

    triton_available = True
except ImportError as e:
    print("Import error:", e)
    triton_available = False


def quantize_blockwise(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(A.dtype == torch.float32, lambda: f"A must be float32 on xpu, got {A.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=A.dtype)
    out = torch.empty_like(A.flatten(), dtype=torch.uint8)

    triton_kernels.quantize_blockwise_triton(A, blocksize, code, blocks, absmax, out)
    out = out.reshape(A.shape)

    return out, absmax.float()


def dequantize_blockwise(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    # torch._check(dtype == torch.float32, lambda: f"dtype must be float32 on xpu, got {dtype}")

    out = torch.empty_like(A, dtype=dtype, device=A.device)
    triton_kernels.dequant_int8_blockwise(
        A,
        code,
        absmax,
        out,
        blocksize,
    )

    return out


def dequantize_blockwise_inplace(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")

    triton_kernels.dequant_int8_blockwise(
        A,
        code,
        absmax,
        out,
        blocksize,
    )


def quantize_4bit(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()

    # TODO: Support when weight matrix is not divisible by blocksize
    torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    blocks = -(n // -(blocksize * 2))

    absmax = torch.empty((blocks * 2,), device=A.device, dtype=A.dtype)
    out = torch.empty((n // 2, 1), device=A.device, dtype=torch.uint8)

    if quant_type == "fp4":
        triton_kernels.quantize_4bit_blockwise_triton(A, blocksize, _FP4_QUANT_TABLE, blocks, absmax, out)
    else:
        triton_kernels.quantize_4bit_blockwise_triton(A, blocksize, _NF4_QUANT_TABLE, blocks, absmax, out)
    packed = out

    if quant_storage != torch.uint8:
        packed = out.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


def dequantize_4bit(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on XPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    # torch._check(
    #     A.dtype == torch.uint8,
    #     lambda: f"Blockwise 4bit dequantization on XPU only supports uint8 storage, got {A.dtype}",
    # )
    # Check if this is fine and fast
    if A.dtype != torch.uint8:
        A = A.squeeze().view(torch.uint8).unsqueeze(1)

    out = torch.empty(shape, dtype=dtype, device=A.device)

    triton_kernels._dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out


def dequantize_4bit_inplace(
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
    triton_kernels._dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def gemv_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # TODO: We need to determine whether `code` is NF4, FP4, or other.
    # Right now we assume NF4, as this is the only one supported on CPU.
    quant_type = "fp4" if code[1] > 0 else "nf4"
    B_dq = dequantize_4bit(B, absmax, blocksize, quant_type, shapeB, A.dtype)

    # For some reason directly passing code causes errors in some cases like:
    # tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=128-uint8-fp32-fc1-fp4-DQ_True-xpu]
    #
    # B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)
    # code = code.to(A.device)
    # if B.dtype != torch.uint8:
    #     B = B.squeeze().view(torch.uint8).unsqueeze(1)

    # triton_kernels._dequantize_4bit_impl_passing_code(
    #     B,
    #     absmax,
    #     blocksize,
    #     code,
    #     dtype=A.dtype,
    #     out=B_dq,
    # )

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
