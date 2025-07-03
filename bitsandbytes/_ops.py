from collections.abc import Sequence
from math import prod
from typing import Optional

import torch

_IS_TORCH_GTE_24 = False

if hasattr(torch.library, "register_fake"):
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl

# Int8 mixed precision matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_mixed_scaled_mm",
    "(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB, Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_mixed_scaled_mm")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    shapeC = (*CA.shape[:-1], CB.shape[0])

    out = torch.empty(shapeC, device=A.device, dtype=A.dtype)

    outlier_cols = torch.library.get_ctx().new_dynamic_size()
    subA = A.new_empty(outlier_cols, dtype=torch.int64)

    return out, subA


# Higher level op: int8 matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, Tensor? bias=None, ScalarType? dtype=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_scaled_mm")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B) -> Tensor",
)


@register_fake("bitsandbytes::int8_linear_matmul")
def _(A: torch.Tensor, B: torch.Tensor):
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=torch.int32)


# More info on `out` overloads:
# https://github.com/pytorch/pytorch/issues/125044
torch.library.define(
    "bitsandbytes::int8_linear_matmul.out",
    "(Tensor A, Tensor B, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::int8_linear_matmul.out")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    shapeC = (*A.shape[:-1], B.shape[0])

    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    torch._check(out.shape == shapeC, lambda: f"Expected out.shape == {shapeC}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == torch.int32, lambda: f"Expected out.dtype == int32, got {out.dtype}")


torch.library.define(
    "bitsandbytes::int8_vectorwise_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_vectorwise_quant")
def _(A: torch.Tensor, threshold=0.0):
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)

    if threshold == 0.0:
        return out_row, row_stats, None

    outlier_cols = torch.library.get_ctx().new_dynamic_size()

    return out_row, row_stats, A.new_empty(outlier_cols, dtype=torch.int64)


torch.library.define("bitsandbytes::int8_vectorwise_dequant", "(Tensor A, Tensor stats) -> Tensor")


@register_fake("bitsandbytes::int8_vectorwise_dequant")
def _(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    return torch.empty_like(A, dtype=torch.float32)


# Default PyTorch-native implementation
@register_kernel("bitsandbytes::int8_vectorwise_dequant", "default")
def _(A: torch.Tensor, stats: torch.Tensor):
    # To dequantize we divide by 127, or multiply by the reciprocal.
    return A * stats.view(-1, 1) * 7.874015718698502e-3


torch.library.define(
    "bitsandbytes::int8_mm_dequant",
    "(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_mm_dequant")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: "A must be int32")
    return torch.empty_like(A, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_double_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_double_quant")
def _(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    out_row = torch.empty_like(A, dtype=torch.int8)
    out_col = torch.empty_like(A, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)
    col_stats = torch.empty(A.shape[-1], device=A.device, dtype=torch.float32)
    outlier_n = torch.library.get_ctx().new_dynamic_size()
    outlier_cols = A.new_empty(outlier_n, dtype=torch.int64)
    return out_row, out_col, row_stats, col_stats, outlier_cols


torch.library.define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_4bit")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    return torch.empty(shape, dtype=dtype, device=A.device)


torch.library.define(
    "bitsandbytes::dequantize_4bit.out",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_4bit.out")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define(
    "bitsandbytes::quantize_4bit",
    "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_4bit")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)
    return out, absmax


torch.library.define(
    "bitsandbytes::dequantize_blockwise",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_blockwise")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    return torch.empty_like(A, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_blockwise.out",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_blockwise.out")
def _(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
):
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define("bitsandbytes::quantize_blockwise", "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)")


@register_fake("bitsandbytes::quantize_blockwise")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)
    return out, absmax


torch.library.define(
    "bitsandbytes::gemv_4bit",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor",
)


@register_fake("bitsandbytes::gemv_4bit")
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
    shape = (*A.shape[:-1], shapeB[0])
    return torch.empty(shape, device=A.device, dtype=A.dtype)


torch.library.define(
    "bitsandbytes::gemv_4bit.out",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::gemv_4bit.out")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
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
    torch._check(
        out.shape == (*A.shape[:-1], shapeB[0]),
        lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
    )
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
