from collections.abc import Sequence
from functools import cache, wraps
from math import prod, sqrt
from typing import Optional

import torch

from ..._ops import register_kernel
from ..utils import _get_4bit_code


def _try_torch_compile(func=None, **compile_kwargs):
    """
    Wrapper around torch.compile that falls back to the original function if compilation fails.
    """

    def decorator(fn):
        try:
            compiled_fn = torch.compile(fn, **compile_kwargs)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return compiled_fn(*args, **kwargs)
                except Exception:
                    return fn(*args, **kwargs)

            return wrapper
        except Exception:
            return fn

    if func is None:
        return decorator
    else:
        return decorator(func)


@register_kernel("bitsandbytes::int8_mm_dequant", "default")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if A.dtype != torch.int32:
        raise ValueError(f"A must be int32, got {A.dtype}")
    if row_stats.dtype != torch.float32:
        raise ValueError(f"row_stats must be float32, got {row_stats.dtype}")
    if col_stats.dtype != torch.float32:
        raise ValueError(f"col_stats must be float32, got {col_stats.dtype}")

    A_calc = A.view(-1, A.shape[-1])
    row_stats = row_stats.reshape(-1).unsqueeze(-1)
    col_stats = col_stats.reshape(-1).unsqueeze(0)

    out = A_calc * (row_stats * col_stats) * 6.200124e-05
    if bias is not None:
        out += bias

    return out.to(dtype or torch.float16)


@register_kernel("bitsandbytes::int8_mixed_scaled_mm", "default")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    subB = None

    if outlier_cols is not None and outlier_cols.numel():
        # Extract the inputs with outliers in original precision
        subA = A[:, outlier_cols].contiguous()

        # Dequantize the corresponding weight columns
        subB = (
            torch.ops.bitsandbytes.int8_vectorwise_dequant.default(CB[:, outlier_cols].contiguous(), SCB)
            .to(A.dtype)
            .t()
        )

        # TODO: if state.has_fp16_weights: subB = B[:, outlier_cols].t()

    else:
        # Needed for torch.compile when there are no outliers.
        subA = torch.empty(0, device=A.device, dtype=A.dtype)

    # Int8 Matmul + Dequant + Bias
    output = torch.ops.bitsandbytes.int8_scaled_mm.default(CA, CB, SCA, SCB, bias=bias, dtype=A.dtype)

    if subB is not None:
        # Add the outlier columns back to the output
        output = output.addmm(subA, subB)

    return output, subA


@register_kernel("bitsandbytes::int8_scaled_mm", "default")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out_i32 = torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)
    return torch.ops.bitsandbytes.int8_mm_dequant.default(
        out_i32,
        row_stats,
        col_stats,
        dtype=dtype or torch.float16,
        bias=bias,
    )


@register_kernel("bitsandbytes::int8_linear_matmul", "default")
def _(A: torch.Tensor, B: torch.Tensor):
    return _int8_linear_matmul_impl(A, B)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "default")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result


@register_kernel("bitsandbytes::int8_vectorwise_quant", "default")
def _(A: torch.Tensor, threshold=0.0):
    rows = A.numel() // A.shape[-1]
    outlier_cols = None

    outlier_restore = None

    if threshold > 0.0:
        outliers = A.abs() >= threshold

        if outliers.any():
            # Determine which columns contain outliers, and zero out the
            # outliers ahead of quantization. We need to keep a backup of these
            # outliers to restore them after quantization.
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
            outlier_restore = A[outliers].clone()
            A[outliers] = 0
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

    # Get absmax for each row.
    row_stats = torch.max(A.abs(), dim=1).values.float()

    # Quantize row-wise to int8.
    out_row = torch.round(A * (127.0 / row_stats.unsqueeze(-1))).to(torch.int8)

    # Zero out values from outlier columns across all rows.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    # Restore outliers.
    if outlier_restore is not None:
        A[outliers] = outlier_restore

    return out_row, row_stats, outlier_cols


@register_kernel("bitsandbytes::quantize_blockwise", "default")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    A_flat = A.reshape(-1).float()
    n = A_flat.numel()
    rem = n % blocksize
    full = n - rem
    blocks = full // blocksize
    A_com = A_flat[:full].reshape(blocks, blocksize)
    absmax = A_com.abs().max(dim=-1)[0]
    scaled = torch.clamp(A_com * (1.0 / absmax.clamp(min=1e-38).view(-1, 1)), -1, 1).reshape(-1)
    if rem:
        am = A_flat[full:].abs().max().clamp(min=1e-38)
        absmax = torch.cat([absmax, am.unsqueeze(0)])
        scaled = torch.cat([scaled, torch.clamp(A_flat[full:] / am, -1, 1)])
    bounds = (code[:-1] + code[1:]) / 2  # code is always sorted (same assumption as CUDA kernel)
    q = torch.bucketize(scaled, bounds, out_int32=True).to(torch.uint8)
    return q.reshape(A.shape), absmax


@_try_torch_compile(dynamic=False)
def _dequantize_blockwise_compute(
    A_flat: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
):
    n = A_flat.numel()
    out = code[A_flat.to(torch.int64)]
    rem = n % blocksize
    if rem == 0:
        out = (out.reshape(-1, blocksize) * absmax.view(-1, 1)).reshape(n)
    else:
        full = n - rem
        blocks = full // blocksize
        out = torch.cat(
            [
                (out[:full].reshape(blocks, blocksize) * absmax[:blocks].view(-1, 1)).reshape(full),
                out[full:] * absmax[blocks],
            ]
        )
    return out.to(dtype)


@register_kernel("bitsandbytes::dequantize_blockwise", "default")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    return _dequantize_blockwise_compute(A.reshape(-1), absmax, code, blocksize, dtype).reshape(A.shape)


@cache
def _get_4bit_quantize_bounds(quant_type: str, device: torch.device):
    code = _get_4bit_code(quant_type, device)
    order = torch.argsort(code)
    midpoints = (code[order[:-1]] + code[order[1:]]) / 2
    return midpoints, order  # NF4 order is identity (sorted); FP4 needs remap


@register_kernel("bitsandbytes::quantize_4bit", "default")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds, order = _get_4bit_quantize_bounds(quant_type, A.device)
    A_flat = A.reshape(-1).float()
    n = A_flat.numel()
    rem = n % blocksize
    full = n - rem
    blocks = full // blocksize
    A_com = A_flat[:full].reshape(blocks, blocksize)
    absmax = A_com.abs().max(dim=-1)[0]
    scaled = torch.clamp(A_com * (1.0 / absmax.clamp(min=1e-38).view(-1, 1)), -1, 1).reshape(-1)
    if rem:
        am = A_flat[full:].abs().max().clamp(min=1e-38)
        absmax = torch.cat([absmax, am.unsqueeze(0)])
        scaled = torch.cat([scaled, torch.clamp(A_flat[full:] / am, -1, 1)])
    if scaled.numel() % 2:
        scaled = torch.nn.functional.pad(scaled, (0, 1))
    q = torch.bucketize(scaled, bounds, out_int32=True)
    if quant_type != "nf4":
        q = order[q]
    q8 = q.to(torch.uint8)
    packed = ((q8[::2] << 4) | q8[1::2]).unsqueeze(1)
    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)
    return packed, absmax


@_try_torch_compile(dynamic=False)
def _dequantize_4bit_compute(
    A_flat: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    shape: Sequence[int],
    dtype: torch.dtype,
):
    n = prod(shape)
    out_dq = torch.empty(A_flat.size(0) * 2, dtype=torch.int32, device=A_flat.device)
    out_dq[1::2] = A_flat & 0xF
    out_dq[::2] = A_flat >> 4
    out_dq = code[out_dq][:n]  # stays fp32, matches C++ / CUDA behavior
    rem = n % blocksize
    if rem:
        full = n - rem
        blocks = full // blocksize
        out = torch.empty(n, dtype=torch.float32, device=A_flat.device)
        out[:full] = (out_dq[:full].view(-1, blocksize) * absmax[:blocks].view(-1, 1)).reshape(full)
        out[full:] = out_dq[full:] * absmax[blocks]
    else:
        out = (out_dq.view(-1, blocksize) * absmax.view(-1, 1)).reshape(n)
    return out.reshape(-1, *shape[1:]).to(dtype)


@register_kernel("bitsandbytes::dequantize_4bit", "default")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)
    code = _get_4bit_code(quant_type, A.device)
    return _dequantize_4bit_compute(A.reshape(-1), absmax, code, blocksize, shape, dtype)


@register_kernel("bitsandbytes::gemv_4bit", "default")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # Applied from dequantize_4bit
    quant_type = "fp4" if code[1] > 0 else "nf4"
    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(B, absmax, blocksize, quant_type, shapeB, A.dtype)

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )


def _gemm_4bit_default_impl(
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
    # When nested, per-block scale = absmax_code[absmax_8bit[i]] * absmax[i // 256] + absmax_offset
    if absmax_8bit is not None:
        absmax = (
            torch.ops.bitsandbytes.dequantize_blockwise.default(absmax_8bit, absmax, absmax_code, 256, torch.float32)
            + absmax_offset
        )
    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(B, absmax, blocksize, quant_type, shapeB, A.dtype)
    return torch.nn.functional.linear(A, B_dq, bias)


register_kernel("bitsandbytes::gemm_4bit", "default")(_gemm_4bit_default_impl)


MOMENTUM = 0
RMSPROP = 1
ADAGRAD = 2
ADAM = 3
# LION should be larger than MOMENTUM, RMSPROP, ADAGRAD due to comparison in kernels
LION = 4
ADEMAMIX = 5

name2optimizer_id = {
    "momentum": MOMENTUM,
    "lars": MOMENTUM,
    "rmsprop": RMSPROP,
    "adagrad": ADAGRAD,
    "adam": ADAM,
    "lamb": ADAM,
    "lion": LION,
    "ademamix": ADEMAMIX,
}


@_try_torch_compile
def _optimizer_precondition_32bit(
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    unorm_vec: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    lr: float,
    gnorm_scale: float,
    optimizer_id: int,
):
    """Preprocessing optimizer, computing update norm"""

    g_vals = gnorm_scale * g

    if optimizer_id == 3:  # ADAM
        correction1 = 1.0 / (1.0 - beta1**step)
        correction2 = 1.0 / (1.0 - beta2**step)

        s1_vals = state1 * beta1 + (1.0 - beta1) * g_vals
        s2_vals = state2 * beta2 + (1.0 - beta2) * g_vals * g_vals

        s1_vals = s1_vals * correction1
        s2_vals = s2_vals * correction2

        update_vals = s1_vals / (torch.sqrt(s2_vals) + eps)
        update_norm = update_vals * update_vals

    elif optimizer_id == 5:  # ADEMAMIX
        update_norm = state1

    elif optimizer_id == 0:  # MOMENTUM
        if step == 1:
            s1_vals = g_vals
        else:
            s1_vals = state1 * beta1 + g_vals
        update_norm = s1_vals * s1_vals

    elif optimizer_id == 4:  # LION
        s1_vals = state1 * beta2 + (1.0 - beta2) * g_vals
        update_norm = s1_vals

    elif optimizer_id == 1:  # RMSPROP
        s1_vals = state1 * beta1 + (1.0 - beta1) * g_vals * g_vals
        update_vals = g_vals / (torch.sqrt(s1_vals) + eps)
        update_norm = update_vals * update_vals

    elif optimizer_id == 2:  # ADAGRAD
        s1_vals = state1 + g_vals * g_vals
        update_vals = g_vals / (torch.sqrt(s1_vals) + eps)
        update_norm = update_vals * update_vals

    total_norm = torch.sum(update_norm)
    unorm_vec.add_(total_norm)


@_try_torch_compile
def _optimizer_update_32bit(
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
    optimizer_id: int,
):
    """Unified optimizer update kernel"""

    p_vals = p.float()
    g_vals = (gnorm_scale * g).float()
    # Coupled (L2) weight decay: fold wd into the gradient. This is correct for
    # MOMENTUM/RMSPROP/ADAGRAD, but NOT for LION (id 4), which uses *decoupled*
    # (AdamW-style) weight decay applied to the param directly (see the LION branch
    # below and Chen et al. 2023). LION is intentionally excluded here.
    if optimizer_id in [0, 1, 2] and weight_decay > 0.0:
        g_vals = g_vals + p_vals * weight_decay

    update_scale = 1.0
    if max_unorm > 0.0:
        current_unorm = torch.sqrt(unorm_vec)
        if optimizer_id in [0, 1, 2, 4]:  # 1-state optimizers
            if current_unorm > max_unorm * param_norm + eps:
                update_scale = (max_unorm * param_norm + eps) / current_unorm
        else:  # 2-state optimizers
            if current_unorm > max_unorm * param_norm:
                update_scale = (max_unorm * param_norm) / current_unorm

    if optimizer_id == 3:  # ADAM
        s1_vals = state1 * beta1 + (1.0 - beta1) * g_vals
        s2_vals = state2 * beta2 + (1.0 - beta2) * g_vals * g_vals

        correction1 = 1.0 - beta1**step
        correction2 = sqrt(1.0 - beta2**step)
        step_size = -lr * correction2 / correction1

        if weight_decay > 0.0:
            p_vals = p_vals * (1.0 - lr * weight_decay)

        update_val = update_scale * step_size * (s1_vals / (torch.sqrt(s2_vals) + eps * correction2))
        p_vals = p_vals + update_val

        state1.copy_(s1_vals)
        state2.copy_(s2_vals)

    elif optimizer_id == 5:  # ADEMAMIX
        s1_vals = state1[0]
        s3_vals = state1[1]
        s2_vals = state2

        m1 = s1_vals * beta1 + (1.0 - beta1) * g_vals
        m2 = s3_vals * beta3 + (1.0 - beta3) * g_vals
        nu = s2_vals * beta2 + (1.0 - beta2) * g_vals * g_vals

        correction1 = 1.0 - beta1**step
        correction2 = sqrt(1.0 - beta2**step)

        if weight_decay > 0.0:
            p_vals = p_vals * (1.0 - lr * weight_decay)

        mixed_momentum = (m1 / correction1) + (alpha * m2)
        adaptive_term = (torch.sqrt(nu) / correction2) + eps
        p_vals = p_vals - lr * (mixed_momentum / adaptive_term)

        state1[0].copy_(m1)
        state1[1].copy_(m2)
        state2.copy_(nu)

    elif optimizer_id == 0:  # MOMENTUM
        if step == 1:
            s1_vals = g_vals
        else:
            s1_vals = state1 * beta1 + g_vals

        update_val = update_scale * (-lr * s1_vals)
        p_vals = p_vals + update_val

        state1.copy_(s1_vals)

    elif optimizer_id == 4:  # LION
        # Lion uses decoupled weight decay: shrink the param directly (p *= 1 - lr*wd)
        # rather than folding wd into the gradient. Matches the cpu backend, the CUDA
        # 8-bit blockwise kernel, and the Lion paper (Chen et al. 2023).
        if weight_decay > 0.0:
            p_vals = p_vals * (1.0 - lr * weight_decay)

        momentum_update = state1 * beta1 + (1.0 - beta1) * g_vals
        update_val = update_scale * lr * torch.sign(momentum_update)
        p_vals = p_vals - update_val

        s1_vals = state1 * beta2 + (1.0 - beta2) * g_vals
        state1.copy_(s1_vals)

    elif optimizer_id == 1:  # RMSPROP
        s1_vals = state1 * beta1 + (1.0 - beta1) * g_vals * g_vals
        update_val = update_scale * lr * g_vals / (torch.sqrt(s1_vals) + eps)
        p_vals = p_vals - update_val

        state1.copy_(s1_vals)

    elif optimizer_id == 2:  # ADAGRAD
        s1_vals = state1 + g_vals * g_vals
        update_val = lr * g_vals / (torch.sqrt(s1_vals) + eps)
        p_vals = p_vals - update_val

        state1.copy_(s1_vals)

    p.copy_(p_vals)


@register_kernel("bitsandbytes::optimizer_update_32bit", "default")
def _(
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
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    """
    32-bit optimizer implemented by PyTorch with @torch.compile
    """
    if skip_zeros:
        raise NotImplementedError("skip_zeros is not supported yet")

    optimizer_id = name2optimizer_id[optimizer_name]

    if optimizer_name == "lion":
        _optimizer_update_32bit(
            g,
            p,
            state1,
            state2,
            unorm_vec,
            max_unorm,
            param_norm,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            weight_decay,
            step,
            lr,
            gnorm_scale,
            optimizer_id,
        )

        if max_unorm > 0.0:
            unorm_vec.zero_()
            _optimizer_precondition_32bit(
                g, p, state1, state2, unorm_vec, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, optimizer_id
            )
    else:
        if max_unorm > 0.0:
            unorm_vec.zero_()
            _optimizer_precondition_32bit(
                g, p, state1, state2, unorm_vec, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, optimizer_id
            )

        _optimizer_update_32bit(
            g,
            p,
            state1,
            state2,
            unorm_vec,
            max_unorm,
            param_norm,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            weight_decay,
            step,
            lr,
            gnorm_scale,
            optimizer_id,
        )
