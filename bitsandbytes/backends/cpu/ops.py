from collections.abc import Sequence
import ctypes as ct
import logging
import math
from math import prod
from typing import Optional

import torch

from bitsandbytes.functional import get_ptr, has_avx512bf16

from ..._ops import register_kernel
from ...cextension import ErrorHandlerMockBNBNativeLibrary, lib

logger = logging.getLogger(__name__)

_has_avx512 = torch.backends.cpu.get_cpu_capability() == "AVX512"

# torch._int_mm for s8@s8->s32 is supported on CPU from torch 2.4+.
# However, we can overflow if we use this without AVX512_VNNI support.
# This is fixed in torch 2.6+, so we set this as the minimum to be safe.
# For more information: https://github.com/pytorch/pytorch/pull/136942
# TODO(matthewdouglas): aarch64?
if torch.__version__ >= (2, 6):

    @register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])


if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):

    @register_kernel("bitsandbytes::quantize_blockwise", "cpu")
    def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check_is_size(blocksize)

        n = A.numel()
        blocks = -(n // -blocksize)

        absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
        out = torch.empty(A.shape, device=A.device, dtype=torch.uint8)

        if A.dtype == torch.float32:
            lib.cquantize_blockwise_cpu_fp32(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_cpu_bf16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
        elif A.dtype == torch.float16:
            lib.cquantize_blockwise_cpu_fp16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(n),
            )
        else:
            # Generic fallback for other dtypes
            A_flat = A.reshape(n).float()
            rem = n % blocksize
            has_rem = rem > 0
            A_com = A_flat[: n - rem]
            A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
            absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
            scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
            scaled_A = scaled_A.reshape(-1)
            if has_rem:
                absmax[-1] = torch.abs(A_flat[n - rem :]).max()
                scaled_A_rem = torch.clamp(A_flat[n - rem :] * (1 / absmax[-1]), -1, 1)
                scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

            diff = torch.abs(scaled_A.unsqueeze(-1) - code.to(scaled_A.device))
            out = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device).reshape(A.shape)

        return out, absmax

    @register_kernel("bitsandbytes::dequantize_blockwise", "cpu")
    def _(
        A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
    ) -> torch.Tensor:
        torch._check_is_size(blocksize)
        torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")

        out = torch.empty_like(A, dtype=dtype)
        if dtype == torch.float32:
            lib.cdequantize_blockwise_cpu_fp32(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        elif dtype == torch.bfloat16:
            lib.cdequantize_blockwise_cpu_bf16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        elif dtype == torch.float16:
            lib.cdequantize_blockwise_cpu_fp16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_longlong(blocksize),
                ct.c_longlong(A.numel()),
            )
        else:
            out = code[A.reshape(-1).int()]
            blocks = out.shape[-1] // blocksize
            res = out.shape[-1] % blocksize
            if res != 0:
                out = torch.nn.functional.pad(out, (0, blocksize - res), mode="constant", value=0)
            out = (out.view(-1, blocksize) * absmax.view(-1, 1)).to(dtype).reshape(-1)
            out = out[: blocks * blocksize + res]
            out = out.reshape(A.shape)

        return out

    @register_kernel("bitsandbytes::dequantize_4bit", "cpu")
    def _(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        torch._check_is_size(blocksize)
        torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
        torch._check(
            dtype in [torch.bfloat16, torch.float16, torch.float32],
            lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
        )

        # Fallback as AVX512 implementation has accuracy issues with fp16/fp32 and blocksize >= 2048
        # Note: this is not a common use case.
        avx512_fallback = _has_avx512 and blocksize >= 2048 and dtype != torch.bfloat16

        # Odd shape is not supported by this kernel; fallback to generic implementation
        shape_fallback = shape[-1] % 2 != 0

        if avx512_fallback or shape_fallback:
            from ..default.ops import _dequantize_4bit_impl

            return _dequantize_4bit_impl(A, absmax, blocksize, quant_type, shape, dtype)

        # Enable non uint8 dtype
        if A.dtype != torch.uint8:
            A = A.view(torch.uint8)

        # TODO: support half precision absmax
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

        if len(shape) == 1:
            shape = (1, shape[0])

        m = prod(shape[:-1])
        n = shape[-1]

        A = A.reshape(m, n // 2)
        out = torch.empty(shape, dtype=dtype, device=A.device)

        if quant_type == "fp4":
            if dtype == torch.float32:
                lib.cdequantize_blockwise_cpu_fp4_fp32(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
            elif dtype == torch.bfloat16:
                lib.cdequantize_blockwise_cpu_fp4_bf16(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
            elif dtype == torch.float16:
                lib.cdequantize_blockwise_cpu_fp4_fp16(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
        elif quant_type == "nf4":
            if dtype == torch.float32:
                lib.cdequantize_blockwise_cpu_nf4_fp32(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
            elif dtype == torch.bfloat16:
                lib.cdequantize_blockwise_cpu_nf4_bf16(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
            elif dtype == torch.float16:
                lib.cdequantize_blockwise_cpu_nf4_fp16(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_longlong(blocksize),
                    ct.c_longlong(m),
                    ct.c_longlong(n),
                )
        else:
            raise ValueError

        return out

    if has_avx512bf16():
        gemm_4bit_forward_kernel = None
        try:
            from kernels import get_kernel

            gemm_4bit_forward_kernel = get_kernel("kernels-community/quantization_bitsandbytes").gemm_4bit_forward
        except Exception as exc:  # pragma: no cover - best effort fallback
            gemm_4bit_forward_kernel = None
            logger.warning(
                "Failed to load CPU gemm_4bit_forward from kernels-community: %s. Please make sure you already `pip install kernels` and the kernels >= 0.11.1",
                exc,
            )

        @register_kernel("bitsandbytes::gemv_4bit", "cpu")
        def _(
            A: torch.Tensor,
            B: torch.Tensor,
            shapeB: Sequence[int],
            absmax: torch.Tensor,
            code: torch.Tensor,
            blocksize: int,
        ) -> torch.Tensor:
            assert B.dtype == torch.uint8, "Only support uint8 qweight"
            dtype = A.dtype
            quant_type = "fp4" if code[1] > 0 else "nf4"
            # cpu fused op only support bf16 for now.
            if dtype != torch.bfloat16:
                A = A.to(torch.bfloat16)

            final_out_shape = (*A.shape[:-1], shapeB[0])
            A = A.reshape(-1, A.shape[-1])
            out_shape = (*A.shape[:-1], shapeB[0])
            if gemm_4bit_forward_kernel is not None:
                quant_type_num = 1 if quant_type == "fp4" else 0
                out = gemm_4bit_forward_kernel(A, B, absmax, blocksize, quant_type_num)
            else:
                out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
                M = A.shape[0]
                N = shapeB[0]
                K = A.shape[1]
                x_strideM = A.stride(0)
                out_strideM = out.stride(0)
                if quant_type == "fp4":
                    lib.gemv_4bit_inference_cpu_fp4_bf16(
                        ct.c_int64(M),
                        ct.c_int64(N),
                        ct.c_int64(K),
                        get_ptr(A),
                        get_ptr(B),
                        get_ptr(absmax),
                        get_ptr(out),
                        ct.c_int64(blocksize),
                        ct.c_int64(x_strideM),
                        ct.c_int64(out_strideM),
                    )
                elif quant_type == "nf4":
                    lib.gemv_4bit_inference_cpu_nf4_bf16(
                        ct.c_int64(M),
                        ct.c_int64(N),
                        ct.c_int64(K),
                        get_ptr(A),
                        get_ptr(B),
                        get_ptr(absmax),
                        get_ptr(out),
                        ct.c_int64(blocksize),
                        ct.c_int64(x_strideM),
                        ct.c_int64(out_strideM),
                    )

            if dtype != torch.bfloat16:
                out = out.to(dtype)

            return out.reshape(final_out_shape)


# ==================== CPU Optimizer Kernels ====================


def _compute_update_norm_and_scale(
    update: torch.Tensor,
    unorm_vec: Optional[torch.Tensor],
    max_unorm: float,
    param_norm: float,
) -> float:
    """Compute trust-ratio scaling factor for LAMB/LARS and store update norm."""
    if max_unorm <= 0.0:
        return 1.0
    unorm = torch.norm(update).item()
    if unorm_vec is not None:
        unorm_vec.fill_(unorm)
    if unorm > max_unorm * param_norm:
        return (max_unorm * param_norm) / unorm
    return 1.0


@torch.no_grad()
def _optimizer_update_32bit_cpu(
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
    skip_zeros: bool = False,
) -> None:
    g_float = g.float() * gnorm_scale
    p_float = p.data.float()

    if optimizer_name in ("adam", "lamb"):
        # Adam / LAMB (2-state): m and v
        state1.mul_(beta1).add_(g_float, alpha=1.0 - beta1)
        state2.mul_(beta2).addcmul_(g_float, g_float, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)
        step_size = -lr * correction2 / correction1

        if weight_decay > 0.0:
            p_float.mul_(1.0 - lr * weight_decay)

        update = state1 / (state2.sqrt() + eps * correction2)

        update_scale = _compute_update_norm_and_scale(update, unorm_vec, max_unorm, param_norm)
        p_float.add_(update, alpha=step_size * update_scale)

    elif optimizer_name == "ademamix":
        # AdEMAMix (2-state): state1 shape is (2, *p.shape), state1[0]=m1, state1[1]=m2
        m1 = state1[0]
        m2 = state1[1]
        nu = state2

        m1.mul_(beta1).add_(g_float, alpha=1.0 - beta1)
        m2.mul_(beta3).add_(g_float, alpha=1.0 - beta3)
        nu.mul_(beta2).addcmul_(g_float, g_float, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)

        if weight_decay > 0.0:
            p_float.mul_(1.0 - lr * weight_decay)

        mixed_momentum = (m1 / correction1) + (alpha * m2)
        adaptive_term = (nu.sqrt() / correction2) + eps
        p_float.add_(mixed_momentum / adaptive_term, alpha=-lr)

    elif optimizer_name in ("momentum", "lars"):
        # SGD with momentum / LARS (1-state)
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float

        if step == 1:
            state1.copy_(g_wd)
        else:
            state1.mul_(beta1).add_(g_wd)

        update_scale = _compute_update_norm_and_scale(state1, unorm_vec, max_unorm, param_norm)
        p_float.add_(state1, alpha=-lr * update_scale)

    elif optimizer_name == "lion":
        # Lion (2-state sign update)
        if weight_decay > 0.0:
            p_float.mul_(1.0 - lr * weight_decay)

        update = state1.mul(beta1).add(g_float, alpha=1.0 - beta1)
        p_float.add_(update.sign(), alpha=-lr)

        state1.mul_(beta2).add_(g_float, alpha=1.0 - beta2)

    elif optimizer_name == "rmsprop":
        # RMSprop (1-state)
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float
        state1.mul_(beta1).addcmul_(g_wd, g_wd, value=1.0 - beta1)

        update = g_wd / (state1.sqrt() + eps)
        update_scale = _compute_update_norm_and_scale(update, unorm_vec, max_unorm, param_norm)
        p_float.add_(update, alpha=-lr * update_scale)

    elif optimizer_name == "adagrad":
        # Adagrad (1-state)
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float
        state1.addcmul_(g_wd, g_wd, value=1.0)

        update = g_wd / (state1.sqrt() + eps)
        p_float.add_(update, alpha=-lr)

    else:
        raise ValueError(f"Unsupported optimizer for CPU: {optimizer_name}")

    # Write back to original precision
    p.data.copy_(p_float)


register_kernel("bitsandbytes::optimizer_update_32bit", "cpu")(_optimizer_update_32bit_cpu)


@torch.no_grad()
def _dequant_blockwise_fp32_direct(
    A_uint8: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    """Dequantize blockwise via direct C lib call, avoiding torch.ops dispatch overhead."""
    n = A_uint8.numel()
    out = torch.empty(n, dtype=torch.float32, device=A_uint8.device)
    lib.cdequantize_blockwise_cpu_fp32(
        get_ptr(code),
        get_ptr(A_uint8.reshape(-1)),
        get_ptr(absmax),
        get_ptr(out),
        ct.c_longlong(blocksize),
        ct.c_longlong(n),
    )
    return out.reshape(A_uint8.shape)


def _quant_blockwise_fp32_direct(
    A_fp32: torch.Tensor, code: torch.Tensor, absmax_out: torch.Tensor, out_uint8: torch.Tensor, blocksize: int
) -> None:
    """Quantize blockwise via direct C lib call, writing into existing buffers (zero-alloc)."""
    n = A_fp32.numel()
    lib.cquantize_blockwise_cpu_fp32(
        get_ptr(code),
        get_ptr(A_fp32.reshape(-1)),
        get_ptr(absmax_out),
        get_ptr(out_uint8.reshape(-1)),
        ct.c_longlong(blocksize),
        ct.c_longlong(n),
    )


def _optimizer_update_8bit_blockwise_cpu(
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
    skip_zeros: bool = False,
) -> None:
    blocksize = 256

    # Dequantize states — direct C lib calls (no torch.ops dispatch overhead)
    if optimizer_name == "ademamix" and absmax1.ndim == 2:
        s1_1 = _dequant_blockwise_fp32_direct(state1[0], absmax1[0], qmap1, blocksize)
        s1_2 = _dequant_blockwise_fp32_direct(state1[1], absmax1[1], qmap1, blocksize)
        state1_fp32 = torch.stack([s1_1, s1_2])
    else:
        state1_fp32 = _dequant_blockwise_fp32_direct(state1, absmax1, qmap1, blocksize)

    state2_fp32 = None
    if state2 is not None and qmap2 is not None and absmax2 is not None:
        state2_fp32 = _dequant_blockwise_fp32_direct(state2, absmax2, qmap2, blocksize)

    grad = g.float() * gnorm_scale
    p_fp32 = p.data.float()

    if optimizer_name in ("adam", "lamb"):
        state1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        state2_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)

        denom = (state2_fp32.sqrt() / correction2).add_(eps)
        if weight_decay > 0.0:
            p_fp32.mul_(1.0 - lr * weight_decay)
        p_fp32.addcdiv_(state1_fp32, denom, value=-lr / correction1)

    elif optimizer_name == "ademamix":
        m1_fp32, m2_fp32 = state1_fp32[0], state1_fp32[1]
        nu_fp32 = state2_fp32

        m1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        m2_fp32.mul_(beta3).add_(grad, alpha=1.0 - beta3)
        nu_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)

        update = (m1_fp32 / correction1 + alpha * m2_fp32) / (nu_fp32.sqrt() / correction2 + eps)
        if weight_decay > 0.0:
            p_fp32.mul_(1.0 - lr * weight_decay)
        p_fp32.add_(update, alpha=-lr)

        state1_fp32 = torch.stack([m1_fp32, m2_fp32])

    elif optimizer_name in ("momentum", "lars"):
        grad.add_(p_fp32, alpha=weight_decay)
        if step == 1:
            state1_fp32.copy_(grad)
        else:
            state1_fp32.mul_(beta1).add_(grad)
        p_fp32.add_(state1_fp32, alpha=-lr)

    elif optimizer_name == "lion":
        if weight_decay > 0.0:
            p_fp32.mul_(1.0 - lr * weight_decay)

        update_dir = torch.sign(state1_fp32.mul(beta1) + grad.mul(1.0 - beta1))
        p_fp32.add_(update_dir, alpha=-lr)

        state1_fp32.mul_(beta2).add_(grad, alpha=1.0 - beta2)

    elif optimizer_name == "rmsprop":
        grad.add_(p_fp32, alpha=weight_decay)
        state1_fp32.mul_(beta1).addcmul_(grad, grad, value=1.0 - beta1)
        p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

    elif optimizer_name == "adagrad":
        grad.add_(p_fp32, alpha=weight_decay)
        state1_fp32.addcmul_(grad, grad, value=1.0)
        p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

    else:
        raise ValueError(f"Unsupported optimizer for CPU 8-bit: {optimizer_name}")

    p.data.copy_(p_fp32)

    # Re-quantize states — direct C lib calls, zero-alloc (write into existing buffers)
    if optimizer_name == "ademamix":
        _quant_blockwise_fp32_direct(state1_fp32[0], qmap1, absmax1[0], state1[0], blocksize)
        _quant_blockwise_fp32_direct(state1_fp32[1], qmap1, absmax1[1], state1[1], blocksize)
        _quant_blockwise_fp32_direct(state2_fp32, qmap2, absmax2, state2, blocksize)
    else:
        _quant_blockwise_fp32_direct(state1_fp32, qmap1, absmax1, state1, blocksize)
        if state2_fp32 is not None:
            _quant_blockwise_fp32_direct(state2_fp32, qmap2, absmax2, state2, blocksize)


register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "cpu")(_optimizer_update_8bit_blockwise_cpu)
