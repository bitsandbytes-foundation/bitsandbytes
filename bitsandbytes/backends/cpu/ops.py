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
#
# Without AVX-512 (including aarch64), torch._int_mm uses a scalar fallback
# that is much slower than fp32 matmul. Only use it when AVX-512 is available.
if torch.__version__ >= (2, 6) and _has_avx512:

    @register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
    def _(A: torch.Tensor, B: torch.Tensor):
        return torch._int_mm(
            A.reshape(-1, A.shape[-1]),
            B.t(),
        ).reshape(*A.shape[:-1], B.shape[0])


if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):

    @register_kernel("bitsandbytes::quantize_blockwise", "cpu")
    def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
        A = A.contiguous()
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
        A = A.contiguous()
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
        # Fallback as AVX512 implementation has accuracy issues with blocksize >= 2048.
        # Note: this is not a common use case.
        avx512_fallback = _has_avx512 and blocksize >= 2048

        # Odd shape is not supported by this kernel; fallback to generic implementation
        shape_fallback = shape[-1] % 2 != 0

        if avx512_fallback or shape_fallback:
            from ..default.ops import _dequantize_4bit_compute
            from ..utils import _get_4bit_code

            if A.dtype != torch.uint8:
                A = A.view(torch.uint8)
            code = _get_4bit_code(quant_type, A.device)
            return _dequantize_4bit_compute(A.reshape(-1), absmax, code, blocksize, shape, dtype)

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

            gemm_4bit_forward_kernel = get_kernel(
                "kernels-community/quantization-bitsandbytes", version=1
            ).gemm_4bit_forward
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
            if B.dtype != torch.uint8:
                B = B.contiguous().view(torch.uint8)
            dtype = A.dtype
            quant_type = "fp4" if code[1] > 0 else "nf4"
            # cpu fused op only support bf16 for now.
            if dtype != torch.bfloat16:
                A = A.to(torch.bfloat16)
            if absmax.dtype != torch.bfloat16:
                absmax = absmax.to(torch.bfloat16)

            final_out_shape = (*A.shape[:-1], shapeB[0])
            A = A.reshape(-1, A.shape[-1])
            out_shape = (*A.shape[:-1], shapeB[0])
            if gemm_4bit_forward_kernel is not None:
                quant_type_num = 1 if quant_type == "fp4" else 0
                # C++ kernel expects weight shape (N, K_packed), ensure 2D contiguous
                B_2d = B.reshape(shapeB[0], -1).contiguous()
                out = gemm_4bit_forward_kernel(A, B_2d, absmax, blocksize, quant_type_num)
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
    update_mask = g != 0 if skip_zeros else None
    if update_mask is not None and not torch.any(update_mask):
        if unorm_vec is not None:
            unorm_vec.zero_()
        return

    if update_mask is None:
        g_float = g.float() * gnorm_scale
        p_float = p.data.float()
    else:
        g_float = g[update_mask].float() * gnorm_scale
        p_float = p.data[update_mask].float()

    if optimizer_name in ("adam", "lamb"):
        # Adam / LAMB (2-state): m and v
        state1_active = state1 if update_mask is None else state1[update_mask]
        state2_active = state2 if update_mask is None else state2[update_mask]
        state1_active.mul_(beta1).add_(g_float, alpha=1.0 - beta1)
        state2_active.mul_(beta2).addcmul_(g_float, g_float, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)
        step_size = -lr * correction2 / correction1

        if weight_decay > 0.0:
            p_float.mul_(1.0 - lr * weight_decay)

        update = state1_active / (state2_active.sqrt() + eps * correction2)

        update_scale = _compute_update_norm_and_scale(update, unorm_vec, max_unorm, param_norm)
        p_float.add_(update, alpha=step_size * update_scale)

    elif optimizer_name == "ademamix":
        # AdEMAMix (2-state): state1 shape is (2, *p.shape), state1[0]=m1, state1[1]=m2
        if update_mask is None:
            m1 = state1[0]
            m2 = state1[1]
            nu = state2
        else:
            m1 = state1[0][update_mask]
            m2 = state1[1][update_mask]
            nu = state2[update_mask]

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
        state1_active = state1 if update_mask is None else state1[update_mask]
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float

        if step == 1:
            state1_active.copy_(g_wd)
        else:
            state1_active.mul_(beta1).add_(g_wd)

        update_scale = _compute_update_norm_and_scale(state1_active, unorm_vec, max_unorm, param_norm)
        p_float.add_(state1_active, alpha=-lr * update_scale)

    elif optimizer_name == "lion":
        # Lion (2-state sign update)
        state1_active = state1 if update_mask is None else state1[update_mask]
        if weight_decay > 0.0:
            p_float.mul_(1.0 - lr * weight_decay)

        update = state1_active.mul(beta1).add(g_float, alpha=1.0 - beta1)
        p_float.add_(update.sign(), alpha=-lr)

        state1_active.mul_(beta2).add_(g_float, alpha=1.0 - beta2)

    elif optimizer_name == "rmsprop":
        # RMSprop (1-state)
        state1_active = state1 if update_mask is None else state1[update_mask]
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float
        state1_active.mul_(beta1).addcmul_(g_wd, g_wd, value=1.0 - beta1)

        update = g_wd / (state1_active.sqrt() + eps)
        update_scale = _compute_update_norm_and_scale(update, unorm_vec, max_unorm, param_norm)
        p_float.add_(update, alpha=-lr * update_scale)

    elif optimizer_name == "adagrad":
        # Adagrad (1-state)
        state1_active = state1 if update_mask is None else state1[update_mask]
        g_wd = g_float.add(p_float, alpha=weight_decay) if weight_decay > 0.0 else g_float
        state1_active.addcmul_(g_wd, g_wd, value=1.0)

        update = g_wd / (state1_active.sqrt() + eps)
        p_float.add_(update, alpha=-lr)

    else:
        raise ValueError(f"Unsupported optimizer for CPU: {optimizer_name}")

    if update_mask is None:
        p.data.copy_(p_float)
        return

    # Advanced indexing requires the source and destination dtypes to match.
    # Optimizer math is carried out in fp32 even when the model parameter is
    # fp16/bf16, so cast only at the masked write-back boundary.
    p.data[update_mask] = p_float.to(dtype=p.dtype)
    if optimizer_name == "ademamix":
        state1[0][update_mask] = m1
        state1[1][update_mask] = m2
        state2[update_mask] = nu
    else:
        state1[update_mask] = state1_active
        if optimizer_name in ("adam", "lamb"):
            state2[update_mask] = state2_active


register_kernel("bitsandbytes::optimizer_update_32bit", "cpu")(_optimizer_update_32bit_cpu)


@torch.no_grad()
def _dequant_blockwise_fp32_direct(
    A_uint8: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    return torch.ops.bitsandbytes.dequantize_blockwise(A_uint8, absmax, code, blocksize, torch.float32)


def _quant_blockwise_fp32_direct(
    A_fp32: torch.Tensor, code: torch.Tensor, absmax_out: torch.Tensor, out_uint8: torch.Tensor, blocksize: int
) -> None:
    out, absmax = torch.ops.bitsandbytes.quantize_blockwise(A_fp32, code, blocksize)
    out_uint8.copy_(out)
    absmax_out.copy_(absmax)


def _quant_blockwise_fp32_masked(
    A_fp32: torch.Tensor,
    code: torch.Tensor,
    absmax_out: torch.Tensor,
    out_uint8: torch.Tensor,
    blocksize: int,
    active_blocks: torch.Tensor,
) -> None:
    """Quantize only blocks touched by a non-zero gradient.

    The blockwise representation stores one scale per block.  Re-quantizing
    an untouched block can therefore change its decoded values even when its
    optimizer state was not updated (the quantized codebook does not generally
    contain the exact old maximum).  ``skip_zeros`` promises that zero-gradient
    entries remain untouched, so retain the original bytes and scale for every
    fully inactive block while still using the normal quantizer for active
    blocks (including inactive entries that share an active block).
    """
    out, absmax = torch.ops.bitsandbytes.quantize_blockwise(A_fp32, code, blocksize)

    active_blocks = active_blocks.reshape(-1).to(device=out_uint8.device, dtype=torch.bool)
    if active_blocks.numel() != absmax_out.numel():
        raise ValueError(
            f"Expected one active-block flag per absmax entry, got "
            f"{active_blocks.numel()} flags and {absmax_out.numel()} scales"
        )

    flat_out = out_uint8.reshape(-1)
    block_ids = torch.arange(flat_out.numel(), device=flat_out.device) // blocksize
    active_elements = active_blocks[block_ids].reshape(out.shape)
    # ``copy_`` also handles a caller-provided non-contiguous state buffer,
    # whereas assigning through a flattened view would silently update only a
    # temporary copy in that case.
    out_uint8.copy_(torch.where(active_elements, out, out_uint8))
    absmax_out.copy_(torch.where(active_blocks, absmax, absmax_out))


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

    update_mask = g != 0 if skip_zeros else None
    if update_mask is not None and not torch.any(update_mask):
        return

    active_blocks = None
    if update_mask is not None:
        numel = update_mask.numel()
        num_blocks = (numel + blocksize - 1) // blocksize
        padded_mask = torch.nn.functional.pad(update_mask.reshape(-1), (0, num_blocks * blocksize - numel))
        active_blocks = padded_mask.reshape(num_blocks, blocksize).any(dim=-1)

    # Dequantize states
    if optimizer_name == "ademamix" and absmax1.ndim == 2:
        s1_1 = _dequant_blockwise_fp32_direct(state1[0], absmax1[0], qmap1, blocksize)
        s1_2 = _dequant_blockwise_fp32_direct(state1[1], absmax1[1], qmap1, blocksize)
        state1_fp32 = torch.stack([s1_1, s1_2])
    else:
        state1_fp32 = _dequant_blockwise_fp32_direct(state1, absmax1, qmap1, blocksize)

    state2_fp32 = None
    if state2 is not None and qmap2 is not None and absmax2 is not None:
        state2_fp32 = _dequant_blockwise_fp32_direct(state2, absmax2, qmap2, blocksize)

    if update_mask is None:
        grad = g.float() * gnorm_scale
        p_active = p.data.float()
    else:
        grad = g[update_mask].float() * gnorm_scale
        p_active = p.data[update_mask].float()

    if optimizer_name in ("adam", "lamb"):
        state1_active = state1_fp32 if update_mask is None else state1_fp32[update_mask]
        state2_active = state2_fp32 if update_mask is None else state2_fp32[update_mask]
        state1_active.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        state2_active.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)

        denom = (state2_active.sqrt() / correction2).add_(eps)
        if weight_decay > 0.0:
            p_active.mul_(1.0 - lr * weight_decay)
        p_active.addcdiv_(state1_active, denom, value=-lr / correction1)

    elif optimizer_name == "ademamix":
        if update_mask is None:
            m1_active, m2_active = state1_fp32[0], state1_fp32[1]
            nu_active = state2_fp32
        else:
            m1_active = state1_fp32[0][update_mask]
            m2_active = state1_fp32[1][update_mask]
            nu_active = state2_fp32[update_mask]

        m1_active.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        m2_active.mul_(beta3).add_(grad, alpha=1.0 - beta3)
        nu_active.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        correction1 = 1.0 - beta1**step
        correction2 = math.sqrt(1.0 - beta2**step)

        update = (m1_active / correction1 + alpha * m2_active) / (nu_active.sqrt() / correction2 + eps)
        if weight_decay > 0.0:
            p_active.mul_(1.0 - lr * weight_decay)
        p_active.add_(update, alpha=-lr)

    elif optimizer_name in ("momentum", "lars"):
        state1_active = state1_fp32 if update_mask is None else state1_fp32[update_mask]
        grad.add_(p_active, alpha=weight_decay)
        if step == 1:
            state1_active.copy_(grad)
        else:
            state1_active.mul_(beta1).add_(grad)
        p_active.add_(state1_active, alpha=-lr)

    elif optimizer_name == "lion":
        state1_active = state1_fp32 if update_mask is None else state1_fp32[update_mask]
        if weight_decay > 0.0:
            p_active.mul_(1.0 - lr * weight_decay)

        update_dir = torch.sign(state1_active.mul(beta1) + grad.mul(1.0 - beta1))
        p_active.add_(update_dir, alpha=-lr)

        state1_active.mul_(beta2).add_(grad, alpha=1.0 - beta2)

    elif optimizer_name == "rmsprop":
        state1_active = state1_fp32 if update_mask is None else state1_fp32[update_mask]
        grad.add_(p_active, alpha=weight_decay)
        state1_active.mul_(beta1).addcmul_(grad, grad, value=1.0 - beta1)
        p_active.addcdiv_(grad, state1_active.sqrt().add_(eps), value=-lr)

    elif optimizer_name == "adagrad":
        state1_active = state1_fp32 if update_mask is None else state1_fp32[update_mask]
        grad.add_(p_active, alpha=weight_decay)
        state1_active.addcmul_(grad, grad, value=1.0)
        p_active.addcdiv_(grad, state1_active.sqrt().add_(eps), value=-lr)

    else:
        raise ValueError(f"Unsupported optimizer for CPU 8-bit: {optimizer_name}")

    if update_mask is None:
        p.data.copy_(p_active)
    else:
        # Keep the optimizer computation in fp32, then restore the parameter's
        # original dtype for the indexed write-back (fp16/bf16 are supported).
        p.data[update_mask] = p_active.to(dtype=p.dtype)
        if optimizer_name == "ademamix":
            state1_fp32[0][update_mask] = m1_active
            state1_fp32[1][update_mask] = m2_active
            state2_fp32[update_mask] = nu_active
        else:
            state1_fp32[update_mask] = state1_active
            if optimizer_name in ("adam", "lamb"):
                state2_fp32[update_mask] = state2_active

    # Re-quantize states
    if optimizer_name == "ademamix":
        quantize_fn = _quant_blockwise_fp32_direct if active_blocks is None else _quant_blockwise_fp32_masked
        if active_blocks is None:
            quantize_fn(state1_fp32[0], qmap1, absmax1[0], state1[0], blocksize)
            quantize_fn(state1_fp32[1], qmap1, absmax1[1], state1[1], blocksize)
            quantize_fn(state2_fp32, qmap2, absmax2, state2, blocksize)
        else:
            quantize_fn(state1_fp32[0], qmap1, absmax1[0], state1[0], blocksize, active_blocks)
            quantize_fn(state1_fp32[1], qmap1, absmax1[1], state1[1], blocksize, active_blocks)
            quantize_fn(state2_fp32, qmap2, absmax2, state2, blocksize, active_blocks)
    else:
        if active_blocks is None:
            _quant_blockwise_fp32_direct(state1_fp32, qmap1, absmax1, state1, blocksize)
        else:
            _quant_blockwise_fp32_masked(state1_fp32, qmap1, absmax1, state1, blocksize, active_blocks)
        if state2_fp32 is not None:
            if active_blocks is None:
                _quant_blockwise_fp32_direct(state2_fp32, qmap2, absmax2, state2, blocksize)
            else:
                _quant_blockwise_fp32_masked(state2_fp32, qmap2, absmax2, state2, blocksize, active_blocks)


register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "cpu")(_optimizer_update_8bit_blockwise_cpu)
