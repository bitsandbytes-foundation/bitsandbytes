"""
MPS backend for bitsandbytes - Optimized Metal kernels for Apple Silicon

This backend provides GPU-accelerated quantization operations using Metal shaders,
offering significant performance improvements over the default PyTorch fallback.

Requires: mps-bitsandbytes (pip install mps-bitsandbytes)
"""

from collections.abc import Sequence
from typing import Optional

import torch

from ..._ops import register_kernel
from ..utils import CODE

# Try to import mps-bitsandbytes for Metal kernels
try:
    import mps_bitsandbytes as mps_bnb
    from mps_bitsandbytes import _C as mps_lib
    HAS_MPS_BITSANDBYTES = True
except ImportError:
    HAS_MPS_BITSANDBYTES = False
    mps_bnb = None
    mps_lib = None


if not HAS_MPS_BITSANDBYTES:
    # If mps-bitsandbytes is not installed, don't register any kernels
    # The default backend will be used instead
    pass
else:
    # ========================================================================
    # 4-bit Quantization (NF4/FP4)
    # ========================================================================

    @register_kernel("bitsandbytes::quantize_4bit", "mps")
    def _(
        A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check_is_size(blocksize)
        torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4, got {quant_type}")

        # Flatten to 1D to avoid per-row padding overhead
        A_flat = A.flatten()

        if quant_type == "nf4":
            packed, quant_state = mps_bnb.quantize_nf4(A_flat, blocksize=blocksize)
        else:
            packed, quant_state = mps_bnb.quantize_fp4(A_flat, blocksize=blocksize)

        # Extract absmax from QuantState
        absmax = quant_state.absmax

        # Handle quant_storage dtype conversion if needed
        if quant_storage != torch.uint8:
            packed = packed.view(quant_storage)

        # Match CUDA layout: (N, 1) 2D tensor for transpose logic in functional.py
        packed = packed.reshape(-1, 1)

        return packed, absmax

    @register_kernel("bitsandbytes::dequantize_4bit", "mps")
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

        # Handle non-uint8 storage
        if A.dtype != torch.uint8:
            A = A.view(torch.uint8)

        if quant_type == "nf4":
            out = mps_bnb.dequantize_nf4(A, absmax=absmax, blocksize=blocksize)
        else:
            out = mps_bnb.dequantize_fp4(A, absmax=absmax, blocksize=blocksize)

        # Truncate block padding and reshape
        from math import prod
        out = out.flatten()[:prod(shape)].reshape(shape)

        return out.to(dtype)

    @register_kernel("bitsandbytes::gemv_4bit", "mps")
    def _(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB: Sequence[int],
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
    ) -> torch.Tensor:
        # Determine quant_type from code
        quant_type = "fp4" if code[1] > 0 else "nf4"

        # Use dequantize + matmul path (same as MatMul4Bit.apply) for numerical consistency
        # This ensures gemv_4bit and matmul_4bit produce identical results
        B_dequant = torch.ops.bitsandbytes.dequantize_4bit(
            B, absmax, blocksize, quant_type, shapeB, A.dtype
        )
        return torch.nn.functional.linear(A, B_dequant)

    # ========================================================================
    # Blockwise 8-bit Quantization (Dynamic Codebook)
    # ========================================================================

    @register_kernel("bitsandbytes::quantize_blockwise", "mps")
    def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor using dynamic codebook (NOT linear INT8).

        The dynamic codebook has 256 values optimized for optimizer states.
        Values are normalized by blockwise absmax to [-1, 1], then mapped
        to the nearest codebook entry via argmin.
        """
        torch._check_is_size(blocksize)

        n = A.numel()
        rem = n % blocksize
        has_rem = rem > 0
        blocks = n // blocksize + has_rem
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
        A_reshaped = A.reshape(n)
        A_com = A_reshaped[: n - rem]
        A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
        absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]

        # Clamp absmax to avoid division by zero
        absmax_clamped = absmax.clamp(min=1e-8)

        # Scale values to [-1, 1]
        scaled_A = torch.clamp(A_com_reshaped * (1 / absmax_clamped[: blocks - has_rem].reshape(-1, 1)), -1, 1)
        scaled_A = scaled_A.reshape(-1)

        if has_rem:
            absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
            absmax_rem = absmax[-1].clamp(min=1e-8)
            scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax_rem), -1, 1)
            scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

        # Find nearest codebook entry
        code_mps = code.to(A.device)
        diff = torch.abs(scaled_A.unsqueeze(-1) - code_mps)
        out = torch.argmin(diff, dim=-1).to(torch.uint8).reshape(A.shape)

        return out, absmax

    @register_kernel("bitsandbytes::dequantize_blockwise", "mps")
    def _(
        A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Dequantize tensor using dynamic codebook lookup.

        Maps quantized indices back to codebook values, then scales by blockwise absmax.
        """
        torch._check_is_size(blocksize)
        torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")

        code_mps = code.to(A.device)
        out = code_mps[A.reshape(-1).int()]

        blocks = out.shape[-1] // blocksize
        res = out.shape[-1] % blocksize

        if res != 0:
            out = torch.nn.functional.pad(out, (0, blocksize - res), mode="constant", value=0)

        out = (out.reshape(-1, blocksize) * absmax.reshape(-1, 1)).to(dtype).reshape(-1)
        out = out[: blocks * blocksize + res]
        out = out.reshape(A.shape)

        return out

    # ========================================================================
    # INT8 Linear Operations
    # ========================================================================

    @register_kernel("bitsandbytes::int8_linear_matmul", "mps")
    def _(A: torch.Tensor, B: torch.Tensor):
        # int8 matmul: A @ B.T -> int32
        return torch.matmul(A.float(), B.float().t()).to(torch.int32)

    @register_kernel("bitsandbytes::int8_mm_dequant", "mps")
    def _(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")

        A_calc = A.reshape(-1, A.shape[-1])
        row_stats = row_stats.reshape(-1).unsqueeze(-1)
        col_stats = col_stats.reshape(-1).unsqueeze(0)

        # Dequantize: scale by row_stats * col_stats * (1/127^2)
        out = A_calc.float() * (row_stats * col_stats) * 6.200124e-05

        if bias is not None:
            out = out + bias

        return out.to(dtype or torch.float16)

    @register_kernel("bitsandbytes::int8_vectorwise_quant", "mps")
    def _(A: torch.Tensor, threshold=0.0):
        from math import prod
        rows = prod(A.shape[:-1])
        outlier_cols = None
        outlier_restore = None

        if threshold > 0.0:
            # Handle outliers - zero them BEFORE computing absmax (matches default backend)
            outliers = A.abs() >= threshold
            if outliers.any():
                outlier_cols = torch.argwhere(outliers.any(dim=0)).reshape(-1)
                outlier_restore = A[outliers].clone()
                A[outliers] = 0
            else:
                outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

        out_row, row_stats = mps_bnb.quantize_rowwise(A)

        # Zero out values from outlier columns across all rows
        if rows > 1 and outlier_cols is not None:
            out_row[:, outlier_cols] = 0

        # Restore outliers in A
        if outlier_restore is not None:
            A[outliers] = outlier_restore

        return out_row, row_stats, outlier_cols

    @register_kernel("bitsandbytes::int8_vectorwise_dequant", "mps")
    def _(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return mps_bnb.dequantize_rowwise(A, stats)

    @register_kernel("bitsandbytes::int8_scaled_mm", "mps")
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
            out_i32, row_stats, col_stats, dtype=dtype or torch.float16, bias=bias
        )

    # ========================================================================
    # 8-bit Optimizers
    # ========================================================================

    @register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "mps")
    def _(
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
        """8-bit optimizer update kernel for MPS."""
        blocksize = 256  # Standard blocksize for optimizer states

        # Dequantize states using torch.ops (consistent with their F.dequantize_blockwise)
        m = torch.ops.bitsandbytes.dequantize_blockwise(state1, absmax1, qmap1, blocksize, torch.float32)
        if state2 is not None and absmax2 is not None and qmap2 is not None:
            v = torch.ops.bitsandbytes.dequantize_blockwise(state2, absmax2, qmap2, blocksize, torch.float32)
        else:
            v = None

        # Apply gradient scaling
        grad = g.float()
        if gnorm_scale != 1.0:
            grad = grad * gnorm_scale

        # Skip zeros if requested
        if skip_zeros:
            mask = grad != 0
        else:
            mask = None

        # Optimizer-specific update
        if optimizer_name == "adam":
            # Adam update
            if weight_decay > 0:
                grad = grad + weight_decay * p.float()
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            update = m / denom
            if mask is not None:
                update = update * mask
            p.add_(update.to(p.dtype), alpha=-step_size)

        elif optimizer_name == "momentum":
            # SGD with momentum
            if weight_decay > 0:
                grad = grad + weight_decay * p.float()
            m.mul_(beta1).add_(grad)
            update = m
            if mask is not None:
                update = update * mask
            p.add_(update.to(p.dtype), alpha=-lr)

        elif optimizer_name == "rmsprop":
            # RMSprop update - 1-state optimizer, state1 is variance (v), state2 is None
            # m holds the variance for rmsprop
            # RMSprop uses beta1 (=alpha) for smoothing, not beta2
            if weight_decay > 0:
                grad = grad + weight_decay * p.float()
            m.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)
            denom = m.sqrt().add_(eps)
            update = grad / denom
            if mask is not None:
                update = update * mask
            p.add_(update.to(p.dtype), alpha=-lr)

        elif optimizer_name == "lion":
            # Lion update: sign(beta1 * m + (1-beta1) * g)
            if weight_decay > 0:
                p.mul_(1 - lr * weight_decay)
            update = (beta1 * m + (1 - beta1) * grad).sign_()
            if mask is not None:
                update = update * mask
            p.add_(update.to(p.dtype), alpha=-lr)
            # Update momentum for next step
            m.mul_(beta2).add_(grad, alpha=1 - beta2)

        elif optimizer_name == "ademamix":
            # AdEMAMix: uses m1 (fast EMA), m2 (slow EMA), and nu (second moment)
            # state1 is shape (2, N) containing [m1, m2]
            # absmax1 is shape (2, blocks) - separate absmax for each buffer
            # state2 is nu (second moment), absmax2 is (blocks,)
            # beta1 -> m1, beta2 -> nu, beta3 -> m2, alpha -> blend factor

            # Dequantize m1 and m2 separately (they have separate absmax rows)
            m1 = torch.ops.bitsandbytes.dequantize_blockwise(
                state1[0], absmax1[0], qmap1, blocksize, torch.float32
            )
            m2 = torch.ops.bitsandbytes.dequantize_blockwise(
                state1[1], absmax1[1], qmap1, blocksize, torch.float32
            )
            nu = v  # v is state2 (second moment), already dequantized

            # Update EMAs
            m1.mul_(beta1).add_(grad, alpha=1 - beta1)
            m2.mul_(beta3).add_(grad, alpha=1 - beta3)
            nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Compute update: (m1/bias_correction1 + alpha * m2) / denom
            denom = (nu.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            update = (m1 / bias_correction1 + alpha * m2) / denom

            # Add weight decay (decoupled)
            update.add_(p.float(), alpha=weight_decay)

            if mask is not None:
                update = update * mask
            p.add_(update.to(p.dtype), alpha=-lr)

            # Requantize m1 and m2 separately back to state1
            new_state1_0, new_absmax1_0 = torch.ops.bitsandbytes.quantize_blockwise(m1, qmap1, blocksize)
            new_state1_1, new_absmax1_1 = torch.ops.bitsandbytes.quantize_blockwise(m2, qmap1, blocksize)
            state1[0].copy_(new_state1_0)
            state1[1].copy_(new_state1_1)
            absmax1[0].copy_(new_absmax1_0)
            absmax1[1].copy_(new_absmax1_1)

            # Requantize nu
            new_state2, new_absmax2 = torch.ops.bitsandbytes.quantize_blockwise(nu, qmap2, blocksize)
            state2.copy_(new_state2)
            absmax2.copy_(new_absmax2)
            return  # Early return since we handled requantization

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Requantize states in-place using torch.ops (returns tensor, absmax)
        new_state1, new_absmax1 = torch.ops.bitsandbytes.quantize_blockwise(m, qmap1, blocksize)
        state1.copy_(new_state1)
        absmax1.copy_(new_absmax1)

        if v is not None and state2 is not None and absmax2 is not None:
            new_state2, new_absmax2 = torch.ops.bitsandbytes.quantize_blockwise(v, qmap2, blocksize)
            state2.copy_(new_state2)
            absmax2.copy_(new_absmax2)
