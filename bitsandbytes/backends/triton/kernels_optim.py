import math
from typing import Optional

import torch

import triton
import triton.language as tl

# from triton.language.extra import libdevice
from .kernels_8bit_quant import (
    dequant_8bit_blockwise,
    dequant_8bit_kernel_util,
    quantize_8bit_blockwise_core,
    quantize_blockwise_triton,
)

###########################################
# Pure torch implementation for reference #
###########################################


@torch.compile
def _dequantize_blockwise_pytorch(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for block-wise dequantization.
    """
    if A.numel() == 0:
        return torch.empty_like(A, dtype=dtype)

    A_flat = A.flatten()
    num_elements = A_flat.numel()

    dequantized_flat = code.to(A.device)[A_flat.long()].to(dtype)

    num_blocks = math.ceil(num_elements / blocksize)
    pad_len = num_blocks * blocksize - num_elements
    if pad_len > 0:
        dequantized_flat = torch.nn.functional.pad(dequantized_flat, (0, pad_len))

    dequantized_blocks = dequantized_flat.reshape(num_blocks, blocksize)

    rescaled_blocks = dequantized_blocks * absmax.unsqueeze(1).to(dtype)

    rescaled_flat = rescaled_blocks.flatten()
    if pad_len > 0:
        rescaled_flat = rescaled_flat[:-pad_len]

    return rescaled_flat.reshape(A.shape)


@torch.compile
def _quantize_blockwise_pytorch(
    A: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference implementation for block-wise quantization.
    """
    if A.numel() == 0:
        return torch.empty_like(A, dtype=torch.uint8), torch.empty(0, dtype=torch.float32, device=A.device)

    A_flat = A.flatten()
    num_elements = A_flat.numel()

    num_blocks = math.ceil(num_elements / blocksize)

    pad_len = num_blocks * blocksize - num_elements
    if pad_len > 0:
        A_flat = torch.nn.functional.pad(A_flat, (0, pad_len))

    A_blocks = A_flat.reshape(num_blocks, blocksize)

    absmax = torch.max(torch.abs(A_blocks), dim=1, keepdim=True)[0]
    absmax[absmax == 0] = 1.0

    scaled_blocks = A_blocks / absmax

    # Inefficient but straightforward quantization, takes a lot of memory
    diff = torch.abs(scaled_blocks.unsqueeze(2) - code.to(A.device))
    quantized_indices = torch.argmin(diff, dim=2).to(torch.uint8)

    quantized_flat = quantized_indices.flatten()
    if pad_len > 0:
        quantized_flat = quantized_flat[:-pad_len]

    return quantized_flat.reshape(A.shape), absmax.flatten()


# Main updated function
def optimizer_update_8bit_blockwise_pytorch(
    p: torch.Tensor,
    g: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,  # ADEMIX
    alpha: float,  # ADEMIX
    eps: float,
    step: int,
    lr: float,
    qmap1: torch.Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: torch.Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float,
    gnorm_scale: float,
    skip_zeros: bool,
    # ADEMIX
    *,
    optimizer_name: str,
) -> None:
    """
    Pure PyTorch implementation of the 8-bit block-wise optimizer update step.
    This version ensures high-precision updates for float16 parameters.
    """
    if skip_zeros:
        raise ValueError("skip_zeros is not supported on XPU yet.")

    blocksize = 256

    with torch.no_grad():
        # Dequantize states to perform updates in 32-bit precision
        if optimizer_name == "ademamix" and absmax1.ndim == 2:
            # For AdEMAMix, state1 holds two EMAs, so absmax1 is stacked.
            s1_1_fp32 = _dequantize_blockwise_pytorch(state1[0], absmax1[0], qmap1, blocksize, torch.float32)
            s1_2_fp32 = _dequantize_blockwise_pytorch(state1[1], absmax1[1], qmap1, blocksize, torch.float32)
            state1_fp32 = torch.stack([s1_1_fp32, s1_2_fp32])
        else:
            state1_fp32 = _dequantize_blockwise_pytorch(state1, absmax1, qmap1, blocksize, torch.float32)

        state2_fp32 = None
        if state2 is not None:
            state2_fp32 = _dequantize_blockwise_pytorch(state2, absmax2, qmap2, blocksize, torch.float32)

        grad = g.float() * gnorm_scale

        # Create a 32-bit copy of the parameter for high-precision updates
        p_fp32 = p.data.float()

        if optimizer_name == "adam":
            state1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            state2_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            denom = (state2_fp32.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)
            p_fp32.addcdiv_(state1_fp32, denom, value=-lr / bias_correction1)

        elif optimizer_name == "ademamix":
            m1_fp32, m2_fp32 = state1_fp32[0], state1_fp32[1]
            nu_fp32 = state2_fp32

            m1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            m2_fp32.mul_(beta3).add_(grad, alpha=1.0 - beta3)
            nu_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = math.sqrt(1.0 - beta2**step)

            update = (m1_fp32 / bias_correction1 + alpha * m2_fp32) / (nu_fp32.sqrt() / bias_correction2 + eps)

            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)

            p_fp32.add_(update, alpha=-lr)
            state1_fp32 = torch.stack([m1_fp32, m2_fp32])

        elif optimizer_name == "momentum":
            grad.add_(p_fp32, alpha=weight_decay)
            if step == 1:
                state1_fp32.copy_(grad)
            else:
                state1_fp32.mul_(beta1).add_(grad)
            p_fp32.add_(state1_fp32, alpha=-lr)

        elif optimizer_name == "rmsprop":
            grad.add_(p_fp32, alpha=weight_decay)
            state1_fp32.mul_(beta1).addcmul_(grad, grad, value=1.0 - beta1)
            p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

        elif optimizer_name == "lion":
            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)

            update_dir = torch.sign(state1_fp32.mul(beta1) + grad.mul(1.0 - beta1))
            p_fp32.add_(update_dir, alpha=-lr)

            state1_fp32.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        elif optimizer_name == "adagrad":
            grad.add_(p_fp32, alpha=weight_decay)
            state1_fp32.addcmul_(grad, grad, value=1.0)
            p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

        else:
            raise NotImplementedError(
                f"Pure PyTorch implementation for optimizer '{optimizer_name}' is not available."
            )

        # Copy the updated 32-bit parameter back to the original tensor
        p.data.copy_(p_fp32)

        # Re-quantize states and update state tensors in-place
        if optimizer_name == "ademamix":
            new_m1_8bit, new_absmax_m1 = _quantize_blockwise_pytorch(state1_fp32[0], qmap1, blocksize)
            new_m2_8bit, new_absmax_m2 = _quantize_blockwise_pytorch(state1_fp32[1], qmap1, blocksize)
            state1[0].copy_(new_m1_8bit)
            state1[1].copy_(new_m2_8bit)
            absmax1[0].copy_(new_absmax_m1)
            absmax1[1].copy_(new_absmax_m2)

            new_state2_8bit, new_absmax2 = _quantize_blockwise_pytorch(state2_fp32, qmap2, blocksize)
            state2.copy_(new_state2_8bit)
            absmax2.copy_(new_absmax2)
        else:
            new_state1_8bit, new_absmax1 = _quantize_blockwise_pytorch(state1_fp32, qmap1, blocksize)
            state1.copy_(new_state1_8bit)
            absmax1.copy_(new_absmax1)

            if state2_fp32 is not None:
                new_state2_8bit, new_absmax2 = _quantize_blockwise_pytorch(state2_fp32, qmap2, blocksize)
                state2.copy_(new_state2_8bit)
                absmax2.copy_(new_absmax2)


#######################################
# Mixed torch + triton implementation #
#######################################


# Much more memory efficient due to using triton for quantization/dequantization
def optimizer_update_8bit_blockwise_triton_quant(
    p: torch.Tensor,
    g: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,  # ADEMIX
    alpha: float,  # ADEMIX
    eps: float,
    step: int,
    lr: float,
    qmap1: torch.Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: torch.Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float,
    gnorm_scale: float,
    skip_zeros: bool,
    # ADEMIX
    *,
    optimizer_name: str,
) -> None:
    """
    Pure PyTorch implementation of the 8-bit block-wise optimizer update step.
    This version ensures high-precision updates for float16 parameters.
    """
    if skip_zeros and not torch.any(g):
        return

    blocksize = 256
    grad = g.float() * gnorm_scale

    with torch.no_grad():
        # Create a 32-bit copy of the parameter for high-precision updates
        p_fp32 = p.data.float()

        # Dequantize states to perform updates in 32-bit precision
        if optimizer_name == "ademamix" and absmax1.ndim == 2:
            # For AdEMAMix, state1 holds two EMAs, so absmax1 is stacked.
            s1_1_fp32 = dequant_8bit_blockwise(state1[0], absmax1[0], qmap1, blocksize, dtype=torch.float32)
            s1_2_fp32 = dequant_8bit_blockwise(state1[1], absmax1[1], qmap1, blocksize, dtype=torch.float32)
            state1_fp32 = torch.stack([s1_1_fp32, s1_2_fp32])
        else:
            state1_fp32 = dequant_8bit_blockwise(state1, absmax1, qmap1, blocksize, dtype=torch.float32)

        state2_fp32 = None
        if state2 is not None:
            state2_fp32 = dequant_8bit_blockwise(state2, absmax2, qmap2, blocksize, dtype=torch.float32)

        # Apply optimizer-specific update logic
        if optimizer_name == "adam":
            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)

            state1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            state2_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            denom = (state2_fp32.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p_fp32.addcdiv_(state1_fp32, denom, value=-lr / bias_correction1)

        elif optimizer_name == "ademamix":
            m1_fp32, m2_fp32 = state1_fp32[0], state1_fp32[1]
            nu_fp32 = state2_fp32

            m1_fp32.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            m2_fp32.mul_(beta3).add_(grad, alpha=1.0 - beta3)
            nu_fp32.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = math.sqrt(1.0 - beta2**step)

            update = (m1_fp32 / bias_correction1 + alpha * m2_fp32) / (nu_fp32.sqrt() / bias_correction2 + eps)

            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)

            p_fp32.add_(update, alpha=-lr)
            state1_fp32 = torch.stack([m1_fp32, m2_fp32])

        elif optimizer_name == "momentum":
            grad.add_(p_fp32, alpha=weight_decay)
            if step == 1:
                state1_fp32.copy_(grad)
            else:
                state1_fp32.mul_(beta1).add_(grad)
            p_fp32.add_(state1_fp32, alpha=-lr)

        elif optimizer_name == "rmsprop":
            grad.add_(p_fp32, alpha=weight_decay)
            state1_fp32.mul_(beta1).addcmul_(grad, grad, value=1.0 - beta1)
            p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

        elif optimizer_name == "lion":
            if weight_decay > 0.0:
                p_fp32.mul_(1.0 - lr * weight_decay)

            update_dir = torch.sign(state1_fp32.mul(beta1) + grad.mul(1.0 - beta1))
            p_fp32.add_(update_dir, alpha=-lr)

            state1_fp32.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        elif optimizer_name == "adagrad":
            grad.add_(p_fp32, alpha=weight_decay)
            state1_fp32.addcmul_(grad, grad, value=1.0)
            p_fp32.addcdiv_(grad, state1_fp32.sqrt().add_(eps), value=-lr)

        else:
            raise NotImplementedError(
                f"Pure PyTorch implementation for optimizer '{optimizer_name}' is not available."
            )

        # Copy the updated 32-bit parameter back to the original tensor
        p.data.copy_(p_fp32)

        # Re-quantize states and update state tensors in-place
        if optimizer_name == "ademamix":
            new_m1_8bit, new_absmax_m1 = quantize_blockwise_triton(state1_fp32[0], qmap1, blocksize)
            new_m2_8bit, new_absmax_m2 = quantize_blockwise_triton(state1_fp32[1], qmap1, blocksize)
            state1[0].copy_(new_m1_8bit)
            state1[1].copy_(new_m2_8bit)
            absmax1[0].copy_(new_absmax_m1)
            absmax1[1].copy_(new_absmax_m2)

            new_state2_8bit, new_absmax2 = quantize_blockwise_triton(state2_fp32, qmap2, blocksize)
            state2.copy_(new_state2_8bit)
            absmax2.copy_(new_absmax2)
        else:
            new_state1_8bit, new_absmax1 = quantize_blockwise_triton(state1_fp32, qmap1, blocksize)
            state1.copy_(new_state1_8bit)
            absmax1.copy_(new_absmax1)

            if state2_fp32 is not None:
                new_state2_8bit, new_absmax2 = quantize_blockwise_triton(state2_fp32, qmap2, blocksize)
                state2.copy_(new_state2_8bit)
                absmax2.copy_(new_absmax2)


#########################
# Triton implementation #
#########################

MOMENTUM = 0
RMSPROP = 1
ADAGRAD = 2
ADAM = 3
# LION should be larger than MOMENTUM, RMSPROP, ADAGRAD due to comparison in kernels
LION = 4
ADEMAMIX = 5

name2optimizer_id = {
    "momentum": MOMENTUM,
    "rmsprop": RMSPROP,
    "adagrad": ADAGRAD,
    "adam": ADAM,
    "lion": LION,
    "ademamix": ADEMAMIX,
}


@triton.jit
def _optimizer_update_1state_8bit_blockwise_triton_kernel(
    # Tensors
    p_ptr,
    g_ptr,
    state1_ptr,
    state2_ptr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    beta3,
    alpha,
    eps: tl.constexpr,
    step,
    beta1_step,
    beta2_step,
    lr,
    qmap1_ptr,
    qmap2_ptr,
    absmax1_ptr,
    absmax2_ptr,
    weight_decay,
    gnorm_scale,
    # Meta-parameters
    n_elements,
    BLOCK_SIZE_N: tl.constexpr,
    N_PER_TH: tl.constexpr,
    OPTIMIZER_ID: tl.constexpr,
):
    """
    Triton kernel for 8-bit optimizers that use one momentum state.
    Supports: Momentum, RMSprop, Adagrad, Lion.
    """
    # 1. Boilerplate: pid, offsets, mask
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N * N_PER_TH)
    mask = offsets < n_elements

    # 2. Load and dequantize tensors
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * gnorm_scale
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s1 = dequant_8bit_kernel_util(state1_ptr, offsets, qmap1_ptr, absmax1_ptr, mask, BLOCK_SIZE_N)

    # 3. Optimizer-specific updates
    # LION
    if weight_decay > 0.0 and OPTIMIZER_ID == 2:
        p *= 1.0 - lr * weight_decay
    # Apply weight decay for momentum, rmsprop, adagrad
    elif weight_decay > 0.0:
        g += p * weight_decay

    # Momentum update
    if OPTIMIZER_ID == 0:  # MOMENTUM
        if step == 1:
            s1 = g
        else:
            s1 = s1 * beta1 + g
        p -= lr * s1

    # RMSprop update
    elif OPTIMIZER_ID == 1:  # RMSPROP
        s1 = s1 * beta1 + (1.0 - beta1) * g * g
        p -= lr * (g / (tl.sqrt(s1) + eps))

    # Adagrad update
    elif OPTIMIZER_ID == 2:  # ADAGRAD
        s1 += g * g
        p -= lr * (g / (tl.sqrt(s1) + eps))

    # Lion update
    elif OPTIMIZER_ID == 4:  # LION
        val = s1 * beta1 + (1.0 - beta1) * g
        update = tl.where(val > 0.0, 1.0, tl.where(val < 0.0, -1.0, 0.0))
        p -= lr * update
        s1 = s1 * beta2 + (1.0 - beta2) * g

    # 4. Store updated parameter and requantized state
    tl.store(p_ptr + offsets, p.to(p_ptr.dtype.element_ty), mask=mask)
    s1_codes, new_absmax1 = quantize_8bit_blockwise_core(s1, qmap1_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
    tl.store(state1_ptr + offsets, s1_codes, mask=mask)
    tl.store(absmax1_ptr + block_start_idx + tl.arange(0, N_PER_TH), new_absmax1)


@triton.jit
def _optimizer_update_2state_8bit_blockwise_triton_kernel(
    # Tensors
    p_ptr,
    g_ptr,
    state1_ptr,
    state2_ptr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    # ademamix changes alpha and beta3
    beta3,
    # ademamix changes alpha and beta3
    alpha,
    eps: tl.constexpr,
    step,
    beta1_step,
    beta2_step,
    lr,
    qmap1_ptr,
    qmap2_ptr,
    absmax1_ptr,
    absmax2_ptr,
    weight_decay: tl.constexpr,
    gnorm_scale: tl.constexpr,
    # Meta-parameters
    n_elements,
    BLOCK_SIZE_N: tl.constexpr,
    N_PER_TH: tl.constexpr,
    OPTIMIZER_ID: tl.constexpr,
):
    """
    Triton kernel for 8-bit optimizers that use two momentum states.
    Supports: Adam, AdEMAMix.
    """
    # 1. Boilerplate: pid, offsets, mask
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N * N_PER_TH)
    mask = offsets < n_elements

    # 2. Load and dequantize tensors
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * gnorm_scale
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # 3. Optimizer-specific updates
    if OPTIMIZER_ID == 3:  # ADAM
        s1 = dequant_8bit_kernel_util(state1_ptr, offsets, qmap1_ptr, absmax1_ptr, mask, BLOCK_SIZE_N)
        s2 = dequant_8bit_kernel_util(state2_ptr, offsets, qmap2_ptr, absmax2_ptr, mask, BLOCK_SIZE_N)

        s1 = s1 * beta1 + (1.0 - beta1) * g
        s2 = s2 * beta2 + (1.0 - beta2) * g * g

        # In torch=2.7 on XPU there is an issue with libdevice.pow, leading to an error.
        # For backwards compatibility we precompute the bias correction factors.
        # bias_correction1 = 1.0 - libdevice.pow(beta1, step)
        # bias_correction2 = 1.0 - libdevice.pow(beta2, step)
        bias_correction1 = 1.0 - beta1_step
        bias_correction2 = 1.0 - beta2_step

        if weight_decay > 0.0:
            p *= 1.0 - lr * weight_decay

        denom = tl.sqrt(s2) / tl.sqrt(bias_correction2) + eps
        p -= (lr / bias_correction1) * (s1 / denom)

        # Store updated parameter
        tl.store(p_ptr + offsets, p.to(p_ptr.dtype.element_ty), mask=mask)

        # Requantize and store states
        s1_codes, new_absmax1 = quantize_8bit_blockwise_core(s1, qmap1_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
        tl.store(state1_ptr + offsets, s1_codes, mask=mask)
        tl.store(absmax1_ptr + block_start_idx + tl.arange(0, N_PER_TH), new_absmax1)

        s2_codes, new_absmax2 = quantize_8bit_blockwise_core(s2, qmap2_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
        tl.store(state2_ptr + offsets, s2_codes, mask=mask)
        tl.store(absmax2_ptr + block_start_idx + tl.arange(0, N_PER_TH), new_absmax2)

    elif OPTIMIZER_ID == 5:  # ADEMAMIX
        # AdEMAMix has a stacked state1 (m1, m2) and state2 (nu)
        m1 = dequant_8bit_kernel_util(state1_ptr, offsets, qmap1_ptr, absmax1_ptr, mask, BLOCK_SIZE_N)
        m2 = dequant_8bit_kernel_util(
            state1_ptr + n_elements,
            offsets,
            qmap1_ptr,
            absmax1_ptr + n_elements // BLOCK_SIZE_N,
            mask,
            BLOCK_SIZE_N,
        )
        nu = dequant_8bit_kernel_util(state2_ptr, offsets, qmap2_ptr, absmax2_ptr, mask, BLOCK_SIZE_N)

        m1 = m1 * beta1 + (1.0 - beta1) * g
        m2 = m2 * beta3 + (1.0 - beta3) * g
        nu = nu * beta2 + (1.0 - beta2) * g * g

        # In torch=2.7 on XPU there is an issue with libdevice.pow, leading to an error.
        # For backwards compatibility we precompute the bias correction factors.
        # bias_correction1 = 1.0 - libdevice.pow(beta1, step)
        # bias_correction2 = tl.sqrt(1.0 - libdevice.pow(beta2, step))
        bias_correction1 = 1.0 - beta1_step
        bias_correction2 = tl.sqrt(1.0 - beta2_step)

        update = (m1 / bias_correction1 + alpha * m2) / (tl.sqrt(nu) / bias_correction2 + eps)

        if weight_decay > 0.0:
            p *= 1.0 - lr * weight_decay

        p -= lr * update

        # Store updated parameter
        tl.store(p_ptr + offsets, p.to(p_ptr.dtype.element_ty), mask=mask)

        # Requantize and store all three states
        m1_codes, new_absmax_m1 = quantize_8bit_blockwise_core(m1, qmap1_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
        tl.store(state1_ptr + offsets, m1_codes, mask=mask)
        tl.store(absmax1_ptr + block_start_idx + tl.arange(0, N_PER_TH), new_absmax_m1)

        m2_codes, new_absmax_m2 = quantize_8bit_blockwise_core(m2, qmap1_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
        tl.store(state1_ptr + n_elements + offsets, m2_codes, mask=mask)
        tl.store(
            absmax1_ptr + block_start_idx + tl.arange(0, N_PER_TH) + n_elements // BLOCK_SIZE_N,
            new_absmax_m2,
        )

        nu_codes, new_absmax_nu = quantize_8bit_blockwise_core(nu, qmap2_ptr, 256, BLOCK_SIZE_N, N_PER_TH)
        tl.store(state2_ptr + offsets, nu_codes, mask=mask)
        tl.store(absmax2_ptr + block_start_idx + tl.arange(0, N_PER_TH), new_absmax_nu)


name2optimizer_fn = {
    "momentum": _optimizer_update_1state_8bit_blockwise_triton_kernel,
    "rmsprop": _optimizer_update_1state_8bit_blockwise_triton_kernel,
    "adagrad": _optimizer_update_1state_8bit_blockwise_triton_kernel,
    "adam": _optimizer_update_2state_8bit_blockwise_triton_kernel,
    "lion": _optimizer_update_1state_8bit_blockwise_triton_kernel,
    "ademamix": _optimizer_update_2state_8bit_blockwise_triton_kernel,
}


def optimizer_update_8bit_blockwise_triton_impl(
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
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    if skip_zeros:
        raise NotImplementedError("skip_zeros is not supported on XPU yet")

    if optimizer_name == "ademamix":
        # Handle AdEMAMIX's stacked state tensors
        if state1.dim() < 2 or state1.shape[0] != 2:
            raise ValueError(
                f"For ademamix, state1 must be a stacked tensor of shape (2, ...), but got {state1.shape}"
            )
        if absmax1.dim() < 2 or absmax1.shape[0] != 2:
            raise ValueError(
                f"For ademamix, absmax1 must be a stacked tensor of shape (2, ...), but got {absmax1.shape}"
            )

    BLOCK_SIZE = 256
    N_PER_TH = 1  # Number of blocks processed per thread.
    grid = (triton.cdiv(p.numel(), BLOCK_SIZE * N_PER_TH),)
    fn = name2optimizer_fn[optimizer_name]
    optimizer_id = name2optimizer_id[optimizer_name]

    # In torch=2.7 on XPU there is an issue with libdevice.pow, leading to an error.
    # For backwards compatibility we precompute the bias correction factors.
    beta1_step = beta1**step
    beta2_step = beta2**step

    fn[grid](
        p,
        g,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        beta1_step,
        beta2_step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay,
        gnorm_scale,
        p.numel(),
        BLOCK_SIZE_N=BLOCK_SIZE,
        N_PER_TH=N_PER_TH,
        OPTIMIZER_ID=optimizer_id,
        num_warps=2,
    )


# optimizer_update_8bit_blockwise_impl = optimizer_update_8bit_blockwise_pytorch
# optimizer_update_8bit_blockwise_impl = torch.compile(optimizer_update_8bit_blockwise_pytorch_impl)
# optimizer_update_8bit_blockwise_impl = optimizer_update_8bit_blockwise_triton_quant
# optimizer_update_8bit_blockwise_impl = torch.compile(optimizer_update_8bit_blockwise_triton_quant)
optimizer_update_8bit_blockwise_impl = optimizer_update_8bit_blockwise_triton_impl
