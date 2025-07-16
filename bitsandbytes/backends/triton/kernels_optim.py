import math
from typing import Optional

import torch

import triton
import triton.language as tl
# from triton.language.extra import libdevice

###########################################
# Pure torch implementation for reference #
###########################################

@torch.compile
def optimizer_update_32bit_impl_torch(
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
    step: int,
    lr: float,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    """
    32-bit optimizer implemented by Torch for performance comparison
    """
    if skip_zeros:
        raise NotImplementedError("skip_zeros is not supported on XPU yet")

    g_scaled = gnorm_scale * g
    update_scale = 1.0

    if optimizer_name == "adam":
        state1.mul_(beta1).add_(g_scaled, alpha=1.0 - beta1)
        state2.mul_(beta2).addcmul_(g_scaled, g_scaled, value=1.0 - beta2)
        
        correction1 = 1.0 - beta1 ** step
        correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
        
        if max_unorm > 0.0 and unorm_vec is not None:
            s1_corrected = state1 / correction1
            update_vals = s1_corrected / (torch.sqrt(state2) + eps)
            update_norm = torch.sum(update_vals * update_vals)
            unorm_vec.fill_(update_norm)
            current_unorm = torch.sqrt(update_norm)
            if current_unorm > max_unorm * param_norm:
                update_scale = (max_unorm * param_norm) / current_unorm

        if weight_decay > 0.0:
            p.mul_(1.0 - lr * weight_decay)
        
        step_size = -lr * correction2_sqrt / correction1
        update_val = state1 / (torch.sqrt(state2) + eps * correction2_sqrt)
        p.add_(update_val, alpha=update_scale * step_size)
        
    elif optimizer_name == "ademamix":
        s1_vals = state1[0]
        s3_vals = state1[1]
        
        s1_vals.mul_(beta1).add_(g_scaled, alpha=1.0 - beta1)
        s3_vals.mul_(beta3).add_(g_scaled, alpha=1.0 - beta3)
        state2.mul_(beta2).addcmul_(g_scaled, g_scaled, value=1.0 - beta2)
        
        correction1 = 1.0 - beta1 ** step
        correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
        
        numerator = (s1_vals / correction1) + (alpha * s3_vals)
        denominator = (torch.sqrt(state2) / correction2_sqrt) + eps
        update_vals = numerator / denominator

        if max_unorm > 0.0 and unorm_vec is not None:
            update_norm = torch.sum(update_vals * update_vals)
            unorm_vec.fill_(update_norm)
            current_unorm = torch.sqrt(update_norm)
            if current_unorm > max_unorm * param_norm:
                update_scale = (max_unorm * param_norm) / current_unorm

        if weight_decay > 0.0:
            p.mul_(1.0 - lr * weight_decay)
        
        p.add_(update_vals, alpha=-lr * update_scale)
        
    elif optimizer_name in ["momentum", "rmsprop", "adagrad", "lion"]:
        g_with_decay = g_scaled
        if weight_decay > 0.0:
            g_with_decay = g_with_decay.add(p, alpha=weight_decay)

        if optimizer_name == "momentum":
            state1.mul_(beta1).add_(g_with_decay)
            update_vals = state1
        elif optimizer_name == "rmsprop":
            state1.mul_(beta1).addcmul_(g_with_decay, g_with_decay, value=1.0 - beta1)
            update_vals = g_with_decay / (torch.sqrt(state1) + eps)
        elif optimizer_name == "adagrad":
            state1.addcmul_(g_with_decay, g_with_decay, value=1.0)
            update_vals = g_with_decay / (torch.sqrt(state1) + eps)
        elif optimizer_name == "lion":
            momentum_update = state1.mul(beta1).add(g_with_decay, alpha=1.0 - beta1)
            update_vals = torch.sign(momentum_update)
            state1.mul_(beta2).add_(g_with_decay, alpha=1.0 - beta2)

        if max_unorm > 0.0 and unorm_vec is not None:
            unorm_calc_source = state1 if optimizer_name == "lion" else update_vals
            update_norm = torch.sum(unorm_calc_source * unorm_calc_source)
            unorm_vec.fill_(update_norm)
            current_unorm = torch.sqrt(update_norm)
            if current_unorm > max_unorm * param_norm:
                update_scale = (max_unorm * param_norm) / current_unorm
        
        if optimizer_name == "lion":
            p.add_(update_vals, alpha=-lr * update_scale)
        else:
            p.add_(update_vals, alpha=-lr * update_scale)

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
def _optimizer_precondition_2state_32bit(
    g_ptr,
    p_ptr,
    state1_ptr,
    state2_ptr,
    unorm_ptr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    step,
    beta1_step,
    beta2_step,
    lr,
    gnorm_scale: tl.constexpr,
    n_elements,
    OPTIMIZER_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    """Preprocessing optimizer, computing update norm (2-state optimizer)"""
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE * N_PER_TH)
    mask = offsets < n_elements
    
    g_vals = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    s1_vals = tl.load(state1_ptr + offsets, mask=mask, other=0.0)
    s2_vals = tl.load(state2_ptr + offsets, mask=mask, other=0.0)
    
    g_vals = gnorm_scale * g_vals
    
    correction1 = 1.0 / (1.0 - beta1_step)
    correction2 = 1.0 / (1.0 - beta2_step)
    
    if OPTIMIZER_ID == 3: # ADAM
        s1_vals = s1_vals * beta1 + (1.0 - beta1) * g_vals
        s2_vals = s2_vals * beta2 + (1.0 - beta2) * g_vals * g_vals
        
        s1_vals = s1_vals * correction1
        s2_vals = s2_vals * correction2
        
        update_vals = s1_vals / (tl.sqrt(s2_vals) + eps)

        update_norm = update_vals * update_vals

    elif OPTIMIZER_ID == 5: # ADEMAMIX
        update_norm = s1_vals

    total_norm = tl.sum(tl.where(mask, update_norm, 0.0))
    
    tl.atomic_add(unorm_ptr, total_norm)


@triton.jit
def _optimizer_precondition_1state_32bit(
    g_ptr,
    p_ptr,
    state1_ptr,
    state2_ptr,
    unorm_ptr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay,
    step, 
    beta1_step,
    beta2_step,
    lr,
    gnorm_scale: tl.constexpr,
    n_elements,
    OPTIMIZER_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    """Preprocessing optimizer, computing update norm (1-state optimizer)"""
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE * N_PER_TH)
    mask = offsets < n_elements
    
    g_vals = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    s1_vals = tl.load(state1_ptr + offsets, mask=mask, other=0.0)
    
    g_vals = gnorm_scale * g_vals
    
    if OPTIMIZER_ID == 0:  # MOMENTUM
        if step == 1:
            s1_vals = g_vals
        else:
            s1_vals = s1_vals * beta1 + g_vals
        update_norm = s1_vals * s1_vals

    elif OPTIMIZER_ID == 4:  # LION
        s1_vals = s1_vals * beta2 + (1.0 - beta2) * g_vals
        update_norm = s1_vals

    elif OPTIMIZER_ID == 1:  # RMSPROP
        s1_vals = s1_vals * beta1 + (1.0 - beta1) * g_vals * g_vals
        update_vals = g_vals / (tl.sqrt(s1_vals) + eps)
        update_norm = update_vals * update_vals

    elif OPTIMIZER_ID == 2:  # ADAGRAD
        s1_vals = s1_vals + g_vals * g_vals
        update_vals = g_vals / (tl.sqrt(s1_vals) + eps)
        update_norm = update_vals * update_vals
    
    total_norm = tl.sum(tl.where(mask, update_norm, 0.0))
    
    tl.atomic_add(unorm_ptr, total_norm)


@triton.jit
def _optimizer_update_2state_32bit_triton_kernel(
    g_ptr,
    p_ptr,
    state1_ptr,
    state2_ptr,
    unorm_ptr,
    max_unorm: tl.constexpr,
    param_norm,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    beta3,
    alpha,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    step, 
    beta1_step,
    beta2_step,
    lr,
    gnorm_scale: tl.constexpr,
    skip_zeros,
    n_elements,
    OPTIMIZER_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    """2-state optimizer kernel"""
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE * N_PER_TH)
    mask = offsets < n_elements
    
    g_vals = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p_vals = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s1_vals = tl.load(state1_ptr + offsets, mask=mask, other=0.0)
    s2_vals = tl.load(state2_ptr + offsets, mask=mask, other=0.0)
    
    if OPTIMIZER_ID == 5:  # ADEMAMIX
        s3_vals = tl.load(state1_ptr + n_elements + offsets, mask=mask, other=0.0)
    
    g_vals = gnorm_scale * g_vals
    
    update_scale = 1.0
    if max_unorm > 0.0:
        current_unorm = tl.sqrt(tl.load(unorm_ptr))
        if current_unorm > max_unorm * param_norm:
            update_scale = (max_unorm * param_norm) / current_unorm
    
    if OPTIMIZER_ID == 3:  # ADAM
        s1_vals = s1_vals * beta1 + (1.0 - beta1) * g_vals
        s2_vals = s2_vals * beta2 + (1.0 - beta2) * g_vals * g_vals

        correction1 = 1.0 - beta1_step
        correction2 = tl.sqrt(1.0 - beta2_step)
        step_size = -lr * correction2 / correction1

        if weight_decay > 0.0:
            p_vals = p_vals * (1.0 - lr * weight_decay)

        update_val = update_scale * step_size * (s1_vals / (tl.sqrt(s2_vals) + eps * correction2))
        p_vals = p_vals + update_val
    
    elif OPTIMIZER_ID == 5:  # ADEMAMIX     
        s1_vals = s1_vals * beta1 + (1.0 - beta1) * g_vals  # m1
        s3_vals = s3_vals * beta3 + (1.0 - beta3) * g_vals  # m2
        s2_vals = s2_vals * beta2 + (1.0 - beta2) * g_vals * g_vals  # nu

        correction1 = 1.0 - beta1_step
        correction2 = tl.sqrt(1.0 - beta2_step)

        if weight_decay > 0.0:
            p_vals = p_vals * (1.0 - lr * weight_decay)
        
        numerator = (s1_vals / correction1) + (alpha * s3_vals)
        denominator = (tl.sqrt(s2_vals) / correction2) + eps
        p_vals = p_vals - lr * (numerator / denominator)
    
    tl.store(p_ptr + offsets, p_vals, mask=mask)
    tl.store(state1_ptr + offsets, s1_vals, mask=mask)
    tl.store(state2_ptr + offsets, s2_vals, mask=mask)
    
    if OPTIMIZER_ID == 5:  # ADEMAMIX
        tl.store(state1_ptr + n_elements + offsets, s3_vals, mask=mask)


@triton.jit
def _optimizer_update_1state_32bit_triton_kernel(
    g_ptr,
    p_ptr, 
    state1_ptr,
    state2_ptr,
    unorm_ptr,
    max_unorm: tl.constexpr,
    param_norm,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    beta3,
    alpha,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    step,
    beta1_step,
    beta2_step,
    lr,
    gnorm_scale: tl.constexpr,
    skip_zeros,
    n_elements,
    OPTIMIZER_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    """1-state optimizer kernel"""
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE * N_PER_TH)
    mask = offsets < n_elements
    
    g_vals = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p_vals = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s1_vals = tl.load(state1_ptr + offsets, mask=mask, other=0.0)

    g_vals = gnorm_scale * g_vals
    if weight_decay > 0.0:
        g_vals = g_vals + p_vals * weight_decay
    
    update_scale = 1.0
    if max_unorm > 0.0:
        current_unorm = tl.sqrt(tl.load(unorm_ptr))
        if current_unorm > max_unorm * param_norm + eps:
            update_scale = (max_unorm * param_norm + eps) / current_unorm
    
    if OPTIMIZER_ID == 0:  # MOMENTUM
        if step == 1:
            s1_vals = g_vals
        else:
            s1_vals = s1_vals * beta1 + g_vals
        
        update_val = update_scale * (-lr * s1_vals)
        p_vals = p_vals + update_val

    elif OPTIMIZER_ID == 4:  # LION
        momentum_update = s1_vals * beta1 + (1.0 - beta1) * g_vals
        update_val = update_scale * lr * tl.where(momentum_update > 0, 1.0, tl.where(momentum_update < 0, -1.0, 0.0))
        p_vals = p_vals - update_val
        
        s1_vals = s1_vals * beta2 + (1.0 - beta2) * g_vals

    elif OPTIMIZER_ID == 1:  # RMSPROP
        s1_vals = s1_vals * beta1 + (1.0 - beta1) * g_vals * g_vals
        
        update_val = update_scale * lr * g_vals / (tl.sqrt(s1_vals) + eps)
        p_vals = p_vals - update_val

    elif OPTIMIZER_ID == 2:  # ADAGRAD
        s1_vals = s1_vals + g_vals * g_vals
        
        update_val = lr * g_vals / (tl.sqrt(s1_vals) + eps)
        p_vals = p_vals - update_val
    
    tl.store(p_ptr + offsets, p_vals, mask=mask)
    tl.store(state1_ptr + offsets, s1_vals, mask=mask)


name2optimizer_32bit_fn = {
    "adam": {
        "preprocess": _optimizer_precondition_2state_32bit,
        "update": _optimizer_update_2state_32bit_triton_kernel,
    },
    "ademamix": {
        "preprocess": _optimizer_precondition_2state_32bit,
        "update": _optimizer_update_2state_32bit_triton_kernel,
    },
    "momentum": {
        "preprocess": _optimizer_precondition_1state_32bit,
        "update": _optimizer_update_1state_32bit_triton_kernel,
    },
    "rmsprop": {
        "preprocess": _optimizer_precondition_1state_32bit,
        "update": _optimizer_update_1state_32bit_triton_kernel,
    },
    "adagrad": {
        "preprocess": _optimizer_precondition_1state_32bit,
        "update": _optimizer_update_1state_32bit_triton_kernel,
    },
    "lion": {
        "preprocess": _optimizer_precondition_1state_32bit,
        "update": _optimizer_update_1state_32bit_triton_kernel,
    },
}


def optimizer_update_32bit_impl(
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
    step: int,
    lr: float,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    """
    32-bit optimizer implemented by Triton
    """
    if skip_zeros:
        raise NotImplementedError("skip_zeros is not supported on XPU yet")

    BLOCK_SIZE = 256
    N_PER_TH = 1  # Number of blocks processed per thread.
    grid = (triton.cdiv(p.numel(), BLOCK_SIZE * N_PER_TH),)
    optimizer_id = name2optimizer_id[optimizer_name]
    fn_preprocess = name2optimizer_32bit_fn[optimizer_name]["preprocess"]
    fn_update = name2optimizer_32bit_fn[optimizer_name]["update"]

    # In torch=2.7 on XPU there is an issue with libdevice.pow, leading to an error.
    # For backwards compatibility we precompute the bias correction factors.
    beta1_step = beta1**step
    beta2_step = beta2**step

    if optimizer_name == "lion":
        fn_update[grid](
            g, p, state1, state2, unorm_vec, max_unorm, param_norm,
            beta1, beta2, beta3, alpha, eps, weight_decay, step,
            beta1_step, beta2_step, lr, gnorm_scale, skip_zeros,
            p.numel(), optimizer_id, BLOCK_SIZE, N_PER_TH, num_warps=2,
        )

        if max_unorm > 0.0:
            unorm_vec.zero_()
            fn_preprocess[grid](
                g, p, state1, state2, unorm_vec,
                beta1, beta2, eps, weight_decay, step, 
                beta1_step, beta2_step, lr, gnorm_scale,
                p.numel(), optimizer_id, BLOCK_SIZE, N_PER_TH, num_warps=2,
            )

    else:
        if max_unorm > 0.0:
            unorm_vec.zero_()
            fn_preprocess[grid](
                g, p, state1, state2, unorm_vec,
                beta1, beta2, eps, weight_decay, step, 
                beta1_step, beta2_step, lr, gnorm_scale,
                p.numel(), optimizer_id, BLOCK_SIZE, N_PER_TH, num_warps=2,
            )

        fn_update[grid](
            g, p, state1, state2, unorm_vec, max_unorm, param_norm,
            beta1, beta2, beta3, alpha, eps, weight_decay, step,
            beta1_step, beta2_step, lr, gnorm_scale, skip_zeros,
            p.numel(), optimizer_id, BLOCK_SIZE, N_PER_TH, num_warps=2,
        )
