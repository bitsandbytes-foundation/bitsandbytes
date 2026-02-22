"""torch.autograd.Function wrappers for CUDA training kernels.

Wraps the low-level CUDA ops (SwiGLU, RMSNorm, RoPE) into autograd-aware
functions that can be used directly in PyTorch training.
"""

import torch


class SwiGLUFunction(torch.autograd.Function):
    """SwiGLU activation: h = silu(gate) * up.

    Forward:  h = (gate * sigmoid(gate)) * up
    Backward: grad_gate = grad_h * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
              grad_up   = grad_h * silu(gate)
    """

    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return torch.ops.bitsandbytes.swiglu_forward(gate, up)

    @staticmethod
    def backward(ctx, grad_h):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = torch.ops.bitsandbytes.swiglu_backward(
            grad_h.contiguous(), gate, up,
        )
        return grad_gate, grad_up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation with autograd support.

    Args:
        gate: Gate tensor (any shape, fp16 or bf16).
        up: Up tensor (same shape as gate).

    Returns:
        silu(gate) * up
    """
    return SwiGLUFunction.apply(gate, up)


class RMSNormFunction(torch.autograd.Function):
    """RMS normalization: y = x * rsqrt(mean(x^2) + eps) * w.

    Supports Gemma variant with ``add_unit_offset=True`` (uses w + 1).
    """

    @staticmethod
    def forward(ctx, x, w, eps=1e-6, add_unit_offset=False):
        # Flatten to 2D for the CUDA kernel
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        out_2d, rrms = torch.ops.bitsandbytes.rmsnorm_forward(
            x_2d, w, eps, add_unit_offset,
        )

        ctx.save_for_backward(x_2d, w, rrms)
        ctx.add_unit_offset = add_unit_offset
        ctx.orig_shape = orig_shape

        return out_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, w, rrms = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(x_2d.shape).contiguous()

        grad_x_2d, grad_w = torch.ops.bitsandbytes.rmsnorm_backward(
            grad_out_2d, x_2d, w, rrms, ctx.add_unit_offset,
        )

        grad_x = grad_x_2d.reshape(ctx.orig_shape)
        return grad_x, grad_w.to(w.dtype), None, None


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    """RMS normalization with autograd support.

    Args:
        x: Input tensor (*, hidden_size), fp16 or bf16.
        w: Weight tensor (hidden_size,).
        eps: Epsilon for numerical stability.
        add_unit_offset: If True, uses (w + 1) instead of w (Gemma convention).

    Returns:
        Normalized tensor of same shape as x.
    """
    return RMSNormFunction.apply(x, w, eps, add_unit_offset)


class RoPEFunction(torch.autograd.Function):
    """Rotary Position Embedding (in-place).

    Forward:  q[..., :half] = q[..., :half] * cos - q[..., half:] * sin
              q[..., half:] = q[..., half:] * cos + q[..., :half] * sin
    Backward: same operation with sin negated.
    """

    @staticmethod
    def forward(ctx, q, cos_cache, sin_cache, n_heads):
        # q: [total_tokens, n_heads, head_dim]
        ctx.save_for_backward(cos_cache, sin_cache)
        ctx.n_heads = n_heads

        q_out = q.clone()
        torch.ops.bitsandbytes.rope_forward(q_out, cos_cache, sin_cache, n_heads)
        return q_out

    @staticmethod
    def backward(ctx, grad_q):
        cos_cache, sin_cache = ctx.saved_tensors

        # Backward of RoPE is the same operation with sin negated
        grad_q_out = grad_q.clone()
        torch.ops.bitsandbytes.rope_forward(
            grad_q_out, cos_cache, -sin_cache, ctx.n_heads,
        )
        return grad_q_out, None, None, None


def rope(
    q: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """Apply Rotary Position Embedding with autograd support.

    Args:
        q: Query tensor [total_tokens, n_heads, head_dim], fp16 or bf16.
        cos_cache: Cosine cache [total_tokens, head_dim/2].
        sin_cache: Sine cache [total_tokens, head_dim/2].
        n_heads: Number of attention heads.

    Returns:
        Rotated query tensor (same shape).
    """
    return RoPEFunction.apply(q, cos_cache, sin_cache, n_heads)
