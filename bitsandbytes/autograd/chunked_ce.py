"""Chunked fused linear cross-entropy loss.

Computes cross-entropy loss on kbit-quantized LM head weights WITHOUT
materializing the full [B*S, vocab_size] logits tensor. Instead, loops
over vocab chunks: dequantize the weight, compute partial logits via
cuBLAS, update a running logsumexp, and accumulate the loss.

Memory: O(B*S * chunk_size) instead of O(B*S * vocab_size).
"""

import torch

import bitsandbytes.functional as F


class ChunkedCrossEntropy(torch.autograd.Function):
    """Cross-entropy on kbit-quantized LM head with vocab chunking.

    Forward:
        For each vocab chunk [c_start, c_end):
            partial_logits = hidden @ W_chunk^T           [B, chunk_size]
            Update running max and sum_exp for logsumexp
            Extract label logits falling in this chunk
        loss = logsumexp - label_logit (per token)

    Backward:
        For each vocab chunk:
            Recompute partial_logits = hidden @ W_chunk^T
            partial_softmax = exp(partial_logits - logsumexp)
            Subtract one-hot for labels in chunk
            grad_hidden += partial_softmax @ W_chunk
    """

    @staticmethod
    def forward(
        ctx,
        hidden,       # [N_tokens, hidden_dim], bf16/fp16
        packed,       # int32, kbit packed LM head weight
        absmax,       # per-block absmax
        codebook,     # codebook for dequantization
        labels,       # [N_tokens], int64
        k,            # bit width
        K_dim,        # hidden dimension
        N_padded,     # vocab_size padded to 128
        N,            # actual vocab_size
        compute_dtype,
        chunk_size,   # vocab chunk size (e.g. 8192)
        ignore_index, # label to ignore (default -100)
    ):
        # Dequantize full LM head weight [vocab_size, hidden_dim]
        n_elements = N_padded * K_dim
        w_deq = F.dequantize_kbit(packed, absmax, codebook, k, n_elements, compute_dtype)
        W = w_deq[:n_elements].reshape(N_padded, K_dim)[:N, :]

        B = hidden.shape[0]
        device = hidden.device

        # Online logsumexp accumulators
        max_logit = torch.full((B,), -float("inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros(B, device=device, dtype=torch.float32)
        label_logit = torch.zeros(B, device=device, dtype=torch.float32)

        for c_start in range(0, N, chunk_size):
            c_end = min(c_start + chunk_size, N)
            W_chunk = W[c_start:c_end]
            partial = hidden @ W_chunk.t()  # [B, chunk_size]
            partial_f = partial.float()

            # Online logsumexp update (numerically stable)
            chunk_max = partial_f.max(dim=-1).values
            new_max = torch.max(max_logit, chunk_max)
            sum_exp = (
                sum_exp * torch.exp(max_logit - new_max)
                + torch.exp(partial_f - new_max.unsqueeze(-1)).sum(dim=-1)
            )
            max_logit = new_max

            # Extract label logits in this chunk
            in_chunk = (labels >= c_start) & (labels < c_end) & (labels != ignore_index)
            if in_chunk.any():
                local_idx = labels[in_chunk] - c_start
                label_logit[in_chunk] = partial_f[in_chunk, local_idx]

        logsumexp = max_logit + torch.log(sum_exp)

        # Per-token loss
        valid_mask = labels != ignore_index
        losses = logsumexp - label_logit
        n_valid = valid_mask.sum()
        if n_valid > 0:
            mean_loss = losses[valid_mask].sum() / n_valid.float()
        else:
            mean_loss = losses.sum() * 0.0

        ctx.save_for_backward(hidden, packed, absmax, codebook, labels, logsumexp)
        ctx.k = k
        ctx.K_dim = K_dim
        ctx.N_padded = N_padded
        ctx.N = N
        ctx.compute_dtype = compute_dtype
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid

        return mean_loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden, packed, absmax, codebook, labels, logsumexp = ctx.saved_tensors

        # Re-dequantize LM head weight
        n_elements = ctx.N_padded * ctx.K_dim
        w_deq = F.dequantize_kbit(
            packed, absmax, codebook, ctx.k, n_elements, ctx.compute_dtype,
        )
        W = w_deq[:n_elements].reshape(ctx.N_padded, ctx.K_dim)[:ctx.N, :]

        B = hidden.shape[0]
        grad_hidden = torch.zeros_like(hidden)

        # Per-sample gradient scale: grad_output / n_valid
        valid_mask = labels != ctx.ignore_index
        grad_scale = torch.zeros(B, device=hidden.device, dtype=torch.float32)
        if ctx.n_valid > 0:
            grad_scale[valid_mask] = grad_output.float() / ctx.n_valid.float()

        for c_start in range(0, ctx.N, ctx.chunk_size):
            c_end = min(c_start + ctx.chunk_size, ctx.N)
            W_chunk = W[c_start:c_end]

            # Recompute partial logits
            partial = hidden @ W_chunk.t()  # [B, chunk_size]

            # Softmax using stored logsumexp
            partial_sm = torch.exp(partial.float() - logsumexp.unsqueeze(-1))

            # Subtract one-hot for labels in this chunk
            in_chunk = (labels >= c_start) & (labels < c_end) & (labels != ctx.ignore_index)
            if in_chunk.any():
                local_idx = labels[in_chunk] - c_start
                partial_sm[in_chunk, local_idx] -= 1.0

            # Scale by gradient
            partial_sm *= grad_scale.unsqueeze(-1)

            # Accumulate grad_hidden: [B, chunk] @ [chunk, hidden]
            grad_hidden += partial_sm.to(hidden.dtype) @ W_chunk

        # Return: hidden, packed, absmax, codebook, labels, k, K_dim, N_padded, N,
        #         compute_dtype, chunk_size, ignore_index
        return grad_hidden, None, None, None, None, None, None, None, None, None, None, None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    packed: torch.Tensor,
    absmax: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    K_dim: int,
    N_padded: int,
    N: int,
    compute_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 8192,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Chunked cross-entropy loss on kbit-quantized LM head.

    Computes CE loss without materializing the full [B*S, vocab] logits
    tensor. Memory scales with chunk_size instead of vocab_size.

    Args:
        hidden: Hidden states [N_tokens, hidden_dim], bf16/fp16.
        packed: Kbit-packed LM head weight.
        absmax: Per-block absmax for the LM head.
        codebook: Dequantization codebook.
        labels: Target labels [N_tokens], int64.
        k: Bit width (2-5).
        K_dim: Hidden dimension.
        N_padded: Vocab size padded to 128.
        N: Actual vocab size.
        compute_dtype: Dtype for matmul computation.
        chunk_size: Number of vocab entries per chunk.
        ignore_index: Label value to ignore.

    Returns:
        Scalar mean loss.
    """
    return ChunkedCrossEntropy.apply(
        hidden, packed, absmax, codebook, labels,
        k, K_dim, N_padded, N, compute_dtype, chunk_size, ignore_index,
    )
