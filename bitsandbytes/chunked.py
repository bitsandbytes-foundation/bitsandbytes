"""Sequence-chunked wrappers for LoRA autograd functions.

Chunks the input along the sequence dimension and processes each chunk
through the underlying autograd function, optionally wrapping each
chunk with torch.utils.checkpoint for activation recomputation.

This reduces peak MLP activation memory from O(B*S*intermediate)
to O(B*chunk*intermediate) at the cost of ~50% more MLP FLOPs
during backward (recomputation).
"""

import torch
from torch.utils.checkpoint import checkpoint

from bitsandbytes.autograd.lora_kbit import LoRA_MLP_Kbit


def chunked_mlp_forward(
    X: torch.Tensor,
    chunk_size: int,
    # Gate projection
    packed_gate: torch.Tensor,
    absmax_gate: torch.Tensor,
    codebook_gate: torch.Tensor,
    A_gate: torch.Tensor,
    B_gate: torch.Tensor,
    s_gate: float,
    # Up projection
    packed_up: torch.Tensor,
    absmax_up: torch.Tensor,
    codebook_up: torch.Tensor,
    A_up: torch.Tensor,
    B_up: torch.Tensor,
    s_up: float,
    # Down projection
    packed_down: torch.Tensor,
    absmax_down: torch.Tensor,
    codebook_down: torch.Tensor,
    A_down: torch.Tensor,
    B_down: torch.Tensor,
    s_down: float,
    # Shared params
    k: int,
    K_dim_in: int,
    N_hidden: int,
    N_hidden_padded: int,
    K_dim_hidden: int,
    N_out: int,
    N_out_padded: int,
    compute_dtype: torch.dtype,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Process MLP in sequence chunks with optional gradient checkpointing.

    Chunks X along dim 0 (the token dimension) and calls LoRA_MLP_Kbit.apply
    on each chunk. When use_checkpoint=True, wraps each chunk with
    torch.utils.checkpoint so that intermediate activations (gate, up, SwiGLU)
    are recomputed during backward instead of stored.

    Args:
        X: Input tensor [M, K_dim_in] where M = B*S (flattened tokens).
        chunk_size: Number of tokens per chunk.
        packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate:
            Gate projection parameters (kbit packed weight + LoRA).
        packed_up, absmax_up, codebook_up, A_up, B_up, s_up:
            Up projection parameters.
        packed_down, absmax_down, codebook_down, A_down, B_down, s_down:
            Down projection parameters.
        k: Bit width.
        K_dim_in: Input dimension.
        N_hidden: MLP intermediate dimension.
        N_hidden_padded: Padded intermediate dimension.
        K_dim_hidden: Hidden dimension (= N_hidden for standard MLP).
        N_out: Output dimension.
        N_out_padded: Padded output dimension.
        compute_dtype: Computation dtype (fp16/bf16).
        use_checkpoint: If True, use gradient checkpointing per chunk.
            Default True.

    Returns:
        Output tensor [M, N_out].
    """
    M = X.shape[0]

    # If input fits in one chunk, process directly (no chunking overhead)
    if M <= chunk_size:
        return LoRA_MLP_Kbit.apply(
            X,
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
            packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
            packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
            k, K_dim_in, N_hidden, N_hidden_padded,
            K_dim_hidden, N_out, N_out_padded, compute_dtype,
        )

    chunks_out = []

    for c_start in range(0, M, chunk_size):
        c_end = min(c_start + chunk_size, M)
        x_chunk = X[c_start:c_end]

        if use_checkpoint:
            # Wrap with gradient checkpointing: saves MLP intermediates
            # (gate, up, SwiGLU outputs) from being stored; recomputes
            # during backward. use_reentrant=False is the modern API.
            chunk_out = checkpoint(
                _mlp_chunk_fn,
                x_chunk,
                packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
                packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
                packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
                k, K_dim_in, N_hidden, N_hidden_padded,
                K_dim_hidden, N_out, N_out_padded, compute_dtype,
                use_reentrant=False,
            )
        else:
            chunk_out = LoRA_MLP_Kbit.apply(
                x_chunk,
                packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
                packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
                packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
                k, K_dim_in, N_hidden, N_hidden_padded,
                K_dim_hidden, N_out, N_out_padded, compute_dtype,
            )

        chunks_out.append(chunk_out)

    return torch.cat(chunks_out, dim=0)


def _mlp_chunk_fn(
    x_chunk,
    packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
    packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
    packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
    k, K_dim_in, N_hidden, N_hidden_padded,
    K_dim_hidden, N_out, N_out_padded, compute_dtype,
):
    """Wrapper function for checkpoint compatibility.

    torch.utils.checkpoint requires a plain function (not a method or
    autograd.Function.apply directly). This wraps LoRA_MLP_Kbit.apply.
    """
    return LoRA_MLP_Kbit.apply(
        x_chunk,
        packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
        packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
        packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
        k, K_dim_in, N_hidden, N_hidden_padded,
        K_dim_hidden, N_out, N_out_padded, compute_dtype,
    )
