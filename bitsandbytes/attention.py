"""Chunked Flash Attention (single-GPU ring attention).

Chunks the query along the sequence dimension and processes each chunk
through flash_attn, keeping K/V in full memory (efficient with GQA).
For very long K/V, supports chunking both Q and K/V with logsumexp
merging for correct softmax normalization.

Requires: flash_attn (pip install flash-attn)
"""

import torch


def _import_flash_attn():
    """Lazy import of flash_attn to give clear error messages."""
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except ImportError:
        raise ImportError(
            "Chunked attention requires the flash_attn package. "
            "Install with: pip install flash-attn --no-build-isolation"
        )


def chunked_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    chunk_size: int = 4096,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Chunked causal attention using flash_attn.

    Chunks Q along the sequence dimension and processes each chunk
    against K/V with correct causal masking. K and V are kept in
    full memory (efficient with GQA where K/V have fewer heads).

    For each Q chunk at positions [c, c+chunk_size), we pass
    K[:c+chunk_size] and V[:c+chunk_size] with causal=True. flash_attn's
    bottom-right causal alignment means Q positions correctly attend
    only to their past and present keys.

    Args:
        Q: Query tensor [B, S, H_q, D] where H_q is number of query heads.
        K: Key tensor [B, S, H_kv, D] where H_kv <= H_q (GQA supported).
        V: Value tensor [B, S, H_kv, D].
        chunk_size: Number of query positions per chunk. Should be a
            multiple of 128 for optimal flash_attn performance.
        causal: Whether to apply causal masking. Default True.
        softmax_scale: Scaling factor for QK^T. Default 1/sqrt(D).

    Returns:
        Output tensor [B, S, H_q, D].
    """
    flash_attn_func = _import_flash_attn()

    B, S, H_q, D = Q.shape
    device = Q.device

    # If sequence fits in one chunk, just call flash_attn directly
    if S <= chunk_size:
        return flash_attn_func(
            Q, K, V,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    output = torch.empty_like(Q)

    for c_start in range(0, S, chunk_size):
        c_end = min(c_start + chunk_size, S)
        q_chunk = Q[:, c_start:c_end]  # [B, cs, H_q, D]

        if causal:
            # Only need K/V up to c_end for causal attention.
            # flash_attn aligns Q to the bottom-right of K, so Q[0]
            # maps to key position c_start and can attend to K[0:c_start+1].
            k_slice = K[:, :c_end]  # [B, c_end, H_kv, D]
            v_slice = V[:, :c_end]  # [B, c_end, H_kv, D]
        else:
            k_slice = K
            v_slice = V

        out_chunk = flash_attn_func(
            q_chunk, k_slice, v_slice,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        output[:, c_start:c_end] = out_chunk

    return output


def chunked_flash_attention_full(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q_chunk_size: int = 4096,
    kv_chunk_size: int = 4096,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Fully chunked attention with logsumexp merging.

    Chunks both Q and K/V, merging partial attention results using
    the online softmax trick. Use this when K/V are too large to
    keep in full memory (very long sequences without GQA compression).

    For most cases, chunked_flash_attention (Q-only chunking) is
    preferred since K/V are small with GQA.

    Args:
        Q: Query tensor [B, S, H_q, D].
        K: Key tensor [B, S, H_kv, D].
        V: Value tensor [B, S, H_kv, D].
        q_chunk_size: Query chunk size.
        kv_chunk_size: Key/Value chunk size.
        causal: Whether to apply causal masking.
        softmax_scale: Scaling factor for QK^T. Default 1/sqrt(D).

    Returns:
        Output tensor [B, S, H_q, D].
    """
    flash_attn_func = _import_flash_attn()

    B, S, H_q, D = Q.shape
    device = Q.device

    # If everything fits in one chunk, call directly
    if S <= q_chunk_size and S <= kv_chunk_size:
        return flash_attn_func(
            Q, K, V,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    output = torch.empty_like(Q)

    for q_start in range(0, S, q_chunk_size):
        q_end = min(q_start + q_chunk_size, S)
        q_chunk = Q[:, q_start:q_end]  # [B, qcs, H_q, D]
        qcs = q_end - q_start

        # Determine KV range for this Q chunk
        if causal:
            kv_end_max = q_end  # No need to attend past q_end for causal
        else:
            kv_end_max = S

        # Accumulate partial attention results with online softmax
        # running_out: [B, qcs, H_q, D] weighted sum
        # running_lse: [B, H_q, qcs] log-sum-exp
        running_out = None
        running_lse = None

        for kv_start in range(0, kv_end_max, kv_chunk_size):
            kv_end = min(kv_start + kv_chunk_size, kv_end_max)
            k_chunk = K[:, kv_start:kv_end]
            v_chunk = V[:, kv_start:kv_end]

            # Determine if causal masking applies to this chunk pair
            # Causal only matters when Q and K/V chunks overlap or Q is after K/V
            if causal and kv_end > q_start:
                # There is overlap — need causal masking within this block
                chunk_causal = True
            else:
                # K/V chunk is fully before Q chunk — no causal needed, full attend
                chunk_causal = False

            # Get partial attention output and LSE
            partial_out, partial_lse, _ = flash_attn_func(
                q_chunk, k_chunk, v_chunk,
                causal=chunk_causal,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
            )
            # partial_lse: [B, H_q, qcs]

            if running_out is None:
                running_out = partial_out
                running_lse = partial_lse
            else:
                # Online softmax merge
                # new_lse = log(exp(running_lse) + exp(partial_lse))
                # = max(running_lse, partial_lse) + log(exp(running_lse - max) + exp(partial_lse - max))
                new_lse = torch.logaddexp(running_lse, partial_lse)

                # Weight for running output: exp(running_lse - new_lse)
                # Weight for partial output: exp(partial_lse - new_lse)
                # Both have shape [B, H_q, qcs] -> need to reshape for broadcast with [B, qcs, H_q, D]
                w_running = torch.exp(running_lse - new_lse)  # [B, H_q, qcs]
                w_partial = torch.exp(partial_lse - new_lse)  # [B, H_q, qcs]

                # Reshape weights: [B, H_q, qcs] -> [B, qcs, H_q, 1]
                w_running = w_running.permute(0, 2, 1).unsqueeze(-1)
                w_partial = w_partial.permute(0, 2, 1).unsqueeze(-1)

                running_out = running_out * w_running + partial_out * w_partial
                running_lse = new_lse

        output[:, q_start:q_end] = running_out

    return output
