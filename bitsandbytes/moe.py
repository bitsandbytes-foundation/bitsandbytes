"""MoE (Mixture of Experts) routing and expert dispatch.

Implements top-k token-to-expert routing with gather/scatter indices,
and chunked expert forward pass using kbit_grouped_gemm.

Designed for MoE architectures like Qwen3.5 (397B-A17B with 512 experts,
top-8 routing) and DeepSeek-style MoE models.
"""

import torch
import torch.nn.functional as torch_F


def moe_router_dispatch(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    router_jitter: float = 0.0,
) -> dict:
    """Top-k token-to-expert routing.

    Computes router logits, selects top-k experts per token, and builds
    gather/scatter indices for efficient expert dispatch.

    Args:
        hidden: Input hidden states [N_tokens, hidden_dim].
        router_weight: Router weight matrix [num_experts, hidden_dim].
        num_experts: Number of experts.
        top_k: Number of experts per token.
        router_jitter: Optional noise added to logits during training.

    Returns:
        dict with:
            expert_indices: [N_tokens, top_k] — selected expert IDs per token
            expert_weights: [N_tokens, top_k] — softmax weights (sum to 1 per token)
            token_indices_per_expert: list of [n_i] tensors — which tokens go to each expert
            expert_offsets: [num_experts + 1] — cumulative token counts for grouped GEMM
            sorted_token_indices: [total_assignments] — flat sorted token indices
            sorted_expert_indices: [total_assignments] — flat sorted expert indices
    """
    N = hidden.shape[0]
    device = hidden.device

    # Router logits: [N_tokens, num_experts]
    logits = hidden.float() @ router_weight.float().t()

    # Optional jitter for training
    if router_jitter > 0.0 and hidden.requires_grad:
        logits = logits + torch.randn_like(logits) * router_jitter

    # Top-k selection
    top_k_logits, expert_indices = torch.topk(logits, top_k, dim=-1)  # [N, top_k]

    # Softmax over selected experts (normalized per token)
    expert_weights = torch_F.softmax(top_k_logits, dim=-1)  # [N, top_k]

    # Build per-expert token indices
    # Flatten: each token appears top_k times
    flat_token_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_expert_indices = expert_indices.reshape(-1)

    # Sort by expert index for grouped GEMM
    sort_order = torch.argsort(flat_expert_indices, stable=True)
    sorted_token_indices = flat_token_indices[sort_order]
    sorted_expert_indices = flat_expert_indices[sort_order]

    # Per-expert token lists and offsets
    token_indices_per_expert = []
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)

    for e in range(num_experts):
        mask = sorted_expert_indices == e
        indices = sorted_token_indices[mask]
        token_indices_per_expert.append(indices)
        expert_counts[e] = indices.shape[0]

    # Cumulative offsets for grouped GEMM: [0, n_0, n_0+n_1, ..., total]
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0).to(torch.int32)

    return {
        "expert_indices": expert_indices,
        "expert_weights": expert_weights.to(hidden.dtype),
        "token_indices_per_expert": token_indices_per_expert,
        "expert_offsets": expert_offsets,
        "sorted_token_indices": sorted_token_indices,
        "sorted_expert_indices": sorted_expert_indices,
    }


class MoEExpertForward(torch.autograd.Function):
    """Chunked expert forward pass using kbit_grouped_gemm.

    For each expert chunk:
        1. Gather tokens routed to these experts
        2. Gate projection via grouped GEMM
        3. Up projection via grouped GEMM
        4. SwiGLU activation
        5. Down projection via grouped GEMM
        6. Scatter-add weighted results back to output
    """

    @staticmethod
    def forward(
        ctx,
        hidden,               # [N_tokens, hidden_dim]
        router_result,        # dict from moe_router_dispatch
        # Expert weights (all experts stacked)
        gate_packed_all,      # [num_experts, packed_size_gate]
        gate_absmax_all,      # [num_experts, absmax_size_gate]
        up_packed_all,        # [num_experts, packed_size_up]
        up_absmax_all,        # [num_experts, absmax_size_up]
        down_packed_all,      # [num_experts, packed_size_down]
        down_absmax_all,      # [num_experts, absmax_size_down]
        codebook,             # shared codebook
        k,                    # bit width
        hidden_dim,           # input/output dim
        intermediate_dim,     # MLP intermediate dim
        num_experts,
        expert_chunk_size,    # how many experts to process at once
    ):
        N = hidden.shape[0]
        device = hidden.device
        dtype = hidden.dtype

        output = torch.zeros(N, hidden_dim, device=device, dtype=dtype)

        expert_indices = router_result["expert_indices"]  # [N, top_k]
        expert_weights = router_result["expert_weights"]  # [N, top_k]
        sorted_token_indices = router_result["sorted_token_indices"]
        sorted_expert_indices = router_result["sorted_expert_indices"]
        expert_offsets = router_result["expert_offsets"]

        for chunk_start in range(0, num_experts, expert_chunk_size):
            chunk_end = min(chunk_start + expert_chunk_size, num_experts)
            chunk_experts = list(range(chunk_start, chunk_end))

            # Find which sorted entries belong to this chunk
            chunk_mask = (sorted_expert_indices >= chunk_start) & (sorted_expert_indices < chunk_end)
            if not chunk_mask.any():
                continue

            chunk_token_indices = sorted_token_indices[chunk_mask]
            chunk_expert_ids = sorted_expert_indices[chunk_mask]

            # Gather input tokens
            A_concat = hidden[chunk_token_indices]  # [n_chunk_tokens, hidden_dim]

            # Build local expert offsets for this chunk
            local_offsets = torch.zeros(len(chunk_experts) + 1, dtype=torch.int32, device=device)
            for i, e in enumerate(chunk_experts):
                local_offsets[i + 1] = local_offsets[i] + (chunk_expert_ids == e).sum().to(torch.int32)

            # Gate projection: grouped GEMM
            gate_out = torch.ops.bitsandbytes.kbit_grouped_gemm(
                A_concat,
                gate_packed_all[chunk_start:chunk_end],
                gate_absmax_all[chunk_start:chunk_end],
                codebook,
                local_offsets,
                hidden_dim, intermediate_dim, k, len(chunk_experts),
            )  # [n_chunk_tokens, intermediate_dim]

            # Up projection: grouped GEMM
            up_out = torch.ops.bitsandbytes.kbit_grouped_gemm(
                A_concat,
                up_packed_all[chunk_start:chunk_end],
                up_absmax_all[chunk_start:chunk_end],
                codebook,
                local_offsets,
                hidden_dim, intermediate_dim, k, len(chunk_experts),
            )  # [n_chunk_tokens, intermediate_dim]

            # SwiGLU: silu(gate) * up
            h = torch_F.silu(gate_out) * up_out

            # Down projection: grouped GEMM
            down_out = torch.ops.bitsandbytes.kbit_grouped_gemm(
                h,
                down_packed_all[chunk_start:chunk_end],
                down_absmax_all[chunk_start:chunk_end],
                codebook,
                local_offsets,
                intermediate_dim, hidden_dim, k, len(chunk_experts),
            )  # [n_chunk_tokens, hidden_dim]

            # Scatter-add with expert weights
            # For each token in this chunk, find its weight
            for i, token_idx in enumerate(chunk_token_indices):
                expert_id = chunk_expert_ids[i]
                # Find which top-k slot this expert is in for this token
                token_experts = expert_indices[token_idx]
                slot_mask = token_experts == expert_id
                weight = expert_weights[token_idx][slot_mask].sum()
                output[token_idx] += down_out[i] * weight

        # Save for backward (not implementing backward for grouped GEMM yet)
        # The backward pass would require differentiating through the grouped GEMM,
        # which needs the transposed grouped GEMM kernel
        ctx.mark_non_differentiable(output)

        return output


def moe_expert_forward(
    hidden: torch.Tensor,
    router_result: dict,
    gate_packed_all: torch.Tensor,
    gate_absmax_all: torch.Tensor,
    up_packed_all: torch.Tensor,
    up_absmax_all: torch.Tensor,
    down_packed_all: torch.Tensor,
    down_absmax_all: torch.Tensor,
    codebook: torch.Tensor,
    k: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    expert_chunk_size: int = 32,
) -> torch.Tensor:
    """Forward pass through MoE experts with chunked dispatch.

    Args:
        hidden: Input hidden states [N_tokens, hidden_dim].
        router_result: Output from moe_router_dispatch.
        gate_packed_all: All expert gate weights [num_experts, packed_size].
        gate_absmax_all: All expert gate absmax [num_experts, absmax_size].
        up_packed_all: All expert up weights.
        up_absmax_all: All expert up absmax.
        down_packed_all: All expert down weights.
        down_absmax_all: All expert down absmax.
        codebook: Shared dequantization codebook.
        k: Bit width.
        hidden_dim: Model hidden dimension.
        intermediate_dim: MLP intermediate dimension.
        num_experts: Total number of experts.
        expert_chunk_size: Number of experts to process at once.

    Returns:
        Output tensor [N_tokens, hidden_dim].
    """
    return MoEExpertForward.apply(
        hidden, router_result,
        gate_packed_all, gate_absmax_all,
        up_packed_all, up_absmax_all,
        down_packed_all, down_absmax_all,
        codebook, k, hidden_dim, intermediate_dim,
        num_experts, expert_chunk_size,
    )
