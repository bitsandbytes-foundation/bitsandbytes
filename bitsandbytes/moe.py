"""MoE (Mixture of Experts) routing and expert dispatch.

Implements top-k token-to-expert routing with gather/scatter indices,
and chunked expert forward pass with per-expert dequantization.

Expert weights are stored in flat kbit-packed format (from quantize_kbit).
Forward and backward use per-expert dequantize_kbit + cuBLAS matmul,
following the same approach as LoRA_W_Kbit's backward pass.

Designed for MoE architectures like Qwen3.5 (397B-A17B with 512 experts,
top-8 routing) and DeepSeek-style MoE models.
"""

import torch
import torch.nn.functional as torch_F

from bitsandbytes.functional import dequantize_kbit


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
            sorted_weights: [total_assignments] — weights matching sorted order
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
    flat_weights = expert_weights.reshape(-1)

    # Sort by expert index for grouped dispatch
    sort_order = torch.argsort(flat_expert_indices, stable=True)
    sorted_token_indices = flat_token_indices[sort_order]
    sorted_expert_indices = flat_expert_indices[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Per-expert token lists and offsets
    token_indices_per_expert = []
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)

    for e in range(num_experts):
        mask = sorted_expert_indices == e
        indices = sorted_token_indices[mask]
        token_indices_per_expert.append(indices)
        expert_counts[e] = indices.shape[0]

    # Cumulative offsets: [0, n_0, n_0+n_1, ..., total]
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0).to(torch.int32)

    return {
        "expert_indices": expert_indices,
        "expert_weights": expert_weights.to(hidden.dtype),
        "token_indices_per_expert": token_indices_per_expert,
        "expert_offsets": expert_offsets,
        "sorted_token_indices": sorted_token_indices,
        "sorted_expert_indices": sorted_expert_indices,
        "sorted_weights": sorted_weights.to(hidden.dtype),
    }


def _dequant_expert_weight(packed_all, absmax_all, expert_idx, packed_per, absmax_per,
                           codebook, k, n_elements, N, N_padded, K, dtype):
    """Dequantize a single expert's weight from the concatenated flat-format tensors.

    Args:
        packed_all: All experts' packed weights concatenated [num_experts * packed_per]
        absmax_all: All experts' absmax concatenated [num_experts * absmax_per]
        expert_idx: Which expert to dequantize
        packed_per: Number of packed int32 elements per expert
        absmax_per: Number of absmax elements per expert
        codebook: Shared codebook
        k: Bit width
        n_elements: N_padded * K (total padded elements per expert)
        N: Original output dim
        N_padded: Padded output dim (multiple of 128)
        K: Input dim
        dtype: Output dtype

    Returns:
        Dequantized weight [N, K]
    """
    packed_e = packed_all[expert_idx * packed_per: (expert_idx + 1) * packed_per]
    absmax_e = absmax_all[expert_idx * absmax_per: (expert_idx + 1) * absmax_per]
    w_deq = dequantize_kbit(packed_e, absmax_e, codebook, k, n_elements, dtype)
    W = w_deq[:n_elements].reshape(N_padded, K)[:N, :]
    return W


class MoEExpertForward(torch.autograd.Function):
    """Chunked expert forward pass with differentiable backward.

    Expert weights are stored in flat kbit-packed format. Forward and backward
    dequantize per-expert weights and use cuBLAS matmul, processed in chunks
    to limit peak activation memory.

    Forward for each expert chunk:
        1. Gather tokens routed to these experts
        2. Per-expert: dequant gate weight, compute gate projection
        3. Per-expert: dequant up weight, compute up projection
        4. SwiGLU activation
        5. Per-expert: dequant down weight, compute down projection
        6. Weighted scatter-add results to output

    Backward recomputes forward per chunk (gradient-checkpoint style) to
    avoid saving intermediate activations.
    """

    @staticmethod
    def forward(
        ctx,
        hidden,                # [N_tokens, hidden_dim]
        sorted_token_indices,  # [total_assignments] from router
        sorted_weights,        # [total_assignments] from router
        expert_offsets,        # [num_experts + 1] cumulative counts
        gate_packed_all,       # flat-format packed gate weights, all experts concatenated
        gate_absmax_all,       # flat-format absmax gate weights, all experts concatenated
        up_packed_all,
        up_absmax_all,
        down_packed_all,
        down_absmax_all,
        codebook,
        k,                     # bit width
        hidden_dim,            # input/output dim (K for gate/up, N for down)
        intermediate_dim,      # MLP intermediate dim (N for gate/up, K for down)
        num_experts,
        expert_chunk_size,
    ):
        N_tokens = hidden.shape[0]
        device = hidden.device
        dtype = hidden.dtype

        output = torch.zeros(N_tokens, hidden_dim, device=device, dtype=dtype)

        # Compute per-expert packed sizes (all experts have same dims)
        gate_packed_per = gate_packed_all.numel() // num_experts
        gate_absmax_per = gate_absmax_all.numel() // num_experts
        up_packed_per = up_packed_all.numel() // num_experts
        up_absmax_per = up_absmax_all.numel() // num_experts
        down_packed_per = down_packed_all.numel() // num_experts
        down_absmax_per = down_absmax_all.numel() // num_experts

        # Padded dims for dequantization
        inter_padded = ((intermediate_dim + 127) // 128) * 128
        hidden_padded = ((hidden_dim + 127) // 128) * 128
        n_elements_gate = inter_padded * hidden_dim   # gate/up: [intermediate, hidden] mapped as [N_padded, K]
        n_elements_down = hidden_padded * intermediate_dim  # down: [hidden, intermediate] mapped as [N_padded, K]

        for chunk_start in range(0, num_experts, expert_chunk_size):
            chunk_end = min(chunk_start + expert_chunk_size, num_experts)

            # Global sorted range for this chunk
            g_start = expert_offsets[chunk_start].item()
            g_end = expert_offsets[chunk_end].item()
            if g_start == g_end:
                continue

            chunk_token_idx = sorted_token_indices[g_start:g_end]
            chunk_weights = sorted_weights[g_start:g_end]

            # Gather input tokens for this chunk
            A_concat = hidden[chunk_token_idx]  # [n_chunk, hidden_dim]

            # Process each expert in the chunk
            chunk_down_out = torch.zeros_like(A_concat)  # accumulate per-expert down outputs

            for e in range(chunk_start, chunk_end):
                e_start = expert_offsets[e].item() - g_start
                e_end = expert_offsets[e + 1].item() - g_start
                if e_start == e_end:
                    continue

                A_e = A_concat[e_start:e_end]  # [n_e, hidden_dim]

                # Gate projection
                W_gate = _dequant_expert_weight(
                    gate_packed_all, gate_absmax_all, e,
                    gate_packed_per, gate_absmax_per,
                    codebook, k, n_elements_gate,
                    intermediate_dim, inter_padded, hidden_dim, dtype,
                )
                gate_out = A_e @ W_gate.t()  # [n_e, intermediate_dim]

                # Up projection
                W_up = _dequant_expert_weight(
                    up_packed_all, up_absmax_all, e,
                    up_packed_per, up_absmax_per,
                    codebook, k, n_elements_gate,
                    intermediate_dim, inter_padded, hidden_dim, dtype,
                )
                up_out = A_e @ W_up.t()  # [n_e, intermediate_dim]

                # SwiGLU
                h = torch_F.silu(gate_out) * up_out  # [n_e, intermediate_dim]

                # Down projection
                W_down = _dequant_expert_weight(
                    down_packed_all, down_absmax_all, e,
                    down_packed_per, down_absmax_per,
                    codebook, k, n_elements_down,
                    hidden_dim, hidden_padded, intermediate_dim, dtype,
                )
                down_out = h @ W_down.t()  # [n_e, hidden_dim]

                chunk_down_out[e_start:e_end] = down_out

            # Weighted scatter-add to output
            weighted_out = chunk_down_out * chunk_weights.unsqueeze(1)
            output.index_add_(0, chunk_token_idx, weighted_out)

        # Save for backward (recompute intermediates per chunk)
        ctx.save_for_backward(
            hidden, sorted_token_indices, sorted_weights, expert_offsets,
            gate_packed_all, gate_absmax_all,
            up_packed_all, up_absmax_all,
            down_packed_all, down_absmax_all,
            codebook,
        )
        ctx.k = k
        ctx.hidden_dim = hidden_dim
        ctx.intermediate_dim = intermediate_dim
        ctx.num_experts = num_experts
        ctx.expert_chunk_size = expert_chunk_size
        ctx.compute_dtype = dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradient w.r.t. hidden input.

        Expert weights are frozen (kbit-quantized), so no weight gradients needed.
        Recomputes forward intermediates per chunk to limit memory.

        Backward through MoE expert MLP:
            dL/dhidden[token] += sum over assigned experts:
                weight * (dL/ddown_out @ W_gate + dL/dup_out @ W_up)
            where dL/ddown_out, dL/dup_out come from SwiGLU and down-projection backward.
        """
        (
            hidden, sorted_token_indices, sorted_weights, expert_offsets,
            gate_packed_all, gate_absmax_all,
            up_packed_all, up_absmax_all,
            down_packed_all, down_absmax_all,
            codebook,
        ) = ctx.saved_tensors

        k = ctx.k
        hidden_dim = ctx.hidden_dim
        intermediate_dim = ctx.intermediate_dim
        num_experts = ctx.num_experts
        expert_chunk_size = ctx.expert_chunk_size
        dtype = ctx.compute_dtype

        grad_hidden = torch.zeros_like(hidden)

        # Per-expert packed sizes
        gate_packed_per = gate_packed_all.numel() // num_experts
        gate_absmax_per = gate_absmax_all.numel() // num_experts
        up_packed_per = up_packed_all.numel() // num_experts
        up_absmax_per = up_absmax_all.numel() // num_experts
        down_packed_per = down_packed_all.numel() // num_experts
        down_absmax_per = down_absmax_all.numel() // num_experts

        inter_padded = ((intermediate_dim + 127) // 128) * 128
        hidden_padded = ((hidden_dim + 127) // 128) * 128
        n_elements_gate = inter_padded * hidden_dim
        n_elements_down = hidden_padded * intermediate_dim

        for chunk_start in range(0, num_experts, expert_chunk_size):
            chunk_end = min(chunk_start + expert_chunk_size, num_experts)

            g_start = expert_offsets[chunk_start].item()
            g_end = expert_offsets[chunk_end].item()
            if g_start == g_end:
                continue

            chunk_token_idx = sorted_token_indices[g_start:g_end]
            chunk_weights = sorted_weights[g_start:g_end]

            # Gather input and grad_output for this chunk
            A_concat = hidden[chunk_token_idx]
            grad_out_chunk = grad_output[chunk_token_idx]  # [n_chunk, hidden_dim]

            # Per-expert backward
            grad_A_chunk = torch.zeros_like(A_concat)

            for e in range(chunk_start, chunk_end):
                e_start = expert_offsets[e].item() - g_start
                e_end = expert_offsets[e + 1].item() - g_start
                if e_start == e_end:
                    continue

                A_e = A_concat[e_start:e_end]
                e_weights = chunk_weights[e_start:e_end]
                grad_out_e = grad_out_chunk[e_start:e_end] * e_weights.unsqueeze(1)

                # --- Recompute forward ---
                W_gate = _dequant_expert_weight(
                    gate_packed_all, gate_absmax_all, e,
                    gate_packed_per, gate_absmax_per,
                    codebook, k, n_elements_gate,
                    intermediate_dim, inter_padded, hidden_dim, dtype,
                )
                gate_out = A_e @ W_gate.t()

                W_up = _dequant_expert_weight(
                    up_packed_all, up_absmax_all, e,
                    up_packed_per, up_absmax_per,
                    codebook, k, n_elements_gate,
                    intermediate_dim, inter_padded, hidden_dim, dtype,
                )
                up_out = A_e @ W_up.t()

                sig_e = torch.sigmoid(gate_out)
                silu_e = gate_out * sig_e

                # --- Down projection backward ---
                # h = silu_e * up_out
                h = silu_e * up_out

                W_down = _dequant_expert_weight(
                    down_packed_all, down_absmax_all, e,
                    down_packed_per, down_absmax_per,
                    codebook, k, n_elements_down,
                    hidden_dim, hidden_padded, intermediate_dim, dtype,
                )
                # Forward: down_out = h @ W_down^T
                # Backward: dL/dh = grad_out_e @ W_down
                grad_h = grad_out_e @ W_down  # [n_e, intermediate_dim]

                # --- SwiGLU backward ---
                # h = silu(gate_out) * up_out
                # dh/d(gate_out) = up_out * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
                # dh/d(up_out) = silu(e)
                grad_gate = grad_h * up_out * sig_e * (1.0 + gate_out * (1.0 - sig_e))
                grad_up = grad_h * silu_e

                # --- Gate/Up projection backward ---
                # gate_out = A_e @ W_gate^T => dL/dA += grad_gate @ W_gate
                # up_out = A_e @ W_up^T   => dL/dA += grad_up @ W_up
                grad_A_e = grad_gate @ W_gate + grad_up @ W_up  # [n_e, hidden_dim]
                grad_A_chunk[e_start:e_end] = grad_A_e

            # Scatter-add gradients back to grad_hidden
            grad_hidden.index_add_(0, chunk_token_idx, grad_A_chunk)

        # Return gradients: only hidden gets gradient, all others are non-differentiable
        return (grad_hidden,) + (None,) * 15


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

    Expert weights must be in flat kbit-packed format (from quantize_kbit),
    concatenated across all experts. Each expert's weight is quantized
    separately and then concatenated.

    Args:
        hidden: Input hidden states [N_tokens, hidden_dim].
        router_result: Output from moe_router_dispatch.
        gate_packed_all: All expert gate packed weights, concatenated.
        gate_absmax_all: All expert gate absmax, concatenated.
        up_packed_all: All expert up packed weights, concatenated.
        up_absmax_all: All expert up absmax, concatenated.
        down_packed_all: All expert down packed weights, concatenated.
        down_absmax_all: All expert down absmax, concatenated.
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
        hidden,
        router_result["sorted_token_indices"],
        router_result["sorted_weights"],
        router_result["expert_offsets"],
        gate_packed_all, gate_absmax_all,
        up_packed_all, up_absmax_all,
        down_packed_all, down_absmax_all,
        codebook, k, hidden_dim, intermediate_dim,
        num_experts, expert_chunk_size,
    )
