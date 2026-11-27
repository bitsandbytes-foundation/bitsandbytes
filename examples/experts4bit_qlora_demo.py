"""QLoRA-style training of fused MoE experts on a frozen ``Experts4bit`` base.

This is a *reference pattern*, intentionally **not** part of the bitsandbytes public API. It
shows that the ``Experts4bit`` 4-bit storage primitive can serve as the frozen base of a
QLoRA-style fine-tune of fused Mixture-of-Experts weights: the 4-bit expert weights stay
frozen, and small per-expert low-rank (LoRA) adapters are the only trainable parameters.

The adapter wiring shown here is the kind of thing that would ultimately live in PEFT /
Unsloth rather than in bitsandbytes itself — the point of this file is to demonstrate that
the base primitive is already differentiable and trainable as a frozen base today.

Run:
    python examples/experts4bit_qlora_demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.nn import Experts4bit


class ExpertsLoRA(nn.Module):
    """Per-expert LoRA adapters over a frozen :class:`Experts4bit` base.

    For each expert ``e``, the two frozen 4-bit projections are augmented with a trainable
    low-rank term ``scaling * (x @ A[e].T) @ B[e].T``:

      * ``gate_up``: ``A[e]`` is ``[r, hidden]``, ``B[e]`` is ``[gate_up_out, r]``
      * ``down``:    ``A[e]`` is ``[r, intermediate]``, ``B[e]`` is ``[hidden, r]``

    ``B`` is initialised to zero, so the adapted module is identical to the frozen base at
    step 0 and only departs from it as the adapters train (standard LoRA initialisation).
    """

    def __init__(self, base: Experts4bit, r: int = 8, alpha: int = 16, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r

        num_experts = base.num_experts
        gate_up_out, hidden = base._gate_up_shape  # [2*intermediate (or intermediate), hidden]
        _, intermediate = base._down_shape  # [hidden, intermediate]

        self.gate_up_lora_A = nn.Parameter(torch.empty(num_experts, r, hidden, dtype=dtype))
        self.gate_up_lora_B = nn.Parameter(torch.zeros(num_experts, gate_up_out, r, dtype=dtype))
        self.down_lora_A = nn.Parameter(torch.empty(num_experts, r, intermediate, dtype=dtype))
        self.down_lora_B = nn.Parameter(torch.zeros(num_experts, hidden, r, dtype=dtype))

        # A ~ small random, B = 0  =>  the initial LoRA delta is exactly zero.
        nn.init.normal_(self.gate_up_lora_A, std=1.0 / r)
        nn.init.normal_(self.down_lora_A, std=1.0 / r)

    def _lora(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # x: [n, in]; A: [r, in]; B: [out, r]  ->  [n, out]
        return self.scaling * F.linear(F.linear(x, A), B)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        compute_dtype = base.compute_dtype if base.compute_dtype is not None else hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)

        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]

            # Frozen 4-bit base projection + trainable low-rank delta.
            gate_up_w = base._dequantize_expert(
                base.gate_up_proj, base.gate_up_absmax, base._gate_up_shape, expert_idx, compute_dtype
            )
            proj = F.linear(x, gate_up_w) + self._lora(
                x, self.gate_up_lora_A[expert_idx], self.gate_up_lora_B[expert_idx]
            )

            if base.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = base.act_fn(gate) * up
            else:
                current_hidden = base.act_fn(proj)

            down_w = base._dequantize_expert(
                base.down_proj, base.down_absmax, base._down_shape, expert_idx, compute_dtype
            )
            current_hidden = F.linear(current_hidden, down_w) + self._lora(
                current_hidden, self.down_lora_A[expert_idx], self.down_lora_B[expert_idx]
            )

            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


def main() -> None:
    torch.manual_seed(0)

    num_experts, hidden, intermediate = 8, 128, 256
    num_tokens, top_k = 64, 2

    # A full-precision fused-expert stack (the shape transformers v5 stores MoE experts in).
    gate_up = torch.randn(num_experts, 2 * intermediate, hidden) * 0.1
    down = torch.randn(num_experts, hidden, intermediate) * 0.1

    # Freeze it in 4-bit, then attach trainable LoRA adapters.
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
    model = ExpertsLoRA(base, r=8, alpha=16)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_base_bytes = base.gate_up_proj.numel() + base.down_proj.numel()
    print(f"trainable LoRA params: {n_train:,}   frozen packed base bytes: {n_base_bytes:,}")

    hidden_states = torch.randn(num_tokens, hidden)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    target = torch.randn(num_tokens, hidden)

    gate_up_before = base.gate_up_proj.clone()

    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    print("\nstep   loss")
    for step in range(50):
        optimizer.zero_grad()
        out = model(hidden_states, top_k_index, top_k_weights)
        loss = F.mse_loss(out, target)
        loss.backward()
        assert base.gate_up_proj.grad is None, "frozen base must never receive a gradient"
        optimizer.step()
        if step % 10 == 0 or step == 49:
            print(f"{step:4d}   {loss.item():.5f}")

    assert torch.equal(base.gate_up_proj, gate_up_before), "frozen base bytes must be unchanged"
    print("\nbase packed weights unchanged after training:", torch.equal(base.gate_up_proj, gate_up_before))


if __name__ == "__main__":
    main()
