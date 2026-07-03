# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Callable
import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

import bitsandbytes.functional as F
from bitsandbytes.functional import QuantState


class _FrozenLinearRecomputeBackward(torch.autograd.Function):
    """``F.linear`` against a frozen dequantized weight, re-dequantizing it in backward.

    The weight produced by ``dequant_fn`` (a closure over the packed buffers) is an
    intermediate, not a Parameter, so a plain ``F.linear`` would stash it as a saved
    activation for the whole forward-to-backward window — one full-precision expert
    weight per projection per layer. Because the base is frozen, backward needs no
    gradient for the weight and only computes ``grad_output @ weight``; the weight can
    therefore be dropped after the forward matmul and re-dequantized on demand, keeping
    training memory independent of the number of experts held between forward and
    backward. Numerically identical to dequantize-then-``linear`` by construction — the
    forward *is* dequantize-then-linear; recomputation only changes what is saved, never
    what is computed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dequant_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        ctx.dequant_fn = dequant_fn
        return F_nn.linear(x, dequant_fn())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ ctx.dequant_fn()
        return grad_x, None


class Experts4bit(nn.Module):
    """4-bit quantized storage for fused Mixture-of-Experts expert weights.

    A growing number of models in the Hugging Face ecosystem store their MoE expert
    weights as a single 3D ``nn.Parameter`` of shape ``[num_experts, out_features,
    in_features]`` (e.g. ``OlmoeExperts``, ``Qwen3MoeExperts``) rather than as a
    collection of ``nn.Linear`` layers. The default 4-bit quantization walker only
    replaces ``nn.Linear`` modules, so these fused experts are silently skipped and
    stay in full precision — the dominant contribution to the model's memory footprint
    (see https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849).

    ``Experts4bit`` holds the two expert projections (``gate_up_proj`` and ``down_proj``)
    in 4-bit NF4/FP4 precision. Unlike :class:`Linear4bit`, the packed weights are kept
    as plain ``nn.Parameter`` buffers and the per-expert quantization statistics
    (``absmax``) live on the module as ordinary buffers. This avoids bending
    :class:`Params4bit`'s tensor-subclass and device-movement machinery around a 3D
    stack, and it means the module serializes through the standard ``state_dict``
    mechanism with no custom save/load hooks.

    The forward pass dequantizes a single expert at a time (a per-expert loop), mirroring
    the reference fused-experts forward. In training, the dequantized weight is not kept
    as a saved activation: it is re-dequantized on demand in backward (see
    :class:`_FrozenLinearRecomputeBackward`), so activation memory stays independent of
    the number of experts. Grouped-GEMM is intentionally left for future work.

    <Tip warning={true}>This feature is experimental and may change in future releases.</Tip>

    Args:
        num_experts (`int`): Number of experts in the layer.
        hidden_dim (`int`): Model hidden size (the ``in_features`` of ``gate_up_proj`` and
            the ``out_features`` of ``down_proj``).
        intermediate_dim (`int`): Expert intermediate size (the ``in_features`` of
            ``down_proj``).
        has_gate (`bool`, *optional*, defaults to `True`): Whether ``gate_up_proj`` packs a
            gate and an up projection (SwiGLU-style). When `False`, the projection is a
            plain up projection of size ``intermediate_dim``.
        activation (`Callable`, *optional*): The activation applied to the gate. Defaults
            to ``torch.nn.functional.silu`` (SwiGLU), matching OLMoE / Qwen3-MoE.
        compute_dtype (`torch.dtype`, *optional*): The dtype expert weights are
            dequantized to for the matmul. When `None`, the input's dtype is used.
        quant_type (`str`, *optional*, defaults to `"nf4"`): The 4-bit data type, ``nf4``
            or ``fp4``.
        blocksize (`int`, *optional*, defaults to `64`): The quantization block size.
        device (*optional*): The device for the (empty) packed buffers.

    Raises:
        ValueError: If ``quant_type`` is invalid, or if ``hidden_dim`` / ``intermediate_dim``
            is not divisible by ``blocksize`` (required so per-expert quantization blocks
            never straddle an expert boundary).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        has_gate: bool = True,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
        blocksize: int = 64,
        device=None,
    ):
        super().__init__()

        if quant_type not in ("nf4", "fp4"):
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type!r}")

        # Each expert is quantized independently, so an expert occupies a contiguous
        # `out_features * in_features` run of elements. Requiring the in_features dim to
        # be a multiple of the blocksize guarantees `out_features * in_features` is too,
        # so blocks tile each expert exactly and absmax reshapes cleanly to
        # [num_experts, blocks_per_expert]. (gate_up in_features is hidden_dim; down_proj
        # in_features is intermediate_dim.)
        for name, in_features in (("hidden_dim", hidden_dim), ("intermediate_dim", intermediate_dim)):
            if in_features % blocksize != 0:
                raise ValueError(
                    f"{name} ({in_features}) must be divisible by blocksize ({blocksize}) "
                    "so per-expert quantization blocks align with expert boundaries"
                )

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.has_gate = has_gate
        self.act_fn = activation if activation is not None else F_nn.silu
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.blocksize = blocksize

        gate_up_out = 2 * intermediate_dim if has_gate else intermediate_dim
        self._gate_up_shape = (gate_up_out, hidden_dim)
        self._down_shape = (hidden_dim, intermediate_dim)

        gate_up_numel = gate_up_out * hidden_dim
        down_numel = hidden_dim * intermediate_dim

        # Packed 4-bit weights as plain (frozen) parameters: two 4-bit values per byte.
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, gate_up_numel // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, down_numel // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )

        # Per-expert quantization scales.
        self.register_buffer(
            "gate_up_absmax",
            torch.empty(num_experts, gate_up_numel // blocksize, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "down_absmax",
            torch.empty(num_experts, down_numel // blocksize, dtype=torch.float32, device=device),
        )

        # The 4-bit codebook is identical for every expert and fully determined by
        # quant_type, so it is reconstructed at init rather than serialized.
        self.register_buffer("code", F.get_4bit_type(quant_type, device=device), persistent=False)

    @classmethod
    def from_float(
        cls,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        has_gate: bool = True,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
        blocksize: int = 64,
    ) -> "Experts4bit":
        """Build an :class:`Experts4bit` by quantizing full-precision expert weights.

        Args:
            gate_up_proj (`torch.Tensor`): Shape ``[num_experts, gate_up_out, hidden_dim]``,
                where ``gate_up_out`` is ``2 * intermediate_dim`` when ``has_gate`` else
                ``intermediate_dim``.
            down_proj (`torch.Tensor`): Shape ``[num_experts, hidden_dim, intermediate_dim]``.

        Returns:
            `Experts4bit`: A module holding the quantized weights on the inputs' device.
        """
        if gate_up_proj.dim() != 3 or down_proj.dim() != 3:
            raise ValueError("gate_up_proj and down_proj must be 3D [num_experts, out, in] tensors")

        num_experts, _, hidden_dim = gate_up_proj.shape
        intermediate_dim = down_proj.shape[2]

        module = cls(
            num_experts,
            hidden_dim,
            intermediate_dim,
            has_gate=has_gate,
            activation=activation,
            compute_dtype=compute_dtype if compute_dtype is not None else gate_up_proj.dtype,
            quant_type=quant_type,
            blocksize=blocksize,
            device=gate_up_proj.device,
        )

        gate_up_packed, gate_up_absmax = module._quantize_stack(gate_up_proj)
        down_packed, down_absmax = module._quantize_stack(down_proj)

        module.gate_up_proj = nn.Parameter(gate_up_packed, requires_grad=False)
        module.down_proj = nn.Parameter(down_packed, requires_grad=False)
        module.gate_up_absmax = gate_up_absmax
        module.down_absmax = down_absmax
        return module

    def _quantize_stack(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a ``[num_experts, out, in]`` stack to packed bytes + per-expert absmax."""
        packed = []
        absmax = []
        for e in range(weights.shape[0]):
            q, state = F.quantize_4bit(
                weights[e].contiguous(),
                blocksize=self.blocksize,
                compress_statistics=False,
                quant_type=self.quant_type,
            )
            packed.append(q.reshape(-1))
            absmax.append(state.absmax.reshape(-1))
        return torch.stack(packed), torch.stack(absmax)

    def _dequantize_expert(
        self,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        shape: tuple[int, int],
        expert_idx: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize a single expert's 2D weight ``[out, in]`` for the matmul."""
        quant_state = QuantState(
            absmax=absmax[expert_idx],
            shape=torch.Size(shape),
            code=self.code,
            blocksize=self.blocksize,
            quant_type=self.quant_type,
            dtype=dtype,
        )
        # Restore the [packed, 1] layout quantize_4bit emits (and which keeps the
        # transpose back-compat shim — keyed on A.shape[0] == 1 — from firing).
        return F.dequantize_4bit(packed[expert_idx].reshape(-1, 1), quant_state=quant_state)

    def _project(self, packed, absmax, shape, expert_idx, x, compute_dtype):
        """One expert projection: dequantize + ``linear``, re-dequantizing in backward.

        The recompute closure is just :meth:`_dequantize_expert`; no gradient is ever
        produced for the frozen packed storage.
        """
        dequant_fn = functools.partial(self._dequantize_expert, packed, absmax, shape, expert_idx, compute_dtype)
        return _FrozenLinearRecomputeBackward.apply(x, dequant_fn)


    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = self.compute_dtype if self.compute_dtype is not None else hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)

        # Accumulate in float32 for numerical stability with bf16/fp16 routing weights.
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = F_nn.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            proj = self._project(
                self.gate_up_proj, self.gate_up_absmax, self._gate_up_shape, expert_idx, current_state, compute_dtype
            )
            if self.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = self.act_fn(gate) * up
            else:
                current_hidden = self.act_fn(proj)

            current_hidden = self._project(
                self.down_proj, self.down_absmax, self._down_shape, expert_idx, current_hidden, compute_dtype
            )
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)
