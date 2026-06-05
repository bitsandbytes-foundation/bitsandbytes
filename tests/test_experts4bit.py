import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Experts4bit
from tests.helpers import describe_dtype, get_available_devices, id_formatter

# Small but representative MoE dims. hidden_dim and intermediate_dim are both multiples
# of the default blocksize (64), as required by Experts4bit.
NUM_EXPERTS = 4
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 128
TOP_K = 2
NUM_TOKENS = 12


def _random_expert_weights(dtype, device, has_gate=True):
    gate_up_out = 2 * INTERMEDIATE_DIM if has_gate else INTERMEDIATE_DIM
    gate_up = torch.randn(NUM_EXPERTS, gate_up_out, HIDDEN_DIM, dtype=dtype, device=device) * 0.1
    down = torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=dtype, device=device) * 0.1
    return gate_up, down


def _random_routing(device):
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_DIM, device=device)
    top_k_index = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), device=device)
    top_k_weights = torch.softmax(torch.randn(NUM_TOKENS, TOP_K, device=device), dim=-1)
    return hidden_states, top_k_index, top_k_weights


def _reference_forward(gate_up, down, hidden_states, top_k_index, top_k_weights, act_fn=torch.nn.functional.silu):
    """Plain full-precision fused-experts forward (mirrors OlmoeExperts.forward)."""
    compute_dtype = gate_up.dtype
    hidden_states = hidden_states.to(compute_dtype)
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=gate_up.shape[0]).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)
    for expert_idx in expert_hit:
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(current_state, gate_up[expert_idx]).chunk(2, dim=-1)
        current = act_fn(gate) * up
        current = torch.nn.functional.linear(current, down[expert_idx])
        current = current * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final.to(hidden_states.dtype)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_experts4bit_roundtrip(device, dtype, quant_type):
    gate_up, down = _random_expert_weights(dtype, device)
    module = Experts4bit.from_float(gate_up, down, quant_type=quant_type)

    # Packed-weight and absmax shapes/dtypes.
    gate_up_out = 2 * INTERMEDIATE_DIM
    assert module.gate_up_proj.dtype == torch.uint8
    assert module.gate_up_proj.shape == (NUM_EXPERTS, gate_up_out * HIDDEN_DIM // 2)
    assert module.down_proj.shape == (NUM_EXPERTS, HIDDEN_DIM * INTERMEDIATE_DIM // 2)
    assert module.gate_up_absmax.shape == (NUM_EXPERTS, gate_up_out * HIDDEN_DIM // module.blocksize)
    assert not module.gate_up_proj.requires_grad

    # Per-expert dequantization round-trips within 4-bit tolerance.
    for e in range(NUM_EXPERTS):
        deq = module._dequantize_expert(module.gate_up_proj, module.gate_up_absmax, module._gate_up_shape, e, dtype)
        assert deq.shape == (gate_up_out, HIDDEN_DIM)
        assert deq.dtype == dtype
        torch.testing.assert_close(deq.float(), gate_up[e].float(), rtol=0.15, atol=0.05)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("has_gate", [True, False], ids=id_formatter("has_gate"))
def test_experts4bit_forward_matches_reference(device, has_gate):
    # float32 compute so the only difference vs. the reference is float accumulation order.
    gate_up, down = _random_expert_weights(torch.float32, device, has_gate=has_gate)
    module = Experts4bit.from_float(gate_up, down, has_gate=has_gate, compute_dtype=torch.float32)

    hidden_states, top_k_index, top_k_weights = _random_routing(device)

    # Reference uses the exact weights the module holds internally (dequantized bytes),
    # isolating forward/routing correctness from quantization error.
    gate_up_deq = torch.stack(
        [
            module._dequantize_expert(
                module.gate_up_proj, module.gate_up_absmax, module._gate_up_shape, e, torch.float32
            )
            for e in range(NUM_EXPERTS)
        ]
    )
    down_deq = torch.stack(
        [
            module._dequantize_expert(module.down_proj, module.down_absmax, module._down_shape, e, torch.float32)
            for e in range(NUM_EXPERTS)
        ]
    )

    if has_gate:
        ref = _reference_forward(gate_up_deq, down_deq, hidden_states, top_k_index, top_k_weights)
    else:
        # no-gate reference: act_fn applied to the whole projection
        ref = torch.zeros_like(hidden_states, dtype=torch.float32)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=NUM_EXPERTS).permute(2, 1, 0)
        for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            cur = torch.nn.functional.silu(
                torch.nn.functional.linear(hidden_states[token_idx], gate_up_deq[expert_idx])
            )
            cur = torch.nn.functional.linear(cur, down_deq[expert_idx])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            ref.index_add_(0, token_idx, cur)
        ref = ref.to(hidden_states.dtype)

    out = module(hidden_states, top_k_index, top_k_weights)
    assert out.shape == hidden_states.shape
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", get_available_devices())
def test_experts4bit_state_dict_roundtrip(device):
    gate_up, down = _random_expert_weights(torch.float16, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float16)

    # Default state_dict carries everything (plain Parameters + buffers — no custom hooks).
    sd = module.state_dict()
    assert "gate_up_proj" in sd and "down_proj" in sd
    assert "gate_up_absmax" in sd and "down_absmax" in sd
    assert "code" not in sd  # codebook is non-persistent (reconstructed at init)

    reloaded = Experts4bit(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, compute_dtype=torch.float16, device=device)
    missing, unexpected = reloaded.load_state_dict(sd, strict=True), None
    assert missing.missing_keys == [] and missing.unexpected_keys == []

    # Bit-exact restore of packed weights + absmax.
    torch.testing.assert_close(reloaded.gate_up_proj, module.gate_up_proj, rtol=0, atol=0)
    torch.testing.assert_close(reloaded.down_absmax, module.down_absmax, rtol=0, atol=0)

    # Identical forward after reload.
    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    out_a = module(hidden_states, top_k_index, top_k_weights)
    out_b = reloaded(hidden_states, top_k_index, top_k_weights)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


def test_experts4bit_blocksize_validation():
    # in_features (hidden_dim / intermediate_dim) must be divisible by blocksize.
    with pytest.raises(ValueError, match="divisible by blocksize"):
        Experts4bit(NUM_EXPERTS, hidden_dim=100, intermediate_dim=128, blocksize=64)
    with pytest.raises(ValueError, match="divisible by blocksize"):
        Experts4bit(NUM_EXPERTS, hidden_dim=64, intermediate_dim=100, blocksize=64)
    with pytest.raises(ValueError, match="quant_type"):
        Experts4bit(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, quant_type="int4")


def test_experts4bit_is_exported():
    assert bnb.nn.Experts4bit is Experts4bit
