import copy

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Experts4bit


@pytest.fixture
def experts_kwargs():
    return dict(num_experts=4, hidden_dim=128, intermediate_dim=256, quant_type="nf4", blocksize=64)


@pytest.fixture
def fp_weights(experts_kwargs):
    """Return randomly initialised fp16 fused-expert weights on CUDA."""
    n = experts_kwargs["num_experts"]
    h = experts_kwargs["hidden_dim"]
    i = experts_kwargs["intermediate_dim"]
    device = "cuda"
    gate_up_proj = torch.randn(n, 2 * i, h, dtype=torch.float16, device=device)
    down_proj = torch.randn(n, h, i, dtype=torch.float16, device=device)
    return gate_up_proj, down_proj


# Quantisation round-trip: quantise each expert and dequantise back, checking
# that the result is within expected 4-bit tolerance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quant_round_trip(quant_type, input_dtype, experts_kwargs, fp_weights):
    gate_up_proj, down_proj = fp_weights
    gate_up_proj = gate_up_proj.to(input_dtype)
    down_proj = down_proj.to(input_dtype)

    module = Experts4bit.from_float(gate_up_proj, down_proj, quant_type=quant_type)

    for expert_idx in range(module.num_experts):
        # Dequantise gate_up weights
        w_gu = module._dequantize_expert(
            module.gate_up_packed[expert_idx],
            module.gate_up_absmax[expert_idx],
            2 * module.intermediate_dim,
            module.hidden_dim,
        )
        original_gu = gate_up_proj[expert_idx].to(torch.float32)
        err_gu = (w_gu.to(torch.float32) - original_gu).abs().mean()
        assert err_gu < 0.20, f"gate_up expert {expert_idx} mean-abs error {err_gu:.5f} >= 0.20"

        # Dequantise down weights
        w_down = module._dequantize_expert(
            module.down_packed[expert_idx],
            module.down_absmax[expert_idx],
            module.hidden_dim,
            module.intermediate_dim,
        )
        original_d = down_proj[expert_idx].to(torch.float32)
        err_d = (w_down.to(torch.float32) - original_d).abs().mean()
        assert err_d < 0.20, f"down expert {expert_idx} mean-abs error {err_d:.5f} >= 0.20"

    # Check shapes and dtypes of packed weights and absmax
    assert module.gate_up_packed.dtype == torch.uint8
    assert module.down_packed.dtype == torch.uint8
    assert module.gate_up_absmax.dtype == torch.float32
    assert module.down_absmax.dtype == torch.float32

    gate_up_out = 2 * module.intermediate_dim if module.has_activation else module.intermediate_dim
    gu_blocks = module.hidden_dim // module.blocksize
    down_blocks = module.intermediate_dim // module.blocksize
    assert module.gate_up_absmax.shape == (module.num_experts, gate_up_out * gu_blocks)
    assert module.down_absmax.shape == (module.num_experts, module.hidden_dim * down_blocks)


def test_from_float_shape_validation(experts_kwargs, fp_weights):
    gate_up_proj, down_proj = fp_weights
    module = Experts4bit.from_float(gate_up_proj, down_proj)
    assert module.num_experts == 4
    assert module.hidden_dim == 128
    assert module.intermediate_dim == 256
    assert module.quant_type == "nf4"


def test_invalid_blocksize():
    with pytest.raises(ValueError, match="hidden_dim.*divisible by blocksize"):
        Experts4bit(num_experts=2, hidden_dim=100, intermediate_dim=64, blocksize=64)


def test_invalid_quant_type(fp_weights):
    gate_up_proj, down_proj = fp_weights
    with pytest.raises((ValueError, NotImplementedError)):
        Experts4bit.from_float(gate_up_proj, down_proj, quant_type="invalid")


# Forward pass correctness vs. a full-precision reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("has_activation", [True, False])
def test_forward_vs_reference(has_activation, experts_kwargs, fp_weights):
    gate_up_proj, down_proj = fp_weights
    n = experts_kwargs["num_experts"]
    h = experts_kwargs["hidden_dim"]
    i = experts_kwargs["intermediate_dim"]
    k = 2

    if not has_activation:
        gate_up_proj = gate_up_proj[:, :i, :]  # remove the gate split

    module = Experts4bit.from_float(
        gate_up_proj, down_proj, quant_type="nf4", has_activation=has_activation,
    )

    hidden = torch.randn(2, 4, h, dtype=torch.float16, device="cuda")
    top_k_idx = torch.randint(0, n, (2, 4, k), device="cuda")
    top_k_w = torch.softmax(torch.randn(2, 4, k, device="cuda"), dim=-1)

    # Reference: dequantize all weights and compute in fp16 (same as forward)
    gate_up_out = 2 * i if has_activation else i
    ref_output = torch.zeros_like(hidden)

    for expert_idx in range(n):
        w_gu_ref = module._dequantize_expert(
            module.gate_up_packed[expert_idx],
            module.gate_up_absmax[expert_idx],
            gate_up_out, h,
        )
        w_down_ref = module._dequantize_expert(
            module.down_packed[expert_idx],
            module.down_absmax[expert_idx],
            h, i,
        )
        for batch in range(2):
            for seq in range(4):
                for k_idx in range(k):
                    if top_k_idx[batch, seq, k_idx] == expert_idx:
                        x = hidden[batch, seq]
                        if has_activation:
                            gate, up = w_gu_ref.chunk(2, dim=0)
                            intermediate = torch.nn.functional.silu(gate @ x) * (up @ x)
                        else:
                            intermediate = w_gu_ref @ x
                        expert_out = w_down_ref @ intermediate
                        ref_output[batch, seq] += top_k_w[batch, seq, k_idx] * expert_out

    module_output = module(hidden, top_k_idx, top_k_w)

    torch.testing.assert_close(
        module_output.to(torch.float32), ref_output.to(torch.float32),
        rtol=5e-2, atol=5e-2,
    )


# state_dict round-trip: bit-exact restore of packed weights + absmax
# ---------------------------------------------------------------------------
def test_state_dict_round_trip(experts_kwargs, fp_weights):
    gate_up_proj, down_proj = fp_weights
    module = Experts4bit.from_float(gate_up_proj, down_proj)

    # Save and restore
    sd = module.state_dict()
    module2 = Experts4bit(**experts_kwargs, device="cuda")
    module2.load_state_dict(sd)

    # Check that packed weights and absmax match exactly
    assert torch.equal(module.gate_up_packed, module2.gate_up_packed)
    assert torch.equal(module.down_packed, module2.down_packed)
    assert torch.equal(module.gate_up_absmax, module2.gate_up_absmax)
    assert torch.equal(module.down_absmax, module2.down_absmax)

    # Forward pass after restore should produce identical output
    hidden = torch.randn(2, 4, 128, dtype=torch.float16, device="cuda")
    top_k_idx = torch.randint(0, 4, (2, 4, 2), device="cuda")
    top_k_w = torch.softmax(torch.randn(2, 4, 2, device="cuda"), dim=-1)

    out1 = module(hidden, top_k_idx, top_k_w)
    out2 = module2(hidden, top_k_idx, top_k_w)
    assert torch.equal(out1, out2), "state_dict round-trip outputs differ"
