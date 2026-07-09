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


# --- Backward / autograd ---------------------------------------------------------------
# Experts4bit is a *frozen* 4-bit base: the packed weights are requires_grad=False, so they
# never receive gradients, but the per-expert dequant + linear + index_add_ forward is fully
# differentiable w.r.t. the input activations. That makes it usable as the frozen base of a
# QLoRA-style setup (gradients flow to adapters/earlier layers, not to the quantized weights).
# These tests lock that contract in.


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
def test_experts4bit_backward_flows_to_input(device, dtype):
    gate_up, down = _random_expert_weights(dtype, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=dtype)

    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    hidden_states = hidden_states.to(dtype).detach().requires_grad_(True)

    out = module(hidden_states, top_k_index, top_k_weights)
    out.float().sum().backward()

    # Gradient reaches the input activations, is finite, and is nonzero (every token is routed
    # to TOP_K experts here, so every row contributes).
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.float().abs().sum() > 0


@pytest.mark.parametrize("device", get_available_devices())
def test_experts4bit_base_weights_stay_frozen(device):
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)

    # Packed weights are frozen by construction ...
    assert module.gate_up_proj.requires_grad is False
    assert module.down_proj.requires_grad is False

    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    hidden_states = hidden_states.requires_grad_(True)
    module(hidden_states, top_k_index, top_k_weights).sum().backward()

    # ... and a backward pass leaves no gradient on them (so an optimizer can never nudge the
    # quantized base, and the absmax buffers are not trainable either).
    assert module.gate_up_proj.grad is None
    assert module.down_proj.grad is None


@pytest.mark.parametrize("device", get_available_devices())
def test_experts4bit_backward_matches_reference(device):
    # float32 throughout: the module's autograd path must match a plain full-precision forward
    # built from the *same* dequantized weights, isolating gradient correctness from quant error.
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)

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

    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    hs_mod = hidden_states.detach().clone().requires_grad_(True)
    hs_ref = hidden_states.detach().clone().requires_grad_(True)

    out_mod = module(hs_mod, top_k_index, top_k_weights)
    out_ref = _reference_forward(gate_up_deq, down_deq, hs_ref, top_k_index, top_k_weights)

    out_mod.sum().backward()
    out_ref.sum().backward()

    torch.testing.assert_close(out_mod, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hs_mod.grad, hs_ref.grad, rtol=1e-4, atol=1e-4)


def _load_experts_lora():
    """Load the ExpertsLoRA reference wrapper from examples/ (kept out of the bnb API)."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "examples", "experts4bit_qlora_demo.py")
    spec = importlib.util.spec_from_file_location("experts4bit_qlora_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpertsLoRA


def test_experts4bit_lora_training_reduces_loss():
    # End-to-end QLoRA-style step: a frozen 4-bit Experts4bit base + trainable per-expert LoRA.
    # Proves the primitive supports training today — only the adapters move, the base stays put.
    torch.manual_seed(0)
    experts_lora = _load_experts_lora()

    gate_up, down = _random_expert_weights(torch.float32, "cpu")
    base = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)
    model = experts_lora(base, r=4, alpha=8)

    # Only LoRA adapters are trainable; the 4-bit base is frozen.
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names and all("lora" in name for name in trainable_names)

    gate_up_before = base.gate_up_proj.clone()
    down_before = base.down_proj.clone()

    hidden_states, top_k_index, top_k_weights = _random_routing("cpu")
    target = torch.randn_like(hidden_states)

    # Standard LoRA init (B=0) => the adapted forward equals the frozen base forward at step 0.
    torch.testing.assert_close(
        model(hidden_states, top_k_index, top_k_weights),
        base(hidden_states, top_k_index, top_k_weights),
        rtol=1e-5,
        atol=1e-5,
    )

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(hidden_states, top_k_index, top_k_weights), target)
        loss.backward()
        assert base.gate_up_proj.grad is None and base.down_proj.grad is None
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]  # training reduces loss
    # The frozen 4-bit base is bit-identical before and after training.
    assert torch.equal(base.gate_up_proj, gate_up_before)
    assert torch.equal(base.down_proj, down_before)


# --- #1849 regression + shape coverage ------------------------------------------------------------


def test_experts4bit_1849_regression_fused_experts_get_quantized():
    """Regression for #1849. transformers v5 stores MoE experts as a single fused 3D ``nn.Parameter``
    (``[num_experts, out, in]``, e.g. ``Qwen3MoeExperts``), which the default 4-bit walker skips because
    there is no ``nn.Linear`` to replace — so the experts stay full-precision and dominate memory.
    ``Experts4bit`` is the fix: it actually 4-bit-quantizes the fused stack. Assert (a) the fused module
    exposes no ``nn.Linear`` for the walker to catch, and (b) ``from_float`` yields ``uint8``-packed 4-bit
    weights materially smaller than the fp16 originals."""
    num_experts, hidden, inter = 4, 128, 256

    class FusedExperts(torch.nn.Module):  # mirrors OlmoeExperts / Qwen3MoeExperts from #1849
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
            self.down_proj = torch.nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)

    fused = FusedExperts()
    # (a) the walker's target type is absent -> a Linear4bit conversion would be a silent no-op here.
    assert not any(isinstance(m, torch.nn.Linear) for m in fused.modules())
    fp16_bytes = (fused.gate_up_proj.numel() + fused.down_proj.numel()) * 2

    # (b) Experts4bit quantizes the fused stack: uint8-packed 4-bit weights + small fp32 absmax.
    q = Experts4bit.from_float(
        fused.gate_up_proj.data.half(), fused.down_proj.data.half(), compute_dtype=torch.float16
    )
    assert q.gate_up_proj.dtype == torch.uint8 and q.down_proj.dtype == torch.uint8
    quantized_bytes = (
        q.gate_up_proj.numel()
        + q.down_proj.numel()  # uint8 packed
        + (q.gate_up_absmax.numel() + q.down_absmax.numel()) * 4  # fp32 absmax
    )
    assert quantized_bytes < fp16_bytes / 3  # ~4x on the weights, minus small absmax overhead


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "num_experts,hidden,inter",
    [(2, 64, 64), (8, 128, 256), (4, 192, 320)],
    ids=["e2_h64_i64", "e8_h128_i256", "e4_h192_i320"],
)
def test_experts4bit_shapes(device, num_experts, hidden, inter):
    """Forward is correct across a spread of MoE dims (all multiples of the blocksize)."""
    gate_up = torch.randn(num_experts, 2 * inter, hidden, dtype=torch.float32, device=device) * 0.1
    down = torch.randn(num_experts, hidden, inter, dtype=torch.float32, device=device) * 0.1
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)

    n_tok = 10
    hidden_states = torch.randn(n_tok, hidden, device=device)
    top_k_index = torch.randint(0, num_experts, (n_tok, TOP_K), device=device)
    top_k_weights = torch.softmax(torch.randn(n_tok, TOP_K, device=device), dim=-1)
    out = module(hidden_states, top_k_index, top_k_weights)
    assert out.shape == (n_tok, hidden) and torch.isfinite(out).all()


# --- recompute-in-backward projection path ----------------------------------------------
# _project routes every expert projection through _FrozenLinearRecomputeBackward: the forward IS
# dequantize + F.linear (bit-exact by construction, every device and grad mode), and the backward
# re-dequantizes the frozen weight on demand instead of keeping it as a saved activation. These
# tests pin both halves of that contract — numerics (grad mode changes nothing; matches a plain
# dequantize+linear reference bit-for-bit) and memory (no [out, in] weight is ever saved).


def _dequantized_expert_stacks(module):
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
    return gate_up_deq, down_deq


@pytest.mark.parametrize("device", get_available_devices())
def test_experts4bit_forward_is_bit_exact_dequantize_linear(device):
    """Forward equals a plain dequantize+linear reference bit-for-bit, in and out of grad mode."""
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)
    hidden_states, top_k_index, top_k_weights = _random_routing(device)

    gate_up_deq, down_deq = _dequantized_expert_stacks(module)
    ref = _reference_forward(gate_up_deq, down_deq, hidden_states, top_k_index, top_k_weights)

    out_grad_mode = module(hidden_states, top_k_index, top_k_weights)
    with torch.no_grad():
        out_no_grad = module(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(out_grad_mode, ref, rtol=0, atol=0)
    torch.testing.assert_close(out_no_grad, ref, rtol=0, atol=0)


@pytest.mark.parametrize("device", get_available_devices())
def test_experts4bit_backward_saves_no_dequantized_weight(device):
    """The dequantized expert weights are dropped after each forward matmul (re-dequantized in
    backward), so nothing weight-shaped reaches autograd's saved-tensor storage — while a plain
    dequantize+linear control does save them. Gradients still match the control exactly."""
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = Experts4bit.from_float(gate_up, down, compute_dtype=torch.float32)
    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    # F.linear's backward may save the weight either as-is or pre-transposed (device/impl
    # dependent), so match both orientations.
    weight_shapes = set()
    for shape in (module._gate_up_shape, module._down_shape):
        weight_shapes.add(tuple(shape))
        weight_shapes.add(tuple(reversed(shape)))

    def run_recording_saved_shapes(fn, x):
        saved = []

        def pack(t):
            saved.append(tuple(t.shape))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = fn(x)
        return out, saved

    x_mod = hidden_states.detach().clone().requires_grad_(True)
    out_mod, saved_mod = run_recording_saved_shapes(lambda x: module(x, top_k_index, top_k_weights), x_mod)
    assert not (set(saved_mod) & weight_shapes)

    gate_up_deq, down_deq = _dequantized_expert_stacks(module)
    x_ref = hidden_states.detach().clone().requires_grad_(True)
    out_ref, saved_ref = run_recording_saved_shapes(
        lambda x: _reference_forward(gate_up_deq, down_deq, x, top_k_index, top_k_weights), x_ref
    )
    assert set(saved_ref) & weight_shapes

    out_mod.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(out_mod, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_mod.grad, x_ref.grad, rtol=0, atol=0)
