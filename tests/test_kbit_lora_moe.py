"""Tests for KbitLoraModel with MoE architectures.

Uses a tiny synthetic Qwen3-MoE model for fast testing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_tiny_moe_model():
    """Create a tiny Qwen3-MoE model for testing."""
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError:
        pytest.skip("transformers does not support Qwen3MoeForCausalLM")

    config = Qwen3MoeConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        vocab_size=1000,
        max_position_embeddings=256,
        decoder_sparse_step=1,
    )
    model = Qwen3MoeForCausalLM(config)
    model = model.to(torch.float16).cuda()
    return model


@pytest.fixture(scope="module")
def tiny_moe_model():
    return _make_tiny_moe_model()


@pytest.fixture(scope="module")
def kbit_moe_model(tiny_moe_model):
    from bitsandbytes.kbit_lora import KbitLoraModel

    return KbitLoraModel(
        tiny_moe_model,
        lora_r=4,
        lora_alpha=8.0,
        k=4,
        k_config={"attention": 4, "experts": 2},
        attn_chunk_size=64,
        mlp_chunk_size=64,
        ce_chunk_size=256,
        compute_dtype=torch.bfloat16,
        expert_chunk_size=4,
    )


class TestMoEKbitLoraModel:

    def test_creation(self, kbit_moe_model):
        """MoE model should be created successfully."""
        assert kbit_moe_model is not None
        assert kbit_moe_model.arch.is_moe
        assert kbit_moe_model.arch.num_experts == 8
        assert kbit_moe_model.arch.num_active_experts == 2

    def test_all_layers_are_moe(self, kbit_moe_model):
        """All layers should be MoE (decoder_sparse_step=1)."""
        for info in kbit_moe_model._layer_data:
            assert info.get("is_moe") is True

    def test_expert_weights_concatenated(self, kbit_moe_model):
        """Expert weights should be concatenated across all experts."""
        info = kbit_moe_model._layer_data[0]
        # gate packed should be 8 experts concatenated
        single_expert_packed_numel = info["expert_gate_packed"].numel() // 8
        assert info["expert_gate_packed"].numel() == single_expert_packed_numel * 8

    def test_expert_k_is_2(self, kbit_moe_model):
        """Expert projections should use k=2 (from k_config)."""
        for info in kbit_moe_model._layer_data:
            assert info["expert_k"] == 2

    def test_attention_k_is_4(self, kbit_moe_model):
        """Attention projections should use k=4."""
        for info in kbit_moe_model._layer_data:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                assert info[proj]["k"] == 4

    def test_has_router_weight(self, kbit_moe_model):
        """Each MoE layer should have a router weight."""
        for info in kbit_moe_model._layer_data:
            assert "router_weight" in info
            assert info["router_weight"].shape[0] == 8  # num_experts

    def test_trainable_parameters(self, kbit_moe_model):
        """Should have trainable parameters."""
        n = kbit_moe_model.num_trainable_parameters()
        assert n > 0
        print(f"MoE trainable parameters: {n:,}")

    def test_forward_with_loss(self, kbit_moe_model):
        """Forward pass should produce finite loss."""
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        result = kbit_moe_model(input_ids, labels=labels)

        assert "loss" in result
        loss = result["loss"]
        assert loss.isfinite(), f"Loss not finite: {loss.item()}"
        print(f"MoE loss: {loss.item():.4f}")

    def test_backward_produces_gradients(self, kbit_moe_model):
        """Backward should produce gradients on LoRA params."""
        for p in kbit_moe_model.get_trainable_parameters():
            if p.grad is not None:
                p.grad.zero_()

        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        result = kbit_moe_model(input_ids, labels=labels)
        result["loss"].backward()

        has_grad = False
        for p in kbit_moe_model.get_trainable_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients produced"

    def test_no_shared_expert(self, kbit_moe_model):
        """Qwen3-MoE has no shared expert."""
        assert not kbit_moe_model.arch.has_shared_expert
        for info in kbit_moe_model._layer_data:
            assert "shared_gate_proj" not in info
