"""Tests for KbitLoraModel (model patcher).

Tests:
- Model creation from Qwen3 0.6B (smallest available)
- Trainable parameter count
- Forward pass produces finite loss
- Backward pass produces gradients on LoRA params
- Gradient accumulation works
- Only LoRA + norms are trainable
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from bitsandbytes.kbit_lora import KbitLoraModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def qwen3_model():
    """Load Qwen3 0.6B once for all tests."""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    return model


@pytest.fixture(scope="module")
def kbit_model(qwen3_model):
    """Create KbitLoraModel once for all tests."""
    return KbitLoraModel(
        qwen3_model,
        lora_r=8,
        lora_alpha=16.0,
        k=4,
        attn_chunk_size=128,
        mlp_chunk_size=128,
        ce_chunk_size=1024,
        compute_dtype=torch.bfloat16,
    )


class TestKbitLoraModel:

    def test_creation(self, kbit_model):
        """Model should be created successfully."""
        assert kbit_model is not None
        assert kbit_model.model_type == "qwen3"
        assert kbit_model.num_layers == 28

    def test_trainable_parameters(self, kbit_model):
        """Should have trainable LoRA + norm parameters."""
        n_trainable = kbit_model.num_trainable_parameters()
        assert n_trainable > 0
        # With r=8, each LoRA pair has 2 * (r * dim) params
        # 28 layers * 7 projections * 2 matrices = 392 LoRA matrices
        # Plus norm weights
        print(f"Trainable parameters: {n_trainable:,}")

    def test_only_lora_and_norms_trainable(self, kbit_model):
        """Base model weights should be frozen."""
        trainable = kbit_model.get_trainable_parameters()
        for name, p in kbit_model.named_parameters():
            if p.requires_grad:
                assert "_lora_params" in name or "_norm_weights" in name, \
                    f"Unexpected trainable parameter: {name}"

    def test_forward_with_loss(self, kbit_model):
        """Forward pass with labels should produce finite loss."""
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()
        labels[:, :5] = -100  # Mask first 5 tokens

        result = kbit_model(input_ids, labels=labels)

        assert "loss" in result
        loss = result["loss"]
        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
        print(f"Loss: {loss.item():.4f}")

    def test_forward_without_labels(self, kbit_model):
        """Forward pass without labels should produce logits."""
        input_ids = torch.randint(0, 100, (1, 16), device="cuda")
        result = kbit_model(input_ids)

        assert "logits" in result
        logits = result["logits"]
        assert logits.shape == (1, kbit_model.vocab_size)
        assert logits.isfinite().all()

    def test_backward_produces_gradients(self, kbit_model):
        """Backward pass should produce gradients on trainable params."""
        # Zero all gradients first
        for p in kbit_model.get_trainable_parameters():
            if p.grad is not None:
                p.grad.zero_()

        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        result = kbit_model(input_ids, labels=labels)
        result["loss"].backward()

        # Check that at least some LoRA params have gradients
        has_grad = False
        for p in kbit_model.get_trainable_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients produced for trainable parameters"

    def test_gradient_accumulation(self, kbit_model):
        """Gradient accumulation over 2 micro-batches should work."""
        for p in kbit_model.get_trainable_parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Two micro-batches
        for _ in range(2):
            input_ids = torch.randint(0, 100, (1, 16), device="cuda")
            labels = input_ids.clone()
            result = kbit_model(input_ids, labels=labels)
            (result["loss"] / 2).backward()  # Scale for accumulation

        # All LoRA A matrices should have gradients
        for name, p in kbit_model._lora_params.named_parameters():
            if "_A" in name:
                assert p.grad is not None, f"No gradient for {name}"


class TestMixedKQuantization:
    """Tests for mixed-k quantization (different k for attention/MLP/LM head)."""

    @pytest.fixture(scope="class")
    def mixed_k_model(self, qwen3_model):
        """Create KbitLoraModel with mixed k values."""
        return KbitLoraModel(
            qwen3_model,
            lora_r=8,
            lora_alpha=16.0,
            k=4,  # default fallback
            k_config={"attention": 4, "mlp": 3, "lm_head": 2},
            attn_chunk_size=128,
            mlp_chunk_size=128,
            ce_chunk_size=1024,
            compute_dtype=torch.bfloat16,
        )

    def test_mixed_k_creation(self, mixed_k_model):
        """Mixed-k model should be created successfully."""
        assert mixed_k_model.k_attention == 4
        assert mixed_k_model.k_mlp == 3
        assert mixed_k_model.k_lm_head == 2

    def test_attention_uses_correct_k(self, mixed_k_model):
        """Attention projections should use k=4."""
        for layer_info in mixed_k_model._layer_data:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                assert layer_info[proj]["k"] == 4, \
                    f"Attention {proj} should have k=4"

    def test_mlp_uses_correct_k(self, mixed_k_model):
        """MLP projections should use k=3."""
        for layer_info in mixed_k_model._layer_data:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                assert layer_info[proj]["k"] == 3, \
                    f"MLP {proj} should have k=3"

    def test_lm_head_uses_correct_k(self, mixed_k_model):
        """LM head should use k=2."""
        assert mixed_k_model._lm_head_info["k"] == 2

    def test_forward_with_mixed_k(self, mixed_k_model):
        """Forward pass should work with mixed k values."""
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        result = mixed_k_model(input_ids, labels=labels)
        loss = result["loss"]
        assert loss.isfinite(), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0

    def test_backward_with_mixed_k(self, mixed_k_model):
        """Backward pass should work with mixed k values."""
        for p in mixed_k_model.get_trainable_parameters():
            if p.grad is not None:
                p.grad.zero_()

        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        result = mixed_k_model(input_ids, labels=labels)
        result["loss"].backward()

        has_grad = False
        for p in mixed_k_model.get_trainable_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
