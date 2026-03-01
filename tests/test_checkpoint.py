"""Tests for pre-quantized checkpoint save/load."""

import os
import tempfile

import pytest
import torch

from bitsandbytes.checkpoint import save_quantized, save_lora, load_lora

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_tiny_dense_model():
    """Create a tiny Llama model for testing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config)
    model = model.to(torch.float16).cuda()
    return model


@pytest.fixture(scope="module")
def kbit_model():
    from bitsandbytes.kbit_lora import KbitLoraModel

    model = _make_tiny_dense_model()
    return KbitLoraModel(
        model, lora_r=4, lora_alpha=8.0, k=4,
        attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
        compute_dtype=torch.bfloat16,
    )


class TestSaveQuantized:

    def test_save_creates_file(self, kbit_model):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_tensor_names_layer_ordered(self, kbit_model):
        """Tensor names should be grouped by layer for sequential NVMe reads."""
        from safetensors import safe_open

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            sf = safe_open(path, framework="pt", device="cpu")
            keys = list(sf.keys())

            # All layer.0.* should come before layer.1.*
            layer_0_last = max(i for i, k in enumerate(keys) if k.startswith("layer.0."))
            layer_1_first = min(i for i, k in enumerate(keys) if k.startswith("layer.1."))
            assert layer_0_last < layer_1_first, \
                f"Layer 0 tensors should precede layer 1: last L0={layer_0_last}, first L1={layer_1_first}"
        finally:
            os.unlink(path)

    def test_metadata_present(self, kbit_model):
        from safetensors import safe_open

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            sf = safe_open(path, framework="pt", device="cpu")
            meta = sf.metadata()
            assert meta["model_type"] == "llama"
            assert meta["k_attention"] == "4"
            assert int(meta["num_layers"]) == 2
        finally:
            os.unlink(path)


class TestSaveLoadLora:

    def test_lora_round_trip(self, kbit_model):
        """Save LoRA, modify params, load LoRA, verify restoration."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            # Save current LoRA weights
            save_lora(kbit_model, path)

            # Record original values
            original_values = {}
            for name, param in kbit_model._lora_params.items():
                original_values[name] = param.data.clone()

            # Modify LoRA params
            for param in kbit_model._lora_params.parameters():
                param.data.fill_(999.0)

            # Load should restore
            load_lora(kbit_model, path)

            # Verify restoration
            for name, param in kbit_model._lora_params.items():
                assert torch.allclose(param.data, original_values[name].to(param.device)), \
                    f"LoRA param {name} not restored correctly"
        finally:
            os.unlink(path)
