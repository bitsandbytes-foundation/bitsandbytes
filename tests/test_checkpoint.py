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

            # Model architecture
            assert meta["model_type"] == "llama"
            assert int(meta["hidden_size"]) == 256
            assert int(meta["num_layers"]) == 2
            assert int(meta["num_attention_heads"]) == 4
            assert int(meta["num_key_value_heads"]) == 2
            assert int(meta["head_dim"]) == 64  # 256 / 4
            assert int(meta["intermediate_size"]) == 512
            assert int(meta["vocab_size"]) == 1000
            assert float(meta["rms_norm_eps"]) > 0
            assert float(meta["rope_theta"]) > 0

            # Quantization config
            assert meta["k_attention"] == "4"
            assert meta["k_mlp"] == "4"
            assert meta["k_lm_head"] == "4"
            assert meta["k_experts"] == "4"
            assert meta["k_shared_expert"] == "4"

            # MoE config
            assert meta["is_moe"] == "False"
            assert meta["has_shared_expert"] == "False"
            assert meta["has_qk_norm"] == "False"
            assert meta["dense_layer_indices"] == ""

            # Per-projection dims for layer 0 attention
            assert int(meta["layer.0.attn.q_proj.N"]) == 256  # q_dim = 4 * 64
            assert int(meta["layer.0.attn.q_proj.K"]) == 256  # hidden_size
            assert int(meta["layer.0.attn.q_proj.N_padded"]) == 256  # already mult of 128
            assert int(meta["layer.0.attn.q_proj.k"]) == 4

            assert int(meta["layer.0.attn.k_proj.N"]) == 128  # kv_dim = 2 * 64
            assert int(meta["layer.0.attn.k_proj.K"]) == 256

            # MLP dims
            assert int(meta["layer.0.mlp.gate_proj.N"]) == 512  # intermediate
            assert int(meta["layer.0.mlp.gate_proj.K"]) == 256  # hidden

            # LM head dims
            assert int(meta["lm_head.N"]) == 1000  # vocab_size
            assert int(meta["lm_head.K"]) == 256  # hidden_size

            # Check all layers have dims
            for i in range(2):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    assert f"layer.{i}.attn.{proj}.N" in meta
                    assert f"layer.{i}.attn.{proj}.K" in meta
                    assert f"layer.{i}.attn.{proj}.N_padded" in meta
                    assert f"layer.{i}.attn.{proj}.k" in meta
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    assert f"layer.{i}.mlp.{proj}.N" in meta
                    assert f"layer.{i}.mlp.{proj}.K" in meta
        finally:
            os.unlink(path)


class TestFromQuantized:
    """Test save_quantized → from_quantized round-trip."""

    def test_round_trip_dense_data_match(self, kbit_model):
        """Quantized weights must be bitwise identical after round-trip."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
                weight_streaming=False,
                target_device=torch.device("cuda:0"),
            )

            # Compare _layer_data quantized tensors
            for i in range(len(kbit_model._layer_data)):
                orig = kbit_model._layer_data[i]
                load = loaded._layer_data[i]

                for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"]:
                    if proj not in orig:
                        continue
                    for wk in ["packed", "absmax", "codebook"]:
                        assert torch.equal(
                            orig[proj][wk].cpu(), load[proj][wk].cpu()
                        ), f"Layer {i} {proj}.{wk} mismatch"
                    assert orig[proj]["N"] == load[proj]["N"]
                    assert orig[proj]["K"] == load[proj]["K"]
                    assert orig[proj]["N_padded"] == load[proj]["N_padded"]
                    assert orig[proj]["k"] == load[proj]["k"]

            # Compare LM head
            for wk in ["packed", "absmax", "codebook"]:
                assert torch.equal(
                    kbit_model._lm_head_info[wk].cpu(),
                    loaded._lm_head_info[wk].cpu(),
                ), f"LM head {wk} mismatch"

            # Compare embedding
            assert torch.equal(
                kbit_model.embed_tokens.weight.data.cpu(),
                loaded.embed_tokens.weight.data.cpu(),
            )
        finally:
            os.unlink(path)

    def test_round_trip_dense_forward_match(self, kbit_model):
        """Forward pass output must match between original and loaded model."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)

            # Save LoRA weights and load them into the new model
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as lf:
                lora_path = lf.name
            save_lora(kbit_model, lora_path)

            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
                weight_streaming=False,
                target_device=torch.device("cuda:0"),
                lora_checkpoint=lora_path,
            )

            # Run forward on same input
            input_ids = torch.randint(0, 100, (1, 32), device="cuda")
            labels = input_ids.clone()

            kbit_model.eval()
            loaded.eval()

            with torch.no_grad():
                orig_result = kbit_model(input_ids, labels=labels)
                load_result = loaded(input_ids, labels=labels)

            assert torch.allclose(
                orig_result["loss"], load_result["loss"], atol=1e-5
            ), f"Loss mismatch: {orig_result['loss'].item()} vs {load_result['loss'].item()}"
        finally:
            os.unlink(path)
            os.unlink(lora_path)

    def test_round_trip_dense_streaming(self, kbit_model):
        """from_quantized with weight_streaming=True should work."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
                weight_streaming=True,
                target_device=torch.device("cuda:0"),
            )

            # Verify streaming infrastructure exists
            assert hasattr(loaded, "_cpu_weights")
            assert hasattr(loaded, "_gpu_slots")
            assert len(loaded._cpu_weights) == len(kbit_model._layer_data)
            assert len(loaded._gpu_slots) == 2

            # Verify _layer_data quantized weights are None (moved to CPU pinned)
            for i, layer_info in enumerate(loaded._layer_data):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    assert layer_info[proj]["packed"] is None, \
                        f"Layer {i} {proj}.packed should be None after streaming init"
                    # LoRA params should still exist on GPU
                    assert layer_info[proj]["A"].device.type == "cuda"
                    assert layer_info[proj]["B"].device.type == "cuda"
        finally:
            os.unlink(path)

    def test_round_trip_attributes(self, kbit_model):
        """Model attributes must be correctly reconstructed."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(kbit_model, path)
            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                weight_streaming=False,
            )

            assert loaded.model_type == kbit_model.model_type
            assert loaded.hidden_size == kbit_model.hidden_size
            assert loaded.num_heads == kbit_model.num_heads
            assert loaded.num_kv_heads == kbit_model.num_kv_heads
            assert loaded.head_dim == kbit_model.head_dim
            assert loaded.intermediate_size == kbit_model.intermediate_size
            assert loaded.vocab_size == kbit_model.vocab_size
            assert loaded.num_layers == kbit_model.num_layers
            assert loaded._num_loaded_layers == kbit_model._num_loaded_layers
            assert loaded.k_attention == kbit_model.k_attention
            assert loaded.k_mlp == kbit_model.k_mlp
            assert loaded.k_lm_head == kbit_model.k_lm_head
        finally:
            os.unlink(path)


class TestFromQuantizedMoE:
    """Test save_quantized → from_quantized round-trip for MoE models."""

    @pytest.fixture(scope="class")
    def moe_model(self):
        from bitsandbytes.kbit_lora import KbitLoraModel

        try:
            from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        except ImportError:
            pytest.skip("transformers does not support Qwen3MoeForCausalLM")

        config = Qwen3MoeConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            vocab_size=1000,
            max_position_embeddings=256,
            decoder_sparse_step=1,
        )
        model = Qwen3MoeForCausalLM(config)
        model = model.to(torch.float16).cuda()

        return KbitLoraModel(
            model, lora_r=4, lora_alpha=8.0, k=4,
            k_config={"attention": 4, "experts": 2},
            attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
            compute_dtype=torch.bfloat16, expert_chunk_size=2,
        )

    def test_round_trip_moe_data_match(self, moe_model):
        """MoE quantized weights must be bitwise identical after round-trip."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(moe_model, path)
            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
                weight_streaming=False,
            )

            for i in range(len(moe_model._layer_data)):
                orig = moe_model._layer_data[i]
                load = loaded._layer_data[i]

                # Attention projections
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    for wk in ["packed", "absmax", "codebook"]:
                        assert torch.equal(
                            orig[proj][wk].cpu(), load[proj][wk].cpu()
                        ), f"Layer {i} {proj}.{wk} mismatch"

                # MoE fields
                assert load.get("is_moe") is True
                assert torch.equal(
                    orig["router_weight"].cpu(), load["router_weight"].cpu()
                )

                # Expert concatenated weights
                for expert_proj in ["gate", "up", "down"]:
                    for suffix in ["packed", "absmax"]:
                        key = f"expert_{expert_proj}_{suffix}"
                        assert torch.equal(
                            orig[key].cpu(), load[key].cpu()
                        ), f"Layer {i} {key} mismatch"
                assert torch.equal(
                    orig["expert_codebook"].cpu(), load["expert_codebook"].cpu()
                )
                assert orig["expert_k"] == load["expert_k"]
                assert orig["expert_N"] == load["expert_N"]
                assert orig["expert_K"] == load["expert_K"]
                assert orig["expert_N_padded"] == load["expert_N_padded"]
        finally:
            os.unlink(path)

    def test_round_trip_moe_streaming(self, moe_model):
        """MoE from_quantized with weight_streaming=True should work."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_quantized(moe_model, path)
            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
                weight_streaming=True,
            )

            assert hasattr(loaded, "_cpu_weights")
            assert len(loaded._cpu_weights) == 2

            # Expert weights should be in CPU pinned memory
            for cpu_layer in loaded._cpu_weights:
                assert "expert_gate_packed" in cpu_layer
                assert cpu_layer["expert_gate_packed"].is_pinned()
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
