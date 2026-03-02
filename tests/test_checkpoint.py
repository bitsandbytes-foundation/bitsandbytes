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
            assert hasattr(loaded, "_n_resident")

            n = len(kbit_model._layer_data)
            nr = loaded._n_resident
            n_streamed = n - nr

            # CPU weights should match number of non-resident layers
            assert len(loaded._cpu_weights) == n_streamed
            # GPU slots allocated only if there are streamed layers
            if n_streamed > 0:
                assert len(loaded._gpu_slots) == 2
            else:
                assert len(loaded._gpu_slots) == 0

            # Verify layer data:
            # - Resident layers keep weights on GPU
            # - Streamed layers have weights = None (moved to CPU pinned)
            for i, layer_info in enumerate(loaded._layer_data):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if i < nr:
                        # Resident: weights on GPU
                        assert layer_info[proj]["packed"] is not None
                        assert layer_info[proj]["packed"].device.type == "cuda"
                    else:
                        # Streamed: weights moved to CPU
                        assert layer_info[proj]["packed"] is None
                    # LoRA params always on GPU
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
            assert hasattr(loaded, "_n_resident")

            nr = loaded._n_resident
            n_streamed = 2 - nr

            # CPU weights should match number of non-resident layers
            assert len(loaded._cpu_weights) == n_streamed

            # If there are streamed layers, expert weights in CPU pinned
            for cpu_layer in loaded._cpu_weights:
                assert "expert_gate_packed" in cpu_layer
                assert cpu_layer["expert_gate_packed"].is_pinned()

            # If all layers are resident, expert weights on GPU
            for i in range(nr):
                li = loaded._layer_data[i]
                if li.get("is_moe"):
                    assert li["expert_gate_packed"] is not None
                    assert li["expert_gate_packed"].device.type == "cuda"
        finally:
            os.unlink(path)


class TestPartialResidency:
    """Test partial residency: some layers on GPU, rest streamed."""

    @pytest.fixture
    def quantized_path(self, kbit_model):
        """Save kbit_model to a temporary quantized checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        save_quantized(kbit_model, path)
        yield path
        os.unlink(path)

    def test_all_resident_with_enough_vram(self, quantized_path):
        """With enough VRAM, all layers should be resident (no streaming)."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        loaded = KbitLoraModel.from_quantized(
            quantized_path, lora_r=4, lora_alpha=8.0,
            attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
            weight_streaming=True, batch_size=1, seq_len=32,
        )

        # Tiny model fits entirely on GPU
        assert loaded._n_resident == loaded._num_loaded_layers
        assert len(loaded._cpu_weights) == 0

        # Weights should be on GPU, not None
        for layer_info in loaded._layer_data:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                assert layer_info[proj]["packed"] is not None
                assert layer_info[proj]["packed"].device.type == "cuda"

    def test_forced_partial_residency(self, quantized_path):
        """Monkey-patch _compute_residency to force partial split."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        # Force only 1 of 2 layers to be resident
        with patch.object(KbitLoraModel, "_compute_residency", return_value=1):
            loaded = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert loaded._n_resident == 1
        assert len(loaded._cpu_weights) == 1  # 1 layer streamed
        assert len(loaded._gpu_slots) == 2    # double buffer allocated

        # Layer 0: resident, weights on GPU
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            assert loaded._layer_data[0][proj]["packed"] is not None
            assert loaded._layer_data[0][proj]["packed"].device.type == "cuda"

        # Layer 1: streamed, weights moved to CPU
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            assert loaded._layer_data[1][proj]["packed"] is None
            assert proj in loaded._cpu_weights[0]
            assert loaded._cpu_weights[0][proj]["packed"].is_pinned()

    def test_forced_partial_forward_backward(self, quantized_path):
        """Partial residency should produce correct forward/backward results."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        # Force 1 resident + 1 streamed
        with patch.object(KbitLoraModel, "_compute_residency", return_value=1):
            model = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        model.train()
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        loss, ctx = model.forward_streaming(input_ids, labels)
        assert loss.item() > 0
        model.backward_streaming(ctx)

        # Should have gradients for all LoRA params
        # Note: LoRA A gradients are zero at initialization because B is
        # zero-initialized (d(loss)/dA depends on B). Only B has non-zero grads.
        for name, param in model._lora_params.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            if name.endswith("_B"):
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_zero_resident_streaming(self, quantized_path):
        """Force 0 resident layers — everything streamed."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            model = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert model._n_resident == 0
        assert len(model._cpu_weights) == 2

        # Forward/backward should work
        model.train()
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        loss, ctx = model.forward_streaming(input_ids, labels)
        assert loss.item() > 0
        model.backward_streaming(ctx)

        for name, param in model._lora_params.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_partial_vs_full_resident_gradient_match(self, quantized_path):
        """Partial residency must give same gradients as fully resident."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        def _run_fwd_bwd(n_resident):
            # Same seed for LoRA initialization
            torch.manual_seed(123)
            with patch.object(KbitLoraModel, "_compute_residency", return_value=n_resident):
                m = KbitLoraModel.from_quantized(
                    quantized_path, lora_r=4, lora_alpha=8.0,
                    attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                    weight_streaming=True, batch_size=1, seq_len=32,
                )
            m.train()
            torch.manual_seed(42)
            ids = torch.randint(0, 100, (1, 32), device="cuda")
            lb = ids.clone()
            loss, ctx = m.forward_streaming(ids, lb)
            m.backward_streaming(ctx)
            grads = {}
            for name, p in m._lora_params.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.clone()
            return loss, grads

        loss_full, grads_full = _run_fwd_bwd(2)  # fully resident
        loss_part, grads_part = _run_fwd_bwd(1)  # 1 resident + 1 streamed

        assert torch.allclose(loss_full, loss_part, atol=1e-5), (
            f"Loss mismatch: full={loss_full.item()}, partial={loss_part.item()}"
        )

        for name in grads_full:
            assert torch.allclose(grads_full[name], grads_part[name], atol=1e-4), (
                f"Gradient mismatch for {name}"
            )


class TestRAMStrategy:
    """Test RAM strategy auto-detection (pinned, hybrid, mmap)."""

    @pytest.fixture
    def quantized_path(self, kbit_model):
        """Save kbit_model to a temporary quantized checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        save_quantized(kbit_model, path)
        yield path
        os.unlink(path)

    def test_default_pinned_with_enough_ram(self, quantized_path):
        """With plenty of RAM, strategy should be 'pinned'."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        # Force 0 resident to exercise streaming, with plenty of RAM
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            m = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert m._ram_strategy == "pinned"
        assert m._safetensors_file is None
        assert len(m._staging_buffers) == 0
        # All cpu_weights should be non-None (pinned)
        for cl in m._cpu_weights:
            assert cl is not None

    def test_mmap_with_low_ram(self, quantized_path):
        """With very low RAM, strategy should be 'mmap'."""
        from bitsandbytes.kbit_lora import KbitLoraModel, get_available_ram_bytes
        from unittest.mock import patch

        # Force 0 resident and very low available RAM (1 byte)
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0), \
             patch("bitsandbytes.kbit_lora.get_available_ram_bytes", return_value=1):
            m = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert m._ram_strategy == "mmap"
        assert m._safetensors_file is not None
        assert len(m._staging_buffers) == 2
        # All cpu_weights should be None (mmap)
        for cl in m._cpu_weights:
            assert cl is None

    def test_hybrid_with_limited_ram(self, quantized_path):
        """With limited RAM, strategy should be 'hybrid'."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        # First determine layer sizes to craft the right RAM value
        from unittest.mock import patch
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            m_probe = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        # Get size of 1 layer (all layers same size for this model)
        layer_bytes = sum(
            (sum(t.nbytes for t in v.values()) if isinstance(v, dict) else v.nbytes)
            for v in m_probe._cpu_weights[0].values()
        )
        total_bytes = layer_bytes * 2  # 2 layers

        # Set RAM to 4GB headroom + 1.5 layers worth (enough for 1 layer but not 2)
        fake_ram = 4 * 1024**3 + int(layer_bytes * 1.5)

        with patch.object(KbitLoraModel, "_compute_residency", return_value=0), \
             patch("bitsandbytes.kbit_lora.get_available_ram_bytes", return_value=fake_ram):
            m = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert m._ram_strategy == "hybrid"
        assert m._safetensors_file is not None
        assert len(m._staging_buffers) == 2
        # First layer pinned, second mmap
        assert m._cpu_weights[0] is not None  # pinned
        assert m._cpu_weights[1] is None      # mmap

    def test_mmap_forward_backward(self, quantized_path):
        """Mmap strategy should produce correct forward/backward."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        # Force mmap for all layers
        torch.manual_seed(123)
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0), \
             patch("bitsandbytes.kbit_lora.get_available_ram_bytes", return_value=1):
            m = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        m.train()
        torch.manual_seed(42)
        ids = torch.randint(0, 100, (1, 32), device="cuda")
        lb = ids.clone()
        loss, ctx = m.forward_streaming(ids, lb)
        assert loss.item() > 0
        m.backward_streaming(ctx)

        for name, p in m._lora_params.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_hybrid_forward_backward(self, quantized_path):
        """Hybrid strategy should produce correct forward/backward."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        # First get layer size
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            m_probe = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )
        layer_bytes = sum(
            (sum(t.nbytes for t in v.values()) if isinstance(v, dict) else v.nbytes)
            for v in m_probe._cpu_weights[0].values()
        )
        fake_ram = 4 * 1024**3 + int(layer_bytes * 1.5)

        torch.manual_seed(123)
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0), \
             patch("bitsandbytes.kbit_lora.get_available_ram_bytes", return_value=fake_ram):
            m = KbitLoraModel.from_quantized(
                quantized_path, lora_r=4, lora_alpha=8.0,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                weight_streaming=True, batch_size=1, seq_len=32,
            )

        assert m._ram_strategy == "hybrid"
        m.train()
        torch.manual_seed(42)
        ids = torch.randint(0, 100, (1, 32), device="cuda")
        lb = ids.clone()
        loss, ctx = m.forward_streaming(ids, lb)
        assert loss.item() > 0
        m.backward_streaming(ctx)

        for name, p in m._lora_params.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_mmap_matches_pinned_gradients(self, quantized_path):
        """Mmap strategy should produce same gradients as pinned."""
        from bitsandbytes.kbit_lora import KbitLoraModel
        from unittest.mock import patch

        def _run(strategy_ram):
            torch.manual_seed(123)
            patches = [patch.object(KbitLoraModel, "_compute_residency", return_value=0)]
            if strategy_ram is not None:
                patches.append(
                    patch("bitsandbytes.kbit_lora.get_available_ram_bytes",
                          return_value=strategy_ram)
                )
            with patches[0] if len(patches) == 1 else patches[0], \
                 (patches[1] if len(patches) > 1 else patch.object(
                     KbitLoraModel, "_compute_residency", return_value=0)):
                m = KbitLoraModel.from_quantized(
                    quantized_path, lora_r=4, lora_alpha=8.0,
                    attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                    weight_streaming=True, batch_size=1, seq_len=32,
                )
            m.train()
            torch.manual_seed(42)
            ids = torch.randint(0, 100, (1, 32), device="cuda")
            lb = ids.clone()
            loss, ctx = m.forward_streaming(ids, lb)
            m.backward_streaming(ctx)
            grads = {}
            for name, p in m._lora_params.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.clone()
            return loss, grads

        loss_pinned, grads_pinned = _run(None)  # default: enough RAM → pinned
        loss_mmap, grads_mmap = _run(1)  # very low RAM → mmap

        assert torch.allclose(loss_pinned, loss_mmap, atol=1e-5)
        for name in grads_pinned:
            assert torch.allclose(grads_pinned[name], grads_mmap[name], atol=1e-4), (
                f"Gradient mismatch for {name}"
            )


class TestGDS:
    """Test GDS/kvikio integration for NVMe → GPU weight streaming."""

    @pytest.fixture
    def quantized_path(self, kbit_model):
        """Save kbit_model to a temporary quantized checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        save_quantized(kbit_model, path)
        yield path
        os.unlink(path)

    @pytest.mark.skipif(
        "GeForce" not in torch.cuda.get_device_name(0),
        reason="Test requires GeForce GPU (verifies GDS fallback)",
    )
    def test_gds_fallback_on_geforce(self, quantized_path):
        """On GeForce GPUs, use_gds=True should warn and fall back to CPU path."""
        import warnings
        from bitsandbytes.kbit_lora import KbitLoraModel

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = KbitLoraModel.from_quantized(
                quantized_path, weight_streaming=True, use_gds=True,
            )
        # Should warn about GDS fallback (GeForce detected)
        gds_warnings = [x for x in w if "GDS requested but not available" in str(x.message)]
        assert len(gds_warnings) == 1, f"Expected GDS fallback warning, got: {w}"
        # Model should work (fell back to non-GDS)
        assert not model._use_gds
        # If streaming is active, strategy should not be gds
        if hasattr(model, "_ram_strategy"):
            assert model._ram_strategy != "gds"

    def test_gds_strategy_with_mock(self, quantized_path):
        """With mocked GDS support, strategy should be 'gds'."""
        from unittest.mock import patch
        from bitsandbytes.kbit_lora import KbitLoraModel

        with patch.object(KbitLoraModel, "_detect_gds_support", return_value=True), \
             patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            model = KbitLoraModel.from_quantized(
                quantized_path, weight_streaming=True, use_gds=True,
            )
        assert model._use_gds
        assert model._ram_strategy == "gds"
        # GDS layer info should be populated for all streamed layers
        n_streamed = model._num_loaded_layers - model._n_resident
        assert len(model._gds_layer_info) == n_streamed
        # Each GDS layer should have offset info with correct structure
        for si, gds_layer in model._gds_layer_info.items():
            for key, value in gds_layer.items():
                if isinstance(value, dict):
                    # Nested projection: {packed: (off, size, shape, dtype), ...}
                    for wk, info in value.items():
                        assert len(info) == 4  # (offset, size, shape, dtype)
                        offset, size, shape, dtype = info
                        assert offset > 0
                        assert size > 0
                else:
                    # Flat tensor: (offset, size, shape, dtype)
                    offset, size, shape, dtype = value
                    assert offset > 0
                    assert size > 0
        # GPU slots should be allocated
        assert len(model._gpu_slots) == 2

    def test_gds_forward_backward_compat_mode(self, quantized_path):
        """Test full GDS path using kvikio in compat mode (works on GeForce)."""
        from unittest.mock import patch
        from bitsandbytes.kbit_lora import KbitLoraModel

        with patch.object(KbitLoraModel, "_detect_gds_support", return_value=True), \
             patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            model = KbitLoraModel.from_quantized(
                quantized_path, weight_streaming=True, use_gds=True,
            )
        assert model._ram_strategy == "gds"

        # Forward + backward
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")
        loss, ctx = model.forward_streaming(input_ids, labels)
        assert loss.item() > 0
        model.backward_streaming(ctx)

        # Verify gradients exist (LoRA B should have non-zero grads)
        has_nonzero_grad = False
        for name, param in model._lora_params.named_parameters():
            if param.grad is not None and "_B" in name:
                if param.grad.abs().sum() > 0:
                    has_nonzero_grad = True
        assert has_nonzero_grad, "No non-zero gradients found for LoRA B"

    def test_gds_matches_pinned_gradients(self, quantized_path):
        """GDS path should produce identical gradients as pinned path."""
        from unittest.mock import patch
        from bitsandbytes.kbit_lora import KbitLoraModel

        # Load with pinned (zero-resident to force streaming)
        torch.manual_seed(42)
        with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            model_pinned = KbitLoraModel.from_quantized(
                quantized_path, weight_streaming=True, use_gds=False,
            )
        # Load with GDS
        torch.manual_seed(42)
        with patch.object(KbitLoraModel, "_detect_gds_support", return_value=True), \
             patch.object(KbitLoraModel, "_compute_residency", return_value=0):
            model_gds = KbitLoraModel.from_quantized(
                quantized_path, weight_streaming=True, use_gds=True,
            )

        # Same forward + backward
        torch.manual_seed(123)
        input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
        labels = torch.randint(0, 1000, (1, 32), device="cuda")

        loss_p, ctx_p = model_pinned.forward_streaming(input_ids.clone(), labels.clone())
        model_pinned.backward_streaming(ctx_p)

        loss_g, ctx_g = model_gds.forward_streaming(input_ids.clone(), labels.clone())
        model_gds.backward_streaming(ctx_g)

        # Loss should match
        assert torch.allclose(
            torch.tensor(loss_p.item()), torch.tensor(loss_g.item()), atol=1e-4
        ), f"Loss mismatch: pinned={loss_p.item()}, gds={loss_g.item()}"

        # Gradients should match
        grads_pinned = {
            name: param.grad.clone()
            for name, param in model_pinned._lora_params.named_parameters()
            if param.grad is not None
        }
        grads_gds = {
            name: param.grad.clone()
            for name, param in model_gds._lora_params.named_parameters()
            if param.grad is not None
        }
        for name in grads_pinned:
            assert torch.allclose(grads_pinned[name], grads_gds[name], atol=1e-4), (
                f"Gradient mismatch for {name}"
            )

    def test_parse_safetensors_offsets(self, quantized_path):
        """Test that parse_safetensors_offsets correctly reads tensor metadata."""
        from bitsandbytes.kbit_lora import parse_safetensors_offsets
        from safetensors import safe_open

        offsets = parse_safetensors_offsets(quantized_path)
        # Should have entries for all tensors
        assert len(offsets) > 0

        # Verify a few entries against safetensors API
        sf = safe_open(quantized_path, framework="pt", device="cpu")
        for name in list(offsets.keys())[:5]:
            offset, size, shape, dtype = offsets[name]
            tensor = sf.get_tensor(name)
            assert list(shape) == list(tensor.shape), f"Shape mismatch for {name}"
            assert size == tensor.nbytes, f"Size mismatch for {name}: {size} vs {tensor.nbytes}"

    def test_detect_gds_support(self):
        """Verify _detect_gds_support matches GPU type."""
        from bitsandbytes.kbit_lora import KbitLoraModel

        gpu_name = torch.cuda.get_device_name(0)
        result = KbitLoraModel._detect_gds_support()
        if "GeForce" in gpu_name:
            assert result is False, "GDS should not be supported on GeForce"
        else:
            # Non-GeForce GPU (workstation/datacenter): GDS should be True
            # if kvikio is installed
            try:
                import kvikio  # noqa: F401
                assert result is True, f"GDS should be supported on {gpu_name}"
            except ImportError:
                assert result is False


class TestStreamingQuantize:
    """Test streaming_quantize produces bitwise-identical output to save_quantized."""

    def test_dense_matches_in_memory(self):
        """Streaming quantize of dense model must match in-memory quantize."""
        from bitsandbytes.checkpoint import streaming_quantize
        from bitsandbytes.kbit_lora import KbitLoraModel
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save tiny model to disk
            model = _make_tiny_dense_model()
            model.save_pretrained(os.path.join(tmpdir, "hf_model"))

            # Path A: In-memory quantize → save_quantized
            kbit = KbitLoraModel(
                model, lora_r=4, lora_alpha=8.0, k=4,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
            )
            path_a = os.path.join(tmpdir, "inmemory.safetensors")
            save_quantized(kbit, path_a)
            del kbit

            # Path B: Streaming quantize from saved model
            path_b = os.path.join(tmpdir, "streamed.safetensors")
            streaming_quantize(
                os.path.join(tmpdir, "hf_model"), path_b, k=4,
            )

            # Compare all tensors
            sf_a = safe_open(path_a, framework="pt", device="cpu")
            sf_b = safe_open(path_b, framework="pt", device="cpu")

            keys_a = set(sf_a.keys())
            keys_b = set(sf_b.keys())
            assert keys_a == keys_b, f"Key mismatch: {keys_a - keys_b} vs {keys_b - keys_a}"

            for key in sorted(keys_a):
                t_a = sf_a.get_tensor(key)
                t_b = sf_b.get_tensor(key)
                assert t_a.shape == t_b.shape, f"{key}: shape {t_a.shape} vs {t_b.shape}"
                assert t_a.dtype == t_b.dtype, f"{key}: dtype {t_a.dtype} vs {t_b.dtype}"
                assert torch.equal(t_a, t_b), f"{key}: values differ"

    def test_dense_metadata_matches(self):
        """Streaming quantize metadata must match in-memory metadata."""
        from bitsandbytes.checkpoint import streaming_quantize
        from bitsandbytes.kbit_lora import KbitLoraModel
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_tiny_dense_model()
            model.save_pretrained(os.path.join(tmpdir, "hf_model"))

            kbit = KbitLoraModel(
                model, lora_r=4, lora_alpha=8.0, k=4,
                attn_chunk_size=64, mlp_chunk_size=64, ce_chunk_size=256,
                compute_dtype=torch.bfloat16,
            )
            path_a = os.path.join(tmpdir, "inmemory.safetensors")
            save_quantized(kbit, path_a)
            del kbit

            path_b = os.path.join(tmpdir, "streamed.safetensors")
            streaming_quantize(os.path.join(tmpdir, "hf_model"), path_b, k=4)

            sf_a = safe_open(path_a, framework="pt", device="cpu")
            sf_b = safe_open(path_b, framework="pt", device="cpu")
            meta_a = sf_a.metadata()
            meta_b = sf_b.metadata()

            # Check key metadata fields match
            for field in ["model_type", "hidden_size", "num_layers",
                         "num_attention_heads", "num_key_value_heads", "head_dim",
                         "intermediate_size", "vocab_size",
                         "k_attention", "k_mlp", "k_lm_head",
                         "is_moe", "has_qk_norm"]:
                assert meta_a[field] == meta_b[field], \
                    f"Metadata {field}: {meta_a[field]} vs {meta_b[field]}"

            # Per-projection dims
            for i in range(2):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    for dim in ["N", "K", "N_padded", "k"]:
                        key = f"layer.{i}.attn.{proj}.{dim}"
                        assert meta_a[key] == meta_b[key], \
                            f"Metadata {key}: {meta_a[key]} vs {meta_b[key]}"

    def test_streamed_loadable_by_from_quantized(self):
        """Output of streaming_quantize should be loadable by from_quantized."""
        from bitsandbytes.checkpoint import streaming_quantize
        from bitsandbytes.kbit_lora import KbitLoraModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_tiny_dense_model()
            model.save_pretrained(os.path.join(tmpdir, "hf_model"))
            del model

            path = os.path.join(tmpdir, "quantized.safetensors")
            streaming_quantize(os.path.join(tmpdir, "hf_model"), path, k=4)

            loaded = KbitLoraModel.from_quantized(
                path, lora_r=4, lora_alpha=8.0,
                weight_streaming=False,
            )

            # Forward pass should work
            input_ids = torch.randint(0, 100, (1, 32), device="cuda")
            labels = input_ids.clone()
            loaded.eval()
            with torch.no_grad():
                result = loaded(input_ids, labels=labels)
            assert result["loss"].isfinite()

    def test_copies_config_json(self):
        """streaming_quantize should copy config.json alongside output."""
        from bitsandbytes.checkpoint import streaming_quantize

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_tiny_dense_model()
            model.save_pretrained(os.path.join(tmpdir, "hf_model"))
            del model

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            path = os.path.join(output_dir, "model.safetensors")
            streaming_quantize(os.path.join(tmpdir, "hf_model"), path, k=4)

            assert os.path.exists(os.path.join(output_dir, "config.json"))


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
