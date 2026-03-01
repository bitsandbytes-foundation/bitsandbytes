"""Tests for ArchConfig architecture adapter system."""

import pytest
from dataclasses import replace

from bitsandbytes.arch_config import (
    ArchConfig,
    LLAMA_CONFIG,
    MISTRAL_CONFIG,
    QWEN2_CONFIG,
    QWEN3_DENSE_CONFIG,
    QWEN3_MOE_CONFIG,
    GLM4_MOE_CONFIG,
    detect_arch_config,
)


class MockConfig:
    """Mock HuggingFace model config for testing."""

    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestArchConfigDetection:

    def test_detect_llama(self):
        config = MockConfig("llama")
        arch = detect_arch_config(config)
        assert arch.layers_path == "model.layers"
        assert arch.attn_module == "self_attn"
        assert not arch.is_moe

    def test_detect_mistral(self):
        config = MockConfig("mistral")
        arch = detect_arch_config(config)
        assert arch.q_proj == "q_proj"
        assert not arch.is_moe

    def test_detect_qwen2(self):
        config = MockConfig("qwen2")
        arch = detect_arch_config(config)
        assert not arch.has_qk_norm

    def test_detect_qwen3_dense(self):
        config = MockConfig("qwen3")
        arch = detect_arch_config(config)
        assert arch.has_qk_norm
        assert not arch.is_moe

    def test_detect_qwen3_moe(self):
        config = MockConfig("qwen3_moe", num_experts=128, num_experts_per_tok=8)
        arch = detect_arch_config(config)
        assert arch.is_moe
        assert arch.has_qk_norm
        assert arch.num_experts == 128
        assert arch.num_active_experts == 8
        assert not arch.has_shared_expert

    def test_detect_qwen3_moe_override_experts(self):
        """Should override num_experts from config when it differs."""
        config = MockConfig("qwen3_moe", num_experts=64, num_experts_per_tok=4)
        arch = detect_arch_config(config)
        assert arch.num_experts == 64
        assert arch.num_active_experts == 4

    def test_detect_glm4(self):
        config = MockConfig("glm4")
        arch = detect_arch_config(config)
        assert arch.is_moe
        assert arch.has_shared_expert
        assert arch.num_experts == 160
        assert arch.dense_layer_indices == [0, 1, 2]

    def test_detect_unsupported(self):
        config = MockConfig("gpt2")
        with pytest.raises(ValueError, match="Unsupported"):
            detect_arch_config(config)

    def test_detect_no_model_type(self):
        config = object()  # no model_type attribute
        with pytest.raises(ValueError, match="model_type"):
            detect_arch_config(config)


class TestArchConfigMoELayer:

    def test_all_moe_layers(self):
        """When dense_layer_indices is None, all layers are MoE."""
        arch = QWEN3_MOE_CONFIG
        assert arch.is_moe_layer(0)
        assert arch.is_moe_layer(47)

    def test_mixed_dense_moe(self):
        """GLM-4.7 has first 3 dense, rest MoE."""
        arch = GLM4_MOE_CONFIG
        assert not arch.is_moe_layer(0)
        assert not arch.is_moe_layer(1)
        assert not arch.is_moe_layer(2)
        assert arch.is_moe_layer(3)
        assert arch.is_moe_layer(91)

    def test_dense_model(self):
        """Dense models always return False for is_moe_layer."""
        arch = LLAMA_CONFIG
        assert not arch.is_moe_layer(0)
        assert not arch.is_moe_layer(100)


class TestGetNestedAttr:

    def test_simple_path(self):

        class Inner:
            value = 42

        class Outer:
            inner = Inner()

        result = ArchConfig.get_nested_attr(Outer(), "inner.value")
        assert result == 42

    def test_deep_path(self):

        class A:
            val = "found"

        class B:
            a = A()

        class C:
            b = B()

        result = ArchConfig.get_nested_attr(C(), "b.a.val")
        assert result == "found"


class TestMoeIntermediateOverride:

    def test_moe_intermediate_override(self):
        """moe_intermediate_size from config should override default."""
        config = MockConfig(
            "qwen3_moe",
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=1024,
        )
        arch = detect_arch_config(config)
        assert arch.expert_intermediate_size == 1024
