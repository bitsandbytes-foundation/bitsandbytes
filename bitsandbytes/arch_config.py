"""Architecture adapter configs for KbitLoraModel.

Each ArchConfig maps generic projection/module names to model-specific
attribute paths, so KbitLoraModel can handle different HF architectures
with a single code path.
"""

from dataclasses import dataclass, field


@dataclass
class ArchConfig:
    """Architecture-specific configuration for a model family."""

    # How to access layers from the HF model
    layers_path: str  # e.g., "model.layers"

    # Embedding, final norm, and LM head paths
    embed_path: str  # e.g., "model.embed_tokens"
    final_norm_path: str  # e.g., "model.norm"
    lm_head_path: str  # e.g., "lm_head"

    # Attention projection names (attributes on the attn_module)
    attn_module: str  # e.g., "self_attn"
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str

    # MLP projection names (attributes on the mlp_module)
    mlp_module: str  # e.g., "mlp"
    gate_proj: str
    up_proj: str
    down_proj: str

    # Norm names (attributes on the layer)
    input_norm: str  # e.g., "input_layernorm"
    post_attn_norm: str  # e.g., "post_attention_layernorm"

    # QK norm (Qwen3 has per-head QK normalization)
    has_qk_norm: bool = False
    q_norm: str = "q_norm"
    k_norm: str = "k_norm"

    # MoE configuration
    is_moe: bool = False
    moe_router_path: str = ""  # e.g., "mlp.gate" — path from layer to router
    moe_experts_path: str = ""  # e.g., "mlp.experts" — path from layer to expert list
    shared_expert_path: str = ""  # e.g., "mlp.shared_expert"
    has_shared_expert: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    expert_intermediate_size: int = 0
    # Which layers are dense (not MoE). None = check all layers.
    # For GLM-4.7: first 3 layers are dense, rest are MoE.
    dense_layer_indices: list[int] | None = None
    # Expert projection names (attributes on each expert module).
    # Defaults match Qwen3-MoE / standard HF MoE.
    expert_gate_proj: str = "gate_proj"
    expert_up_proj: str = "up_proj"
    expert_down_proj: str = "down_proj"

    def is_moe_layer(self, global_layer_idx: int) -> bool:
        """Check if a specific layer index is an MoE layer."""
        if not self.is_moe:
            return False
        if self.dense_layer_indices is None:
            return True
        return global_layer_idx not in self.dense_layer_indices

    @staticmethod
    def get_nested_attr(obj, path: str):
        """Navigate dotted path like 'model.layers' to get the attribute."""
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj


# ─── Pre-defined architecture configs ───


LLAMA_CONFIG = ArchConfig(
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_module="self_attn",
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    o_proj="o_proj",
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",
    post_attn_norm="post_attention_layernorm",
)

MISTRAL_CONFIG = ArchConfig(
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_module="self_attn",
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    o_proj="o_proj",
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",
    post_attn_norm="post_attention_layernorm",
)

QWEN3_DENSE_CONFIG = ArchConfig(
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_module="self_attn",
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    o_proj="o_proj",
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",
    post_attn_norm="post_attention_layernorm",
    has_qk_norm=True,
    q_norm="q_norm",
    k_norm="k_norm",
)

QWEN2_CONFIG = ArchConfig(
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_module="self_attn",
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    o_proj="o_proj",
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",
    post_attn_norm="post_attention_layernorm",
)

QWEN3_MOE_CONFIG = ArchConfig(
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_module="self_attn",
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    o_proj="o_proj",
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",
    post_attn_norm="post_attention_layernorm",
    has_qk_norm=True,
    q_norm="q_norm",
    k_norm="k_norm",
    is_moe=True,
    moe_router_path="mlp.gate",
    moe_experts_path="mlp.experts",
    has_shared_expert=False,
    num_experts=128,
    num_active_experts=8,
    expert_intermediate_size=768,
    dense_layer_indices=None,  # all layers are MoE (decoder_sparse_step=1)
)

# GLM-4.7 config — attribute paths marked VERIFY need checking against the
# actual model before use. Load with device_map="meta" and inspect named_modules().
GLM4_MOE_CONFIG = ArchConfig(
    layers_path="model.layers",  # VERIFY
    embed_path="model.embed_tokens",  # VERIFY
    final_norm_path="model.norm",  # VERIFY
    lm_head_path="lm_head",  # VERIFY
    attn_module="self_attn",  # VERIFY
    q_proj="q_proj",  # VERIFY
    k_proj="k_proj",  # VERIFY
    v_proj="v_proj",  # VERIFY
    o_proj="o_proj",  # VERIFY
    mlp_module="mlp",
    gate_proj="gate_proj",
    up_proj="up_proj",
    down_proj="down_proj",
    input_norm="input_layernorm",  # VERIFY
    post_attn_norm="post_attention_layernorm",  # VERIFY
    is_moe=True,
    moe_router_path="mlp.gate",  # VERIFY
    moe_experts_path="mlp.experts",  # VERIFY
    shared_expert_path="mlp.shared_expert",  # VERIFY
    has_shared_expert=True,
    num_experts=160,
    num_active_experts=8,
    expert_intermediate_size=1536,
    dense_layer_indices=[0, 1, 2],  # first_k_dense_replace=3
)


# ─── Auto-detection ───

_MODEL_TYPE_MAP = {
    "llama": LLAMA_CONFIG,
    "mistral": MISTRAL_CONFIG,
    "qwen2": QWEN2_CONFIG,
    "qwen3": QWEN3_DENSE_CONFIG,
    "qwen3_moe": QWEN3_MOE_CONFIG,
    "glm4": GLM4_MOE_CONFIG,
}


def detect_arch_config(config) -> ArchConfig:
    """Detect architecture config from a HuggingFace model config."""
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError("Model config has no model_type attribute")
    if model_type not in _MODEL_TYPE_MAP:
        supported = ", ".join(sorted(_MODEL_TYPE_MAP.keys()))
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported: {supported}"
        )

    arch = _MODEL_TYPE_MAP[model_type]

    # For MoE models, override num_experts etc from the actual config if present
    if arch.is_moe:
        num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
        if num_experts is not None and num_experts != arch.num_experts:
            # Create a copy with updated values
            from dataclasses import replace
            arch = replace(arch, num_experts=num_experts)

        num_active = getattr(config, "num_experts_per_tok", None) or getattr(config, "num_selected_experts", None)
        if num_active is not None and num_active != arch.num_active_experts:
            from dataclasses import replace
            arch = replace(arch, num_active_experts=num_active)

        moe_inter = getattr(config, "moe_intermediate_size", None)
        if moe_inter is not None and moe_inter != arch.expert_intermediate_size:
            from dataclasses import replace
            arch = replace(arch, expert_intermediate_size=moe_inter)

    return arch
