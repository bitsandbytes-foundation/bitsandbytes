"""Pre-quantized checkpoint save/load for KbitLoraModel.

Saves quantized weights to layer-ordered safetensors files for efficient
NVMe streaming. Saves/loads LoRA adapters separately.
"""

from collections import OrderedDict
from typing import Optional

import torch

from safetensors.torch import save_file
from safetensors import safe_open


def save_quantized(model, path: str):
    """Save pre-quantized model weights to layer-ordered safetensors.

    Tensors are inserted in layer order so that on-disk layout is optimal
    for sequential NVMe reads during weight streaming.

    Args:
        model: KbitLoraModel instance.
        path: Output safetensors file path.
    """
    tensors = OrderedDict()

    for i, layer_info in enumerate(model._layer_data):
        prefix = f"layer.{i}"

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            for wk in ["packed", "absmax", "codebook"]:
                tensors[f"{prefix}.attn.{proj}.{wk}"] = layer_info[proj][wk]

        # MLP or MoE
        if layer_info.get("is_moe"):
            # Router weight
            tensors[f"{prefix}.moe.router_weight"] = layer_info["router_weight"]

            # Shared expert (if present)
            if "shared_gate_proj" in layer_info:
                for proj in ["shared_gate_proj", "shared_up_proj", "shared_down_proj"]:
                    for wk in ["packed", "absmax", "codebook"]:
                        tensors[f"{prefix}.moe.{proj}.{wk}"] = layer_info[proj][wk]

            # Expert weights (concatenated)
            for expert_proj in ["gate", "up", "down"]:
                for suffix in ["packed", "absmax"]:
                    key = f"expert_{expert_proj}_{suffix}"
                    tensors[f"{prefix}.moe.experts.{expert_proj}.{suffix}"] = layer_info[key]
            tensors[f"{prefix}.moe.experts.codebook"] = layer_info["expert_codebook"]
        else:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for wk in ["packed", "absmax", "codebook"]:
                    tensors[f"{prefix}.mlp.{proj}.{wk}"] = layer_info[proj][wk]

        # Norm weights
        for nk in ["input_layernorm", "post_attention_layernorm"]:
            if nk in layer_info:
                tensors[f"{prefix}.{nk}.weight"] = layer_info[nk].data

        # QK norms
        for nk in ["q_norm", "k_norm"]:
            if nk in layer_info:
                tensors[f"{prefix}.{nk}.weight"] = layer_info[nk].data

    # LM head
    if model._lm_head_info is not None:
        lm = model._lm_head_info
        for wk in ["packed", "absmax", "codebook"]:
            tensors[f"lm_head.{wk}"] = lm[wk]

    # Final norm
    if "final_norm_weight" in model._norm_weights:
        tensors["final_norm.weight"] = model._norm_weights["final_norm_weight"].data

    # Embedding
    if model.embed_tokens is not None:
        tensors["embed_tokens.weight"] = model.embed_tokens.weight.data

    # Metadata — comprehensive, enables load_quantized without the HF model
    metadata = {
        # Model architecture
        "model_type": model.model_type,
        "hidden_size": str(model.hidden_size),
        "num_layers": str(model.num_layers),
        "num_loaded_layers": str(model._num_loaded_layers),
        "layer_start": str(model._layer_start),
        "layer_end": str(model._layer_end),
        "num_attention_heads": str(model.num_heads),
        "num_key_value_heads": str(model.num_kv_heads),
        "head_dim": str(model.head_dim),
        "intermediate_size": str(model.intermediate_size),
        "vocab_size": str(model.vocab_size),
        "rms_norm_eps": str(model.rms_norm_eps),
        "rope_theta": str(model.rope_theta),
        # Quantization config
        "k_attention": str(model.k_attention),
        "k_mlp": str(model.k_mlp),
        "k_lm_head": str(model.k_lm_head),
        "k_experts": str(model.k_experts),
        "k_shared_expert": str(model.k_shared_expert),
        # MoE config
        "is_moe": str(model.arch.is_moe),
        "num_experts": str(model.arch.num_experts),
        "num_active_experts": str(model.arch.num_active_experts),
        "expert_intermediate_size": str(model.arch.expert_intermediate_size),
        "has_shared_expert": str(model.arch.has_shared_expert),
        "has_qk_norm": str(model.arch.has_qk_norm),
    }

    # Dense layer indices (comma-separated, empty if None or all MoE)
    if model.arch.dense_layer_indices is not None:
        metadata["dense_layer_indices"] = ",".join(
            str(i) for i in model.arch.dense_layer_indices
        )
    else:
        metadata["dense_layer_indices"] = ""

    # Per-projection dimensions (needed for LoRA initialization in load_quantized)
    for i, layer_info in enumerate(model._layer_data):
        prefix = f"layer.{i}"

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            metadata[f"{prefix}.attn.{proj}.N"] = str(layer_info[proj]["N"])
            metadata[f"{prefix}.attn.{proj}.K"] = str(layer_info[proj]["K"])
            metadata[f"{prefix}.attn.{proj}.N_padded"] = str(layer_info[proj]["N_padded"])
            metadata[f"{prefix}.attn.{proj}.k"] = str(layer_info[proj]["k"])

        # MLP or MoE
        if layer_info.get("is_moe"):
            # Shared expert dims
            if "shared_gate_proj" in layer_info:
                for proj in ["shared_gate_proj", "shared_up_proj", "shared_down_proj"]:
                    metadata[f"{prefix}.moe.{proj}.N"] = str(layer_info[proj]["N"])
                    metadata[f"{prefix}.moe.{proj}.K"] = str(layer_info[proj]["K"])
                    metadata[f"{prefix}.moe.{proj}.N_padded"] = str(layer_info[proj]["N_padded"])
                    metadata[f"{prefix}.moe.{proj}.k"] = str(layer_info[proj]["k"])

            # Expert dims (same for all experts — store once)
            metadata[f"{prefix}.moe.experts.N"] = str(layer_info.get("expert_N", 0))
            metadata[f"{prefix}.moe.experts.K"] = str(layer_info.get("expert_K", 0))
            metadata[f"{prefix}.moe.experts.N_padded"] = str(layer_info.get("expert_N_padded", 0))
            metadata[f"{prefix}.moe.experts.k"] = str(layer_info.get("expert_k", 0))
        else:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                metadata[f"{prefix}.mlp.{proj}.N"] = str(layer_info[proj]["N"])
                metadata[f"{prefix}.mlp.{proj}.K"] = str(layer_info[proj]["K"])
                metadata[f"{prefix}.mlp.{proj}.N_padded"] = str(layer_info[proj]["N_padded"])
                metadata[f"{prefix}.mlp.{proj}.k"] = str(layer_info[proj]["k"])

    # LM head dims
    if model._lm_head_info is not None:
        lm = model._lm_head_info
        metadata["lm_head.N"] = str(lm["N"])
        metadata["lm_head.K"] = str(lm["K"])
        metadata["lm_head.N_padded"] = str(lm["N_padded"])
        metadata["lm_head.k"] = str(lm["k"])

    # Move all tensors to CPU for saving
    cpu_tensors = OrderedDict()
    for k, v in tensors.items():
        cpu_tensors[k] = v.contiguous().cpu()

    save_file(cpu_tensors, path, metadata=metadata)


def save_lora(model, path: str):
    """Save LoRA adapter weights + norm weights to safetensors.

    Args:
        model: KbitLoraModel instance.
        path: Output safetensors file path.
    """
    state = OrderedDict()

    for name, param in model._lora_params.items():
        state[f"lora.{name}"] = param.data.contiguous().cpu()

    for name, param in model._norm_weights.items():
        state[f"norm.{name}"] = param.data.contiguous().cpu()

    metadata = {
        "lora_r": str(model.lora_r),
        "lora_s": str(model.lora_s),
        "model_type": model.model_type,
    }

    save_file(state, path, metadata=metadata)


def load_lora(model, path: str, device: Optional[torch.device] = None):
    """Load LoRA adapter weights + norm weights from safetensors.

    Args:
        model: KbitLoraModel instance.
        path: Input safetensors file path.
        device: Device to load onto. Defaults to model's target device.
    """
    if device is None:
        device = model._target_device

    f = safe_open(path, framework="pt", device=str(device))

    for name, param in model._lora_params.items():
        key = f"lora.{name}"
        if key in f.keys():
            param.data.copy_(f.get_tensor(key))

    for name, param in model._norm_weights.items():
        key = f"norm.{name}"
        if key in f.keys():
            param.data.copy_(f.get_tensor(key))
