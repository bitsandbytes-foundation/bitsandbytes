"""Pre-quantized checkpoint save/load for KbitLoraModel.

Saves quantized weights to layer-ordered safetensors files for efficient
NVMe streaming. Saves/loads LoRA adapters separately. Includes a streaming
quantizer that converts HF checkpoints layer-by-layer with minimal memory.
"""

from collections import OrderedDict
import json
import os
import shutil
import struct
from typing import Optional

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from bitsandbytes.arch_config import ArchConfig, detect_arch_config


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
        metadata["dense_layer_indices"] = ",".join(str(i) for i in model.arch.dense_layer_indices)
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


# ─── Streaming quantizer ───


def _compute_quantized_sizes(N: int, K: int, k: int):
    """Compute output tensor sizes for quantize_kbit without running it.

    Returns (N_padded, packed_numel, absmax_numel, codebook_numel).
    """
    N_padded = ((N + 127) // 128) * 128
    n_elements = N_padded * K
    num_blocks = -(n_elements // -32)  # ceil_div
    packed_numel = num_blocks * k + k
    absmax_numel = num_blocks + 1
    codebook_numel = 1 << k
    return N_padded, packed_numel, absmax_numel, codebook_numel


def streaming_quantize(
    model_name_or_path: str,
    output_path: str,
    k: int = 4,
    k_config: Optional[dict[str, int]] = None,
    arch_config: Optional[ArchConfig] = None,
    device: Optional[torch.device] = None,
):
    """Quantize a HuggingFace model layer-by-layer and write to safetensors.

    Two-pass approach:
      Pass 1: Read tensor shapes from shard headers, compute quantized sizes,
              build output safetensors header and metadata.
      Pass 2: Load each layer's weights onto GPU, quantize, write to the
              pre-allocated output file.

    Memory: only one layer's fp16 weights on GPU at a time (~200 MB dense,
    ~2.4 GB for 160 MoE experts per projection type). Total RAM footprint
    is ~4 GB for concatenated expert packed/absmax buffers.

    Args:
        model_name_or_path: Local directory or HuggingFace Hub model ID.
        output_path: Output safetensors file path.
        k: Default bit width for quantization (2-5).
        k_config: Optional per-module bit width overrides.
        arch_config: Optional ArchConfig override.
        device: GPU device for quantization kernels.
    """
    if device is None:
        device = torch.device("cuda:0")

    from transformers import AutoConfig

    import bitsandbytes.functional as F

    k_config = k_config or {}
    k_attn = k_config.get("attention", k)
    k_mlp = k_config.get("mlp", k)
    k_lm_head = k_config.get("lm_head", k)
    k_experts = k_config.get("experts", k)
    k_shared_expert = k_config.get("shared_expert", k_mlp)

    # ─── Load model config and detect architecture ───
    config = AutoConfig.from_pretrained(model_name_or_path)
    arch = arch_config or detect_arch_config(config)

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size
    num_layers = config.num_hidden_layers
    rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
    rope_theta = getattr(config, "rope_theta", 10000.0)

    # ─── Resolve model directory ───
    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_name_or_path)

    # ─── Build weight map: tensor_name → shard_filename ───
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fp:
            index_data = json.load(fp)
        weight_map = index_data["weight_map"]
        shard_set = set(weight_map.values())
    else:
        weight_map = None
        shard_set = {"model.safetensors"}

    # ─── Parse shard headers to get tensor shapes without loading data ───
    shard_headers = {}
    for shard_name in shard_set:
        shard_path = os.path.join(model_dir, shard_name)
        with open(shard_path, "rb") as fp:
            hs = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(hs))
        header.pop("__metadata__", None)
        shard_headers[shard_name] = header

    def _get_shard(hf_name: str) -> str:
        if weight_map is not None:
            return weight_map[hf_name]
        return "model.safetensors"

    def _get_shape(hf_name: str) -> list:
        shard = _get_shard(hf_name)
        return shard_headers[shard][hf_name]["shape"]

    def _get_dtype(hf_name: str) -> str:
        shard = _get_shard(hf_name)
        return shard_headers[shard][hf_name]["dtype"]

    # ─── HF tensor name helpers ───
    def _hf_attn(layer_idx, proj_attr):
        return f"{arch.layers_path}.{layer_idx}.{arch.attn_module}.{proj_attr}.weight"

    def _hf_mlp(layer_idx, proj_attr):
        return f"{arch.layers_path}.{layer_idx}.{arch.mlp_module}.{proj_attr}.weight"

    def _hf_expert(layer_idx, expert_idx, proj_attr):
        return f"{arch.layers_path}.{layer_idx}.{arch.moe_experts_path}.{expert_idx}.{proj_attr}.weight"

    def _hf_shared_expert(layer_idx, proj_attr):
        return f"{arch.layers_path}.{layer_idx}.{arch.shared_expert_path}.{proj_attr}.weight"

    def _hf_router(layer_idx):
        return f"{arch.layers_path}.{layer_idx}.{arch.moe_router_path}.weight"

    def _hf_norm(layer_idx, norm_attr):
        return f"{arch.layers_path}.{layer_idx}.{norm_attr}.weight"

    def _hf_qk_norm(layer_idx, norm_name):
        return f"{arch.layers_path}.{layer_idx}.{arch.attn_module}.{norm_name}.weight"

    # ─── PASS 1: Build output header ───

    # tensor_specs: name → (dtype_str, shape_list, byte_size)
    tensor_specs = OrderedDict()
    metadata = {}

    _DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I32": 4, "U8": 1}

    def _add_quantized(out_name, hf_name, k_val, meta_prefix):
        """Register a quantized projection in the output layout."""
        shape = _get_shape(hf_name)
        N, K_dim = shape[0], shape[1]
        N_padded, packed_n, absmax_n, cb_n = _compute_quantized_sizes(N, K_dim, k_val)

        tensor_specs[f"{out_name}.packed"] = ("I32", [packed_n], packed_n * 4)
        tensor_specs[f"{out_name}.absmax"] = ("F32", [absmax_n], absmax_n * 4)
        tensor_specs[f"{out_name}.codebook"] = ("F32", [cb_n], cb_n * 4)

        metadata[f"{meta_prefix}.N"] = str(N)
        metadata[f"{meta_prefix}.K"] = str(K_dim)
        metadata[f"{meta_prefix}.N_padded"] = str(N_padded)
        metadata[f"{meta_prefix}.k"] = str(k_val)

    def _add_copy(out_name, hf_name, force_dtype=None):
        """Register a non-quantized tensor copy."""
        shape = _get_shape(hf_name)
        dtype_str = force_dtype or _get_dtype(hf_name)
        numel = 1
        for s in shape:
            numel *= s
        tensor_specs[out_name] = (dtype_str, shape, numel * _DTYPE_BYTES[dtype_str])

    def _add_expert_concat(out_prefix, layer_idx, proj_attr, k_val, meta_prefix):
        """Register concatenated expert projections."""
        hf_0 = _hf_expert(layer_idx, 0, proj_attr)
        shape = _get_shape(hf_0)
        N, K_dim = shape[0], shape[1]
        N_padded, packed_per, absmax_per, _ = _compute_quantized_sizes(N, K_dim, k_val)

        n_exp = arch.num_experts
        total_packed = packed_per * n_exp
        total_absmax = absmax_per * n_exp

        tensor_specs[f"{out_prefix}.packed"] = ("I32", [total_packed], total_packed * 4)
        tensor_specs[f"{out_prefix}.absmax"] = ("F32", [total_absmax], total_absmax * 4)

        # Metadata is stored once per layer (same dims for all experts)
        metadata[f"{meta_prefix}.N"] = str(N)
        metadata[f"{meta_prefix}.K"] = str(K_dim)
        metadata[f"{meta_prefix}.N_padded"] = str(N_padded)
        metadata[f"{meta_prefix}.k"] = str(k_val)

    # --- Per-layer tensor specs ---
    _attn_projs = [
        ("q_proj", arch.q_proj),
        ("k_proj", arch.k_proj),
        ("v_proj", arch.v_proj),
        ("o_proj", arch.o_proj),
    ]
    _mlp_projs = [
        ("gate_proj", arch.gate_proj),
        ("up_proj", arch.up_proj),
        ("down_proj", arch.down_proj),
    ]
    _expert_projs = [
        ("gate", arch.expert_gate_proj),
        ("up", arch.expert_up_proj),
        ("down", arch.expert_down_proj),
    ]

    for i in range(num_layers):
        pfx = f"layer.{i}"

        # Attention
        for name, attr in _attn_projs:
            _add_quantized(f"{pfx}.attn.{name}", _hf_attn(i, attr), k_attn, f"{pfx}.attn.{name}")

        # MLP or MoE
        if arch.is_moe_layer(i):
            # Router weight
            _add_copy(f"{pfx}.moe.router_weight", _hf_router(i), force_dtype="BF16")

            # Shared expert
            if arch.has_shared_expert:
                for name, attr in [
                    ("shared_gate_proj", arch.gate_proj),
                    ("shared_up_proj", arch.up_proj),
                    ("shared_down_proj", arch.down_proj),
                ]:
                    _add_quantized(
                        f"{pfx}.moe.{name}",
                        _hf_shared_expert(i, attr),
                        k_shared_expert,
                        f"{pfx}.moe.{name}",
                    )

            # Experts (concatenated)
            for name, attr in _expert_projs:
                _add_expert_concat(
                    f"{pfx}.moe.experts.{name}",
                    i,
                    attr,
                    k_experts,
                    f"{pfx}.moe.experts",
                )

            # Expert codebook (shared across projection types)
            hf_0 = _hf_expert(i, 0, arch.expert_gate_proj)
            shape_0 = _get_shape(hf_0)
            _, _, _, cb_n = _compute_quantized_sizes(shape_0[0], shape_0[1], k_experts)
            tensor_specs[f"{pfx}.moe.experts.codebook"] = ("F32", [cb_n], cb_n * 4)
        else:
            for name, attr in _mlp_projs:
                _add_quantized(f"{pfx}.mlp.{name}", _hf_mlp(i, attr), k_mlp, f"{pfx}.mlp.{name}")

        # Norms
        _add_copy(f"{pfx}.input_layernorm.weight", _hf_norm(i, arch.input_norm), force_dtype="BF16")
        _add_copy(f"{pfx}.post_attention_layernorm.weight", _hf_norm(i, arch.post_attn_norm), force_dtype="BF16")

        if arch.has_qk_norm:
            _add_copy(f"{pfx}.q_norm.weight", _hf_qk_norm(i, arch.q_norm), force_dtype="BF16")
            _add_copy(f"{pfx}.k_norm.weight", _hf_qk_norm(i, arch.k_norm), force_dtype="BF16")

    # LM head
    lm_hf = f"{arch.lm_head_path}.weight"
    _add_quantized("lm_head", lm_hf, k_lm_head, "lm_head")

    # Final norm
    _add_copy("final_norm.weight", f"{arch.final_norm_path}.weight", force_dtype="BF16")

    # Embedding (keep original dtype)
    _add_copy("embed_tokens.weight", f"{arch.embed_path}.weight")

    # --- Global metadata ---
    metadata.update(
        {
            "model_type": config.model_type,
            "hidden_size": str(hidden_size),
            "num_layers": str(num_layers),
            "num_loaded_layers": str(num_layers),
            "layer_start": "0",
            "layer_end": str(num_layers),
            "num_attention_heads": str(num_heads),
            "num_key_value_heads": str(num_kv_heads),
            "head_dim": str(head_dim),
            "intermediate_size": str(intermediate_size),
            "vocab_size": str(vocab_size),
            "rms_norm_eps": str(rms_norm_eps),
            "rope_theta": str(rope_theta),
            "k_attention": str(k_attn),
            "k_mlp": str(k_mlp),
            "k_lm_head": str(k_lm_head),
            "k_experts": str(k_experts),
            "k_shared_expert": str(k_shared_expert),
            "is_moe": str(arch.is_moe),
            "num_experts": str(arch.num_experts),
            "num_active_experts": str(arch.num_active_experts),
            "expert_intermediate_size": str(arch.expert_intermediate_size),
            "has_shared_expert": str(arch.has_shared_expert),
            "has_qk_norm": str(arch.has_qk_norm),
            "dense_layer_indices": ",".join(str(x) for x in (arch.dense_layer_indices or [])),
        }
    )

    # ─── Write safetensors header + pre-allocate file ───

    sf_header = {"__metadata__": metadata}
    data_offset = 0
    for name, (dtype_str, shape, byte_size) in tensor_specs.items():
        sf_header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [data_offset, data_offset + byte_size],
        }
        data_offset += byte_size

    header_json = json.dumps(sf_header, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)
    data_start = 8 + header_size

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fp:
        fp.write(struct.pack("<Q", header_size))
        fp.write(header_json)
        if data_offset > 0:
            fp.seek(data_start + data_offset - 1)
            fp.write(b"\0")

    # ─── PASS 2: Quantize and write ───

    # Build byte offset lookup from tensor_specs order
    tensor_byte_offsets = {}
    offset = 0
    for name, (_, _, byte_size) in tensor_specs.items():
        tensor_byte_offsets[name] = offset
        offset += byte_size

    # Lazy shard handle cache
    _shard_handles = {}

    def _load_hf(hf_name):
        shard = _get_shard(hf_name)
        if shard not in _shard_handles:
            _shard_handles[shard] = safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu")
        return _shard_handles[shard].get_tensor(hf_name)

    _TORCH_DTYPE = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}

    with open(output_path, "r+b") as out_fp:

        def _write(out_name, tensor):
            t = tensor.contiguous().cpu()
            if t.dtype == torch.bfloat16:
                # numpy doesn't support bfloat16; write raw bytes via storage
                nbytes = t.element_size() * t.numel()
                raw = bytes(t.untyped_storage())[:nbytes]
            else:
                raw = t.numpy().tobytes()
            out_fp.seek(data_start + tensor_byte_offsets[out_name])
            out_fp.write(raw)

        def _quantize_and_write(out_prefix, hf_name, k_val):
            """Load, pad, quantize one projection, write packed/absmax/codebook."""
            weight = _load_hf(hf_name).to(device)
            N, _K_dim = weight.shape
            N_padded = ((N + 127) // 128) * 128
            if N_padded != N:
                w = torch.nn.functional.pad(weight.float(), (0, 0, 0, N_padded - N))
            else:
                w = weight.float()
            del weight

            packed, absmax, codebook = F.quantize_kbit(w.reshape(-1), k=k_val, absmax_format="fp32")
            del w

            _write(f"{out_prefix}.packed", packed)
            _write(f"{out_prefix}.absmax", absmax)
            _write(f"{out_prefix}.codebook", codebook)
            del packed, absmax, codebook
            torch.cuda.empty_cache()

        def _copy_and_write(out_name, hf_name):
            """Load a tensor, optionally convert dtype, write."""
            tensor = _load_hf(hf_name)
            target_dtype_str = tensor_specs[out_name][0]
            target_dtype = _TORCH_DTYPE.get(target_dtype_str)
            if target_dtype is not None and tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            _write(out_name, tensor)
            del tensor

        # --- Process layers ---
        for i in range(num_layers):
            pfx = f"layer.{i}"

            # Attention
            for name, attr in _attn_projs:
                _quantize_and_write(f"{pfx}.attn.{name}", _hf_attn(i, attr), k_attn)

            # MLP or MoE
            if arch.is_moe_layer(i):
                # Router
                _copy_and_write(f"{pfx}.moe.router_weight", _hf_router(i))

                # Shared expert
                if arch.has_shared_expert:
                    for name, attr in [
                        ("shared_gate_proj", arch.gate_proj),
                        ("shared_up_proj", arch.up_proj),
                        ("shared_down_proj", arch.down_proj),
                    ]:
                        _quantize_and_write(
                            f"{pfx}.moe.{name}",
                            _hf_shared_expert(i, attr),
                            k_shared_expert,
                        )

                # Experts (concatenated per projection type)
                expert_codebook = None
                for name, attr in _expert_projs:
                    all_packed = []
                    all_absmax = []

                    for e in range(arch.num_experts):
                        w = _load_hf(_hf_expert(i, e, attr)).to(device)
                        N, _K_dim = w.shape
                        N_padded = ((N + 127) // 128) * 128
                        if N_padded != N:
                            w = torch.nn.functional.pad(w.float(), (0, 0, 0, N_padded - N))
                        else:
                            w = w.float()

                        packed, absmax, codebook = F.quantize_kbit(w.reshape(-1), k=k_experts, absmax_format="fp32")
                        del w

                        all_packed.append(packed.cpu())
                        all_absmax.append(absmax.cpu())
                        if expert_codebook is None:
                            expert_codebook = codebook.cpu()
                        del packed, absmax, codebook

                    torch.cuda.empty_cache()

                    cat_packed = torch.cat(all_packed)
                    cat_absmax = torch.cat(all_absmax)
                    _write(f"{pfx}.moe.experts.{name}.packed", cat_packed)
                    _write(f"{pfx}.moe.experts.{name}.absmax", cat_absmax)
                    del all_packed, all_absmax, cat_packed, cat_absmax

                # Expert codebook (captured from first expert of first proj type)
                _write(f"{pfx}.moe.experts.codebook", expert_codebook)
                del expert_codebook
            else:
                for name, attr in _mlp_projs:
                    _quantize_and_write(f"{pfx}.mlp.{name}", _hf_mlp(i, attr), k_mlp)

            # Norms
            _copy_and_write(f"{pfx}.input_layernorm.weight", _hf_norm(i, arch.input_norm))
            _copy_and_write(f"{pfx}.post_attention_layernorm.weight", _hf_norm(i, arch.post_attn_norm))

            if arch.has_qk_norm:
                _copy_and_write(f"{pfx}.q_norm.weight", _hf_qk_norm(i, arch.q_norm))
                _copy_and_write(f"{pfx}.k_norm.weight", _hf_qk_norm(i, arch.k_norm))

            print(f"  Layer {i}/{num_layers} done")

        # LM head
        _quantize_and_write("lm_head", f"{arch.lm_head_path}.weight", k_lm_head)

        # Final norm
        _copy_and_write("final_norm.weight", f"{arch.final_norm_path}.weight")

        # Embedding
        _copy_and_write("embed_tokens.weight", f"{arch.embed_path}.weight")

    # Close shard handles
    _shard_handles.clear()

    # Copy config.json alongside output
    config_src = os.path.join(model_dir, "config.json")
    output_dir = os.path.dirname(os.path.abspath(output_path))
    config_dst = os.path.join(output_dir, "config.json")
    if os.path.exists(config_src) and os.path.abspath(config_src) != os.path.abspath(config_dst):
        shutil.copy(config_src, config_dst)

    print(f"Streaming quantize complete: {output_path}")
