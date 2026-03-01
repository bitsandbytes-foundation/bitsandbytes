"""KbitLoraModel: Model patcher for dense and MoE transformer architectures.

Replaces all linear layers with kbit-quantized weights + LoRA adapters,
patches attention with chunked Flash Attention, patches MLP with chunked
LoRA_MLP_Kbit, and patches norms with CUDA RMSNorm.

No PEFT dependency — manages LoRA adapters directly for efficiency.

Supported model_types: llama, mistral, qwen2, qwen3, qwen3_moe, glm4
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from bitsandbytes.arch_config import ArchConfig, detect_arch_config
from bitsandbytes.attention import chunked_flash_attention
from bitsandbytes.autograd.chunked_ce import chunked_cross_entropy
from bitsandbytes.autograd.lora_kbit import LoRA_W_Kbit
from bitsandbytes.autograd.training_kernels import rmsnorm, rope
from bitsandbytes.chunked import chunked_mlp_forward
import bitsandbytes.functional as F
from bitsandbytes.moe import moe_expert_forward, moe_router_dispatch
from bitsandbytes.training import checkpoint_cpu_offload


class KbitLoraModel(nn.Module):
    """Wraps a HuggingFace CausalLM model with kbit quantization + LoRA.

    Quantizes all linear weights (attention, MLP/MoE, LM head) to k-bit,
    adds trainable LoRA adapters, and patches forward methods to use
    our optimized CUDA kernels.

    Supports dense models (Llama, Mistral, Qwen) and MoE models
    (Qwen3-MoE, GLM-4.7) via ArchConfig adapters.

    Args:
        model: HuggingFace CausalLM model (e.g., from AutoModelForCausalLM).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor (effective scale = lora_alpha / lora_r).
        k: Bit width for quantization (2-5). Default 4. Used as fallback
            when k_config doesn't specify a value for a module type.
        k_config: Optional dict mapping module types to bit widths.
            Supported keys: "attention", "mlp", "lm_head", "experts",
            "shared_expert". Example: {"attention": 4, "mlp": 3, "experts": 2}
        attn_chunk_size: Sequence chunk size for attention. Default 4096.
        mlp_chunk_size: Sequence chunk size for MLP. Default 4096.
        ce_chunk_size: Vocab chunk size for cross-entropy. Default 8192.
        compute_dtype: Computation dtype. Default bf16.
        cpu_offload: If True, offload inter-layer activations to CPU during
            forward and reload during backward. Saves GPU memory at cost
            of CPU<->GPU bandwidth. Default False.
        layer_range: Optional tuple (start, end) to only load decoder layers
            [start, end). Used for pipeline parallelism so each rank only
            loads its assigned layers. Default None (all layers).
        include_embed: Whether to keep the embedding layer. Default True.
            Set False for non-first pipeline stages.
        include_lm_head: Whether to quantize and keep the LM head. Default True.
            Set False for non-last pipeline stages.
        target_device: Device for quantized weights and LoRA params. If None,
            uses the model's device. Set this when loading the HF model on
            CPU to stream weights to GPU one layer at a time (minimizes
            peak GPU memory). Example: torch.device("cuda:0").
        weight_streaming: If True, keep frozen quantized weights in CPU pinned
            memory and stream them to GPU layer-by-layer during forward/backward.
            Uses a double-buffered async pipeline: while the GPU computes on one
            layer, the next layer's weights transfer via PCIe DMA on a dedicated
            CUDA stream. Requires cpu_offload=True (gradient checkpointing) so
            that backward also streams one layer at a time.
        arch_config: Optional ArchConfig override. Auto-detected from
            model.config.model_type if not provided.
        lora_on_experts: If True, add LoRA adapters to MoE expert projections
            in addition to attention and shared expert. Default False.
        expert_chunk_size: Number of experts to process at once in MoE forward.
            Default 32.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_r: int = 64,
        lora_alpha: float = 16.0,
        k: int = 4,
        k_config: Optional[dict[str, int]] = None,
        attn_chunk_size: int = 4096,
        mlp_chunk_size: int = 4096,
        ce_chunk_size: int = 8192,
        compute_dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False,
        layer_range: Optional[tuple[int, int]] = None,
        include_embed: bool = True,
        include_lm_head: bool = True,
        target_device: Optional[torch.device] = None,
        weight_streaming: bool = False,
        arch_config: Optional[ArchConfig] = None,
        lora_on_experts: bool = False,
        expert_chunk_size: int = 32,
    ):
        super().__init__()

        config = model.config

        # Detect or validate architecture
        if arch_config is not None:
            self.arch = arch_config
        else:
            self.arch = detect_arch_config(config)

        self.config = config
        self.model_type = config.model_type
        self.lora_r = lora_r
        self.lora_s = lora_alpha / lora_r
        self.k = k
        self.k_config = k_config or {}
        self.k_attention = self.k_config.get("attention", k)
        self.k_mlp = self.k_config.get("mlp", k)
        self.k_lm_head = self.k_config.get("lm_head", k)
        self.k_experts = self.k_config.get("experts", k)
        self.k_shared_expert = self.k_config.get("shared_expert", self.k_mlp)
        self.attn_chunk_size = attn_chunk_size
        self.mlp_chunk_size = mlp_chunk_size
        self.ce_chunk_size = ce_chunk_size
        self.compute_dtype = compute_dtype
        self.cpu_offload = cpu_offload
        self.weight_streaming = weight_streaming
        self.include_embed = include_embed
        self.include_lm_head = include_lm_head
        self.lora_on_experts = lora_on_experts
        self.expert_chunk_size = expert_chunk_size

        if weight_streaming and not cpu_offload:
            raise ValueError(
                "weight_streaming=True requires cpu_offload=True. "
                "Without gradient checkpointing, autograd saves all layers' "
                "weights on GPU for backward, defeating the memory savings."
            )

        # Extract model dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.intermediate_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        # Determine layer range
        total_layers = config.num_hidden_layers
        if layer_range is not None:
            self._layer_start, self._layer_end = layer_range
            assert 0 <= self._layer_start < self._layer_end <= total_layers
        else:
            self._layer_start, self._layer_end = 0, total_layers
        self._num_loaded_layers = self._layer_end - self._layer_start

        # Determine target device for quantized weights.
        self._streaming = target_device is not None
        if target_device is not None:
            self._target_device = target_device
        else:
            self._target_device = next(model.parameters()).device

        # Keep reference to original model for embeddings
        self.model = model
        embed = self.arch.get_nested_attr(model, self.arch.embed_path)
        if include_embed:
            self.embed_tokens = embed.to(self._target_device)
        else:
            self.embed_tokens = None

        lm_head = self.arch.get_nested_attr(model, self.arch.lm_head_path)
        self.lm_head_tied = (
            lm_head.weight.data_ptr() == embed.weight.data_ptr()
        )

        # Quantize and create LoRA adapters
        self._quantized_weights = nn.ParameterDict()
        self._lora_params = nn.ParameterDict()
        self._norm_weights = nn.ParameterDict()

        self._quantize_and_create_lora(model)

        # Set up weight streaming
        if self.weight_streaming:
            self._init_weight_streaming()

        # Freeze all base model parameters (any that remain)
        for p in model.parameters():
            p.requires_grad_(False)

        # Our LoRA params and norm weights are trainable
        for p in self._lora_params.parameters():
            p.requires_grad_(True)
        for p in self._norm_weights.parameters():
            p.requires_grad_(True)

    # ─── Quantization & LoRA creation ───

    def _quantize_weight(self, weight: torch.Tensor, name: str, k: int | None = None):
        """Quantize a weight matrix and store packed data."""
        if k is None:
            k = self.k
        weight = weight.to(self._target_device)
        N, K = weight.shape
        N_padded = ((N + 127) // 128) * 128
        if N_padded != N:
            w_padded = torch.nn.functional.pad(weight.float(), (0, 0, 0, N_padded - N))
        else:
            w_padded = weight.float()
        del weight

        packed, absmax, codebook = F.quantize_kbit(
            w_padded.reshape(-1),
            k=k,
            absmax_format="fp32",
        )
        del w_padded

        safe_name = name.replace(".", "_")
        self.register_buffer(f"_packed_{safe_name}", packed)
        self.register_buffer(f"_absmax_{safe_name}", absmax)
        self.register_buffer(f"_codebook_{safe_name}", codebook)

        return packed, absmax, codebook, N_padded, N, K

    def _create_lora(self, name: str, N: int, K: int):
        """Create LoRA A and B parameters for a weight matrix on _target_device."""
        safe_name = name.replace(".", "_")
        device = self._target_device
        A = nn.Parameter(torch.empty(self.lora_r, K, dtype=self.compute_dtype, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = nn.Parameter(torch.zeros(N, self.lora_r, dtype=self.compute_dtype, device=device))
        self._lora_params[f"{safe_name}_A"] = A
        self._lora_params[f"{safe_name}_B"] = B
        return A, B

    def _quantize_proj(self, module, proj_attr: str, name: str, k: int):
        """Quantize a single projection weight and create LoRA adapter."""
        weight = getattr(module, proj_attr).weight.data
        packed, absmax, codebook, N_padded, N, K = self._quantize_weight(weight, name, k=k)
        A, B = self._create_lora(name, N, K)
        return {
            "packed": packed, "absmax": absmax, "codebook": codebook,
            "N_padded": N_padded, "N": N, "K": K,
            "A": A, "B": B, "k": k,
        }

    def _quantize_attention(self, layer, layer_idx: int) -> dict:
        """Quantize attention projections for one layer."""
        attn = getattr(layer, self.arch.attn_module)
        prefix = f"layers_{layer_idx}"
        info = {}
        for generic, attr in [
            ("q_proj", self.arch.q_proj),
            ("k_proj", self.arch.k_proj),
            ("v_proj", self.arch.v_proj),
            ("o_proj", self.arch.o_proj),
        ]:
            info[generic] = self._quantize_proj(
                attn, attr, f"{prefix}_attn_{generic}", self.k_attention
            )
        return info

    def _quantize_dense_mlp(self, layer, layer_idx: int) -> dict:
        """Quantize dense MLP projections for one layer."""
        mlp = getattr(layer, self.arch.mlp_module)
        prefix = f"layers_{layer_idx}"
        info = {}
        for generic, attr in [
            ("gate_proj", self.arch.gate_proj),
            ("up_proj", self.arch.up_proj),
            ("down_proj", self.arch.down_proj),
        ]:
            info[generic] = self._quantize_proj(
                mlp, attr, f"{prefix}_mlp_{generic}", self.k_mlp
            )
        return info

    def _quantize_moe_layer(self, layer, layer_idx: int) -> dict:
        """Quantize MoE layer: router, shared expert (if any), routing experts."""
        prefix = f"layers_{layer_idx}"
        info = {"is_moe": True}

        # Router weight (NOT quantized — small, needs full precision)
        router = self.arch.get_nested_attr(layer, self.arch.moe_router_path)
        if hasattr(router, "weight"):
            router_weight = router.weight.data
        else:
            router_weight = router.data
        buf_name = f"_router_{prefix}"
        router_w = router_weight.to(self._target_device, dtype=self.compute_dtype)
        self.register_buffer(buf_name, router_w)
        info["router_weight"] = router_w

        # Shared expert (if present)
        if self.arch.has_shared_expert:
            shared = self.arch.get_nested_attr(layer, self.arch.shared_expert_path)
            for generic, attr in [
                ("shared_gate_proj", self.arch.gate_proj),
                ("shared_up_proj", self.arch.up_proj),
                ("shared_down_proj", self.arch.down_proj),
            ]:
                info[generic] = self._quantize_proj(
                    shared, attr, f"{prefix}_moe_{generic}", self.k_shared_expert
                )

        # Routing experts — quantize each expert and concatenate
        experts = self.arch.get_nested_attr(layer, self.arch.moe_experts_path)
        n_experts = self.arch.num_experts

        for proj_generic, proj_attr in [
            ("gate", self.arch.expert_gate_proj),
            ("up", self.arch.expert_up_proj),
            ("down", self.arch.expert_down_proj),
        ]:
            all_packed = []
            all_absmax = []
            codebook_ref = None
            meta = None  # N, K, N_padded from first expert

            for e_idx in range(n_experts):
                expert = experts[e_idx]
                weight = getattr(expert, proj_attr).weight.data.to(self._target_device)
                N, K = weight.shape
                N_padded = ((N + 127) // 128) * 128
                if N_padded != N:
                    w_padded = torch.nn.functional.pad(weight.float(), (0, 0, 0, N_padded - N))
                else:
                    w_padded = weight.float()
                del weight

                packed, absmax, codebook = F.quantize_kbit(
                    w_padded.reshape(-1), k=self.k_experts, absmax_format="fp32"
                )
                del w_padded

                all_packed.append(packed)
                all_absmax.append(absmax)
                if codebook_ref is None:
                    codebook_ref = codebook
                    meta = (N, K, N_padded)

            cat_packed = torch.cat(all_packed, dim=0)
            cat_absmax = torch.cat(all_absmax, dim=0)

            safe = f"{prefix}_moe_experts_{proj_generic}"
            self.register_buffer(f"_packed_{safe}", cat_packed)
            self.register_buffer(f"_absmax_{safe}", cat_absmax)
            if proj_generic == "gate":
                self.register_buffer(f"_codebook_{prefix}_moe_experts", codebook_ref)

            info[f"expert_{proj_generic}_packed"] = cat_packed
            info[f"expert_{proj_generic}_absmax"] = cat_absmax

        info["expert_codebook"] = getattr(self, f"_codebook_{prefix}_moe_experts")
        info["expert_k"] = self.k_experts
        N, K, N_padded = meta
        info["expert_N"] = N
        info["expert_K"] = K
        info["expert_N_padded"] = N_padded

        return info

    def _quantize_norms(self, layer, layer_idx: int) -> dict:
        """Extract and store norm weights for one layer."""
        device = self._target_device
        prefix = f"layers_{layer_idx}"
        info = {}

        for generic, attr in [
            ("input_layernorm", self.arch.input_norm),
            ("post_attention_layernorm", self.arch.post_attn_norm),
        ]:
            norm = getattr(layer, attr)
            safe = f"{prefix}_{generic}_weight"
            self._norm_weights[safe] = nn.Parameter(
                norm.weight.data.to(device=device, dtype=self.compute_dtype).clone()
            )
            info[generic] = self._norm_weights[safe]

        # QK norms (Qwen3)
        if self.arch.has_qk_norm:
            attn = getattr(layer, self.arch.attn_module)
            for generic, attr in [
                ("q_norm", self.arch.q_norm),
                ("k_norm", self.arch.k_norm),
            ]:
                norm = getattr(attn, attr)
                safe = f"{prefix}_attn_{generic}_weight"
                self._norm_weights[safe] = nn.Parameter(
                    norm.weight.data.to(device=device, dtype=self.compute_dtype).clone()
                )
                info[generic] = self._norm_weights[safe]

        return info

    def _quantize_and_create_lora(self, model: nn.Module):
        """Walk model, quantize weights, create LoRA adapters."""
        device = self._target_device
        layers = self.arch.get_nested_attr(model, self.arch.layers_path)
        self._layer_data = []

        for i in range(self._layer_start, self._layer_end):
            layer = layers[i]

            # Attention (always dense)
            layer_info = self._quantize_attention(layer, i)

            # MLP: dense or MoE
            if self.arch.is_moe_layer(i):
                moe_info = self._quantize_moe_layer(layer, i)
                layer_info.update(moe_info)
            else:
                mlp_info = self._quantize_dense_mlp(layer, i)
                layer_info.update(mlp_info)

            # Norms
            norm_info = self._quantize_norms(layer, i)
            layer_info.update(norm_info)

            self._layer_data.append(layer_info)

            # In streaming mode, free each layer from the source model
            if self._streaming:
                layers[i] = nn.Module()
                del layer
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Final norm
        if self.include_lm_head:
            final_norm = self.arch.get_nested_attr(model, self.arch.final_norm_path)
            self._norm_weights["final_norm_weight"] = nn.Parameter(
                final_norm.weight.data.to(device=device, dtype=self.compute_dtype).clone()
            )

        # LM head
        self._lm_head_info = None
        if self.include_lm_head:
            lm_head = self.arch.get_nested_attr(model, self.arch.lm_head_path)
            lm_weight = lm_head.weight.data
            name = "lm_head"
            packed, absmax, codebook, N_padded, N, K = self._quantize_weight(
                lm_weight, name, k=self.k_lm_head,
            )
            self._lm_head_info = {
                "packed": packed, "absmax": absmax, "codebook": codebook,
                "N_padded": N_padded, "N": N, "K": K, "k": self.k_lm_head,
            }

        # Precompute RoPE cos/sin cache
        self._build_rope_cache(device)

    # ─── RoPE ───

    def _build_rope_cache(self, device, max_seq_len: int = 8192):
        """Build rotary position embedding cos/sin cache."""
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos_cache = torch.cos(freqs).to(self.compute_dtype)
        sin_cache = torch.sin(freqs).to(self.compute_dtype)
        self.register_buffer("_cos_cache", cos_cache)
        self.register_buffer("_sin_cache", sin_cache)

    def _extend_rope_cache(self, seq_len: int, device):
        """Extend RoPE cache if needed for longer sequences."""
        if seq_len <= self._cos_cache.shape[0]:
            return
        self._build_rope_cache(device, max_seq_len=seq_len)

    # ─── Weight streaming ───

    def _get_streaming_weight_keys(self, layer_info: dict) -> list[str]:
        """Get the list of projection keys that have quantized weights for this layer."""
        if layer_info.get("is_moe"):
            keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if self.arch.has_shared_expert:
                keys += ["shared_gate_proj", "shared_up_proj", "shared_down_proj"]
            # Expert weights stored separately
            return keys
        else:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def _init_weight_streaming(self):
        """Move quantized weights to CPU pinned memory and pre-allocate GPU buffers."""
        device = self._target_device
        weight_keys = ["packed", "absmax", "codebook"]

        self._cpu_weights = []
        max_slot_bytes = 0

        for layer_info in self._layer_data:
            cpu_layer = {}
            layer_bytes = 0

            # Dense projections (attention + MLP/shared expert)
            proj_keys = self._get_streaming_weight_keys(layer_info)
            for proj in proj_keys:
                cpu_proj = {}
                for wk in weight_keys:
                    gpu_tensor = layer_info[proj][wk]
                    cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
                    cpu_tensor.copy_(gpu_tensor)
                    cpu_proj[wk] = cpu_tensor
                    layer_bytes += cpu_tensor.nbytes
                    layer_info[proj][wk] = None
                cpu_layer[proj] = cpu_proj

            # MoE expert weights (concatenated)
            if layer_info.get("is_moe"):
                for expert_proj in ["gate", "up", "down"]:
                    for suffix in ["packed", "absmax"]:
                        key = f"expert_{expert_proj}_{suffix}"
                        gpu_tensor = layer_info[key]
                        cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
                        cpu_tensor.copy_(gpu_tensor)
                        cpu_layer[key] = cpu_tensor
                        layer_bytes += cpu_tensor.nbytes
                        layer_info[key] = None
                # Codebook (shared across expert projections)
                cb = layer_info["expert_codebook"]
                cpu_cb = torch.empty_like(cb, device="cpu", pin_memory=True)
                cpu_cb.copy_(cb)
                cpu_layer["expert_codebook"] = cpu_cb
                layer_bytes += cpu_cb.nbytes
                layer_info["expert_codebook"] = None

            self._cpu_weights.append(cpu_layer)
            max_slot_bytes = max(max_slot_bytes, layer_bytes)

        # Free registered buffers
        buffers_to_remove = []
        for name, buf in self.named_buffers():
            if any(name.startswith(p) for p in ("_packed_", "_absmax_", "_codebook_", "_router_")):
                if "lm_head" in name:
                    continue
                buffers_to_remove.append(name)
        for name in buffers_to_remove:
            delattr(self, name)
        torch.cuda.empty_cache()

        # Pre-allocate 2 GPU buffer slots sized for the largest layer
        self._copy_stream = torch.cuda.Stream(device=device)
        self._gpu_slots = []
        for _ in range(2):
            slot = {}
            for i, cpu_layer in enumerate(self._cpu_weights):
                if i == 0:
                    for key, cpu_tensor in cpu_layer.items():
                        slot[key] = torch.empty_like(cpu_tensor, device=device)
                    break
            self._gpu_slots.append(slot)
        self._current_slot = 0

        total_cpu_bytes = sum(
            sum(t.nbytes for t in cl.values()) for cl in self._cpu_weights
        )
        slot_bytes = sum(t.nbytes for t in self._gpu_slots[0].values())
        print(
            f"Weight streaming: {total_cpu_bytes / 1e9:.1f} GB on CPU pinned, "
            f"{2 * slot_bytes / 1e6:.0f} MB GPU double-buffer "
            f"({len(self._cpu_weights)} layers)"
        )

    def _stream_load_layer(self, layer_idx: int, slot: int, sync: bool = False):
        """Copy a layer's quantized weights from CPU pinned to a GPU slot."""
        cpu_layer = self._cpu_weights[layer_idx]
        gpu_slot = self._gpu_slots[slot]

        if sync:
            for key, cpu_tensor in cpu_layer.items():
                if key not in gpu_slot:
                    gpu_slot[key] = torch.empty_like(cpu_tensor, device=self._target_device)
                gpu_slot[key].copy_(cpu_tensor)
        else:
            with torch.cuda.stream(self._copy_stream):
                for key, cpu_tensor in cpu_layer.items():
                    if key not in gpu_slot:
                        gpu_slot[key] = torch.empty_like(cpu_tensor, device=self._target_device)
                    gpu_slot[key].copy_(cpu_tensor, non_blocking=True)

    def _get_layer_gpu_weights(self, layer_idx: int, slot: int) -> dict:
        """Build a layer_info-compatible dict from GPU slot + always-resident data."""
        info = self._layer_data[layer_idx]
        gpu_slot = self._gpu_slots[slot]
        merged = {}

        # Dense projections
        proj_keys = self._get_streaming_weight_keys(info)
        for proj in proj_keys:
            merged[proj] = {
                "packed": gpu_slot[proj]["packed"],
                "absmax": gpu_slot[proj]["absmax"],
                "codebook": gpu_slot[proj]["codebook"],
                "A": info[proj]["A"],
                "B": info[proj]["B"],
                "N_padded": info[proj]["N_padded"],
                "N": info[proj]["N"],
                "K": info[proj]["K"],
                "k": info[proj]["k"],
            }

        # MoE expert weights
        if info.get("is_moe"):
            merged["is_moe"] = True
            merged["router_weight"] = info["router_weight"]
            for expert_proj in ["gate", "up", "down"]:
                for suffix in ["packed", "absmax"]:
                    key = f"expert_{expert_proj}_{suffix}"
                    merged[key] = gpu_slot[key]
            merged["expert_codebook"] = gpu_slot["expert_codebook"]
            merged["expert_k"] = info["expert_k"]
            merged["expert_N"] = info["expert_N"]
            merged["expert_K"] = info["expert_K"]
            merged["expert_N_padded"] = info["expert_N_padded"]

        # Norm weights (always on GPU)
        for key in ["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"]:
            if key in info:
                merged[key] = info[key]

        return merged

    # ─── Layer forward ───

    def _attention_forward(self, info: dict, hidden: torch.Tensor, position_ids: torch.Tensor, B: int, S: int, H: int):
        """Compute attention sub-block."""
        hidden_2d = hidden.reshape(-1, H)
        normed = rmsnorm(
            hidden_2d,
            info["input_layernorm"],
            eps=self.rms_norm_eps,
        ).reshape(B, S, H)
        normed_2d = normed.reshape(-1, H)

        # Q, K, V projections
        def _proj(proj_info, x):
            return LoRA_W_Kbit.apply(
                x,
                proj_info["packed"], proj_info["absmax"], proj_info["codebook"],
                proj_info["A"], proj_info["B"],
                self.lora_s, proj_info["k"], proj_info["K"],
                proj_info["N_padded"], proj_info["N"], self.compute_dtype,
            )

        Q = _proj(info["q_proj"], normed_2d)
        K_proj = _proj(info["k_proj"], normed_2d)
        V_proj = _proj(info["v_proj"], normed_2d)

        Q = Q.reshape(B * S, self.num_heads, self.head_dim)
        K_proj = K_proj.reshape(B * S, self.num_kv_heads, self.head_dim)
        V_proj = V_proj.reshape(B * S, self.num_kv_heads, self.head_dim)

        # QK norm
        if self.arch.has_qk_norm:
            Q = rmsnorm(Q.reshape(-1, self.head_dim), info["q_norm"], eps=self.rms_norm_eps)
            Q = Q.reshape(B * S, self.num_heads, self.head_dim)
            K_proj = rmsnorm(K_proj.reshape(-1, self.head_dim), info["k_norm"], eps=self.rms_norm_eps)
            K_proj = K_proj.reshape(B * S, self.num_kv_heads, self.head_dim)

        # RoPE
        positions = position_ids.reshape(-1)
        cos = self._cos_cache[positions]
        sin = self._sin_cache[positions]
        Q = rope(Q, cos, sin, self.num_heads)
        K_proj = rope(K_proj, cos, sin, self.num_kv_heads)

        # Flash attention
        Q = Q.reshape(B, S, self.num_heads, self.head_dim)
        K_proj = K_proj.reshape(B, S, self.num_kv_heads, self.head_dim)
        V_proj = V_proj.reshape(B, S, self.num_kv_heads, self.head_dim)

        attn_out = chunked_flash_attention(Q, K_proj, V_proj, chunk_size=self.attn_chunk_size, causal=True)
        attn_out = attn_out.reshape(B * S, self.q_dim)

        # Output projection
        attn_out = _proj(info["o_proj"], attn_out)
        return attn_out.reshape(B, S, H)

    def _dense_mlp_forward(self, info: dict, normed: torch.Tensor):
        """Compute dense MLP sub-block with chunked forward."""
        g = info["gate_proj"]
        u = info["up_proj"]
        d = info["down_proj"]
        return chunked_mlp_forward(
            normed, self.mlp_chunk_size,
            g["packed"], g["absmax"], g["codebook"], g["A"], g["B"], self.lora_s,
            u["packed"], u["absmax"], u["codebook"], u["A"], u["B"], self.lora_s,
            d["packed"], d["absmax"], d["codebook"], d["A"], d["B"], self.lora_s,
            g["k"],
            self.hidden_size, self.intermediate_size,
            ((self.intermediate_size + 127) // 128) * 128,
            self.intermediate_size, self.hidden_size,
            ((self.hidden_size + 127) // 128) * 128,
            self.compute_dtype, use_checkpoint=True,
        )

    def _moe_mlp_forward(self, info: dict, normed: torch.Tensor):
        """Compute MoE MLP sub-block: router dispatch + expert forward + shared expert."""
        # Router dispatch
        router_result = moe_router_dispatch(
            normed, info["router_weight"],
            num_experts=self.arch.num_experts,
            top_k=self.arch.num_active_experts,
        )

        # Expert forward (chunked)
        expert_out = moe_expert_forward(
            normed, router_result,
            info["expert_gate_packed"], info["expert_gate_absmax"],
            info["expert_up_packed"], info["expert_up_absmax"],
            info["expert_down_packed"], info["expert_down_absmax"],
            info["expert_codebook"],
            k=info["expert_k"],
            hidden_dim=self.hidden_size,
            intermediate_dim=self.arch.expert_intermediate_size,
            num_experts=self.arch.num_experts,
            expert_chunk_size=self.expert_chunk_size,
        )

        # Shared expert (if present)
        if self.arch.has_shared_expert:
            g = info["shared_gate_proj"]
            u = info["shared_up_proj"]
            d = info["shared_down_proj"]
            shared_inter = g["N"]  # shared expert intermediate size
            shared_out = chunked_mlp_forward(
                normed, self.mlp_chunk_size,
                g["packed"], g["absmax"], g["codebook"], g["A"], g["B"], self.lora_s,
                u["packed"], u["absmax"], u["codebook"], u["A"], u["B"], self.lora_s,
                d["packed"], d["absmax"], d["codebook"], d["A"], d["B"], self.lora_s,
                g["k"],
                self.hidden_size, shared_inter,
                ((shared_inter + 127) // 128) * 128,
                shared_inter, self.hidden_size,
                ((self.hidden_size + 127) // 128) * 128,
                self.compute_dtype, use_checkpoint=True,
            )
            return expert_out + shared_out
        else:
            return expert_out

    def _layer_forward(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """Forward pass for one decoder layer (dense or MoE)."""
        if self.weight_streaming:
            if torch.is_grad_enabled():
                self._stream_load_layer(layer_idx, 0, sync=True)
                info = self._get_layer_gpu_weights(layer_idx, 0)
            else:
                slot = layer_idx % 2
                info = self._get_layer_gpu_weights(layer_idx, slot)
        else:
            info = self._layer_data[layer_idx]

        B, S, H = hidden.shape

        # Attention
        residual = hidden
        attn_out = self._attention_forward(info, hidden, position_ids, B, S, H)
        hidden = residual + attn_out

        # MLP (dense or MoE)
        residual = hidden
        hidden_2d = hidden.reshape(-1, H)
        normed = rmsnorm(
            hidden_2d, info["post_attention_layernorm"], eps=self.rms_norm_eps,
        )

        if info.get("is_moe"):
            mlp_out = self._moe_mlp_forward(info, normed)
        else:
            mlp_out = self._dense_mlp_forward(info, normed)

        mlp_out = mlp_out.reshape(B, S, H)
        hidden = residual + mlp_out

        return hidden

    # ─── Streaming forward ───

    def _forward_streaming(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        """Double-buffered streaming forward pass."""
        n = self._num_loaded_layers

        self._current_slot = 0
        self._stream_load_layer(0, slot=0, sync=True)

        for i in range(n):
            next_slot = 1 - (i % 2)

            if i + 1 < n:
                self._stream_load_layer(i + 1, slot=next_slot, sync=False)

            def _make_stream_fn(layer_idx, pos_ids):
                def _fn(h):
                    return self._layer_forward(layer_idx, h, pos_ids)
                return _fn

            hidden = checkpoint_cpu_offload(
                _make_stream_fn(i, position_ids), hidden,
            )

            if i + 1 < n:
                torch.cuda.current_stream().wait_stream(self._copy_stream)

        return hidden

    # ─── Explicit backward for streaming ───

    def get_layer_lora_params(self, layer_idx: int) -> list[nn.Parameter]:
        """Get all LoRA parameters for a specific layer."""
        info = self._layer_data[layer_idx]
        params = []
        proj_keys = self._get_streaming_weight_keys(info)
        for proj in proj_keys:
            params.append(info[proj]["A"])
            params.append(info[proj]["B"])
        return params

    def forward_streaming_explicit(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Forward + backward with explicit per-layer autograd.grad() control.

        Returns loss value. Gradients are accumulated on LoRA params.
        """
        B, S = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        self._extend_rope_cache(S, device)

        # ─── FORWARD: save checkpoints at block boundaries ───
        if self.embed_tokens is not None:
            hidden = self.embed_tokens(input_ids).to(self.compute_dtype)
        else:
            hidden = input_ids

        n = self._num_loaded_layers
        checkpoints = []

        # Save input to first layer on CPU pinned
        ckpt = torch.empty(hidden.shape, dtype=hidden.dtype, device="cpu", pin_memory=True)
        ckpt.copy_(hidden, non_blocking=True)
        checkpoints.append(ckpt)

        # Pre-load layer 0
        self._stream_load_layer(0, slot=0, sync=True)

        for i in range(n):
            next_slot = 1 - (i % 2)
            if i + 1 < n:
                self._stream_load_layer(i + 1, slot=next_slot, sync=False)

            with torch.no_grad():
                hidden = self._layer_forward(i, hidden, position_ids)

            # Save checkpoint
            ckpt = torch.empty(hidden.shape, dtype=hidden.dtype, device="cpu", pin_memory=True)
            ckpt.copy_(hidden, non_blocking=True)
            checkpoints.append(ckpt)

            if i + 1 < n:
                torch.cuda.current_stream().wait_stream(self._copy_stream)

        # ─── LOSS (with grad) ───
        hidden_final = checkpoints[-1].to(device, non_blocking=True).requires_grad_(True)
        torch.cuda.current_stream().synchronize()

        hidden_2d = hidden_final.reshape(-1, self.hidden_size)
        hidden_2d = rmsnorm(
            hidden_2d, self._norm_weights["final_norm_weight"],
            eps=self.rms_norm_eps,
        )

        shift_hidden = hidden_2d[:-1]
        shift_labels = labels.reshape(-1)[1:]

        lm = self._lm_head_info
        loss = chunked_cross_entropy(
            shift_hidden,
            lm["packed"], lm["absmax"], lm["codebook"],
            shift_labels,
            lm["k"], lm["K"], lm["N_padded"], lm["N"],
            self.compute_dtype, self.ce_chunk_size,
        )

        # Also get grad for final norm weights
        norm_params = [self._norm_weights["final_norm_weight"]]
        all_grads = torch.autograd.grad(
            loss, [hidden_final] + norm_params,
            retain_graph=False,
        )
        grad = all_grads[0]
        for param, g in zip(norm_params, all_grads[1:]):
            if param.grad is None:
                param.grad = g.detach()
            else:
                param.grad.add_(g.detach())

        loss_val = loss.detach()

        # ─── BACKWARD: reverse layer order, double-buffered ───
        # Pre-load last layer
        last_slot = (n - 1) % 2
        self._stream_load_layer(n - 1, slot=last_slot, sync=True)

        for i in reversed(range(n)):
            cur_slot = i % 2
            next_bwd_slot = 1 - cur_slot

            # Prefetch next backward layer (i-1)
            if i > 0:
                self._stream_load_layer(i - 1, slot=next_bwd_slot, sync=False)

            # Restore checkpoint and recompute forward with grad
            input_act = checkpoints[i].to(device, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            input_act = input_act.requires_grad_(True)

            with torch.enable_grad():
                output = self._layer_forward(i, input_act, position_ids)

            # Get LoRA params + norm params for this layer
            lora_params = self.get_layer_lora_params(i)
            info = self._layer_data[i]
            layer_norm_params = []
            for nk in ["input_layernorm", "post_attention_layernorm"]:
                if nk in info:
                    layer_norm_params.append(info[nk])

            all_params = [input_act] + lora_params + layer_norm_params
            grads = torch.autograd.grad(
                output, all_params,
                grad_outputs=grad,
                retain_graph=False,
            )

            grad = grads[0]  # gradient w.r.t. input → pass to previous layer

            # Accumulate LoRA gradients
            for param, g in zip(lora_params, grads[1:1 + len(lora_params)]):
                if param.grad is None:
                    param.grad = g.detach()
                else:
                    param.grad.add_(g.detach())

            # Accumulate norm gradients
            for param, g in zip(layer_norm_params, grads[1 + len(lora_params):]):
                if param.grad is None:
                    param.grad = g.detach()
                else:
                    param.grad.add_(g.detach())

            # Wait for prefetch
            if i > 0:
                torch.cuda.current_stream().wait_stream(self._copy_stream)

        return loss_val

    # ─── Standard forward ───

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the full model."""
        B, S = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        self._extend_rope_cache(S, device)

        if self.embed_tokens is not None:
            hidden = self.embed_tokens(input_ids).to(self.compute_dtype)
        else:
            hidden = input_ids

        if self.weight_streaming and self.training:
            hidden = self._forward_streaming(hidden, position_ids)
        else:
            for i in range(self._num_loaded_layers):
                if self.cpu_offload and self.training:
                    def _make_layer_fn(layer_idx, pos_ids):
                        def _fn(h):
                            return self._layer_forward(layer_idx, h, pos_ids)
                        return _fn
                    hidden = checkpoint_cpu_offload(_make_layer_fn(i, position_ids), hidden)
                else:
                    hidden = self._layer_forward(i, hidden, position_ids)

        if not self.include_lm_head:
            return {"hidden": hidden}

        hidden_2d = hidden.reshape(-1, self.hidden_size)
        hidden_2d = rmsnorm(
            hidden_2d, self._norm_weights["final_norm_weight"],
            eps=self.rms_norm_eps,
        )

        result = {}

        if labels is not None:
            shift_hidden = hidden_2d[:-1]
            shift_labels = labels.reshape(-1)[1:]
            lm = self._lm_head_info
            loss = chunked_cross_entropy(
                shift_hidden,
                lm["packed"], lm["absmax"], lm["codebook"],
                shift_labels,
                lm["k"], lm["K"], lm["N_padded"], lm["N"],
                self.compute_dtype, self.ce_chunk_size,
            )
            result["loss"] = loss
        else:
            last_hidden = hidden_2d[-B:]
            lm = self._lm_head_info
            W_deq = F.dequantize_kbit(
                lm["packed"], lm["absmax"], lm["codebook"],
                lm["k"], lm["N_padded"] * lm["K"], self.compute_dtype,
            )
            W = W_deq[: lm["N_padded"] * lm["K"]].reshape(lm["N_padded"], lm["K"])[: lm["N"], :]
            logits = last_hidden @ W.t()
            result["logits"] = logits

        return result

    # ─── Parameter access ───

    def get_trainable_parameters(self):
        """Return only trainable parameters (LoRA adapters + norm weights)."""
        params = []
        for p in self._lora_params.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self._norm_weights.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def num_trainable_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def num_total_parameters(self):
        """Count all parameters (including quantized base model)."""
        total = sum(p.numel() for p in self.parameters())
        for buf in self.buffers():
            total += buf.numel()
        return total
