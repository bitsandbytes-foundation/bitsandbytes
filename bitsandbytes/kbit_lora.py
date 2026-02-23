"""KbitLoraModel: Model patcher for Llama/Mistral/Qwen families.

Replaces all linear layers with kbit-quantized weights + LoRA adapters,
patches attention with chunked Flash Attention, patches MLP with chunked
LoRA_MLP_Kbit, and patches norms with CUDA RMSNorm.

No PEFT dependency — manages LoRA adapters directly for efficiency.

Supported model_types: llama, mistral, qwen2, qwen3
"""

import math
from typing import Optional

import torch
import torch.nn as nn

import bitsandbytes.functional as F
from bitsandbytes.attention import chunked_flash_attention
from bitsandbytes.autograd.chunked_ce import chunked_cross_entropy
from bitsandbytes.autograd.lora_kbit import LoRA_MLP_Kbit, LoRA_W_Kbit
from bitsandbytes.autograd.training_kernels import rmsnorm, rope
from bitsandbytes.chunked import chunked_mlp_forward
from bitsandbytes.training import checkpoint_cpu_offload

SUPPORTED_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3"}


class KbitLoraModel(nn.Module):
    """Wraps a HuggingFace CausalLM model with kbit quantization + LoRA.

    Quantizes all linear weights (attention, MLP, LM head) to k-bit,
    adds trainable LoRA adapters, and patches forward methods to use
    our optimized CUDA kernels.

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
    ):
        super().__init__()

        config = model.config
        if config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported architecture: {config.model_type}. "
                f"Supported: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}"
            )

        self.config = config
        self.model_type = config.model_type
        self.lora_r = lora_r
        self.lora_s = lora_alpha / lora_r
        self.k = k
        self.k_config = k_config or {}
        self.k_attention = self.k_config.get("attention", k)
        self.k_mlp = self.k_config.get("mlp", k)
        self.k_lm_head = self.k_config.get("lm_head", k)
        self.attn_chunk_size = attn_chunk_size
        self.mlp_chunk_size = mlp_chunk_size
        self.ce_chunk_size = ce_chunk_size
        self.compute_dtype = compute_dtype
        self.cpu_offload = cpu_offload
        self.include_embed = include_embed
        self.include_lm_head = include_lm_head

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
        self.has_qk_norm = self.model_type == "qwen3"

        # Determine layer range
        total_layers = config.num_hidden_layers
        if layer_range is not None:
            self._layer_start, self._layer_end = layer_range
            assert 0 <= self._layer_start < self._layer_end <= total_layers
        else:
            self._layer_start, self._layer_end = 0, total_layers
        self._num_loaded_layers = self._layer_end - self._layer_start

        # Keep reference to original model for embeddings
        self.model = model
        if include_embed:
            self.embed_tokens = model.model.embed_tokens
        else:
            self.embed_tokens = None
        self.lm_head_tied = hasattr(model, "lm_head") and (
            model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        )

        # Quantize and create LoRA adapters
        self._quantized_weights = nn.ParameterDict()
        self._lora_params = nn.ParameterDict()
        self._norm_weights = nn.ParameterDict()

        self._quantize_and_create_lora(model)

        # Freeze all base model parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # Our LoRA params and norm weights are trainable
        for p in self._lora_params.parameters():
            p.requires_grad_(True)
        for p in self._norm_weights.parameters():
            p.requires_grad_(True)

    def _quantize_weight(self, weight: torch.Tensor, name: str, k: int | None = None):
        """Quantize a weight matrix and store packed data."""
        if k is None:
            k = self.k
        N, K = weight.shape
        N_padded = ((N + 127) // 128) * 128
        if N_padded != N:
            w_padded = torch.nn.functional.pad(weight.float(), (0, 0, 0, N_padded - N))
        else:
            w_padded = weight.float()

        packed, absmax, codebook = F.quantize_kbit(
            w_padded.reshape(-1), k=k, absmax_format="fp32",
        )

        # Store as non-trainable buffers
        safe_name = name.replace(".", "_")
        self.register_buffer(f"_packed_{safe_name}", packed)
        self.register_buffer(f"_absmax_{safe_name}", absmax)
        self.register_buffer(f"_codebook_{safe_name}", codebook)

        return packed, absmax, codebook, N_padded, N, K

    def _create_lora(self, name: str, N: int, K: int, device: torch.device):
        """Create LoRA A and B parameters for a weight matrix."""
        safe_name = name.replace(".", "_")
        # A: [r, K] initialized with Kaiming uniform
        A = nn.Parameter(torch.empty(self.lora_r, K, dtype=self.compute_dtype, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        # B: [N, r] initialized to zero (so LoRA contribution starts at zero)
        B = nn.Parameter(torch.zeros(N, self.lora_r, dtype=self.compute_dtype, device=device))
        self._lora_params[f"{safe_name}_A"] = A
        self._lora_params[f"{safe_name}_B"] = B
        return A, B

    def _quantize_and_create_lora(self, model: nn.Module):
        """Walk model, quantize weights, create LoRA adapters.

        Only processes layers in [_layer_start, _layer_end) and optionally
        skips embedding and LM head for pipeline parallelism.
        """
        device = next(model.parameters()).device

        # Process only the decoder layers in our range
        layers = model.model.layers
        self._layer_data = []

        for i in range(self._layer_start, self._layer_end):
            layer = layers[i]
            attn = layer.self_attn
            mlp = layer.mlp
            prefix = f"layers_{i}"

            layer_info = {}

            # Attention projections (use k_attention)
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight = getattr(attn, proj_name).weight.data.to(device)
                name = f"{prefix}_attn_{proj_name}"
                packed, absmax, codebook, N_padded, N, K = self._quantize_weight(
                    weight, name, k=self.k_attention,
                )
                A, B = self._create_lora(name, N, K, device)
                layer_info[proj_name] = {
                    "packed": packed, "absmax": absmax, "codebook": codebook,
                    "N_padded": N_padded, "N": N, "K": K, "A": A, "B": B,
                    "k": self.k_attention,
                }

            # MLP projections (use k_mlp)
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                weight = getattr(mlp, proj_name).weight.data.to(device)
                name = f"{prefix}_mlp_{proj_name}"
                packed, absmax, codebook, N_padded, N, K = self._quantize_weight(
                    weight, name, k=self.k_mlp,
                )
                A, B = self._create_lora(name, N, K, device)
                layer_info[proj_name] = {
                    "packed": packed, "absmax": absmax, "codebook": codebook,
                    "N_padded": N_padded, "N": N, "K": K, "A": A, "B": B,
                    "k": self.k_mlp,
                }

            # Norm weights (trainable, not quantized)
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                norm = getattr(layer, norm_name)
                safe = f"{prefix}_{norm_name}_weight"
                self._norm_weights[safe] = nn.Parameter(
                    norm.weight.data.to(self.compute_dtype).clone()
                )
                layer_info[norm_name] = self._norm_weights[safe]

            # QK norms (Qwen3 only)
            if self.has_qk_norm:
                for norm_name in ["q_norm", "k_norm"]:
                    norm = getattr(attn, norm_name)
                    safe = f"{prefix}_attn_{norm_name}_weight"
                    self._norm_weights[safe] = nn.Parameter(
                        norm.weight.data.to(self.compute_dtype).clone()
                    )
                    layer_info[norm_name] = self._norm_weights[safe]

            self._layer_data.append(layer_info)

        # Final norm (only needed by last stage or full model)
        if self.include_lm_head:
            final_norm = model.model.norm
            self._norm_weights["final_norm_weight"] = nn.Parameter(
                final_norm.weight.data.to(self.compute_dtype).clone()
            )

        # LM head (only needed by last stage or full model)
        self._lm_head_info = None
        if self.include_lm_head:
            lm_weight = model.lm_head.weight.data.to(device)
            name = "lm_head"
            packed, absmax, codebook, N_padded, N, K = self._quantize_weight(
                lm_weight, name, k=self.k_lm_head,
            )
            self._lm_head_info = {
                "packed": packed, "absmax": absmax, "codebook": codebook,
                "N_padded": N_padded, "N": N, "K": K,
                "k": self.k_lm_head,
            }

        # Precompute RoPE cos/sin cache
        self._build_rope_cache(device)

    def _build_rope_cache(self, device, max_seq_len: int = 8192):
        """Build rotary position embedding cos/sin cache."""
        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        cos_cache = torch.cos(freqs).to(self.compute_dtype)
        sin_cache = torch.sin(freqs).to(self.compute_dtype)
        self.register_buffer("_cos_cache", cos_cache)
        self.register_buffer("_sin_cache", sin_cache)

    def _extend_rope_cache(self, seq_len: int, device):
        """Extend RoPE cache if needed for longer sequences."""
        if seq_len <= self._cos_cache.shape[0]:
            return
        self._build_rope_cache(device, max_seq_len=seq_len)

    def _layer_forward(self, layer_idx: int, hidden: torch.Tensor, position_ids: torch.Tensor):
        """Forward pass for one decoder layer.

        Args:
            layer_idx: Local index (0-based within this model's loaded layers).
            hidden: Input hidden states [B, S, H].
            position_ids: Position IDs [B, S].

        Returns:
            Output hidden states [B, S, H].
        """
        info = self._layer_data[layer_idx]
        B, S, H = hidden.shape

        # --- Attention ---
        # Input layernorm
        residual = hidden
        hidden_2d = hidden.reshape(-1, H)
        normed = rmsnorm(
            hidden_2d, info["input_layernorm"], eps=self.rms_norm_eps,
        ).reshape(B, S, H)
        normed_2d = normed.reshape(-1, H)

        # Q, K, V projections (separate calls to handle GQA dims)
        q_info = info["q_proj"]
        Q = LoRA_W_Kbit.apply(
            normed_2d, q_info["packed"], q_info["absmax"], q_info["codebook"],
            q_info["A"], q_info["B"], self.lora_s,
            q_info["k"], q_info["K"], q_info["N_padded"], q_info["N"], self.compute_dtype,
        )  # [B*S, q_dim]

        k_info = info["k_proj"]
        K_proj = LoRA_W_Kbit.apply(
            normed_2d, k_info["packed"], k_info["absmax"], k_info["codebook"],
            k_info["A"], k_info["B"], self.lora_s,
            k_info["k"], k_info["K"], k_info["N_padded"], k_info["N"], self.compute_dtype,
        )  # [B*S, kv_dim]

        v_info = info["v_proj"]
        V_proj = LoRA_W_Kbit.apply(
            normed_2d, v_info["packed"], v_info["absmax"], v_info["codebook"],
            v_info["A"], v_info["B"], self.lora_s,
            v_info["k"], v_info["K"], v_info["N_padded"], v_info["N"], self.compute_dtype,
        )  # [B*S, kv_dim]

        # Reshape to [B*S, n_heads, head_dim] for RoPE
        Q = Q.reshape(B * S, self.num_heads, self.head_dim)
        K_proj = K_proj.reshape(B * S, self.num_kv_heads, self.head_dim)
        V_proj = V_proj.reshape(B * S, self.num_kv_heads, self.head_dim)

        # QK norm (Qwen3 only)
        if self.has_qk_norm:
            Q_2d = Q.reshape(-1, self.head_dim)
            Q_2d = rmsnorm(Q_2d, info["q_norm"], eps=self.rms_norm_eps)
            Q = Q_2d.reshape(B * S, self.num_heads, self.head_dim)

            K_2d = K_proj.reshape(-1, self.head_dim)
            K_2d = rmsnorm(K_2d, info["k_norm"], eps=self.rms_norm_eps)
            K_proj = K_2d.reshape(B * S, self.num_kv_heads, self.head_dim)

        # RoPE
        positions = position_ids.reshape(-1)  # [B*S]
        cos = self._cos_cache[positions]  # [B*S, head_dim/2]
        sin = self._sin_cache[positions]

        Q = rope(Q, cos, sin, self.num_heads)
        K_proj = rope(K_proj, cos, sin, self.num_kv_heads)

        # Reshape for flash attention: [B, S, H, D]
        Q = Q.reshape(B, S, self.num_heads, self.head_dim)
        K_proj = K_proj.reshape(B, S, self.num_kv_heads, self.head_dim)
        V_proj = V_proj.reshape(B, S, self.num_kv_heads, self.head_dim)

        # Chunked Flash Attention
        attn_out = chunked_flash_attention(
            Q, K_proj, V_proj,
            chunk_size=self.attn_chunk_size,
            causal=True,
        )  # [B, S, num_heads, head_dim]

        # Reshape back to [B*S, q_dim]
        attn_out = attn_out.reshape(B * S, self.q_dim)

        # Output projection
        o_info = info["o_proj"]
        attn_out = LoRA_W_Kbit.apply(
            attn_out, o_info["packed"], o_info["absmax"], o_info["codebook"],
            o_info["A"], o_info["B"], self.lora_s,
            o_info["k"], o_info["K"], o_info["N_padded"], o_info["N"], self.compute_dtype,
        )  # [B*S, hidden_size]
        attn_out = attn_out.reshape(B, S, H)

        # Residual connection
        hidden = residual + attn_out

        # --- MLP ---
        residual = hidden
        hidden_2d = hidden.reshape(-1, H)
        normed = rmsnorm(
            hidden_2d, info["post_attention_layernorm"], eps=self.rms_norm_eps,
        )

        # Chunked MLP with gradient checkpointing
        g = info["gate_proj"]
        u = info["up_proj"]
        d = info["down_proj"]
        mlp_out = chunked_mlp_forward(
            normed, self.mlp_chunk_size,
            g["packed"], g["absmax"], g["codebook"], g["A"], g["B"], self.lora_s,
            u["packed"], u["absmax"], u["codebook"], u["A"], u["B"], self.lora_s,
            d["packed"], d["absmax"], d["codebook"], d["A"], d["B"], self.lora_s,
            g["k"], self.hidden_size, self.intermediate_size,
            ((self.intermediate_size + 127) // 128) * 128,
            self.intermediate_size, self.hidden_size,
            ((self.hidden_size + 127) // 128) * 128,
            self.compute_dtype,
            use_checkpoint=True,
        )  # [B*S, hidden_size]

        mlp_out = mlp_out.reshape(B, S, H)
        hidden = residual + mlp_out

        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the full model.

        Args:
            input_ids: Input token IDs [B, S].
            labels: Target labels [B, S] for CE loss (shifted internally).
            position_ids: Position IDs [B, S]. Auto-generated if None.

        Returns:
            dict with 'loss' (if labels provided) and 'logits' (always None
            when using chunked CE to save memory).
        """
        B, S = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # Extend RoPE cache if needed
        self._extend_rope_cache(S, device)

        # Embedding (only if this model has the embedding layer)
        if self.embed_tokens is not None:
            hidden = self.embed_tokens(input_ids).to(self.compute_dtype)
        else:
            # input_ids is actually hidden states from previous pipeline stage
            hidden = input_ids

        # Decoder layers (local indices, 0-based)
        for i in range(self._num_loaded_layers):
            if self.cpu_offload and self.training:
                # Wrap each layer with CPU offload: saves inter-layer
                # activations to CPU during forward, reloads during backward
                def _make_layer_fn(layer_idx, pos_ids):
                    def _fn(h):
                        return self._layer_forward(layer_idx, h, pos_ids)
                    return _fn
                hidden = checkpoint_cpu_offload(_make_layer_fn(i, position_ids), hidden)
            else:
                hidden = self._layer_forward(i, hidden, position_ids)

        # Final norm + LM head (only if this model has the LM head)
        if not self.include_lm_head:
            return {"hidden": hidden}

        hidden_2d = hidden.reshape(-1, self.hidden_size)
        hidden_2d = rmsnorm(
            hidden_2d, self._norm_weights["final_norm_weight"], eps=self.rms_norm_eps,
        )

        result = {}

        if labels is not None:
            # Shift labels for next-token prediction
            shift_hidden = hidden_2d[:-1]  # Drop last position (B*S-1 tokens)
            shift_labels = labels.reshape(-1)[1:]  # Drop first label

            # Chunked cross-entropy (no logits materialization)
            lm = self._lm_head_info
            loss = chunked_cross_entropy(
                shift_hidden, lm["packed"], lm["absmax"], lm["codebook"],
                shift_labels,
                lm["k"], lm["K"], lm["N_padded"], lm["N"],
                self.compute_dtype, self.ce_chunk_size,
            )
            result["loss"] = loss
        else:
            # For inference: compute logits for the last position only
            last_hidden = hidden_2d[-B:]  # Last position per batch
            lm = self._lm_head_info
            W_deq = F.dequantize_kbit(
                lm["packed"], lm["absmax"], lm["codebook"],
                lm["k"], lm["N_padded"] * lm["K"], self.compute_dtype,
            )
            W = W_deq[:lm["N_padded"] * lm["K"]].reshape(lm["N_padded"], lm["K"])[:lm["N"], :]
            logits = last_hidden @ W.t()
            result["logits"] = logits

        return result

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
        # Add buffer sizes (quantized weights stored as buffers)
        for buf in self.buffers():
            total += buf.numel()
        return total
