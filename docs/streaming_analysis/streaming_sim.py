#!/usr/bin/env python3
"""
GLM-4.7 355B MoE — Realistic Streaming Simulation

Models the COMPLETE training step for QLoRA with weight streaming,
including all memory consumers, all compute phases, and all transfer
overheads. Finds the maximum batch size that fits, then computes
wall-clock step time and streaming overhead for each hardware config.

=== CONSIDERATIONS CHECKLIST ===

GPU MEMORY (what competes for VRAM):
  [x] Resident quantized weights (NF4/NF3/NF2 per layer)
  [x] Double buffer for streamed layers (2 × layer_size)
  [x] LoRA adapter weights (A and B matrices, bf16, per layer)
  [x] LoRA gradients (bf16, same size as LoRA weights)
  [x] LoRA optimizer states (AdamW: fp32 momentum + variance = 8 bytes/param)
  [x] Activation memory per layer (depends on B, S, H, grad checkpoint strategy)
  [x] Attention intermediate memory (Q, K, V, scores — flash attention reduces this)
  [x] MLP intermediate memory (gate, up projections before down)
  [x] Gradient checkpointing: only 1 layer's activations at a time
  [x] PyTorch CUDA context + allocator overhead
  [x] Temporary buffers: dequantized weight tiles during matmul

COMPUTE PHASES (what happens per training step):
  [x] Forward pass through all layers (resident + streamed)
  [x] Backward pass through all layers (reverse order)
  [x] Gradient checkpointing: recompute forward during backward
  [x] LoRA weight updates (optimizer step) — negligible time
  [x] Per-layer: attention (QKV proj, score, output proj) + MLP (gate, up, down)
  [x] Only ACTIVE expert params compute (8/160 for MoE)
  [x] Non-matmul ops: layernorm, softmax, activation fn, routing — ~10% overhead

TRANSFER PIPELINE (three-stage with overlap):
  [x] NVMe → CPU buffer (if weights not in CPU RAM)
  [x] CPU buffer → GPU buffer (PCIe DMA, per-GPU link)
  [x] NVMe bandwidth shared across GPUs in pipeline parallel
  [x] PCIe bandwidth is per-GPU (each has own x16 link)
  [x] Double buffering: load next layer while computing current
  [x] Forward pass: sequential layer order, can prefetch ahead
  [x] Backward pass: reverse order, need to reload layers again
  [x] With grad checkpoint: each streamed layer loaded TWICE per step
      (once for recompute-forward, once for gradient computation)

PIPELINE PARALLELISM:
  [x] G GPUs, each handles ceil(92/G) layers
  [x] M micro-batches to fill the pipeline
  [x] Pipeline bubble: (G-1) idle stages at start and end
  [x] Effective batch = M × micro_batch_size
  [x] Each GPU processes M forward + M backward passes per step
  [x] NVMe reads happen M × 2 times per step (forward + backward per micro-batch)
      Actually: with gradient checkpointing, streamed layers loaded 2x per micro-batch

BATCH SIZE CONSTRAINTS:
  [x] Must fit activations in free VRAM after weights + LoRA + optimizer + buffer
  [x] With gradient checkpointing: activation mem = O(B × S × H) per layer
  [x] With flash attention: attention mem = O(B × S) not O(B × S²)
  [x] MLP intermediate: B × S × intermediate_size × 2 bytes (bf16)
  [x] Input embeddings + final logits (usually small)
  [x] Micro-batch size for pipeline parallel may differ from total batch

WHAT WE COMPUTE:
  For each hardware config:
    1. Memory budget → max micro-batch size (B_max)
    2. Forward compute time per layer (matmul FLOPs / TFLOPS)
    3. Backward compute time per layer (~2× forward)
    4. Transfer time per streamed layer (layer_size / effective_bandwidth)
    5. Total step time with streaming overlap
    6. Overhead percentage vs no-streaming baseline
    7. Training throughput (tokens/sec)
"""

import math
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# =============================================================================
# MODEL DEFINITION
# =============================================================================

@dataclass
class MoEModel:
    """
    GLM-4.7 355B MoE architecture.

    Source: https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json

    Key facts:
    - hidden_size=5120, heads=96 (GQA: 96 Q heads, 8 KV heads, head_dim=128)
    - Note: Q dim = 96*128 = 12288, which is LARGER than hidden_size (5120)
    - shared expert intermediate = 12288 (config field: intermediate_size)
    - routing expert intermediate = 1536 (config field: moe_intermediate_size)
    - 160 routed experts, top-8 per token
    - first_k_dense_replace = 3: first 3 layers are dense (no MoE)
    - 89 MoE layers + 3 dense layers = 92 total

    Calculated:
    - MoE layer: ~4.10B params
    - Dense layer: ~0.325B params
    - Active params per MoE layer: ~515M
    - Total: ~367B (model card says ~355B; difference likely counting convention)
    """
    name: str = "GLM-4.7-355B"
    n_layers: int = 92
    n_dense_layers: int = 3         # first_k_dense_replace: no MoE in first 3 layers
    hidden_size: int = 5120
    num_attention_heads: int = 96
    num_kv_heads: int = 8            # GQA
    head_dim: int = 128              # Q/K/V head dimension
    # The shared expert acts like a dense FFN
    shared_intermediate_size: int = 12288  # config: intermediate_size
    # Each routing expert is small (160 of them)
    expert_intermediate_size: int = 1536   # config: moe_intermediate_size
    num_experts: int = 160                 # config: n_routed_experts
    num_active_experts: int = 8            # config: num_experts_per_tok
    has_shared_expert: bool = True         # config: n_shared_experts = 1

    @property
    def attention_params(self) -> int:
        h, nh, kv, d = self.hidden_size, self.num_attention_heads, self.num_kv_heads, self.head_dim
        # Q: h → nh*d,  K: h → kv*d,  V: h → kv*d,  O: nh*d → h
        return h * nh * d + h * kv * d + h * kv * d + nh * d * h

    @property
    def shared_expert_params(self) -> int:
        # gate + up + down: each h × shared_inter
        return 3 * self.hidden_size * self.shared_intermediate_size

    @property
    def per_routing_expert_params(self) -> int:
        return 3 * self.hidden_size * self.expert_intermediate_size

    @property
    def total_params_per_layer(self) -> float:
        router = self.hidden_size * self.num_experts
        return (self.attention_params
                + self.shared_expert_params
                + self.num_experts * self.per_routing_expert_params
                + router)

    @property
    def active_params_per_layer(self) -> float:
        router = self.hidden_size * self.num_experts
        return (self.attention_params
                + self.shared_expert_params
                + self.num_active_experts * self.per_routing_expert_params
                + router)

    @property
    def expert_fraction(self) -> float:
        expert_params = self.num_experts * self.per_routing_expert_params
        return expert_params / self.total_params_per_layer

    @property
    def active_mlp_intermediate_total(self) -> int:
        """Total intermediate dimension across all active MLP paths."""
        return self.shared_intermediate_size + self.num_active_experts * self.expert_intermediate_size


# =============================================================================
# QUANTIZATION
# =============================================================================

@dataclass
class QuantConfig:
    """
    Quantization configuration.

    Uses EMPIRICAL layer sizes (validated against actual quantized model files)
    rather than deriving from architecture params, since the exact overhead
    from absmax scales, codebook entries, and block structure varies.

    compute_speedup: effective layer-level speedup from hardware-accelerated
    quantized matmul (e.g., NVFP4 on Blackwell tensor cores). This accounts
    for the fact that only weight matmuls benefit — attention QK^T/score×V
    remains BF16. For GLM-4.7 at S=1024:
      - Weight matmuls = 95.3% of FLOPs → benefit from FP4 TCs
      - Attention compute = 4.7% → always BF16
      - At 3x raw kernel speedup: full NVFP4 effective = 2.74x
    """
    name: str
    layer_mb_empirical: float   # measured/validated layer size in MB
    compute_speedup: float = 1.0  # effective layer-level speedup (1.0 = no speedup)

    def layer_bytes(self, model: MoEModel) -> float:
        return self.layer_mb_empirical * (1024 ** 2)

    def layer_mb(self, model: MoEModel) -> float:
        return self.layer_mb_empirical

    def layer_gb(self, model: MoEModel) -> float:
        return self.layer_mb_empirical / 1024

    def total_gb(self, model: MoEModel) -> float:
        return self.layer_gb(model) * model.n_layers


# Empirical layer sizes from previous validated analysis.
# NF4d+NF3e computed from cross-check: dense@NF4(4.60bpp) + expert@NF3(3.36bpp).
# NVFP4: MXFP4 format (E2M1 + FP8 microscaling per 32 elements ≈ 4.25 bpp).
#   Effective speedup from Blackwell FP4 tensor cores (benchmarked at ~3x raw vs BF16):
#   Full NVFP4 = 2.74x layer-level (95% of FLOPs are weight matmuls).
#   NVFP4 ONLY valid on GPUs with FP4 tensor cores (Blackwell: RTX 5090, B100, B200).
QUANT_CONFIGS = {
    "NF4":       QuantConfig("NF4",       layer_mb_empirical=2250),
    "NF3":       QuantConfig("NF3",       layer_mb_empirical=1640),
    "NF2":       QuantConfig("NF2",       layer_mb_empirical=1150),
    "NF4d+NF2e": QuantConfig("NF4d+NF2e", layer_mb_empirical=1237),
    "NF4d+NF3e": QuantConfig("NF4d+NF3e", layer_mb_empirical=1690),
    "NVFP4":     QuantConfig("NVFP4",     layer_mb_empirical=2100, compute_speedup=2.74),
}


# =============================================================================
# LORA CONFIG
# =============================================================================

@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 64
    # Which projections get LoRA
    # Attention: Q, K, V, O = 4 projections
    # Shared expert MLP: gate, up, down = 3 projections
    # Routing experts: NOT adapted (too many, would explode param count)
    n_projections: int = 7  # 4 attn + 3 shared expert

    def params_per_layer(self, model: MoEModel) -> int:
        """LoRA parameters per layer (A + B matrices)."""
        h = model.hidden_size
        nh, kv, d = model.num_attention_heads, model.num_kv_heads, model.head_dim
        shared_inter = model.shared_intermediate_size

        # Attention: Q [h→nh*d], K [h→kv*d], V [h→kv*d], O [nh*d→h]
        # Each gets A [r, dim_in] + B [dim_out, r]
        q_params = self.rank * h + nh * d * self.rank
        k_params = self.rank * h + kv * d * self.rank
        v_params = self.rank * h + kv * d * self.rank
        o_params = self.rank * nh * d + h * self.rank
        attn_params = q_params + k_params + v_params + o_params

        # Shared expert MLP: gate [h→inter], up [h→inter], down [inter→h]
        gate_params = self.rank * h + shared_inter * self.rank
        up_params = self.rank * h + shared_inter * self.rank
        down_params = self.rank * shared_inter + h * self.rank
        mlp_params = gate_params + up_params + down_params

        # Routing experts: NOT adapted (too many, would explode param count)
        return attn_params + mlp_params

    def params_total(self, model: MoEModel) -> int:
        return self.params_per_layer(model) * model.n_layers

    def weight_bytes_per_layer(self, model: MoEModel) -> float:
        """LoRA weights in bf16."""
        return self.params_per_layer(model) * 2  # bf16

    def grad_bytes_per_layer(self, model: MoEModel) -> float:
        """Gradients in bf16."""
        return self.params_per_layer(model) * 2

    def optimizer_bytes_per_layer(self, model: MoEModel) -> float:
        """AdamW: fp32 copy + momentum + variance = 12 bytes/param."""
        # Actually: master weight (fp32) + momentum (fp32) + variance (fp32) = 12
        # But if we do bf16 training with fp32 optimizer, it's:
        # param (bf16) + grad (bf16) + master (fp32) + momentum (fp32) + variance (fp32)
        # = 2 + 2 + 4 + 4 + 4 = 16 bytes/param
        # The param and grad are already counted separately
        # Optimizer states only: master_weight(fp32) + momentum(fp32) + variance(fp32) = 12
        return self.params_per_layer(model) * 12

    def total_gpu_bytes_per_layer(self, model: MoEModel) -> float:
        """Total LoRA-related GPU memory per layer."""
        return (self.weight_bytes_per_layer(model)
                + self.grad_bytes_per_layer(model)
                + self.optimizer_bytes_per_layer(model))


# =============================================================================
# GPU HARDWARE
# =============================================================================

@dataclass
class GPU:
    """GPU hardware specification."""
    name: str
    vram_gb: float
    pcie_bw_gbs: float    # effective PCIe bandwidth (GB/s)
    bf16_tflops: float    # dense BF16 tensor core TFLOPS
    pcie_gen: int = 4

    @property
    def peak_flops(self) -> float:
        return self.bf16_tflops * 1e12


GPUS = {
    "RTX 4090":  GPU("RTX 4090",  vram_gb=24,  pcie_bw_gbs=22, bf16_tflops=165, pcie_gen=4),
    "RTX 5090":  GPU("RTX 5090",  vram_gb=32,  pcie_bw_gbs=44, bf16_tflops=209, pcie_gen=5),
    "A100 80G":  GPU("A100 80G",  vram_gb=80,  pcie_bw_gbs=22, bf16_tflops=312, pcie_gen=4),
    # H100 PCIe: BF16 TC dense = 756 TFLOPS. SXM5: 990 TFLOPS.
    # (495 was TF32 dense SXM5, not BF16)
    "H100 80G":  GPU("H100 80G",  vram_gb=80,  pcie_bw_gbs=44, bf16_tflops=756, pcie_gen=5),
    "RTX6000P":  GPU("RTX6000P",  vram_gb=96,  pcie_bw_gbs=44, bf16_tflops=300, pcie_gen=5),
}


# =============================================================================
# STORAGE
# =============================================================================

@dataclass
class StorageConfig:
    """NVMe + CPU RAM configuration."""
    name: str
    nvme_bw_gbs: float    # NVMe sequential read bandwidth
    cpu_ram_gb: float      # total system RAM
    cpu_pinned_gb: float   # available for pinned memory (after OS, PyTorch)

    @property
    def description(self) -> str:
        return f"{self.name}, {self.cpu_ram_gb:.0f}G RAM"


STORAGE_CONFIGS = {
    "Gen4x1_32G":     StorageConfig("Gen4x1",     nvme_bw_gbs=7,  cpu_ram_gb=32, cpu_pinned_gb=26),
    "Gen4x1_64G":     StorageConfig("Gen4x1",     nvme_bw_gbs=7,  cpu_ram_gb=64, cpu_pinned_gb=56),
    "Gen4R0x4_32G":   StorageConfig("Gen4 R0x4",  nvme_bw_gbs=28, cpu_ram_gb=32, cpu_pinned_gb=26),
    "Gen5AICx4_32G":  StorageConfig("Gen5 AICx4", nvme_bw_gbs=48, cpu_ram_gb=32, cpu_pinned_gb=26),
    "Gen5AICx4_64G":  StorageConfig("Gen5 AICx4", nvme_bw_gbs=48, cpu_ram_gb=64, cpu_pinned_gb=56),
    "Gen4R0x4_64G":   StorageConfig("Gen4 R0x4",  nvme_bw_gbs=28, cpu_ram_gb=64, cpu_pinned_gb=56),
    "Gen4R0x4_128G":  StorageConfig("Gen4 R0x4",  nvme_bw_gbs=28, cpu_ram_gb=128, cpu_pinned_gb=120),
}


# =============================================================================
# ACTIVATION MEMORY MODEL
# =============================================================================

def activation_memory_per_layer_bytes(
    model: MoEModel,
    batch_size: int,
    seq_len: int,
    grad_checkpoint: bool = True,
    flash_attention: bool = True,
) -> float:
    """
    Estimate peak activation memory for one transformer layer during training.

    With gradient checkpointing: only 1 layer's activations at a time.
    With flash attention: no S×S attention matrix materialized.

    Peak is during backward (with recompute) for the MLP block:
    - Input checkpoint: B × S × H × 2
    - Attention output (input to MLP): B × S × H × 2
    - Shared expert intermediates: 2 × B × S × shared_inter × 2 (gate + up for SwiGLU)
    - Routing expert intermediates: 2 × B × S × k × expert_inter × 2
      (each token routed to k=8 experts, gate + up for SwiGLU)
    - Router logits: B × S × n_experts × 4 (fp32)
    - Gradient: B × S × H × 2

    Note: attention backward peak is usually smaller than MLP backward for MoE
    because MLP has many active expert intermediates.
    """
    B, S = batch_size, seq_len
    H = model.hidden_size
    nh = model.num_attention_heads
    kv = model.num_kv_heads
    d = model.head_dim
    shared_inter = model.shared_intermediate_size
    expert_inter = model.expert_intermediate_size
    k = model.num_active_experts

    bpe = 2  # bytes per element (bf16)

    # Input activation checkpoint
    input_act = B * S * H * bpe

    # Attention intermediates
    qkv = B * S * (nh * d + 2 * kv * d) * bpe
    attn_out = B * S * H * bpe
    if flash_attention:
        attn_ws = B * nh * S * 4  # O(B × nh × S) workspace
    else:
        attn_ws = B * nh * S * S * bpe

    # MLP intermediates (peak during SwiGLU: gate and up are both live)
    # Shared expert: gate(B×S×shared_inter) + up(B×S×shared_inter)
    shared_mlp = 2 * B * S * shared_inter * bpe
    # Routing experts: each token goes to k experts
    # gate(B×S×expert_inter) + up(B×S×expert_inter) per active expert
    routing_mlp = k * 2 * B * S * expert_inter * bpe
    # Router logits (fp32 for numerical stability)
    router = B * S * model.num_experts * 4

    # Gradient tensor
    grad_out = B * S * H * bpe

    # Peak during attention backward
    peak_attn = input_act + qkv + attn_ws + attn_out + grad_out

    # Peak during MLP backward (usually higher for MoE)
    peak_mlp = input_act + attn_out + shared_mlp + routing_mlp + router + grad_out

    # Add 20% for PyTorch allocator fragmentation
    peak = max(peak_attn, peak_mlp) * 1.2

    return peak


# =============================================================================
# COMPUTE TIME MODEL
# =============================================================================

def layer_forward_flops(model: MoEModel, batch_size: int, seq_len: int) -> float:
    """FLOPs for one forward pass through one layer."""
    B, S = batch_size, seq_len
    tokens = B * S
    H = model.hidden_size
    nh = model.num_attention_heads
    kv = model.num_kv_heads
    d = model.head_dim
    shared_inter = model.shared_intermediate_size
    expert_inter = model.expert_intermediate_size

    # Attention projections: Q, K, V, O
    # Q: tokens × [H → nh*d] = 2 × tokens × H × nh*d
    # K: tokens × [H → kv*d]
    # V: tokens × [H → kv*d]
    # O: tokens × [nh*d → H]
    attn_proj = 2 * tokens * H * (nh * d + 2 * kv * d + nh * d)

    # Attention scores: Q @ K^T then @ V
    # Score: B × nh × S × d @ B × nh × d × S = 2 × B × nh × S × S × d
    # Context: B × nh × S × S @ B × nh × S × d = 2 × B × nh × S × S × d
    attn_compute = 2 * 2 * B * nh * S * S * d

    # MLP for active experts:
    # Each expert: gate(H→inter) + up(H→inter) + down(inter→H)
    # = 3 matmuls × 2 × tokens × H × inter
    # Shared expert (large intermediate)
    shared_mlp_flops = 3 * 2 * tokens * H * shared_inter
    # Routing experts (small intermediate each, k active)
    routing_mlp_flops = model.num_active_experts * (3 * 2 * tokens * H * expert_inter)
    mlp_flops = shared_mlp_flops + routing_mlp_flops

    # Router: tokens × H × num_experts
    router_flops = 2 * tokens * H * model.num_experts

    # LayerNorm, softmax, activation fn: ~5% of matmul FLOPs
    non_matmul_factor = 1.05

    return (attn_proj + attn_compute + mlp_flops + router_flops) * non_matmul_factor


def layer_backward_flops(model: MoEModel, batch_size: int, seq_len: int) -> float:
    """FLOPs for backward pass. ~2× forward (grad w.r.t. input + grad w.r.t. LoRA weights)."""
    return 2 * layer_forward_flops(model, batch_size, seq_len)


def compute_time_seconds(
    flops: float, gpu: GPU, utilization: float = 0.70, compute_speedup: float = 1.0,
) -> float:
    """
    Wall-clock time for given FLOPs on given GPU.

    utilization: fraction of peak TFLOPS actually achieved.

    Benchmarked on RTX 4090 with NF4 dequant + BF16 matmul (CUDA 12.8, PyTorch 2.9):
      B=1:  81% of 165 TFLOPS peak (NF4 dequant adds 20% over pure BF16)
      B=4:  94% (dequant adds 4%)
      B=8:  96% (dequant adds 3%)
      B=16: 97% (dequant adds 2%)

    These are isolated matmul numbers. End-to-end training adds:
      - Non-matmul ops (layernorm, softmax, activation fn): ~5%
      - Training loop / scheduling overhead: ~5-10%
      - Gradient checkpoint recomputation scheduling: ~5%

    Conservative end-to-end estimate: 70% (vs measured 81-97% for isolated matmuls).

    compute_speedup: effective layer-level speedup from hardware-accelerated
    formats (e.g., NVFP4 on Blackwell FP4 tensor cores = 2.74x).
    """
    return flops / (gpu.peak_flops * utilization * compute_speedup)


# =============================================================================
# MEMORY BUDGET AND MAX BATCH SIZE
# =============================================================================

@dataclass
class MemoryBudget:
    """Complete GPU memory breakdown."""
    gpu_vram_gb: float
    resident_weight_gb: float
    stream_buffer_gb: float
    lora_weight_gb: float
    lora_grad_gb: float
    lora_optimizer_gb: float
    cuda_overhead_gb: float = 2.5
    # Computed
    free_for_activations_gb: float = 0.0
    n_resident: int = 0
    n_streamed: int = 0
    layers_per_gpu: int = 0

    @property
    def total_fixed_gb(self) -> float:
        return (self.resident_weight_gb + self.stream_buffer_gb
                + self.lora_weight_gb + self.lora_grad_gb
                + self.lora_optimizer_gb + self.cuda_overhead_gb)


def compute_memory_budget(
    gpu: GPU,
    n_gpus: int,
    model: MoEModel,
    quant: QuantConfig,
    lora: LoRAConfig,
    n_resident_override: Optional[int] = None,
) -> Optional[MemoryBudget]:
    """
    Compute the full memory breakdown.

    If n_resident_override is None, maximize resident layers (greedy).
    If specified, use exactly that many resident layers (for optimization sweep).
    """
    layers_per_gpu = math.ceil(model.n_layers / n_gpus)
    layer_gb = quant.layer_gb(model)

    # LoRA memory (all layers on this GPU need LoRA regardless of streaming)
    lora_w_gb = lora.weight_bytes_per_layer(model) * layers_per_gpu / (1024**3)
    lora_g_gb = lora.grad_bytes_per_layer(model) * layers_per_gpu / (1024**3)
    lora_o_gb = lora.optimizer_bytes_per_layer(model) * layers_per_gpu / (1024**3)

    cuda_overhead = 2.5  # GB

    if n_resident_override is not None:
        n_resident = min(n_resident_override, layers_per_gpu)
        n_streamed = layers_per_gpu - n_resident
        buffer_gb = 2 * layer_gb if n_streamed > 0 else 0
        resident_gb = n_resident * layer_gb
        free = gpu.vram_gb - (resident_gb + buffer_gb + lora_w_gb + lora_g_gb
                              + lora_o_gb + cuda_overhead)
        if free < 0:
            return None
        return MemoryBudget(
            gpu_vram_gb=gpu.vram_gb,
            resident_weight_gb=resident_gb,
            stream_buffer_gb=buffer_gb,
            lora_weight_gb=lora_w_gb,
            lora_grad_gb=lora_g_gb,
            lora_optimizer_gb=lora_o_gb,
            cuda_overhead_gb=cuda_overhead,
            free_for_activations_gb=max(free, 0),
            n_resident=n_resident,
            n_streamed=n_streamed,
            layers_per_gpu=layers_per_gpu,
        )

    # Default: try all-on-GPU first
    all_weight = layers_per_gpu * layer_gb
    total_no_stream = all_weight + lora_w_gb + lora_g_gb + lora_o_gb + cuda_overhead
    if total_no_stream < gpu.vram_gb:
        free = gpu.vram_gb - total_no_stream
        return MemoryBudget(
            gpu_vram_gb=gpu.vram_gb,
            resident_weight_gb=all_weight,
            stream_buffer_gb=0,
            lora_weight_gb=lora_w_gb,
            lora_grad_gb=lora_g_gb,
            lora_optimizer_gb=lora_o_gb,
            cuda_overhead_gb=cuda_overhead,
            free_for_activations_gb=free,
            n_resident=layers_per_gpu,
            n_streamed=0,
            layers_per_gpu=layers_per_gpu,
        )

    # Streaming: need double buffer
    buffer_gb = 2 * layer_gb
    fixed = cuda_overhead + lora_w_gb + lora_g_gb + lora_o_gb + buffer_gb
    avail_for_weights = gpu.vram_gb - fixed
    if avail_for_weights <= 0:
        return None

    n_resident = min(int(avail_for_weights / layer_gb), layers_per_gpu)
    n_streamed = layers_per_gpu - n_resident

    resident_gb = n_resident * layer_gb
    free = gpu.vram_gb - (resident_gb + buffer_gb + lora_w_gb + lora_g_gb + lora_o_gb + cuda_overhead)

    if free < 0.2:
        n_resident = max(0, n_resident - 1)
        n_streamed = layers_per_gpu - n_resident
        resident_gb = n_resident * layer_gb
        free = gpu.vram_gb - (resident_gb + buffer_gb + lora_w_gb + lora_g_gb + lora_o_gb + cuda_overhead)

    return MemoryBudget(
        gpu_vram_gb=gpu.vram_gb,
        resident_weight_gb=resident_gb,
        stream_buffer_gb=buffer_gb,
        lora_weight_gb=lora_w_gb,
        lora_grad_gb=lora_g_gb,
        lora_optimizer_gb=lora_o_gb,
        cuda_overhead_gb=cuda_overhead,
        free_for_activations_gb=max(free, 0),
        n_resident=n_resident,
        n_streamed=n_streamed,
        layers_per_gpu=layers_per_gpu,
    )


def find_max_batch_size(
    model: MoEModel,
    mem: MemoryBudget,
    seq_len: int,
) -> int:
    """Find maximum micro-batch size that fits in free VRAM."""
    free_bytes = mem.free_for_activations_gb * (1024**3)
    # Binary search
    lo, hi = 1, 256
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        act_bytes = activation_memory_per_layer_bytes(model, mid, seq_len)
        # Add a 20% fragmentation overhead for PyTorch allocator
        act_bytes_with_frag = act_bytes * 1.2
        if act_bytes_with_frag <= free_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# =============================================================================
# STREAMING SIMULATION
# =============================================================================

@dataclass
class StepSimulation:
    """Result of simulating one complete training step."""
    # Config
    gpu_name: str
    n_gpus: int
    quant_name: str
    storage_name: str
    seq_len: int
    # Memory
    max_micro_batch: int
    free_vram_gb: float
    n_resident: int
    n_streamed: int
    layers_per_gpu: int
    # Compute
    forward_time_s: float      # total forward pass time (all layers, this GPU)
    backward_time_s: float     # total backward pass (includes recompute)
    compute_time_s: float      # forward + backward
    # Transfer
    transfer_source: str       # "RAM" or "NVMe"
    effective_bw_gbs: float    # bottleneck bandwidth
    bottleneck: str            # "PCIe" or "NVMe"
    # Per-layer transfer time (for one streamed layer)
    layer_transfer_time_s: float
    # Total transfer demand: each streamed layer loaded twice (fwd recompute + bwd)
    total_transfer_demand_s: float
    # Overlap
    compute_time_per_step_s: float  # total compute for all micro-batches
    transfer_time_per_step_s: float # total transfer needed
    step_time_s: float              # max(compute, transfer) — with overlap
    overhead_pct: float             # (step_time / compute_time - 1) × 100
    # Throughput
    tokens_per_step: int
    tokens_per_sec: float
    # Pipeline
    n_micro_batches: int
    pipeline_bubble_frac: float


def simulate_step(
    model: MoEModel,
    gpu: GPU,
    n_gpus: int,
    quant: QuantConfig,
    lora: LoRAConfig,
    storage: StorageConfig,
    seq_len: int = 1024,
    n_micro_batches: int = 1,
    gpu_utilization: float = 0.5,
    n_resident_override: Optional[int] = None,
) -> Optional[StepSimulation]:
    """Simulate a complete training step."""

    mem = compute_memory_budget(gpu, n_gpus, model, quant, lora,
                                n_resident_override=n_resident_override)
    if mem is None:
        return None

    # Find max micro-batch size
    max_mb = find_max_batch_size(model, mem, seq_len)
    if max_mb < 1:
        return None

    B = max_mb  # micro-batch size
    S = seq_len
    lpg = mem.layers_per_gpu
    layer_gb = quant.layer_gb(model)

    # --- Compute times ---
    fwd_flops_per_layer = layer_forward_flops(model, B, S)
    bwd_flops_per_layer = layer_backward_flops(model, B, S)

    # With gradient checkpointing: backward includes recompute of forward
    # So total per layer = forward + (forward_recompute + backward) = 1 fwd + 1 fwd + 2 fwd = 4 fwd
    # Actually: fwd(1x) + bwd(2x fwd) + recompute_fwd(1x) = 4x fwd per layer
    recompute_flops = fwd_flops_per_layer  # recompute during backward

    cs = quant.compute_speedup  # hardware-accelerated format speedup (1.0 for NF4, 2.74 for NVFP4)
    fwd_time = sum(
        compute_time_seconds(fwd_flops_per_layer, gpu, gpu_utilization, cs)
        for _ in range(lpg)
    )
    bwd_time = sum(
        compute_time_seconds(bwd_flops_per_layer + recompute_flops, gpu, gpu_utilization, cs)
        for _ in range(lpg)
    )
    compute_per_microbatch = fwd_time + bwd_time

    # --- Transfer times ---
    n_str = mem.n_streamed

    if n_str == 0:
        # All on GPU — no streaming
        transfer_source = "GPU"
        effective_bw = float('inf')
        bneck = "—"
        layer_xfer = 0
        total_xfer = 0
    else:
        # Determine source: CPU RAM or NVMe
        total_streamed_gb = n_str * layer_gb * n_gpus
        fits_in_ram = total_streamed_gb <= storage.cpu_pinned_gb

        if fits_in_ram:
            transfer_source = "RAM"
            effective_bw = gpu.pcie_bw_gbs  # PCIe only
            bneck = "PCIe"
        else:
            transfer_source = "NVMe"
            nvme_per_gpu = storage.nvme_bw_gbs / n_gpus
            if nvme_per_gpu >= gpu.pcie_bw_gbs:
                effective_bw = gpu.pcie_bw_gbs
                bneck = "PCIe"
            else:
                effective_bw = nvme_per_gpu
                bneck = "NVMe"

        layer_xfer = layer_gb / effective_bw  # seconds per layer transfer

        # Each streamed layer must be loaded TWICE per micro-batch:
        # 1. During forward (or prefetched before)
        # 2. During backward (reverse order, recompute needs weights again)
        loads_per_microbatch = 2 * n_str
        total_xfer = loads_per_microbatch * layer_xfer

    # --- Pipeline parallelism ---
    # With M micro-batches and G stages:
    # Total compute = M × compute_per_microbatch (each GPU does M forward + M backward)
    # Total transfer = M × total_xfer_per_microbatch
    # Pipeline bubble = (G-1) × compute_per_microbatch (approximately)
    M = n_micro_batches
    G = n_gpus

    total_compute = M * compute_per_microbatch
    total_transfer = M * total_xfer

    # Pipeline bubble overhead
    bubble_stages = G - 1
    bubble_time = bubble_stages * compute_per_microbatch if G > 1 else 0
    bubble_frac = bubble_stages / (M + G - 1) if (M + G - 1) > 0 else 0

    # The step time is determined by the overlap of compute and transfer.
    # During compute phases (forward + backward), we can overlap transfers.
    # The total time is max(total_compute, total_transfer) + bubble.
    # But more precisely: transfers can happen during compute of ANY layer,
    # including resident layers (which give "free" transfer time).
    step_time = max(total_compute, total_transfer) + bubble_time

    if total_compute > 0:
        overhead = max(0, (step_time - bubble_time) / total_compute - 1) * 100
    else:
        overhead = 0

    tokens = M * B * S
    tps = tokens / step_time if step_time > 0 else 0

    return StepSimulation(
        gpu_name=gpu.name,
        n_gpus=n_gpus,
        quant_name=quant.name,
        storage_name=storage.description,
        seq_len=S,
        max_micro_batch=B,
        free_vram_gb=mem.free_for_activations_gb,
        n_resident=mem.n_resident,
        n_streamed=mem.n_streamed,
        layers_per_gpu=lpg,
        forward_time_s=fwd_time,
        backward_time_s=bwd_time,
        compute_time_s=compute_per_microbatch,
        transfer_source=transfer_source,
        effective_bw_gbs=effective_bw if effective_bw != float('inf') else 0,
        bottleneck=bneck,
        layer_transfer_time_s=layer_xfer,
        total_transfer_demand_s=total_xfer,
        compute_time_per_step_s=total_compute,
        transfer_time_per_step_s=total_transfer,
        step_time_s=step_time,
        overhead_pct=overhead,
        tokens_per_step=tokens,
        tokens_per_sec=tps,
        n_micro_batches=M,
        pipeline_bubble_frac=bubble_frac,
    )


# =============================================================================
# OPTIMAL RESIDENT/BATCH TRADE-OFF
# =============================================================================

def find_optimal_resident(
    model: MoEModel,
    gpu: GPU,
    n_gpus: int,
    quant: QuantConfig,
    lora: LoRAConfig,
    storage: StorageConfig,
    seq_len: int = 1024,
    n_micro_batches: int = 1,
    gpu_utilization: float = 0.5,
) -> Optional[StepSimulation]:
    """
    Sweep n_resident to find the split that minimizes step time.

    The trade-off: fewer resident layers → more free VRAM → larger batch →
    more compute per transfer → lower streaming overhead.
    Transfer time per layer is fixed (independent of batch size), but compute
    time scales with batch size. There's an optimal point.
    """
    layers_per_gpu = math.ceil(model.n_layers / n_gpus)

    best_sim = None
    best_tps = 0.0  # maximize tokens/sec, not minimize step time

    for n_res in range(layers_per_gpu + 1):
        sim = simulate_step(
            model, gpu, n_gpus, quant, lora, storage,
            seq_len=seq_len,
            n_micro_batches=n_micro_batches,
            gpu_utilization=gpu_utilization,
            n_resident_override=n_res,
        )
        if sim is None:
            continue
        if sim.max_micro_batch < 1:
            continue
        if sim.tokens_per_sec > best_tps:
            best_tps = sim.tokens_per_sec
            best_sim = sim

    return best_sim


# =============================================================================
# VALIDATION: Check all assumptions against cross-references
# =============================================================================

def validate(model: MoEModel, lora: LoRAConfig) -> bool:
    """
    Run all sanity checks. Prints results and returns False if any FAIL.

    Sources of truth (ranked by reliability):
    1. HARD FACTS: GPU specs (VRAM, TFLOPS, PCIe gen) — from vendor datasheets
    2. HARD FACTS: NVMe bandwidth — from published benchmarks
    3. EMPIRICAL: NF4 layer size = 2250 MB — from previous analysis
    4. ESTIMATED: Total model = ~355B params, 92 layers
    5. ESTIMATED: Architecture dims (hidden, experts, intermediates) — inferred
    6. DERIVED: Everything else (FLOPs, activation mem, timing) — computed from above

    The critical chain: architecture → FLOPs → compute time → overhead ratio.
    If architecture is wrong, everything downstream is wrong.
    """
    ok = True
    warnings = []

    print("=" * 90)
    print("VALIDATION: Checking all assumptions")
    print("=" * 90)
    print()

    # ─── 1. Architecture → total params ───
    total_params = model.total_params_per_layer * model.n_layers
    target_params = 355e9
    pct_off = abs(total_params - target_params) / target_params * 100
    status = "OK" if pct_off < 5 else "WARN" if pct_off < 10 else "FAIL"
    if status != "OK":
        ok = False if status == "FAIL" else ok
        warnings.append(f"Total params {total_params/1e9:.1f}B vs target 355B ({pct_off:.1f}% off)")
    print(f"  [{status:4s}] Total params: {total_params/1e9:.2f}B (target: 355B, {pct_off:.1f}% off)")

    # ─── 2. Architecture → active params ───
    active = model.active_params_per_layer
    target_active = 514e6
    pct_off_active = abs(active - target_active) / target_active * 100
    status = "OK" if pct_off_active < 5 else "WARN" if pct_off_active < 10 else "FAIL"
    if status != "OK":
        ok = False if status == "FAIL" else ok
        warnings.append(f"Active params {active/1e6:.0f}M vs target 514M ({pct_off_active:.1f}% off)")
    print(f"  [{status:4s}] Active params/layer: {active/1e6:.0f}M (target: ~514M, {pct_off_active:.1f}% off)")

    # ─── 3. Cross-check: theoretical NF4 size vs empirical ───
    # NF4: 4 bits + absmax scales. With group_size=64, fp16 scale per group:
    # effective = 4 + 16/64 = 4.25 bits/param. With double-quant overhead: ~4.5-4.7
    params_per_layer = model.total_params_per_layer
    empirical_nf4_mb = 2250
    implied_bits = empirical_nf4_mb * 1024 * 1024 * 8 / params_per_layer
    status = "OK" if 4.0 <= implied_bits <= 5.5 else "WARN" if 3.5 <= implied_bits <= 6.0 else "FAIL"
    if status != "OK":
        ok = False if status == "FAIL" else ok
        warnings.append(f"Implied NF4 bits/param = {implied_bits:.2f} (expected 4.0-5.5)")
    theoretical_nf4_mb = params_per_layer * 4.5 / 8 / (1024**2)
    print(f"  [{status:4s}] NF4 cross-check: empirical={empirical_nf4_mb}MB, "
          f"theoretical@4.5bpp={theoretical_nf4_mb:.0f}MB, "
          f"implied={implied_bits:.2f} bits/param")

    # ─── 4. Cross-check: NF4d+NF2e size ───
    # Dense params at NF4 (~4.5 bpp), expert params at NF2 (~2.5 bpp)
    dense_params = model.attention_params + model.shared_expert_params + model.hidden_size * model.num_experts
    expert_params = model.num_experts * model.per_routing_expert_params
    theoretical_mixed = (dense_params * implied_bits + expert_params * (implied_bits * 2/4.5)) / 8 / (1024**2)
    # Better estimate: use the NF4/NF2 ratio from empirical values
    # NF2 empirical = 1150 MB → implied NF2 bits = 1150 * 1024^2 * 8 / 3.87B = 2.52 bits
    empirical_nf2_mb = 1150
    implied_nf2_bits = empirical_nf2_mb * 1024 * 1024 * 8 / params_per_layer
    # NF4d+NF2e: dense at NF4 rate, experts at NF2 rate
    mixed_mb = (dense_params * implied_bits + expert_params * implied_nf2_bits) / 8 / (1024**2)
    empirical_mixed = 1237
    pct_mixed = abs(mixed_mb - empirical_mixed) / empirical_mixed * 100
    status = "OK" if pct_mixed < 10 else "WARN" if pct_mixed < 20 else "FAIL"
    if status != "OK":
        ok = False if status == "FAIL" else ok
        warnings.append(f"NF4d+NF2e cross-check: predicted={mixed_mb:.0f}MB vs empirical={empirical_mixed}MB ({pct_mixed:.0f}%)")
    print(f"  [{status:4s}] NF4d+NF2e cross-check: predicted={mixed_mb:.0f}MB, "
          f"empirical={empirical_mixed}MB ({pct_mixed:.1f}% off)")
    print(f"         (implied bits: NF4={implied_bits:.2f}, NF2={implied_nf2_bits:.2f})")

    # ─── 5. LoRA param count sanity ───
    lora_params = lora.params_per_layer(model)
    # Cross-check: 7 projections × 2 matrices × rank × avg_dim
    avg_proj_dim = (model.hidden_size + model.num_attention_heads * model.head_dim) / 2
    expected_lora_order = 7 * 2 * lora.rank * avg_proj_dim
    ratio = lora_params / expected_lora_order
    status = "OK" if 0.5 < ratio < 2.0 else "WARN"
    if status != "OK":
        warnings.append(f"LoRA params ratio unexpected: {ratio:.2f}")
    print(f"  [{status:4s}] LoRA params/layer: {lora_params/1e6:.2f}M "
          f"(7 projections, rank={lora.rank})")
    lora_total_gb = lora.total_gpu_bytes_per_layer(model) * model.n_layers / (1024**3)
    print(f"         LoRA total GPU footprint: {lora_total_gb:.1f} GB "
          f"(weights + grads + optimizer, all 92 layers)")

    # ─── 6. GPU specs cross-check ───
    print()
    print("  GPU specs (from vendor datasheets):")
    known_specs = {
        "RTX 4090":  {"vram": 24, "bf16": 165, "pcie_gen": 4},
        "RTX 5090":  {"vram": 32, "bf16": 209, "pcie_gen": 5},
        "A100 80G":  {"vram": 80, "bf16": 312, "pcie_gen": 4},
        "H100 80G":  {"vram": 80, "bf16": 756, "pcie_gen": 5},  # PCIe dense BF16; SXM: 990
        "RTX6000P":  {"vram": 96, "bf16": 300, "pcie_gen": 5},  # placeholder
    }
    for name, gpu in GPUS.items():
        spec = known_specs.get(name, {})
        notes = []
        if gpu.vram_gb != spec.get("vram", gpu.vram_gb):
            notes.append(f"VRAM mismatch: {gpu.vram_gb} vs known {spec['vram']}")
            ok = False
        # PCIe BW: Gen4 x16 ≈ 25 GB/s theoretical, ~22 effective
        #          Gen5 x16 ≈ 63 GB/s theoretical, ~44 effective
        expected_pcie = 22 if gpu.pcie_gen == 4 else 44
        if abs(gpu.pcie_bw_gbs - expected_pcie) > 5:
            notes.append(f"PCIe BW unusual: {gpu.pcie_bw_gbs} vs expected ~{expected_pcie}")
        note_str = f" !! {'; '.join(notes)}" if notes else ""
        print(f"    {name:12s}: {gpu.vram_gb}GB, {gpu.bf16_tflops} BF16 TFLOPS, "
              f"PCIe Gen{gpu.pcie_gen} @{gpu.pcie_bw_gbs}GB/s{note_str}")

    # ─── 7. H100 TFLOPS note ───
    h100 = GPUS.get("H100 80G")
    if h100:
        print()
        print(f"  [INFO] H100 80G BF16={h100.bf16_tflops} TFLOPS (PCIe dense).")
        print(f"         H100 SXM5 dense BF16 = 990 TFLOPS (1.31× higher).")

    # ─── 8. Compute model: FLOPs per token sanity check ───
    print()
    flops_b1 = layer_forward_flops(model, 1, 1)  # 1 token
    # For a dense transformer: ~6H² FLOPs per token for attention + MLP
    # For MoE: attention is ~2×H×(nh*d + 2*kv*d + nh*d) = ~4H²
    #          MLP is (shared + k*expert) × 3×2×H = 6H×(shared + k*expert)
    expected_attn_flops = 2 * 1 * model.hidden_size * (
        model.num_attention_heads * model.head_dim +
        2 * model.num_kv_heads * model.head_dim +
        model.num_attention_heads * model.head_dim
    )
    expected_mlp_flops = (3 * 2 * 1 * model.hidden_size * model.shared_intermediate_size +
                          model.num_active_experts * 3 * 2 * 1 * model.hidden_size * model.expert_intermediate_size)
    expected_total = (expected_attn_flops + expected_mlp_flops) * 1.05  # +5% non-matmul
    ratio_flops = flops_b1 / expected_total
    status = "OK" if 0.95 < ratio_flops < 1.15 else "WARN"
    # At S=1 there's no attention QK^T/softmax*V, so we should use S=1024
    flops_1024 = layer_forward_flops(model, 1, 1024) / 1024  # per token at S=1024
    print(f"  [{status:4s}] FLOPs/token (S=1024): {flops_1024/1e6:.1f} MFLOP "
          f"(attn: {expected_attn_flops/1e6:.1f}M, mlp: {expected_mlp_flops/1e6:.1f}M per token)")

    # ─── 9. Activation memory model: cross-check against known formulas ───
    # Megatron-LM formula for activation mem per layer (with grad ckpt, flash attn):
    # ≈ 2 × B × S × H bytes (input checkpoint) + MLP intermediates
    # Our model should be in the right ballpark
    act_b1 = activation_memory_per_layer_bytes(model, 1, 1024)
    act_b8 = activation_memory_per_layer_bytes(model, 8, 1024)
    # Should scale roughly linearly with B
    linearity = (act_b8 / act_b1) / 8
    status = "OK" if 0.95 < linearity < 1.05 else "WARN"
    print(f"  [{status:4s}] Activation memory linearity: act(B=8)/act(B=1)/8 = {linearity:.3f} (expect ~1.0)")
    print(f"         act(B=1,S=1024) = {act_b1/(1024**3):.3f} GB, "
          f"act(B=8,S=1024) = {act_b8/(1024**3):.3f} GB")

    # ─── 10. Compute vs transfer dominance check ───
    # At B=1, compute time should be short relative to a 80GB GPU
    # Single layer forward: ~1.1 TFLOP at B=1 S=1024 → time on A100 (312 TFLOPS × 0.5):
    flops_layer = layer_forward_flops(model, 1, 1024)
    a100_time = flops_layer / (312e12 * 0.5)
    transfer_time = 1237 / 1024 / 22  # NF4d+NF2e layer in seconds on PCIe Gen4
    print(f"  [INFO] At B=1: A100 forward time/layer = {a100_time*1000:.1f} ms, "
          f"transfer/layer = {transfer_time*1000:.1f} ms")
    print(f"         Ratio compute/transfer = {a100_time/transfer_time:.2f} "
          f"({'compute-bound' if a100_time > transfer_time else 'TRANSFER-BOUND'})")

    # ─── 11. Memory budget sanity: does the model even fit? ───
    print()
    nf4de = QUANT_CONFIGS["NF4d+NF2e"]
    for gpu_name, gpu in GPUS.items():
        layer_gb = nf4de.layer_gb(model)
        all_layers = layer_gb * model.n_layers
        lora_all = lora.total_gpu_bytes_per_layer(model) * model.n_layers / (1024**3)
        print(f"  {gpu_name:12s}: {gpu.vram_gb:.0f}GB VRAM, "
              f"NF4d+NF2e all layers={all_layers:.0f}GB, "
              f"fits on 1 GPU: {'YES' if all_layers + lora_all + 2.5 < gpu.vram_gb else 'NO'}, "
              f"min GPUs: {math.ceil((all_layers + lora_all + 2.5) / gpu.vram_gb)}")

    # ─── Summary ───
    print()
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    ⚠ {w}")
    print()
    if ok:
        print("  RESULT: All critical checks PASSED. Warnings above are non-fatal.")
    else:
        print("  RESULT: Some checks FAILED. Results may be unreliable.")
    print()

    return ok


# =============================================================================
# MAIN: RUN ALL CONFIGURATIONS
# =============================================================================

def main():
    model = MoEModel()
    lora = LoRAConfig()

    # Run validation first
    valid = validate(model, lora)
    if "--validate-only" in sys.argv:
        sys.exit(0 if valid else 1)

    # Print model summary
    print("=" * 110)
    print("GLM-4.7 355B MoE — COMPLETE STREAMING SIMULATION")
    print("=" * 110)
    print()
    print(f"Model: {model.name}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Hidden: {model.hidden_size}, Heads: {model.num_attention_heads}, KV heads: {model.num_kv_heads}")
    print(f"  Shared expert MLP intermediate: {model.shared_intermediate_size}")
    print(f"  Routing expert MLP intermediate: {model.expert_intermediate_size}")
    print(f"  Experts: {model.num_experts} total, {model.num_active_experts} active + 1 shared")
    print(f"  Total params/layer: {model.total_params_per_layer/1e9:.2f}B")
    print(f"  Active params/layer: {model.active_params_per_layer/1e6:.0f}M")
    print(f"  Expert fraction: {model.expert_fraction*100:.1f}%")
    print()

    print(f"LoRA: rank={lora.rank}, {lora.n_projections} projections/layer")
    print(f"  Params/layer: {lora.params_per_layer(model)/1e6:.2f}M")
    print(f"  Weight/layer: {lora.weight_bytes_per_layer(model)/1e6:.1f} MB (bf16)")
    print(f"  Grad/layer: {lora.grad_bytes_per_layer(model)/1e6:.1f} MB")
    print(f"  Optimizer/layer: {lora.optimizer_bytes_per_layer(model)/1e6:.1f} MB (AdamW fp32)")
    print(f"  Total LoRA GPU mem/layer: {lora.total_gpu_bytes_per_layer(model)/1e6:.1f} MB")
    print(f"  Total LoRA GPU mem (92 layers): {lora.total_gpu_bytes_per_layer(model)*92/1e9:.2f} GB")
    print()

    print("Quantization formats:")
    for qn, qc in QUANT_CONFIGS.items():
        print(f"  {qn:12s}: {qc.layer_mb(model):7.1f} MB/layer, "
              f"{qc.total_gb(model):6.1f} GB total")
    print()

    # Sample activation memory
    print("Activation memory (1 layer, grad checkpoint, flash attn):")
    for b in [1, 2, 4, 8, 16]:
        act = activation_memory_per_layer_bytes(model, b, 1024) / (1024**3)
        print(f"  B={b:3d}, S=1024: {act:.2f} GB")
    print()

    # Compute FLOPs
    print("Forward FLOPs per layer:")
    for b in [1, 2, 4, 8, 16]:
        flops = layer_forward_flops(model, b, 1024)
        print(f"  B={b:3d}, S=1024: {flops/1e12:.2f} TFLOP")
    print()

    # =================================================================
    # Run all configs with OPTIMAL resident/batch trade-off
    # =================================================================
    SEQ_LEN = 1024
    GPU_UTILIZATION = 0.70  # benchmarked: NF4 matmul 81-97%, minus ~15% training overhead

    print("=" * 110)
    print(f"SIMULATION RESULTS — OPTIMAL RESIDENT/BATCH SPLIT")
    print(f"(seq_len={SEQ_LEN}, GPU utilization={GPU_UTILIZATION*100:.0f}%)")
    print(f"Optimizer sweeps n_resident to minimize step time.")
    print("=" * 110)
    print()

    # Header
    hdr = (f"{'Config':24s} {'Quant':>11s} {'Storage':>18s} "
           f"{'B':>3s} {'Res':>4s} {'Str':>4s} {'Free':>5s} "
           f"{'Src':>4s} {'Bnk':>5s} "
           f"{'Comp':>6s} {'Xfer':>6s} {'Step':>6s} "
           f"{'OH%':>5s} {'tok/s':>7s}")
    print(hdr)
    print("-" * len(hdr))

    def format_sim_line(config, qn, storage_desc, sim):
        comp_s = f"{sim.compute_time_per_step_s:.1f}s" if sim.compute_time_per_step_s < 100 else f"{sim.compute_time_per_step_s:.0f}s"
        xfer_s = f"{sim.transfer_time_per_step_s:.1f}s" if sim.transfer_time_per_step_s < 100 else f"{sim.transfer_time_per_step_s:.0f}s"
        step_s = f"{sim.step_time_s:.1f}s" if sim.step_time_s < 100 else f"{sim.step_time_s:.0f}s"
        oh = f"{sim.overhead_pct:.0f}%" if sim.overhead_pct > 0 else "0%"
        tps = f"{sim.tokens_per_sec:.0f}"
        return (f"{config:24s} {qn:>11s} {storage_desc:>18s} "
                f"{sim.max_micro_batch:>3d} {sim.n_resident:>4d} {sim.n_streamed:>4d} {sim.free_vram_gb:>4.1f}G "
                f"{sim.transfer_source:>4s} {sim.bottleneck:>5s} "
                f"{comp_s:>6s} {xfer_s:>6s} {step_s:>6s} "
                f"{oh:>5s} {tps:>7s}")

    # Focus on key quant configs. NVFP4 only valid on Blackwell GPUs (RTX 5090).
    BLACKWELL_GPUS = {"RTX 5090"}
    for qn in ["NF4d+NF2e", "NF4d+NF3e", "NVFP4", "NF4"]:
        quant = QUANT_CONFIGS[qn]
        for gpu_name, gpu in GPUS.items():
            # NVFP4 requires FP4 tensor cores (Blackwell only)
            if quant.compute_speedup > 1.0 and gpu_name not in BLACKWELL_GPUS:
                continue
            for ng in [1, 2, 4]:
                config = f"{ng}x {gpu_name}"
                n_mb = max(2 * ng, 4) if ng > 1 else 1

                # Use optimal trade-off for each storage config, dedup
                seen = set()
                for st_name, storage in STORAGE_CONFIGS.items():
                    sim = find_optimal_resident(
                        model, gpu, ng, quant, lora, storage,
                        seq_len=SEQ_LEN, n_micro_batches=n_mb,
                        gpu_utilization=GPU_UTILIZATION,
                    )
                    if sim is None or sim.max_micro_batch < 1:
                        continue

                    if sim.n_streamed == 0:
                        key = ("GPU", "—", 0.0, sim.max_micro_batch)
                    else:
                        key = (sim.transfer_source, sim.bottleneck,
                               round(sim.overhead_pct, 1), sim.max_micro_batch)
                    if key in seen:
                        continue
                    seen.add(key)

                    st_desc = "(all on GPU)" if sim.n_streamed == 0 else storage.description
                    print(format_sim_line(config, qn, st_desc, sim))

        print()

    # =================================================================
    # Detailed comparison: greedy vs optimal for key configs
    # =================================================================
    print()
    print("=" * 110)
    print("GREEDY vs OPTIMAL RESIDENT/BATCH TRADE-OFF")
    print("=" * 110)
    print()
    print("Greedy = maximize resident layers (minimize streaming).")
    print("Optimal = sweep n_resident to find best step time.")
    print("The trade-off: fewer resident layers → more free VRAM → larger batch")
    print("→ more compute per transfer → lower streaming overhead.")
    print()

    key_configs = [
        ("RTX 4090", 1, "NF4d+NF2e", "Gen4x1_32G"),
        ("RTX 4090", 1, "NF4d+NF3e", "Gen4x1_32G"),
        ("RTX 5090", 1, "NF4d+NF2e", "Gen5AICx4_32G"),
        ("RTX 5090", 1, "NF4d+NF3e", "Gen5AICx4_32G"),
        ("RTX 5090", 1, "NVFP4",     "Gen5AICx4_32G"),
        ("A100 80G", 1, "NF4d+NF2e", "Gen4x1_64G"),
        ("H100 80G", 1, "NF4d+NF2e", "Gen4x1_32G"),
        ("RTX6000P", 1, "NF4d+NF2e", "Gen4x1_32G"),
    ]

    for gpu_name, ng, qn, st_name in key_configs:
        gpu = GPUS[gpu_name]
        quant = QUANT_CONFIGS[qn]
        storage = STORAGE_CONFIGS[st_name]
        n_mb = max(2 * ng, 4) if ng > 1 else 1

        # Greedy (default)
        greedy = simulate_step(model, gpu, ng, quant, lora, storage,
                              seq_len=SEQ_LEN, n_micro_batches=n_mb,
                              gpu_utilization=GPU_UTILIZATION)
        # Optimal
        optimal = find_optimal_resident(model, gpu, ng, quant, lora, storage,
                                       seq_len=SEQ_LEN, n_micro_batches=n_mb,
                                       gpu_utilization=GPU_UTILIZATION)

        if greedy is None and optimal is None:
            continue

        print(f"{'─'*3} {ng}x {gpu_name} | {qn} | {storage.description} {'─'*30}")
        print(f"  {'':20s} {'Greedy':>12s}  {'Optimal':>12s}  {'Δ':>8s}")

        if greedy and optimal:
            g, o = greedy, optimal
            def delta_pct(g_val, o_val):
                if g_val == 0:
                    return ""
                return f"{(o_val/g_val - 1)*100:+.0f}%"

            print(f"  {'Resident layers':20s} {g.n_resident:>12d}  {o.n_resident:>12d}")
            print(f"  {'Streamed layers':20s} {g.n_streamed:>12d}  {o.n_streamed:>12d}")
            print(f"  {'Micro-batch (B)':20s} {g.max_micro_batch:>12d}  {o.max_micro_batch:>12d}")
            print(f"  {'Free VRAM (GB)':20s} {g.free_vram_gb:>11.1f}G  {o.free_vram_gb:>11.1f}G")
            print(f"  {'Tokens/micro-batch':20s} {g.max_micro_batch*SEQ_LEN:>12,d}  {o.max_micro_batch*SEQ_LEN:>12,d}")
            print(f"  {'Compute (s)':20s} {g.compute_time_per_step_s:>12.2f}  {o.compute_time_per_step_s:>12.2f}")
            print(f"  {'Transfer (s)':20s} {g.transfer_time_per_step_s:>12.2f}  {o.transfer_time_per_step_s:>12.2f}")
            print(f"  {'Step time (s)':20s} {g.step_time_s:>12.2f}  {o.step_time_s:>12.2f}  {delta_pct(g.step_time_s, o.step_time_s):>8s}")
            print(f"  {'Overhead':20s} {g.overhead_pct:>11.1f}%  {o.overhead_pct:>11.1f}%")
            print(f"  {'Tokens/sec':20s} {g.tokens_per_sec:>12.0f}  {o.tokens_per_sec:>12.0f}  {delta_pct(g.tokens_per_sec, o.tokens_per_sec):>8s}")
        print()

    # =================================================================
    # Sweep: resident/batch curves
    # =================================================================
    sweep_configs = [
        ("RTX 4090", "NF4d+NF2e", "Gen4x1_32G",    1),
        ("RTX 4090", "NF4d+NF3e", "Gen4x1_32G",    1),
        ("RTX 5090", "NVFP4",     "Gen5AICx4_32G", 1),
        ("A100 80G", "NF4d+NF2e", "Gen4x1_64G",    1),
        ("H100 80G", "NF4d+NF2e", "Gen4x1_32G",    1),
    ]

    for gpu_name, qn, st_name, ng in sweep_configs:
        gpu = GPUS[gpu_name]
        quant = QUANT_CONFIGS[qn]
        storage = STORAGE_CONFIGS[st_name]
        lpg = math.ceil(model.n_layers / ng)

        print()
        print("=" * 90)
        print(f"TRADE-OFF CURVE: {ng}x {gpu_name} | {qn} | {storage.description}")
        print("Sweeping n_resident from 0 to max. Fewer resident → larger batch → more compute overlap.")
        print("=" * 90)
        print()

        hdr2 = f"{'Res':>4s} {'Str':>4s} {'Free':>6s} {'B':>3s} {'tok':>6s} {'Comp':>7s} {'Xfer':>7s} {'Step':>7s} {'OH%':>6s} {'tok/s':>7s}  {'note':s}"
        print(hdr2)
        print("-" * len(hdr2))

        best_tps = 0
        best_n_res = 0
        results = []
        for n_res in range(lpg + 1):
            sim = simulate_step(
                model, gpu, ng, quant, lora, storage,
                seq_len=SEQ_LEN, n_micro_batches=1,
                gpu_utilization=GPU_UTILIZATION,
                n_resident_override=n_res,
            )
            if sim is None or sim.max_micro_batch < 1:
                continue
            if sim.tokens_per_sec > best_tps:
                best_tps = sim.tokens_per_sec
                best_n_res = n_res
            results.append((n_res, sim))

        for n_res, sim in results:
            oh = f"{sim.overhead_pct:.0f}%" if sim.overhead_pct > 0 else "0%"
            note = " ← OPTIMAL" if n_res == best_n_res else ""
            print(f"{sim.n_resident:>4d} {sim.n_streamed:>4d} {sim.free_vram_gb:>5.1f}G "
                  f"{sim.max_micro_batch:>3d} {sim.tokens_per_step:>6d} "
                  f"{sim.compute_time_per_step_s:>6.1f}s {sim.transfer_time_per_step_s:>6.1f}s "
                  f"{sim.step_time_s:>6.1f}s {oh:>6s} {sim.tokens_per_sec:>7.0f}{note}")


if __name__ == "__main__":
    main()
