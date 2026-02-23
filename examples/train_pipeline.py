"""Pipeline parallelism training example using bitsandbytes kbit quantization.

Demonstrates distributed pipeline training across 2+ GPUs:
- Loads a HuggingFace model and applies KbitLoraModel
- Splits decoder layers across GPUs (first stage = embedding + first layers,
  last stage = remaining layers + norm + LM head)
- Trains using DistributedPipelineEngine with NCCL
- Reports per-GPU memory and throughput

Usage:
    # 2-GPU pipeline training on Qwen3-0.6B
    torchrun --nproc_per_node=2 examples/train_pipeline.py

    # Larger model
    torchrun --nproc_per_node=2 examples/train_pipeline.py --model Qwen/Qwen3-4B

    # More micro-batches for better pipeline utilization
    torchrun --nproc_per_node=2 examples/train_pipeline.py --micro-batches 8
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

if "BNB_CUDA_VERSION" not in os.environ:
    pass

import bitsandbytes  # noqa: F401
from bitsandbytes.kbit_lora import KbitLoraModel
from bitsandbytes.pipeline import DistributedPipelineEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline QLoRA training")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model name")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--k", type=int, default=4, help="Quantization bit width")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--micro-batches", type=int, default=4, help="Number of micro-batches")
    return parser.parse_args()


class KbitFirstStage(nn.Module):
    """First pipeline stage: embedding + first layers.

    Takes input_ids [B, S], returns hidden states [B, S, H].
    """

    def __init__(self, kbit_model, layer_start, layer_end):
        super().__init__()
        self.km = kbit_model
        self.layer_start = layer_start
        self.layer_end = layer_end

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        self.km._extend_rope_cache(S, device)
        hidden = self.km.embed_tokens(input_ids).to(self.km.compute_dtype)
        for i in range(self.layer_start, self.layer_end):
            hidden = self.km._layer_forward(i, hidden, position_ids)
        return hidden


class KbitLastStage(nn.Module):
    """Last pipeline stage: remaining layers + final norm.

    Takes hidden states [B, S, H], returns hidden states after norm [B*S, H].
    Loss is computed externally by the engine's loss_fn.
    """

    def __init__(self, kbit_model, layer_start, layer_end):
        super().__init__()
        self.km = kbit_model
        self.layer_start = layer_start
        self.layer_end = layer_end

    def forward(self, hidden):
        from bitsandbytes.autograd.training_kernels import rmsnorm

        B, S, H = hidden.shape
        device = hidden.device
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        self.km._extend_rope_cache(S, device)

        for i in range(self.layer_start, self.layer_end):
            hidden = self.km._layer_forward(i, hidden, position_ids)

        # Final norm
        hidden_2d = hidden.reshape(-1, self.km.hidden_size)
        hidden_2d = rmsnorm(
            hidden_2d, self.km._norm_weights["final_norm_weight"],
            eps=self.km.rms_norm_eps,
        )
        return hidden_2d


def make_loss_fn(kbit_model):
    """Create a loss function closure that uses chunked cross-entropy."""
    from bitsandbytes.autograd.chunked_ce import chunked_cross_entropy

    km = kbit_model
    lm = km._lm_head_info

    def loss_fn(hidden_2d, labels):
        """Compute chunked cross-entropy loss.

        Args:
            hidden_2d: [B*S, H] hidden states from last stage.
            labels: [B, S] target token IDs.
        """
        shift_hidden = hidden_2d[:-1]
        shift_labels = labels.reshape(-1)[1:]
        loss = chunked_cross_entropy(
            shift_hidden, lm["packed"], lm["absmax"], lm["codebook"],
            shift_labels,
            lm["k"], lm["K"], lm["N_padded"], lm["N"],
            km.compute_dtype, km.ce_chunk_size,
        )
        return loss

    return loss_fn


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"{'=' * 60}")
        print(f"Pipeline QLoRA Training ({world_size} GPUs)")
        print(f"{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"LoRA rank: {args.lora_r}, k={args.k}")
        print(f"Seq len: {args.seq_len}, Micro-batches: {args.micro_batches}")
        print(f"Steps: {args.steps}")
        print()

    # Load model
    from transformers import AutoModelForCausalLM

    if rank == 0:
        print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    # Quantize
    if rank == 0:
        print("Quantizing and creating LoRA adapters...")
    kbit_model = KbitLoraModel(
        model,
        lora_r=args.lora_r,
        lora_alpha=16.0,
        k=args.k,
        compute_dtype=torch.bfloat16,
    )
    del model
    torch.cuda.empty_cache()

    num_layers = kbit_model.num_layers
    layers_per_stage = num_layers // world_size
    layer_start = rank * layers_per_stage
    layer_end = (rank + 1) * layers_per_stage if rank < world_size - 1 else num_layers

    is_first = (rank == 0)
    is_last = (rank == world_size - 1)

    if is_first:
        stage = KbitFirstStage(kbit_model, layer_start, layer_end)
    else:
        stage = KbitLastStage(kbit_model, layer_start, layer_end)

    if rank == 0:
        print(f"  Total layers: {num_layers}")
        print(f"  Trainable params: {kbit_model.num_trainable_parameters():,}")

    for r in range(world_size):
        if r == rank:
            ls = r * layers_per_stage
            le = (r + 1) * layers_per_stage if r < world_size - 1 else num_layers
            role = "first" if r == 0 else ("last" if r == world_size - 1 else "mid")
            print(f"  GPU {r}: layers {ls}-{le-1} ({role} stage)")
        dist.barrier()

    # Loss function for the last stage
    loss_fn = make_loss_fn(kbit_model) if is_last else None

    # Hidden shape for inter-stage communication: [B, S, H]
    hidden_shape = (1, args.seq_len, kbit_model.hidden_size)

    # Pipeline engine
    engine = DistributedPipelineEngine(
        stage_module=stage,
        rank=rank,
        world_size=world_size,
        loss_fn=loss_fn,
        num_micro_batches=args.micro_batches,
        hidden_shape=hidden_shape,
        dtype=torch.bfloat16,
    )

    # Optimizer — each rank has its own view of the parameters
    trainable_params = kbit_model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Training loop
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Training")
        print(f"{'=' * 60}")

    vocab_size = kbit_model.vocab_size
    losses = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        t_step = time.time()
        optimizer.zero_grad()

        # Generate micro-batches (all ranks generate same data for labels)
        # Use deterministic seed per step so last rank has correct labels
        torch.manual_seed(step * 1000 + 42)
        micro_batch_inputs = []
        micro_batch_labels = []
        for mb in range(args.micro_batches):
            input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=device)
            labels = input_ids.clone()
            labels[:, :1] = -100
            micro_batch_inputs.append(input_ids)
            micro_batch_labels.append(labels)

        # Run pipeline step
        result = engine.step(
            micro_batch_inputs=micro_batch_inputs if is_first else None,
            micro_batch_labels=micro_batch_labels if is_last else None,
        )

        # Get loss from last rank
        loss_val = result["loss"] if is_last else 0.0
        loss_tensor = torch.tensor([loss_val], device=device)
        dist.broadcast(loss_tensor, src=world_size - 1)
        loss_val = loss_tensor.item()

        optimizer.step()
        losses.append(loss_val)

        dt = time.time() - t_step
        tokens = args.micro_batches * args.seq_len

        if rank == 0 and (step % 5 == 0 or step == args.steps - 1):
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(
                f"  Step {step:3d}/{args.steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Time: {dt:.2f}s | "
                f"Tok/s: {tokens/dt:.0f} | "
                f"Peak mem: {peak_mb:.0f} MB"
            )

    # Results
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")

        if len(losses) >= 10:
            early = sum(losses[:5]) / 5
            late = sum(losses[-5:]) / 5
            if late < early:
                print(f"  Loss DECREASED from {early:.4f} to {late:.4f} (PASS)")
            else:
                print(f"  WARNING: Loss did not decrease ({early:.4f} -> {late:.4f})")

    # Report per-GPU peak memory
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1024 / 1024], device=device)
    peaks = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(peaks, peak)
    if rank == 0:
        for r, p in enumerate(peaks):
            print(f"  GPU {r} peak memory: {p.item():.0f} MB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
