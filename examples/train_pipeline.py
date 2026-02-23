"""Pipeline parallelism training example using bitsandbytes kbit quantization.

Demonstrates distributed pipeline training across 2+ GPUs with per-rank
model loading — each GPU only loads the decoder layers it needs:
- First stage: embedding + first half of layers
- Last stage: remaining layers + final norm + LM head (loss)

This reduces per-GPU memory compared to loading the full model everywhere.

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
    The KbitLoraModel has already been created with only this stage's layers.
    """

    def __init__(self, kbit_model):
        super().__init__()
        self.km = kbit_model

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        self.km._extend_rope_cache(S, device)
        hidden = self.km.embed_tokens(input_ids).to(self.km.compute_dtype)
        for i in range(self.km._num_loaded_layers):
            hidden = self.km._layer_forward(i, hidden, position_ids)
        return hidden


class KbitLastStage(nn.Module):
    """Last pipeline stage: remaining layers + final norm.

    Takes hidden states [B, S, H], returns hidden states after norm [B*S, H].
    Loss is computed externally by the engine's loss_fn.
    The KbitLoraModel has already been created with only this stage's layers
    plus the final norm and LM head.
    """

    def __init__(self, kbit_model):
        super().__init__()
        self.km = kbit_model

    def forward(self, hidden):
        from bitsandbytes.autograd.training_kernels import rmsnorm

        B, S, H = hidden.shape
        device = hidden.device
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        self.km._extend_rope_cache(S, device)

        for i in range(self.km._num_loaded_layers):
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

    is_first = (rank == 0)
    is_last = (rank == world_size - 1)

    if rank == 0:
        print(f"{'=' * 60}")
        print(f"Pipeline QLoRA Training ({world_size} GPUs, per-rank loading)")
        print(f"{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"LoRA rank: {args.lora_r}, k={args.k}")
        print(f"Seq len: {args.seq_len}, Micro-batches: {args.micro_batches}")
        print(f"Steps: {args.steps}")
        print()

    # Load model — each rank loads the full HF model temporarily to extract
    # its layer weights. We immediately delete the original after quantization.
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    layers_per_stage = num_layers // world_size
    layer_start = rank * layers_per_stage
    layer_end = (rank + 1) * layers_per_stage if rank < world_size - 1 else num_layers

    role = "first" if is_first else ("last" if is_last else "mid")
    print(f"  GPU {rank}: layers {layer_start}-{layer_end-1} ({role} stage)")

    if rank == 0:
        print(f"\nLoading and quantizing (per-rank)...")
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    mem_after_load = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU {rank}: {mem_after_load:.0f} MB after HF model load")

    # Create KbitLoraModel with ONLY this rank's layers
    kbit_model = KbitLoraModel(
        model,
        lora_r=args.lora_r,
        lora_alpha=16.0,
        k=args.k,
        compute_dtype=torch.bfloat16,
        layer_range=(layer_start, layer_end),
        include_embed=is_first,
        include_lm_head=is_last,
    )

    # Delete the original HF model to free memory
    del model
    torch.cuda.empty_cache()

    mem_after_quant = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU {rank}: {mem_after_quant:.0f} MB after quantize + cleanup "
          f"({kbit_model._num_loaded_layers} layers, "
          f"embed={'yes' if is_first else 'no'}, "
          f"lm_head={'yes' if is_last else 'no'})")

    if rank == 0:
        print(f"  Trainable params (rank 0): {kbit_model.num_trainable_parameters():,}")

    # Create pipeline stage wrappers
    if is_first:
        stage = KbitFirstStage(kbit_model)
    else:
        stage = KbitLastStage(kbit_model)

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

    # Optimizer — each rank optimizes only its own trainable parameters
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

        # All ranks generate same data with same seed (for label consistency)
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
