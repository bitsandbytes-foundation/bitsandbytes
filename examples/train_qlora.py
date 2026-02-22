"""End-to-end QLoRA training example using bitsandbytes kbit quantization.

Demonstrates the full training stack:
- Load a HuggingFace model
- Apply KbitLoraModel (kbit quantization + LoRA adapters)
- Train with AdamW on synthetic data
- Verify loss decreases
- Log memory usage

Usage:
    python examples/train_qlora.py                           # Default: Qwen3-0.6B
    python examples/train_qlora.py --model Qwen/Qwen3-4B    # Larger model
    python examples/train_qlora.py --steps 200 --lora-r 128  # More steps, higher rank
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM

# Force BNB_CUDA_VERSION if not set
if "BNB_CUDA_VERSION" not in os.environ:
    # Auto-detect from the installed library
    pass

import bitsandbytes  # noqa: F401
from bitsandbytes.kbit_lora import KbitLoraModel


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA training with bitsandbytes kbit")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model name")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--k", type=int, default=4, help="Quantization bit width (2-5)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--attn-chunk", type=int, default=256, help="Attention chunk size")
    parser.add_argument("--mlp-chunk", type=int, default=256, help="MLP chunk size")
    parser.add_argument("--ce-chunk", type=int, default=4096, help="CE vocab chunk size")
    return parser.parse_args()


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_gpu_peak_mb():
    """Get peak GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def generate_synthetic_batch(batch_size, seq_len, vocab_size, device):
    """Generate a synthetic training batch (random tokens)."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :1] = -100  # Mask first token (no label for BOS)
    return input_ids, labels


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print(f"QLoRA Training with bitsandbytes kbit quantization")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Quantization: k={args.k}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Steps: {args.steps}")
    print()

    # Load base model
    print("Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  GPU memory after load: {get_gpu_memory_mb():.0f} MB")

    # Apply KbitLoraModel
    print("\nQuantizing and creating LoRA adapters...")
    t0 = time.time()
    kbit_model = KbitLoraModel(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        k=args.k,
        attn_chunk_size=args.attn_chunk,
        mlp_chunk_size=args.mlp_chunk,
        ce_chunk_size=args.ce_chunk,
        compute_dtype=torch.bfloat16,
    )
    print(f"  Quantized in {time.time() - t0:.1f}s")
    print(f"  Trainable parameters: {kbit_model.num_trainable_parameters():,}")
    print(f"  GPU memory after quantization: {get_gpu_memory_mb():.0f} MB")

    # Free the original model weights (they're now quantized)
    del model
    torch.cuda.empty_cache()
    print(f"  GPU memory after cleanup: {get_gpu_memory_mb():.0f} MB")

    # Set up optimizer
    trainable_params = kbit_model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Training loop
    print(f"\n{'=' * 60}")
    print("Training")
    print(f"{'=' * 60}")

    vocab_size = kbit_model.vocab_size
    losses = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        t_step = time.time()

        # Generate synthetic batch
        input_ids, labels = generate_synthetic_batch(
            args.batch_size, args.seq_len, vocab_size, "cuda",
        )

        # Forward
        result = kbit_model(input_ids, labels=labels)
        loss = result["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        dt = time.time() - t_step

        if step % 10 == 0 or step == args.steps - 1:
            peak_mb = get_gpu_peak_mb()
            print(
                f"  Step {step:4d}/{args.steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Time: {dt:.2f}s | "
                f"Peak mem: {peak_mb:.0f} MB"
            )

    # Verify loss decrease
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss change:  {losses[-1] - losses[0]:.4f}")
    print(f"  Peak GPU memory: {get_gpu_peak_mb():.0f} MB")

    # Check if loss decreased over time
    # Compare first 10 steps vs last 10 steps
    if len(losses) >= 20:
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        if late_avg < early_avg:
            print(f"  Loss DECREASED from {early_avg:.4f} to {late_avg:.4f} (OK)")
        else:
            print(f"  WARNING: Loss did not decrease ({early_avg:.4f} -> {late_avg:.4f})")
    else:
        if losses[-1] < losses[0]:
            print("  Loss decreased (OK)")
        else:
            print("  WARNING: Loss did not decrease")


if __name__ == "__main__":
    main()
