"""End-to-end QLoRA training example using bitsandbytes kbit quantization.

Demonstrates the full training stack:
- Load a HuggingFace model
- Apply KbitLoraModel (kbit quantization + LoRA adapters)
- Chunked attention + chunked MLP + chunked CE + gradient checkpointing
- Optional CPU offload for inter-layer activations
- Train on Alpaca dataset or synthetic data
- Report tokens/sec, peak GPU memory, time per step
- Optional memory comparison mode (chunked vs unchunked)

Usage:
    # Default: Qwen3-0.6B on Alpaca dataset
    python examples/train_qlora.py

    # Synthetic data (no dataset download needed)
    python examples/train_qlora.py --synthetic

    # With CPU offload for lower memory
    python examples/train_qlora.py --cpu-offload

    # Memory comparison (chunked vs unchunked)
    python examples/train_qlora.py --compare-memory --steps 5

    # Larger model
    python examples/train_qlora.py --model Qwen/Qwen3-4B --cpu-offload

    # Custom settings
    python examples/train_qlora.py --steps 200 --lora-r 128 --seq-len 1024
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload for inter-layer activations")
    parser.add_argument(
        "--weight-streaming",
        action="store_true",
        help="Stream frozen weights from CPU pinned memory to GPU layer-by-layer. "
        "Reduces GPU memory from O(n_layers) to O(1) for base weights. "
        "Implies --cpu-offload. Effective when per-layer compute >= PCIe transfer.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of Alpaca")
    parser.add_argument("--compare-memory", action="store_true", help="Run memory comparison: chunked vs unchunked")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
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


def load_alpaca_dataset(tokenizer, seq_len, num_samples=None):
    """Load and tokenize the Alpaca dataset for next-token prediction."""
    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def format_sample(sample):
        """Format an Alpaca sample as instruction-following text."""
        if sample["input"]:
            text = (
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input']}\n\n"
                f"### Response:\n{sample['output']}"
            )
        else:
            text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
        return text

    # Pre-tokenize all samples
    tokenized = []
    for sample in dataset:
        text = format_sample(sample)
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) >= 4:  # Skip very short samples
            tokenized.append(ids)

    return tokenized


class AlpacaDataLoader:
    """Simple iterator over tokenized Alpaca samples, packed to seq_len."""

    def __init__(self, tokenized_samples, batch_size, seq_len, device, pad_token_id=0):
        self.samples = tokenized_samples
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.pad_token_id = pad_token_id
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        input_ids_list = []
        labels_list = []
        for _ in range(self.batch_size):
            # Get next sample, wrap around
            ids = self.samples[self.idx % len(self.samples)]
            self.idx += 1

            # Truncate or pad to seq_len
            if len(ids) > self.seq_len:
                ids = ids[: self.seq_len]
            pad_len = self.seq_len - len(ids)
            labels = list(ids)

            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len  # Don't compute loss on padding

            # Shift labels for next-token prediction: mask first position
            labels[0] = -100

            input_ids_list.append(ids)
            labels_list.append(labels)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)
        return input_ids, labels


def run_training(args, kbit_model, data_source, label):
    """Run a training loop and return metrics."""
    trainable_params = kbit_model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    kbit_model.train()
    vocab_size = kbit_model.vocab_size
    losses = []
    step_times = []
    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Training ({label})")
    print(f"{'=' * 60}")

    if isinstance(data_source, AlpacaDataLoader):
        data_iter = iter(data_source)
    else:
        data_iter = None

    for step in range(args.steps):
        t_step = time.time()

        optimizer.zero_grad()

        accum_loss = 0.0
        step_tokens = 0

        for accum_step in range(args.grad_accum):
            # Get batch
            if data_iter is not None:
                input_ids, labels = next(data_iter)
            else:
                input_ids, labels = generate_synthetic_batch(
                    args.batch_size,
                    args.seq_len,
                    vocab_size,
                    "cuda",
                )

            # Forward
            result = kbit_model(input_ids, labels=labels)
            loss = result["loss"] / args.grad_accum

            # Backward
            loss.backward()

            accum_loss += loss.item()
            # Count non-masked tokens
            step_tokens += (labels != -100).sum().item()

        optimizer.step()
        total_tokens += step_tokens

        losses.append(accum_loss)
        dt = time.time() - t_step
        step_times.append(dt)
        tokens_per_sec = step_tokens / dt

        if step % 10 == 0 or step == args.steps - 1:
            peak_mb = get_gpu_peak_mb()
            print(
                f"  Step {step:4d}/{args.steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"Time: {dt:.2f}s | "
                f"Tok/s: {tokens_per_sec:.0f} | "
                f"Peak mem: {peak_mb:.0f} MB"
            )

    peak_mb = get_gpu_peak_mb()
    avg_step_time = sum(step_times[1:]) / max(len(step_times) - 1, 1)  # Skip warmup step
    avg_tokens_per_sec = total_tokens / sum(step_times)

    return {
        "losses": losses,
        "peak_mb": peak_mb,
        "avg_step_time": avg_step_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "total_tokens": total_tokens,
    }


def print_results(metrics, label):
    """Print training results summary."""
    losses = metrics["losses"]
    print(f"\n{'=' * 60}")
    print(f"Results ({label})")
    print(f"{'=' * 60}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss change:  {losses[-1] - losses[0]:.4f}")
    print(f"  Peak GPU memory: {metrics['peak_mb']:.0f} MB")
    print(f"  Avg step time: {metrics['avg_step_time']:.3f}s")
    print(f"  Avg tokens/sec: {metrics['avg_tokens_per_sec']:.0f}")
    print(f"  Total tokens: {metrics['total_tokens']:,}")

    if len(losses) >= 20:
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        if late_avg < early_avg:
            print(f"  Loss DECREASED from {early_avg:.4f} to {late_avg:.4f} (PASS)")
        else:
            print(f"  WARNING: Loss did not decrease ({early_avg:.4f} -> {late_avg:.4f})")
    elif losses[-1] < losses[0]:
        print("  Loss decreased (PASS)")
    else:
        print("  WARNING: Loss did not decrease")


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print("QLoRA Training with bitsandbytes kbit quantization")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Quantization: k={args.k}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Steps: {args.steps}, Grad accum: {args.grad_accum}")
    # --weight-streaming implies --cpu-offload
    if args.weight_streaming:
        args.cpu_offload = True
    print(f"CPU offload: {args.cpu_offload}")
    print(f"Weight streaming: {args.weight_streaming}")
    print(f"Data: {'synthetic' if args.synthetic else 'Alpaca'}")
    print(f"Chunks: attn={args.attn_chunk}, mlp={args.mlp_chunk}, ce={args.ce_chunk}")
    print()

    # Load tokenizer (needed for Alpaca dataset)
    tokenizer = None
    if not args.synthetic:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model on CPU, then stream weights to GPU during quantization.
    # This keeps peak GPU memory minimal — only 1 layer of fp16 weights at a
    # time on GPU, plus the growing quantized data.
    print("Loading base model on CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  GPU memory after load: {get_gpu_memory_mb():.0f} MB (model on CPU)")

    # Apply KbitLoraModel — streams weights CPU->GPU one layer at a time
    print("\nQuantizing and streaming to GPU...")
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
        cpu_offload=args.cpu_offload,
        weight_streaming=args.weight_streaming,
        target_device=torch.device("cuda"),
    )
    print(f"  Quantized in {time.time() - t0:.1f}s")
    print(f"  Trainable parameters: {kbit_model.num_trainable_parameters():,}")
    print(f"  GPU memory after quantization: {get_gpu_memory_mb():.0f} MB")
    print(f"  Peak GPU memory during load: {get_gpu_peak_mb():.0f} MB")

    # Free the original model (CPU memory)
    del model

    # Prepare dataset
    if not args.synthetic:
        print("\nLoading Alpaca dataset...")
        t0 = time.time()
        tokenized = load_alpaca_dataset(
            tokenizer,
            args.seq_len,
            num_samples=max(args.steps * args.batch_size * args.grad_accum * 2, 1000),
        )
        print(f"  Tokenized {len(tokenized)} samples in {time.time() - t0:.1f}s")
        data_source = AlpacaDataLoader(
            tokenized,
            args.batch_size,
            args.seq_len,
            "cuda",
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        data_source = None  # Will use synthetic

    # Run training
    metrics = run_training(args, kbit_model, data_source, "full stack")
    print_results(metrics, "full stack")

    # Memory comparison mode
    if args.compare_memory:
        print(f"\n{'=' * 60}")
        print("Memory Comparison: chunked vs unchunked")
        print(f"{'=' * 60}")

        # Already have chunked metrics
        chunked_peak = metrics["peak_mb"]

        # Run unchunked: set chunk sizes very large so no chunking occurs
        print("\nRunning with large chunk sizes (effectively unchunked)...")
        kbit_model.attn_chunk_size = 999999
        kbit_model.mlp_chunk_size = 999999
        kbit_model.ce_chunk_size = 999999
        kbit_model.cpu_offload = False

        # Re-initialize optimizer (LoRA params may have accumulated state)
        if not args.synthetic:
            data_source_unchunked = AlpacaDataLoader(
                tokenized,
                args.batch_size,
                args.seq_len,
                "cuda",
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            data_source_unchunked = None

        unchunked_args = argparse.Namespace(**vars(args))
        unchunked_args.steps = min(args.steps, 5)  # Just a few steps for comparison
        metrics_unchunked = run_training(unchunked_args, kbit_model, data_source_unchunked, "unchunked")

        print(f"\n{'=' * 60}")
        print("Memory Comparison Results")
        print(f"{'=' * 60}")
        print(f"  Chunked peak:   {chunked_peak:.0f} MB")
        print(f"  Unchunked peak: {metrics_unchunked['peak_mb']:.0f} MB")
        savings = metrics_unchunked["peak_mb"] - chunked_peak
        pct = (savings / metrics_unchunked["peak_mb"]) * 100 if metrics_unchunked["peak_mb"] > 0 else 0
        print(f"  Savings:        {savings:.0f} MB ({pct:.1f}%)")


if __name__ == "__main__":
    main()
