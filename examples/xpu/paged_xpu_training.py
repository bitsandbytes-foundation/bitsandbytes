"""
Real training case for XPU Paged Optimizer using JackFram/llama-68m + Alpaca Clean.

Usage:
    python tests/test_paged_xpu_training.py
    python tests/test_paged_xpu_training.py --optimizer paged_adamw --steps 50
    python tests/test_paged_xpu_training.py --compare  # compare paged vs non-paged loss curves
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import bitsandbytes as bnb


def get_args():
    parser = argparse.ArgumentParser(description="XPU Paged Optimizer Training Test")
    parser.add_argument("--model", type=str, default="JackFram/llama-68m")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--optimizer", type=str, default="paged_adamw",
                        choices=["paged_adamw", "paged_adamw8bit", "paged_adamw32bit",
                                 "paged_adam", "paged_adam8bit", "paged_adam32bit",
                                 "paged_lion", "paged_lion8bit", "paged_lion32bit",
                                 "adamw", "adamw8bit", "adamw32bit",
                                 "adam", "adam8bit", "adam32bit"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Compare paged vs non-paged optimizer")
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    return parser.parse_args()


def format_alpaca(example):
    if example.get("input", ""):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def prepare_data(tokenizer, dataset_name, max_length, num_samples=200):
    """Load and tokenize a small subset of Alpaca."""
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(num_samples, len(ds))))

    def tokenize(example):
        text = format_alpaca(example)
        enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def collate_fn(batch):
    return {
        k: torch.tensor([ex[k] for ex in batch])
        for k in batch[0].keys()
    }


def create_optimizer(model, name, lr):
    """Create a bnb optimizer by name."""
    optim_map = {
        "paged_adamw": bnb.optim.PagedAdamW,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "paged_adam": bnb.optim.PagedAdam,
        "paged_adam8bit": bnb.optim.PagedAdam8bit,
        "paged_adam32bit": bnb.optim.PagedAdam32bit,
        "paged_lion": bnb.optim.PagedLion,
        "paged_lion8bit": bnb.optim.PagedLion8bit,
        "paged_lion32bit": bnb.optim.PagedLion32bit,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit,
        "adam": bnb.optim.Adam,
        "adam8bit": bnb.optim.Adam8bit,
        "adam32bit": bnb.optim.Adam32bit,
    }
    cls = optim_map[name]
    return cls(model.parameters(), lr=lr)


def train_loop(model, optimizer, dataloader, steps, log_interval, device):
    """Run training and return list of (step, loss, time) tuples."""
    model.train()
    history = []
    step = 0
    t0 = time.time()

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            elapsed = time.time() - t0
            history.append((step, loss_val, elapsed))

            if step % log_interval == 0:
                print(f"  step {step:4d} | loss {loss_val:.4f} | time {elapsed:.1f}s")

            step += 1

    return history


def get_torch_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def run_single(args):
    """Train with one optimizer and report results."""
    device = args.device
    dtype = get_torch_dtype(args.dtype)
    print(f"=== Training with {args.optimizer} on {device} ({args.dtype}) ===")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Steps: {args.steps} | LR: {args.lr} | Batch: {args.batch_size} | MaxLen: {args.max_length}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)

    ds = prepare_data(tokenizer, args.dataset, args.max_length)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = create_optimizer(model, args.optimizer, args.lr)

    history = train_loop(model, optimizer, dataloader, args.steps, args.log_interval, torch.device(device))

    loss_start = history[0][1]
    loss_end = history[-1][1]
    total_time = history[-1][2]
    print(f"\n--- Results ---")
    print(f"Loss: {loss_start:.4f} -> {loss_end:.4f} (delta={loss_start - loss_end:+.4f})")
    print(f"Total time: {total_time:.1f}s ({args.steps / total_time:.1f} steps/s)")
    print(f"Optimizer: {args.optimizer} | Dtype: {args.dtype}")

    if loss_end >= loss_start:
        print("WARNING: Loss did not decrease! Training may not be working correctly.")
    else:
        print("OK: Loss decreased as expected.")

    return history


def run_compare(args):
    """Compare paged_adamw vs adamw numerically."""
    device = args.device
    dtype = get_torch_dtype(args.dtype)
    print(f"=== Comparing paged_adamw vs adamw on {device} ({args.dtype}) ===\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = prepare_data(tokenizer, args.dataset, args.max_length, num_samples=100)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    results = {}
    for opt_name in ["adamw", "paged_adamw"]:
        print(f"\n>> {opt_name}")
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
        optimizer = create_optimizer(model, opt_name, args.lr)
        history = train_loop(model, optimizer, dataloader, args.steps, args.log_interval, torch.device(device))
        results[opt_name] = history

    print("\n=== Comparison ===")
    print(f"{'Step':>5} | {'AdamW Loss':>11} | {'PagedAdamW Loss':>16} | {'Diff':>10}")
    print("-" * 55)
    h_normal = results["adamw"]
    h_paged = results["paged_adamw"]
    for i in range(0, min(len(h_normal), len(h_paged)), max(1, args.log_interval)):
        s1, l1, _ = h_normal[i]
        s2, l2, _ = h_paged[i]
        print(f"{s1:5d} | {l1:11.4f} | {l2:16.4f} | {abs(l1 - l2):10.6f}")

    final_diff = abs(h_normal[-1][1] - h_paged[-1][1])
    print(f"\nFinal loss difference: {final_diff:.6f}")
    if final_diff < 0.1:
        print("OK: Paged and non-paged optimizers produce similar results.")
    else:
        print("NOTE: Some divergence detected. This may be expected due to async paging operations.")


def main():
    args = get_args()

    # Sanity check device
    if args.device == "xpu":
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available!"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available!"

    if args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()


# python paged_xpu_training.py --compare
# === Comparison ===
#  Step |  AdamW Loss |  PagedAdamW Loss |       Diff
# -------------------------------------------------------
#     0 |      4.9552 |           4.9552 |   0.000000
#     5 |      5.0027 |           5.0053 |   0.002588
#    10 |      2.7280 |           2.7284 |   0.000325
#    15 |      1.7927 |           1.7960 |   0.003312
#    20 |      2.8800 |           2.8778 |   0.002215
#    25 |      2.6720 |           2.6712 |   0.000807

# Final loss difference: 0.000739
# OK: Paged and non-paged optimizers produce similar results.
