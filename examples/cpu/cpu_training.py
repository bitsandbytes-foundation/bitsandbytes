"""
End-to-end finetuning on CPU using bitsandbytes optimizers.

Demonstrates that bnb.optim.AdamW / AdamW8bit / Adam / SGD etc. work
on CPU with a real model, using JackFram/llama-68m + Alpaca Clean.

Usage:
    python cpu_training.py
    python cpu_training.py --optimizer adamw8bit --steps 50
    python cpu_training.py --optimizer sgd --lr 0.001 --steps 30
    python cpu_training.py --compare  # compare bnb AdamW vs torch AdamW
    python cpu_training.py --use_trainer --optimizer adamw8bit  # use HF Trainer
"""

import argparse
import time

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

import bitsandbytes as bnb


def get_args():
    parser = argparse.ArgumentParser(description="CPU bitsandbytes optimizer training")
    parser.add_argument("--model", type=str, default="JackFram/llama-68m")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=[
            "adamw",
            "adamw8bit",
            "adamw32bit",
            "adam",
            "adam8bit",
            "adam32bit",
            "sgd",
            "sgd8bit",
            "lion",
            "lion8bit",
            "rmsprop",
            "rmsprop8bit",
            "adagrad",
            "adagrad8bit",
            "lamb",
            "lars",
        ],
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Compare bnb AdamW vs torch AdamW")
    parser.add_argument("--use_trainer", action="store_true", help="Use HF Trainer instead of manual training loop")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


def format_alpaca(example):
    if example.get("input", ""):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def prepare_data(tokenizer, dataset_name, max_length, num_samples=200):
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
    return {k: torch.tensor([ex[k] for ex in batch]) for k in batch[0].keys()}


def create_optimizer(model, name, lr):
    optim_map = {
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit,
        "adam": bnb.optim.Adam,
        "adam8bit": bnb.optim.Adam8bit,
        "adam32bit": bnb.optim.Adam32bit,
        "lion": bnb.optim.Lion,
        "lion8bit": bnb.optim.Lion8bit,
        "rmsprop": bnb.optim.RMSprop,
        "rmsprop8bit": bnb.optim.RMSprop8bit,
        "adagrad": bnb.optim.Adagrad,
        "adagrad8bit": bnb.optim.Adagrad8bit,
        "lamb": bnb.optim.LAMB,
        "lars": lambda p, lr: bnb.optim.LARS(p, lr, momentum=0.9),
        "sgd": lambda p, lr: bnb.optim.SGD(p, lr, momentum=0.9),
        "sgd8bit": lambda p, lr: bnb.optim.SGD8bit(p, lr, momentum=0.9),
    }
    factory = optim_map[name]
    return factory(model.parameters(), lr=lr)


def train_loop(model, optimizer, dataloader, steps, log_interval):
    model.train()
    history = []
    step = 0
    t0 = time.time()

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

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
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


def run_single(args):
    dtype = get_torch_dtype(args.dtype)
    print(f"=== Training with bnb {args.optimizer} on CPU ({args.dtype}) ===")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Steps: {args.steps} | LR: {args.lr} | Batch: {args.batch_size} | MaxLen: {args.max_length}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    ds = prepare_data(tokenizer, args.dataset, args.max_length)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )

    optimizer = create_optimizer(model, args.optimizer, args.lr)

    history = train_loop(model, optimizer, dataloader, args.steps, args.log_interval)

    loss_start = history[0][1]
    loss_end = history[-1][1]
    total_time = history[-1][2]
    print("\n--- Results ---")
    print(f"Loss: {loss_start:.4f} -> {loss_end:.4f} (delta={loss_start - loss_end:+.4f})")
    print(f"Total time: {total_time:.1f}s ({args.steps / total_time:.1f} steps/s)")
    print(f"Optimizer: bnb.optim.{args.optimizer} | Dtype: {args.dtype}")

    if loss_end >= loss_start:
        print("WARNING: Loss did not decrease! Training may not be working correctly.")
    else:
        print("OK: Loss decreased as expected.")

    return history


def run_compare(args):
    """Compare bnb AdamW vs torch AdamW on CPU to verify correctness."""
    dtype = get_torch_dtype(args.dtype)
    print(f"=== Comparing bnb AdamW vs torch AdamW on CPU ({args.dtype}) ===\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = prepare_data(tokenizer, args.dataset, args.max_length, num_samples=100)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    results = {}
    for label, make_opt in [
        ("bnb.AdamW", lambda m: bnb.optim.AdamW(m.parameters(), lr=args.lr)),
        ("torch.AdamW", lambda m: torch.optim.AdamW(m.parameters(), lr=args.lr)),
    ]:
        print(f"\n>> {label}")
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        optimizer = make_opt(model)
        history = train_loop(model, optimizer, dataloader, args.steps, args.log_interval)
        results[label] = history

    print(f"\n{'Step':>5} | {'bnb Loss':>10} | {'torch Loss':>11} | {'Diff':>10}")
    print("-" * 50)
    h_bnb = results["bnb.AdamW"]
    h_pt = results["torch.AdamW"]
    for i in range(0, min(len(h_bnb), len(h_pt)), max(1, args.log_interval)):
        s1, l1, _ = h_bnb[i]
        _, l2, _ = h_pt[i]
        print(f"{s1:5d} | {l1:10.4f} | {l2:11.4f} | {abs(l1 - l2):10.6f}")

    final_diff = abs(h_bnb[-1][1] - h_pt[-1][1])
    print(f"\nFinal loss difference: {final_diff:.6f}")
    if final_diff < 0.01:
        print("OK: bnb and torch AdamW produce nearly identical results on CPU.")
    else:
        print("NOTE: Some divergence detected (may grow over many steps).")


def run_with_trainer(args):
    """Train using HuggingFace Trainer with a bnb optimizer on CPU."""
    dtype = get_torch_dtype(args.dtype)
    print(f"=== Trainer mode with bnb {args.optimizer} on CPU ({args.dtype}) ===")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Steps: {args.steps} | LR: {args.lr} | Batch: {args.batch_size} | MaxLen: {args.max_length}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    ds = prepare_data(tokenizer, args.dataset, args.max_length)

    training_args = TrainingArguments(
        output_dir="./cpu_trainer_output",
        per_device_train_batch_size=args.batch_size,
        max_steps=args.steps,
        logging_steps=args.log_interval,
        learning_rate=args.lr,
        save_strategy="steps",
        save_steps=args.steps,
        save_total_limit=1,
        report_to="none",
        bf16=(args.dtype == "bf16"),
        no_cuda=True,
        dataloader_pin_memory=False,
    )

    optimizer = create_optimizer(model, args.optimizer, args.lr)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print("\n--- Trainer Results ---")
    print(f"Training loss: {metrics['train_loss']:.4f}")
    print(f"Training runtime: {metrics['train_runtime']:.1f}s")
    print(f"Steps/sec: {metrics['train_steps_per_second']:.1f}")
    print(f"Optimizer: bnb.optim.{args.optimizer} | Dtype: {args.dtype}")

    save_dir = "./cpu_trainer_output/final"
    print(f"\nSaving model and tokenizer to {save_dir} ...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Save complete.")

    # Verify saved model can be loaded back
    print("Verifying saved model loads correctly ...")
    loaded_model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=dtype)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    test_input = loaded_tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        out = loaded_model(**test_input)
    print(f"Reload OK — output logits shape: {out.logits.shape}")
    print("Full CPU finetune pipeline completed successfully.")


def main():
    args = get_args()

    if args.compare:
        run_compare(args)
    elif args.use_trainer:
        run_with_trainer(args)
    else:
        run_single(args)


if __name__ == "__main__":
    set_seed(42)
    main()
