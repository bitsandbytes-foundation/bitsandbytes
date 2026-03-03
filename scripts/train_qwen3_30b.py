"""End-to-end training validation for Qwen3-30B-A3B.

Trains with both streaming (from_quantized) and non-streaming (standard) paths
and compares loss curves. Success criterion: loss must match within 5% per step.
"""

import json
import os
import time

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from bitsandbytes.checkpoint import save_lora
from bitsandbytes.kbit_lora import KbitLoraModel


def prepare_data(tokenizer, n_samples=200, max_len=256):
    """Load Alpaca and tokenize."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(n_samples))

    all_input_ids = []
    all_labels = []
    for example in ds:
        text = example["text"]
        tokens = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        if len(input_ids) < 10:
            continue
        all_input_ids.append(input_ids)
        all_labels.append(input_ids.clone())

    return all_input_ids, all_labels


def train_streaming(model, input_ids_list, labels_list, n_steps=100, lr=1e-4):
    """Train with forward_streaming / backward_streaming."""
    optimizer = torch.optim.AdamW(
        [p for p in model._lora_params.parameters() if p.requires_grad],
        lr=lr,
    )
    # Also add norm params
    norm_params = [p for p in model.parameters() if p.requires_grad and p not in set(model._lora_params.parameters())]
    if norm_params:
        optimizer.add_param_group({"params": norm_params, "lr": lr})

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        idx = step % len(input_ids_list)
        input_ids = input_ids_list[idx].unsqueeze(0).cuda()
        labels = labels_list[idx].unsqueeze(0).cuda()

        optimizer.zero_grad()
        loss, ctx = model.forward_streaming(input_ids, labels)
        model.backward_streaming(ctx)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step:3d} | loss={loss_val:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Training complete: {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    return losses


def train_standard(model, input_ids_list, labels_list, n_steps=100, lr=1e-4):
    """Train with standard forward + loss.backward()."""
    optimizer = torch.optim.AdamW(
        [p for p in model._lora_params.parameters() if p.requires_grad],
        lr=lr,
    )
    norm_params = [p for p in model.parameters() if p.requires_grad and p not in set(model._lora_params.parameters())]
    if norm_params:
        optimizer.add_param_group({"params": norm_params, "lr": lr})

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        idx = step % len(input_ids_list)
        input_ids = input_ids_list[idx].unsqueeze(0).cuda()
        labels = labels_list[idx].unsqueeze(0).cuda()

        optimizer.zero_grad()
        result = model(input_ids, labels)
        loss = result["loss"]
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step:3d} | loss={loss_val:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Training complete: {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    return losses


def compare_losses(losses_streaming, losses_standard, tolerance=0.05):
    """Compare two loss curves. Returns True if they match within tolerance."""
    assert len(losses_streaming) == len(losses_standard)
    max_rel_diff = 0
    mismatches = 0
    for i, (ls, ln) in enumerate(zip(losses_streaming, losses_standard)):
        if ln == 0:
            continue
        rel_diff = abs(ls - ln) / abs(ln)
        max_rel_diff = max(max_rel_diff, rel_diff)
        if rel_diff > tolerance:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Step {i}: streaming={ls:.4f} standard={ln:.4f} diff={rel_diff:.4f}")

    print(f"  Max relative difference: {max_rel_diff:.4f}")
    print(f"  Steps exceeding {tolerance * 100}% tolerance: {mismatches}/{len(losses_streaming)}")
    return mismatches == 0, max_rel_diff


def main():
    quantized_path = os.path.expanduser("~/quantized/qwen3-30b-a3b-4bit.safetensors")
    model_name = "Qwen/Qwen3-30B-A3B"
    n_steps = 100
    lr = 1e-4

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    print("Preparing data...")
    input_ids_list, labels_list = prepare_data(tokenizer, n_samples=200, max_len=256)
    print(f"  {len(input_ids_list)} samples prepared")

    # === Path 1: Streaming (from_quantized) ===
    print("\n=== Streaming path (from_quantized) ===")
    torch.manual_seed(42)
    model_stream = KbitLoraModel.from_quantized(
        quantized_path,
        weight_streaming=True,
        lora_r=16,
    )
    losses_streaming = train_streaming(model_stream, input_ids_list, labels_list, n_steps=n_steps, lr=lr)

    # Save LoRA
    lora_path = os.path.expanduser("~/quantized/qwen3-30b-lora.pt")
    save_lora(model_stream, lora_path)
    print(f"  LoRA saved to {lora_path}")

    # Free memory
    del model_stream
    torch.cuda.empty_cache()

    # === Path 2: Non-streaming (standard forward) ===
    print("\n=== Non-streaming path (standard forward) ===")
    torch.manual_seed(42)
    model_standard = KbitLoraModel.from_quantized(
        quantized_path,
        weight_streaming=False,
        lora_r=16,
    )
    losses_standard = train_standard(model_standard, input_ids_list, labels_list, n_steps=n_steps, lr=lr)

    del model_standard
    torch.cuda.empty_cache()

    # === Compare loss curves ===
    print("\n=== Loss curve comparison ===")
    matches, max_diff = compare_losses(losses_streaming, losses_standard)
    if matches:
        print("  PASS: Loss curves match within 5%")
    else:
        print("  FAIL: Loss curves diverge by more than 5%")

    # === Reload LoRA and verify ===
    print("\n=== LoRA reload test ===")
    torch.manual_seed(42)
    model_reload = KbitLoraModel.from_quantized(
        quantized_path,
        weight_streaming=False,
        lora_r=16,
        lora_checkpoint=lora_path,
    )
    # Quick inference test
    prompt = "What is machine learning?"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].cuda()

    with torch.no_grad():
        result = model_reload(input_ids, labels=None)
        logits = result["logits"]
        # Generate a few tokens greedily
        next_tokens = logits.argmax(dim=-1)
        generated = tokenizer.decode(next_tokens[0], skip_special_tokens=True)
        print(f"  LoRA reload OK, generated: {generated[:100]}")

    # Save results
    results = {
        "losses_streaming": losses_streaming,
        "losses_standard": losses_standard,
        "max_rel_diff": max_diff,
        "matches": matches,
        "n_steps": n_steps,
        "lr": lr,
        "model": "Qwen3-30B-A3B",
        "lora_r": 16,
    }
    results_path = os.path.expanduser("~/quantized/training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
