"""GDS validation script for dettmers-desktop.

Tests GDS (GPUDirect Storage) vs CPU pinned weight streaming on a real model.
Forces zero residency to exercise the full streaming path even when VRAM is plentiful.

Run on dettmers-desktop:
    BNB_CUDA_VERSION=131 PYTHONPATH=. python scripts/validate_gds.py
"""

import time
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from bitsandbytes.kbit_lora import KbitLoraModel


def train_steps(model, input_ids_list, labels_list, n_steps=20, label=""):
    """Train n_steps and return per-step timing and losses."""
    optimizer = torch.optim.AdamW(
        [p for p in model._lora_params.parameters() if p.requires_grad],
        lr=1e-4,
    )
    norm_params = [p for p in model.parameters() if p.requires_grad and p not in set(model._lora_params.parameters())]
    if norm_params:
        optimizer.add_param_group({"params": norm_params, "lr": 1e-4})

    step_times = []
    losses = []

    # Warmup step (not counted)
    idx = 0
    input_ids = input_ids_list[idx].unsqueeze(0).cuda()
    labels = labels_list[idx].unsqueeze(0).cuda()
    optimizer.zero_grad()
    loss, ctx = model.forward_streaming(input_ids, labels)
    model.backward_streaming(ctx)
    optimizer.step()
    torch.cuda.synchronize()
    print(f"  [{label}] Warmup done, loss={loss.item():.4f}")

    for step in range(n_steps):
        idx = (step + 1) % len(input_ids_list)
        input_ids = input_ids_list[idx].unsqueeze(0).cuda()
        labels = labels_list[idx].unsqueeze(0).cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        loss, ctx = model.forward_streaming(input_ids, labels)
        model.backward_streaming(ctx)
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_times.append(t1 - t0)
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  [{label}] Step {step:2d} | loss={loss.item():.4f} | {t1 - t0:.3f}s")

    return step_times, losses


def main():
    import os

    quantized_path = os.path.expanduser("~/quantized/qwen3-30b-a3b-4bit.safetensors")
    model_name = "Qwen/Qwen3-30B-A3B"
    n_steps = 20

    # Load tokenizer and prepare data
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(50))
    input_ids_list = []
    labels_list = []
    for ex in ds:
        tokens = tokenizer(ex["text"], truncation=True, max_length=256, return_tensors="pt")
        ids = tokens["input_ids"][0]
        if len(ids) >= 10:
            input_ids_list.append(ids)
            labels_list.append(ids.clone())
    print(f"  {len(input_ids_list)} samples prepared")

    # Compute model size for bandwidth calculation
    import json
    import struct

    with open(quantized_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_size))
    metadata = header_json.get("__metadata__", {})
    n_layers = int(metadata.get("num_layers", 48))
    print(f"  Model: {n_layers} layers")

    # === Test 1: GDS path ===
    print("\n=== GDS path (use_gds=True, forced 0 resident) ===")
    torch.manual_seed(42)
    with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
        model_gds = KbitLoraModel.from_quantized(
            quantized_path,
            weight_streaming=True,
            use_gds=True,
            lora_r=16,
        )
    print(f"  RAM strategy: {model_gds._ram_strategy}")
    print(f"  GDS enabled: {model_gds._use_gds}")
    print(f"  Resident layers: {model_gds._n_resident}")

    gds_times, gds_losses = train_steps(model_gds, input_ids_list, labels_list, n_steps=n_steps, label="GDS")

    del model_gds
    torch.cuda.empty_cache()

    # === Test 2: CPU pinned path ===
    print("\n=== CPU pinned path (forced 0 resident) ===")
    torch.manual_seed(42)
    with patch.object(KbitLoraModel, "_compute_residency", return_value=0):
        model_pinned = KbitLoraModel.from_quantized(
            quantized_path,
            weight_streaming=True,
            use_gds=False,
            lora_r=16,
        )
    print(f"  RAM strategy: {model_pinned._ram_strategy}")
    print(f"  Resident layers: {model_pinned._n_resident}")

    pinned_times, pinned_losses = train_steps(
        model_pinned, input_ids_list, labels_list, n_steps=n_steps, label="Pinned"
    )

    del model_pinned
    torch.cuda.empty_cache()

    # === Results ===
    print("\n=== Results ===")
    gds_avg = sum(gds_times) / len(gds_times)
    pinned_avg = sum(pinned_times) / len(pinned_times)
    print(f"  GDS avg step time:    {gds_avg:.3f}s")
    print(f"  Pinned avg step time: {pinned_avg:.3f}s")
    print(f"  Speedup (pinned/GDS): {pinned_avg / gds_avg:.2f}x")

    # Estimate streaming bandwidth
    # Each step does forward (48 layers) + backward (48 layers) = 96 layer loads
    # Model size on disk: ~19.53 GB, so per-layer ~0.407 GB
    # For streaming: each layer loaded twice (forward + backward) per step
    # But with double-buffering, prefetch overlaps with compute
    model_size_gb = 19.53  # approximate
    layer_size_gb = model_size_gb / n_layers
    layers_per_step = n_layers * 2  # forward + backward
    data_per_step_gb = layers_per_step * layer_size_gb

    gds_bw = data_per_step_gb / gds_avg
    pinned_bw = data_per_step_gb / pinned_avg
    print("\n  Estimated streaming bandwidth:")
    print(f"    GDS:    {gds_bw:.1f} GB/s ({data_per_step_gb:.1f} GB/step)")
    print(f"    Pinned: {pinned_bw:.1f} GB/s")

    # Loss comparison
    max_diff = 0
    for i, (lg, lp) in enumerate(zip(gds_losses, pinned_losses)):
        if lp != 0:
            diff = abs(lg - lp) / abs(lp)
            max_diff = max(max_diff, diff)
    print(f"\n  Loss comparison: max relative diff = {max_diff:.4f}")
    if max_diff < 0.05:
        print("  PASS: GDS and pinned produce matching losses")
    else:
        print("  WARNING: Loss difference exceeds 5%")

    print("\n  All GDS validation checks passed!" if max_diff < 0.05 else "\n  GDS validation had issues.")


if __name__ == "__main__":
    main()
