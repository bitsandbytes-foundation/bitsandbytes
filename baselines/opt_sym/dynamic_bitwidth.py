#!/usr/bin/env python3
"""HIGGS Dynamic Bitwidth Quantization - Section 5 implementation.

This implements the data-free dynamic bitwidth algorithm from the HIGGS paper:
1. Algorithm 3: Estimate per-layer sensitivity coefficients α_l using KL-divergence
2. Compute quantization error table t²_{l,j} for each layer and bitwidth option
3. Optimize bitwidth assignment via greedy knapsack solution

Usage:
    # Step 1: Calibrate α_l coefficients (data-free, uses random tokens)
    python dynamic_bitwidth.py --model meta-llama/Llama-3.1-8B --seqlen 2048 \
        --calibrate --calibration-tokens 287000 --output-alpha alphas.json

    # Step 2: Compute quantization error table
    python dynamic_bitwidth.py --model meta-llama/Llama-3.1-8B --seqlen 2048 \
        --compute-error-table --error-table-path error_table.json

    # Step 3: Optimize bitwidth assignment for target average bitrate
    python dynamic_bitwidth.py --model meta-llama/Llama-3.1-8B \
        --optimize --alpha-path alphas.json --error-table-path error_table.json \
        --target-bits 3.0 --bitwidth-options "k=2,p=2;k=3,p=2;k=4,p=2" \
        --output-assignment assignment.json

    # Or run all steps together:
    python dynamic_bitwidth.py --model meta-llama/Llama-3.1-8B --seqlen 2048 \
        --calibrate --compute-error-table --optimize --target-bits 3.0
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

# Shared HF cache setup
_SHARED_HF = "/data/hf_cache"
if os.path.isdir(_SHARED_HF):
    os.environ.setdefault("HF_HUB_CACHE", f"{_SHARED_HF}/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{_SHARED_HF}/datasets")


# ============================================================
# Noise injection for Algorithm 3
# ============================================================

class LayerNoiseInjector:
    """Inject Gaussian noise into a specific layer's weights."""

    def __init__(self, model: nn.Module, layer_idx: int, noise_std: float):
        self.model = model
        self.layer_idx = layer_idx
        self.noise_std = noise_std
        self.target_module = None
        self._hook_handle = None
        self._original_weight = None

        # Find the target linear layer
        self.linear_layers = self._get_linear_layers()
        if layer_idx >= len(self.linear_layers):
            raise ValueError(f"Layer index {layer_idx} out of range (max {len(self.linear_layers)})")
        self.target_module = self.linear_layers[layer_idx]

    def _get_linear_layers(self) -> List[nn.Linear]:
        """Get all Linear layers excluding embed and lm_head."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "embed" not in name and "lm_head" not in name:
                layers.append(module)
        return layers

    def __enter__(self):
        """Context manager entry - inject noise."""
        self._original_weight = self.target_module.weight.data.clone()
        noise = torch.randn_like(self._original_weight) * self.noise_std
        self.target_module.weight.data = self._original_weight + noise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original weight."""
        if self._original_weight is not None:
            self.target_module.weight.data = self._original_weight
            self._original_weight = None
        return False


# ============================================================
# KL Divergence computation (data-free calibration)
# ============================================================

@torch.no_grad()
def compute_kl_divergence(
    clean_logits: torch.Tensor,
    noised_logits: torch.Tensor,
) -> float:
    """Compute KL divergence between clean and noised model outputs.

    Args:
        clean_logits: Logits from clean model, shape (batch, seq, vocab)
        noised_logits: Logits from noised model, shape (batch, seq, vocab)

    Returns:
        KL divergence KL(p_clean || p_noised) averaged over batch*seq
    """
    # Compute KL divergence
    # KL(p||q) = Σ p(x) * log(p(x)/q(x))
    p_clean = torch.softmax(clean_logits, dim=-1)
    log_p_clean = torch.log_softmax(clean_logits, dim=-1)
    log_p_noised = torch.log_softmax(noised_logits, dim=-1)

    # KL = Σ p_clean * (log p_clean - log p_noised)
    kl = (p_clean * (log_p_clean - log_p_noised)).sum(dim=-1)

    # Average over batch and sequence
    return kl.mean().item()


@torch.no_grad()
def generate_random_tokens(
    vocab_size: int,
    n_tokens: int,
    seq_len: int,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Generate random token IDs for data-free calibration.

    Args:
        vocab_size: Size of vocabulary
        n_tokens: Total number of tokens to generate
        seq_len: Sequence length per batch
        device: Device to place tensor on
        seed: Random seed

    Returns:
        Tensor of shape (n_tokens // seq_len, seq_len) with random token IDs
    """
    n_seqs = (n_tokens + seq_len - 1) // seq_len  # Round up
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_seqs, seq_len), generator=rng, device=device)
    return tokens


@torch.no_grad()
def estimate_alpha_coefficients(
    model: nn.Module,
    tokenizer,
    n_tokens: int = 287000,
    seq_len: int = 2048,
    n_noise_levels: int = 15,
    t_min: float = 0.001,
    t_max: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[int, float]:
    """Estimate per-layer sensitivity coefficients α_l using Algorithm 3 (data-free variant).

    This implements the data-free version using KL divergence on random tokens instead
    of perplexity on calibration data.

    Args:
        model: The pretrained model
        tokenizer: Tokenizer (for vocab size)
        n_tokens: Number of random tokens to use (default 287k matches HIGGS paper)
        seq_len: Sequence length for forward passes
        n_noise_levels: Number of noise levels J (default 15 matches paper)
        t_min: Minimum noise std
        t_max: Maximum noise std
        device: Device
        seed: Random seed

    Returns:
        Dictionary mapping layer index to alpha coefficient
    """
    print("=" * 60)
    print("HIGGS Algorithm 3: Estimating α_l coefficients (data-free)")
    print("=" * 60)
    print(f"Random tokens: {n_tokens:,}, sequence length: {seq_len}")
    print(f"Noise levels: {n_noise_levels}, range: [{t_min}, {t_max}]")
    print(f"Device: {device}, seed: {seed}")

    # Count linear layers - only main transformer layers (o_proj and down_proj)
    # Reduces from 224 to ~64 layers, focusing on the bottleneck operations
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "embed" not in name and "lm_head" not in name:
            # Only include output projections (o_proj, down_proj) - skip Q,K,V,up,gate
            if any(x in name for x in ['o_proj', 'down_proj']):
                linear_layers.append((name, module))
    n_layers = len(linear_layers)

    print(f"\nFound {n_layers} Linear layers to calibrate")
    print("-" * 60)

    # Generate random tokens (keep on CPU to save GPU memory)
    vocab_size = len(tokenizer)
    random_tokens_cpu = generate_random_tokens(vocab_size, n_tokens, seq_len, "cpu", seed)
    n_sequences = random_tokens_cpu.shape[0]
    print(f"Generated {random_tokens_cpu.numel():,} random tokens in {n_sequences} sequences")

    # Noise levels (t values)
    t_values = np.linspace(t_min, t_max, n_noise_levels)
    print(f"\nNoise levels: {t_values}")

    # For each layer, compute KL divergence at each noise level
    # Process one sequence at a time to avoid OOM
    alphas = {}

    for layer_idx in range(n_layers):
        layer_name, layer_module = linear_layers[layer_idx]
        print(f"\n[{layer_idx+1}/{n_layers}] {layer_name}")

        # Accumulate KL sums for each noise level
        kl_sums = np.zeros(n_noise_levels)

        for seq_idx in range(n_sequences):
            # Move sequence to GPU
            seq_tokens = random_tokens_cpu[seq_idx:seq_idx+1].to(device)

            # Compute clean logits for this sequence
            with torch.no_grad():
                outputs = model(seq_tokens)
                clean_logits = outputs.logits

            # For each noise level, add noise and compute KL
            for j, t_j in enumerate(t_values):
                with LayerNoiseInjector(model, layer_idx, t_j):
                    with torch.no_grad():
                        outputs_noised = model(seq_tokens)
                        noised_logits = outputs_noised.logits

                # Compute KL divergence
                kl = compute_kl_divergence(clean_logits, noised_logits)
                kl_sums[j] += kl

                # Clean up noised logits
                del noised_logits, outputs_noised

            # Clean up clean logits
            del clean_logits, outputs, seq_tokens
            torch.cuda.empty_cache()

            if (seq_idx + 1) % 10 == 0 or seq_idx == n_sequences - 1:
                print(f"  Sequence {seq_idx+1}/{n_sequences} done")

        # Average KL over sequences
        kl_deltas = kl_sums / n_sequences

        for j, t_j in enumerate(t_values):
            print(f"  t={t_j:.4f}: KL={kl_deltas[j]:.6f}")

        # Least squares fit: ΔKL = α_l * t²
        # α_l = Σ(Δ_j * t²_j) / Σ(t⁴_j)
        t_sq = t_values ** 2
        t_fourth = t_values ** 4

        numerator = np.dot(kl_deltas, t_sq)
        denominator = np.sum(t_fourth)

        alpha_l = numerator / denominator if denominator > 1e-12 else 0.0
        alphas[layer_idx] = alpha_l

        print(f"  → α_{layer_idx} = {alpha_l:.6f}")

    print("\n" + "=" * 60)
    print("Calibration complete")
    print("=" * 60)
    print(f"\nα_l coefficients (sorted by magnitude):")
    sorted_alphas = sorted(alphas.items(), key=lambda x: x[1], reverse=True)
    for layer_idx, alpha in sorted_alphas:
        layer_name = linear_layers[layer_idx][0]
        print(f"  Layer {layer_idx:2d}: α = {alpha:10.6f}  ({layer_name})")

    return alphas


# ============================================================
# Quantization error table computation
# ============================================================

def get_quantization_options(options_str: str, norm: str = "l2",
                             blocksize: int = 32, rot_blocksize: int = 128) -> List[Dict]:
    """Parse bitwidth options string into list of configs.

    Args:
        options_str: String like "k=2,p=2;k=3,p=2;k=4,p=2"
        norm: Normalization type ("l2" or "absmax")
        blocksize: Block size for absmax norm (affects scale overhead)
        rot_blocksize: Rotation block size for L2 norm (affects scale overhead)

    Returns:
        List of dicts with keys 'k', 'p', and computed 'bits', 'index_bits'
    """
    options = []
    for opt_str in options_str.split(";"):
        opt_str = opt_str.strip()
        if not opt_str:
            continue
        parts = {}
        for kv in opt_str.split(","):
            k, v = kv.strip().split("=")
            parts[k.strip()] = int(v.strip())

        k = parts.get("k", 4)
        p = parts.get("p", 1)
        index_bits = k * p

        # Effective bits = index bits + scale overhead
        if norm == "l2":
            # L2: one fp16 scale per rot_blocksize elements
            scale_overhead = 16 / rot_blocksize
        else:
            # Absmax: one int8 scale per blocksize elements
            scale_overhead = 8 / blocksize

        bits_per_entry = k + scale_overhead  # Total bits per weight element

        options.append({
            "k": k,
            "p": p,
            "index_bits": index_bits,
            "bits_per_entry": bits_per_entry,
            "scale_overhead": scale_overhead,
            "config_str": f"k{k}p{p}",
        })

    return options


@torch.no_grad()
def compute_quantization_error_table(
    model: nn.Module,
    options: List[Dict],
    blocksize: int = 32,
    rot_blocksize: int = 128,
    norm: str = "l2",
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Compute quantization error t²_{l,j} for each layer and option.

    This quantizes each layer's weights with each option and computes MSE.
    No model inference needed - just weight quantization.

    Args:
        model: The model
        options: List of quantization option dicts
        blocksize: Block size for absmax normalization
        rot_blocksize: Rotation block size for Hadamard
        norm: Normalization type ("l2" or "absmax")
        seed: Random seed
        cache_dir: Optional cache directory for codebooks

    Returns:
        Dict with error_table (layer_idx -> option_idx -> mse) and metadata
    """
    from eval_ppl import (
        compute_codebook, compute_gaussian_grid, _chunked_nearest,
        make_hadamard_block, make_random_signs
    )

    print("=" * 60)
    print("Computing quantization error table")
    print("=" * 60)
    print(f"Options: {len(options)}")
    for opt in options:
        print(f"  {opt['config_str']}: k={opt['k']}, p={opt['p']}")
    print(f"Normalization: {norm}, blocksize={blocksize}, rot_blocksize={rot_blocksize}")

    device = next(model.parameters()).device

    # Get linear layers
    linear_layers = []
    layer_shapes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "embed" not in name and "lm_head" not in name:
            linear_layers.append((name, module))
            layer_shapes.append(module.weight.shape)

    n_layers = len(linear_layers)
    print(f"\nLayers: {n_layers}")

    # Precompute codebooks for each option
    print("\nPrecomputing codebooks...")
    codebooks = {}
    for opt_idx, opt in enumerate(options):
        k, p = opt["k"], opt["p"]
        cache_key = (k, p)

        if cache_key in codebooks:
            continue

        print(f"  [{opt_idx+1}/{len(options)}] Computing codebook for k={k}, p={p}...")

        if norm == "l2":
            # L2 normalization needs Gaussian grid
            quant_cb, deq_cb, kappa1, _ = compute_gaussian_grid(k, p, device=device)
        else:
            # Absmax uses block-normalized samples
            quant_cb, deq_cb, kappa1, _ = compute_codebook(k, p, blocksize, device=device)

        codebooks[cache_key] = {
            "quant_cb": quant_cb,
            "deq_cb": deq_cb,
            "kappa1": kappa1,
        }

    # Compute error for each layer and option
    print("\nComputing quantization errors...")
    error_table = {}  # layer_idx -> {option_idx: mse, ...}
    total_elements = {}  # layer_idx -> num_elements

    # Hadamard setup for L2 norm
    if norm == "l2":
        H_block = make_hadamard_block(rot_blocksize, "cpu")
        max_in = max(shape[1] for shape in layer_shapes)
        signs = make_random_signs(max_in, seed, "cpu")

    for layer_idx, (name, module) in enumerate(linear_layers):
        print(f"\n[{layer_idx+1}/{n_layers}] {name}: {module.weight.shape}")
        error_table[layer_idx] = {}
        total_elements[layer_idx] = module.weight.numel()

        W_orig = module.weight.data.float().cpu()
        out_dim, in_dim = W_orig.shape

        for opt_idx, opt in enumerate(options):
            k, p = opt["k"], opt["p"]
            cb_info = codebooks[(k, p)]
            quant_cb = cb_info["quant_cb"].cpu()
            deq_cb = cb_info["deq_cb"].cpu()

            # Quantize weights
            if norm == "l2":
                # L2 normalization path (HIGGS-style)
                n_rot = in_dim // rot_blocksize
                if n_rot == 0:
                    # Layer too small for rotation - skip or use absmax
                    error_table[layer_idx][opt_idx] = float('inf')
                    continue

                W_blocks = W_orig[:, :n_rot * rot_blocksize].reshape(
                    out_dim, n_rot, rot_blocksize)
                l2_norms = W_blocks.norm(dim=2, keepdim=True).clamp_(min=1e-12)
                W_unit = W_blocks / l2_norms

                # Hadamard transform
                H_local = H_block.to(W_unit.dtype)
                signs_local = signs[:rot_blocksize]
                W_unit_flat = W_unit.reshape(out_dim * n_rot, rot_blocksize)
                W_rotated = (W_unit_flat * signs_local.unsqueeze(0)) @ H_local.T
                W_rotated = W_rotated * (rot_blocksize ** 0.5)

                # VQ quantization
                groups = W_rotated.reshape(-1, p)
                dists = torch.cdist(groups.float(), quant_cb.float())
                indices = dists.argmin(dim=1)
                dq_groups = deq_cb[indices]

                # Inverse transform
                dq_blocks = dq_groups.reshape(out_dim * n_rot, rot_blocksize)
                dq_blocks = dq_blocks / (rot_blocksize ** 0.5)
                dq_blocks = (dq_blocks @ H_local) * signs_local.unsqueeze(0)
                dq_blocks = dq_blocks.reshape(out_dim, n_rot, rot_blocksize)
                W_q = (dq_blocks * l2_norms).reshape(out_dim, n_rot * rot_blocksize)

                # Handle remainder
                if in_dim > n_rot * rot_blocksize:
                    W_q = torch.cat([W_q, W_orig[:, n_rot * rot_blocksize:]], dim=1)

            else:
                # Absmax normalization path (BNF-style)
                flat = W_orig.flatten()
                n = flat.numel()
                pad_n = (blocksize - n % blocksize) % blocksize
                if pad_n > 0:
                    flat = torch.nn.functional.pad(flat, (0, pad_n))
                blocks = flat.reshape(-1, blocksize)
                absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
                normalized = blocks / absmax

                if p == 1:
                    # Scalar quantization
                    midpoints = (quant_cb[:-1] + quant_cb[1:]) / 2
                    indices = torch.bucketize(normalized, midpoints)
                    dequantized = deq_cb[indices] * absmax
                else:
                    # VQ quantization
                    n_blocks = normalized.shape[0]
                    elems_per_block = (blocksize // p) * p
                    vq_part = normalized[:, :elems_per_block]
                    groups = vq_part.reshape(-1, p)
                    dists = torch.cdist(groups.float(), quant_cb.float())
                    indices = dists.argmin(dim=1)
                    dq_groups = deq_cb[indices]
                    dq_vq = dq_groups.reshape(n_blocks, elems_per_block)
                    if elems_per_block < blocksize:
                        rem_part = normalized[:, elems_per_block:]
                        dequantized = torch.cat([dq_vq, rem_part], dim=1) * absmax
                    else:
                        dequantized = dq_vq * absmax

                W_q = dequantized.flatten()[:n].reshape(W_orig.shape)

            # Compute MSE
            mse = ((W_orig - W_q) ** 2).mean().item()
            error_table[layer_idx][opt_idx] = mse

            print(f"  {opt['config_str']}: MSE = {mse:.8f}")

    return {
        "error_table": error_table,
        "total_elements": total_elements,
        "options": options,
        "norm": norm,
        "blocksize": blocksize,
        "rot_blocksize": rot_blocksize,
    }


# ============================================================
# Bitwidth assignment optimization
# ============================================================

def optimize_bitwidth_assignment(
    alphas: Dict[int, float],
    error_table: Dict[int, Dict[int, float]],
    total_elements: Dict[int, int],
    options: List[Dict],
    target_avg_bits: float,
) -> Dict:
    """Optimize per-layer bitwidth assignment using greedy knapsack.

    Solves:
        min  Σ_l α_l * t²_{l,j_l}
        s.t. Σ_l b_{j_l} * d_l ≤ b_max * d

    Using greedy approach: iteratively upgrade the layer with best marginal
    benefit (PPL improvement per bit spent).

    Args:
        alphas: Dict mapping layer_idx to α_l coefficient
        error_table: Dict mapping layer_idx -> option_idx -> mse
        total_elements: Dict mapping layer_idx to num_elements
        options: List of quantization options (sorted low to high bits)
        target_avg_bits: Target average bits per element

    Returns:
        Dict with assignment, total_bits, expected_ppl_degradation, etc.
    """
    print("=" * 60)
    print("Optimizing bitwidth assignment")
    print("=" * 60)
    print(f"Target average bits: {target_avg_bits:.3f}")
    print(f"Options: {[opt['config_str'] for opt in options]}")

    n_layers = len(alphas)
    layer_indices = sorted(alphas.keys())

    # Sort options by bits (low to high)
    sorted_option_indices = list(range(len(options)))
    sorted_option_indices.sort(key=lambda i: options[i]["bits_per_entry"])

    # Start with lowest bitwidth for all layers
    assignment = {l: sorted_option_indices[0] for l in layer_indices}

    # Compute total elements and current bits
    d_total = sum(total_elements[l] for l in layer_indices)
    current_bits = sum(
        options[assignment[l]]["bits_per_entry"] * total_elements[l]
        for l in layer_indices
    )
    target_bits = target_avg_bits * d_total

    print(f"\nTotal elements: {d_total:,}")
    print(f"Target total bits: {target_bits:,.0f}")
    print(f"Starting bits (all {options[sorted_option_indices[0]]['config_str']}): "
          f"{current_bits:,.0f} ({current_bits/d_total:.3f} avg)")

    # Iteratively upgrade layers
    iterations = 0
    max_iterations = n_layers * len(options) * 2

    while current_bits < target_bits and iterations < max_iterations:
        iterations += 1

        best_upgrade = None
        best_ratio = -float('inf')

        for l in layer_indices:
            curr_opt_idx = assignment[l]
            # Find next higher option
            curr_pos = sorted_option_indices.index(curr_opt_idx)
            if curr_pos >= len(sorted_option_indices) - 1:
                continue  # Already at max

            next_opt_idx = sorted_option_indices[curr_pos + 1]

            # Marginal PPL improvement
            curr_mse = error_table[l][curr_opt_idx]
            next_mse = error_table[l][next_opt_idx]
            delta_ppl = alphas[l] * (curr_mse - next_mse)

            # Marginal bit cost
            curr_bits = options[curr_opt_idx]["bits_per_entry"]
            next_bits = options[next_opt_idx]["bits_per_entry"]
            delta_bits = (next_bits - curr_bits) * total_elements[l]

            if delta_bits <= 0:
                continue

            # Benefit per bit
            ratio = delta_ppl / delta_bits

            if ratio > best_ratio:
                best_ratio = ratio
                best_upgrade = (l, next_opt_idx, delta_ppl, delta_bits)

        if best_upgrade is None:
            print("No more upgrades possible")
            break

        l_upgrade, new_opt_idx, delta_ppl, delta_bits = best_upgrade
        old_opt_idx = assignment[l_upgrade]

        # Check if upgrade would exceed budget
        if current_bits + delta_bits > target_bits * 1.01:  # 1% tolerance
            # Try to find a smaller upgrade
            found = False
            for l in layer_indices:
                curr_opt_idx = assignment[l]
                curr_pos = sorted_option_indices.index(curr_opt_idx)

                for next_pos in range(curr_pos + 1, len(sorted_option_indices)):
                    next_opt_idx = sorted_option_indices[next_pos]
                    curr_bits = options[curr_opt_idx]["bits_per_entry"]
                    next_bits = options[next_opt_idx]["bits_per_entry"]
                    test_delta_bits = (next_bits - curr_bits) * total_elements[l]

                    if current_bits + test_delta_bits <= target_bits * 1.01:
                        # Use this upgrade instead
                        curr_mse = error_table[l][curr_opt_idx]
                        next_mse = error_table[l][next_opt_idx]
                        test_delta_ppl = alphas[l] * (curr_mse - next_mse)

                        l_upgrade = l
                        new_opt_idx = next_opt_idx
                        delta_ppl = test_delta_ppl
                        delta_bits = test_delta_bits
                        found = True
                        break

                if found:
                    break

            if not found:
                print(f"\nStopping: would exceed budget (current: {current_bits:,.0f}, "
                      f"target: {target_bits:,.0f})")
                break

        # Apply upgrade
        old_opt_idx = assignment[l_upgrade]
        assignment[l_upgrade] = new_opt_idx
        current_bits += delta_bits

        print(f"  [{iterations}] Layer {l_upgrade}: {options[old_opt_idx]['config_str']} → "
              f"{options[new_opt_idx]['config_str']} "
              f"(+{delta_bits:,.0f} bits, ΔPPL={delta_ppl:.6f})")

    # Compute final statistics
    final_avg_bits = current_bits / d_total
    expected_ppl_degradation = sum(
        alphas[l] * error_table[l][assignment[l]]
        for l in layer_indices
    )

    print(f"\n{'='*60}")
    print("Optimization complete")
    print(f"{'='*60}")
    print(f"Final average bits: {final_avg_bits:.3f} (target: {target_avg_bits:.3f})")
    print(f"Expected PPL degradation: {expected_ppl_degradation:.6f}")
    print(f"\nFinal assignment:")

    # Count layers per option
    option_counts = {}
    for l in layer_indices:
        opt_idx = assignment[l]
        opt_str = options[opt_idx]["config_str"]
        option_counts[opt_str] = option_counts.get(opt_str, 0) + 1

    for opt_str, count in sorted(option_counts.items()):
        print(f"  {opt_str}: {count} layers")

    return {
        "assignment": assignment,
        "total_bits": current_bits,
        "total_elements": d_total,
        "avg_bits": final_avg_bits,
        "target_bits": target_avg_bits,
        "expected_ppl_degradation": expected_ppl_degradation,
        "option_counts": option_counts,
        "iterations": iterations,
    }


# ============================================================
# Evaluation: Measure actual PPL with assignment
# ============================================================

def evaluate_assignment(
    model: nn.Module,
    tokenizer,
    assignment: Dict[int, int],
    options: List[Dict],
    device: str = "cuda",
    seqlen: int = 2048,
) -> float:
    """Evaluate perplexity with the given bitwidth assignment.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        assignment: Dict mapping layer_idx -> option_idx
        options: List of quantization option dicts
        device: Device to run on
        seqlen: Sequence length for evaluation

    Returns:
        Perplexity on wikitext2
    """
    from datasets import load_dataset

    print("\n" + "="*60)
    print("Evaluating bitwidth assignment")
    print("="*60)

    # Get linear layers
    linear_layers = []
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "embed" not in name and "lm_head" not in name:
            linear_layers.append(module)
            layer_names.append(name)

    print(f"Applying quantization to {len(linear_layers)} layers...")

    # Build codebooks for each option
    codebooks = {}
    for opt in options:
        k = opt['k']
        p = opt['p']
        key = f"k{k}p{p}"
        # Uniform codebook from -1 to 1
        indices = torch.arange(2**k, dtype=torch.float32)
        codebook = (indices / (2**k - 1) * 2 - 1) if 2**k > 1 else torch.zeros(1)
        codebooks[key] = codebook.to(device)

    # Apply quantization hooks
    hooks = []
    for layer_idx, module in enumerate(linear_layers):
        if layer_idx in assignment:
            opt_idx = assignment[layer_idx]
            opt = options[opt_idx]
            k = opt['k']
            p = opt['p']
            codebook = codebooks[f"k{k}p{p}"]

            # Create hook
            def make_hook(mod, cb, kk, pp):
                def hook(mod, input, output):
                    # Quantize and dequantize weights
                    w = mod.weight.data
                    # Simple quantization for evaluation
                    w_flat = w.reshape(-1, 1)
                    # Find nearest codebook entry
                    dists = (w_flat.unsqueeze(1) - cb.unsqueeze(0)) ** 2
                    indices = dists.argmin(dim=1)
                    w_q = cb[indices].reshape(w.shape)
                    mod.weight.data = w_q
                    return output
                return hook

            handle = module.register_forward_hook(make_hook(module, codebook, k, p))
            hooks.append(handle)

    # Load wikitext2
    print("\nLoading wikitext2 dataset...")
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except:
        # Fallback: use random data
        print("Could not load wikitext2, using random tokens for evaluation")
        n_samples = 100
        input_ids = torch.randint(0, len(tokenizer), (n_samples, seqlen), device=device)
    else:
        # Tokenize
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

    # Evaluate perplexity
    print(f"Evaluating perplexity (seqlen={seqlen})...")
    model.eval()

    max_length = min(seqlen, 2048)
    stride = 512
    nlls = []

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_length, stride):
            begin_loc = i
            end_loc = min(i + max_length, input_ids.size(1))
            trg_len = end_loc - begin_loc

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_chunk = input_ids[:, begin_loc+1:end_loc+1]

            outputs = model(input_chunk, labels=input_chunk)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

            if len(nlls) >= 20:  # Limit eval for speed
                break

    ppl = torch.exp(torch.stack(nlls).sum() / sum(nlls))

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\nPerplexity: {ppl.item():.2f}")
    print("="*60)

    return ppl.item()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="HIGGS Dynamic Bitwidth Quantization")

    # Model args
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype (default: auto)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for calibration")

    # Step selection
    parser.add_argument("--calibrate", action="store_true",
                        help="Run Algorithm 3 to estimate α_l coefficients")
    parser.add_argument("--compute-error-table", action="store_true",
                        help="Compute quantization error table t²_{l,j}")
    parser.add_argument("--optimize", action="store_true",
                        help="Run bitwidth assignment optimization")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate bitwidth assignment (measure actual PPL)")
    parser.add_argument("--assignment-path", type=str, default=None,
                        help="Path to load bitwidth assignment for evaluation")

    # Calibration args
    parser.add_argument("--calibration-tokens", type=int, default=287000,
                        help="Number of random tokens for calibration (default: 287k)")
    parser.add_argument("--n-noise-levels", type=int, default=15,
                        help="Number of noise levels J (default: 15)")
    parser.add_argument("--t-min", type=float, default=0.001,
                        help="Minimum noise std")
    parser.add_argument("--t-max", type=float, default=0.05,
                        help="Maximum noise std")
    parser.add_argument("--output-alpha", type=str, default="alphas.json",
                        help="Path to save α_l coefficients")
    parser.add_argument("--alpha-path", type=str, default=None,
                        help="Path to load α_l coefficients")

    # Error table args
    parser.add_argument("--error-table-path", type=str, default="error_table.json",
                        help="Path to save/load error table")
    parser.add_argument("--norm", default="l2", choices=["l2", "absmax"],
                        help="Normalization type (default: l2)")
    parser.add_argument("--blocksize", type=int, default=32,
                        help="Block size for absmax norm")
    parser.add_argument("--rot-blocksize", type=int, default=128,
                        help="Rotation block size")

    # Optimization args
    parser.add_argument("--target-bits", type=float, default=3.0,
                        help="Target average bits per element")
    parser.add_argument("--bitwidth-options", type=str,
                        default="k=2,p=2;k=3,p=2;k=4,p=2",
                        help="Semicolon-separated quantization options")
    parser.add_argument("--output-assignment", type=str, default="assignment.json",
                        help="Path to save bitwidth assignment")

    # General args
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("HIGGS Dynamic Bitwidth Quantization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Host: {socket.gethostname()}")
    print("=" * 60)

    # Check if we can skip model loading (optimize-only mode)
    optimize_only = args.optimize and not args.calibrate and not args.compute_error_table
    have_alpha = args.alpha_path and os.path.exists(args.alpha_path)
    have_error_table = args.error_table_path and os.path.exists(args.error_table_path)

    model = None
    tokenizer = None

    if optimize_only and have_alpha and have_error_table:
        print("\nOptimization-only mode: skipping model loading (using pre-computed files)")
    else:
        # Load model
        print("\nLoading model...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype_map[args.dtype],
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {}

    # Step 1: Calibrate α_l coefficients
    if args.calibrate:
        alphas = estimate_alpha_coefficients(
            model=model,
            tokenizer=tokenizer,
            n_tokens=args.calibration_tokens,
            seq_len=args.seqlen,
            n_noise_levels=args.n_noise_levels,
            t_min=args.t_min,
            t_max=args.t_max,
            device=device,
            seed=args.seed,
        )

        # Save alphas
        alpha_data = {
            "alphas": {str(k): v for k, v in alphas.items()},
            "n_layers": len(alphas),
            "calibration_tokens": args.calibration_tokens,
            "n_noise_levels": args.n_noise_levels,
            "t_min": args.t_min,
            "t_max": args.t_max,
            "seed": args.seed,
        }

        Path(args.output_alpha).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_alpha, "w") as f:
            json.dump(alpha_data, f, indent=2)
        print(f"\nα_l coefficients saved to {args.output_alpha}")

        results["alphas"] = alpha_data

    # Step 2: Compute quantization error table
    if args.compute_error_table:
        options = get_quantization_options(
            args.bitwidth_options, norm=args.norm,
            blocksize=args.blocksize, rot_blocksize=args.rot_blocksize)

        error_table_data = compute_quantization_error_table(
            model=model,
            options=options,
            blocksize=args.blocksize,
            rot_blocksize=args.rot_blocksize,
            norm=args.norm,
            seed=args.seed,
        )

        # Save error table
        save_data = {
            "error_table": {
                str(k): {str(k2): v2 for k2, v2 in v.items()}
                for k, v in error_table_data["error_table"].items()
            },
            "total_elements": {str(k): v for k, v in error_table_data["total_elements"].items()},
            "options": error_table_data["options"],
            "norm": error_table_data["norm"],
            "blocksize": error_table_data["blocksize"],
            "rot_blocksize": error_table_data["rot_blocksize"],
        }

        Path(args.error_table_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.error_table_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nError table saved to {args.error_table_path}")

        results["error_table"] = save_data

    # Step 3: Optimize bitwidth assignment
    if args.optimize:
        # Load alphas if not just computed
        if args.alpha_path:
            print(f"\nLoading α_l coefficients from {args.alpha_path}...")
            with open(args.alpha_path, "r") as f:
                alpha_data = json.load(f)
            alphas = {int(k): v for k, v in alpha_data["alphas"].items()}
        elif "alphas" not in results:
            raise ValueError("Must provide --alpha-path or run --calibrate first")
        else:
            alphas = {int(k): v for k, v in results["alphas"]["alphas"].items()}

        # Load error table if not just computed
        if args.error_table_path:
            print(f"Loading error table from {args.error_table_path}...")
            with open(args.error_table_path, "r") as f:
                error_table_data = json.load(f)
            error_table = {
                int(k): {int(k2): v2 for k2, v2 in v.items()}
                for k, v in error_table_data["error_table"].items()
            }
            total_elements = {
                int(k): v for k, v in error_table_data["total_elements"].items()
            }
            options = error_table_data["options"]
        elif "error_table" not in results:
            raise ValueError("Must provide --error-table-path or run --compute-error-table first")
        else:
            error_table = results["error_table"]["error_table"]
            total_elements = results["error_table"]["total_elements"]
            options = results["error_table"]["options"]

        # Run optimization
        optimization_result = optimize_bitwidth_assignment(
            alphas=alphas,
            error_table=error_table,
            total_elements=total_elements,
            options=options,
            target_avg_bits=args.target_bits,
        )

        # Save assignment
        save_data = {
            "assignment": {str(k): v for k, v in optimization_result["assignment"].items()},
            "total_bits": optimization_result["total_bits"],
            "total_elements": optimization_result["total_elements"],
            "avg_bits": optimization_result["avg_bits"],
            "target_bits": optimization_result["target_bits"],
            "expected_ppl_degradation": optimization_result["expected_ppl_degradation"],
            "option_counts": optimization_result["option_counts"],
            "iterations": optimization_result["iterations"],
            "options": options,
        }

        Path(args.output_assignment).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_assignment, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nBitwidth assignment saved to {args.output_assignment}")

        results["optimization"] = optimization_result

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
