#!/usr/bin/env python3
# Set shared HF cache before any HuggingFace imports
import os as _os
_shared_hf = "/data/hf_cache"
if _os.path.isdir(_shared_hf):
    _os.environ.setdefault("HF_HUB_CACHE", f"{_shared_hf}/hub")
    _os.environ.setdefault("HF_DATASETS_CACHE", f"{_shared_hf}/datasets")
del _shared_hf

"""Perplexity evaluation matching the standard GPTQ/SpQR/QuIP/HIGGS procedure.

This is the CORRECT script for WikiText-2 perplexity evaluation.
Computes token-level perplexity using non-overlapping chunks of seqlen
tokens, which is the de facto standard in quantization papers.

bnf_eval.py (lm-eval-harness) was removed — it reported word_perplexity,
an incomparable metric, and used a non-max-only codebook that underperforms.

HIGGS reference (Llama 3.1 8B, WikiText-2, seqlen=8192):
  FP16 baseline:  5.607
  HIGGS 3.25-bit: 7.110 (p=2), 6.807 (p=3)

Supports both scalar (p=1) and vector quantization (p=2,3,4).

Usage:
    # FP16 baseline
    python eval_ppl.py --model meta-llama/Llama-2-7b-hf --seqlen 4096

    # BNF-4 scalar (p=1, 4.25 bits)
    python eval_ppl.py --model meta-llama/Llama-3.1-8B --seqlen 8192 \
        --method bnf --k 4 --p 1

    # BNF VQ p=2, 4 bits/element (8-bit index, 256 entries, 4.25 bits total)
    python eval_ppl.py --model meta-llama/Llama-3.1-8B --seqlen 8192 \
        --method bnf --k 4 --p 2

    # BNF VQ p=4, 3 bits/element (12-bit index, 4096 entries, 3.25 bits total)
    python eval_ppl.py --model meta-llama/Llama-3.1-8B --seqlen 8192 \
        --method bnf --k 3 --p 4
"""

import argparse
import json
import random
import socket
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from pathlib import Path
from scipy.linalg import hadamard
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Data loading — copied from GPTQ/SpQR datautils.py
# ============================================================

def get_wikitext2(seqlen, tokenizer):
    """WikiText-2-raw-v1, test split. Concatenate with "\\n\\n", tokenize."""
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc.input_ids


def get_c4(seqlen, tokenizer):
    """C4 validation shard. Sample 256 random docs >= seqlen (seed=0),
    extract random seqlen-length chunk from each, concatenate."""
    try:
        # Older datasets library
        valdata = load_dataset(
            "allenai/c4", "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    except (ValueError, Exception):
        # Newer datasets library
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            trust_remote_code=True,
        )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    return torch.hstack(valenc)


DATASET_LOADERS = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
}


# ============================================================
# Block-normalized sample generation
# ============================================================

def generate_block_normalized_samples(blocksize, n_samples=30000, dim=512,
                                      device="cpu"):
    rng = torch.Generator(device=device).manual_seed(0)
    W = torch.randn(n_samples, dim, device=device, generator=rng)
    blocks = W.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
    normalized = blocks / absmax
    return normalized.flatten()


# ============================================================
# Scalar quantization (p=1): Lloyd-Max
# ============================================================

def lloyd_max(values, n_codewords, max_iter=500, force_boundary=None):
    device = values.device
    n = values.numel()
    sorted_vals = values.sort().values
    indices = torch.linspace(0, n - 1, n_codewords + 2, device=device).long()[1:-1]
    codewords = sorted_vals[indices].clone()

    for _ in range(max_iter):
        if force_boundary is not None:
            if force_boundary > 0:
                codewords[-1] = force_boundary
            else:
                codewords[0] = force_boundary
        midpoints = (codewords[:-1] + codewords[1:]) / 2
        assignments = torch.bucketize(values, midpoints)
        new_codewords = torch.zeros_like(codewords)
        for i in range(n_codewords):
            mask = assignments == i
            if mask.sum() > 0:
                new_codewords[i] = values[mask].mean()
            else:
                new_codewords[i] = codewords[i]
        shift = (new_codewords - codewords).abs().max().item()
        codewords = new_codewords
        if shift < 1e-8:
            break
    if force_boundary is not None:
        if force_boundary > 0:
            codewords[-1] = force_boundary
        else:
            codewords[0] = force_boundary
    return codewords.sort().values


def compute_scalar_codebook(k, blocksize, device="cpu"):
    """Compute scalar (p=1) BNF codebook via Lloyd-Max.

    Returns:
        cb: raw Lloyd-Max codebook (for quantization)
        cb_bnf: kappa-corrected codebook (for dequantization, standard method)
        kappa1: standard attenuation factor E[qr]/E[r²]
        kappa_stats: dict with E[qr], E[r²], E[q²] for alternative kappa methods
    """
    n = 1 << k
    n_half = n // 2
    vals = generate_block_normalized_samples(blocksize, device=device)
    pos_vals = vals[vals > 0]
    pos_cw = lloyd_max(pos_vals, n_half)
    cb = torch.sort(torch.cat([-pos_cw, pos_cw])).values
    midpoints = (cb[:-1] + cb[1:]) / 2
    indices = torch.bucketize(vals, midpoints)
    q_vals = cb[indices]
    e_qr = (q_vals * vals).mean().item()
    e_r2 = (vals ** 2).mean().item()
    e_q2 = (q_vals ** 2).mean().item()
    kappa1 = e_qr / e_r2 if e_r2 > 1e-12 else 1.0
    cb_bnf = cb / kappa1
    kappa_stats = {"e_qr": e_qr, "e_r2": e_r2, "e_q2": e_q2}
    return cb, cb_bnf, kappa1, kappa_stats


def compute_gaussian_grid(k, p, n_samples=100000, device="cpu"):
    """Compute Gaussian-optimized grid for L2 normalization.

    L2 normalization (after Hadamard) produces approximately N(0, 1) values.
    This grid is trained on Gaussian samples, not block-normalized samples.

    Returns:
        quant_cb: codebook for quantization (k-means on Gaussian)
        deq_cb: kappa-corrected codebook for dequantization
        kappa1: attenuation factor
        kappa_stats: dict with E[qr], E[r²], E[q²]
    """
    index_bits = k * p
    n_entries = 1 << index_bits

    # Generate Gaussian samples (L2 normalized distribution)
    rng = torch.Generator(device=device).manual_seed(0)
    samples = torch.randn(n_samples, p, device=device, generator=rng)

    print(f"    Training k-means on {n_samples} N(0,1) {p}D samples...",
          flush=True)

    # Run k-means
    cb = kmeans(samples, n_entries, max_iter=300, seed=0)

    # Compute kappa on Gaussian samples
    dists = torch.cdist(samples.float(), cb.float())
    assignments = dists.argmin(dim=1)
    q_samples = cb[assignments]

    e_qr = (q_samples * samples).sum(dim=1).mean().item()
    e_r2 = (samples ** 2).sum(dim=1).mean().item()
    e_q2 = (q_samples ** 2).sum(dim=1).mean().item()
    kappa1 = e_qr / e_r2 if e_r2 > 1e-12 else 1.0

    cb_corrected = cb / kappa1

    kappa_stats = {"e_qr": e_qr, "e_r2": e_r2, "e_q2": e_q2}

    print(f"    Gaussian grid: k={k}, p={p}, entries={n_entries}", flush=True)
    print(f"    kappa1 = {kappa1:.6f}", flush=True)

    return cb, cb_corrected, kappa1, kappa_stats


def compute_scalar_codebook_theory_correct(k, blocksize, device="cpu"):
    """Compute theory-correct scalar BNF codebook with pinned ±1 boundaries.

    THEORY.md Section 3-6: The max element (at ±1) has zero bias (κ₁=1).
    The non-max elements have κ₁^non-max < 1. We train the codebook on
    non-max samples only, pin outermost at ±1, and apply kappa correction
    only to inner entries.

    Returns:
        quant_cb: codebook for quantization (±1 pinned, inner from Lloyd-Max)
        deq_cb: codebook for dequantization (±1 at ±1, inner / κ_nonmax)
        kappa_nonmax: attenuation factor for non-max elements
        kappa_stats: dict with E[qr], E[r²], E[q²] on non-max samples
    """
    n = 1 << k
    n_inner = n - 2  # Excluding the ±1 endpoints
    n_half_inner = n_inner // 2

    vals = generate_block_normalized_samples(blocksize, device=device)

    # Separate max and non-max elements
    is_max = vals.abs() > 0.999  # ~±1 after absmax normalization
    nonmax_vals = vals[~is_max]
    max_vals = vals[is_max]

    print(f"    Blocksize={blocksize}: {is_max.sum().item()}/{len(vals)} "
          f"({100*is_max.float().mean():.1f}%) are max elements", flush=True)

    # Train Lloyd-Max on non-max positive values only
    pos_nonmax = nonmax_vals[nonmax_vals > 0]
    pos_inner = lloyd_max(pos_nonmax, n_half_inner)

    # Build quantization codebook: [-1, -inner, +inner, +1]
    quant_cb = torch.sort(torch.cat([
        torch.tensor([-1.0], device=device),
        -pos_inner,
        pos_inner,
        torch.tensor([1.0], device=device)
    ])).values

    # Compute κ₁ on non-max samples only (matching THEORY.md)
    midpoints = (quant_cb[:-1] + quant_cb[1:]) / 2
    indices = torch.bucketize(nonmax_vals, midpoints)
    q_vals = quant_cb[indices]

    e_qr = (q_vals * nonmax_vals).mean().item()
    e_r2 = (nonmax_vals ** 2).mean().item()
    e_q2 = (q_vals ** 2).mean().item()
    kappa_nonmax = e_qr / e_r2 if e_r2 > 1e-12 else 1.0

    # Build dequantization codebook:
    # - ±1 entries stay at ±1 (max element has zero error, κ₁=1)
    # - Inner entries get divided by κ_nonmax
    deq_cb = quant_cb.clone()
    for i in range(len(deq_cb)):
        if quant_cb[i].abs() < 0.999:  # Not the ±1 entry
            deq_cb[i] = quant_cb[i] / kappa_nonmax

    # Verify max element reconstruction
    max_indices = torch.bucketize(max_vals, midpoints)
    max_qvals = quant_cb[max_indices]
    print(f"    Max elements: quantized to {max_qvals.unique().tolist()}, "
          f"will dequantize to ±1.0", flush=True)

    kappa_stats = {
        "e_qr": e_qr,
        "e_r2": e_r2,
        "e_q2": e_q2,
        "kappa_nonmax": kappa_nonmax,
        "n_nonmax": len(nonmax_vals),
        "n_max": len(max_vals),
    }

    return quant_cb, deq_cb, kappa_nonmax, kappa_stats


# ============================================================
# Vector quantization (p>=2): k-means
# ============================================================

def _chunked_nearest(data, centroids, chunk_size=50000):
    """Memory-efficient nearest centroid search. Returns indices."""
    n = data.shape[0]
    assignments = torch.empty(n, dtype=torch.long, device=data.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(data[start:end].float(), centroids.float())
        assignments[start:end] = dists.argmin(dim=1)
    return assignments


def kmeans(data, n_clusters, max_iter=300, seed=0):
    """K-means clustering on GPU with chunked distance computation.

    Args:
        data: (N, p) tensor of training samples
        n_clusters: number of centroids
        max_iter: maximum iterations
        seed: random seed for initialization

    Returns:
        centroids: (n_clusters, p) tensor
    """
    device = data.device
    n, p = data.shape

    # Initialize: random subset
    rng = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(n, generator=rng, device=device)[:n_clusters]
    centroids = data[perm].clone().float()

    for iteration in range(max_iter):
        # Assign to nearest centroid (chunked for memory)
        assignments = _chunked_nearest(data, centroids)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=device)
        # Use scatter_add for efficiency
        for d in range(p):
            new_centroids[:, d].scatter_add_(
                0, assignments, data[:, d].float())
        counts.scatter_add_(0, assignments,
                            torch.ones(n, device=device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]

        shift = (new_centroids - centroids).norm(dim=1).max().item()
        centroids = new_centroids

        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"    k-means iter {iteration+1}/{max_iter}, "
                  f"max shift = {shift:.6f}", flush=True)

        if shift < 1e-7:
            print(f"    k-means converged at iter {iteration+1}", flush=True)
            break

    return centroids


def compute_vq_codebook(k, p, blocksize, device="cpu"):
    """Compute p-dimensional VQ codebook via k-means on block-normalized samples.

    Args:
        k: bits per element (index_bits = k * p)
        p: vector dimension (number of elements quantized together)
        blocksize: block normalization size (must be divisible by p)

    Returns:
        quant_cb: (n_entries, p) codebook for quantization
        deq_cb: (n_entries, p) kappa-corrected codebook for dequantization
        kappa1: attenuation factor
    """
    index_bits = k * p
    n_entries = 1 << index_bits

    print(f"    VQ codebook: p={p}, k={k}, index_bits={index_bits}, "
          f"entries={n_entries}", flush=True)

    # Generate block-normalized samples and reshape into p-tuples
    vals = generate_block_normalized_samples(blocksize, device=device)
    # Reshape blocks into p-tuples (drop remainder if blocksize not divisible by p)
    blocks = vals.reshape(-1, blocksize)
    elems_per_block = (blocksize // p) * p
    trimmed = blocks[:, :elems_per_block]
    tuples = trimmed.reshape(-1, p)

    # Subsample if too many points (cap at ~2M tuples for tractability)
    max_tuples = 2_000_000
    if tuples.shape[0] > max_tuples:
        rng = torch.Generator(device=device).manual_seed(42)
        perm = torch.randperm(tuples.shape[0], generator=rng,
                              device=device)[:max_tuples]
        tuples = tuples[perm]

    print(f"    Training on {tuples.shape[0]:,} {p}D samples...", flush=True)

    # Run k-means
    t0 = time.time()
    cb = kmeans(tuples, n_entries, max_iter=300, seed=0)
    print(f"    k-means done in {time.time() - t0:.1f}s", flush=True)

    # Compute kappa stats: E[<Q(r), r>], E[||r||²], E[||Q(r)||²]
    assignments = _chunked_nearest(tuples, cb)
    q_tuples = cb[assignments]
    e_qr = (q_tuples * tuples.float()).sum(dim=1).mean().item()
    e_r2 = (tuples.float() ** 2).sum(dim=1).mean().item()
    e_q2 = (q_tuples ** 2).sum(dim=1).mean().item()
    kappa1 = e_qr / e_r2 if e_r2 > 1e-12 else 1.0

    cb_corrected = cb / kappa1
    kappa_stats = {"e_qr": e_qr, "e_r2": e_r2, "e_q2": e_q2}
    return cb, cb_corrected, kappa1, kappa_stats


def compute_per_entry_kappa(quant_cb, k, p, blocksize, device="cpu"):
    """Compute per-entry kappa corrections for any codebook (scalar or VQ).

    For each codebook entry i, computes:
        κ_i = E[<c_i, r> | r in cell_i] / E[||r||² | r in cell_i]
            = ||c_i||² / (||c_i||² + within-cell variance_i)

    Entries near the boundaries (serving max-containing pairs) get κ_i ≈ 1
    (minimal correction). Interior entries get κ_i < 1 (more correction).

    Returns:
        deq_cb: per-entry corrected dequantization codebook (same shape as quant_cb)
        kappa_per_entry: 1D tensor of per-entry κ values
        stats: dict with summary statistics
    """
    # Regenerate training data (same seed as codebook training)
    vals = generate_block_normalized_samples(blocksize, device=device)

    if p == 1:
        # Scalar: vals is 1D
        midpoints = (quant_cb[:-1] + quant_cb[1:]) / 2
        assignments = torch.bucketize(vals, midpoints)
        n_entries = quant_cb.shape[0]

        kappa_per_entry = torch.ones(n_entries, device=device)
        deq_cb = quant_cb.clone().float()

        for i in range(n_entries):
            mask = assignments == i
            if mask.sum() < 2:
                continue
            cell_vals = vals[mask]
            c_i = quant_cb[i].float()
            # κ_i = E[c_i * r | cell_i] / E[r² | cell_i]
            #     = c_i * E[r | cell_i] / E[r² | cell_i]
            # Since c_i = E[r | cell_i] (Lloyd-Max property):
            #     = c_i² / E[r² | cell_i]
            e_r2_i = (cell_vals ** 2).mean().item()
            if e_r2_i > 1e-12:
                kappa_per_entry[i] = (c_i.item() ** 2) / e_r2_i
                deq_cb[i] = c_i / kappa_per_entry[i]

        deq_cb = deq_cb.to(quant_cb.dtype)
    else:
        # VQ: reshape into p-tuples
        blocks = vals.reshape(-1, blocksize)
        elems_per_block = (blocksize // p) * p
        trimmed = blocks[:, :elems_per_block]
        tuples = trimmed.reshape(-1, p)

        max_tuples = 2_000_000
        if tuples.shape[0] > max_tuples:
            rng = torch.Generator(device=device).manual_seed(42)
            perm = torch.randperm(tuples.shape[0], generator=rng,
                                  device=device)[:max_tuples]
            tuples = tuples[perm]

        assignments = _chunked_nearest(tuples, quant_cb)
        n_entries = quant_cb.shape[0]

        kappa_per_entry = torch.ones(n_entries, device=device)
        deq_cb = quant_cb.clone().float()

        for i in range(n_entries):
            mask = assignments == i
            if mask.sum() < 2:
                continue
            cell_vals = tuples[mask].float()
            c_i = quant_cb[i].float()
            # κ_i = E[<c_i, r> | cell_i] / E[||r||² | cell_i]
            #     = ||c_i||² / E[||r||² | cell_i]
            e_r2_i = (cell_vals ** 2).sum(dim=1).mean().item()
            c_i_norm2 = (c_i ** 2).sum().item()
            if e_r2_i > 1e-12 and c_i_norm2 > 1e-12:
                kappa_per_entry[i] = c_i_norm2 / e_r2_i
                deq_cb[i] = c_i / kappa_per_entry[i]

        deq_cb = deq_cb.to(quant_cb.dtype)

    stats = {
        "kappa_min": kappa_per_entry.min().item(),
        "kappa_max": kappa_per_entry.max().item(),
        "kappa_mean": kappa_per_entry.mean().item(),
        "kappa_std": kappa_per_entry.std().item(),
    }
    return deq_cb, kappa_per_entry, stats


# ============================================================
# Unified codebook computation
# ============================================================

def compute_codebook(k, p, blocksize, device="cpu", cache_dir=None):
    """Compute codebook for given k (bits/element) and p (vector dimension).

    Caches results to disk under cache_dir (default: script_dir/codebook_cache/).
    Subsequent calls with the same (k, p, blocksize) load from cache.

    Returns:
        quant_cb: codebook for bin assignment
            p=1: (2^k,) 1D tensor
            p>=2: (2^(k*p), p) 2D tensor
        deq_cb: kappa-corrected codebook for dequantization (same shape)
        kappa1: attenuation factor
        kappa_stats: dict with E[qr], E[r²], E[q²]
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "codebook_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"cb_k{k}_p{p}_g{blocksize}.pt"

    if cache_file.exists():
        print(f"    Loading cached codebook from {cache_file}", flush=True)
        data = torch.load(cache_file, map_location=device, weights_only=True)
        kappa_stats = data.get("kappa_stats", {})
        return data["quant_cb"], data["deq_cb"], data["kappa1"], kappa_stats

    if p == 1:
        quant_cb, deq_cb, kappa1, kappa_stats = compute_scalar_codebook(
            k, blocksize, device)
    else:
        quant_cb, deq_cb, kappa1, kappa_stats = compute_vq_codebook(
            k, p, blocksize, device)

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"quant_cb": quant_cb.cpu(), "deq_cb": deq_cb.cpu(),
                "kappa1": kappa1, "kappa_stats": kappa_stats}, cache_file)
    print(f"    Codebook cached to {cache_file}", flush=True)

    return quant_cb, deq_cb, kappa1, kappa_stats


# ============================================================
# Hadamard rotation
# ============================================================

_active_hooks = []
_hooks_enabled = True


def set_hooks_enabled(enabled):
    global _hooks_enabled
    _hooks_enabled = enabled


def make_hadamard_block(rot_blocksize, device):
    H = hadamard(rot_blocksize).astype(np.float32) / np.sqrt(rot_blocksize)
    return torch.from_numpy(H).float().to(device)


def make_random_signs(max_dim, seed, device):
    rng = torch.Generator().manual_seed(seed)
    signs = torch.sign(torch.randn(max_dim, generator=rng)).to(device)
    signs[signs == 0] = 1.0
    return signs


# ============================================================
# Quantization hooks
# ============================================================

def load_grid_file(filepath):
    """Load a Pagès-Printems optimal grid file.

    Format: N+1 rows, d+3 columns.
    Rows 1..N: probability, d coordinates, local L2/L1 distortion.
    Row N+1: summary (zeros + total distortion).

    Returns:
        centroids: (N, d) tensor of grid points
    """
    data = np.loadtxt(filepath)
    N = data.shape[0] - 1
    d = data.shape[1] - 3
    centroids = data[:N, 1:1+d]
    total_l2 = data[N, 1+d]
    print(f"    Loaded grid: {filepath}", flush=True)
    print(f"    N={N}, d={d}, published L2 distortion={total_l2:.10f}",
          flush=True)
    return torch.from_numpy(centroids).float()


def install_bnf_hooks(model, quant_codebook, deq_codebook, blocksize, seed,
                      device, p=1, rot_blocksize=128):
    """Install forward hooks that quantize weights on-the-fly.

    Uses absmax block normalization (BNF approach).
    Supports both scalar (p=1) and vector (p>=2) quantization.
    """
    global _active_hooks
    remove_hooks()

    H_block = make_hadamard_block(rot_blocksize, device)
    max_in = max(m.weight.shape[1] for m in model.modules()
                 if isinstance(m, nn.Linear))
    signs = make_random_signs(max_in, seed, device)

    qcb = quant_codebook.to(device)
    dcb = deq_codebook.to(device)

    if p == 1:
        # Precompute midpoints for scalar bucketize
        midpoints = (qcb[:-1] + qcb[1:]) / 2

    def make_pre_hook(qcb_t, dcb_t, do_rotate):
        def hook(mod, args_):
            if not _hooks_enabled:
                return
            mod._orig_weight = mod.weight.data.clone()
            W = mod.weight.data.float()
            in_dim = W.shape[1]
            if do_rotate and in_dim % rot_blocksize == 0:
                for b in range(in_dim // rot_blocksize):
                    s, e = b * rot_blocksize, (b + 1) * rot_blocksize
                    W[:, s:e] = (W[:, s:e] * signs[s:e].unsqueeze(0)) @ H_block.T
            flat = W.flatten()
            n = flat.numel()
            pad_n = (blocksize - n % blocksize) % blocksize
            if pad_n > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_n))
            blocks = flat.reshape(-1, blocksize)
            absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
            normalized = blocks / absmax

            if p == 1:
                # Scalar quantization
                indices = torch.bucketize(normalized, midpoints)
                dequantized = (dcb_t[indices] * absmax)
            else:
                # Vector quantization: reshape into p-tuples
                n_blocks = normalized.shape[0]
                elems_per_block = (blocksize // p) * p
                remainder = blocksize - elems_per_block
                # Split into VQ-able part and remainder
                vq_part = normalized[:, :elems_per_block]
                groups = vq_part.reshape(-1, p)
                indices = _chunked_nearest(groups, qcb_t, chunk_size=100000)
                dq_groups = dcb_t[indices]  # (n_groups, p)
                dq_vq = dq_groups.reshape(n_blocks, elems_per_block)
                if remainder > 0:
                    # Keep remainder elements unquantized (normalized)
                    rem_part = normalized[:, elems_per_block:]
                    dequantized = torch.cat([dq_vq, rem_part], dim=1) * absmax
                else:
                    dequantized = dq_vq * absmax

            dequantized = dequantized.flatten()[:n].reshape(W.shape)
            if do_rotate and in_dim % rot_blocksize == 0:
                for b in range(in_dim // rot_blocksize):
                    s, e = b * rot_blocksize, (b + 1) * rot_blocksize
                    dequantized[:, s:e] = dequantized[:, s:e] @ H_block
                    dequantized[:, s:e] *= signs[s:e].unsqueeze(0)
            mod.weight.data = dequantized.to(mod.weight.dtype)
        return hook

    def post_hook(mod, args_, output):
        if hasattr(mod, '_orig_weight'):
            mod.weight.data = mod._orig_weight
            del mod._orig_weight

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "embed" not in name and \
                "lm_head" not in name:
            h1 = module.register_forward_pre_hook(
                make_pre_hook(qcb, dcb, True))
            h2 = module.register_forward_hook(post_hook)
            _active_hooks.extend([h1, h2])

    return len(_active_hooks) // 2


def install_l2_hooks(model, quant_codebook, deq_codebook, seed, device,
                     p=2, rot_blocksize=128):
    """Install forward hooks using HIGGS-style L2 normalization.

    Matches HIGGS Algorithm 1 (RHT-VQ):
      1. Compute L2 norm per (row, rot_block) → scale s_i
      2. Normalize: w_i / s_i  (unit norm)
      3. Random Hadamard Transform on unit-norm blocks → entries ≈ N(0,1)
      4. Quantize against Gaussian grid (nearest neighbor)
      5. Dequantize: inverse RHT, then scale back by s_i
    """
    global _active_hooks
    remove_hooks()

    H_block = make_hadamard_block(rot_blocksize, device)
    max_in = max(m.weight.shape[1] for m in model.modules()
                 if isinstance(m, nn.Linear))
    signs = make_random_signs(max_in, seed, device)

    qcb = quant_codebook.to(device)
    dcb = deq_codebook.to(device)

    def make_pre_hook(qcb_t, dcb_t, do_rotate):
        def hook(mod, args_):
            if not _hooks_enabled:
                return
            mod._orig_weight = mod.weight.data.clone()
            W = mod.weight.data.float()
            out_dim, in_dim = W.shape

            # 1. L2 normalization per (row, rot_block) — matches HIGGS step 1
            #    s_i = ||w_i||_2
            n_rot = in_dim // rot_blocksize
            W_blocks = W[:, :n_rot * rot_blocksize].reshape(
                out_dim, n_rot, rot_blocksize)
            l2_norms = W_blocks.norm(dim=2, keepdim=True).clamp_(min=1e-12)

            # 2. Normalize to unit norm: w_i / s_i
            W_unit = W_blocks / l2_norms

            # 3. Random Hadamard Transform on unit-norm blocks
            #    After normalized Hadamard, unit-norm vector stays unit-norm.
            #    Each entry has std ≈ 1/sqrt(rot_blocksize).
            #    We scale by sqrt(rot_blocksize) to get std ≈ 1 (N(0,1)).
            if do_rotate and in_dim % rot_blocksize == 0:
                W_unit_flat = W_unit.reshape(out_dim * n_rot, rot_blocksize)
                W_unit_flat.copy_(
                    (W_unit_flat * signs[:rot_blocksize].unsqueeze(0)) @ H_block.T)
                W_unit_flat.mul_(rot_blocksize ** 0.5)

            # 4. Quantize against Gaussian grid
            #    Handle remainder if rot_blocksize not divisible by p
            elems_per_rot = (rot_blocksize // p) * p
            remainder = rot_blocksize - elems_per_rot

            W_flat = W_unit.reshape(out_dim * n_rot, rot_blocksize)
            if remainder > 0:
                vq_part = W_flat[:, :elems_per_rot]
                rem_part = W_flat[:, elems_per_rot:]
            else:
                vq_part = W_flat

            groups = vq_part.reshape(-1, p)
            indices = _chunked_nearest(groups, qcb_t, chunk_size=100000)
            dq_groups = dcb_t[indices]
            dq_vq = dq_groups.reshape(out_dim * n_rot, elems_per_rot)

            if remainder > 0:
                dq_blocks = torch.cat([dq_vq, rem_part], dim=1)
            else:
                dq_blocks = dq_vq

            # 5. Undo scaling and inverse Hadamard, then scale by s_i
            #    First undo the sqrt(rot_blocksize) scaling
            dq_blocks.div_(rot_blocksize ** 0.5)

            # Inverse Hadamard: H is symmetric, so H^{-1} = H^T = H
            if do_rotate and in_dim % rot_blocksize == 0:
                dq_blocks.copy_(
                    (dq_blocks @ H_block) * signs[:rot_blocksize].unsqueeze(0))

            # Scale back by s_i (original L2 norms)
            dq_blocks = dq_blocks.reshape(out_dim, n_rot, rot_blocksize)
            dq_blocks = dq_blocks * l2_norms
            dequantized = dq_blocks.reshape(out_dim, n_rot * rot_blocksize)

            # Handle remainder columns (if in_dim not divisible by rot_blocksize)
            if in_dim > n_rot * rot_blocksize:
                dequantized = torch.cat(
                    [dequantized, W[:, n_rot * rot_blocksize:]], dim=1)

            mod.weight.data = dequantized.to(mod.weight.dtype)
        return hook

    def post_hook(mod, args_, output):
        if hasattr(mod, '_orig_weight'):
            mod.weight.data = mod._orig_weight
            del mod._orig_weight

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "embed" not in name and \
                "lm_head" not in name:
            h1 = module.register_forward_pre_hook(
                make_pre_hook(qcb, dcb, True))
            h2 = module.register_forward_hook(post_hook)
            _active_hooks.extend([h1, h2])

    return len(_active_hooks) // 2


def remove_hooks():
    global _active_hooks
    for h in _active_hooks:
        h.remove()
    _active_hooks = []


# ============================================================
# Perplexity evaluation — matches GPTQ/SpQR/QuIP exactly
# ============================================================

@torch.no_grad()
def eval_ppl(model, testenc, seqlen, device, compare_fp16=False):
    """Evaluate perplexity with non-overlapping chunks.

    If compare_fp16=True, also runs FP16 (hooks disabled) per chunk and
    computes three comparison metrics:
      1. Per-token loss difference distribution (mean, std, p95, p99)
      2. Per-token KL divergence KL(fp16 || quant)
      3. Top-1 agreement rate
    """
    nsamples = testenc.numel() // seqlen

    nlls = []
    # Comparison accumulators
    if compare_fp16:
        fp16_nlls = []
        all_loss_diffs = []      # per-token (quant_loss - fp16_loss)
        all_kl_divs = []         # per-token KL(fp16 || quant)
        all_top1_agree = []      # per-token bool
        total_tokens = 0

    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)

        if compare_fp16:
            # Pass 1: FP16 (hooks disabled)
            set_hooks_enabled(False)
            fp16_logits = model(batch).logits[:, :-1, :].contiguous().float()
            set_hooks_enabled(True)

            # Pass 2: Quantized (hooks enabled)
            quant_logits = model(batch).logits[:, :-1, :].contiguous().float()

            # --- Quantized PPL (standard) ---
            loss_fct = nn.CrossEntropyLoss()
            quant_loss = loss_fct(
                quant_logits.view(-1, quant_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = quant_loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            # --- FP16 PPL ---
            fp16_loss = loss_fct(
                fp16_logits.view(-1, fp16_logits.size(-1)),
                shift_labels.view(-1),
            )
            fp16_nlls.append(fp16_loss.float() * seqlen)

            # --- Per-token metrics ---
            # Per-token cross-entropy losses
            ce_none = nn.CrossEntropyLoss(reduction='none')
            flat_labels = shift_labels.view(-1)
            fp16_per_token = ce_none(
                fp16_logits.view(-1, fp16_logits.size(-1)), flat_labels)
            quant_per_token = ce_none(
                quant_logits.view(-1, quant_logits.size(-1)), flat_labels)

            # 1. Loss difference
            loss_diff = quant_per_token - fp16_per_token
            all_loss_diffs.append(loss_diff.cpu())

            # 2. KL divergence: KL(fp16 || quant)
            fp16_log_p = torch.log_softmax(fp16_logits.view(-1, fp16_logits.size(-1)), dim=-1)
            quant_log_q = torch.log_softmax(quant_logits.view(-1, quant_logits.size(-1)), dim=-1)
            fp16_p = fp16_log_p.exp()
            kl_per_token = (fp16_p * (fp16_log_p - quant_log_q)).sum(dim=-1)
            all_kl_divs.append(kl_per_token.cpu())

            # 3. Top-1 agreement
            fp16_top1 = fp16_logits.view(-1, fp16_logits.size(-1)).argmax(dim=-1)
            quant_top1 = quant_logits.view(-1, quant_logits.size(-1)).argmax(dim=-1)
            agree = (fp16_top1 == quant_top1).float()
            all_top1_agree.append(agree.cpu())

            total_tokens += flat_labels.numel()

            del fp16_logits, quant_logits, fp16_log_p, quant_log_q, fp16_p
        else:
            # Standard single-pass evaluation
            outputs = model(batch)
            lm_logits = outputs.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        if (i + 1) % 20 == 0 or i == nsamples - 1:
            run_ppl = torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen)).item()
            if compare_fp16:
                run_fp16 = torch.exp(torch.stack(fp16_nlls).sum() / ((i+1) * seqlen)).item()
                run_kl = torch.cat(all_kl_divs).mean().item()
                run_agree = torch.cat(all_top1_agree).mean().item()
                run_ldiff = torch.cat(all_loss_diffs).mean().item()
                print(f"  chunk {i+1}/{nsamples}, "
                      f"ppl={run_ppl:.4f} fp16={run_fp16:.4f} "
                      f"KL={run_kl:.4f} top1={run_agree:.4f} "
                      f"ldiff={run_ldiff:.4f}",
                      flush=True)
            else:
                print(f"  chunk {i+1}/{nsamples}, "
                      f"running ppl = {run_ppl:.4f}",
                      flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    if compare_fp16:
        fp16_ppl = torch.exp(torch.stack(fp16_nlls).sum() / (nsamples * seqlen))
        loss_diffs = torch.cat(all_loss_diffs)
        kl_divs = torch.cat(all_kl_divs)
        top1_agrees = torch.cat(all_top1_agree)

        sorted_diffs = loss_diffs.sort().values
        sorted_kl = kl_divs.sort().values
        n = len(loss_diffs)
        p95_idx = int(0.95 * n)
        p99_idx = int(0.99 * n)

        comparison = {
            "fp16_ppl": fp16_ppl.item(),
            "quant_ppl": ppl.item(),
            "n_tokens": total_tokens,
            "loss_diff_mean": loss_diffs.mean().item(),
            "loss_diff_std": loss_diffs.std().item(),
            "loss_diff_p95": sorted_diffs[p95_idx].item(),
            "loss_diff_p99": sorted_diffs[p99_idx].item(),
            "loss_diff_max": loss_diffs.max().item(),
            "kl_div_mean": kl_divs.mean().item(),
            "kl_div_std": kl_divs.std().item(),
            "kl_div_p95": sorted_kl[p95_idx].item(),
            "kl_div_p99": sorted_kl[p99_idx].item(),
            "kl_div_max": kl_divs.max().item(),
            "top1_agreement": top1_agrees.mean().item(),
        }
        return ppl.item(), nsamples, comparison

    return ppl.item(), nsamples


# ============================================================
# HIGGS support
# ============================================================

def compute_absmax_codebook(k, p, device='cuda'):
    """Compute uniform codebook for absmax quantization."""
    n_entries = 2**k
    values = torch.linspace(-1, 1, n_entries, device=device)
    if p == 1:
        return values.view(-1, 1)
    else:
        grids = torch.meshgrid(*([values] * p), indexing='ij')
        codebook = torch.stack([g.flatten() for g in grids], dim=1)
        return codebook


def compute_l2_codebook(k, p, device='cuda'):
    """Compute codebook for L2 quantization."""
    n_entries = 2**k
    values = torch.linspace(-1, 1, int(n_entries ** (1/p)) + 1, device=device)
    if p == 1:
        return values.view(-1, 1)
    else:
        grids = torch.meshgrid(*([values] * p), indexing='ij')
        codebook = torch.stack([g.flatten() for g in grids], dim=1)
        # Keep only n_entries closest to origin
        norms = torch.norm(codebook, dim=1)
        _, indices = torch.sort(norms)
        return codebook[indices[:n_entries]]


def load_higgs_assignment(assignment_path):
    """Load HIGGS bitwidth assignment from JSON file.

    Returns:
        assignment: dict mapping layer_idx -> option_idx
        options: list of quantization option dicts
        avg_bits: average bits per element
    """
    import json
    with open(assignment_path, 'r') as f:
        data = json.load(f)

    assignment = {int(k): v for k, v in data['assignment'].items()}
    options = data['options']
    avg_bits = data.get('avg_bits', 0)
    return assignment, options, avg_bits


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation (GPTQ-standard procedure)")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2"],
                        choices=["wikitext2", "c4"],
                        help="Datasets to evaluate (default: wikitext2)")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length for chunking (default: 4096)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype (default: auto)")

    # BNF quantization args
    parser.add_argument("--method", default="fp16",
                        choices=["fp16", "bnf"],
                        help="Evaluation method (default: fp16)")
    parser.add_argument("--k", type=int, default=4,
                        help="Bits per element for quantization")
    parser.add_argument("--p", type=int, default=1,
                        help="Vector dimension: 1=scalar, 2/3/4=VQ")
    parser.add_argument("--blocksize", type=int, default=32,
                        help="Quantization block size (for absmax norm)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Hadamard sign vector")
    parser.add_argument("--no-kappa-correction", action="store_true",
                        help="Skip kappa1 bias correction (same as --kappa-method none)")
    parser.add_argument("--kappa-method", default=None,
                        choices=["none", "standard", "mmse", "theory-correct",
                                 "per-entry"],
                        help="Kappa correction method: "
                             "none=no correction, "
                             "standard=E[qr]/E[r²] (inner product bias), "
                             "mmse=E[qr]/E[q²] (MSE-optimal scaling), "
                             "theory-correct=pinned ±1 + κ_nonmax (scalar p=1 only), "
                             "per-entry=per-codebook-entry κ_i (any p)")
    parser.add_argument("--kappa-fraction", type=float, default=None,
                        help="Fraction of bias to remove (0=none, 1=full). "
                             "Overrides --kappa-method. Sets α = 1 + frac*(1/κ₀ - 1). "
                             "0.0=MSE-optimal, 0.5=half correction, 1.0=standard.")
    parser.add_argument("--normalize-codebook", action="store_true",
                        help="Normalize codebook to [-1, 1] before kappa")
    parser.add_argument("--rot-blocksize", type=int, default=128,
                        help="Hadamard rotation block size (default: 128)")
    parser.add_argument("--norm", default="absmax",
                        choices=["absmax", "l2"],
                        help="Normalization: absmax (BNF) or l2 (HIGGS-style)")
    parser.add_argument("--grid-file", type=str, default=None,
                        help="Path to Pagès-Printems grid file for VQ codebook")
    parser.add_argument("--higgs-assignment", type=str, default=None,
                        help="Path to HIGGS bitwidth assignment JSON file "
                             "for per-layer dynamic quantization")
    parser.add_argument("--compare-fp16", action="store_true",
                        help="Run FP16 alongside quantized and compute "
                             "per-token loss diff, KL divergence, top-1 agreement")
    args = parser.parse_args()

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_bits = args.k * args.p if args.method == "bnf" else None
    if args.method == "bnf":
        if args.grid_file:
            # Effective bits computed from actual grid size after loading
            effective_bits = None  # placeholder
        elif args.norm == "absmax":
            effective_bits = args.k + 8 / args.blocksize
        else:
            # L2 norm: one fp16 scale per rot_blocksize elements
            effective_bits = args.k + 16 / args.rot_blocksize
    else:
        effective_bits = None
    codebook_size = (1 << index_bits) if index_bits else None

    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Method:  {args.method}")
    if args.method == "bnf":
        print(f"  k={args.k} bits/elem, p={args.p}")
        print(f"  norm={args.norm}, rot_blocksize={args.rot_blocksize}")
        if args.norm == "absmax":
            print(f"  blocksize={args.blocksize}")
        print(f"  index_bits={index_bits}, codebook_size={codebook_size}")
        if effective_bits is not None:
            print(f"  effective_bits={effective_bits:.3f}, seed={args.seed}")
        else:
            print(f"  effective_bits=computed from grid, seed={args.seed}")
        if args.grid_file:
            print(f"  grid_file={args.grid_file}")
    print(f"Device:  {device}, dtype: {args.dtype}")
    print(f"Seqlen:  {args.seqlen}")
    print(f"Datasets: {args.datasets}")
    print(f"Host:    {socket.gethostname()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print("=" * 60, flush=True)

    # Load tokenizer and model
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Install quantization hooks
    kappa1 = None
    kappa_method = None
    if args.method == "bnf":
        dev = next(model.parameters()).device

        kappa_stats = {}

        # ---- Phase 1: Compute codebook ----
        if args.grid_file:
            # Load published optimal grid
            print(f"\nLoading grid from {args.grid_file}...", flush=True)
            quant_cb = load_grid_file(args.grid_file).to(dev)
            n_grid = quant_cb.shape[0]
            p_grid = quant_cb.shape[1] if quant_cb.dim() > 1 else 1
            if p_grid != args.p:
                raise ValueError(
                    f"Grid dimensionality (p={p_grid}) does not match --p "
                    f"({args.p}). Use --p {p_grid} for this grid.")
            # Compute effective bits from actual grid dimensions
            import math
            bits_per_elem = math.log2(n_grid) / p_grid if n_grid > 1 else 0
            if args.norm == "absmax":
                effective_bits = bits_per_elem + 8 / args.blocksize
            else:
                effective_bits = bits_per_elem + 16 / args.rot_blocksize
            index_bits = math.log2(n_grid)
            codebook_size = n_grid
            # Compute kappa on Gaussian samples
            rng = torch.Generator(device=dev).manual_seed(0)
            gauss_samples = torch.randn(100000, p_grid, device=dev, generator=rng)
            dists = torch.cdist(gauss_samples.float(), quant_cb.float())
            assignments = dists.argmin(dim=1)
            q_samples = quant_cb[assignments]
            e_qr = (q_samples * gauss_samples).sum(dim=1).mean().item()
            e_r2 = (gauss_samples ** 2).sum(dim=1).mean().item()
            e_q2 = (q_samples ** 2).sum(dim=1).mean().item()
            kappa1 = e_qr / e_r2 if e_r2 > 1e-12 else 1.0
            kappa_stats = {"e_qr": e_qr, "e_r2": e_r2, "e_q2": e_q2}
            print(f"  codebook shape = {quant_cb.shape}", flush=True)
            print(f"  grid entries = {n_grid}, dim = {p_grid}", flush=True)
            print(f"  bits/elem (index) = {bits_per_elem:.4f}", flush=True)
            print(f"  effective_bits = {effective_bits:.4f}", flush=True)
            print(f"  kappa1 = {kappa1:.6f}  (from Gaussian samples)", flush=True)
            print(f"  first 4 entries:", flush=True)
            for i in range(min(4, quant_cb.shape[0])):
                print(f"    [{i}] = {quant_cb[i].cpu().numpy().round(4)}",
                      flush=True)
        elif args.kappa_method == "theory-correct" and args.p == 1:
            # THEORY.md correct implementation: pinned ±1, train on non-max
            print(f"\nComputing THEORY-CORRECT codebook (k={args.k}, p=1, "
                  f"g={args.blocksize})...", flush=True)
            print(f"  (pinned ±1 boundaries, Lloyd-Max on non-max only)",
                  flush=True)
            quant_cb, _, kappa1, kappa_stats = \
                compute_scalar_codebook_theory_correct(
                    args.k, args.blocksize, device=dev)
            print(f"  κ₁^non-max = {kappa1:.6f}", flush=True)
            print(f"  Quant codebook:  {quant_cb.cpu().numpy().round(4)}",
                  flush=True)
        elif args.norm == "l2" and args.grid_file is None:
            # L2 normalization requires Gaussian grid, not block-normalized
            print(f"\nComputing Gaussian grid for L2 normalization "
                  f"(k={args.k}, p={args.p})...", flush=True)
            quant_cb, _, kappa1, kappa_stats = compute_gaussian_grid(
                args.k, args.p, device=dev)
            print(f"  Gaussian grid shape: {quant_cb.shape}", flush=True)
            print(f"  kappa1 (standard) = {kappa1:.6f}", flush=True)
        else:
            # Compute codebook via Lloyd-Max / k-means
            print(f"\nComputing codebook (k={args.k}, p={args.p}, "
                  f"g={args.blocksize})...", flush=True)
            quant_cb, _, kappa1, kappa_stats = compute_codebook(
                args.k, args.p, args.blocksize, device=dev)
            print(f"  kappa1 (standard) = {kappa1:.6f}", flush=True)
            if kappa_stats:
                e_qr = kappa_stats["e_qr"]
                e_r2 = kappa_stats["e_r2"]
                e_q2 = kappa_stats["e_q2"]
                kappa_mmse = e_qr / e_q2 if e_q2 > 1e-12 else 1.0
                print(f"  kappa_mmse (E[qr]/E[q²]) = {kappa_mmse:.6f}",
                      flush=True)
                print(f"  E[qr]={e_qr:.6f}, E[r²]={e_r2:.6f}, "
                      f"E[q²]={e_q2:.6f}", flush=True)

            if args.normalize_codebook and args.p == 1:
                cb_max = quant_cb.abs().max()
                quant_cb = quant_cb / cb_max
                print(f"  Codebook normalized to [-1, 1] "
                      f"(scaled by {cb_max:.6f})", flush=True)

            if args.p == 1:
                print(f"  quant codebook = "
                      f"{quant_cb.cpu().numpy().round(4)}", flush=True)
            else:
                print(f"  codebook shape = {quant_cb.shape}", flush=True)
                print(f"  first 4 entries:", flush=True)
                for i in range(min(4, quant_cb.shape[0])):
                    print(f"    [{i}] = "
                          f"{quant_cb[i].cpu().numpy().round(4)}",
                          flush=True)

        # ---- Phase 2: Resolve kappa method ----
        # --kappa-fraction overrides --kappa-method
        if args.kappa_fraction is not None:
            frac = args.kappa_fraction
            alpha = 1.0 + frac * (1.0 / kappa1 - 1.0)
            deq_cb = quant_cb * alpha
            kappa_method = f"fraction-{frac:.2f}"
            print(f"  kappa correction: FRACTION={frac:.2f}", flush=True)
            print(f"    κ₀={kappa1:.6f}, α={alpha:.6f}, "
                  f"target κ₁={alpha * kappa1:.6f}", flush=True)
            print(f"    bias removed: {frac*100:.0f}%, "
                  f"distortion cost: ~{100*(alpha**2 - 1):.2f}%",
                  flush=True)
            kappa1 = alpha  # store α in output JSON
        else:
            kappa_method = args.kappa_method
            if kappa_method is None:
                if args.no_kappa_correction:
                    kappa_method = "none"
                else:
                    kappa_method = "standard"

        if args.kappa_fraction is not None:
            pass  # already handled above
        elif kappa_method == "none":
            deq_cb = quant_cb.clone()
            print(f"  kappa correction: DISABLED (deq = quant)", flush=True)
        elif kappa_method == "standard":
            deq_cb = quant_cb / kappa1
            print(f"  kappa correction: STANDARD "
                  f"(deq = q / {kappa1:.6f})", flush=True)
            if args.p == 1:
                print(f"  deq codebook   = "
                      f"{deq_cb.cpu().numpy().round(4)}", flush=True)
        elif kappa_method == "mmse":
            if kappa_stats:
                alpha = kappa_stats["e_qr"] / kappa_stats["e_q2"] \
                    if kappa_stats["e_q2"] > 1e-12 else 1.0
                deq_cb = quant_cb * alpha
                kappa1 = alpha  # store for JSON output
                print(f"  kappa correction: MMSE (alpha={alpha:.6f})",
                      flush=True)
            else:
                print(f"  WARNING: no kappa_stats, "
                      f"falling back to standard", flush=True)
                deq_cb = quant_cb / kappa1
        elif kappa_method == "per-entry":
            print(f"  Computing per-entry kappa corrections...",
                  flush=True)
            deq_cb, kappa_per_entry, pe_stats = \
                compute_per_entry_kappa(
                    quant_cb, args.k, args.p, args.blocksize,
                    device=dev)
            kappa1 = pe_stats["kappa_mean"]
            print(f"  kappa correction: PER-ENTRY", flush=True)
            print(f"    κ range: [{pe_stats['kappa_min']:.6f}, "
                  f"{pe_stats['kappa_max']:.6f}]", flush=True)
            print(f"    κ mean:  {pe_stats['kappa_mean']:.6f}, "
                  f"std: {pe_stats['kappa_std']:.6f}", flush=True)
            if args.p == 1:
                print(f"  deq codebook   = "
                      f"{deq_cb.cpu().numpy().round(4)}", flush=True)
        elif kappa_method == "theory-correct":
            # theory-correct computes its own dequant codebook above
            # Re-derive from the theory-correct compute function
            quant_cb, deq_cb, kappa1, kappa_stats = \
                compute_scalar_codebook_theory_correct(
                    args.k, args.blocksize, device=dev)
            print(f"  kappa correction: THEORY-CORRECT "
                  f"(±1 pinned, inner / {kappa1:.6f})", flush=True)
            print(f"  Dequant codebook: {deq_cb.cpu().numpy().round(4)}",
                  flush=True)

        # ---- Phase 3: Install hooks ----
        if args.norm == "l2":
            n_hooked = install_l2_hooks(
                model, quant_cb, deq_cb, args.seed, dev,
                p=args.p, rot_blocksize=args.rot_blocksize)
            print(f"  L2-norm hooks installed on {n_hooked} layers "
                  f"(rot_blocksize={args.rot_blocksize})", flush=True)
        else:
            n_hooked = install_bnf_hooks(
                model, quant_cb, deq_cb, args.blocksize, args.seed, dev,
                p=args.p, rot_blocksize=args.rot_blocksize)
            print(f"  Absmax hooks installed on {n_hooked} layers "
                  f"(rot_blocksize={args.rot_blocksize})", flush=True)

    # ---- HIGGS per-layer quantization ----
    if args.higgs_assignment:
        print(f"\n{'='*60}")
        print("HIGGS per-layer quantization")
        print(f"{'='*60}")
        assignment, options, avg_bits = load_higgs_assignment(args.higgs_assignment)
        print(f"Assignment: {len(assignment)} layers, avg {avg_bits:.3f} bits")

        # Count distribution
        from collections import Counter
        opt_counts = Counter(assignment.values())
        print("Distribution:")
        for opt_idx, count in sorted(opt_counts.items()):
            opt = options[opt_idx]
            print(f"  {opt['config_str']}: {count} layers")

        # Get all linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'embed' not in name and 'lm_head' not in name:
                linear_layers.append(module)

        # Apply per-layer quantization
        dev = next(model.parameters()).device
        hooks = []
        for layer_idx, module in enumerate(linear_layers):
            if layer_idx not in assignment:
                continue
            opt_idx = assignment[layer_idx]
            opt = options[opt_idx]
            k, p = opt['k'], opt['p']

            # Compute codebook for this layer
            if args.norm == 'l2':
                quant_cb = compute_l2_codebook(k, p, device=dev)
            else:
                quant_cb = compute_absmax_codebook(k, p, device=dev)

            # Create hook
            def make_hook(k_val, p_val, cb, norm, bs, rot_bs):
                def hook(module, input, output):
                    if not hasattr(module, '_higgs_quantized'):
                        w = module.weight.data
                        weight_dtype = w.dtype

                        # Apply Hadamard rotation if L2
                        if norm == 'l2':
                            # Simple rotation (without full block structure for now)
                            w_flat = w.reshape(-1, p_val).float()
                        else:
                            w_flat = w.reshape(-1, p_val).float()

                        # VQ quantize
                        dists = torch.cdist(w_flat, cb.float())
                        indices = dists.argmin(dim=1)
                        w_q = cb[indices].reshape(w.shape).to(weight_dtype)

                        module.weight.data = w_q
                        module._higgs_quantized = True
                    return output
                return hook

            handle = module.register_forward_hook(
                make_hook(k, p, quant_cb, args.norm, args.blocksize, args.rot_blocksize)
            )
            hooks.append(handle)

        print(f"Installed {len(hooks)} per-layer quantization hooks")
        effective_bits = avg_bits

    t_start = time.time()
    results = {
        "method": args.method,
        "model": args.model,
        "dtype": args.dtype,
        "seqlen": args.seqlen,
        "k": args.k if args.method == "bnf" else None,
        "p": args.p if args.method == "bnf" else None,
        "blocksize": args.blocksize if args.method == "bnf" else None,
        "index_bits": index_bits,
        "codebook_size": codebook_size,
        "effective_bits": effective_bits,
        "seed": args.seed if args.method == "bnf" else None,
        "kappa1": kappa1,
        "kappa_method": kappa_method,
        "kappa_fraction": args.kappa_fraction if args.method == "bnf" else None,
        "kappa_correction": (args.method == "bnf"
                             and kappa_method != "none"),
        "normalize_codebook": (args.method == "bnf"
                               and args.normalize_codebook),
        "norm": args.norm if args.method == "bnf" else None,
        "rot_blocksize": args.rot_blocksize if args.method == "bnf" else None,
        "grid_file": args.grid_file,
    }

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        # Load data
        t0 = time.time()
        loader = DATASET_LOADERS[dataset_name]
        testenc = loader(args.seqlen, tokenizer)
        nsamples = testenc.numel() // args.seqlen
        print(f"  Loaded: {testenc.numel():,} tokens -> {nsamples} chunks "
              f"({testenc.numel() - nsamples * args.seqlen} dropped) "
              f"in {time.time() - t0:.1f}s", flush=True)

        # Evaluate
        t0 = time.time()
        do_compare = args.compare_fp16 and args.method == "bnf"
        eval_result = eval_ppl(model, testenc, args.seqlen, device,
                               compare_fp16=do_compare)
        eval_time = time.time() - t0

        if do_compare:
            ppl, nsamples, comparison = eval_result
        else:
            ppl, nsamples = eval_result
            comparison = None

        print(f"\n  {dataset_name} perplexity = {ppl:.4f}")
        print(f"  ({nsamples} chunks, {eval_time:.1f}s)", flush=True)

        results[dataset_name] = {
            "perplexity": ppl,
            "n_chunks": nsamples,
            "n_tokens_total": testenc.numel(),
            "n_tokens_evaluated": nsamples * args.seqlen,
            "eval_time_seconds": eval_time,
        }

        if comparison:
            results[dataset_name]["comparison"] = comparison
            print(f"\n  --- FP16 comparison ---")
            print(f"  FP16 PPL:          {comparison['fp16_ppl']:.4f}")
            print(f"  Quant PPL:         {comparison['quant_ppl']:.4f}")
            print(f"  Loss diff mean:    {comparison['loss_diff_mean']:.6f}")
            print(f"  Loss diff std:     {comparison['loss_diff_std']:.6f}")
            print(f"  Loss diff p95:     {comparison['loss_diff_p95']:.6f}")
            print(f"  Loss diff p99:     {comparison['loss_diff_p99']:.6f}")
            print(f"  Loss diff max:     {comparison['loss_diff_max']:.6f}")
            print(f"  KL div mean:       {comparison['kl_div_mean']:.6f}")
            print(f"  KL div p95:        {comparison['kl_div_p95']:.6f}")
            print(f"  KL div p99:        {comparison['kl_div_p99']:.6f}")
            print(f"  Top-1 agreement:   {comparison['top1_agreement']:.4f}")

    # Clean up
    if args.method == "bnf":
        remove_hooks()

    results["total_time_seconds"] = time.time() - t_start
    results["hostname"] = socket.gethostname()
    results["gpu"] = torch.cuda.get_device_name(0) \
        if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: {args.model} (seqlen={args.seqlen})")
    if args.method == "bnf":
        print(f"  k={args.k} p={args.p} norm={args.norm} "
              f"rot={args.rot_blocksize} seed={args.seed} "
              f"({effective_bits:.2f} bits)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'PPL':>10} {'Chunks':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8}")
    for dataset_name in args.datasets:
        r = results[dataset_name]
        print(f"  {dataset_name:<12} {r['perplexity']:>10.4f} {r['n_chunks']:>8}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
