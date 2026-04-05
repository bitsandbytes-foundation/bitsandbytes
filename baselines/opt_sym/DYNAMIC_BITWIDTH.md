# HIGGS Dynamic Bitwidth Quantization

This implements Section 5 from the HIGGS paper: **Variable Bitwidth Quantization** with data-free calibration.

## Overview

The algorithm finds optimal per-layer bitwidths that minimize perplexity degradation while meeting a target average bitrate. It consists of three steps:

### Step 1: Algorithm 3 — Estimate α_l Coefficients (Data-Free)

Instead of measuring perplexity on a calibration dataset, we use **KL divergence on random tokens**:

1. Generate random token IDs (no dataset needed)
2. Run through the clean model → get clean logits
3. For each layer `l` and noise level `t_j`:
   - Add Gaussian noise `N(0, t_j²)` to layer `l`'s weights
   - Run through noised model → get noised logits
   - Compute `KL(p_clean || p_noised)`
4. Fit α_l via least squares: minimize `Σ_j (ΔKL_j - α_l · t²_j)²`

**Why this works:** We don't need coherent text to measure how much a layer perturbation distorts the output distribution. Random tokens are sufficient because we're measuring relative distortion, not absolute quality.

### Step 2: Compute Quantization Error Table

For each layer `l` and each quantization option `j`, compute:
- `t²_{l,j} = MSE of quantizing layer l with option j`

This is purely about weight quantization error — no model inference needed.

### Step 3: Optimize Bitwidth Assignment

Solve the constrained optimization:

```
min  Σ_l α_l · t²_{l,j_l}
s.t. Σ_l b_{j_l} · d_l ≤ b_max · d
```

where:
- `α_l` = layer sensitivity coefficient (from Step 1)
- `t²_{l,j}` = quantization error for layer l with option j (from Step 2)
- `b_j` = bits per element for option j
- `d_l` = number of elements in layer l
- `b_max` = target average bitrate

We use a **greedy knapsack approach**: iteratively upgrade the layer with the best marginal benefit (PPL improvement per bit spent) until the bit budget is exhausted.

## Usage

### Quick Start — Run All Steps

```bash
cd baselines/opt_sym

# Single target bitrate
python dynamic_bitwidth.py \
    --model meta-llama/Llama-3.1-8B \
    --seqlen 2048 \
    --calibrate \
    --compute-error-table \
    --optimize \
    --target-bits 3.0 \
    --bitwidth-options "k=2,p=2;k=3,p=2;k=4,p=2"
```

This will create:
- `alphas.json` — Per-layer sensitivity coefficients
- `error_table.json` — Quantization error for each layer/option
- `assignment.json` — Optimized bitwidth assignment

### Step-by-Step

**Step 1: Calibrate α_l**

```bash
python dynamic_bitwidth.py \
    --model meta-llama/Llama-3.1-8B \
    --seqlen 2048 \
    --calibrate \
    --calibration-tokens 287000 \
    --output-alpha alphas.json
```

Options:
- `--calibration-tokens`: Number of random tokens (default: 287k, matching HIGGS paper)
- `--n-noise-levels`: Number of noise levels J (default: 15)
- `--t-min`, `--t-max`: Noise range (default: 0.001 to 0.05)

**Step 2: Compute Error Table**

```bash
python dynamic_bitwidth.py \
    --model meta-llama/Llama-3.1-8B \
    --compute-error-table \
    --bitwidth-options "k=2,p=2;k=3,p=2;k=4,p=2" \
    --norm l2 \
    --error-table-path error_table.json
```

Options:
- `--bitwidth-options`: Semicolon-separated options, each as `k=X,p=Y`
- `--norm`: `l2` (HIGGS-style) or `absmax` (BNF-style)
- `--rot-blocksize`: Rotation block size for L2 norm (default: 128)

**Step 3: Optimize Assignment**

```bash
python dynamic_bitwidth.py \
    --model meta-llama/Llama-3.1-8B \
    --optimize \
    --alpha-path alphas.json \
    --error-table-path error_table.json \
    --target-bits 3.0 \
    --output-assignment assignment.json
```

### Batch Job on Babel

```bash
# Single target
sbatch run_dynamic_bitwidth.sh \
    --model meta-llama/Llama-3.1-8B \
    --target-bits 3.0

# Array job for multiple bitrates
sbatch --array=2.0,2.5,3.0,3.5,4.0 \
    run_dynamic_bitwidth.sh \
    --model meta-llama/Llama-3.1-8B
```

## Output Format

### alphas.json

```json
{
  "alphas": {
    "0": 0.123,
    "1": 0.456,
    ...
  },
  "n_layers": 32,
  "calibration_tokens": 287000,
  "n_noise_levels": 15,
  ...
}
```

### error_table.json

```json
{
  "error_table": {
    "0": {"0": 0.0012, "1": 0.0008, "2": 0.0005},
    ...
  },
  "total_elements": {"0": 4194304, ...},
  "options": [
    {"k": 2, "p": 2, "index_bits": 4, "bits_per_entry": 2, "config_str": "k2p2"},
    ...
  ],
  "norm": "l2",
  ...
}
```

### assignment.json

```json
{
  "assignment": {
    "0": 2,  // Layer 0 uses option index 2 (k=4,p=2)
    "1": 1,  // Layer 1 uses option index 1 (k=3,p=2)
    ...
  },
  "avg_bits": 3.012,
  "target_bits": 3.0,
  "expected_ppl_degradation": 0.0456,
  "option_counts": {
    "k2p2": 8,
    "k3p2": 18,
    "k4p2": 6
  }
}
```

## Implementation Notes

### Algorithm 3 (Data-Free Calibration)

The key insight is that we can estimate per-layer sensitivity without a calibration dataset. Instead of:

```
Δ_{l,j} = PPL(W*(l, t_j)) - PPL(W*)  // needs WikiText-2
```

We use:

```
Δ_{l,j} = KL(p_clean || p_noised)  // works with random tokens
```

This makes the calibration **fully self-contained** — no data download or preparation needed.

### Computational Cost

- **Calibration**: `L × J` forward passes = ~32 layers × 15 noise levels = 480 passes
- **Error table**: One quantization per layer per option
- **Optimization**: Negligible (analytical solution)

For Llama-3.1-8B with 287k tokens at seqlen=2048:
- Calibration: ~4-6 hours on single GPU
- Error table: ~30 minutes

### Extending to Per-Entry Quantization

The current implementation assigns the same bitwidth to all weights in a layer. To extend to per-entry quantization (like HIGGS with per-entry sensitivity):

1. Compute κ_l,i for each weight element (like existing κ analysis)
2. Modify error table to track per-entry MSE
3. Add constraint to the optimization ensuring each entry gets at least one bit
4. Solve with a more sophisticated solver (e.g., water-filling)

This is future work — the current layer-wise assignment is already a significant improvement over uniform bitwidths.
