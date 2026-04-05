#!/bin/bash
# Example: Precompute alphas and error table once, then run multiple target bitrates
#
# This workflow is for the specific targets:
#   - 3.25 bits: k=3, absmax norm (3 + 0.25 = 3.25)
#   - 4.0 bits: closest is k=4, L2 norm (4 + 0.125 = 4.125)
#   - 4.25 bits: k=4, absmax norm (4 + 0.25 = 4.25)

set -e

MODEL="meta-llama/Llama-3.1-8B"
SEQLEN=2048
CALIBRATION_TOKENS=287000

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
RESULTS_DIR="../../results/dynamic_bw/${MODEL_SAFE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Precompute Workflow: HIGGS Dynamic Bitwidth"
echo "============================================================"
echo "Model: $MODEL"
echo "Results dir: $RESULTS_DIR"
echo "============================================================"

# ============================================================================
# STEP 1: Precompute alpha coefficients (data-free, run once)
# ============================================================================
ALPHA_FILE="${RESULTS_DIR}/alphas.json"
if [ -f "$ALPHA_FILE" ]; then
    echo ""
    echo "Alpha coefficients already exist: $ALPHA_FILE"
    echo "Skipping Step 1 (calibration)..."
else
    echo ""
    echo "=== STEP 1: Calibrating alpha coefficients (data-free) ==="
    echo "This uses KL-divergence on random tokens - no dataset needed."
    echo "Running ${CALIBRATION_TOKENS} tokens, ~480 forward passes..."
    echo ""

    python dynamic_bitwidth.py \
        --model "$MODEL" \
        --seqlen $SEQLEN \
        --calibrate \
        --calibration-tokens $CALIBRATION_TOKENS \
        --output-alpha "$ALPHA_FILE"

    echo ""
    echo "Alpha coefficients saved to: $ALPHA_FILE"
fi

# ============================================================================
# STEP 2: Precompute quantization error table (run once per norm type)
# ============================================================================
#
# We need TWO error tables because the targets use different norms:
#   - 3.25 bits: absmax (k=3)
#   - 4.0 bits: L2 (k=4)
#   - 4.25 bits: absmax (k=4)

# Error table for absmax norm (targets: 3.25, 4.25 bits)
ERROR_TABLE_ABSMAX="${RESULTS_DIR}/error_table_absmax.json"
if [ -f "$ERROR_TABLE_ABSMAX" ]; then
    echo ""
    echo "Absmax error table already exists: $ERROR_TABLE_ABSMAX"
    echo "Skipping Step 2a (absmax error table)..."
else
    echo ""
    echo "=== STEP 2a: Computing error table for absmax norm ==="
    echo "Options: k=2,3,4,5 (effective bits: 2.25, 3.25, 4.25, 5.25)"
    echo ""

    python dynamic_bitwidth.py \
        --model "$MODEL" \
        --seqlen $SEQLEN \
        --compute-error-table \
        --norm absmax \
        --blocksize 32 \
        --bitwidth-options "k=2,p=2;k=3,p=2;k=4,p=2;k=5,p=2" \
        --error-table-path "$ERROR_TABLE_ABSMAX"

    echo ""
    echo "Absmax error table saved to: $ERROR_TABLE_ABSMAX"
fi

# Error table for L2 norm (targets: 4.0 bits)
ERROR_TABLE_L2="${RESULTS_DIR}/error_table_l2.json"
if [ -f "$ERROR_TABLE_L2" ]; then
    echo ""
    echo "L2 error table already exists: $ERROR_TABLE_L2"
    echo "Skipping Step 2b (L2 error table)..."
else
    echo ""
    echo "=== STEP 2b: Computing error table for L2 norm ==="
    echo "Options: k=2,3,4,5 (effective bits: 2.125, 3.125, 4.125, 5.125)"
    echo ""

    python dynamic_bitwidth.py \
        --model "$MODEL" \
        --seqlen $SEQLEN \
        --compute-error-table \
        --norm l2 \
        --rot-blocksize 128 \
        --bitwidth-options "k=2,p=2;k=3,p=2;k=4,p=2;k=5,p=2" \
        --error-table-path "$ERROR_TABLE_L2"

    echo ""
    echo "L2 error table saved to: $ERROR_TABLE_L2"
fi

# ============================================================================
# STEP 3: Optimize for specific target bitrates (fast, runs multiple times)
# ============================================================================

echo ""
echo "=== STEP 3: Optimizing bitwidth assignments ==="

# 3.25 bits: absmax, k=3 (3 + 0.25 = 3.25)
echo ""
echo "--- Target: 3.25 bits (absmax k=3) ---"
python dynamic_bitwidth.py \
    --model "$MODEL" \
    --optimize \
    --alpha-path "$ALPHA_FILE" \
    --error-table-path "$ERROR_TABLE_ABSMAX" \
    --target-bits 3.25 \
    --output-assignment "${RESULTS_DIR}/assignment_3.25bits_absmax.json"

# 4.0 bits: L2, k=4 (4 + 0.125 = 4.125, closest to 4.0)
echo ""
echo "--- Target: 4.0 bits (L2 k=4) ---"
python dynamic_bitwidth.py \
    --model "$MODEL" \
    --optimize \
    --alpha-path "$ALPHA_FILE" \
    --error-table-path "$ERROR_TABLE_L2" \
    --target-bits 4.0 \
    --output-assignment "${RESULTS_DIR}/assignment_4.0bits_l2.json"

# 4.25 bits: absmax, k=4 (4 + 0.25 = 4.25)
echo ""
echo "--- Target: 4.25 bits (absmax k=4) ---"
python dynamic_bitwidth.py \
    --model "$MODEL" \
    --optimize \
    --alpha-path "$ALPHA_FILE" \
    --error-table-path "$ERROR_TABLE_ABSMAX" \
    --target-bits 4.25 \
    --output-assignment "${RESULTS_DIR}/assignment_4.25bits_absmax.json"

echo ""
echo "============================================================"
echo "All optimizations complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  3.25 bits: ${RESULTS_DIR}/assignment_3.25bits_absmax.json"
echo "  4.00 bits: ${RESULTS_DIR}/assignment_4.0bits_l2.json"
echo "  4.25 bits: ${RESULTS_DIR}/assignment_4.25bits_absmax.json"
echo ""
echo "Next step: Use these assignments to actually quantize and evaluate PPL"
