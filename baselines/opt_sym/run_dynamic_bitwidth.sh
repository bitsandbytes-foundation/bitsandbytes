#!/bin/bash
# Run HIGGS dynamic bitwidth optimization pipeline on Babel

#SBATCH --job-name=higgs-dynamic
#SBATCH --output=logs/higgs_dynamic_%A_%a.out
#SBATCH --error=logs/higgs_dynamic_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Usage:
#   sbatch run_dynamic_bitwidth.sh --model meta-llama/Llama-3.1-8B --target-bits 3.0
#
# Or as an array job for multiple target bitrates:
#   sbatch --array=2.0,2.5,3.0,3.5,4.0 run_dynamic_bitwidth.sh --model meta-llama/Llama-3.1-8B

set -e

cd "$(dirname "$0")"

# Parse arguments
MODEL=""
TARGET_BITS=""
SEQLEN=2048
CALIBRATION_TOKENS=287000
# Bitwidth options - these define the available quantization configs
# Format: k=BITS,p=DIM where k=bits per element, p=vector dimension
#
# For absmax norm (blocksize=32): effective_bits = k + 8/32 = k + 0.25
#   k=3: 3.25 bits  |  k=4: 4.25 bits
#
# For L2 norm (rot_blocksize=128): effective_bits = k + 16/128 = k + 0.125
#   k=3: 3.125 bits |  k=4: 4.125 bits
#
# To hit specific targets:
#   3.25 bits: use k=3 with absmax norm
#   4.0 bits: use k=4 with L2 norm (4.125 is closest)
#   4.25 bits: use k=4 with absmax norm
BITWIDTH_OPTIONS="k=2,p=2;k=3,p=2;k=4,p=2;k=5,p=2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --target-bits)
            TARGET_BITS="$2"
            shift 2
            ;;
        --seqlen)
            SEQLEN="$2"
            shift 2
            ;;
        --calibration-tokens)
            CALIBRATION_TOKENS="$2"
            shift 2
            ;;
        --bitwidth-options)
            BITWIDTH_OPTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If running as array job, use SLURM_ARRAY_TASK_ID as target bits
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    TARGET_BITS="$SLURM_ARRAY_TASK_ID"
fi

if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    exit 1
fi

if [ -z "$TARGET_BITS" ]; then
    echo "Error: --target-bits is required (or use --array)"
    exit 1
fi

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
RESULTS_DIR="../../results/dynamic_bw/${MODEL_SAFE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "HIGGS Dynamic Bitwidth Optimization"
echo "============================================================"
echo "Model: $MODEL"
echo "Target bits: $TARGET_BITS"
echo "Sequence length: $SEQLEN"
echo "Calibration tokens: $CALIBRATION_TOKENS"
echo "Bitwidth options: $BITWIDTH_OPTIONS"
echo "Results dir: $RESULTS_DIR"
echo "============================================================"

# Step 1: Calibrate alpha coefficients (data-free)
ALPHA_FILE="${RESULTS_DIR}/alphas.json"
if [ -f "$ALPHA_FILE" ]; then
    echo "Alpha coefficients already exist: $ALPHA_FILE"
    echo "Skipping calibration..."
else
    echo ""
    echo "Step 1: Calibrating alpha coefficients..."
    python dynamic_bitwidth.py \
        --model "$MODEL" \
        --seqlen $SEQLEN \
        --calibrate \
        --calibration-tokens $CALIBRATION_TOKENS \
        --output-alpha "$ALPHA_FILE"
fi

# Step 2: Compute quantization error table
ERROR_TABLE_FILE="${RESULTS_DIR}/error_table.json"
if [ -f "$ERROR_TABLE_FILE" ]; then
    echo "Error table already exists: $ERROR_TABLE_FILE"
    echo "Skipping error table computation..."
else
    echo ""
    echo "Step 2: Computing quantization error table..."
    python dynamic_bitwidth.py \
        --model "$MODEL" \
        --seqlen $SEQLEN \
        --compute-error-table \
        --error-table-path "$ERROR_TABLE_FILE" \
        --bitwidth-options "$BITWIDTH_OPTIONS"
fi

# Step 3: Optimize bitwidth assignment
ASSIGNMENT_FILE="${RESULTS_DIR}/assignment_${TARGET_BITS}bits.json"
echo ""
echo "Step 3: Optimizing bitwidth assignment..."
python dynamic_bitwidth.py \
    --model "$MODEL" \
    --optimize \
    --alpha-path "$ALPHA_FILE" \
    --error-table-path "$ERROR_TABLE_FILE" \
    --target-bits "$TARGET_BITS" \
    --output-assignment "$ASSIGNMENT_FILE"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results: $ASSIGNMENT_FILE"
echo "============================================================"

# Display summary
echo ""
echo "Assignment summary:"
python3 << EOF
import json
with open("$ASSIGNMENT_FILE") as f:
    data = json.load(f)
print(f"Target bits: {data['target_bits']}")
print(f"Actual avg bits: {data['avg_bits']:.3f}")
print(f"Expected PPL degradation: {data['expected_ppl_degradation']:.6f}")
print(f"Layer distribution:")
for opt, count in data['option_counts'].items():
    print(f"  {opt}: {count} layers")
EOF
