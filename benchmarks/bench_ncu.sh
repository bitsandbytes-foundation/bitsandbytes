#!/bin/bash
# Full kernel benchmark: MMA + scalar + grouped (ncu) + cuBLAS fp16 (CUDA events).
# Then computes end-to-end model summary for Qwen3-Coder-Next 70B.
#
# Usage:
#   bash benchmarks/bench_ncu.sh           # default M=1..8
#   M_VALS=1,4 bash benchmarks/bench_ncu.sh  # custom M values
#
# Output: raw kernel tables, then one summary table per M value showing
# all kernels side by side for every (shape, k) combination.
#
# Runtime: ~2-4 minutes for M=1..8.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/.bench_results"
mkdir -p "$RESULTS_DIR"

export M_VALS="${M_VALS:-1,2,3,4,5,6,7,8}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"
WARMUP=5
PROFILED=5

# Compute M subsets: scalar/grouped only support M<=4
SCALAR_M=$(python3 -c "print(','.join(str(m) for m in [int(x) for x in '$M_VALS'.split(',')] if m <= 4))")
ALL_M="$M_VALS"

echo "START: $(date)"
echo "M values: $M_VALS (scalar/grouped: $SCALAR_M)"
echo "MoE experts: $NUM_EXPERTS"

# Helper: run ncu and parse output for a kernel
run_ncu_bench() {
    local KTYPE="$1"   # mma, scalar, grouped
    local KNAME="$2"   # ncu kernel name filter
    local SHAPES="$3"  # Python list literal for shape names
    local MVALS="$4"   # M values to use

    KERNEL=$KTYPE M_VALS=$MVALS NUM_EXPERTS=$NUM_EXPERTS \
        ncu --kernel-name "$KNAME" --metrics gpu__time_duration.avg \
        python "$SCRIPT_DIR/ncu_driver.py" 2>/dev/null | \
        grep "gpu__time_duration.avg" | awk '{print $NF}' | \
    python3 -c "
import sys
vals = [float(l.strip()) for l in sys.stdin]
shapes = $SHAPES
kbits = [2,3,4,5]
mvals = [int(x) for x in '$MVALS'.split(',')]
W, P = $WARMUP, $PROFILED
i = 0
for s in shapes:
    for k in kbits:
        for m in mvals:
            samples = vals[i+W:i+W+P]
            avg = sum(samples)/len(samples) if samples else 0
            print(f'{s:<8} {k:>2} {m:>2} {avg:>10.2f}')
            i += W + P
"
}

# ---- MMA kernel (all M values) ----
echo ""
echo "=== MMA kernel ==="
printf "%-8s %2s %2s %10s\n" "shape" "k" "M" "avg_us"
echo "---"
run_ncu_bench mma "kbit_gemm_prod" "['gateup','down','Q','O','KV']" "$ALL_M" | tee "$RESULTS_DIR/mma.txt"

# ---- Scalar GEMV (M<=4 only) ----
echo ""
echo "=== Scalar GEMV (M<=4) ==="
printf "%-8s %2s %2s %10s\n" "shape" "k" "M" "avg_us"
echo "---"
if [ -n "$SCALAR_M" ]; then
    run_ncu_bench scalar "kbit_scalar_gemv" "['gateup','down','Q','O','KV']" "$SCALAR_M" | tee "$RESULTS_DIR/scalar.txt"
else
    echo "(no M<=4 values requested)" | tee "$RESULTS_DIR/scalar.txt"
fi

# ---- Grouped expert kernel (M<=4 only) ----
echo ""
echo "=== Grouped scalar GEMV (${NUM_EXPERTS} experts, M<=4) ==="
printf "%-8s %2s %2s %10s\n" "shape" "k" "M" "avg_us"
echo "---"
if [ -n "$SCALAR_M" ]; then
    run_ncu_bench grouped "kbit_grouped_scalar_gemv" "['moe_gu','moe_dn']" "$SCALAR_M" | tee "$RESULTS_DIR/grouped.txt"
else
    echo "(no M<=4 values requested)" | tee "$RESULTS_DIR/grouped.txt"
fi

# ---- Grouped MMA kernel (all M values) ----
echo ""
echo "=== Grouped MMA (${NUM_EXPERTS} experts, all M) ==="
printf "%-8s %2s %2s %10s\n" "shape" "k" "M" "avg_us"
echo "---"
run_ncu_bench grouped_mma "kbit_grouped_gemm_prod" "['moe_gu','moe_dn']" "$ALL_M" | tee "$RESULTS_DIR/grouped_mma.txt"

# ---- cuBLAS fp16 baselines (CUDA events, all M values) ----
echo ""
echo "=== cuBLAS fp16 (dense mm + MoE bmm) ==="
M_VALS=$ALL_M NUM_EXPERTS=$NUM_EXPERTS python "$SCRIPT_DIR/bench_fp16.py" 2>/dev/null | \
    tee "$RESULTS_DIR/cublas.txt"

# ---- Model-level summary ----
echo ""
echo "=== Qwen3-Coder-Next 70B: weight matmul summary ==="
python3 "$SCRIPT_DIR/model_summary.py" "$RESULTS_DIR"

echo ""
echo "END: $(date)"
