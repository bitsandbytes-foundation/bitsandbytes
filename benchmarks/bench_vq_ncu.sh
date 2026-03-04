#!/bin/bash
# VQ kernel benchmark: scalar GEMV + MMA, with ncu profiling.
# Companion to bench_ncu.sh for kbit kernels.
#
# Usage:
#   bash benchmarks/bench_vq_ncu.sh           # default: scalar M=1, MMA M=5,8,16
#   NCU=1 bash benchmarks/bench_vq_ncu.sh     # enable ncu profiling
#
# Output: raw kernel tables + speedup summary.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/.bench_results"
mkdir -p "$RESULTS_DIR"

USE_NCU="${NCU:-0}"
WARMUP=5
PROFILED=5

echo "START: $(date)"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

if [ "$USE_NCU" = "1" ]; then
    echo "Mode: ncu profiling"

    # ---- VQ scalar GEMV ----
    echo ""
    echo "=== VQ Scalar GEMV (M=1, p=2,4) ==="
    printf "%-8s %2s %2s %10s\n" "shape" "p" "M" "avg_us"
    echo "---"

    for P in 2 4; do
        KERNEL=vq_scalar P_VAL=$P M_VALS=1 \
            ncu --kernel-name "vq_scalar_gemv" --metrics gpu__time_duration.avg \
            python "$SCRIPT_DIR/ncu_vq_driver.py" 2>/dev/null | \
            grep "gpu__time_duration.avg" | awk '{print $NF}' | \
        python3 -c "
import sys
vals = [float(l.strip()) for l in sys.stdin]
shapes = ['gateup','down','Q','O','KV']
W, P = $WARMUP, $PROFILED
i = 0
for s in shapes:
    samples = vals[i+W:i+W+P]
    avg = sum(samples)/len(samples) if samples else 0
    print(f'{s:<8} {$P:>2} {1:>2} {avg:>10.2f}')
    i += W + P
" | tee -a "$RESULTS_DIR/vq_scalar_p${P}.txt"
    done

    # ---- VQ MMA kernel ----
    echo ""
    echo "=== VQ MMA Kernel (M=5,8,16, p=2,4) ==="
    printf "%-8s %2s %2s %10s\n" "shape" "p" "M" "avg_us"
    echo "---"

    for P in 2 4; do
        KERNEL=vq_mma P_VAL=$P M_VALS=5,8,16 \
            ncu --kernel-name "vq_gemm_prod" --metrics gpu__time_duration.avg \
            python "$SCRIPT_DIR/ncu_vq_driver.py" 2>/dev/null | \
            grep "gpu__time_duration.avg" | awk '{print $NF}' | \
        python3 -c "
import sys
vals = [float(l.strip()) for l in sys.stdin]
shapes = ['gateup','down','Q']
mvals = [5, 8, 16]
W, P = $WARMUP, $PROFILED
i = 0
for s in shapes:
    for m in mvals:
        samples = vals[i+W:i+W+P]
        avg = sum(samples)/len(samples) if samples else 0
        print(f'{s:<8} {$P:>2} {m:>2} {avg:>10.2f}')
        i += W + P
" | tee -a "$RESULTS_DIR/vq_mma_p${P}.txt"
    done

else
    echo "Mode: CUDA graph timing (no ncu)"
    echo ""

    # Run the Python benchmark directly
    python "$SCRIPT_DIR/bench_vq_codebook.py" --inner 500 --outer 15 \
        --output "$RESULTS_DIR/vq_bench.json" | tee "$RESULTS_DIR/vq_bench.txt"
fi

echo ""
echo "END: $(date)"
