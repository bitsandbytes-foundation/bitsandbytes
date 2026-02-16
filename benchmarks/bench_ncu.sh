#!/bin/bash
# Full kernel benchmark: MMA + scalar (ncu) + cuBLAS fp16 (CUDA events).
#
# Usage:
#   bash benchmarks/bench_ncu.sh           # default M=1,2,3,4,8
#   M_VALS=3,4 bash benchmarks/bench_ncu.sh  # custom M values
#
# Output: three tables (MMA, scalar, cuBLAS fp16) with avg kernel time
# in microseconds for each shape × k × M combination.
#
# Runtime: ~30-60 seconds depending on M_VALS count.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export M_VALS="${M_VALS:-1,2,3,4,8}"
WARMUP=5
PROFILED=5

echo "START: $(date)"
echo "M values: $M_VALS"

for KERNEL in mma scalar; do
    if [ "$KERNEL" = "mma" ]; then
        KNAME="kbit_gemm_prod"
        echo ""
        echo "=== MMA kernel ==="
    else
        KNAME="kbit_scalar_gemv"
        echo ""
        echo "=== Scalar GEMV ==="
    fi
    printf "%-8s %2s %2s %10s\n" "shape" "k" "M" "avg_us"
    echo "---"

    KERNEL=$KERNEL M_VALS=$M_VALS ncu --kernel-name "$KNAME" --metrics gpu__time_duration.avg \
        python "$SCRIPT_DIR/ncu_driver.py" 2>/dev/null | \
        grep "gpu__time_duration.avg" | awk '{print $NF}' | \
    python3 -c "
import os, sys
vals = [float(l.strip()) for l in sys.stdin]
shapes = ['gateup','down','Q','O','KV']
kbits = [2,3,4,5]
mvals = [int(x) for x in os.environ['M_VALS'].split(',')]
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
done

# cuBLAS fp16 (CUDA events — ncu can't reliably filter cuBLAS kernels)
echo ""
echo "=== cuBLAS fp16 ==="
M_VALS=$M_VALS python "$SCRIPT_DIR/bench_fp16.py" 2>/dev/null

echo ""
echo "END: $(date)"
