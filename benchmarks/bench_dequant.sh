#!/bin/bash
# Dequant + cuBLAS overhead analysis.
# Uses ncu for accurate dequant kernel timing, CUDA events for matmul.
#
# Usage:
#   bash benchmarks/bench_dequant.sh
#   M_VALS=16,32,64,128,256 bash benchmarks/bench_dequant.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Phase 1: measure dequant kernel times via ncu (all shapes × all k)
echo "Measuring dequant kernel times via ncu..."
DEQUANT_CSV=$(ncu --kernel-name "kDequantizeBlockwise_kbit_vec" \
    --metrics gpu__time_duration.avg \
    python3 -c "
import sys, torch; sys.path.insert(0, '.')
import bitsandbytes
from bitsandbytes.functional import create_normal_float_codebook
shapes = [('gateup',2048,5120),('down',5120,2048),('Q',2048,4096),('O',4096,2048),('KV',2048,512)]
dev = torch.device('cuda')
for k in [2,3,4,5]:
    codebook = create_normal_float_codebook(k, device=dev)
    for name, K, N in shapes:
        n = K * N
        W = torch.randn(n, device=dev)
        packed, absmax = torch.ops.bitsandbytes.quantize_kbit(W, codebook, k)
        torch.cuda.synchronize()
        for _ in range(3):
            torch.ops.bitsandbytes.dequantize_kbit(packed, codebook, absmax, k, n, torch.float16)
        torch.cuda.synchronize()
        torch.ops.bitsandbytes.dequantize_kbit(packed, codebook, absmax, k, n, torch.float16)
        torch.cuda.synchronize()
" 2>&1 | grep "gpu__time_duration" | awk '{print $NF}' | \
python3 -c "
import sys
vals = [float(l.strip()) for l in sys.stdin]
# 4 launches per (k, shape): 3 warmup + 1 profiled, take last
result = []
for i in range(0, len(vals), 4):
    result.append(vals[i+3])
# Output: k=2 × 5 shapes, k=3 × 5, k=4 × 5, k=5 × 5
print(','.join(f'{v:.2f}' for v in result))
")

echo "Dequant kernel times (ncu): $DEQUANT_CSV"
echo ""

# Phase 2: run the Python script with injected dequant times
DEQUANT_CSV="$DEQUANT_CSV" python3 "$SCRIPT_DIR/bench_dequant.py"
