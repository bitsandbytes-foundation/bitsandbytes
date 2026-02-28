# GLM-4.7 355B MoE: Streaming QLoRA Training Analysis

Complete simulation of QLoRA fine-tuning with weight streaming for the GLM-4.7
355B MoE model, calibrated against RTX 4090 matmul benchmarks. Covers memory
budgets, compute/transfer overlap, quantization trade-offs, NVFP4 on Blackwell,
and hardware build recommendations.

**Key finding**: With an optimal resident/batch split, streaming overhead is
**0%** across all tested configurations. The trick is counter-intuitive: evict
layers from VRAM to make room for larger batches, so compute dominates transfer.

## Model Architecture

Source: [GLM-4.7 config.json](https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json)

| Parameter | Value |
|---|---|
| Layers | 92 (3 dense + 89 MoE) |
| Hidden size | 5120 |
| Attention heads | 96 Q, 8 KV (GQA), head_dim=128 |
| Shared expert intermediate | 12288 |
| Routing expert intermediate | 1536 |
| Experts | 160 total, 8 active per token + 1 shared |
| Total params/layer | 4.10B |
| Active params/layer | 515M |
| Expert fraction | 92.1% of total params |

The MoE architecture creates a fundamental tension for streaming: each layer
transfers ~1.2-2.3 GB (all 160 experts), but only ~12.5% (8 active + shared +
attention) contributes to compute. This poor weight-to-compute ratio makes naive
streaming heavily transfer-bound.

## Quantization Formats

All layer sizes are empirical (validated against actual quantized weights) or
derived from cross-checked implied bits-per-param.

| Format | Layer size | Total (92 layers) | Description |
|---|---|---|---|
| NF4 | 2250 MB | 202 GB | Standard 4-bit NormalFloat |
| NF3 | 1640 MB | 147 GB | 3-bit NormalFloat |
| NF2 | 1150 MB | 103 GB | 2-bit NormalFloat |
| NF4d+NF2e | 1237 MB | 111 GB | NF4 dense + NF2 experts (best size/quality) |
| NF4d+NF3e | 1690 MB | 152 GB | NF4 dense + NF3 experts (better quality) |
| NVFP4 | 2100 MB | 189 GB | Blackwell HW FP4 (MXFP4, 4.25 bpp) |

Cross-check: NF4 at 2250 MB/layer implies 4.60 bits/param. NF4d+NF2e predicted
from component bits (dense@4.60 + expert@2.35) = 1237 MB, matching the empirical
value exactly.

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | 64 |
| Adapted projections | 7 per layer (Q, K, V, O + gate, up, down of shared expert) |
| Routing experts | NOT adapted (160 experts would explode param count) |
| Params/layer | 6.36M |
| GPU memory/layer | 101.7 MB (weights bf16 + grads bf16 + AdamW fp32 states) |
| Total LoRA GPU footprint | 9.36 GB (all 92 layers) |

## The Resident/Batch Trade-Off

This is the central insight of the analysis.

### The problem with "greedy" (maximize resident layers)

The naive approach packs as many layers as possible into GPU VRAM to minimize
streaming. This leaves almost no VRAM for activations, forcing tiny batch sizes:

```
1x RTX 4090, NF4d+NF2e, Gen4x1 NVMe:
  Greedy:  Res=8, Str=84, B=3   -> compute=11s, transfer=29s -> 161% OH, 106 tok/s
  Optimal: Res=1, Str=91, B=51  -> compute=189s, transfer=31s ->   0% OH, 277 tok/s
```

The greedy approach keeps 8 layers on-GPU and streams 84. But with only B=3,
the GPU finishes computing each layer so fast (11s total) that it sits idle
waiting for the next layer to transfer (29s). Result: 161% streaming overhead.

### Why evicting layers makes you faster

Every byte of VRAM has two competing uses:

1. **Hold a resident layer** — saves one layer's transfer time
2. **Hold activations** — enables larger batch size for ALL layers

When you evict one layer (~1.2 GB for NF4d+NF2e), you gain ~7 more samples in
the batch (at S=1024, activation memory is ~150 MB per sample per layer). Those
7 extra samples increase compute time for every layer — resident and streamed
alike. The compute increase easily exceeds the transfer time for the one evicted
layer.

The optimal point is where compute just exceeds transfer for the streamed layers
(typically 0-2 resident layers). Beyond that, additional resident layers waste
VRAM that could be doing useful compute.

### When residents help

Residents are only beneficial when:
- Batch size is already at maximum (capped at 256 or by convergence needs)
- Transfer STILL exceeds compute at max batch
- This only happens on very fast GPUs with very slow storage

In practice, the optimal sweep almost always lands at Res=0-2.

### Trade-off curve example

```
1x RTX 4090 | NF4d+NF2e | Gen4x1, 32G RAM

 Res  Str  Free   B   Compute  Transfer  Step    OH%   tok/s
   0   92  10.4G  58    214.5s    31.8s  214.5s    0%   277
   1   91   9.2G  51    188.6s    31.4s  188.6s    0%   277  <- OPTIMAL
   2   90   8.0G  44    162.7s    31.1s  162.7s    0%   277
   3   89   6.7G  37    136.8s    30.7s  136.8s    0%   277
   4   88   5.5G  31    114.6s    30.4s  114.6s    0%   277
   5   87   4.3G  24     88.8s    30.0s   88.8s    0%   277
   6   86   3.1G  17     62.9s    29.7s   62.9s    0%   277
   7   85   1.9G  10     37.0s    29.3s   37.0s    0%   277
   8   84   0.7G   3     11.1s    29.0s   29.0s  161%   106  <- GREEDY (2.6x slower)
```

Throughput is flat at 277 tok/s from Res=0 through Res=7 (compute always
dominates transfer). At Res=8, B drops to 3 and the system flips to
transfer-bound with 161% overhead.

## GPU Utilization Calibration

The simulation uses a GPU utilization parameter to translate FLOPs into
wall-clock time. We benchmarked this on an RTX 4090 (CUDA 12.8, PyTorch 2.9).

### BF16 matmul benchmark (RTX 4090, 165 TFLOPS peak)

Using CUDA graphs, measuring total forward pass FLOPs at GLM-4.7 dimensions:

| B | M=B*S | Total ms | TFLOPS | Utilization |
|---|---|---|---|---|
| 1 | 1024 | 7.9 ms | 139.6T | 84.6% |
| 2 | 2048 | 14.7 ms | 149.9T | 90.8% |
| 4 | 4096 | 27.8 ms | 158.9T | 96.3% |
| 8 | 8192 | 56.4 ms | 156.5T | 94.8% |
| 16 | 16384 | 112.5 ms | 157.0T | 95.1% |
| 32 | 32768 | 222.4 ms | 158.8T | 96.2% |

CUDA graphs provide no speedup (<1% difference) at these matrix sizes — kernel
launch overhead is negligible for large matmuls.

### NF4 dequant + matmul benchmark

NF4 dequantization adds minimal overhead at training-relevant batch sizes:

| B | NF4 Utilization | NF4/BF16 ratio |
|---|---|---|
| 1 | 81.3% | 1.20x slower |
| 4 | 93.8% | 1.04x slower |
| 8 | 95.6% | 1.03x slower |
| 16 | 96.6% | 1.02x slower |

At B>=4, NF4 dequant adds only 2-4% overhead — the dequant kernel runs
concurrently with the matmul compute on the SMs.

### Simulation parameter

The simulation uses **70% utilization** as a conservative end-to-end estimate:
- Isolated NF4 matmuls: 81-97% (benchmarked above)
- Non-matmul ops (layernorm, softmax, activation fn): -5%
- Training loop / scheduling overhead: -5-10%
- Gradient checkpoint recomputation scheduling: -5%

This calibration means simulated throughput is within ~15% of reality, erring
on the conservative side.

## Simulation Results

All results use optimal resident/batch split, seq_len=1024, 70% GPU utilization.

### GPU Hardware

| GPU | VRAM | BF16 TFLOPS | PCIe | Price (Feb 2026) |
|---|---|---|---|---|
| RTX 4090 | 24 GB | 165 | Gen4 x16, 22 GB/s | ~$1,800 used |
| RTX 5090 | 32 GB | 209 | Gen5 x16, 44 GB/s | ~$2,900 new |
| A100 80G | 80 GB | 312 | Gen4 x16, 22 GB/s | ~$12,000 |
| H100 80G | 80 GB | 756 | Gen5 x16, 44 GB/s | ~$25,000 |
| RTX PRO 6000 | 96 GB | 300 | Gen5 x16, 44 GB/s | ~$8,000 |

### Single-GPU throughput (NF4d+NF2e, 1237 MB/layer)

| GPU | B | Res | Str | Compute | Transfer | OH% | tok/s |
|---|---|---|---|---|---|---|---|
| RTX 4090 | 51 | 1 | 91 | 189s | 31s | 0% | **277** |
| RTX 5090 | 89 | 2 | 90 | 260s | 31s | 0% | **351** |
| A100 80G | 155 | 32 | 60 | 303s | 21s | 0% | **524** |
| H100 80G | 19 | 52 | 40 | 15s | 14s | 0% | **1269** |
| RTX PRO 6000 | 35 | 63 | 29 | 71s | 10s | 0% | **503** |

Storage shown: Gen4x1 (7 GB/s NVMe), 32G RAM. All configs achieve 0% streaming
overhead with the optimal split. The NVMe speed doesn't affect throughput because
compute dominates in every case.

### NF4d+NF3e vs NF4d+NF2e (NF3 for experts instead of NF2)

| GPU | NF4d+NF2e tok/s | NF4d+NF3e tok/s | Impact |
|---|---|---|---|
| RTX 4090 | 277 | 277 | None |
| RTX 5090 | 351 | 351 | None |
| H100 80G | 1269 | 1269 | None |

Using NF3 instead of NF2 for routing experts increases layer size by 37%
(1237 -> 1690 MB) but has **zero throughput impact** with the optimal strategy.
Transfer time grows but remains fully hidden behind compute. The benefit of NF3
is purely in model quality (less quantization error) with no training speed cost.

### NVFP4 on Blackwell (RTX 5090)

NVFP4 (MXFP4) uses Blackwell's native FP4 tensor cores. Benchmarked kernel-level
speedup vs BF16 cuBLAS:

| M (tokens) | Best implementation | Speedup vs BF16 |
|---|---|---|
| 1-16 | HW (hardware path) | 1.20-1.48x |
| 64-256 | CL (custom library) | 1.31-3.32x |
| 1024 | CL | 3.27x |
| 4096 | CL | 3.76x |

At training-relevant batch sizes (M >= 1024), the speedup is consistently ~3-3.8x.

**Effective layer-level speedup**: Not all FLOPs benefit from FP4 tensor cores.
Attention QK^T/score*V (4.7% of layer FLOPs) remains BF16. For GLM-4.7 at
S=1024: effective speedup = **2.74x** (at 3x raw kernel speedup).

| Config (1x RTX 5090, optimal) | NF4d+NF2e | NVFP4 | Speedup |
|---|---|---|---|
| Gen5 AICx4 | 351 tok/s | **961 tok/s** | 2.74x |
| Gen4x1 | 351 tok/s | **961 tok/s** | 2.74x |

NVFP4 trade-off curve on RTX 5090:

```
 Res  Str  Free   B   Compute  Transfer  Step    OH%   tok/s
   0   92  16.7G  93    99.1s     8.6s   99.1s    0%    961
   1   91  14.6G  82    87.4s     8.5s   87.4s    0%    961  <- OPTIMAL
   ...
   7   85   2.3G  13    13.9s     7.9s   13.9s    0%    961
   8   84   0.3G   1     1.1s     7.8s    7.8s  635%    131  <- GREEDY CLIFF
```

The cliff at Res=8 is dramatic: B drops from 13 to 1, compute drops from 13.9s
to 1.1s, but transfer stays at 7.8s. The GPU finishes its work in 1 second and
waits 7 seconds for the next layer. NVFP4's faster compute makes the greedy
penalty **more** severe, not less.

### Multi-GPU with pipeline parallelism

With G GPUs in a pipeline:
- Each GPU handles ceil(92/G) layers
- M micro-batches fill the pipeline
- NVMe bandwidth is shared across all GPUs
- PCIe bandwidth is per-GPU (separate x16 links)

The NVMe sharing means each GPU sees half the read bandwidth. Since compute per
GPU also halves (fewer layers), you need **2x the micro-batch size** to maintain
the same compute/transfer ratio.

| Config | B | Res | Str | Compute | Transfer | OH% | tok/s |
|---|---|---|---|---|---|---|---|
| 1x RTX 4090, NF4d+NF2e | 51 | 1 | 91 | 189s | 31s | 0% | 277 |
| 2x RTX 4090, NF4d+NF2e | 82 | 0 | 46 | 607s | 127s | 0% | 443 |
| 4x RTX 4090, NF4d+NF2e | 81 | 2 | 21 | 599s | 232s | 0% | 805 |
| 1x RTX 5090, NVFP4 | 82 | 1 | 91 | 87s | 8.5s | 0% | 961 |
| 2x RTX 5090, NVFP4 | 37 | 7 | 39 | 79s | 46s | 0% | 1538 |
| 1x H100 80G, NF4d+NF2e | 19 | 52 | 40 | 15s | 14s | 0% | 1269 |
| 4x H100 80G, NF4d+NF2e | 256 | 0 | 23 | 413s | 254s | 0% | 3691 |

Multi-GPU configs use M = max(2*G, 4) micro-batches. Pipeline bubble overhead
is (G-1)/(M+G-1) and included in the step time. Storage: Gen4x1 for Gen4 GPUs,
Gen5 AICx4 for Gen5 GPUs, 32G RAM.

### Consumer AM5 x8/x8 has no impact

Consumer AM5 motherboards split PCIe into x8/x8 when two GPUs are installed:

| Config | PCIe per GPU | tok/s | vs x16/x16 |
|---|---|---|---|
| 2x RTX 4090 @ x16/x16 (Gen4) | 22 GB/s | 443 | baseline |
| 2x RTX 4090 @ x8/x8 (Gen4) | 11 GB/s | 443 | identical |
| 2x RTX 5090 @ x8/x8 (Gen5) | 22 GB/s | 561 | n/a |
| 2x RTX 5090 @ x8/x8 (Gen5) + NVFP4 | 22 GB/s | 1538 | n/a |

For RTX 4090s: NVMe is the bottleneck (3.5 GB/s per GPU), not PCIe, so halving
PCIe makes no difference. For RTX 5090s: Gen5 x8 = 22 GB/s, matching Gen4 x16.

**No Threadripper needed.** A $320 X870E AM5 board works as well as a $1,800
Threadripper platform for dual-GPU streaming.

## Hardware Build Recommendations

Component prices as of February 2026. DDR5 RAM is severely inflated due to an
ongoing DRAM shortage (64GB kits that were $200 in mid-2025 are now $400+).

### Build A: Budget Single GPU — ~$2,700

| Component | Choice | Price |
|---|---|---|
| GPU | 1x RTX 4090 (used) | $1,800 |
| Motherboard | B650 / X670E AM5 | $150 |
| CPU | Ryzen 5 7600 | $180 |
| RAM | 32GB DDR5 (2x16) | $200 |
| NVMe | 2TB Gen4 (Samsung 990 Pro) | $170 |
| PSU | 850W | $120 |
| Case | Mid-tower | $80 |
| **Total** | | **~$2,700** |

**277 tok/s** at $9.75/tok/s. The cheapest viable setup. 32GB system RAM is
enough — weights stream from NVMe and the optimal batch split ensures 0%
overhead even at Gen4 speeds (7 GB/s). 64GB RAM ($400) is a comfort option for
dataset loading but does not change training throughput.

### Build B: Dual 4090 Consumer — ~$5,000

| Component | Choice | Price |
|---|---|---|
| GPU | 2x RTX 4090 (used) | $3,600 |
| Motherboard | X870E with dual x16 slots | $320 |
| CPU | Ryzen 7 9700X | $300 |
| RAM | 32GB DDR5 (2x16) | $200 |
| NVMe | 2TB Gen4 | $170 |
| PSU | 1600W | $250 |
| Case | Full tower (2x 3-slot GPUs) | $160 |
| **Total** | | **~$5,000** |

**443 tok/s** at $11.29/tok/s. Runs x8/x8 on AM5 — irrelevant because NVMe
(3.5 GB/s per GPU) is the bottleneck, not PCIe. Simulation confirms identical
throughput at x8 vs x16.

### Build C: Single RTX 5090 — ~$4,200

| Component | Choice | Price |
|---|---|---|
| GPU | 1x RTX 5090 | $2,900 |
| Motherboard | X870E AM5 | $320 |
| CPU | Ryzen 7 9700X | $300 |
| RAM | 32GB DDR5 | $200 |
| NVMe | 2TB Gen5 (Crucial T705) | $225 |
| PSU | 1000W | $150 |
| Case | Mid-tower | $100 |
| **Total** | | **~$4,200** |

**351 tok/s** (NF4d+NF2e) or **961 tok/s** (NVFP4) at $4.37/tok/s. If NVFP4
support ships in bitsandbytes, this is the best value by a wide margin — a
single consumer GPU matching H100-class throughput.

### Build D: Dual RTX 5090 — ~$7,600

| Component | Choice | Price |
|---|---|---|
| GPU | 2x RTX 5090 | $5,800 |
| Motherboard | X870E with dual x16 | $320 |
| CPU | Ryzen 9 9900X | $400 |
| RAM | 64GB DDR5 (2x32) | $400 |
| NVMe | 2TB Gen5 (Crucial T705) | $225 |
| PSU | 1600W | $300 |
| Case | Full tower | $160 |
| **Total** | | **~$7,600** |

**561 tok/s** (NF4d+NF2e) or **1538 tok/s** (NVFP4) at $4.94/tok/s. Gen5 x8
on AM5 gives 22 GB/s per GPU, matching the simulation's assumptions.

### Value comparison

| Build | Cost | tok/s | $/tok/s | Time for 50M tokens |
|---|---|---|---|---|
| A: 1x 4090 (used) | $2,700 | 277 | **$9.75** | 50 hrs |
| B: 2x 4090 (used) | $5,000 | 443 | $11.29 | 31 hrs |
| C: 1x 5090 | $4,200 | 351 | $11.97 | 40 hrs |
| C: 1x 5090 + NVFP4 | $4,200 | **961** | **$4.37** | **14 hrs** |
| D: 2x 5090 + NVFP4 | $7,600 | 1538 | $4.94 | 9 hrs |

### Recommendations

**Cheapest**: Build A ($2,700). A used 4090 on a minimal AM5 system. Even a
single Gen4 NVMe provides zero streaming overhead.

**Best value if NVFP4 ships**: Build C ($4,200). One RTX 5090 at 961 tok/s
delivers H100-class throughput at 1/6 the system cost.

**Avoid**: Threadripper/Xeon workstation platforms. The $1,800+ premium for true
x16/x16 PCIe is wasted — the optimal batch strategy achieves 0% streaming
overhead even at x8 bandwidths, and NVMe is the real bottleneck for multi-GPU.

## Simulation Code

- `streaming_sim.py` — Complete simulation with all hardware configs, quantization
  formats, and the optimal resident/batch sweep.
- `bench_matmul.py` — BF16 and NF4 matmul benchmarks for GPU utilization
  calibration. Run on your hardware to validate the 70% utilization assumption.

```bash
# Run full simulation
python docs/streaming_analysis/streaming_sim.py

# Validation only
python docs/streaming_analysis/streaming_sim.py --validate-only

# Matmul benchmark (requires CUDA GPU)
python docs/streaming_analysis/bench_matmul.py
```

## Assumptions and Limitations

1. **Activation memory model** uses analytical estimates with 20% fragmentation
   overhead and 20% allocator overhead. Real PyTorch allocation patterns may
   differ by 10-20%.

2. **GPU utilization at 70%** is calibrated on RTX 4090 NF4 benchmarks.
   Different GPUs may achieve different utilization. Run `bench_matmul.py` on
   your hardware to check.

3. **NVFP4 speedup of 2.74x** assumes 3x raw kernel speedup (from Blackwell
   FP4 benchmarks) reduced by the BF16 attention compute fraction. Actual
   end-to-end speedup depends on the NVFP4 kernel implementation in
   bitsandbytes.

4. **Total params (377B vs 355B)**: Our architecture calculation gives 377B,
   6.3% above the model card's 355B. The difference is likely a counting
   convention (e.g., embedding layers, layer norms). This affects FLOPs
   proportionally but not the streaming overhead conclusions.

5. **Pipeline parallelism bubble** is modeled as (G-1) idle stages. Real 1F1B
   scheduling may achieve slightly better utilization.

6. **NVMe bandwidth** assumes sustained sequential read. Random reads or
   fragmented files will be slower. Use direct I/O and contiguous files.

7. **Component prices** are snapshot values from February 2026 and will change.
   DDR5 prices are particularly volatile due to the ongoing DRAM shortage.
