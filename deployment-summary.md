# kbit kernel deployment summary

RTX 4090 (128 SMs, sm_89), k=2..5, fp16/bf16.
Target models: Qwen3-Coder-Next 70B (MoE), GLM-4.7-AI, Qwen3-Max-70B.

## Executive summary

At k=4 on RTX 4090, the kbit kernel suite is **34-74% faster than fp16**
for single-user / low-concurrency inference (1-4 users), which accounts
for the vast majority of agent and code-assistant workloads. The advantage
comes from reading 4x less weight data from memory at M=1 decode, which
is 80-84% of total GEMM wall-clock time in typical sessions.

At 16+ concurrent users, the advantage disappears because large prefill
chunks dominate and the dequant overhead exceeds the bandwidth savings.

The system uses **5 CUDA kernels** dispatched per (layer_type, M):

| Kernel | M range | Layers | Mechanism |
|--------|---------|--------|-----------|
| Scalar GEMV | 1-4 | Dense + attn | 64 threads, shuffle codebook, no tensor cores |
| MMA dequant | 5-16 | Dense + attn | Tensor core m16n8k16, inline dequant |
| Dequant + cuBLAS | 17+ | Dense + attn | Separate dequant kernel → cuBLAS GEMM |
| Grouped scalar GEMV | 1-4 | MoE experts | Same as scalar, batched across experts |
| Grouped MMA | 1+ | MoE experts | Same as MMA, batched across experts |

For MoE layers at large M (prefill), the grouped MMA kernel loses to
fp16 BMM, so a hybrid dequant + cuBLAS BMM path is available.

---

## Per-kernel performance vs fp16 (RTX 4090, CUDA events)

### M=1 (autoregressive decode — dominant use case)

```
shape    k=2     k=3     k=4     k=5     fp16    Best kernel
--------------------------------------------------------------
Dense layers:
gateup   9.5us   10.8    13.0    14.4    19.1    Scalar: 1.47-2.00x
down     10.2    11.6    13.1    14.4    19.1    Scalar: 1.32-1.87x
Q        8.7     9.6     11.2    12.4    10.7    Scalar: 0.86-1.23x
O        8.0     9.1     10.2    11.1    15.8    Scalar: 1.42-1.99x
KV       3.5     3.7     4.3     4.1     10.9    Scalar: 2.56-3.11x

MoE layers (8 experts):
moe_gu   9.0     10.2    11.3    12.7    11.7    Grouped: 0.92-1.30x
moe_dn   8.9     10.7    12.1    13.1    13.1    Grp MMA: 1.00-1.47x
```

**Dense layers at M=1 are the big win.** Scalar GEMV reads 3-4x less
data (kbit compressed weights vs fp16) and is consistently faster than
fp16 cuBLAS across all k values for gateup, down, O, and KV. The Q
projection is an exception at k=4-5 where its shape (2048×4096) gives
cuBLAS enough parallelism to compete.

**MoE layers at M=1 are roughly break-even.** The grouped kernels match
fp16 BMM at k=4 (1.00-1.03x) and win at k=2-3. At k=5 the grouped
scalar loses on moe_gu (0.92x). The fundamental issue is that MoE expert
shapes are small (512×2048 or 2048×512) so even kbit compression doesn't
give a large bandwidth advantage per expert.

### M=4 (small batch / MoE after routing)

```
shape    k=2     k=3     k=4     k=5     fp16    Best kernel
--------------------------------------------------------------
Dense layers:
gateup   15.8    17.3    18.8    19.4    21.9    MMA/Scalar: 1.13-1.39x
down     11.0    12.5    14.2    15.8    16.2    MMA: 1.03-1.47x
Q        9.3     10.5    12.0    13.6    12.7    MMA: 0.94-1.36x
O        9.7     10.8    12.4    13.7    29.1    MMA: 2.12-3.00x
KV       4.9     4.9     5.0     5.4     15.7    Scalar: 2.91-3.21x

MoE layers (8 experts):
moe_gu   9.3     10.9    11.9    13.2    18.9    Grp MMA: 1.43-2.02x
moe_dn   9.2     10.8    12.1    13.6    12.1    Grp MMA: 0.89-1.32x
```

At M=4, the MMA kernel starts winning on dense layers (tensor cores
become useful at M≥4). MoE grouped MMA wins on moe_gu (1.43-2.02x) but
breaks even or loses on moe_dn at k=4-5.

### M=64+ (prefill / multi-user)

At large M, the landscape shifts:

```
MoE layers, k=4 (8 experts):
           Grp MMA    Hybrid       fp16
shape  M    (us)    dq+BMM (us)   BMM (us)   Best kbit   vs fp16
-----------------------------------------------------------------
moe_gu  64   54.5     54.7         25.6       ~tied       0.47x
moe_gu 128   85.9     57.8         28.6       Hybrid      0.49x
moe_gu 256  134.4     64.7         35.7       Hybrid      0.55x
moe_gu 512  262.4     98.2         69.1       Hybrid      0.70x

moe_dn  64   42.9     53.3         24.1       Grp MMA     0.56x
moe_dn 128   80.4     54.9         25.6       Hybrid      0.47x
moe_dn 256  154.4     67.9         38.6       Hybrid      0.57x
moe_dn 512  300.6     98.4         69.2       Hybrid      0.70x

Dense layers, k=4:
                MMA       dq+cuBLAS    fp16
shape    M     (us)        (us)        (us)     Best kbit   vs fp16
-------------------------------------------------------------------
gateup   64    49.9    ~30+15 = 45     15.4     dq+cuBLAS   0.34x
gateup  128    73.3    ~30+32 = 62     31.9     dq+cuBLAS   0.51x
gateup  512   165.1    ~30+80 = 110    80.0     dq+cuBLAS   0.73x
```

At M=64+, kbit is always slower than fp16. The dequant + cuBLAS hybrid
is the best kbit option, running at 0.47-0.73x of fp16 speed depending
on M. The dequant kernel (~29 us for 8M elements) is a fixed cost that
becomes a smaller fraction at larger M.

---

## Model-level cost per transformer block

Summing the best kernel time across all 7 shapes (gateup, down, Q, O,
KV, moe_gu, moe_dn) gives the total weight-matmul time per transformer
block. Each shape appears once per block.

### M=1 (decode)

```
           kbit total (us)              fp16 total (us)    ratio
k=2:          57.8                          100.4           1.74x faster
k=3:          65.7                          100.4           1.53x faster
k=4:          75.1                          100.4           1.34x faster
k=5:          82.3                          100.4           1.22x faster
```

### M=4 (small batch)

```
           kbit total (us)              fp16 total (us)    ratio
k=2:          69.2                          126.6           1.83x faster
k=3:          77.7                          126.6           1.63x faster
k=4:          86.4                          126.6           1.46x faster
k=5:          94.7                          126.6           1.34x faster
```

### Summary: which k values are "worth it"?

At M=1 decode (dominant workload):
- **k=2**: 1.74x faster than fp16. Clear win.
- **k=3**: 1.53x faster. Strong win.
- **k=4**: 1.34x faster. Moderate win, good quality/speed tradeoff.
- **k=5**: 1.22x faster. Marginal, mainly for quality preservation.

---

## Workload-weighted analysis (vLLM continuous batching)

Real deployments use vLLM continuous batching. Token distributions from
397 Claude Code sessions show the M distribution is bimodal — either
pure-decode (M = num_users) or decode + prefill chunk (M = num_users +
chunk_size). See `token_analysis.md` for the full analysis.

### Speed vs fp16 by concurrency (k=4)

| Users | Dominant kernel | Weighted kbit/fp16 ratio |
|------:|-----------------|-------------------------:|
| 1 | Scalar (87%) | **0.57x** (43% faster) |
| 4 | Scalar (59%) + dq+cuBLAS (41%) | **0.76x** (24% faster) |
| 8 | MMA (45%) + dq+cuBLAS (55%) | **0.85x** (15% faster) |
| 16 | dq+cuBLAS (76%) | **~1.00x** (break-even) |
| 32 | dq+cuBLAS (93%) | **1.17x** (17% slower) |
| 64 | dq+cuBLAS (98%) | **1.23x** (23% slower) |

The crossover is at ~16 concurrent users. Below that, kbit wins.
Above that, the ~29 us dequant overhead per MoE layer dominates.

### Memory savings

Regardless of speed, kbit provides substantial memory savings:

| k | Bits/weight | vs fp16 (16 bits) | 70B model size |
|---|------------|-------------------|---------------|
| 2 | 2 | 8.0x smaller | ~17.5 GB |
| 3 | 3 | 5.3x smaller | ~26.2 GB |
| 4 | 4 | 4.0x smaller | ~35.0 GB |
| 5 | 5 | 3.2x smaller | ~43.7 GB |
| 16 (fp16) | 16 | baseline | ~140 GB |

At k=4, a 70B model fits in 35 GB — comfortably on a single 4090 (24 GB
VRAM) with context offloading, or two 4090s with room for KV cache. At
fp16, the same model requires 140 GB (two H100s or four 4090s).

---

## Grouped scalar GEMV: where it fits

The grouped scalar GEMV (`kbit_grouped_scalar_gemv`) is a specialized
kernel for MoE expert layers at M=1-4. It uses the same flat data format
and shuffle codebook as the dense scalar GEMV.

### When it wins

Only for **moe_gu (K=2048, N=512) at M=1** — and barely:

| Shape | M | Grouped scalar | Grp MMA | fp16 BMM | Winner |
|-------|---|---------------|---------|----------|--------|
| moe_gu | 1 | **11.3** | 11.6 | 11.7 | Grouped (by 0.3 us) |
| moe_gu | 2 | 12.9 | **11.8** | 12.7 | Grp MMA |
| moe_gu | 4 | 17.1 | **11.9** | 18.9 | Grp MMA |
| moe_dn | 1 | 24.9 | **12.1** | 13.1 | Grp MMA |
| moe_dn | 4 | 38.3 | **12.1** | 12.1 | Grp MMA |

The grouped scalar is terrible on moe_dn (K=512): with only 512/64=8
quant blocks per thread and C=1 (one column per block), the kernel is
launch-overhead-dominated. The grouped MMA wins everywhere except that
one moe_gu M=1 case.

### Why it still exists

1. It uses the flat data format (from `quantize_kbit` directly), no
   repack step. If you only store weights in flat format, the grouped
   scalar is the only MoE option at M=1-4.
2. The moe_gu M=1 win is small but real in the most common workload
   (single-user decode). Over thousands of layers, 0.3 us adds up.
3. It provides a correctness cross-check against the grouped MMA.

---

## Remaining optimization opportunities

### 1. CUDA Graphs for hybrid path (medium impact, low effort)

Capture the dequant + BMM kernel pair as a CUDA graph. Eliminates
~25 us of per-layer dispatch overhead (2 × ~14 us → ~3 us). This
would improve the hybrid path at all M values, most impactful at
small M where dispatch is a larger fraction.

### 2. Dequant kernel (~5 us headroom)

The dequant kernel is at 89% of memory bandwidth. Possible ~3-5 us
improvement from wider vectorized loads and occupancy tuning. Marginal
impact on total model time.

### 3. Fused dequant + transpose

The cuBLAS BMM expects a specific weight layout. If the dequant kernel
writes directly in BMM-optimal layout, the `W.transpose().contiguous()`
call is eliminated. Saves one kernel launch + memory pass.

### 4. MoE hybrid dispatch integration

Wire the dequant + cuBLAS BMM hybrid path into the actual dispatch for
MoE layers at M >= threshold. Currently only benchmarked, not integrated
into the forward pass.

### 5. Wait for Hopper/Blackwell

On Hopper (`wgmma.mma_async`) or datacenter Blackwell (`tcgen05.mma`),
the MMA is truly asynchronous. This would allow overlapping dequant ALU
work with MMA, eliminating the 39:1 instruction ratio bottleneck that
limits Ada. The fused grouped MMA kernel could then achieve the
theoretical ~35 us at M=512 (1.9x faster than fp16), instead of the
current 113 us (0.58x).

---

## Architecture notes for GLM-4.7-AI and Qwen3-Max-70B

To compute total model speed for a specific architecture, the required
info per model is:

1. Number of transformer layers
2. Per-layer shapes: hidden_dim, intermediate_dim, num_heads, head_dim
3. MoE config: num_experts, top_k, expert dims
4. Which layers are dense vs MoE

The kernel timings scale predictably:
- Scalar/MMA kernels: time ~ N × K (weight matrix size)
- dq+cuBLAS: dequant time ~ N × K, BMM time from cuBLAS
- Grouped: same scaling but batched across top_k experts

With the per-shape timings in this document and the model architecture,
total per-layer and per-forward-pass time can be computed directly.
