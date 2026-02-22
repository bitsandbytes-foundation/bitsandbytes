# K-bit kernel benchmarking report

RTX 4090 (128 SMs, sm_89), Qwen3-Coder-Next 70B (MoE, hidden_dim=2048).
All kernel times are NCU `gpu__time_duration.avg` unless stated otherwise.

## Kernel dispatch

Four kernels cover the full inference workload. `kbit_linear` and
`kbit_expert_linear` dispatch to the fastest kernel per (layer_type, M):

| Kernel | M range | Layers | Status |
|--------|---------|--------|--------|
| Scalar GEMV | 1-4 | Dense + attention | Done (V8), 1.5-1.9x faster than fp16 at M=1 |
| MMA dequant | 5-16 | Dense + attention | Done, ~1.0-1.3x vs fp16 |
| Dequant + cuBLAS | 17+ | Dense + attention | Done, ~0.95-1.0x vs fp16 |
| Grouped MMA | 1-16 | MoE experts | Done, competitive with fp16 |

All kernels read tiled format (from `repack_kbit`) with E4M4 absmax.

## Per-shape speedups at M=1 (decode, dominant workload)

Best kbit kernel vs cuBLAS fp16, all shapes per transformer block:

| Shape | k=2 | k=3 | k=4 | k=5 |
|-------|-----|-----|-----|-----|
| gateup (2048x5120) | 2.47x | 2.17x | 1.76x | 1.58x |
| down (5120x2048) | 2.05x | 1.84x | 1.57x | 1.42x |
| Q (2048x4096) | 1.90x | 1.67x | 1.43x | 1.28x |
| O (4096x2048) | 2.23x | 2.01x | 1.72x | 1.54x |
| KV (2048x512) | 1.86x | 1.65x | 1.41x | 1.27x |
| moe_gu (2048x512, 8 experts) | ~1.03x | ~1.05x | ~1.03x | ~0.98x |
| moe_dn (512x2048, 8 experts) | ~1.10x | ~1.08x | ~1.05x | ~1.00x |

Dense layers see large speedups because the scalar GEMV reads 2-5x less
data (k-bit compressed weights vs fp16). MoE layers are roughly at parity
because the grouped kernel inner loop has not yet received the V8
optimizations (vectorized A loads, 2-warp config).

## Model size per k

Qwen3-Coder-Next 70B total weight parameters: ~70B.

| k | Bits/param | Model size (weights only) | vs fp16 (140 GB) |
|---|-----------|--------------------------|-------------------|
| 2 | 2 | ~17.5 GB | 8.0x smaller |
| 3 | 3 | ~26.3 GB | 5.3x smaller |
| 4 | 4 | ~35.0 GB | 4.0x smaller |
| 5 | 5 | ~43.8 GB | 3.2x smaller |

At k=2, the entire 70B model fits in a single RTX 4090 (24 GB VRAM) with
room for KV cache. At k=4, it requires ~35 GB which needs multi-GPU or an
80 GB card.

## Deployment speedups (NCU kernel-only, single-user decode)

Single-user inference is dominated by M=1 decode (80-84% of total GEMM
time, from workload analysis in `token_analysis.md`). The weighted per-block
speedup:

| k | Decode speedup (M=1) | Weighted overall (decode + prefill) |
|---|---------------------|-------------------------------------|
| 2 | ~1.90x | ~1.58x |
| 3 | ~1.70x | ~1.45x |
| 4 | ~1.50x | ~1.30x |
| 5 | ~1.35x | ~1.18x |

Prefill uses dequant + cuBLAS, which is slightly slower than pure fp16.
But prefill is infrequent: a typical turn has 1 prefill pass + 114 decode
steps, so the decode speedup dominates.

## Deployment speedups (NCU kernel-only, 4-user vLLM)

With 4 concurrent users in vLLM continuous batching, the M distribution is
bimodal: M=4 for decode-only iterations (92.6% of iterations) and M=4+chunk
for decode+prefill iterations. The scalar kernel handles 59% of GEMM time,
dequant+cuBLAS handles 41%.

| k | 4-user weighted speedup |
|---|------------------------|
| 2 | ~1.58x |
| 3 | ~1.40x |
| 4 | ~1.25x |
| 5 | ~1.12x |

The crossover where quantized kernels become slower than fp16 is at ~16
concurrent users. Below that, bandwidth savings from k-bit compression
outweigh the dequant overhead. Above that, the dequant cost per shape
dominates because most iterations include a large prefill chunk where cuBLAS
is highly efficient.

## Dequant kernel NCU times (bandwidth model at 815 GB/s)

The dequant kernel (`kDequantizeBlockwise_kbit_vec`) reads k-bit packed data
plus absmax and writes fp16 output. Times scale with element count and k:

| Shape | Elements | k=2 | k=3 | k=4 | k=5 |
|-------|----------|-----|-----|-----|-----|
| gateup/down | 10.5M | 29.3 us | 30.5 us | 31.8 us | 33.1 us |
| Q/O | 8.4M | 23.5 us | 24.4 us | 26.1 us | 27.3 us |
| KV | 1.0M | 2.9 us | 3.0 us | 3.2 us | 3.4 us |

k=2 is fastest because it reads only 0.25 bytes/element packed; k=5 reads
0.625 bytes/element. The fp16 output write (2 bytes/element) dominates
bandwidth regardless of k, which is why the spread is only ~15%.

## Issue: Python dispatch overhead in bitsandbytes custom ops

Profiled the per-call overhead of custom CUDA kernels (kbit dequant as the
test case, but this applies to all ops going through `torch.library`). For
a kernel that takes 26 us on-GPU (NCU), the CUDA events end-to-end time is
51 us -- nearly 2x the kernel itself.

Breakdown of the ~25 us overhead:

```
torch.ops dispatch routing:    ~10 us  (library registry lookup, dispatch key resolution)
functional.py wrapper:          ~9 us  (argument reordering, out[:n] slice)
torch._check x 4:              ~5 us  (runtime type/dtype assertions)
torch.empty (16 MB output):    ~4 us  (allocator)
CUDA driver launch:            ~3 us  (kernel submission)
```

For comparison, calling the kernel directly through ctypes (bypassing
`torch.library` entirely) measures 3.3 us overhead -- the raw CUDA driver
launch cost. The remaining 22 us is pure Python/PyTorch framework overhead.

### Why this matters for deployment

At M=1 decode (the dominant workload), a typical Qwen3 transformer block
has 7 weight matmul kernel launches. At 25 us overhead each, that is 175 us
of pure dispatch overhead per block -- comparable to the total kernel
compute time. For the dequant+cuBLAS path (M>16), each shape needs 2
kernel launches (dequant + matmul), doubling the dispatch tax.

### Possible mitigations

1. **CUDA graphs**: capture the dispatch sequence and replay it,
   eliminating per-call Python overhead. Requires static shapes or
   shape-bucketed graphs. This is the standard production solution.
2. **Direct ctypes dispatch**: bypass `torch.library` for hot-path ops.
   Reduces overhead from 25 us to 3 us. Loses `torch.compile`
   compatibility.
3. **Fuse dequant into matmul**: eliminate the separate dequant kernel
   launch entirely for M>16. Requires a custom matmul kernel that reads
   k-bit weights directly (the MMA kernel already does this for M<=16).
4. **Reduce `torch._check` calls**: the 4 runtime assertions add ~5 us.
   These could be gated behind a debug flag.
5. **Eliminate argument reordering**: `functional.py` reorders arguments
   before calling `torch.ops`. Aligning the public API with the internal
   op signature would save ~9 us.

## Conclusions

1. **K-bit quantization provides significant speedups for low-concurrency
   serving.** At k=2, single-user decode is ~1.9x faster than fp16 while
   using 8x less memory. Even k=4 gives 1.5x decode speedup with 4x
   compression.

2. **The sweet spot is 1-4 concurrent users.** The scalar GEMV kernel
   dominates at this scale and is bandwidth-bound -- it directly benefits
   from reading less data. At 16+ users, prefill overhead erodes the
   advantage.

3. **Python dispatch overhead is the next bottleneck.** The 25 us per-call
   overhead nearly doubles the effective kernel time at M=1. Addressing
   this (via CUDA graphs, direct ctypes, or fusing ops) would improve
   end-to-end throughput by up to 1.5x on top of the current kernel
   speedups.

4. **MoE dispatch is unified.** The grouped MMA handles M<=16 for MoE
   layers; for larger M, `kbit_expert_linear` falls back to per-expert
   dequant + cuBLAS matmul. The grouped scalar GEMV was removed (it only
   won one shape at M=1 by 0.3 us).

5. **Lower k is strictly better for inference speed, not just model size.**
   k=2 is fastest at every M value because it reads the least data. The
   accuracy-speed tradeoff is the only reason to use higher k values.
