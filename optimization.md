# kbit Kernel Optimization Plan

This document describes the kernel strategy for kbit-quantized inference on
RTX 4090 (128 SMs, 72 MB L2, ~1 TB/s DRAM, ~2 TB/s L2 BW). Target models:
Qwen3-Coder-Next (512 experts top-8, hidden=2048) and GLM-4.7-Flash
(64 experts top-4, hidden=2048).

---

## 1. Core Insight: Why the Fused MMA Kernel Underperforms

The fused kbit GEMM kernel reads 3.6x less data than cuBLAS (fp16), but
only achieves 1.6-2x speedup at MoE scale. The missing speedup is explained
by a bandwidth efficiency gap:

| Kernel | Data read | Time | Effective BW | % peak |
|--------|----------:|-----:|-------------:|-------:|
| Grouped GEMM (kbit) | 60 MB | 195us | 308 GB/s | 31% |
| cuBLAS bmm (fp16) | 228 MB | 332us | 687 GB/s | 69% |

*(Qwen3 batch=16, 114 experts, gate/up 2048x512, M=1)*

cuBLAS is 2.2x more bandwidth-efficient, almost exactly cancelling the
3.6x data reduction: 3.6x / 2.2x ≈ 1.6x observed speedup.

Three factors cause the 31% efficiency:

1. **MMA waste at small M.** TILE_M=16 but M=1 → 93.75% of tensor core
   work computes on zero-padded rows. cuBLAS likely uses a scalar GEMV
   internally at M=1, avoiding this waste entirely.

2. **Dequant instruction overhead.** ~1264 instructions per k_tile for
   bit-plane extraction, codebook lookup, and MMA fragment packing. The
   kernel is partially instruction-limited — it can't consume data as fast
   as DRAM delivers it.

3. **Pipeline overhead.** 2-stage cp.async pipeline has fill/drain bubbles
   per work item. With 32 k_tiles per work item, ~6% overhead.

### Dense layers: even worse

For dense layers (single weight matrix, all L2-resident), the fused kernel
never beats cuBLAS. At M=1:

| Shape | Fused kbit | dq+cuBLAS | cuBLAS fp16 |
|-------|----------:|----------:|------------:|
| dense gate/up (2048x5120) | 70us | 85us | 29us |
| dense down (5120x2048) | 70us | 81us | 30us |
| shared gate/up (2048x10240) | 75us | 83us | 55us |
| shared down (10240x2048) | 73us | 83us | 25us |

The production kernel with split-K brings all shapes to ~70-75us (vs
130-325us without split-K), but cuBLAS at 20-55us is still 1.5-3x faster.
Both fused kbit and dequant+cuBLAS converge to similar times (~70-85us)
because the ~42us dequant cost is unavoidable whether fused or separate.
The data fits in L2, so the 3.6x compression provides no bandwidth advantage.

---

## 2. Kernel Strategy

Three kernels cover all regimes optimally:

### Kernel 1: Scalar GEMV (new — highest priority)

For decode (autoregressive generation), M=1-4, both dense and MoE layers.

**Why it wins:** At M=1-4, both our scalar kernel and cuBLAS are
bandwidth-limited. cuBLAS reads fp16 weights; we read 3.6x less kbit data.
No MMA instructions, no fragment packing, no zero-padded rows. Per-element
cost: ~14 simple integer + FMA instructions vs cuBLAS's ~2-3 (FMA only),
but we read 3.6x less data to compensate.

**Projected performance (1.8x overhead factor):**

| Batch | Qwen3 kbit | Qwen3 cuBLAS | Speedup | GLM4.7 kbit | GLM4.7 cuBLAS | Speedup |
|------:|----------:|-----------:|--------:|----------:|-----------:|--------:|
| 1 | 27us | 141us | 5.3x | 37us | 157us | 4.3x |
| 2 | 35us | 147us | 4.2x | 49us | 149us | 3.1x |
| 4 | 50us | 168us | 3.4x | 70us | 330us | 4.7x |

These numbers are per-layer totals (all dense + MoE projections combined).
The 1.8x overhead factor accounts for realistic bandwidth efficiency
(~55% of peak vs cuBLAS's ~69%).

**Architecture:**
- Same persistent kernel shell as grouped GEMM (work distribution, expert
  descriptor lookup)
- Template parameter `ComputeMode::SCALAR` for inner loop
- No shared memory needed for A tiles (M is tiny, load from registers)
- B tiles loaded to shared memory same as MMA path (same bit-plane layout)
- Each thread accumulates scalar FMA: `acc += dequant(B[k]) * A[m][k]`
- Warp-level reduction across K dimension
- Supports both grouped (MoE) and single-matrix (dense) dispatch

**Implementation:** Same kernel file, same grouped dispatch infrastructure.
Add `SCALAR` template specialization for the inner compute loop. When
`max_M <= 4`, dispatch to SCALAR variant.

### Kernel 2: Grouped GEMM (existing)

For MoE expert layers at batch ≥ 8 (decode) and during prefill.

**Why it wins:** At 60+ active experts, total kbit data exceeds L2 cache
and becomes DRAM-bound. Reading 3.6x less data from DRAM saves real time.
The MMA overhead (~2.2x efficiency gap) is partially offset by the 3.6x
compression, giving 1.6-2x over cuBLAS bmm.

Dequant+bmm can't compete at this scale: dequanting 114 experts separately
costs 114 × 42us = 4,788us, and even a hypothetical batched dequant would
materialize 228 MB of fp16 intermediate data that the fused kernel avoids
entirely (total memory traffic: fused 65 MB vs dequant+bmm 521 MB).

**Measured performance:**

| Batch | #experts | Grouped GEMM | cuBLAS bmm | Speedup |
|------:|---------:|-------------:|-----------:|--------:|
| 8 | 61 | 279us | 314us | 1.13x |
| 16 | 114 | 386us | 618us | 1.60x |
| 32 | 203 | 563us | 1060us | 1.88x |
| 64 | 325 | 804us | 1590us | 1.98x |

*(Qwen3 gate/up + down combined)*

**Status:** Implemented and working. No further optimization needed for now.

### Kernel 3: Dequant + cuBLAS (existing pieces)

For dense layers during prefill (M > ~4-8 tokens).

**Why it wins:** cuBLAS is extremely optimized for large-M GEMM, achieving
near-peak tensor core utilization. The dequant kernel runs at 72-78% of
peak bandwidth (42-55us per dense layer). The combination is ~80-90% of
native fp16 cuBLAS speed.

**Important:** The dequant kernel must receive pre-encoded E4M4 absmax
(uint8), not fp32 absmax. Passing fp32 triggers `encode_absmax_e4m4()`
on every call, adding ~800us of overhead. The E4M4 encoding should be
done once at model load time.

**Status:** Both pieces exist. Need dispatch logic to select this path
when M > threshold.

---

## 3. When to Use Each Kernel

### Decode (autoregressive token generation)

| Batch size | Dense layers | MoE expert layers |
|:----------:|:-------------|:------------------|
| 1-4 | Scalar kernel | Scalar grouped kernel |
| 5-7 | Dequant + cuBLAS | Scalar grouped kernel |
| 8+ | Dequant + cuBLAS | Grouped GEMM |

### Prefill (prompt processing, tool-call output)

| Phase | Dense layers | MoE expert layers |
|:------|:-------------|:------------------|
| All M | Dequant + cuBLAS | Grouped GEMM |

During prefill, M is large (hundreds to thousands of tokens). cuBLAS
handles the large-M GEMM optimally. For MoE, tokens are routed to
experts with average M/expert in the tens — grouped GEMM handles this
efficiently.

Prefill also includes mid-generation prefill events: tool-call outputs,
multi-turn continuations, speculative decoding verification. These
typically have M=10-500 tokens and follow the same dispatch logic.

---

## 4. Implementation Priority

### P0: Scalar Kernel

Highest-impact item. Projected 3-5x full-model speedup at batch=1-4
(the autoregressive decode case — the hot path for interactive inference).

**Full implementation guide:** [`agents/scalar_gemv_guide.md`](agents/scalar_gemv_guide.md)

Steps:
1. CUDA kernel in `csrc/ops.cu` — scalar inner loop with cp.async B-tile
   pipeline, A loaded to registers, codebook via `__shfl_sync`
2. C wrappers in `csrc/pythonInterface.cpp`
3. Python op registration and dispatch
4. Correctness tests against dequant + torch.mm reference
5. Benchmark against cuBLAS at M=1,2,4 for all target shapes

### P1: Dispatch Logic

Wire up the three-kernel strategy in the Python layer:
- `kbit_linear(A, W_packed, W_absmax, codebook, ...)` that auto-selects:
  - Scalar kernel when M <= 4
  - Grouped GEMM for MoE expert batches
  - Dequant + cuBLAS when M > threshold for dense layers

### P2: Benchmarking

Full end-to-end model speed comparison:
- Qwen3-Coder-Next per-layer timing at batch=1,2,4,8,16,32,64
- GLM-4.7-Flash per-layer timing at same batch sizes
- Compare: kbit (best kernel per regime) vs fp16 cuBLAS
- Measure across all layers: attention Q/K/V/O + dense MLP + MoE

---

## 5. What We Tried and Why It Doesn't Work

### Fused MMA for dense shapes

The fused kbit GEMM kernel (stages 3-6, production kernel) was designed
for large-N shapes where SM utilization is high. For Qwen3/GLM4.7 dense
layers:
- All weight data fits in L2 (0.5-10.5 MB per layer)
- L2 bandwidth (2 TB/s) means the kernel is instruction-limited, not
  bandwidth-limited
- The 3.6x data compression provides no benefit when data is L2-resident
- MMA overhead + dequant instructions make it 2-3x slower than cuBLAS

Split-K improved the worst cases dramatically (shared down 10240x2048:
318us → 73us) but still can't beat cuBLAS (25us) because the dequant
instruction cost is fundamental.

### MLP fusion (gate/up → SiLU → down)

Considered fusing the full MLP (gate/up projections → SiLU activation →
down projection) into one kernel, similar to Flash Attention. The
intermediate hidden state would stay in registers/shared memory.

Rejected because the intermediate is tiny relative to weights:
- M=1, intermediate_dim=512: hidden = 1 KB vs weights = 1.12 MB (0.09%)
- Flash Attention's intermediate is O(seq²), making fusion critical there
- MLP's intermediate is O(M × intermediate_dim), negligible next to weights

The weight reads completely dominate. Saving 1 KB of intermediate I/O
while reading 1.12 MB of weights provides no meaningful speedup.

---

## 6. Benchmark Reference

### Dequant kernel throughput (from PR #1858)

| K | bits/elem | fp16 (us) | GB/s | % peak BW |
|---|-----------|-----------|------|-----------|
| 2 | 2.25 | 205 | 781 | 78% |
| 3 | 3.25 | 215 | 786 | 78% |
| 4 | 4.25 | 244 | 729 | 72% |
| 5 | 5.25 | 271 | 689 | 68% |

*(67M elements, RTX 4090, E4M4 absmax)*

Per-layer dequant time for target shapes (10.5M elements): ~42-55us.

### Scalar kernel theoretical roofline

RTX 4090: 128 SMs × 128 INT32 cores × 2.52 GHz = 41.3 TOPS INT32.
L2 BW = 2 TB/s. DRAM BW = 1 TB/s.

For one dense layer at M=1 (e.g., gate/up 2048×5120):
- kbit data: 5.7 MB
- L2 read time: 2.85us
- Compute (dequant + FMA): 0.003us (negligible)
- Estimated with 1.8x overhead: ~5.1us
- cuBLAS fp16 same shape: ~25us
- Projected speedup: ~4.9x

The scalar kernel is purely bandwidth-limited. The 3.6x data compression
translates almost directly to speed because the dequant compute is trivially
cheap on scalar INT32 units (~14 ops/element vs 41.3 TOPS throughput).

---

## 7. Files

| File | Purpose |
|------|---------|
| `csrc/ops.cu` | All CUDA kernels (stages 1-6, grouped GEMM, dequant) |
| `bitsandbytes/backends/cuda/ops.py` | Python dispatch for all kbit ops |
| `benchmarks/bench_crossover.py` | Dense crossover + full model speedup |
| `benchmarks/bench_grouped_gemm.py` | Grouped GEMM vs bmm benchmarks |
| `benchmarks/bench_gemv_theoretical.py` | Scalar kernel roofline model |
| `benchmarks/bench_moe_e2e.py` | End-to-end MoE layer timing |
| `progress.md` | Complete development record |
