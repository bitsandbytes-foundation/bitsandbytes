# MPS Backend Status — Phase 1 audit + Phase 2 first native kernel

**Date:** 2026-07-08 (Phases 1–3) · 2026-07-14 (Phases M1–M4, 4-bit matmul) ·
2026-09-03 (Phase M5, bf16 gemm — §11.5) ·
**Branch:** `feature/mps-metal-kernels` (base: `777c145`), then `feature/mps-matmul`,
then `feat/mps-gemm-4bit-bf16`
**Machine:** Apple Silicon (arm64), macOS **26.4.1** (M5: macOS **26.5**, M4 Max)
**Stack:** Python 3.14.2 · torch **2.12.1** · bitsandbytes 0.50.0.dev0
**Harness:** `tests/test_mps_parity.py` — Phase-1 baseline (no native build): **183 passed, 1 xfailed
(strict), 0 skipped**. Phase-3 source build (`-DCOMPUTE_BACKEND=mps`, `BNB_MPS_REQUIRE_NATIVE=1`):
**293 passed, 1 xfailed**, with `quantize_blockwise`, `dequantize_blockwise`, `dequantize_4bit`, and
`quantize_4bit` all running through hand-written Metal and bit-exact vs the CPU oracle. Post-M4
build (`-DCOMPUTE_BACKEND=mps`, `BNB_MPS_REQUIRE_NATIVE=1`): **329 passed, 0 xfailed** — the lion
weight-decay divergence that was a strict `xfail` is now fixed upstream and is a passing regression
test (§5), and the 4-bit matmuls are native per §11.

This is the Phase-1 deliverable (§3 audit) plus the Phase-2 result (first native kernel end to end).
Re-verify against `bitsandbytes/_ops.py` and `bitsandbytes/backends/mps/ops.py` before trusting this
after a rebase. The Phase-2 native path is described in §7 below.

---

## 1. How an op resolves on the `mps` device

Three tiers, checked in order by the torch dispatcher:

1. **`mps` registration** (`bitsandbytes/backends/mps/ops.py`) — each such kernel first
   tries the **HuggingFace Hub kernel** (`kernels-community/bitsandbytes-mps`, gated to
   macOS ≥ 26 _and_ requiring the `kernels` package), else falls back to a pure-PyTorch
   implementation executed on mps tensors.
2. **`default` registration** (`bitsandbytes/backends/default/ops.py`) — pure-PyTorch,
   device-agnostic; runs on mps tensors through PyTorch's aten MPS kernels.
3. **Nothing registered** → `NotImplementedError` at call time.

### What actually runs on THIS machine

- **Hub kernels: NEVER run here.** macOS 26.4.1 passes the version gate, but the
  `kernels` package is **not installed**, so `_get_kernel()` fails its import once and
  latches `_kernel_load_failed = True` for the process. Every "Hub-first" op silently
  uses its pure-torch fallback. To exercise the Hub path: `pip install kernels` and
  re-run the parity harness (the same tests then cover it, blocksize-gated).
- **bitsandbytes-native Metal kernels: do not exist yet** (see §4). Nothing in this
  audit exercises `csrc/mps_ops.mm` / `csrc/mps_kernels.metal`.
- **No silent CPU fallback.** `PYTORCH_ENABLE_MPS_FALLBACK` is unset, so any aten op
  missing on MPS would raise instead of quietly routing to CPU. All "parity green"
  results below therefore represent genuine execution on the MPS device (via aten MPS
  kernels) — but **zero** of them represent bitsandbytes-native Metal coverage. This is
  a fallback-quality baseline, which is exactly what Phases 2–3 replace.
- **CPU oracle = `default` backend.** On this source checkout the native library is the
  error-handler mock (`cextension.lib` is `ErrorHandlerMockBNBNativeLibrary`) and the
  host is aarch64 (no AVX512), so none of the lib-gated `cpu` registrations for the
  quant ops exist; the `cpu` device resolves to the same `default` pure-torch kernels.
  Exception: `optimizer_update_32bit` and `optimizer_update_8bit_blockwise` have
  unconditional `cpu` registrations (`backends/cpu/ops.py`), which is how the lion
  divergence in §5 was caught.
- **torch.compile:** the `_try_torch_compile` wrappers compile successfully; alternating
  cpu/mps calls trips dynamo's recompile limit (8) after which execution transparently
  falls back to eager. No correctness impact observed.

---

## 2. Op-by-op matrix (mps device, this machine)

Parity = max deviation vs the CPU oracle with seeded inputs (see §3 for tolerances).

| Op (`bitsandbytes::…`)            | `mps` reg?                     | Path that runs here                  | Parity vs CPU oracle                               |
| --------------------------------- | ------------------------------ | ------------------------------------ | -------------------------------------------------- |
| `quantize_blockwise`              | ✅ **native Metal** (P2)       | hand-written kernel (fallback avail) | codes **bit-exact**, absmax **bit-exact**          |
| `dequantize_blockwise`            | ✅ **native Metal** (P3)       | hand-written kernel (fallback avail) | **bit-exact** all dtypes/blocksizes                |
| `dequantize_blockwise.out`        | ❌ (cuda/xpu only, no default) | **`NotImplementedError`**            | — (gap)                                            |
| `quantize_4bit`                   | ✅ **native Metal** (P3)       | hand-written kernel (fallback avail) | packed nibbles **bit-exact**, absmax **bit-exact** |
| `dequantize_4bit` (+`.out`)       | ✅ **native Metal** (P3)       | hand-written kernel (fallback avail) | **bit-exact** all dtypes/blocksizes                |
| `gemv_4bit` (+`.out`)             | ✅ **native Metal** (M2)       | fused dequant+dot kernel (§11.1)     | within per-dtype tolerances (§3); fallback avail   |
| `gemm_4bit`                       | ✅ **native Metal** (M3/M5)    | dequant→scratch + MPSMatMul/Graph    | within per-dtype tolerances, **all 3 dtypes**      |
| `int8_linear_matmul` (+`.out`)    | ❌ → `default`                 | fp32 matmul on MPS                   | exact (int32)                                      |
| `int8_vectorwise_quant`           | ❌ → `default`                 | pure-torch on MPS                    | exact (incl. outlier extraction, threshold=6)      |
| `int8_vectorwise_dequant`         | ❌ → `default`                 | pure-torch on MPS                    | exact                                              |
| `int8_mm_dequant`                 | ❌ → `default`                 | pure-torch on MPS                    | exact                                              |
| `int8_scaled_mm`                  | ❌ → `default`                 | composition of the above             | exact                                              |
| `int8_mixed_scaled_mm`            | ❌ → `default`                 | composition of the above             | covered via components                             |
| `int8_double_quant`               | ❌ (cuda only, no default)     | **`NotImplementedError`**            | — (gap; also unavailable on cpu)                   |
| `optimizer_update_32bit`          | ❌ → `default`                 | pure-torch on MPS                    | exact (incl. lion + weight_decay since #1992; §5)  |
| `optimizer_update_8bit_blockwise` | ❌ (cpu/cuda/xpu, no default)  | **`NotImplementedError`**            | — (gap: 8-bit optimizers unusable on mps)          |

**Round-trip reconstruction** (quantize→dequantize on MPS vs same on CPU, seeded randn,
blocksize ∈ {64, 128, 256, 512}, dtypes fp32/fp16/bf16):

- blockwise-int8 (dynamic map): mean abs error ~1e-2 on both devices, **identical** to
  the oracle (codes bit-exact ⇒ reconstruction bit-exact).
- NF4 / FP4: max abs error ~0.55–0.71 and mean ~6e-2 on randn — the expected 4-bit
  quantization error — **identical** on CPU and MPS, including tail (partial-block)
  handling with numel % blocksize ≠ 0.

No "confident garbage" was observed anywhere in the current fallback stack.

---

## 3. Tolerances (documented, empirically calibrated)

Used by `tests/test_mps_parity.py::assert_parity`; per-dtype (rtol, atol), CT2-style
(fp32 tight, halves looser). Measured headroom on this baseline is large — fp32 matmul
divergence is accumulation-order only (≤ ~8e-6 at K ≤ 256); the looser fp16/bf16 bounds
are chosen so the same harness keeps working when native Metal kernels (fast-math,
different accumulation order) replace the fallbacks in Phase 2+.

| dtype | rtol | atol | observed baseline max deviation |
| ----- | ---- | ---- | ------------------------------- |
| fp32  | 1e-6 | 1e-5 | 7.7e-6 (gemv), 3.9e-6 (gemm)    |
| fp16  | 1e-3 | 1e-2 | 0.0                             |
| bf16  | 1e-2 | 4e-2 | 0.0                             |

Additionally:

- **Quantized artifacts (uint8 codes, packed nibbles) must be bit-exact** — a mismatch
  is a wrong bucket, not a rounding difference (`assert_bit_exact`).
- `absmax`/statistics: fp32 tolerance (observed exact — both paths compute absmax in fp32).
- int8/int32 outputs: exact equality.

fp16/bf16 measuring 0.0 today is _not_ an accident to rely on: both matmul fallback
paths dequantize to the activation dtype and run `F.linear`, whose MPS and CPU results
round identically at these small K. Native kernels will not have this property; the
documented tolerances above are the contract.

---

## 4. Native (`csrc`) path — confirmed doubly dead

Verified against the plan's §1 claims, at `777c145`:

- `csrc/mps_ops.mm` (62 lines): `quantize_mps` is `NSLog(@"Not implemented"); return nil;`.
  `get_library()` loads `bitsandbytes.metallib` by **CWD-relative path** (line 33) —
  will not survive an installed package; must be resolved relative to the dylib/package
  dir in Phase 2.
- `csrc/mps_kernels.metal` (117 lines): exactly one kernel (`quantize`, scalar binary
  search into a 256-entry code table). Its math predates the current op registry and is
  **unvalidated** — validate against the CPU code table before using it as the Phase-2
  starting point.
- `metallib` appears **nowhere** in `bitsandbytes/` Python: no loader, no packaging
  reference. `cextension.py` only handles CUDA/ROCm/XPU libraries; on this machine it
  yields the error-handler mock.
- CMake scaffolding (`-DCOMPUTE_BACKEND=mps` → `libbitsandbytes_mps.dylib` +
  `bitsandbytes/bitsandbytes.metallib`) exists but was **not** built or exercised in
  Phase 1 (per plan: Phase 1 audits the existing backend; no native build required).

---

## 5. Findings / divergences

1. **Lion weight-decay semantics — RESOLVED (was a cross-backend divergence).** The
   harness caught this in Phase 1 (originally a strict `xfail`): the `default` kernel
   used on **mps** applied **coupled** decay for lion (`g += p * weight_decay`, LION
   wrongly included in the coupled group), while the `cpu` and CUDA kernels applied
   **decoupled** decay (`p *= 1 - lr*weight_decay`) per the Lion paper — the `default`
   backend was the outlier, a real upstream bug affecting every device on the default
   optimizer path (mps included). **Fixed upstream (#1992 / #1993):** the default
   backend now excludes LION from the coupled fold and applies decoupled decay, so mps
   and the cpu oracle agree. The former xfail is now the passing regression test
   `test_lion_weight_decay_decoupled_parity`.
2. **8-bit optimizers are unusable on mps** — `optimizer_update_8bit_blockwise` has no
   mps/default registration and raises `NotImplementedError`.
3. **`int8_double_quant` is CUDA-only** — raises on mps _and_ on cpu.
4. **`dequantize_blockwise.out`** raises on mps (only cuda/xpu register the `.out`
   overload; the non-`.out` variant works via `default`).
5. **The Hub-kernel gate is necessary but not sufficient**: macOS 26 alone doesn't
   enable it; the `kernels` package must be installed. A parity report claiming "MPS
   passes" on a macOS-26 machine may still be testing pure-torch fallbacks (as this
   baseline does). §7 risk from the plan: confirmed, resolved by checking
   `bitsandbytes.backends.mps.ops._kernel` at runtime.

---

## 6. Parity harness

`tests/test_mps_parity.py` — mirrors the `tests/test_ops.py` structure and
`tests/helpers.py` parametrization; skips the whole module when
`torch.backends.mps.is_available()` is false.

```bash
pytest tests/test_mps_parity.py -v --tb=short
```

Coverage: quantize/dequantize_blockwise (bit-exactness, parity, round-trip),
quantize/dequantize_4bit (NF4+FP4 × fp32/fp16/bf16 × blocksize {64,128,256,512},
partial-block tail), gemv_4bit, gemm_4bit (± bias, ± nested/compressed absmax),
the int8 op family, optimizer_update_32bit (adam/momentum/rmsprop/lion), and
loud-failure tests pinning the §5 gaps (if a gap op starts working, its test fails,
forcing this document to be updated).

**Baseline record (2026-07-08, torch 2.12.1, macOS 26.4.1): 183 passed, 1 xfailed
(strict; the lion divergence), 0 skipped, ~7 s** (no native build).

`TestNativeMetalPath` gates the Phase-2 native path: it skips when the native library is
absent, or -- with `BNB_MPS_REQUIRE_NATIVE=1` -- fails hard, so a source-build verification
run cannot silently pass on the fallback. It also proves graceful degradation
(`test_graceful_fallback_when_native_absent` forces the native handle off and confirms the
pure-torch path still works).

---

## 7. Phase 2 — first native kernel end to end (`quantize_blockwise`)

**Status: complete and green.** On a source build (`cmake -DCOMPUTE_BACKEND=mps -S . -B . &&
cmake --build . --config Release`), `bitsandbytes::quantize_blockwise` on `mps` runs through a
hand-written Metal kernel and is **bit-exact** vs the CPU oracle (codes AND absmax) across
fp32/fp16/bf16 × blocksize {64,128,256,512}, including partial-block tails. Full suite on the
native build: **199 passed, 1 xfailed**.

### Validation of the pre-existing kernel (plan §7 / step 1)

The old `csrc/mps_kernels.metal::quantize` kernel was validated before trusting it. Its scalar
binary-search core (`quantize_scalar<false>`) is **mathematically correct** — reimplemented in
Python and checked against `torch.bucketize` over the dynamic map: 0/200 000 mismatches. **But the
kernel as a whole was the wrong shape for the op**: no per-block absmax, no scaling by absmax, it
never writes `absmax`, and it used an unrelated `NUM_BLOCK=4096` grid-stride loop instead of the
op's `blocksize`. It predates the current op registry. → **Replaced**, not reused.

### The three connected pieces

1. **MSL kernel** (`csrc/mps_kernels.metal`, `quantize_blockwise`): one thread per block — per-block
   absmax (serial reduction), `scaled = clamp(A * 1/max(absmax,1e-38), -1, 1)`, then a
   searchsorted-left over the 255 midpoint bounds of the 256-entry code table (reproduces
   `torch.bucketize(..., right=False)`). One-thread-per-block is the correctness-first shape; a
   SIMD-group parallel absmax reduction is deferred to a perf phase (per the plan's "correct before
   fast" rule). Compiled with **`-fno-fast-math`** (CMake) so division is correctly rounded and no
   FMA contraction occurs — this is what makes bucket selection identical to the CPU oracle.
2. **Dispatch layer** (`csrc/mps_ops.mm`): replaced the `NSLog("Not implemented")` stub with a real
   encode path — cached device/queue/library/pipeline singletons, `commandBuffer` →
   `computeCommandEncoder` → bind buffers → `dispatchThreads` → `commit` → `waitUntilCompleted`.
   Stable `extern "C" bnb_mps_quantize_blockwise(code, A, out, absmax, n, blocksize)`.
3. **Python load path** (`cextension.py` `MpsBNBNativeLibrary` + `get_mps_library()`;
   `backends/mps/ops.py` routing): native when the lib + metallib are present, else today's
   pure-torch fallback. Never hard-crashes when absent.

### Key implementation decisions / surprises

- **Buffer bridging with zero libtorch linkage.** The CMake `mps` target links only Metal/MPS
  frameworks, not libtorch — so the classic ATen `getMTLBufferStorage` include path isn't available.
  Empirically confirmed on this machine that **a torch MPS tensor's `data_ptr()` IS its
  `id<MTLBuffer>`** (probed: cast to `id<MTLBuffer>`, `[buffer length]` == tensor byte size, class
  `AGXG16XFamilyBuffer`). So the `.mm` casts the ctypes-passed `void*` straight to `id<MTLBuffer>`.
  Ruled out along the way: `data_ptr()` is **not** page-aligned (offsets 4032/6272/…), so
  `newBufferWithBytesNoCopy` fails; and it is **not** a CPU-readable unified pointer (reads returned
  garbage), so a memcpy-in/out bridge is impossible. The object-pointer bridge is the only one that
  works from a ctypes lib.
- **Offset-0 requirement.** `data_ptr()` equals the buffer object only for a `storage_offset == 0`
  tensor; a view's `data_ptr()` is `buffer + offset` and would cast wrong. The Python wrapper forces
  fresh, contiguous, offset-0 fp32 buffers (`_ensure_native_buffer`) before the call. Cost: a copy of
  A per call (acceptable, correctness-first; a later phase can avoid it).
- **Cross-queue synchronization.** The kernel dispatches on its own `MTLCommandQueue`, not torch's
  MPS stream. `torch.mps.synchronize()` is called **before** the dispatch (torch's writes to A/code
  materialized) and the `.mm` blocks on `waitUntilCompleted` **after** commit (outputs complete
  before torch reads). This is the "flush first" lesson from the op-graduation playbook, adapted to a
  separate queue.
- **Install-safe metallib load (plan §4).** The old `get_library()` loaded `bitsandbytes.metallib`
  by CWD-relative path. Now resolved via `dladdr` on a symbol in this dylib → same directory as the
  loaded `.dylib` (both land in `PACKAGE_DIR`), with a `BNB_MPS_METALLIB` env override and a
  CWD-relative last resort.
- **Build layout.** The metallib custom command writes relative paths, so the build must be
  **in-source** (`-B .`, matching the plan's `cmake -S .` recipe) for the metallib and dylib to land
  together in `bitsandbytes/`. An out-of-tree `-B build/` split them. Both files are gitignored.
  **Packaging risk RESOLVED (§9):** the wheel now ships both the `.metallib` and `_mps.dylib` (see
  §9).

Phase 3 (below) graduated dequantize_blockwise + the 4-bit ops onto this exact pipe; the 4-bit
matmuls (`gemv_4bit`/`gemm_4bit`) remain the separate, later hard sub-phase.

---

## 8. Phase 3 — remaining quant/dequant ops on native Metal

**Status: complete and green.** On the source build, three more ops run through hand-written Metal
and are **bit-exact** vs the CPU oracle. Full suite on the native build: **293 passed, 1 xfailed**
(with `BNB_MPS_REQUIRE_NATIVE=1`). Order graduated, each with a green parity test before the next:

| Op                     | New registration?                        | Parity vs CPU oracle (native path)                                            |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `dequantize_blockwise` | **yes** (was missing on mps → `default`) | out **bit-exact** (`torch.equal`), fp32/fp16/bf16 × bs {64,128,256,512}       |
| `dequantize_4bit`      | native swap                              | out **bit-exact**, NF4+FP4 × all dtypes/blocksizes incl. odd-numel tail       |
| `quantize_4bit`        | native swap                              | packed nibbles + absmax **bit-exact**, incl. `quant_storage=bf16` reinterpret |

- **Kernels** (`csrc/mps_kernels.metal`): `dequantize_blockwise` (`out[i]=code[A[i]]*absmax[i/bs]`),
  `dequantize_4bit` (high nibble→even index, low→odd; `out[j]=code4[nib]*absmax[j/bs]`), and
  `quantize_4bit` (per-block absmax + searchsorted over the 15 midpoint bounds of the sorted code +
  `order` remap for FP4 + nibble pack). All one-thread-per-block, fp32 internally; the Python wrapper
  casts dequant output to the requested dtype (matching the reference's trailing `.to(dtype)`), which
  is what makes the fp16/bf16 dequant outputs land **bit-exact** rather than merely within tolerance.
- **Tolerances:** integer/packed outputs (codes, packed nibbles, absmax) asserted **bit-exact**
  (`torch.equal` / view-as-uint8 to dodge NaN≠NaN on the bf16 reinterpret); float dequant outputs use
  the Phase-1 per-dtype tolerances (`assert_parity`) but measured bit-exact in practice.
- **Two reference subtleties reproduced (both were latent bugs risks):**
  1. **Tail-block asymmetry.** The reference stores the tail (partial) block's absmax **clamped** to
     1e-38 and scales it by **direct divide** (`A/absmax`), while full blocks store the **raw** max and
     use **reciprocal-multiply** (`A*(1/absmax)`). Under `-fno-fast-math` these differ by up to 1 ulp,
     which can flip a bucket. The Phase-2 `quantize_blockwise` kernel used reciprocal-multiply for all
     blocks and only passed because test tails were non-zero randn — **hardened in Phase 3** to branch
     on `is_tail` for both `quantize_blockwise` and `quantize_4bit`.
  2. **Odd-numel padding nibble.** For odd numel the reference pads `scaled` with `0.0` and
     **quantizes that** (a nonzero NF4/FP4 index), then packs it as the final low nibble. The kernel
     must quantize `0.0` for that slot, not write a literal `0` — caught by the partial-block test (it
     was a real 1-code mismatch until fixed).
- **`dequantize_4bit` also feeds the matmul fallbacks:** routing it native means `gemv_4bit`/`gemm_4bit`
  on mps now dequantize through Metal before the pure-torch `F.linear`. Their parity tests stay green.
  The matmul itself is untouched (still `F.linear`) — the hard fused-matmul sub-phase has NOT started.

### Sub-task 0 — load-time guard for the `data_ptr()`-is-the-`MTLBuffer` contract

The blind `(__bridge id<MTLBuffer>)` cast rides on an **undocumented** torch internal. Hardened with a
cheap **one-time** check at native-library load (`MpsBNBNativeLibrary.verify_buffer_contract()` →
`extern "C" bnb_mps_check_buffer_contract`): it takes a real MPS tensor's `data_ptr()` + its byte size
and confirms the pointer resolves to a genuine `id<MTLBuffer>` (protocol conformance + `[length]` ≥
size), guarded by `@try/@catch`. If a future torch breaks the contract, `get_mps_library()` **disables
the native path and logs a clear, actionable error** (falls back to pure-torch — no crash, no silent
corruption); `BNB_MPS_REQUIRE_NATIVE=1` then turns that into a hard test failure. Verified: real tensor
→ 1, null pointer → 0, oversize length → 0 (`test_buffer_contract_guard`).

**Unchanged debt (not regressed):** the per-call offset-0 copy of inputs. `gemv_4bit`/`gemm_4bit`
fused matmul and the int8/optimizer ops remain out of scope. (The wheel-packaging gap is now closed
— see §9.)

---

## 9. Packaging — native MPS from a `pip install` (not only source builds)

**Status: resolved.** The wheel now ships both native artifacts and native MPS loads from a plain
`pip install`.

**The gap.** `pyproject.toml` `[tool.setuptools] package-data` matched `libbitsandbytes*.*` — that glob
catches every shared library (all prefixed `lib…`, including `libbitsandbytes_mps.dylib`) but **misses
`bitsandbytes.metallib`**, which has no `lib` prefix. A wheel built before this fix carried the dylib
but not the shader archive, so `get_mps_library()` found the dylib, failed the `metallib.exists()`
gate, and silently fell back to pure-torch.

**The fix (packaging only, one line).** Added a `*.metallib` entry to `package-data`:

```toml
package-data = { "*" = ["libbitsandbytes*.*", "*.metallib", "py.typed"] }
```

Verified the `.dylib` is genuinely covered by the existing glob (not assumed) — see the `unzip -l`
evidence below; both land at `bitsandbytes/…`.

**Build flow (matches how bnb ships prebuilt CUDA `.so`s).** `setup.py`'s `ExtBuildPy` runs a CMake
build (default `COMPUTE_BACKEND=cpu`) during `build_py` **unless `BNB_SKIP_CMAKE=1`**. (`wheel.cmake =
false` is a scikit-build-core _native_-backend setting; this repo uses the `scikit_build_core.setuptools`
shim, where the CMake step is driven by `setup.py` + the `BNB_SKIP_CMAKE` env, so `BNB_SKIP_CMAKE=1` is
the actual switch that skips it.) So the flow is:

```bash
cmake -DCOMPUTE_BACKEND=mps -S . -B . && cmake --build . --config Release   # artifacts -> bitsandbytes/
rm -rf build/ dist/                                                          # avoid staging a stale cpu dylib
BNB_SKIP_CMAKE=1 python -m build --wheel                                     # package pre-built artifacts, no re-run
```

Gotcha found: without `BNB_SKIP_CMAKE=1`, `python -m build` re-runs CMake as `cpu` and adds a stray
`libbitsandbytes_cpu.dylib`; and a stale `build/lib…/` staging dir from an earlier non-skip build gets
swept into the wheel, so clean `build/` first.

**Inclusion proof — `unzip -l dist/*.whl`:**

```
    18074  bitsandbytes/bitsandbytes.metallib
    75928  bitsandbytes/libbitsandbytes_mps.dylib
```

**Runtime proof — isolated throwaway venv (not the source tree).** Fresh venv, `pip install --no-deps`
the built wheel (torch inherited), run from outside the worktree so `import bitsandbytes` resolves to
the _installed_ package. Confirmed against the installed wheel:
`bitsandbytes.__file__` → venv site-packages; both artifacts present in the installed package;
`get_mps_library()` loads native with `metallib_path` resolved (via `dladdr`) inside the venv;
`verify_buffer_contract()` passes; native `quantize_blockwise` bit-exact vs CPU; and the parity subset
`-k "Native or Blockwise8bit or Test4bitParity"` runs **236 passed** with `BNB_MPS_REQUIRE_NATIVE=1`.

**Still open (not this task):** the wheel is a plain-tagged platform wheel; CI matrix / release
automation to actually publish MPS wheels is a separate concern. The per-call offset-0 input copy and
the fused 4-bit matmuls remain as documented above.

---

## 10. Phase M1 — 4-bit matmul baseline + A/B decision (spike)

**Status: measured. Decision made.** This is the `NEXT_MATMUL_PLAN.md` Phase M1 spike, but done
against the _real_ baseline (today's `dequant → F.linear`) rather than an unbuilt native route — the
numbers decide the design fork on their own, so no throwaway MPSMatMul wiring was needed to choose.

**Method.** `torch.mps.synchronize()`-bracketed timing, warmup + 30–50 iters, native dequant forced
(`BNB_MPS_REQUIRE_NATIVE=1`), nf4/blocksize-64. Per shape we isolate the two costs inside today's
unfused path: the native Metal **dequant of B** (materializes full `B_dq`) and the **`F.linear`** GEMM
on that materialized `B_dq`. Bench scripts: `scratchpad/bench_matmul_baseline.py`,
`bench_gemm_baseline.py` (not committed; reproduce from the numbers here).

**gemv (M=1), fp16/bf16** — dequant is the whole cost:

| N     | K     | total  | dequant | linear | dequant share |
| ----- | ----- | ------ | ------- | ------ | ------------- |
| 4096  | 4096  | 0.94ms | 0.75ms  | 0.07ms | ~80%          |
| 11008 | 4096  | 1.80ms | ~2.0ms  | 0.19ms | ~90%+         |
| 4096  | 11008 | 1.81ms | 1.63ms  | 0.21ms | ~90%          |

**gemm (N=K=4096, fp16), sweeping M** — fixed dequant floor, GEMM overtakes it near M≈512:

| M    | total  | dequant | linear | GEMM share |
| ---- | ------ | ------- | ------ | ---------- |
| 8    | 1.41ms | 0.80ms  | 0.10ms | 7%         |
| 64   | 1.21ms | 0.88ms  | 0.42ms | 35%        |
| 512  | 2.50ms | 0.73ms  | 1.27ms | 51%        |
| 2048 | 5.66ms | 0.76ms  | 4.85ms | 86%        |

**Decision (per-op, as the plan anticipated — now with evidence):**

- **`gemv_4bit` (M=1) → Option B (hand-fused dequant+matmul).** 80–90% dequant-bound; Option A
  (`MPSMatrixMultiplication` on materialized `B_dq`) would only touch the ~10% GEMM slice. Fusion —
  never writing `B_dq` to device memory — is the entire win. This is Phase M2.
- **`gemm_4bit` large M (≥~512) → Option A (`MPSMatrixMultiplication`).** GEMM dominates; do not try to
  out-GEMM Apple's tuned kernel by hand. Accept the fixed ~0.75ms dequant tax. This is Phase M3.
- Small-M `gemm` (≤64) is still dequant-bound and behaves like gemv; a fused path helps there too, but
  M3 defaults to Option A for simplicity and lets the fixed dequant floor stand.

The Phase M2 (fused gemv), M3 (native gemm), and M4 (sync/offset closeout) results are consolidated
in **§11** below.

**Load-bearing caveat for the kernel author.** The existing dequant kernel moves ~40 MB in ~0.75ms ≈
**54 GB/s**, on hardware that sustains ~400 GB/s — it's leaving ~85% of memory bandwidth on the floor.
Both matmul routes inherit this: a fused `gemv` kernel that reads packed B no faster than the current
dequant will reproduce the 54 GB/s and win ~nothing. **The M2 target is bandwidth, not "fusion" per se**
— the fused kernel must read packed B + absmax at close to peak bandwidth (coalesced loads, minimal
recompute) or it doesn't beat the baseline. (Separately, this implies the standalone dequant kernel is
itself under-optimized — a possible bigger, simpler lever for the QLoRA M=1 inference case — but that's
Phase-3 kernel scope, out of this phase's remit.)

---

## 11. Phases M2–M4 — native 4-bit matmul (fused gemv + MPSMatMul gemm) and the sync/offset closeout

**Status: complete and green.** Both 4-bit matmuls run natively per the §10 decision. Phase-M4 full
suite: **328 passed** under `BNB_MPS_REQUIRE_NATIVE=1` (plus the known, unrelated lion strict-xfail
XPASS, §5). Wall-clock numbers below are steady-state (warmed, back-to-back calls) on this machine;
DVFS makes idle-gapped calls slower.

### 11.1 Phase M2 — `gemv_4bit` (M == 1): hand-fused Metal kernel (Option B)

`gemv_4bit_fp32/fp16/bf16` in `csrc/mps_kernels.metal`: one SIMD-group per output element, uint4
loads of packed B, in-register nf4/fp4 dequant (weights rounded to the activation dtype, matching
the oracle's `B_dq.to(dtype)`), fp32 fma accumulation with split accumulators, `simd_sum`
reduction. **B_dq is never materialized.** Router guards: true M==1, K % 32 == 0, power-of-two
blocksize ≥ 32, 16-entry code, packed-size check; anything else falls back to dequant+`F.linear`.

- Parity: all gemv tests green under `BNB_MPS_REQUIRE_NATIVE=1`, native asserted via spy; K%32≠0
  and native-absent fallbacks asserted.
- Wall-clock vs the dequant+`F.linear` baseline (nf4/bs64, 50 iters): **3.4–6.2x** across
  fp16/bf16/fp32 on the §10 shapes (e.g. fp16 4096×4096: 0.31ms vs 1.64ms; 11008×4096: 0.51ms vs
  1.76ms). Kernel-only GPU time ~0.11–0.14ms for the ~25 MB shapes when clocked up ≈ **~230 GB/s**
  effective read (vs the standalone dequant kernel's ~54 GB/s). Bench:
  `benchmarks_wip/bench_gemv_fused.py`.

### 11.2 Phase M3 — `gemm_4bit` (general M): chunked dequant + `MPSMatrixMultiplication` (Option A)

`bnb_mps_gemm_4bit` in `csrc/mps_ops.mm` encodes, on **one command buffer / one commit / one
blocking wait**: (1) a chunked dequant kernel (`dequantize_4bit_chunked_fp32/fp16`, one thread per
32-element uint4 chunk) writing `(T)(code[nib]*absmax)` into a growable **private scratch
`MTLBuffer`** — same rounding as the oracle's `B_dq.to(dtype)`; (2) a shape-cached
`MPSMatrixMultiplication` `A[M,K]·B_dq[N,K]ᵀ` (row-major, `transposeRight=YES`); (3) an optional
`out[m,n] += bias[n]` epilogue kernel. The single sync is the structural win over
dequant+`F.linear`, which pays the cross-queue round trip twice.

- **bf16 was excluded by the router at M3:** `MPSMatrixMultiplication` hard-asserts on anything
  but fp32/fp16/int8/int16 (probed on macOS 26.4.1: "Input data type must be one of
  MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16"), so bf16 kept
  the dequant+`F.linear` fallback verbatim. **Phase M5 (§11.5) lifted this** by routing bf16
  through `MPSGraph` instead; the rest of §11.2 is unchanged.
- Other router guards mirror gemv: K % 32 == 0, power-of-two blocksize ≥ 32, packed-size and bias
  checks; nested absmax is unpacked to plain fp32 absmax before routing, unchanged.
- Parity: native asserted via spy incl. ±bias/±nested-absmax; fp32-vs-MPSMatMul accumulation stays
  within the documented 1e-5 atol at the calibrated K ≤ 256. (The one tolerance trip found during
  M3 was in the **pure-torch** fallback composition at K=256/M=4, so the graceful-fallback test
  pins K=64.)
- Wall-clock vs the dequant+`F.linear` fallback (nf4/bs64, N=K=4096, 30 iters): fp16 **2.5x**
  (M=8), **1.5x** (M=64, M=512), **1.08x** (M=2048); fp32 **1.6x/1.5x** (M=8/64), **1.1x** (M=512),
  **~1.0x** (M=2048). The win is the single sync + a much faster chunked dequant at small/medium M;
  ~flat at M=2048 where the GEMM itself dominates and MPSMatMul ≈ `F.linear`'s GEMM. Bench:
  `benchmarks_wip/bench_gemm_baseline.py`.

### 11.3 Phase M4 — the per-call sync tax: measured, decomposed, and why it stays

Every native op runs on a **private** `MTLCommandQueue` and pays: `torch.mps.synchronize()` before
dispatch (torch's pending writes to the inputs must be materialized) and `waitUntilCompleted` after
commit (outputs complete before torch reads them). `BNB_MPS_PROFILE=1` now decomposes each call
(`sched` = commit → GPU start, `gpu` = kernel execution, `done` = GPU end → wait return; timebase
`mach_absolute_time`, same clock as `GPUStartTime`). Steady-state, fp16, idle torch queue:

| call               | total wall | pre-sync | encode  | sched   | gpu     | done    | fixed tax (non-gpu) |
| ------------------ | ---------- | -------- | ------- | ------- | ------- | ------- | ------------------- |
| gemv 4096×4096     | 0.29ms     | ~1µs     | ~0.02ms | ~0.07ms | 0.137ms | ~0.07ms | **~0.15ms (~52%)**  |
| gemv 11008×4096    | 0.50ms     | ~1µs     | ~0.02ms | ~0.10ms | 0.339ms | ~0.06ms | ~0.16ms (~32%)      |
| gemm M=8 (4096²)   | 0.59ms     | ~1µs     | ~0.02ms | ~0.06ms | ~0.39ms | ~0.07ms | ~0.17ms (~30%)      |
| gemm M=512 (4096²) | 1.55ms     | ~2µs     | ~0.02ms | ~0.07ms | 1.30ms  | ~0.07ms | ~0.16ms (~10%)      |

Two distinct costs:

1. **The fixed ~0.15ms/call round trip** (encode + commit→GPU-start scheduling + completion
   delivery). This is inherent to one-command-buffer-per-call on a private queue, NOT to the
   `torch.mps.synchronize()` itself — which is ~1µs when torch's queue is idle.
2. **Lost overlap when torch's queue is busy:** the pre-sync blocks until torch's pending work
   drains (measured 1.3ms with a pending 4096² fp16 matmul). That work would run anyway; the cost
   is serialization — the CPU stalls instead of encoding ahead.

**Why the private queue + both syncs stay (the Task-1 investigation, torch 2.12.1):**

- **torch exposes no queue/stream handle.** Nothing in `torch.mps` / `torch._C._mps_*` returns the
  `MTLCommandQueue` or `MPSStream` (only events, shader compilation, and synchronize exist; there
  is no `torch.mps.current_stream()` in 2.12.1).
- **The C++ internals are reachable only as an ABI trap.** `libtorch_cpu.dylib` exports
  `at::mps::getCurrentMPSStream()` and some `MPSStream` methods, but: `commandQueue()`/`queue()`
  are inline (recovering them means reading ivars at header-derived offsets — layout-dependent);
  `commit()`/`flush()` are private; every encode must run on torch's private `_serialQueue`
  dispatch queue to avoid racing its kernel-coalescing encoder; and `SyncType` enum values would
  be assumed. dlsym-ing mangled C++ internals from a torch-independent ctypes dylib is exactly the
  miscast class of bug the load-time buffer-contract guard exists to prevent. Rejected.
- **Sharing only the queue would not remove the pre-sync anyway.** Command buffers execute in
  COMMIT order, and torch batches encodes into an uncommitted `MPSCommandBuffer` — a buffer we
  commit first can run before torch's earlier-issued-but-uncommitted writes. Correct ordering
  still requires torch to flush, which is also not exposed.
- **Future directions (recorded, not taken):** (a) build the mps backend as a libtorch-linked
  torch extension using `getCurrentMPSStream()` — the sanctioned C++ route, but it couples the
  binary to the torch ABI/version, contrary to bnb's ship-one-binary packaging; (b)
  `torch.mps.compile_shader` (documented since ~torch 2.7) dispatches user MSL on torch's own
  stream and would eliminate both syncs for the pure-MSL kernels — but it is runtime source
  compilation (the `-fno-fast-math` metallib guarantees would need re-validation) and cannot host
  the `MPSMatrixMultiplication` gemm. Either is a re-architecture, out of M4 scope.

**Guard:** `test_sync_discipline_interleave_stress` interleaves dependent torch writes with native
matmuls on the same buffers, parity-checked every iteration. Verified to have teeth: with
`torch.mps.synchronize` no-op'd it fails **30/30** iterations; with the discipline intact it passes
100%. Any future change to the sync must keep this test green.

### 11.4 Phase M4 — offset-0 input copy: pointer semantics verified, clone stays

`_ensure_native_buffer` `.clone()`s any input with `storage_offset != 0`. §5 of
`NEXT_MATMUL_PLAN.md` asked whether that copy can be replaced by binding at a byte offset.
Verified on this build (torch 2.12.1):

- **A view's `data_ptr()` is `base_ptr + storage_offset * itemsize`** — raw pointer arithmetic
  (probed: a fp32 view at offset 128 reads exactly +512 bytes). It is NOT an `id<MTLBuffer>`;
  even objc-probing such an interior pointer (protocol conformance inside `@try/@catch`)
  **SIGSEGVs the process** — the crash is not catchable. The clone is load-bearing; casting a
  view's `data_ptr()` would be silent-corruption-or-crash.
- **New finding (better than §5 feared):** the base buffer object IS recoverable from Python —
  `t.untyped_storage().data_ptr()` returns the base allocation pointer, which passes the
  buffer-contract check as a genuine `id<MTLBuffer>` of the full allocation size. So safe offset
  binding is _possible_: bind `untyped_storage().data_ptr()` with
  `[enc setBuffer:base offset:storage_offset*itemsize atIndex:i]` (plus alignment guards for the
  kernels' vectorized uint4 loads).
- **Decision: not implemented.** Steady-state matmul inputs (activations, quant state) are fresh
  offset-0 allocations — the clone almost never fires in the QLoRA path — and the change would
  touch every C ABI entry point for no measured benefit. The recipe above is recorded for when a
  workload actually hits the copy. `test_view_data_ptr_is_base_plus_offset` pins the verified
  semantics so a future torch that changes them fails loudly instead of silently invalidating
  `_ensure_native_buffer`'s premise.

### 11.5 Phase M5 — `gemm_4bit` in bf16 via MPSGraph

**Status: complete and green.** Correctness verified, wall-clock measured (below).

bf16 was the one dtype with no native 4-bit matmul, and it is the dtype the CogKit CogView4-6B
QLoRA lane actually trains in — so on that lane the M3 work did not apply at all. The blocker
was never Metal, only `MPSMatrixMultiplication`, which accepts fp32/fp16/int8/int16 and nothing
else. **`MPSGraph` does have a bf16 matmul.** Probed directly on macOS 26.5 / M4 Max before any
code was written: `matrixMultiplicationWithPrimaryTensor:` on `MPSDataTypeBFloat16` builds,
runs, and returns the exact expected value.

The M3 structure is unchanged — chunked dequant into a private scratch buffer, then GEMM, then
optional bias epilogue, all on **one command buffer / one commit / one blocking wait**. Only the
middle step differs: bf16 encodes a shape-cached `MPSGraph` (`get_gemm_graph_bf16`) onto an
`MPSCommandBuffer`, where fp32/fp16 keep the shape-cached `MPSMatrixMultiplication`. The Metal
kernels needed no new code, only two instantiations of the existing `T`-templated macros
(`dequantize_4bit_chunked_bf16`, `gemm_bias_add_bf16`) — `bfloat` was already in use by the M2
gemv kernels.

Three things were probed before committing to the design, because each would have sunk it:

1. **Oversized buffers.** `MPSGraphTensorData initWithMTLBuffer:shape:dataType:` accepts a
   buffer larger than the shape requires and reads only the leading elements. This is not an
   edge case: torch's caching allocator hands out oversized buffers as the *normal* case, so a
   strict-size requirement would have meant a copy on every call.
2. **Sharing a command buffer.** `MPSCommandBuffer` conforms to `MTLCommandBuffer`, so our own
   `computeCommandEncoder` for dequant and bias works on it unmodified — the single-sync
   structure survives.
3. **Hazard tracking across the two.** Our compute kernel writes the *private* scratch buffer
   and the graph reads it, on one command buffer, with no explicit barrier. Verified correct at
   the first and last element. (MPSGraph may `commitAndContinue` internally, rolling the root
   command buffer; commit order is preserved regardless, so waiting on the final root also
   waits for anything rolled off. The `BNB_MPS_PROFILE` GPU-time decomposition is the one thing
   that could be skewed by such a split.)

**Known property, inherited from M3: the shape cache is unbounded.** `get_gemm_graph_bf16` keys
on `{M, N, K}` and never evicts, exactly like M3's `get_gemm` cache of `MPSMatrixMultiplication`
objects. For training that is free — M is fixed by the config and a transformer repeats a handful
of `{N, K}` — but a long-running inference server with a varying sequence length would accumulate
one compiled graph per distinct M, and an `MPSGraph` is a heavier object to leak than an
`MPSMatrixMultiplication`. Left as-is deliberately rather than bounding one cache and not the
other; if it ever matters, both want the same LRU.

**Router gating.** bf16 requires the new `bnb_mps_gemm_4bit_supports_bf16` capability symbol,
not just `bnb_mps_gemm_4bit`. Without it, a dylib built before M5 would be handed `dtype_flag=2`
and would fall through to `elsize = 4` / `MPSDataTypeFloat32` — reading a bf16 scratch as fp32
and returning silent garbage. Old dylib, new Python, no crash: exactly the failure mode worth a
symbol.

**`BNB_MPS_DISABLE_BF16_GEMM=1`** forces the dequant+`F.linear` fallback for bf16 with the native
build otherwise intact. It exists so the path can be A/B'd in place — the end-to-end table above
was produced by toggling this between runs of the same binary and config, which is the only way to
know the difference is the GEMM and not a rebuild, a reinstall, or the weather. A speedup you
cannot switch off is a speedup you cannot verify.

**Numerics.** Without bias the native bf16 path reproduces the dequant+`F.linear` fallback it
replaces **bit-exactly** — asserted by `torch.equal` in
`test_gemm_4bit_bf16_is_native_and_matches_the_fallback`, which is the assertion with teeth
here. With bias the two can differ: the epilogue computes `(bfloat)(gemm + bias)` on an already
bf16-rounded GEMM result where `F.linear` rounds once, so the error is one ulp of the *pre-add*
magnitude. Wherever bias largely cancels the GEMM result that is several ulp of the much smaller
output, which is why the biased case is held to the documented CPU parity tolerance rather than
a relative-to-output bound. (A first draft of the test used exactly such a bound and failed —
correctly. The kernel was fine; the tolerance was the wrong shape.) bf16 is also now part of the
`test_gemm_4bit_native` parametrized sweep across nf4/fp4 × ±bias × ±nested-absmax.

Quantified at the CogKit shapes (M=1024, each of the four `(N, K)`): **without bias, bit-exact at
every one, K=16384 included.** With bias, ~25% of output elements differ, always by exactly one ulp
— which is what a double rounding does to uniformly distributed values. CogView4's linears are
`bias=True`, and across 28 layers this is visible end to end: the QLoRA loss sequence is otherwise
identical between the two arms (1.12, 1.36, 1.32, 1.32) and deterministic within each, but the one
high-loss outlier step reads **5.56 on the fallback and 5.59 native**, a 0.5% shift on a step whose
loss is 4x the others. This is the same epilogue design fp32/fp16 have shipped with since M3, so it
is left alone rather than made inconsistent across dtypes; eliminating it would mean an fp32 output
scratch so the bias adds before the single rounding.

**Wall-clock.** Measured on a quiet M4 Max (macOS 26.5, torch `2.15.0a0+gitf6df965`), 30 iters /
8 warmup, with a 64 MB `clone()` control read before and after each table (0.300 → 0.287 ms, i.e.
~427 GB/s and stable, so each table is internally comparable). An earlier attempt the same day was
discarded entirely: the machine sat at load average 200–298 under a runaway editor file-scan, where
plain torch bf16 `F.linear` read 3.0 TFLOP/s — about a tenth of this GPU — and *the sign of some
comparisons flipped*. See the retraction below for how badly.

M sweep, `gemm_4bit` native vs the dequant+`F.linear` fallback, nf4/bs64, N=K=4096:

| M    | bf16       | fp16   | fp32   |
| ---- | ---------- | ------ | ------ |
| 8    | **2.09x**  | 2.16x  | 1.22x  |
| 64   | **1.94x**  | 1.94x  | 1.04x  |
| 512  | **1.09x**  | 1.80x  | 1.24x  |
| 2048 | **1.03x**  | 0.96x  | 0.93x  |

bf16 traces the same curve M3 found for fp16: a large win where the fixed ~0.15 ms sync tax is a big
share of a small op, converging to break-even once the GEMM dominates. At the CogView4-6B shapes
(M=1024/1280 × the four `(N, K)` of qkv/out/mlp-in/mlp-out) bf16 lands between **0.98x and 2.11x,
mostly 1.0–1.16x** — the training lane sits in the flat part of the curve, not the steep part.

**End-to-end: CogKit CogView4-6B QLoRA, 512×512, batch 1.** *Superseded by the three-arm n=6
measurement in §11.6 — the forward figure held, the step figure did not.* Four runs per arm, alternated, toggled
in place with `BNB_MPS_DISABLE_BF16_GEMM` (identical binary and config, so nothing else can drift):

| stage           | fallback (mean, range)   | native (mean, range)     | delta      | ranges overlap? |
| --------------- | ------------------------ | ------------------------ | ---------- | --------------- |
| **forward**     | 4.418 s [4.196–4.646]    | **3.950 s [3.754–4.181]**| **−10.6%** | **no**          |
| backward        | 3.963 s [3.863–4.183]    | 4.111 s [3.919–4.433]    | +3.7%      | yes             |
| **step**        | 8.651 s [8.338–9.098]    | **8.329 s [7.956–8.860]**| −3.7%      | yes             |
| memory_reserved | 17.39 GB [17.05–18.05]   | 16.43 GB [16.18–17.18]   | −5.5%      | yes             |

Read this carefully, because the honest claim is narrower than the headline:

- **Forward is the real result: −10.6%, with no overlap between the two arms' ranges across 4+4
  runs.** That is the only stage `gemm_4bit` touches, and it is the only one that separates.
- **Backward's +3.7% is noise**, and must be: `MatMul4Bit.backward` never calls `gemm_4bit` (verified
  by spy, not by reading — a bf16 `Linear4bit` fwd+bwd gives *forward = 1 native call, backward = 0*).
  A stage this path cannot reach moving by 3.7% is a direct measurement of the run-to-run noise floor.
- **Step time −3.7% overlaps** and is therefore suggestive, not established. At n=2 it looked like a
  clean −7% with no overlap; two more runs per arm dissolved that. The forward win is real and the
  backward noise is comparable in size, so it partly eats the step-level gain.
- The ~1 GB of `memory_reserved` is a plausible side effect — the fallback materializes a full `B_dq`
  torch tensor per layer through the caching allocator, where the native path reuses one private
  scratch `MTLBuffer` outside it — but the arms overlap, and ~134 MB of that scratch is simply
  invisible to torch's accounting rather than saved.

**Retraction.** The pre-code probe recorded during the contended window had `MPSGraph` bf16 at 2.33 ms
against `MPSMatrixMultiplication` fp16 at 3.84 ms (1024×2560×2560) and was written up as a lead that
fp32/fp16 might want MPSGraph too. Re-run on the quiet machine: **1.183 ms vs 1.081 ms — MPSMatrix-
Multiplication is slightly faster, and the ratio reversed.** There is no case for moving fp32/fp16 onto
MPSGraph. Both numbers were ~2x slower under contention *and* their ordering flipped, which is the
cleanest available demonstration that a contended benchmark is not merely imprecise but can be
directionally wrong.

**What this does NOT cover: the backward pass.** `MatMul4Bit.backward` computes
`grad_A = grad_output @ dequantize_4bit(B)` in Python and never calls `gemm_4bit` at all — so it
still pays two syncs and a full weight dequant, and none of M3 or M5 reaches it. In the CogKit
QLoRA profile backward is 4.03 s of an 8.52 s step, comparable to forward. A fused
`gemm_4bit_backward` (same dequant scratch, `transposeRight:NO`, one command buffer) is the
obvious next phase and is not started.

### 11.6 Phase M6 — the fused backward: `gemm_4bit_backward`

**Status: correctness complete; end-to-end numbers below.**

§11.5 closed the last dtype gap in the forward and, in doing so, made the remaining asymmetry
obvious: **`MatMul4Bit.backward` never called `gemm_4bit` at all.** It computed
`grad_A = grad_output @ dequantize_4bit(B)` inline in Python — a native dequant on our private
queue (wait), handed to torch, then a matmul on torch's queue (wait). Two cross-queue round trips
per `Linear4bit` per step, on a stage that is ~48% of a CogKit QLoRA step. Every phase from M3
through M5 optimised the half of the step that was already the faster half.

This was verified by spy before any code was written, not inferred from reading: a bf16
`Linear4bit` fwd+bwd reported **forward = 1 native call, backward = 0**.

**The new op.** `bitsandbytes::gemm_4bit_backward(grad_output, B, shapeB, absmax, blocksize,
quant_type, ...)` returns `grad_A[..., K] = grad_output[..., N] · B_dq[N, K]`. Note the inner
dimension is **N**, not K. Its `default` kernel is exactly the composition it replaces, so every
non-MPS device gets the op for free and the fused kernel has an oracle to be checked against.

**Why this orientation is cheap.** `dequantize_4bit` already emits `B_dq` as `[N, K]`, which is
precisely what `grad_A` wants — so the backward needs **no transpose** where the forward does
(`transposeRight:NO` for `MPSMatrixMultiplication`; no `transposeTensor` node for the bf16
`MPSGraph`). There is also no bias: `grad_bias` is `grad_output.sum(0)`, computed in Python and
never routed here.

The C side is not a second implementation. `bnb_mps_gemm_4bit` and `bnb_mps_gemm_4bit_bwd` are
both thin `extern "C"` shims over one `gemm_4bit_common(..., bool backward)` body, so the chunked
dequant, the private scratch, the shape caches, the profiling hooks and — critically — the
one-command-buffer / one-commit / one-wait discipline are literally the same code in both
directions. The orientation flows through as three expressions (`lhs_cols`, `res_cols`,
`transposeRight`) and one cache-key bit.

**Numerics: bit-exact, with nothing to caveat.** The forward's one-ulp deviation came entirely
from the bias epilogue double-rounding (§11.5); the backward has no bias, so there is no such
term. The fused kernel reproduces `dequant + torch.matmul` **bit-exactly** at every shape tested,
asserted with `torch.equal` in `test_gemm_4bit_backward_native` across fp32/fp16/bf16 × nf4/fp4 ×
±nested-absmax. End to end this is pinned by
`test_linear4bit_bf16_autograd_is_unchanged_by_the_native_paths`: with `bias=False` a whole
`Linear4bit` fwd+bwd is **bit-identical** with the native paths on and off; with `bias=True` only
the *forward* differs and the gradient merely inherits that one ulp. The fused backward adds no
deviation of its own.

**A trap found while writing the tests, which applies to §11.2's forward test too.** The new
backward test first failed at `compress_statistics=True` + fp32, and the fused kernel was not the
cause — it was bit-exact against the on-device `dequant + torch.matmul` (deviation 0.0) in exactly
the failing case. The cause is upstream: with nested statistics the **absmax is itself quantized
with `quantize_blockwise`**, and that is the one op whose CPU kernel is known-approximate (it snaps
to a 65536-point LUT before the codebook lookup — the 26 standing failures in §5). For this tensor
it picked a different code for **1 block out of 512**, shifting that block's scale by ~9.3e-4, which
lands in `B_dq` and blows fp32's 1e-5 tolerance. The plain (non-nested) absmax was bit-identical.

So a test that quantizes *independently* on cpu and mps is not really testing the matmul when
`compress_statistics=True`; it is also re-testing a quantizer that is already known to disagree.
`test_gemm_4bit_backward_native` now quantizes once on cpu and moves the state across, making both
sides bit-identical by construction. **`test_gemm_4bit_native` (forward) still quantizes
independently and passes only because its RNG draw happens not to produce a differing code** — it
is one seed away from the same failure, and should be converted the same way when someone next
touches it.

**Gating.** bf16 requires the `bnb_mps_gemm_4bit_supports_bf16` marker as before; every dtype
additionally requires `bnb_mps_gemm_4bit_bwd` to exist, so a dylib predating M6 keeps the old
composition. `BNB_MPS_DISABLE_BF16_GEMM_BWD=1` disables the backward alone, which is what makes
the three-arm measurement below possible against a single binary — a single switch could only
ever have answered "both or neither".

**Wall-clock: `gemm_4bit_backward` alone**, nf4/bs64, CogView4-6B shapes, quiet machine (64 MB
`clone()` control 0.298 → 0.287 ms across the table):

| M    | bf16 (native vs dequant+matmul)          | fp16                    |
| ---- | ---------------------------------------- | ----------------------- |
| 1024 | 1.18–1.31x across the four `(N, K)`      | 1.02–1.18x              |
| 1280 | 1.01–1.20x                               | 1.10–1.15x              |

Slightly better than the forward at the same shapes (§11.5: 0.98–1.16x), which is what the
orientation predicts: no bias epilogue to encode, and no transpose for the GEMM to absorb.

**End to end: CogKit CogView4-6B QLoRA, 512×512, batch 1, three arms, n=6 each.** Arms toggled in
place against one binary via `BNB_MPS_DISABLE_BF16_GEMM` / `..._BWD`, so nothing but the routing
differs. **Half the reps ran the arms in the opposite order**, because with a fixed arm order any
warming across a rep hands the last arm a free win:

| arm                       | forward         | backward        | step             |
| ------------------------- | --------------- | --------------- | ---------------- |
| NEITHER (pre-M5 baseline) | 4.784 ± 0.363 s | 4.340 ± 0.375 s | 9.394 ± 0.716 s  |
| FWD_ONLY (M5)             | 4.130 ± 0.468 s | 4.171 ± 0.435 s | 8.575 ± 0.891 s  |
| **BOTH (M5 + M6)**        | 4.055 ± 0.264 s | **3.638 ± 0.179 s** | **7.960 ± 0.431 s** |

- **BOTH vs NEITHER: forward −15.2%, backward −16.2%, step −15.3%.**
- **M6's own contribution (BOTH vs FWD_ONLY): backward −12.8%, step −7.2%.**

Two built-in noise checks say those are signal. `FWD_ONLY` should not move the backward at all,
and reports −3.9%; `BOTH` should not move the forward relative to `FWD_ONLY`, and reports −1.8%.
So the per-stage noise floor here is ~2–4%, and the −12.8% backward sits three to six times
above it. The backward arm is also the *tightest* in the table (± 0.179 vs ± 0.375 for the
baseline), which is what removing a per-layer cross-queue round trip should do to variance.

The ordering control earned its keep: reversing the arm order moved the same arm's step time by
−0.50 s to +0.64 s, comparable to the effect being measured. It happened to run **against**
`BOTH` in the original order (`BOTH` was last and *slower* there), so the headline was not an
artefact — but that was luck, not design, and a fixed arm order should not be trusted again.

**This supersedes §11.5's end-to-end row.** That measurement was two arms at n=4 in a single
fixed order and put the step at −3.7% (overlapping) where this one puts M5's step contribution at
−8.7%. The forward figure held up (−10.6% there, −13.7% here); the step figure did not, which is
exactly the stage where n=4 was called insufficient at the time.

**What is left.** Both directions of the 4-bit matmul are now native for every dtype, so the
remaining per-call cost is the ~0.15 ms fixed sync tax dissected in §11.3 — unchanged, and still
gated on either a libtorch-linked extension or `torch.mps.compile_shader`, both re-architectures.
Beyond that the step's remaining time is no longer in `bitsandbytes` at all.
