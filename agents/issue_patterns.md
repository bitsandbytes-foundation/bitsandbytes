# Common Issue Patterns in bitsandbytes

This document catalogs recurring issue patterns across the bitsandbytes issue tracker. Use it during issue triage to quickly identify duplicates, stale reports, and issues that can be closed.

## CUDA Setup / Library Loading

These are the single largest category of issues. Most are environment problems on old bitsandbytes versions, not code bugs.

### Legacy `cuda_setup/main.py` (versions 0.41.x–0.42.x)

**How to identify:** Tracebacks reference `bitsandbytes/cuda_setup/main.py` (line 166 or 167). Error output includes `UserWarning: Welcome to bitsandbytes. For bug reports, please run python -m bitsandbytes` in the old format. The import chain goes through `bitsandbytes/research/__init__.py` → `modules.py` → `GlobalOptimManager` → `cextension.py` line 20.

**What happened:** Versions 0.41.x–0.42.x used a fragile CUDA detection system in `cuda_setup/main.py` that searched for `libcudart.so` in environment paths. It had bugs:
- It re-initialized `cuda_runtime_libs = set()` after already populating it from `CONDA_PREFIX` and `LD_LIBRARY_PATH`, discarding valid search results.
- It failed in conda environments, Docker containers, and systems with multiple CUDA versions.
- It searched for Linux `.so` files on Windows.
- Error messages gave Linux-specific advice (`sudo ldconfig`, `export LD_LIBRARY_PATH`) regardless of platform.

**Resolution:** The entire `cuda_setup/main.py` module was replaced in v0.43.0 with a new library loading mechanism in `cextension.py`. Users should upgrade to the latest version.

**Closing template:**
> Closing this issue. The CUDA detection system (`cuda_setup/main.py`) used in bitsandbytes 0.41.x–0.42.x was fragile and had known bugs — it could fail to find CUDA libraries even when they were correctly installed, particularly in conda environments, Docker containers, and systems with multiple CUDA versions. That entire module was replaced starting in v0.43.0 with a more robust library loading mechanism.
>
> If you're still hitting CUDA setup problems on the **latest** bitsandbytes (v0.45+), please open a new issue with the output of `python -m bitsandbytes` and your environment details (OS, Python version, PyTorch version, GPU).

### Windows pre-support (before v0.43.0)

**How to identify:** Windows paths in tracebacks, but error messages reference `libcudart.so` (Linux) or `libbitsandbytes_cpu.so` (Linux extension on Windows). The `cuda_setup/` subdirectory path appears. May reference the unofficial `jllllll/bitsandbytes` Windows fork. Error advice includes `sudo ldconfig` or `find / -name libcuda.so` — Linux commands on Windows. The `argument of type 'WindowsPath' is not iterable` error is a strong signal.

**What happened:** Official Windows support was added in v0.43.0. Before that, users relied on unofficial forks or got the Linux-only `.so` builds that don't work on Windows.

**Closing template:**
> Closing this issue. This was reported before official Windows support was added in bitsandbytes v0.43.0. The old CUDA detection system also gave Linux-specific guidance on Windows. Both Windows support and the library loading system have been overhauled in recent releases.
>
> If you're still hitting problems on the **latest** bitsandbytes (v0.45+), please open a new issue with the output of `python -m bitsandbytes` and your environment details.

### Missing `libcusparse.so.11` / shared library mismatch

**How to identify:** `OSError: libcusparse.so.11: cannot open shared object file: No such file or directory`. Or similar errors for `libcusparse.so.12`, `libcublasLt.so.11`, etc.

**What happened:** The bnb binary was compiled against one CUDA version (e.g., 11.x) but the system only has another (e.g., 12.x). The shared library dependencies don't exist. Modern releases ship platform-specific wheels with better CUDA version detection and multiple binary variants.

**Closing template:**
> Closing this issue. The error indicates a mismatch between the CUDA version bitsandbytes was compiled against and the system CUDA libraries. Modern bitsandbytes releases (v0.43.0+) ship platform-specific wheels that handle CUDA version detection more reliably.
>
> If you're still hitting this on the **latest** bitsandbytes (v0.45+), please open a new issue with the output of `python -m bitsandbytes` and your environment details.

### C library load failure → `NameError: str2optimizer32bit` / `NoneType` errors

**How to identify:** `NameError: name 'str2optimizer32bit' is not defined`, `AttributeError: 'NoneType' object has no attribute 'cquantize_blockwise_...'`, or `AttributeError: 'NoneType' object has no attribute 'split'` in `cuda_specs.py`. May also show `module 'bitsandbytes' has no attribute 'nn'`.

**What happened:** When the C/CUDA binary fails to load (for any reason — wrong platform, missing deps, version mismatch), the `lib` object is `None` and Python-level dispatch dictionaries are never populated. The resulting errors are confusing symptoms of the real problem. PR #1615 (merged, tracked by #1548) improved error messaging to surface the actual load failure.

**Closing template:**
> Closing this issue. This error is a symptom of the C/CUDA library failing to load — the confusing `NameError`/`AttributeError` was a downstream effect. Error messaging for this case was improved in PR #1615. Please upgrade to the latest bitsandbytes, which will show a clearer error if the library fails to load.
>
> If you're still hitting this on the **latest** bitsandbytes (v0.45+), please open a new issue with the output of `python -m bitsandbytes` and your environment details.

### Unsupported platform / architecture

**How to identify:** Platform is aarch64 (Jetson), ppc64le, or uses a very old GPU (Kepler/compute 3.5). Binary file is missing for the architecture. Error like `libbitsandbytes_cuda122.so: cannot open shared object file`.

**What happened:** Pre-built binaries only cover x86-64 + certain CUDA versions. aarch64 support has improved in recent releases. Kepler (compute 3.5) and ppc64le are not officially supported.

**Closing template:**
> Closing this issue. Pre-built binaries were not available for this platform at the time of reporting. Please check the latest release notes for current platform support. For source builds, see the [installation docs](https://huggingface.co/docs/bitsandbytes/main/en/installation).

## Not bitsandbytes Issues

### Third-party application issues

**How to identify:** User is running Automatic1111, Forge UI, ComfyUI, kohya_ss, or similar Stable Diffusion tools. The error occurs inside bitsandbytes but is caused by the app pinning old bnb versions or misconfiguring the environment. Minimal or no diagnostic info. Often no bnb version specified.

**Resolution:** These are dependency management issues in third-party apps. Close with a note to report to the app's issue tracker and upgrade bitsandbytes.

**Closing template:**
> Closing this issue. This appears to be a dependency/environment issue in the application you're using rather than a bitsandbytes bug. Please ensure the application is using the latest bitsandbytes version (v0.45+). If the issue persists, reporting it to the application's own issue tracker may be more effective.

### Transformers version mismatch

**How to identify:** `ImportError: Using 'bitsandbytes' 8-bit quantization requires Accelerate: pip install accelerate and the latest version of bitsandbytes`. This error message comes from the `transformers` library, not from bitsandbytes.

**What happened:** Older `transformers` versions had a version check that could emit this misleading error even when both accelerate and bitsandbytes were installed. Upgrading `transformers` resolves it.

**Closing template:**
> Closing this issue. This error message originates from the `transformers` library, not from bitsandbytes. Upgrading `transformers` to the latest version resolves it.

### Unrelated errors filed against bitsandbytes

**How to identify:** The traceback's root cause is in another library (sentencepiece, diffusers, ONNX, etc.) but the user filed it here because bitsandbytes appeared somewhere in their stack. Look at the actual exception — if it's about tokenizer parsing, model loading from a different library, or API changes in diffusers/transformers, it's not a bnb issue.

**Closing template:**
> Closing this issue. The error originates in [library name], not in bitsandbytes. Please report it to the appropriate issue tracker.

## Other Recurring Patterns

### Undefined symbol errors (old builds)

**How to identify:** `undefined symbol: cadam32bit_grad_fp32` or `undefined symbol: cdequantize_blockwise_fp32`. Occurs with old builds or version mismatches between the compiled C library and the Python package.

**Resolution:** Upgrade to the latest version. If building from source, do a clean build.

### Questions filed as bugs

**How to identify:** The issue asks about NF4 internals (offset value, data format, quantile bins), how quantization works, or how to use a feature. Often has the `Question` label. No actual error or bug report.

**Resolution:** If answered in comments or by the reporter themselves, close. If useful, convert to a discussion. Consider whether the question reveals a documentation gap worth addressing.

### FSDP + bitsandbytes optimizers

**How to identify:** Errors when using bnb optimizers (Adam8bit, PagedAdamW, etc.) with FSDP or FSDP2. Common errors include `AssertionError` in `_convert_all_state_info`, `AttributeError: 'int' object has no attribute 'cpu'`, or `illegal memory access`.

**Current status:** FSDP support for bnb optimizers is a known gap. The maintainer has stated this repeatedly. Track via #1633 (open, Contributions Welcome). Historical context in #89 (closed).

**Resolution:** Close as duplicate of #1633, noting that FSDP optimizer support is not yet available.
