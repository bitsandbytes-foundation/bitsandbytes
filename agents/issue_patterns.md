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

**How to identify:** User is running Automatic1111, Forge UI, ComfyUI, kohya_ss/kohya-trainer, MimicMotion, or similar Stable Diffusion / fine-tuning tools. The error occurs inside bitsandbytes but is caused by the app pinning old bnb versions or misconfiguring the environment. Minimal or no diagnostic info. Often no bnb version specified. May also manifest as errors from other libraries (e.g., diffusers API changes like `AttributeError: module diffusers.models has no attribute unet_2d_condition`) that the user files against bnb because it was in their stack. Another variant: ComfyUI + Triton on Windows, which isn't officially supported by Triton — the user sees a bnb library loading error but the root cause is the app's dependency packaging. Device placement errors (tensors on different devices) from ComfyUI's execution pipeline also fall in this category.

**Resolution:** These are dependency management issues in third-party apps. Close with a note to report to the app's issue tracker and upgrade bitsandbytes.

**Closing template:**
> Closing this issue. This appears to be a dependency/environment issue in the application you're using rather than a bitsandbytes bug. Please ensure the application is using the latest bitsandbytes version (v0.45+). If the issue persists, reporting it to the application's own issue tracker may be more effective.

### Transformers version mismatch

**How to identify:** `ImportError: Using 'bitsandbytes' 8-bit quantization requires Accelerate: pip install accelerate and the latest version of bitsandbytes`. This error message comes from the `transformers` library, not from bitsandbytes.

**What happened:** Older `transformers` versions had a version check that could emit this misleading error even when both accelerate and bitsandbytes were installed. Upgrading `transformers` resolves it.

**Closing template:**
> Closing this issue. This error message originates from the `transformers` library, not from bitsandbytes. Upgrading `transformers` to the latest version resolves it.

### TensorFlow / non-PyTorch frameworks

**How to identify:** User mentions TensorFlow, JAX (without explicit bnb JAX support), or other non-PyTorch frameworks. They may be searching for CUDA runtime DLLs like `cudart64_118.dll` for TensorFlow GPU support and conflate it with bitsandbytes. Bitsandbytes only works with PyTorch (>= 2.2.2).

**Resolution:** Close, noting that bitsandbytes is PyTorch-only.

**Closing template:**
> Closing this issue. Bitsandbytes is only compatible with PyTorch (>= 2.2.2) and does not support TensorFlow or other frameworks. The issue you're describing appears to be related to your [TensorFlow/other] setup rather than bitsandbytes.

### Unrelated errors filed against bitsandbytes

**How to identify:** The traceback's root cause is in another library (sentencepiece, diffusers, ONNX, etc.) but the user filed it here because bitsandbytes appeared somewhere in their stack. Look at the actual exception — if it's about tokenizer parsing (e.g., `could not parse ModelProto from tokenizer.model` — that's sentencepiece), model loading from a different library, or API changes in diffusers/transformers, it's not a bnb issue.

**Closing template:**
> Closing this issue. The error originates in [library name], not in bitsandbytes. Please report it to the appropriate issue tracker.

## Other Recurring Patterns

### Undefined symbol errors (old builds)

**How to identify:** `undefined symbol: cadam32bit_grad_fp32` or `undefined symbol: cdequantize_blockwise_fp32`. Occurs with old builds or version mismatches between the compiled C library and the Python package.

**Resolution:** Upgrade to the latest version. If building from source, do a clean build.

### Questions filed as bugs

**How to identify:** The issue asks about NF4 internals (offset value, data format, quantile bins), how quantization works, or how to use a feature. Often has the `Question` label. No actual error or bug report. Common specific questions:
- How NF4 values are derived from `create_normal_map` and why they differ slightly from recomputing (floating-point rounding; the hardcoded values are canonical and avoid a scipy runtime dependency).
- Whether NF4 is a floating-point format with sign/exponent/mantissa bits — it is not; NF4 is a lookup table of 16 quantile-based values, not an IEEE-style float format.
- How `Linear8bitLt`'s `threshold` parameter works — users often assume it operates on **weights**, but it actually controls outlier detection on **activations** (inputs). Columns where activation magnitude exceeds the threshold are computed in fp16; the rest use int8.
- How to inspect which columns were quantized vs. kept in fp16 after a forward pass.
- Requests for per-layer mixed quantization (different bit widths for different layers, like llama.cpp's approach) — not currently supported.

**Resolution:** If answered in comments or by the reporter themselves, close. If useful, convert to a discussion. Consider whether the question reveals a documentation gap worth addressing.

### FSDP + bitsandbytes (optimizers and quantized models)

**How to identify:** Errors when using bnb optimizers (Adam8bit, PagedAdamW, etc.) with FSDP or FSDP2. Common errors include `AssertionError` in `_convert_all_state_info`, `AttributeError: 'int' object has no attribute 'cpu'`, or `illegal memory access`. Also includes errors loading 8-bit quantized models with FSDP, e.g., `Must flatten tensors with uniform dtype but got torch.float16 and torch.int8` — FSDP cannot handle mixed-dtype parameter groups from LLM.int8() quantization. FSDP2 optimizer state checkpointing (saving/resuming optimizer state with `bf16 + 8-bit optimizer`) also fails with assertion errors. Paged optimizers (PagedAdamW) also fail with FSDP when resuming from checkpoint.

**Current status:** FSDP support for bnb optimizers is a known gap. The maintainer has stated this repeatedly. LLM.int8() with FSDP1 is not supported and unlikely to be worked on. Track via #1633 (open, Contributions Welcome). Historical context in #89 (closed). Recent duplicates: #1732 (FSDP2 checkpointing), #1709 (FSDP1 + int8 model loading), #1381 (paged optimizer + FSDP checkpoint resume), #1403 (FSDP2 + 8-bit optimizer).

**Resolution:** Close as duplicate of #1633, noting that FSDP optimizer support is not yet available.

### DeepSpeed ZeRO-3 + quantized models

**How to identify:** Errors when using `deepspeed.zero.Init` (ZeRO-3) with bitsandbytes-quantized models. Typically occurs when trying to combine ZeRO-3 weight partitioning with pre-quantized weights or `load_in_4bit`/`load_in_8bit`.

**What happened:** ZeRO-3's weight partitioning mechanism is incompatible with pre-quantized weights. The `zero.Init` context manager expects to shard standard floating-point parameters, but quantized weights have a different internal structure. This is a limitation of the transformers + DeepSpeed integration, not a bitsandbytes bug per se.

**Resolution:** Close, noting that ZeRO-3 `zero.Init` does not support quantized weights. Users should use ZeRO-2 or load the model without ZeRO-3 `zero.Init`.

**Closing template:**
> Closing this issue. DeepSpeed ZeRO-3's `zero.Init` does not support bitsandbytes-quantized weights. The weight partitioning mechanism expects standard floating-point parameters. Consider using ZeRO stage 1 or 2 instead, or loading the model outside of `zero.Init`.

### CPU optimizer support requests

**How to identify:** Feature request asking for 8-bit or other low-bit optimizers to run on CPU (no CUDA). Common use case: DeepSpeed ZeRO-Offload where optimizer states are offloaded to CPU. Users want reduced memory for CPU-side optimizer states (e.g., 8-bit Adam on CPU for full fine-tuning of large models).

**Current status:** CPU optimizer support is tracked in #1226 (open). Recent duplicate: #1402.

**Resolution:** Close as duplicate of #1226.

### ROCm / AMD GPU build issues

**How to identify:** Build failure when compiling bitsandbytes from source with ROCm/HIP backend. Common errors include "Failed to find ROCm root directory" or hipcc-related failures. Often caused by incomplete or broken ROCm installations rather than bnb bugs.

**Resolution:** Verify the ROCm installation is complete and `ROCM_HOME`/`HIP_PATH` are set correctly. Upgrading ROCm often resolves the issue. If the user has a valid ROCm setup and still fails, it may be a real build bug.

**Closing template:**
> Closing this issue. The build failure appears to be caused by an incomplete or misconfigured ROCm installation. Please ensure ROCm is installed correctly, `ROCM_HOME` and `HIP_PATH` are set, and `hipcc` is functional. Upgrading to a recent ROCm version (6.3+) often resolves these issues.

### Colab / Jupyter runtime not restarted after upgrade

**How to identify:** `ImportError: cannot import name 'sync_gpu' from 'bitsandbytes.utils'` or similar errors where a function exists in the installed version but not in the loaded module. The user upgraded bitsandbytes via `pip install` in a Colab or Jupyter notebook but didn't restart the runtime/kernel. The old `.pyc` files or already-imported modules remain in memory, causing version mismatches between submodules (e.g., `optimizer.py` from the new version references `sync_gpu` but `utils.py` from the old version is still loaded).

**Resolution:** Instruct the user to restart their Colab runtime / Jupyter kernel after upgrading bitsandbytes. Also check for outdated dependency versions (e.g., old PEFT).

**Closing template:**
> Closing this issue. The `ImportError` indicates a version mismatch caused by upgrading bitsandbytes without restarting your Colab runtime / Jupyter kernel. After running `pip install -U bitsandbytes`, you must restart the runtime so that all modules are reloaded from the new version. Also consider upgrading related packages (peft, transformers, accelerate) to their latest versions.

### CMake + CUDA version architecture mismatch (source builds)

**How to identify:** Build failure when compiling bitsandbytes from source with CUDA 13+ and CMake < 3.31.9. CMake tries to compile for Maxwell, Pascal, or Volta architectures that CUDA 13 dropped. Error messages reference unsupported `sm_` values or nvcc compilation failures for old compute capabilities.

**What happened:** CMake versions before 3.31.9 don't know which GPU architectures were removed in CUDA 13. CMake's `CMAKE_CUDA_ARCHITECTURES` auto-detection includes architectures that the installed CUDA toolkit no longer supports, causing compilation failures. This is a CMake bug/limitation, not a bitsandbytes bug.

**Resolution:** Upgrade CMake to 3.31.9+, or manually specify supported architectures with `-DCOMPUTE_CAPABILITY=`.

**Closing template:**
> Closing this issue. CMake versions before 3.31.9 don't know which architectures CUDA 13 dropped, so they attempt to compile for unsupported targets (Maxwell, Pascal, Volta). The fix is to either upgrade CMake to 3.31.9+ or manually specify your target architectures with `-DCOMPUTE_CAPABILITY=75;80;86` (or whichever you need). This is a CMake limitation, not a bitsandbytes bug.

### EOL platforms / old glibc preventing upgrades

**How to identify:** User is on CentOS 7, RHEL 7, or another EOL Linux distribution with glibc < 2.24. They cannot install bitsandbytes > 0.42.x from PyPI because the published wheels require glibc >= 2.24 (`manylinux_2_24`). They're stuck on old versions and hitting all the legacy `cuda_setup/main.py` bugs.

**What happened:** Modern bitsandbytes wheels are built with `manylinux_2_24`, which requires glibc >= 2.24. EOL platforms like CentOS 7 (glibc 2.17) can't use them. The user can't upgrade past the broken 0.42.x versions without upgrading their OS or building from source.

**Resolution:** Close, noting that EOL platforms can't be officially supported. Suggest building from source or upgrading the OS.

**Closing template:**
> Closing this issue. The bitsandbytes wheels on PyPI require glibc >= 2.24, which means EOL platforms like CentOS 7 cannot install modern versions. We recommend upgrading your OS or building bitsandbytes from source. See the [installation docs](https://huggingface.co/docs/bitsandbytes/main/en/installation) for source build instructions.

### `prepare_model_for_kbit_training` memory concerns

**How to identify:** User reports that NF4/4-bit quantized model + LoRA uses more memory than expected, sometimes even more than bf16. The traceback or description references `prepare_model_for_kbit_training` from PEFT. Users expect quantization to always reduce memory but find backpropagation memory is higher than anticipated.

**What happened:** `prepare_model_for_kbit_training` intentionally casts adapter (LoRA) weights to float32 for training stability, which increases memory vs. keeping them in bf16. Additionally, quantized models still need to dequantize during the forward pass, and gradient computation through the dequantization step has its own memory overhead. This is by-design behavior in PEFT, not a bitsandbytes bug.

**Resolution:** Close, noting this is expected behavior. Users can skip `prepare_model_for_kbit_training` and call `model.gradient_checkpointing_enable()` directly if they want to trade off training stability for lower memory.

**Closing template:**
> Closing this issue. The higher-than-expected memory usage is by design — `prepare_model_for_kbit_training` (from PEFT) casts adapter weights to float32 for training stability. You can skip it and call `model.gradient_checkpointing_enable()` directly if you prefer lower memory at the cost of potential training instability. This is a PEFT behavior, not a bitsandbytes issue.

### Insufficient information / no reproduction

**How to identify:** Issue reports an error but provides no bitsandbytes version, no `python -m bitsandbytes` output, no minimal reproduction code, or no response to maintainer follow-up questions. May also include screenshot-only bug reports where the image is inaccessible, bare model support requests with no detail (e.g., just a model name with "supported?"), or vague performance complaints without measurements.

**Resolution:** Ask for specifics. If no response after a reasonable period, close.

**Closing template:**
> Closing this issue due to insufficient information to reproduce or investigate. If you're still experiencing this problem, please open a new issue with: (1) the output of `python -m bitsandbytes`, (2) your full environment details (OS, Python, PyTorch, GPU), and (3) a minimal code snippet that reproduces the error.

### Quantized model output quality (NaN, large numeric differences)

**How to identify:** User reports NaN values in model logits/outputs after 8-bit or 4-bit quantization, or reports that quantized model outputs are very different from the unquantized model. Often on old bitsandbytes versions (0.42.x or earlier). May also be caused by using float16 instead of bfloat16 on Ampere+ GPUs.

**Resolution:** Ask the user to upgrade bitsandbytes and try with `torch_dtype=torch.bfloat16`. If on the latest version with bfloat16 and the issue persists with a minimal repro, it may be a real bug. Otherwise close.

**Closing template:**
> Closing this issue. NaN or large numeric differences in quantized outputs are often caused by using an old bitsandbytes version or float16 dtype. Please upgrade to the latest bitsandbytes and use `torch_dtype=torch.bfloat16`. If the issue persists, please open a new issue with a minimal reproduction.

### 4-bit model loading drops certain weights

**How to identify:** Certain model architectures lose specific weights when loaded with `load_in_4bit=True` via transformers. The saved model's `state_dict` is missing expected keys (e.g., `decoder.lm_head.weight`). Works correctly without quantization. Typically affects models with tied/shared weights or non-standard architectures (e.g., VisionEncoderDecoder, Donut).

**What happened:** The transformers `load_in_4bit` integration may not correctly handle tied weights or non-standard model architectures. Weights that are shared or aliased in the original model may get dropped during the quantization loading process.

**Resolution:** This is likely a transformers integration issue. Check if the model architecture has tied weights. Suggest filing against transformers if it's a loading issue in their quantization code path.
