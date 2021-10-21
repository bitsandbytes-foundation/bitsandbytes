### 0.0.21
- Ampere, RTX 30 series GPUs now compatible with the library.

### 0.0.22:

- Fixed an error where a `reset_parameters()` call on the `StableEmbedding` would lead to an error in older PyTorch versions (from 1.7.0).

### 0.0.23:

Bugs:
 - Unified quantization API: each quantization function now returns `Q, S` where `Q` is the quantized tensor and `S` the quantization state which may hold absolute max values, a quantization map or more. For dequantization all functions now accept the inputs `Q, S` so that `Q` is dequantized with the quantization state `S`.
 - Fixed an issue where the CUDA 11.1 binary was not compiled with the right headers

API changes:
 - Block-wise quantization for optimizers now enabled by default

Features:
 - Block-wise quantization routines now support CPU Tensors.


### 0.0.24:

- Fixed a bug where a float/half conversion led to a compilation error for CUDA 11.1 on Turning GPUs.
- removed Apex dependency for bnb LAMB

### 0.0.25:

Features:
 - Added `skip_zeros` for block-wise and 32-bit optimizers. This ensures correct updates for sparse gradients and sparse models.
 - Added support for Kepler GPUs. (#4)

Bug fixes:
 - fixed "undefined symbol: \_\_fatbinwrap_38" error for P100 GPUs on CUDA 10.1 (#5)


