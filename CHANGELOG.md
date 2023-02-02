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
 - Added Analysis Adam to track 8-bit vs 32-bit quantization errors over time.
 - Make compilation more user friendly.

Bug fixes:
 - fixed "undefined symbol: \_\_fatbinwrap_38" error for P100 GPUs on CUDA 10.1 (#5)

Docs:
 - Added docs with instructions to compile from source.


### 0.26.0:

Features:
 - Added Adagrad (without grad clipping) as 32-bit and 8-bit block-wise optimizer.
 - Added AdamW (copy of Adam with weight decay init 1e-2). #10
 - Introduced ModuleConfig overrides which can be seamlessly be used at initialization time of a module.
 - Added `bnb.nn.Embedding` layer which runs at 32-bit but without the layernorm. This works well if you need to fine-tune pretrained models that do not have a embedding layer norm. #19

Bug fixes:
 - Fixed a bug where weight decay was incorrectly applied to 32-bit Adam. #13
 - Fixed an unsafe use of eval. #8
 - Fixed a bug where the StableEmbedding layer 32-bit optimizer override would not work without registering the whole model first (`bnb.optim.GlobalOptimManager.get_instance().register_parameters(model.parameters())`).  #13 #15

Docs:
 - Added instructions how to solve "\_\_fatbinwrap_" errors.


### 0.30.0

#### 8-bit Inference Update

Features:
 - Added 8-bit matrix multiplication form cuBLAS,  and cuBLASLt as well as multiple GEMM kernels (GEMM, GEMMEx, GEMMLt)
 - Added 8-bit Linear layers with 8-bit Params that perform memory efficient inference with an option for 8-bit mixed precision matrix decomposition for inference without performance degradation
 - Added quantization methods for "fake" quantization as well as optimized kernels vector-wise quantization and equalization as well as optimized cuBLASLt transformations
 - CPU only build now available (Thank you, @mryab)

Deprecated:
 - Pre-compiled release for CUDA 9.2, 10.0, 10.2 no longer available

### 0.31.0

#### 8-bit Inference and Packaging Update

Features:
 - added direct outlier extraction. This enables outlier extraction without fp16 weights without performance degradation.
 - Added automatic CUDA SETUP procedure and packaging all binaries into a single bitsandbytes package.

### 0.32.0

#### 8-bit Inference Performance Enhancements

We added performance enhancements for small models. This makes small models about 2x faster for LLM.int8() inference.

Features:
 - Int32 dequantization now supports fused biases.
 - Linear8bitLt now uses a fused bias implementation.
 - Change `.data.storage().data_ptr()` to `.data.data_ptr()` to enhance inference performance.

Bug fixes:
 - Now throws and error if LLM.int8() is used on a GPU that is not supported.
 - Enhances error messaging if CUDA SETUP fails.


### 0.33.0

#### Various bug fixes

Features:
 - CPU quantization now supports a variable `blocksize` variable to enhance quantization speed or precision.

Bug fixes:
 - fixed an issue in CPU quantization where tensors with more than 2^31 elements would fail 19a7adca7a6c9bf7061a384d7e9d9b13676a1a88
 - fixed a bug where cpu binaries would fail if no GPU would be detected eab4d8232d558f2e6bd7f7cc3d00e2e6e94f4e80
 - fixed an issue where cpu binaries cause additional stdout messages 92a3363096e10ad6a5c4e944af898bd1186d806a
 - fixed an import of bnb.utils 2e630b55f51d454f3bd723dffda68a07ef93190c

We thank @mryab, @mbrukman, @chessgecko, @dbaranchuk for pull request with bug fixes and new features.


### 0.34.0

#### Bug fixes and memory efficient backprop

Features:
 - Linear8bitLt layer now supports `memory_efficient_backward=True` which enables backprop of gradients through frozen weights.

Bug fixes:
 - fixed an issue where too many threads were created in blockwise quantization on the CPU for large tensors


### 0.35.0

#### CUDA 11.8 support and bug fixes

Features:
 - CUDA 11.8 support added and binaries added to the PyPI release.

Bug fixes:
 - fixed a bug where too long directory names would crash the CUDA SETUP #35 (thank you @tomaarsen)
 - fixed a bug where CPU installations on Colab would run into an error  #34 (thank you @tomaarsen)
 - fixed an issue where the default CUDA version with fast-DreamBooth was not supported #52

### 0.35.1

Features:
 - Added CUDA instruction generator to fix some installations.

Bug fixes:
 - Fixed a problem where warning messages would be displayed even though everything worked correctly.

### 0.35.2

Bug fixes:
 - Fixed a bug where the CUDA setup failed due to a wrong function call.

### 0.35.3

Bug fixes:
 - Fixed a bug in the CUDA Setup which led to an incomprehensible error if no GPU was detected.

### 0.35.4

Bug fixes:
 - Fixed a bug in the CUDA Setup failed with the cuda runtime was found, but not the cuda library.
 - Fixed a bug where not finding the cuda runtime led to an incomprehensible error.


### 0.36.0

#### Improvements, Ada/Hopper support, fake k-bit quantization.

Features:
 - CUDA 11.8 and 12.0 support added
 - support for Ada and Hopper GPUs added (compute capability 8.9 and 9.0)
 - support for fake k-bit block-wise quantization for Int, Float, quantile quantization, and dynamic exponent data types added
 - Added CUDA instruction generator to fix some installations.
 - Added additional block sizes for quantization {64, 128, 256, 512, 1024}
 - Added SRAM Quantile algorithm to quickly estimate less than 256 quantiles
 - Added option to suppress the bitsandbytes welcome message (@Cyberes)

Regression:
 - Compute capability 3.0 removed: GTX 600s and 700s series is no longer supported (except GTX 780 and GTX 780 Ti)

Bug fixes:
 - fixed a bug where too long directory names would crash the CUDA SETUP #35 (@tomaarsen)
 - fixed a bug where CPU installations on Colab would run into an error  #34 (@tomaarsen)
 - fixed an issue where the default CUDA version with fast-DreamBooth was not supported #52
 - fixed a bug where the CUDA setup failed due to a wrong function call.
 - fixed a bug in the CUDA Setup which led to an incomprehensible error if no GPU was detected.
 - fixed a bug in the CUDA Setup failed with the cuda runtime was found, but not the cuda library.
 - fixed a bug where not finding the cuda runtime led to an incomprehensible error.
 - fixed a bug where with missing CUDA the default was an error instead of the loading the CPU library
 - fixed a bug where the CC version of the GPU was not detected appropriately (@BlackHC)
 - fixed a bug in CPU quantization which lead to errors when the input buffer exceeded 2^31 elements

Improvements:
 - multiple improvements in formatting, removal of unused imports, and slight performance improvements (@tomaarsen)
 - StableEmbedding layer now has device and dtype parameters to make it 1:1 replaceable with regular Embedding layers (@lostmsu)
 - runtime performance of block-wise quantization slightly improved
 - added error message for the case multiple libcudart.so are installed and bitsandbytes picks the wrong one


### 0.37.0

#### Int8 Matmul + backward support for all GPUs

Features:
 - Int8 MatmulLt now supports backward through inversion of the ColTuring/ColAmpere format. Slow, but memory efficient. Big thanks to @borzunov
 - Int8 now supported on all GPUs. On devices with compute capability < 7.5, the Int weights are cast to 16/32-bit for the matrix multiplication. Contributed by @borzunov

Improvements:
 - Improved logging for the CUDA detection mechanism.
