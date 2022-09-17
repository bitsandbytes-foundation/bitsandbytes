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
