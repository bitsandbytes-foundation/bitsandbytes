### 0.43.1

#### Improvements:

- Improved the serialization format for 8-bit weights; this change is fully backwards compatible. (#1164, thanks to @younesbelkada for the contributions and @akx for the review).
- Added CUDA 12.4 support to the Linux x86-64 build workflow, expanding the library's compatibility with the latest CUDA versions. (#1171, kudos to @matthewdouglas for this addition).
- Docs enhancement: Improved the instructions for installing the library from source. (#1149, special thanks to @stevhliu for the enhancements).

#### Bug Fixes

- Fix 4bit quantization with blocksize = 4096, where an illegal memory access was encountered. (#1160, thanks @matthewdouglas for fixing and @YLGH for reporting)

#### Internal Improvements:

- Tests: improve memory usage (#1147, thanks @matthewdouglas)
- Add CUDA 12.4 to docs/install helper (#1136, thanks @matthewdouglas)
- Minor type/doc fixes (#1128, thanks @akx)
- Reformat Python code with Ruff (#1081, thanks @akx)
- Rework of CUDA/native-library setup and diagnostics (#1041, thanks @akx)

### 0.43.0

#### Improvements and New Features:

- QLoRA + FSDP official support is now live! https://github.com/TimDettmers/bitsandbytes/pull/970 by @warner-benjamin and team - with FSDP you can train very large models (70b scale) on multiple 24GB consumer-type GPUs. See https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html for more details.
- Introduced improvements to the CI process for enhanced performance and efficiency during builds, specifically enabling more effective cross-compilation on Linux platforms. This was accomplished by deprecating Make and migrating to Cmake, as well as implementing new corresponding workflows. Huge thanks go to @wkpark, @rickardp, @matthewdouglas and @younesbelkada; #1055, #1050, #1111.
- Windows should be officially supported in bitsandbytes if you install the library from source. See: https://huggingface.co/docs/bitsandbytes/main/en/index for more details
- Updated installation instructions to provide more comprehensive guidance for users. This includes clearer explanations and additional tips for various setup scenarios, making the library more accessible to a broader audience (@rickardp, #1047).
- Enhanced the library's compatibility and setup process, including fixes for CPU-only installations and improvements in CUDA setup error messaging. This effort aims to streamline the installation process and improve user experience across different platforms and setups (@wkpark, @akx, #1038, #996, #1012).
- Setup a new documentation at https://huggingface.co/docs/bitsandbytes/main with extensive new sections and content to help users better understand and utilize the library. Especially notable are the new API docs. (big thanks to @stevhliu and @mishig25 from HuggingFace #1012). The API docs have been also addressed in #1075.

#### Bug Fixes:

- Addressed a race condition in kEstimateQuantiles, enhancing the reliability of quantile estimation in concurrent environments (@pnunna93, #1061).
- Fixed various minor issues, including typos in code comments and documentation, to improve code clarity and prevent potential confusion (@Brian Vaughan, #1063).

#### Backwards Compatibility

- After upgrading from `v0.42` to `v0.43`, when using 4bit quantization, models may generate slightly different outputs (approximately up to the 2nd decimal place) due to a fix in the code. For anyone interested in the details, [see this comment](https://github.com/TimDettmers/bitsandbytes/discussions/1094#discussioncomment-8984069).

#### Internal and Build System Enhancements:

- Implemented several enhancements to the internal and build systems, including adjustments to the CI workflows, portability improvements, and build artifact management. These changes contribute to a more robust and flexible development process, ensuring the library's ongoing quality and maintainability (@rickardp, @akx, @wkpark, @matthewdouglas; #949, #1053, #1045, #1037).

#### Contributors:

This release is made possible thanks to the many active contributors that submitted PRs and many others who contributed to discussions, reviews, and testing. Your efforts greatly enhance the library's quality and user experience. It's truly inspiring to work with such a dedicated and competent group of volunteers and professionals!

We give a special thanks to @TimDettmers for managing to find a little bit of time for valuable consultations on critical topics, despite preparing for and touring the states applying for professor positions. We wish him the utmost success!

We also extend our gratitude to the broader community for your continued support, feedback, and engagement, which play a crucial role in driving the library's development forward.

### 0.42.0

Features:

- 4-bit serialization now supported. This enables 4-bit load/store. Thank you @poedator #753
- the bitsandbytes library now has a version attribute: `bitsandbytes.__version__` @rasbt #710

Bug fixes:

- Fixed bugs in dynamic exponent data type creation. Thank you @RossM, @KohakuBlueleaf, @ArrowM #659 #227 #262 #152
- Fixed an issue where 4-bit serialization would fail for layers without double quantization #868. Thank you, @poedator
- Fixed an issue where calling .to() or .cuda() on a 4-bit layer twice would result in an error #867. Thank you, @jph00
- Fixed a bug where a missing access permission in a path searched for CUDA would lead to an error @osma #677
- Fixed a bug where the GOOGLE_VM_CONFIG_LOCK_FILE variable could cause errors in colab environments @akrentsel @xaptronic #715 #883 #622
- Fixed a bug where kgetColRowStats (LLM.int8()) would fail for certain dimensions @LucQueen @905
- Fixed a bug where the adjusted regular Embedding layer was not available via bnb.nn.Embedding @neel04 #563
- Fixed added missing scipy requirement @dulalbert #525

### 0.41.3

Bug fixes:

- Fixed an issue where 4-bit serialization would fail for layers without double quantization #868. Thank you, @poedator
- Fixed an issue where calling .to() or .cuda() on a 4-bit layer twice would result in an error #867. Thank you, @jph00

### 0.41.2

Feature:

- 4-bit serialization now supported. This enables 4-bit load/store. Thank you @poedator #753

### 0.41.1

Bug fixes:

- Fixed bugs in dynamic exponent data type creation. Thank you @RossM, @KohakuBlueleaf, @ArrowM #659 #227 #262 #152

### 0.41.0

Features:

- Added precompiled CUDA 11.8 binaries to support H100 GPUs without compilation #571
- CUDA SETUP now no longer looks for libcuda and libcudart and relies PyTorch CUDA libraries. To manually override this behavior see: how_to_use_nonpytorch_cuda.md. Thank you @rapsealk

Bug fixes:

- Fixed a bug where the default type of absmax was undefined which leads to errors if the default type is different than torch.float32. # 553
- Fixed a missing scipy dependency in requirements.txt. #544
- Fixed a bug, where a view operation could cause an error in 8-bit layers.
- Fixed a bug where CPU bitsandbytes would during the import. #593 Thank you @bilelomrani
- Fixed a but where a non-existent LD_LIBRARY_PATH variable led to a failure in python -m bitsandbytes #588
- Removed outdated get_cuda_lib_handle calls that lead to errors. #595 Thank you @ihsanturk
- Fixed bug where read-permission was assumed for a file. #497
- Fixed a bug where prefetchAsync lead to errors on GPUs that do not support unified memory but not prefetching (Maxwell, SM52). #470 #451 #453 #477 Thank you @jllllll and @stoperro

Documentation:

- Improved documentation for GPUs that do not support 8-bit matmul. #529
- Added description and pointers for the NF4 data type. #543

User experience:

- Improved handling of default compute_dtype for Linear4bit Layers, so that compute_dtype = input_dtype if the input data type is stable enough (float32, bfloat16, but not float16).

Performance:

- improved 4-bit inference performance for A100 GPUs. This degraded performance for A40/RTX3090 and RTX 4090 GPUs slightly.

### 0.40.2

Bug fixes:

- Fixed a but where a non-existent LD_LIBRARY_PATH variable led to a failure in python -m bitsandbytes #588
- Removed outdated get_cuda_lib_handle calls that lead to errors. #595 Thank you @ihsanturk
- Fixed bug where read-permission was assumed for a file. #497
- Fixed a bug where prefetchAsync lead to errors on GPUs that do not support unified memory but not prefetching (Maxwell, SM52). #470 #451 #453 #477 Thank you @jllllll and @stoperro

### 0.40.1

Features:

- Added precompiled CUDA 11.8 binaries to support H100 GPUs without compilation #571
- CUDA SETUP now no longer looks for libcuda and libcudart and relies PyTorch CUDA libraries. To manually override this behavior see: how_to_use_nonpytorch_cuda.md. Thank you @rapsealk

Bug fixes:

- Fixed a bug where the default type of absmax was undefined which leads to errors if the default type is different than torch.float32. # 553
- Fixed a missing scipy dependency in requirements.txt. #544
- Fixed a bug, where a view operation could cause an error in 8-bit layers.
- Fixed a bug where CPU bitsandbytes would during the import. #593 Thank you @bilelomrani

Documentation:

- Improved documentation for GPUs that do not support 8-bit matmul. #529
- Added description and pointers for the NF4 data type. #543

### 0.40.0

Features:

- Added 4-bit inference kernels for batch size=1. Currently support are the NF4, FP4 data types.
- Added support for quantizations of bfloat16 input data.

Bug fixes:

- Added `device` variable for bitsandbytes layers to be compatible with PyTorch layers.

Deprecated:

- Binaries for CUDA 11.2, 11.6 no longer ship with `pip install bitsandbytes` and need to be compiled from source.

### 0.39.0

Features:

- 4-bit matrix multiplication for Float4 and NormalFloat4 data types.
- Added 4-bit quantization routines
- Doubled quantization routines for 4-bit quantization
- Paged optimizers for Adam and Lion.
- bfloat16 gradient / weight support for Adam and Lion with 8 or 32-bit states.

Bug fixes:

- Fixed a bug where 8-bit models consumed twice the memory as expected after serialization

Deprecated:

- Kepler binaries (GTX 700s and Tesla K40/K80) are not longer provided via pip and need to be compiled from source. Kepler support might be fully removed in the future.

### 0.38.1

Features:

- Added Int8 SwitchBack layers
- Added Fake FP8 layers for research purposes (available under `bnb.research.nn. ...`)

### 0.38.0

#### 8-bit Lion, Load/Store 8-bit Models directly from/to HF Hub

Features:

- Support for 32 and 8-bit Lion has been added. Thank you @lucidrains
- Support for serialization of Linear8bitLt layers (LLM.int8()). This allows to store and load 8-bit weights directly from the HuggingFace Hub. Thank you @myrab
- New bug report features `python -m bitsandbytes` now gives extensive debugging details to debug CUDA setup failures.

Bug fixes:

- Fixed a bug where some bitsandbytes methods failed in a model-parallel setup on multiple GPUs. Thank you @tonylins
- Fixed a bug where cudart.so libraries could not be found in newer PyTorch releases.

Improvements:

- Improved the CUDA Setup procedure by doing a more extensive search for CUDA libraries

Deprecated:

- Devices with compute capability 3.0 (GTX 700s, K10) and 3.2 (Tegra K1, Jetson TK1) are now deprecated and support will be removed in 0.39.0.
- Support for CUDA 10.0 and 10.2 will be removed in bitsandbytes 0.39.0

### 0.37.0

#### Int8 Matmul + backward support for all GPUs

Features:

- Int8 MatmulLt now supports backward through inversion of the ColTuring/ColAmpere format. Slow, but memory efficient. Big thanks to @borzunov
- Int8 now supported on all GPUs. On devices with compute capability \< 7.5, the Int weights are cast to 16/32-bit for the matrix multiplication. Contributed by @borzunov

Improvements:

- Improved logging for the CUDA detection mechanism.

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

### 0.35.4

Bug fixes:

- Fixed a bug in the CUDA Setup failed with the cuda runtime was found, but not the cuda library.
- Fixed a bug where not finding the cuda runtime led to an incomprehensible error.

### 0.35.3

Bug fixes:

- Fixed a bug in the CUDA Setup which led to an incomprehensible error if no GPU was detected.

### 0.35.2

Bug fixes:

- Fixed a bug where the CUDA setup failed due to a wrong function call.

### 0.35.1

Features:

- Added CUDA instruction generator to fix some installations.

Bug fixes:

- Fixed a problem where warning messages would be displayed even though everything worked correctly.

### 0.35.0

#### CUDA 11.8 support and bug fixes

Features:

- CUDA 11.8 support added and binaries added to the PyPI release.

Bug fixes:

- fixed a bug where too long directory names would crash the CUDA SETUP #35 (thank you @tomaarsen)
- fixed a bug where CPU installations on Colab would run into an error  #34 (thank you @tomaarsen)
- fixed an issue where the default CUDA version with fast-DreamBooth was not supported #52

### 0.34.0

#### Bug fixes and memory efficient backprop

Features:

- Linear8bitLt layer now supports `memory_efficient_backward=True` which enables backprop of gradients through frozen weights.

Bug fixes:

- fixed an issue where too many threads were created in blockwise quantization on the CPU for large tensors

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

### 0.31.0

#### 8-bit Inference and Packaging Update

Features:

- added direct outlier extraction. This enables outlier extraction without fp16 weights without performance degradation.
- Added automatic CUDA SETUP procedure and packaging all binaries into a single bitsandbytes package.

### 0.30.0

#### 8-bit Inference Update

Features:

- Added 8-bit matrix multiplication form cuBLAS,  and cuBLASLt as well as multiple GEMM kernels (GEMM, GEMMEx, GEMMLt)
- Added 8-bit Linear layers with 8-bit Params that perform memory efficient inference with an option for 8-bit mixed precision matrix decomposition for inference without performance degradation
- Added quantization methods for "fake" quantization as well as optimized kernels vector-wise quantization and equalization as well as optimized cuBLASLt transformations
- CPU only build now available (Thank you, @mryab)

Deprecated:

- Pre-compiled release for CUDA 9.2, 10.0, 10.2 no longer available

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

- Added instructions how to solve "\_\_fatbinwrap\_" errors.

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

### 0.0.24:

- Fixed a bug where a float/half conversion led to a compilation error for CUDA 11.1 on Turning GPUs.
- removed Apex dependency for bnb LAMB

### 0.0.23:

Bugs:

- Unified quantization API: each quantization function now returns `Q, S` where `Q` is the quantized tensor and `S` the quantization state which may hold absolute max values, a quantization map or more. For dequantization all functions now accept the inputs `Q, S` so that `Q` is dequantized with the quantization state `S`.
- Fixed an issue where the CUDA 11.1 binary was not compiled with the right headers

API changes:

- Block-wise quantization for optimizers now enabled by default

Features:

- Block-wise quantization routines now support CPU Tensors.

### 0.0.22:

- Fixed an error where a `reset_parameters()` call on the `StableEmbedding` would lead to an error in older PyTorch versions (from 1.7.0).

### 0.0.21

- Ampere, RTX 30 series GPUs now compatible with the library.
