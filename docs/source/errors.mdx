# Troubleshoot

## No kernel image available

This problem arises with the cuda version loaded by bitsandbytes is not supported by your GPU, or if you pytorch CUDA version mismatches.

To solve this problem you need to debug ``$LD_LIBRARY_PATH``, ``$CUDA_HOME`` as well as ``$PATH``. You can print these via ``echo $PATH``. You should look for multiple paths to different CUDA versions. This can include versions in your anaconda path, for example ``$HOME/anaconda3/lib``. You can check those versions via ``ls -l $HOME/anaconda3/lib/*cuda*`` or equivalent paths. Look at the CUDA versions of files in these paths. Does it match with ``nvidia-smi``?

If you are feeling lucky, you can also try to compile the library from source. This can be still problematic if your PATH variables have multiple cuda versions. As such, it is recommended to figure out path conflicts before you proceed with compilation.

## `fatbinwrap`

This error occurs if there is a mismatch between CUDA versions in the C++ library and the CUDA part. Make sure you have right CUDA in your `$PATH` and `$LD_LIBRARY_PATH` variable. In the conda base environment you can find the library under:

```bash
ls $CONDA_PREFIX/lib/*cudart*
```
Make sure this path is appended to the `LD_LIBRARY_PATH` so bnb can find the CUDA runtime environment library (cudart).

If this does not fix the issue, please try compilation from source next.

If this does not work, please open an issue and paste the printed environment if you call `make` and the associated error when running bnb.

## Library not found: version mismatch

The library filename encodes the version: `libbitsandbytes_cuda{major}{minor}` for CUDA, `libbitsandbytes_rocm{major}{minor}` for ROCm. bitsandbytes selects which one to load based on what PyTorch reports:

```python
import torch
print(torch.version.cuda)  # e.g. "12.8" -> looks for libbitsandbytes_cuda128
print(torch.version.hip)   # e.g. "7.2"  -> looks for libbitsandbytes_rocm72
```

bitsandbytes will automatically fall back to the closest available pre-compiled version if an exact match is not found, and log a warning. For example, if your PyTorch was built with CUDA 12.9 but bitsandbytes only ships 12.8, it will load 12.8 automatically.

If you see an error like `No compatible CUDA library found`, it means no compatible pre-compiled library could be found at all. To resolve this:

1. **Compile from source** to produce a library matching your exact toolkit version. See the [installation guide](installation) for instructions.
2. **Override the version at runtime** with an environment variable to force loading a specific pre-compiled version:
   ```bash
   # Linux / macOS
   export BNB_CUDA_VERSION=128   # or BNB_ROCM_VERSION=72

   # Windows (cmd)
   set BNB_CUDA_VERSION=128
   ```
   The value must be digits only, e.g. `128` for CUDA 12.8 or `72` for ROCm 7.2.
