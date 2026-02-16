# Compiling bitsandbytes for H100 and L40 GPUs

This guide shows how to compile bitsandbytes from source specifically optimized for NVIDIA H100 and L40 GPUs.

## Prerequisites

- CMake >= 3.22.1
- Python >= 3.9
- GCC (version 9+ recommended)
- CUDA Toolkit (11.8+)
- PyTorch with CUDA support

Verify your system:
```bash
cmake --version
python3 --version
gcc --version
nvcc --version
```

## GPU Compute Capabilities

- **L40**: Compute Capability 8.9 (sm_89)
- **H100**: Compute Capability 9.0 (sm_90)

## Compilation Steps

### 1. Clean any previous build configuration

```bash
cd /path/to/bitsandbytes
rm -rf CMakeCache.txt CMakeFiles/ build/
```

### 2. Configure CMake for H100 and L40

```bash
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="89;90" -S .
```

This configures the build to target only compute capabilities 89 (L40) and 90 (H100), significantly reducing compilation time compared to building for all architectures.

### 3. Compile the library

```bash
make -j$(nproc)
```

This will create `bitsandbytes/libbitsandbytes_cuda<VERSION>.so` where `<VERSION>` matches your CUDA Toolkit version (e.g., `cuda124` for CUDA 12.4).

### 4. Install the package

```bash
pip install -e .
```

Use `-e` flag for editable/development install, or omit it for regular installation.

### 5. Handle PyTorch CUDA version mismatch (if needed)

If your PyTorch was compiled with a different CUDA version than your Toolkit, you may need to create a symlink:

```bash
# Example: PyTorch uses CUDA 12.8, but you compiled with CUDA 12.4
ln -sf libbitsandbytes_cuda124.so bitsandbytes/libbitsandbytes_cuda128.so
```

Alternatively, set the environment variable:
```bash
export BNB_CUDA_VERSION=124  # Use your compiled CUDA version
```

### 6. Verify installation

```bash
python3 -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}'); print('Success!')"
```

## Expected Output

After compilation, you should see:
- Binary file: `bitsandbytes/libbitsandbytes_cuda<VERSION>.so` (approximately 7MB when targeting only sm_89 and sm_90)
- Successful import in Python with no errors

## Compilation Time

Building for only H100/L40 (2 architectures) takes approximately **1-2 minutes** compared to **5+ minutes** when building for all 14+ compute capabilities.

## Troubleshooting

### Warning messages during compilation
Warnings like "variable declared but never referenced" are harmless and can be ignored.

### Wrong CUDA binary error
If you see `Configured CUDA binary not found`, check:
1. The compiled `.so` file exists in `bitsandbytes/` directory
2. The CUDA version matches or create a symlink as shown in step 5
3. Use `BNB_CUDA_VERSION` environment variable to override

### CUDA version check
```bash
# Check your CUDA Toolkit version
nvcc --version

# Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"
```

## Notes

- The compiled library will **only work on GPUs with compute capability 8.9 or 9.0** (L40 and H100)
- For other GPUs, you'll need to recompile with appropriate compute capabilities
- The `-DCOMPUTE_CAPABILITY` flag accepts a semicolon-separated list: e.g., `"75;80;89;90"` for T4, A100, L40, and H100
