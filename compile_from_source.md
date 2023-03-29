# Compiling from source

## Windows

cpu: most tests fail, this library requires CUDA to use the special bits.

Ensure you have your environment you want to bring in bitsandbytes. (a bloom setup, textgen-ui, etc) - via conda.
I'd suggest to install MAMBA  and use it, as it's way faster.

IMPORTANT - ensure your environment.yml matches your installed CUDA version in Visual Studio. Keep One VERSION.

1. environment.yml. Replace/tweak as necesarry (remove cuda, change version):
```
name: mycompileenv
channels:
  - pytorch
  - nvidia
  - huggingface
  - conda-forge
  - anaconda
    
dependencies:
  - python=3.10.9
  - pytorch-cuda=11.7 
  - pytorch
  - torchvision
  - torchaudio 
  - transformers
  - cudatoolkit=11.7
  - jupyter
  - notebook
  - pytest
  - einops
  - scipy
```

2. Then open POWERSHELL
```
conda install -c conda-forge mamba
mamba env create
mamba env activate mycompileenv
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -List -Arch amd64
```
At this point - select your visual studio installation - aka hit 1

3. Go into your bitsandbytes folder and run `cmake-gui -S . -B ./build` 
5. Hit Configure
6. Set `cuda=11.7`, nothing, or other version in the `Optional toolset to use` when selecting the generator to . You can leave it blank. If you don't see the generate, delete the `build` folder and run cmake-gui again with the above command line.

The `Optional toolset to use` will determine the dll name. You need to have that CUDA toolkit and VS integration installed using the NVIDIA installer
7. You should see in the log some info:
```
CONFIGURE_LOG
Configuring using Cuda Compiler 11.7.64; Visual Studio Integration: 11.7

CONFIGURE_LOG CUDA Targeting feature level 11.x, with architectures 52
CONFIGURE_LOG Shared library name being used: libbitsandbytes_cuda117
```
7. Set `CUDA_TARGET_ARCH_FEATURE_LEVEL` to you desired feature level. Supported: 10.x, 11.0, 11.x, 12.x
8. `CMAKE_CUDA_ARCHITECTURES` is overwritten by `CUDA_TARGET_ARCH_FEATURE_LEVEL` for now
9. Select other options. Hit Generate.
10. Open Visual Studio and select Release as configuration. Build Solution
11. Everything should be copied in the right place
12. run tests `python -m pytest`. You may need to use `mamba` to install other modules
13. build wheel `mamba install build` and then `python -m build --wheel`
14. install wheel  `pip install .\dist\*.whl`


## Linux
Basic steps.
1. `make [target]` where `[target]` is among `cuda92, cuda10x, cuda110, cuda11x, cuda12x, cpuonly`
2. `CUDA_VERSION=XXX python setup.py install`

To run these steps you will need to have the nvcc compiler installed that comes with a CUDA installation. If you use anaconda (recommended) then you can figure out which version of CUDA you are using with PyTorch via the command `conda list | grep cudatoolkit`. Then you can install the nvcc compiler by downloading and installing the same CUDA version from the [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).

For your convenience, there is an installation script in the root directory that installs CUDA 11.1 locally and configures it automatically. After installing you should add the `bin` sub-directory to the `$PATH` variable to make the compiler visible to your system. To do this you can add this to your `.bashrc` by executing these commands:
```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
source ~/.bashrc
```

By default, the Makefile will look at your `CUDA_HOME` environmental variable to find your CUDA version for compiling the library. If this path is not set it is inferred from the path of your `nvcc` compiler.

Either `nvcc` needs to be in path for the `CUDA_HOME` variable needs to be set to the CUDA directory root (e.g. `/usr/local/cuda`) in order for compilation to succeed

If you have problems compiling the library with these instructions from source, please open an issue.
