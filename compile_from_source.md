# Compiling from source

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

## ROCm
Basic steps.
1. `make hip`
2. `python setup.py install`

To run these steps you will need to have the hipcc compiler installed that comes with a ROCm installation. hipcc need to be in your path.

If your GPU is not compatible with ROCm, for example if you have an RX 6700 XT, cross compiling to a similar GPU should work.
To do that set HCC_AMDGPU_TARGET to that similar GPU for example gfx1030 should work for an RX 6700 XT.

