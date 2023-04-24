# Common Problems and Solutions

If you encounter any other error not listed here please create an issue. This will help resolve your problem and will help out others in the future.

## No kernel image available

This problem arises with the cuda version loaded by bitsandbytes is not supported by your GPU, or if you pytorch CUDA version mismatches. To solve this problem you need to debug ``$LD_LIBRARY_PATH``, ``$CUDA_HOME``, ``$PATH``. You can print these via ``echo $PATH``. You should look for multiple paths to different CUDA versions. This can include versions in your anaconda path, for example ``$HOME/anaconda3/lib``. You can check those versions via ``ls -l $HOME/anaconda3/lib/*cuda*`` or equivalent paths. Look at the CUDA versions of files in these paths. Does it match with ``nvidia-smi``?

If you are feeling lucky, you can also try to compile the library from source. This can be still problematic if your PATH variables have multiple cuda versions. As such, it is recommended to figure out path conflicts before you proceed with compilation.

## fatbinwrap

This error occurs if there is a mismatch between CUDA versions in the C++ library and the CUDA part. Make sure you have right CUDA in your $PATH and $LD_LIBRARY_PATH variable. In the conda base environment you can find the library under:

```bash
ls $CONDA_PREFIX/lib/*cudart*
```

Make sure this path is appended to the `LD_LIBRARY_PATH` so bnb can find the CUDA runtime environment library (cudart).

If this does not fix the issue, please try [compilation from source](compile_from_source.md) next.

If this does not work, please open an issue and paste the printed environment if you call `make` and the associated error when running bnb.

## GPU not detected on Windows WSL environment

See [Issue #336](https://github.com/TimDettmers/bitsandbytes/issues/336) for the full details. This happens when a CUDA capable
GPU is correctly detected on Windows (as seen by running `nvidia-smi`), but `bitsandbytes` does not seem to have access to it.
This shows up when running `python -m bitsandbytes` in the WSL Ubuntu environment where we installed the module. The output
looks like this:

```text
bin /home/kayvan/.miniconda3/envs/textgen/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so
/home/kayvan/.miniconda3/envs/textgen/lib/python3.10/site-packages/bitsandbytes/cextension.py:33: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
CUDA SETUP: WARNING! libcuda.so not found! Do you have a CUDA driver installed? If you are on a cluster, make sure you are on a CUDA machine!
[...]
```

According to the [official NVIDIA documentation on using CUDA in WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2):

> Once a Windows NVIDIA GPU driver is installed on the system, CUDA becomes available within WSL 2.
> The CUDA driver installed on Windows host will be stubbed inside the WSL 2 as libcuda.so, therefore
> users must not install any NVIDIA GPU Linux driver within WSL 2

This was the clue that pointed to the solution. In the WSL Ubuntu environment, the CUDA stub `libcuda.so` is
installed in `/usr/lib/wsl/lib` and the solution is to add the following two lines to your bash
startup (`~/.bashrc`):

```bash
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64"
```

With this in place, the module finds the CUDA driver and loads correctly. You can verify this by running
the python script [cuda_check.py](https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549)
in your WSL Ubuntu environment in two ways. First, with no `LD_LIBRARY_PATH` environment variable
set:

```bash
$ unset LD_LIBRARY_PATH
$ python cuda_check.py
Traceback (most recent call last):
  File "/mnt/c/Users/kayva/cuda_check.py", line 140, in <module>
    sys.exit(main())
  File "/mnt/c/Users/kayva/cuda_check.py", line 69, in main
    raise OSError("could not load any of: " + ' '.join(libnames))
OSError: could not load any of: libcuda.so libcuda.dylib cuda.dll
```

And then with `LD_LIBRARY_PATH` environment variable set:

```bash
$ export LD_LIBRARY_PATH=/usr/lib/wsl/lib
$ python cuda_check.py
Found 1 device(s).
Device: 0
  Name: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9
  Multiprocessors: 128
  CUDA Cores: unknown
  Concurrent threads: 196608
  GPU clock: 2535 MHz
  Memory clock: 10501 MHz
  Total Memory: 24563 MiB
  Free Memory: 23008 MiB
```
