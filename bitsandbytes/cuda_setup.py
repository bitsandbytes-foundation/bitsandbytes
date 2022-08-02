"""
extract factors the build is dependent on:
[X] compute capability  
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

alle Binaries packagen

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes
import os
from pathlib import Path
from typing import Set, Union

from .utils import print_err, warn_of_missing_prerequisite, execute_and_return


def check_cuda_result(cuda, result_val):
    # 3. Check for CUDA errors
    if result_val != 0:
        error_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result_val, ctypes.byref(error_str))
        raise Exception(f"CUDA exception! ERROR: {error_str}")


# taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def get_compute_capability():
    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    #    init_device -> init variables -> call function by reference
    # 2. call extern C function to determine CC 
    #    (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
    # 3. Check for CUDA errors
    #    https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    libnames = ("libcuda.so",)
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(libnames))

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()

    check_cuda_result(cuda, cuda.cuInit(0))

    check_cuda_result(cuda, cuda.cuDeviceGetCount(ctypes.byref(nGpus)))
    ccs = []
    for i in range(nGpus.value):
        check_cuda_result(cuda, cuda.cuDeviceGet(ctypes.byref(device), i))
        ref_major = ctypes.byref(cc_major)
        ref_minor = ctypes.byref(cc_minor)
        # 2. call extern C function to determine CC 
        check_cuda_result(cuda, cuda.cuDeviceComputeCapability(ref_major, ref_minor, device))
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    # TODO: handle different compute capabilities; for now, take the max
    ccs.sort()
    max_cc = ccs[-1]
    return max_cc


CUDA_RUNTIME_LIB: str = "libcudart.so"


def tokenize_paths(paths: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths.split(":") if ld_path}


def resolve_env_variable(env_var):
    '''Searches a given envirionmental library or path for the CUDA runtime library (libcudart.so)'''
    paths: Set[Path] = tokenize_paths(env_var)

    non_existent_directories: Set[Path] = {
        path for path in paths if not path.exists()
    }

    if non_existent_directories:
        print_err(
            "WARNING: The following directories listed your path were found to "
            f"be non-existent: {non_existent_directories}"
        )

    cuda_runtime_libs: Set[Path] = {
        path / CUDA_RUNTIME_LIB
        for path in paths
        if (path / CUDA_RUNTIME_LIB).is_file()
    } - non_existent_directories

    if len(cuda_runtime_libs) > 1:
        err_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        )
        raise FileNotFoundError(err_msg)
    elif len(cuda_runtime_libs) == 0: return None # this is not en error, since other envs can contain CUDA
    else: return next(iter(cuda_runtime_libs)) # for now just return the first

def get_cuda_runtime_lib_path() -> Union[Path, None]:
    '''Searches conda installation and environmental paths for a cuda installations.'''

    cuda_runtime_libs = []
    # CONDA_PREFIX/lib is the default location for a default conda
    # install of pytorch. This location takes priortiy over all
    # other defined variables
    if 'CONDA_PREFIX' in os.environ:
        lib_conda_path = f'{os.environ["CONDA_PREFIX"]}/lib/'
        print(lib_conda_path)
        cuda_runtime_libs.append(resolve_env_variable(lib_conda_path))

    if len(cuda_runtime_libs) == 1: return cuda_runtime_libs[0]

    # if CONDA_PREFIX does not have the library, search the environment
    # (in particualr LD_LIBRARY PATH)
    for var in os.environ:
         cuda_runtime_libs.append(resolve_env_variable(var))

    if len(cuda_runtime_libs) < 1:
        err_msg = (
            f"Did not find {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        )
        raise FileNotFoundError(err_msg)

    return cuda_runtime_libs.pop()


def evaluate_cuda_setup():
    cuda_path = get_cuda_runtime_lib_path()
    print(f'CUDA SETUP: CUDA path found: {cuda_path}')
    cc = get_compute_capability()
    binary_name = "libbitsandbytes_cpu.so"

    if not (has_gpu := bool(cc)):
        print(
            "WARNING: No GPU detected! Check your CUDA paths. Processing to load CPU-only library..."
        )
        return binary_name

    has_cublaslt = cc in ["7.5", "8.0", "8.6"]

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    cuda_home = str(Path(cuda_path).parent.parent)
    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler
    ls_output, err = execute_and_return(f"ls -l {cuda_path}")
    major, minor, revision = ls_output.split(' ')[-1].replace('libcudart.so.', '').split('.')
    cuda_version_string = f"{major}{minor}"

    binary_name = f'libbitsandbytes_cuda{cuda_version_string}{("" if has_cublaslt else "_nocublaslt")}.so'

    return binary_name
