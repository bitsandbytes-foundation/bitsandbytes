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
from os import environ as env
from pathlib import Path
from typing import Set, Union

from .utils import print_err, warn_of_missing_prerequisite


def check_cuda_result(cuda, result_val):
    if result_val != 0:
        # TODO: undefined name 'error_str'
        cuda.cuGetErrorString(result_val, ctypes.byref(error_str))
        print("Count not initialize CUDA - failure!")
        raise Exception("CUDA exception!")
    return result_val


# taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def get_compute_capability():
    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
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
    # TODO: local variable 'context' is assigned to but never used
    context = ctypes.c_void_p()
    # TODO: local variable 'error_str' is assigned to but never used
    error_str = ctypes.c_char_p()

    result = check_cuda_result(cuda, cuda.cuInit(0))

    result = check_cuda_result(cuda, cuda.cuDeviceGetCount(ctypes.byref(nGpus)))
    ccs = []
    for i in range(nGpus.value):
        result = check_cuda_result(
            cuda, cuda.cuDeviceGet(ctypes.byref(device), i)
        )
        result = check_cuda_result(
            cuda,
            cuda.cuDeviceComputeCapability(
                ctypes.byref(cc_major), ctypes.byref(cc_minor), device
            ),
        )
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    # TODO: handle different compute capabilities; for now, take the max
    ccs.sort()
    # return ccs[-1]
    return ccs


CUDA_RUNTIME_LIB: str = "libcudart.so"


def tokenize_paths(paths: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths.split(":") if ld_path}


def get_cuda_runtime_lib_path(
    # TODO: replace this with logic for all paths in env vars
    LD_LIBRARY_PATH: Union[str, None] = env.get("LD_LIBRARY_PATH")
) -> Union[Path, None]:
    """# TODO: add doc-string"""

    if not LD_LIBRARY_PATH:
        warn_of_missing_prerequisite(
            "LD_LIBRARY_PATH is completely missing from environment!"
        )
        return None

    ld_library_paths: Set[Path] = tokenize_paths(LD_LIBRARY_PATH)

    non_existent_directories: Set[Path] = {
        path for path in ld_library_paths if not path.exists()
    }

    if non_existent_directories:
        print_err(
            "WARNING: The following directories listed your path were found to "
            f"be non-existent: {non_existent_directories}"
        )

    cuda_runtime_libs: Set[Path] = {
        path / CUDA_RUNTIME_LIB
        for path in ld_library_paths
        if (path / CUDA_RUNTIME_LIB).is_file()
    } - non_existent_directories

    if len(cuda_runtime_libs) > 1:
        err_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        )
        raise FileNotFoundError(err_msg)

    elif len(cuda_runtime_libs) < 1:
        err_msg = (
            f"Did not find {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        )
        raise FileNotFoundError(err_msg)

    single_cuda_runtime_lib_dir = next(iter(cuda_runtime_libs))
    return single_cuda_runtime_lib_dir


def evaluate_cuda_setup():
    cuda_path = get_cuda_runtime_lib_path()
    cc = get_compute_capability()
    binary_name = "libbitsandbytes_cpu.so"

    if not (has_gpu := bool(cc)):
        print(
            "WARNING: No GPU detected! Check our CUDA paths. Processing to load CPU-only library..."
        )
        return binary_name

    has_cublaslt = cc in ["7.5", "8.0", "8.6"]

    # TODO:
    # (1) Model missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    cuda_home = str(Path(cuda_path).parent.parent)
    ls_output, err = execute_and_return(f"{cuda_home}/bin/nvcc --version")
    cuda_version = (
        ls_output.split("\n")[3].split(",")[-1].strip().lower().replace("v", "")
    )
    major, minor, revision = cuda_version.split(".")
    cuda_version_string = f"{major}{minor}"

    binary_name = f'libbitsandbytes_cuda{cuda_version_string}_{("cublaslt" if has_cublaslt else "")}.so'

    return binary_name
