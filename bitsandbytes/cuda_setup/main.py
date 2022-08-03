"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes
from pathlib import Path

from ..utils import execute_and_return
from .paths import determine_cuda_runtime_lib_path


def check_cuda_result(cuda, result_val):
    # 3. Check for CUDA errors
    if result_val != 0:
        error_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result_val, ctypes.byref(error_str))
        raise Exception(f"CUDA exception! ERROR: {error_str}")


def get_compute_capabilities():
    """
    1. find libcuda.so library (GPU driver) (/usr/lib)
       init_device -> init variables -> call function by reference
    2. call extern C function to determine CC
       (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
    3. Check for CUDA errors
       https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    # bits taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """

    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    try:
        cuda = ctypes.CDLL("libcuda.so")
    except OSError:
        # TODO: shouldn't we error or at least warn here?
        return None

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
        check_cuda_result(
            cuda, cuda.cuDeviceComputeCapability(ref_major, ref_minor, device)
        )
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    return ccs.sort()


# def get_compute_capability()-> Union[List[str, ...], None]: # FIXME: error
def get_compute_capability():
    """
    Extracts the highest compute capbility from all available GPUs, as compute
    capabilities are downwards compatible. If no GPUs are detected, it returns
    None.
    """
    if ccs := get_compute_capabilities() is not None:
        # TODO: handle different compute capabilities; for now, take the max
        return ccs[-1]
    return None


def evaluate_cuda_setup():
    cuda_path = determine_cuda_runtime_lib_path()
    print(f"CUDA SETUP: CUDA path found: {cuda_path}")
    cc = get_compute_capability()
    binary_name = "libbitsandbytes_cpu.so"

    # FIXME: has_gpu is still unused
    if not (has_gpu := bool(cc)):
        print(
            "WARNING: No GPU detected! Check your CUDA paths. Processing to load CPU-only library..."
        )
        return binary_name

    # 7.5 is the minimum CC vor cublaslt
    has_cublaslt = cc in ["7.5", "8.0", "8.6"]

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    # FIXME: cuda_home is still unused
    cuda_home = str(Path(cuda_path).parent.parent)
    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler
    ls_output, err = execute_and_return(f"ls -l {cuda_path}")
    major, minor, revision = (
        ls_output.split(" ")[-1].replace("libcudart.so.", "").split(".")
    )
    cuda_version_string = f"{major}{minor}"

    def get_binary_name():
        "if not has_cublaslt (CC < 7.5), then we have to choose  _nocublaslt.so"
        bin_base_name = "libbitsandbytes_cuda"
        if has_cublaslt:
            return f"{bin_base_name}{cuda_version_string}.so"
        else:
            return f"{bin_base_name}_nocublaslt.so"

    return binary_name
