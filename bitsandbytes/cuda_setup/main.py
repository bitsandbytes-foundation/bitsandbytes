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
import torch

from .paths import determine_cuda_runtime_lib_path
from bitsandbytes.cextension import CUDASetup


def check_cuda_result(cuda, result_val):
    # 3. Check for CUDA errors
    if result_val != 0:
        error_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result_val, ctypes.byref(error_str))
        CUDASetup.get_instance().add_log_entry(f"CUDA exception! Error code: {error_str.value.decode()}")


# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
def get_cuda_version(cuda, cudart_path):
    if cuda is None: return None

    try:
        cudart = ctypes.CDLL(cudart_path)
    except OSError:
        CUDASetup.get_instance().add_log_entry(f'ERROR: libcudart.so could not be read from path: {cudart_path}!')
        return None

    version = ctypes.c_int()
    check_cuda_result(cuda, cudart.cudaRuntimeGetVersion(ctypes.byref(version)))
    version = int(version.value)
    major = version//1000
    minor = (version-(major*1000))//10

    if major < 11:
       CUDASetup.get_instance().add_log_entry('CUDA SETUP: CUDA version lower than 11 are currenlty not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')

    return f'{major}{minor}'


def get_cuda_lib_handle():
    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    try:
        cuda = ctypes.CDLL("libcuda.so")
    except OSError:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: WARNING! libcuda.so not found! Do you have a CUDA driver installed? If you are on a cluster, make sure you are on a CUDA machine!')
        return None
    check_cuda_result(cuda, cuda.cuInit(0))

    return cuda


def get_compute_capabilities(cuda):
    """
    1. find libcuda.so library (GPU driver) (/usr/lib)
       init_device -> init variables -> call function by reference
    2. call extern C function to determine CC
       (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
    3. Check for CUDA errors
       https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    # bits taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    device = ctypes.c_int()

    check_cuda_result(cuda, cuda.cuDeviceGetCount(ctypes.byref(nGpus)))
    ccs = []
    for i in range(nGpus.value):
        check_cuda_result(cuda, cuda.cuDeviceGet(ctypes.byref(device), i))
        ref_major = ctypes.byref(cc_major)
        ref_minor = ctypes.byref(cc_minor)
        # 2. call extern C function to determine CC
        check_cuda_result(cuda, cuda.cuDeviceComputeCapability(ref_major, ref_minor, device))
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    return ccs


# def get_compute_capability()-> Union[List[str, ...], None]: # FIXME: error
def get_compute_capability(cuda):
    """
    Extracts the highest compute capbility from all available GPUs, as compute
    capabilities are downwards compatible. If no GPUs are detected, it returns
    None.
    """
    if cuda is None: return None

    # TODO: handle different compute capabilities; for now, take the max
    ccs = get_compute_capabilities(cuda)
    if ccs: return ccs[-1]


def evaluate_cuda_setup():
    # we remove this for now and see how things go
    #print('')
    #print('='*35 + 'BUG REPORT' + '='*35)
    #print('Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues')
    #print('For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link')
    #print('='*80)
    if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None

    cuda_setup = CUDASetup.get_instance()
    cudart_path = determine_cuda_runtime_lib_path()
    cuda = get_cuda_lib_handle()
    cc = get_compute_capability(cuda)
    cuda_version_string = get_cuda_version(cuda, cudart_path)

    failure = False
    if cudart_path is None:
        failure = True
        cuda_setup.add_log_entry("WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!", is_warning=True)
    else:
        cuda_setup.add_log_entry((f"CUDA SETUP: CUDA runtime path found: {cudart_path}"))

    if cc == '' or cc is None:
        failure = True
        cuda_setup.add_log_entry("WARNING: No GPU detected! Check your CUDA paths. Proceeding to load CPU-only library...", is_warning=True)
    else:
        cuda_setup.add_log_entry(f"CUDA SETUP: Highest compute capability among GPUs detected: {cc}")

    if cuda is None:
        failure = True
    else:
        cuda_setup.add_log_entry(f'CUDA SETUP: Detected CUDA version {cuda_version_string}')

    # 7.5 is the minimum CC vor cublaslt
    has_cublaslt = cc in ["7.5", "8.0", "8.6"]

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler

    if failure:
        binary_name = "libbitsandbytes_cpu.so"
    elif has_cublaslt:
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}.so"
    else:
        "if not has_cublaslt (CC < 7.5), then we have to choose  _nocublaslt.so"
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}_nocublaslt.so"

    return binary_name, cudart_path, cuda, cc, cuda_version_string
