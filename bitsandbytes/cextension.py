"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multiple)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes as ct
import logging
import os
from pathlib import Path

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs, get_rocm_gpu_arch
from bitsandbytes.npu_specs import get_npu_specs

logger = logging.getLogger(__name__)


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA BNB native library specified by the
    given CUDA specs, taking into account the `BNB_CUDA_VERSION` override environment variable.

    The library is not guaranteed to exist at the returned path.
    """
    if torch.version.hip:
        if BNB_HIP_VERSION < 601:
            return PACKAGE_DIR / f"libbitsandbytes_rocm{BNB_HIP_VERSION_SHORT}_nohipblaslt{DYNAMIC_LIBRARY_SUFFIX}"
        else:
            return PACKAGE_DIR / f"libbitsandbytes_rocm{BNB_HIP_VERSION_SHORT}{DYNAMIC_LIBRARY_SUFFIX}"
    library_name = f"libbitsandbytes_cuda{cuda_specs.cuda_version_string}"
    if not cuda_specs.has_cublaslt:
        # if not has_cublaslt (CC < 7.5), then we have to choose _nocublaslt
        library_name += "_nocublaslt"
    library_name = f"{library_name}{DYNAMIC_LIBRARY_SUFFIX}"

    override_value = os.environ.get("BNB_CUDA_VERSION")
    if override_value:
        library_name_stem, _, library_name_ext = library_name.rpartition(".")
        # `library_name_stem` will now be e.g. `libbitsandbytes_cuda118`;
        # let's remove any trailing numbers:
        library_name_stem = library_name_stem.rstrip("0123456789")
        # `library_name_stem` will now be e.g. `libbitsandbytes_cuda`;
        # let's tack the new version number and the original extension back on.
        library_name = f"{library_name_stem}{override_value}.{library_name_ext}"
        logger.warning(
            f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
            "This can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n"
            "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
            "If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n"
            "For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n",
        )

    return PACKAGE_DIR / library_name


class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, item):
        return getattr(self._lib, item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        if torch.version.cuda:
            lib.get_cusparse.restype = ct.c_void_p
        elif torch.version.hip:
            lib.get_hipsparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


def get_native_library() -> BNBNativeLibrary:
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
    cuda_specs = get_cuda_specs()
    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)
        if cuda_binary_path.exists():
            binary_path = cuda_binary_path
        else:
            logger.warning("Could not find the bitsandbytes %s binary at %r", BNB_BACKEND, cuda_binary_path)
    npu_specs = get_npu_specs()
    if npu_specs:
        binary_path = PACKAGE_DIR / f"libbitsandbytes_npu{DYNAMIC_LIBRARY_SUFFIX}"

    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return CudaBNBNativeLibrary(dll)

    return BNBNativeLibrary(dll)


ROCM_GPU_ARCH = get_rocm_gpu_arch()

try:
    if torch.version.hip:
        hip_major, hip_minor = map(int, torch.version.hip.split(".")[0:2])
        HIP_ENVIRONMENT, BNB_HIP_VERSION = True, hip_major * 100 + hip_minor
        BNB_HIP_VERSION_SHORT = f"{hip_major}{hip_minor}"
        BNB_BACKEND = "ROCm"
    else:
        HIP_ENVIRONMENT, BNB_HIP_VERSION = False, 0
        BNB_HIP_VERSION_SHORT = ""
        BNB_BACKEND = "CUDA"

    lib = get_native_library()
except Exception as e:
    lib = None
    logger.error(f"Could not load bitsandbytes native library: {e}", exc_info=True)
    if torch.cuda.is_available():
        logger.warning(
            f"""
{BNB_BACKEND} Setup failed despite {BNB_BACKEND} being available. Please run the following command to get more information:

python -m bitsandbytes

Inspect the output of the command and see if you can locate {BNB_BACKEND} libraries. You might need to add them
to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
""",
        )
