import ctypes as ct
import logging
import os
from pathlib import Path
import re

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs

logger = logging.getLogger(__name__)


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA BNB native library specified by the
    given CUDA specs, taking into account the `BNB_CUDA_VERSION` override environment variable.

    The library is not guaranteed to exist at the returned path.
    """

    prefix = "rocm" if torch.version.hip else "cuda"
    library_name = f"libbitsandbytes_{prefix}{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"

    override_value = os.environ.get("BNB_CUDA_VERSION")
    if override_value:
        library_name = re.sub(r"cuda\d+", f"cuda{override_value}", library_name, count=1)
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

    def __getitem__(self, item):
        return getattr(self._lib, item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


class MockBNBNativeLibrary(BNBNativeLibrary):
    """
    Mock BNBNativeLibrary that raises an error when trying to use native library functionality without successfully loading the library.

    Any method or attribute access will raise a RuntimeError with a message that points to the original error and provides troubleshooting steps.
    """

    def __init__(self, error_msg: str):
        self.error_msg = error_msg

    def __getattr__(self, name):
        base_msg = "Attempted to use bitsandbytes native library functionality but it's not available.\n\n"
        original_error = f"Original error: {self.error_msg}\n\n" if self.error_msg else ""
        troubleshooting = (
            "This typically happens when:\n"
            "1. BNB doesn't ship with a pre-compiled binary for your CUDA version\n"
            "2. The library wasn't compiled properly during installation\n"
            "3. Missing CUDA dependencies\n"
            "4. PyTorch/bitsandbytes version mismatch\n\n"
            "Run 'python -m bitsandbytes' for diagnostics."
        )
        raise RuntimeError(base_msg + original_error + troubleshooting)

    def __getitem__(self, name):
        return self.__getattr__(name)


def get_native_library() -> BNBNativeLibrary:
    """
    Load CUDA library XOR CPU, as the latter contains a subset of symbols of the former.
    """
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
    cuda_specs = get_cuda_specs()
    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)
        if cuda_binary_path.exists():
            binary_path = cuda_binary_path
        else:
            logger.warning("Could not find the bitsandbytes CUDA binary at %r", cuda_binary_path)
    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return CudaBNBNativeLibrary(dll)

    logger.warning(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable.",
    )
    return BNBNativeLibrary(dll)


try:
    lib = get_native_library()
except Exception as e:
    error_msg = f"Could not load bitsandbytes native library: {e}"
    logger.error(error_msg, exc_info=True)

    diagnostic_help = ""
    if torch.cuda.is_available():
        diagnostic_help = (
            "CUDA Setup failed despite CUDA being available. "
            "Please run the following command to get more information:\n\n"
            "python -m bitsandbytes\n\n"
            "Inspect the output of the command and see if you can locate CUDA libraries. "
            "You might need to add them to your LD_LIBRARY_PATH. "
            "If you suspect a bug, please take the information from the command and open an issue at:\n\n"
            "https://github.com/bitsandbytes-foundation/bitsandbytes/issues\n\n"
            "If you are using a custom CUDA version, you might need to set the BNB_CUDA_VERSION "
            "environment variable to the correct version."
        )

    # create a mock with error messaging as fallback
    lib = MockBNBNativeLibrary(diagnostic_help)
