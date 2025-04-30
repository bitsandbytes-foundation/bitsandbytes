import ctypes as ct
import logging
import os
from pathlib import Path
import re
from typing import Optional

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs, get_cuda_version_tuple

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


def get_available_cuda_binary_versions() -> list[str]:
    """Get formatted CUDA versions from existing library files using cuda_specs logic"""
    lib_pattern = f"libbitsandbytes_cuda*{DYNAMIC_LIBRARY_SUFFIX}"
    versions = []
    for lib in Path(__file__).parent.glob(lib_pattern):
        match = re.search(r"cuda(\d{3})", lib.name)
        if match:
            ver_code = int(match.group(1))
            major = ver_code // 10
            minor = ver_code % 10
            versions.append(f"{major}.{minor}")
    return sorted(versions)


def parse_cuda_version(version_str: str) -> str:
    """Convert raw version string (e.g. '118' from env var) to formatted version (e.g. '11.8')"""
    if version_str.isdigit() and len(version_str) == 3:
        return f"{version_str[:2]}.{version_str[2]}"
    return version_str  # fallback as safety net


def _format_cuda_error_message(
    available_versions: list[str],
    user_cuda_version: str,
    original_error: str = "",
    requested_version: Optional[str] = None,
) -> str:
    base_msg = "Attempted to use bitsandbytes native library functionality but it's not available.\n\n"

    version_alert = ""
    if requested_version not in available_versions:
        version_list_str = "\n  - " + "\n  - ".join(available_versions) if available_versions else "NONE"
        version_alert = (
            f"🚨 CUDA VERSION MISMATCH 🚨\n"
            f"Requested CUDA version:          {requested_version}\n"
            f"Detected PyTorch CUDA version:   {user_cuda_version}\n"
            f"Available pre-compiled versions: {version_list_str}\n\n"
            "This means:\n"
            "1. The version you're trying to use is NOT distributed with this package\n"
            if available_versions
            else "1. You're not using the package but checked-out the source code\n"
            "2. You MUST compile from source for this specific CUDA version\n"
            "3. The installation will NOT work until you compile or choose a CUDA supported version\n\n"
        )

    troubleshooting = (
        "This typically happens when:\n"
        "1. bitsandbytes doesn't ship with a pre-compiled binary for your CUDA version\n"
        "2. The library wasn't compiled properly during installation from source\n"
        "3. Missing CUDA dependencies\n\n"
    )

    note = (
        "To make bitsandbytes work, the compiled library version MUST exactly match the linked CUDA version.\n"
        "If your CUDA version doesn't have a pre-compiled binary, you MUST compile from source.\n\n"
    )

    compile_instructions = (
        "You have three options:\n"
        "1. COMPILE FROM SOURCE (required if no binary exists):\n"
        "   https://huggingface.co/docs/bitsandbytes/main/en/installation#cuda-compile\n"
        "2. Use BNB_CUDA_VERSION to specify a DIFFERENT CUDA version from the detected one, which is installed on your machine and matching an available pre-compiled version listed above\n"
        "3. Check LD_LIBRARY_PATH contains the correct CUDA libraries\n\n"
    )

    diagnostics = (
        "🔍 Run this command for detailed diagnostics:\n"
        "python -m bitsandbytes\n\n"
        "If you've tried everything and still have issues:\n"
        "1. Include ALL version info (operating system, bitsandbytes, pytorch, cuda, python)\n"
        "2. Describe what you've tried in detail\n"
        "3. Open an issue with this information:\n"
        "   https://github.com/bitsandbytes-foundation/bitsandbytes/issues\n\n"
    )

    return f"{version_alert}{base_msg}{troubleshooting}{note}{compile_instructions}{original_error}\n{diagnostics}"


class MockBNBNativeLibrary(BNBNativeLibrary):
    """
    Mock BNBNativeLibrary that raises an error when trying to use native library
    functionality without successfully loading the library.
    Any method or attribute access will raise a RuntimeError with a message that
    points to the original error and provides troubleshooting steps.
    """

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        self.user_cuda_version = get_cuda_version_tuple()

    def __getattr__(self, name):
        available_versions = get_available_cuda_binary_versions()
        override_value = os.environ.get("BNB_CUDA_VERSION")

        requested_version = (
            parse_cuda_version(override_value)
            if override_value
            else f"{self.user_cuda_version[0]}.{self.user_cuda_version[1]}"
        )

        msg = _format_cuda_error_message(
            available_versions=available_versions,
            user_cuda_version=f"{self.user_cuda_version[0]}.{self.user_cuda_version[1]}",
            original_error=f"Original error: {self.error_msg}\n" if self.error_msg else "",
            requested_version=requested_version,
        )
        raise RuntimeError(msg)

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
            available_versions = get_available_cuda_binary_versions()
            env_version = os.environ.get("BNB_CUDA_VERSION")

            requested_version = parse_cuda_version(env_version) if env_version else cuda_specs.cuda_version_string

            msg = _format_cuda_error_message(
                available_versions=available_versions,
                user_cuda_version=cuda_specs.cuda_version_string,
                requested_version=requested_version,
            )
            logger.warning(msg)

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
    logger.error(error_msg, exc_info=False)

    # create a mock with error messaging as fallback
    lib = MockBNBNativeLibrary(error_msg)
