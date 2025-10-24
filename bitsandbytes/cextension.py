import ctypes as ct
import functools
import logging
import os
from pathlib import Path
import re
from typing import Optional

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import (
    CUDASpecs,
    get_cuda_specs,
    get_cuda_version_tuple,
    get_rocm_gpu_arch,
    get_rocm_warpsize,
)

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
        if torch.version.hip:
            raise RuntimeError(
                f"BNB_CUDA_VERSION={override_value} detected for ROCm!! \n"
                f"Clear the variable and retry: export BNB_CUDA_VERSION=\n"
            )
        logger.warning(
            f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
            "This can be used to load a bitsandbytes version built with a CUDA version that is different from the PyTorch CUDA version.\n"
            "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
        )

    return PACKAGE_DIR / library_name


class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    @functools.cache  # noqa: B019
    def __getattr__(self, name):
        fn = getattr(self._lib, name, None)

        if fn is not None:
            return fn

        def throw_on_call(*args, **kwargs):
            raise RuntimeError(
                f"Method '{name}' not available in CPU-only version of bitsandbytes.\n"
                "Reinstall with GPU support or use CUDA-enabled hardware."
            )

        return throw_on_call

    def __getitem__(self, item):
        return self.__getattr__(item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


def get_available_cuda_binary_versions() -> list[str]:
    """Get formatted CUDA versions from existing library files using cuda_specs logic"""
    lib_pattern = f"libbitsandbytes_{BNB_BACKEND.lower()}*{DYNAMIC_LIBRARY_SUFFIX}"
    versions = []
    for lib in Path(__file__).parent.glob(lib_pattern):
        pattern = rf"{BNB_BACKEND.lower()}(\d+)"
        match = re.search(pattern, lib.name)
        if match:
            ver_code = int(match.group(1))
            major = ver_code // 10
            minor = ver_code % 10
            versions.append(f"{major}.{minor}")
    return sorted(versions)


def parse_cuda_version(version_str: str) -> str:
    """Convert raw version string (e.g. '118' from env var) to formatted version (e.g. '11.8')"""
    if version_str.isdigit():
        return f"{version_str[:-1]}.{version_str[-1]}"
    return version_str  # fallback as safety net


class ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary):
    """
    Mock library handler that defers errors until native methods are called.

    This class serves as a fallback when the native bitsandbytes library fails to load.
    It captures the original error and generates detailed troubleshooting guidance.

    Key behaviors:
    - Allows attribute access and method assignment without immediate errors
    - Throws a RuntimeError with diagnostic information only when a native method is called, as otherwise it would error out on import, breaking backward compatibility
    - Handles both missing CUDA dependencies and version mismatch scenarios

    Error scenarios covered:
    1. Missing shared library dependencies (e.g., libcudart.so not in LD_LIBRARY_PATH or through PyTorch CUDA installation)
    2. CUDA version mismatch between PyTorch and available pre-compiled binaries
    3. Completely missing pre-compiled binaries when CUDA is detected
    4. Custom BNB_CUDA_VERSION override but mismatch
    5. CPU-only installation attempts when GPU functionality is requested

    """

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        self.user_cuda_version = get_cuda_version_tuple()
        self.available_versions = get_available_cuda_binary_versions()
        self.override_value = os.environ.get("BNB_CUDA_VERSION")
        self.requested_version = (
            parse_cuda_version(self.override_value)
            if self.override_value
            else f"{self.user_cuda_version[0]}.{self.user_cuda_version[1]}"
            if self.user_cuda_version
            else "unknown"
        )

        # Pre-generate the error message based on error type
        if "cannot open shared object file" in error_msg:
            self.formatted_error = self._format_dependency_error()
        else:  # lib loading errors
            self.formatted_error = self._format_lib_error_message(
                available_versions=self.available_versions,
                user_cuda_version=f"{self.user_cuda_version[0]}.{self.user_cuda_version[1]}"
                if self.user_cuda_version
                else "unknown",
                original_error=f"Original error: {self.error_msg}\n" if self.error_msg else "",
                requested_version=self.requested_version,
            )

    def _format_lib_error_message(
        self,
        available_versions: list[str],
        user_cuda_version: str,
        original_error: str = "",
        requested_version: Optional[str] = None,
    ) -> str:
        """Format detailed error message for library loading failures"""
        analysis = ""
        no_cpu_lib_found = "libbitsandbytes_cpu.so: cannot open" in original_error
        no_cuda_lib_found = f"{BNB_BACKEND} binary not found" in original_error

        if no_cpu_lib_found:
            analysis = "\n🚨 Failed to load CPU-only bitsandbytes library 🚨\n\n"

        elif no_cuda_lib_found:
            version_list_str = "\n  - " + "\n  - ".join(available_versions) if available_versions else "NONE"
            analysis = (
                (
                    f"\n🚨 {BNB_BACKEND} VERSION MISMATCH 🚨\n"
                    f"Requested {BNB_BACKEND} version:          {requested_version}\n"
                    f"Detected PyTorch {BNB_BACKEND} version:   {user_cuda_version}\n"
                    f"Available pre-compiled versions: {version_list_str}\n\n"
                    "This means:\n"
                    "The version you're trying to use is NOT distributed with this package\n\n"
                )
                if available_versions
                else "\n🚨 Forgot to compile the bitsandbytes library? 🚨\n"
                "1. You're not using the package but checked-out the source code\n"
                "2. You MUST compile from source\n\n"
            )

        base_msg = "Attempted to use bitsandbytes native library functionality but it's not available.\n\n"

        troubleshooting = (
            (
                f"This typically happens when:\n"
                f"1. bitsandbytes doesn't ship with a pre-compiled binary for your {BNB_BACKEND} version\n"
                f"2. The library wasn't compiled properly during installation from source\n\n"
            )
            if no_cuda_lib_found
            else f"This typically happens when you checked the code out from source and your torch installation doesn't detect {BNB_BACKEND} on your machine.\n\n"
        )

        note = (
            (
                f"To make bitsandbytes work, the compiled library version MUST exactly match the linked {BNB_BACKEND} version.\n"
                f"If your {BNB_BACKEND} version doesn't have a pre-compiled binary, you MUST compile from source.\n\n"
            )
            if no_cuda_lib_found
            else ""
        )

        compile_instructions = (
            ("COMPILE FROM SOURCE for CPU-only:\n  `cmake -DCOMPUTE_BACKEND=cpu -S . && make`\n\n")
            if not no_cuda_lib_found
            else (
                "You have two options:\n"
                "1. COMPILE FROM SOURCE (required if no binary exists):\n"
                "   https://huggingface.co/docs/bitsandbytes/main/en/installation#cuda-compile\n"
                "2. Use BNB_CUDA_VERSION to specify a DIFFERENT CUDA version from the detected one, which is installed on your machine and matching an available pre-compiled version listed above\n\n"
            )
            if not HIP_ENVIRONMENT
            else (
                "You can COMPILE FROM SOURCE as mentioned here:\n"
                "   https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=AMD+ROCm#amd-gpu\n"
            )
        )

        diagnostics = (
            f"🔍 Run this command for detailed diagnostics:\n"
            f"python -m bitsandbytes\n\n"
            f"If you've tried everything and still have issues:\n"
            f"1. Include ALL version info (operating system, bitsandbytes, pytorch, {BNB_BACKEND.lower()}, python)\n"
            f"2. Describe what you've tried in detail\n"
            f"3. Open an issue with this information:\n"
            f"   https://github.com/bitsandbytes-foundation/bitsandbytes/issues\n\n"
        )

        return f"{analysis}{base_msg}{troubleshooting}{note}{compile_instructions}{original_error}\n{diagnostics}"

    def _format_dependency_error(self) -> str:
        """Format error message for missing shared libraries"""
        # Extract missing library name from error
        error_parts = self.error_msg.split(":")
        missing_lib = error_parts[0].strip() if len(error_parts) > 0 else "unknown library"
        cuda_major_version = (
            self.requested_version.split(".")[0] if "." in self.requested_version else self.requested_version
        )

        return (
            f"\n🚨 {BNB_BACKEND} SETUP ERROR: Missing dependency: {missing_lib} 🚨\n\n"
            f"{BNB_BACKEND} {cuda_major_version}.x runtime libraries were not found in the LD_LIBRARY_PATH.\n\n"
            f"To fix this, make sure that:\n"
            f"1. You have installed {BNB_BACKEND} {cuda_major_version}.x toolkit on your system\n"
            f"2. The {BNB_BACKEND} runtime libraries are in your LD_LIBRARY_PATH\n\n"
            f"You can add them with (and persist the change by adding the line to your .bashrc):\n"
            f"   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/{BNB_BACKEND.lower()}-{cuda_major_version}.x/\
                    {'lib64' if not HIP_ENVIRONMENT else 'lib'}\n\n"
            f"Original error: {self.error_msg}\n\n"
            f"🔍 Run this command for detailed diagnostics:\n"
            f"python -m bitsandbytes\n\n"
            f"If you've tried everything and still have issues:\n"
            f"1. Include ALL version info (operating system, bitsandbytes, pytorch, {BNB_BACKEND.lower()}, python)\n"
            f"2. Describe what you've tried in detail\n"
            f"3. Open an issue with this information:\n"
            f"   https://github.com/bitsandbytes-foundation/bitsandbytes/issues\n\n"
        )

    def __getattr__(self, name):
        """Return a dummy function that throws when called, rather than on attribute access"""

        def throw_on_call(*args, **kwargs):
            raise RuntimeError(f"{self.formatted_error}Native code method attempted to call: lib.{name}()")

        return throw_on_call

    def __getitem__(self, name):
        return self.__getattr__(name)


def get_native_library() -> BNBNativeLibrary:
    """
    Load CUDA library XOR CPU, as the latter contains a subset of symbols of the former.
    """
    cuda_specs = get_cuda_specs()
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"

    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)

        if not cuda_binary_path.exists():
            raise RuntimeError(f"Configured {BNB_BACKEND} binary not found at {cuda_binary_path}")

        binary_path = cuda_binary_path

    if torch._C._has_xpu:
        binary_path = PACKAGE_DIR / f"libbitsandbytes_xpu{DYNAMIC_LIBRARY_SUFFIX}"

    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")

    # Try to load the library - any errors will propagate up
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return CudaBNBNativeLibrary(dll)

    return BNBNativeLibrary(dll)


ROCM_GPU_ARCH = get_rocm_gpu_arch()
ROCM_WARP_SIZE_64 = True if get_rocm_warpsize() == 64 else False

HIP_ENVIRONMENT = False
BNB_BACKEND = "CPU"
if torch.version.hip:
    HIP_ENVIRONMENT = True
    BNB_BACKEND = "ROCm"
elif torch.cuda.is_available():
    BNB_BACKEND = "CUDA"
elif torch._C._has_xpu:
    BNB_BACKEND = "XPU"

try:
    lib = get_native_library()
except Exception as e:
    if BNB_BACKEND in ("CPU", "XPU"):
        lib = ErrorHandlerMockBNBNativeLibrary("XPU/CPU can run without native library.")
    else:
        error_msg = str(e)
        logger.error(
            f"bitsandbytes library load error: {error_msg}",
            exc_info=True,
        )

        # create a mock with error messaging as fallback
        lib = ErrorHandlerMockBNBNativeLibrary(error_msg)
