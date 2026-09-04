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
)

logger = logging.getLogger(__name__)


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the path to the best matching CUDA/ROCm BNB native library for the given specs.

    When no override is set, selects from packaged libraries using the following priority:
    1. Exact version match.
    2. Highest packaged version <= runtime version, same major (e.g. runtime 12.9, ship 12.8).
    3. Lowest packaged version > runtime version, same major (e.g. runtime 12.0, ship 12.1).
    No cross-major fallback: if no same-major library exists, returns the exact non-existent
    path so the caller raises a clear "not found" error.
    A warning is logged when falling back. Override env vars bypass selection entirely
    and load the named version with no fallback. The returned path is not guaranteed to
    exist when no packaged libs are found, or when an override names an absent version.
    """
    is_hip = bool(torch.version.hip)
    prefix = "rocm" if is_hip else "cuda"
    override_var = "BNB_ROCM_VERSION" if is_hip else "BNB_CUDA_VERSION"

    override_value = os.environ.get(override_var)

    if override_value is not None:
        if not override_value.isdigit():
            raise RuntimeError(f"{override_var}={override_value!r}: value must be digits only (e.g. '124' for 12.4).")
        library_name = f"libbitsandbytes_{prefix}{override_value}{DYNAMIC_LIBRARY_SUFFIX}"
        logger.warning(
            f"WARNING: {override_var}={override_value} environment variable detected; loading {library_name}.\n"
            f"This overrides automatic {'ROCm' if is_hip else 'CUDA'} version selection.\n"
            f"If this was unintended clear the variable and retry: unset {override_var}\n",
        )
        return PACKAGE_DIR / library_name

    available = _find_cuda_libs(prefix, is_hip)
    runtime_version = cuda_specs.cuda_version_tuple

    if not available:
        return PACKAGE_DIR / f"libbitsandbytes_{prefix}{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"

    if runtime_version in available:
        return available[runtime_version]

    lower = [v for v in available if v[0] == runtime_version[0] and v < runtime_version]
    if lower:
        selected = max(lower)
    else:
        higher_same = [v for v in available if v[0] == runtime_version[0] and v > runtime_version]
        if higher_same:
            selected = min(higher_same)
        else:
            # No same-major library available. Return the non-existent exact path so
            # get_native_library() raises a clear "not found" error.
            return PACKAGE_DIR / f"libbitsandbytes_{prefix}{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"

    logger.warning(
        f"No prebuilt binary for {'ROCm' if is_hip else 'CUDA'} "
        f"{runtime_version[0]}.{runtime_version[1]}, loading "
        f"{'ROCm' if is_hip else 'CUDA'} {selected[0]}.{selected[1]} instead. "
        f"Set {override_var} to override."
    )
    return available[selected]


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
        lib.cget_managed_ptr.restype = ct.c_void_p


class XpuBNBNativeLibrary(BNBNativeLibrary):
    """XPU native library with SYCL USM paged memory support."""

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        if hasattr(lib, "cget_managed_ptr"):
            lib.cget_managed_ptr.restype = ct.c_void_p


class MpsBNBNativeLibrary(BNBNativeLibrary):
    """Apple Silicon MPS native library: hand-written Metal kernels + companion metallib.

    Loaded independently of the main `lib` (which, on macOS without CUDA/ROCm/XPU, is the
    CPU error-handler mock). Callers must pass torch MPS ``tensor.data_ptr()`` values --
    on torch MPS a tensor's data pointer IS its ``id<MTLBuffer>`` -- for offset-0
    contiguous tensors.
    """

    def __init__(self, lib: ct.CDLL, metallib_path: Path):
        super().__init__(lib)
        self.metallib_path = metallib_path

        lib.bnb_mps_check_buffer_contract.restype = ct.c_int
        lib.bnb_mps_check_buffer_contract.argtypes = [ct.c_void_p, ct.c_int64]

        lib.bnb_mps_quantize_blockwise.restype = None
        lib.bnb_mps_quantize_blockwise.argtypes = [
            ct.c_void_p,  # code (float32[256])
            ct.c_void_p,  # A (float32[n])
            ct.c_void_p,  # out (uint8[n])
            ct.c_void_p,  # absmax (float32[blocks])
            ct.c_int64,  # n
            ct.c_int64,  # blocksize
        ]

        lib.bnb_mps_dequantize_blockwise.restype = None
        lib.bnb_mps_dequantize_blockwise.argtypes = [
            ct.c_void_p,  # code (float32[256])
            ct.c_void_p,  # A (uint8[n])
            ct.c_void_p,  # absmax (float32[blocks])
            ct.c_void_p,  # out (float32[n])
            ct.c_int64,  # n
            ct.c_int64,  # blocksize
        ]

        lib.bnb_mps_dequantize_4bit.restype = None
        lib.bnb_mps_dequantize_4bit.argtypes = [
            ct.c_void_p,  # code (float32[16])
            ct.c_void_p,  # A (uint8 packed)
            ct.c_void_p,  # absmax (float32[blocks])
            ct.c_void_p,  # out (float32[n])
            ct.c_int64,  # n
            ct.c_int64,  # blocksize
        ]

        lib.bnb_mps_quantize_4bit.restype = None
        lib.bnb_mps_quantize_4bit.argtypes = [
            ct.c_void_p,  # bounds (float32[15])
            ct.c_void_p,  # order (uint8[16])
            ct.c_void_p,  # A (float32[n])
            ct.c_void_p,  # out (uint8 packed)
            ct.c_void_p,  # absmax (float32[blocks])
            ct.c_int64,  # n
            ct.c_int64,  # blocksize
        ]

        # Older native builds predate the fused matmul kernels; guard so a stale dylib
        # keeps the quant/dequant native path without breaking library load.
        if hasattr(lib, "bnb_mps_gemv_4bit"):
            lib.bnb_mps_gemv_4bit.restype = None
            lib.bnb_mps_gemv_4bit.argtypes = [
                ct.c_void_p,  # code (float32[16])
                ct.c_void_p,  # B (uint8 packed, N*K/2 bytes)
                ct.c_void_p,  # absmax (float32[blocks])
                ct.c_void_p,  # A (activation dtype [K])
                ct.c_void_p,  # out (activation dtype [N])
                ct.c_int64,  # K
                ct.c_int64,  # N
                ct.c_int64,  # bs_shift = log2(blocksize)
                ct.c_int64,  # dtype_flag (0=fp32, 1=fp16, 2=bf16)
            ]

        if hasattr(lib, "bnb_mps_gemm_4bit"):
            lib.bnb_mps_gemm_4bit.restype = None
            lib.bnb_mps_gemm_4bit.argtypes = [
                ct.c_void_p,  # code (float32[16])
                ct.c_void_p,  # B (uint8 packed, N*K/2 bytes)
                ct.c_void_p,  # absmax (float32[blocks])
                ct.c_void_p,  # A (activation dtype [M*K])
                ct.c_void_p,  # bias (activation dtype [N]; None when absent)
                ct.c_void_p,  # out (activation dtype [M*N])
                ct.c_int64,  # M
                ct.c_int64,  # K
                ct.c_int64,  # N
                ct.c_int64,  # bs_shift = log2(blocksize)
                ct.c_int64,  # dtype_flag (0=fp32, 1=fp16, 2=bf16)
            ]

        if hasattr(lib, "bnb_mps_gemm_4bit_bwd"):
            lib.bnb_mps_gemm_4bit_bwd.restype = None
            lib.bnb_mps_gemm_4bit_bwd.argtypes = [
                ct.c_void_p,  # code (float32[16])
                ct.c_void_p,  # B (uint8 packed, N*K/2 bytes)
                ct.c_void_p,  # absmax (float32[blocks])
                ct.c_void_p,  # G = grad_output (activation dtype [M*N])
                ct.c_void_p,  # out = grad_A (activation dtype [M*K])
                ct.c_int64,  # M
                ct.c_int64,  # K
                ct.c_int64,  # N
                ct.c_int64,  # bs_shift = log2(blocksize)
                ct.c_int64,  # dtype_flag (0=fp32, 1=fp16, 2=bf16)
            ]

        # Capability marker for the bf16 GEMM (MPSGraph path). Gated separately from
        # bnb_mps_gemm_4bit so a dylib built before bf16 support keeps the fallback rather
        # than being handed dtype_flag=2 and reading the scratch as fp32.
        if hasattr(lib, "bnb_mps_gemm_4bit_supports_bf16"):
            lib.bnb_mps_gemm_4bit_supports_bf16.restype = ct.c_int
            lib.bnb_mps_gemm_4bit_supports_bf16.argtypes = []

    def verify_buffer_contract(self) -> None:
        """Verify the undocumented torch contract that an MPS tensor's data_ptr() is its
        id<MTLBuffer>. Raises RuntimeError if a future torch has broken it, so callers can
        disable the native path loudly instead of casting garbage into a Metal kernel.
        """
        probe = torch.empty(1024, dtype=torch.float32, device="mps")
        torch.mps.synchronize()
        ok = self._lib.bnb_mps_check_buffer_contract(ct.c_void_p(probe.data_ptr()), ct.c_int64(probe.numel() * 4))
        if ok != 1:
            raise RuntimeError(
                "bitsandbytes MPS native path DISABLED: a torch MPS tensor's data_ptr() no longer "
                "resolves to its id<MTLBuffer> (this torch version broke the undocumented contract the "
                "native Metal kernels rely on). Falling back to the pure-PyTorch path. "
                f"torch={torch.__version__}. Please report this at "
                "https://github.com/bitsandbytes-foundation/bitsandbytes/issues"
            )


@functools.cache
def get_mps_library() -> Optional[MpsBNBNativeLibrary]:
    """Load the native MPS library if it (and its metallib) were built and are present.

    Returns None -- never raises -- when MPS is unavailable or the native library / metallib
    is missing (source installs without a `-DCOMPUTE_BACKEND=mps` build, wheels without it).
    The MPS backend keeps its Hub/pure-torch fallback in that case.
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None

    lib_path = PACKAGE_DIR / f"libbitsandbytes_mps{DYNAMIC_LIBRARY_SUFFIX}"
    metallib_path = PACKAGE_DIR / "bitsandbytes.metallib"

    if not lib_path.exists() or not metallib_path.exists():
        logger.debug(
            "Native MPS library not found (lib=%s exists=%s, metallib exists=%s); using Hub/pure-torch fallback.",
            lib_path,
            lib_path.exists(),
            metallib_path.exists(),
        )
        return None

    try:
        dll = ct.cdll.LoadLibrary(str(lib_path))
        if not hasattr(dll, "bnb_mps_quantize_blockwise"):
            logger.debug("Native MPS library at %s missing expected symbols; using fallback.", lib_path)
            return None
        native = MpsBNBNativeLibrary(dll, metallib_path)
    except Exception as e:
        logger.debug("Failed to load native MPS library from %s: %s; using fallback.", lib_path, e)
        return None

    # Harden the data_ptr()-is-the-MTLBuffer bridge everything rides on: verify it once,
    # loudly disable native (not crash, not corrupt) if a future torch breaks the contract.
    try:
        native.verify_buffer_contract()
    except Exception as e:
        logger.error("%s", e)
        return None

    return native


def _split_cuda_version(compact: str, is_hip: bool) -> tuple[int, int]:
    """Split a compact CUDA/ROCm version string from a library filename into (major, minor).

    CUDA: major is always 2 digits (11, 12, 13...), e.g. '118' -> (11, 8), '132' -> (13, 2).
    ROCm: major is always 1 digit for now (6, 7...), e.g. '72' -> (7, 2), '713' -> (7, 13).
    Note: revisit if ROCm major reaches 10.
    """
    if is_hip:
        return int(compact[:1]), int(compact[1:])
    return int(compact[:2]), int(compact[2:])


def _find_cuda_libs(prefix: str, is_hip: bool) -> dict[tuple[int, int], Path]:
    """Return a {(major, minor): Path} mapping for all packaged CUDA/ROCm library files."""
    result = {}
    for lib in PACKAGE_DIR.glob(f"libbitsandbytes_{prefix}*{DYNAMIC_LIBRARY_SUFFIX}"):
        match = re.search(rf"{prefix}(\d+)", lib.name)
        if match:
            try:
                result[_split_cuda_version(match.group(1), is_hip)] = lib
            except (ValueError, IndexError):
                continue
    return result


def get_available_cuda_binary_versions() -> list[str]:
    """Get formatted CUDA/ROCm versions from existing library files."""
    is_hip = bool(torch.version.hip)
    prefix = "rocm" if is_hip else "cuda"
    return sorted(f"{major}.{minor}" for major, minor in _find_cuda_libs(prefix, is_hip))


def parse_cuda_version(version_str: str) -> str:
    """Convert a raw version code string (e.g. '118', '713') to a dotted version (e.g. '11.8', '7.13')."""
    if version_str.isdigit():
        is_hip = bool(torch.version.hip)
        try:
            major, minor = _split_cuda_version(version_str, is_hip)
            return f"{major}.{minor}"
        except (ValueError, IndexError):
            pass
    return version_str


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
    4. Custom BNB_CUDA_VERSION or BNB_ROCM_VERSION override but mismatch
    5. CPU-only installation attempts when GPU functionality is requested

    """

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        self.available_versions = get_available_cuda_binary_versions()
        override_value = os.environ.get("BNB_ROCM_VERSION") if HIP_ENVIRONMENT else os.environ.get("BNB_CUDA_VERSION")
        user_version = get_cuda_version_tuple()
        user_version_str = f"{user_version[0]}.{user_version[1]}" if user_version else "unknown"
        self.requested_version = parse_cuda_version(override_value) if override_value else user_version_str

        # Pre-generate the error message based on error type
        if "cannot open shared object file" in error_msg:
            self.formatted_error = self._format_dependency_error()
        else:  # lib loading errors
            self.formatted_error = self._format_lib_error_message(
                available_versions=self.available_versions,
                user_cuda_version=user_version_str,
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
                f"bitsandbytes tried to find a compatible {BNB_BACKEND} binary but none could be loaded.\n"
                f"If your {BNB_BACKEND} version isn't among the available pre-compiled versions above, you must compile from source.\n\n"
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
                "You have two options:\n"
                "1. COMPILE FROM SOURCE as mentioned here:\n"
                "   https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=AMD+ROCm#amd-gpu\n"
                "2. Use BNB_ROCM_VERSION to specify a DIFFERENT ROCm version from the detected one, matching the version the library was built with.\n\n"
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
            f"   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/{BNB_BACKEND.lower()}-{cuda_major_version}.x/"
            f"{'lib64' if not HIP_ENVIRONMENT else 'lib'}\n\n"
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


def get_xpu_bnb_library_path() -> Path:
    """Get the path to the XPU native library matching the oneAPI toolchain.

    Prefers the versioned library (e.g. libbitsandbytes_xpu2026) matching the first
    4 digits of torch.version.xpu, falling back to an unversioned libbitsandbytes_xpu.
    """
    xpu_version = getattr(torch.version, "xpu", None)
    if xpu_version:
        versioned = PACKAGE_DIR / f"libbitsandbytes_xpu{xpu_version[:4]}{DYNAMIC_LIBRARY_SUFFIX}"
        if versioned.exists():
            return versioned
    return PACKAGE_DIR / f"libbitsandbytes_xpu{DYNAMIC_LIBRARY_SUFFIX}"


def get_native_library() -> BNBNativeLibrary:
    """
    Load CUDA library XOR CPU, as the latter contains a subset of symbols of the former.
    """
    cuda_specs = get_cuda_specs()
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"

    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)

        if not cuda_binary_path.exists():
            raise RuntimeError(f"No compatible {BNB_BACKEND} binary found at {cuda_binary_path}")

        binary_path = cuda_binary_path

    if torch._C._has_xpu:
        binary_path = get_xpu_bnb_library_path()

    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")

    # Try to load the library - any errors will propagate up
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return CudaBNBNativeLibrary(dll)

    if torch._C._has_xpu:
        return XpuBNBNativeLibrary(dll)

    return BNBNativeLibrary(dll)


ROCM_GPU_ARCH = get_rocm_gpu_arch()

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
