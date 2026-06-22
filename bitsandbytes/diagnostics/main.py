import importlib
import platform
import sys
import traceback

import torch

from bitsandbytes import __version__ as bnb_version
from bitsandbytes.consts import PACKAGE_GITHUB_URL
from bitsandbytes.cuda_specs import get_cuda_specs
from bitsandbytes.diagnostics.cuda import print_diagnostics
from bitsandbytes.diagnostics.utils import print_dedented, print_header

_RELATED_PACKAGES = [
    "accelerate",
    "diffusers",
    "numpy",
    "pip",
    "peft",
    "safetensors",
    "transformers",
    "triton",
    "trl",
]


def sanity_check():
    from bitsandbytes.optim import Adam

    p = torch.nn.Parameter(torch.rand(10, 10).cuda())
    a = torch.rand(10, 10).cuda()
    p1 = p.data.sum().item()
    adam = Adam([p])
    out = a * p
    loss = out.sum()
    loss.backward()
    adam.step()
    p2 = p.data.sum().item()
    assert p1 != p2


def get_package_version(name: str) -> str:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "not found"
    return version


def show_environment():
    """Simple utility to print out environment information."""

    print(f"Platform: {platform.platform()}")
    if platform.system() == "Linux":
        print(f"  libc: {'-'.join(platform.libc_ver())}")

    print(f"Python: {platform.python_version()}")

    print(f"PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda or 'N/A'}")
    print(f"  HIP: {torch.version.hip or 'N/A'}")
    print(f"  XPU: {getattr(torch.version, 'xpu', 'N/A') or 'N/A'}")

    print("Related packages:")
    for pkg in _RELATED_PACKAGES:
        version = get_package_version(pkg)
        print(f"  {pkg}: {version}")


def main():
    print_header(f"bitsandbytes v{bnb_version}")
    show_environment()
    print_header("")

    cuda_specs = get_cuda_specs()

    if cuda_specs:
        print_diagnostics(cuda_specs)

    has_rocm = torch.version.hip is not None
    has_cuda = not has_rocm and torch.version.cuda is not None and torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    from bitsandbytes.cextension import ErrorHandlerMockBNBNativeLibrary, lib

    lib_loaded = not isinstance(lib, ErrorHandlerMockBNBNativeLibrary)

    if not (has_cuda or has_rocm or has_xpu):
        print(
            f"No CUDA, ROCm, or XPU detected; CPU library {'loaded successfully' if lib_loaded else 'failed to load'}."
        )
    elif has_xpu:
        from bitsandbytes.backends.utils import triton_available

        if not isinstance(lib, ErrorHandlerMockBNBNativeLibrary):
            print("XPU native library loaded successfully.")
        elif triton_available:
            print("XPU native library not loaded; using triton fallback.")
        else:
            print("XPU native library not loaded and triton not available.")
    else:
        if not lib_loaded:
            print_dedented(
                f"""
                See above for details on why the library failed to load.
                Please provide this info when creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose
                WARNING: Please be sure to sanitize sensitive info from the output before posting it.
                """,
            )
            sys.exit(1)

        print("Checking that the library is importable and callable...")
        try:
            sanity_check()
            print("SUCCESS!")
            return
        except RuntimeError as e:
            if "not available in CPU-only" in str(e):
                print("WARNING: bitsandbytes is running as CPU-only!")
                print("8-bit optimizers and GPU quantization are unavailable.")
                print("If you think this is an error, please report an issue.")
            else:
                raise e
        except Exception:
            traceback.print_exc()

        print_dedented(
            f"""
            Above we output some debug information.
            Please provide this info when creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose
            WARNING: Please be sure to sanitize sensitive info from the output before posting it.
            """,
        )
        sys.exit(1)
