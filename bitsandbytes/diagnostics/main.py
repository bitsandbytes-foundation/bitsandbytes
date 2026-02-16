import importlib
import platform
import sys
import traceback

import torch

from bitsandbytes import __version__ as bnb_version
from bitsandbytes.cextension import BNB_BACKEND
from bitsandbytes.consts import PACKAGE_GITHUB_URL
from bitsandbytes.cuda_specs import get_cuda_specs
from bitsandbytes.diagnostics.cuda import (
    print_diagnostics,
)
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

    # TODO: There's a lot of noise in this; needs improvement.
    # print_cuda_runtime_diagnostics()

    if not torch.cuda.is_available():
        print(f"PyTorch says {BNB_BACKEND} is not available. Possible reasons:")
        print(f"1. {BNB_BACKEND} driver not installed")
        print("2. Using a CPU-only PyTorch build")
        print("3. No GPU detected")

    else:
        print(f"Checking that the library is importable and {BNB_BACKEND} is callable...")

        try:
            sanity_check()
            print("SUCCESS!")
            return
        except RuntimeError as e:
            if "not available in CPU-only" in str(e):
                print(
                    f"WARNING: {__package__} is currently running as CPU-only!\n"
                    "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
                    f"If you think that this is so erroneously,\nplease report an issue!",
                )
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
