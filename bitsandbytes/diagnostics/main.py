import importlib
import logging
import os
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

logger = logging.getLogger(__name__)


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

    logger.info("Platform: %s", platform.platform())
    if platform.system() == "Linux":
        logger.info("  libc: %s", "-".join(platform.libc_ver()))

    logger.info("Python: %s", platform.python_version())

    logger.info("PyTorch: %s", torch.__version__)
    logger.info("  CUDA: %s", torch.version.cuda or "N/A")
    logger.info("  HIP: %s", torch.version.hip or "N/A")
    logger.info("  XPU: %s", getattr(torch.version, "xpu", "N/A") or "N/A")

    logger.info("Related packages:")
    for pkg in _RELATED_PACKAGES:
        version = get_package_version(pkg)
        logger.info("  %s: %s", pkg, version)


def main():
    # bitsandbytes' CLI entrypoint: configure logging for human-readable output.
    # Library imports do not configure logging; downstream apps should decide.
    level_name = os.environ.get("BNB_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")

    print_header(f"bitsandbytes v{bnb_version}")
    show_environment()
    print_header("")

    cuda_specs = get_cuda_specs()

    if cuda_specs:
        print_diagnostics(cuda_specs)

    # TODO: There's a lot of noise in this; needs improvement.
    # print_cuda_runtime_diagnostics()

    if not torch.cuda.is_available():
        logger.warning("PyTorch says %s is not available. Possible reasons:", BNB_BACKEND)
        logger.warning("1. %s driver not installed", BNB_BACKEND)
        logger.warning("2. Using a CPU-only PyTorch build")
        logger.warning("3. No GPU detected")

    else:
        logger.info("Checking that the library is importable and %s is callable...", BNB_BACKEND)

        try:
            sanity_check()
            logger.info("SUCCESS!")
            return
        except RuntimeError as e:
            if "not available in CPU-only" in str(e):
                logger.warning(
                    "WARNING: %s is currently running as CPU-only!\n"
                    "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
                    "If you think that this is so erroneously,\nplease report an issue!",
                    __package__,
                )
            else:
                raise e
        except Exception:
            logger.exception("Diagnostics sanity check failed:")

        print_dedented(
            f"""
            Above we output some debug information.
            Please provide this info when creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose
            WARNING: Please be sure to sanitize sensitive info from the output before posting it.
            """,
        )
        sys.exit(1)
