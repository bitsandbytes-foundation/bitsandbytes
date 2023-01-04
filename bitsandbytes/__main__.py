import os
import sys
from warnings import warn

import torch

HEADER_WIDTH = 60


def print_header(
    txt: str, width: int = HEADER_WIDTH, filler: str = "+"
) -> None:
    txt = f" {txt} " if txt else ""
    print(txt.center(width, filler))


def print_debug_info() -> None:
    print(
        "\nAbove we output some debug information. Please provide this info when "
        f"creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose ...\n"
    )


print_header("")
print_header("DEBUG INFORMATION")
print_header("")
print()


from . import COMPILED_WITH_CUDA, PACKAGE_GITHUB_URL
from .cuda_setup.env_vars import to_be_ignored
from .cuda_setup.main import get_compute_capabilities, get_cuda_lib_handle

print_header("POTENTIALLY LIBRARY-PATH-LIKE ENV VARS")
for k, v in os.environ.items():
    if "/" in v and not to_be_ignored(k, v):
        print(f"'{k}': '{v}'")
print_header("")

print(
    "\nWARNING: Please be sure to sanitize sensible info from any such env vars!\n"
)

print_header("OTHER")
print(f"COMPILED_WITH_CUDA = {COMPILED_WITH_CUDA}")
cuda = get_cuda_lib_handle()
print(f"COMPUTE_CAPABILITIES_PER_GPU = {get_compute_capabilities(cuda)}")
print_header("")
print_header("DEBUG INFO END")
print_header("")
print(
    """
Running a quick check that:
    + library is importable
    + CUDA function is callable
"""
)

try:
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
    print("SUCCESS!")
    print("Installation was successful!")
    sys.exit(0)

except ImportError:
    print()
    warn(
        f"WARNING: {__package__} is currently running as CPU-only!\n"
        "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
        f"If you think that this is so erroneously,\nplease report an issue!"
    )
    print_debug_info()
    sys.exit(0)
except Exception as e:
    print(e)
    print_debug_info()
    sys.exit(1)
