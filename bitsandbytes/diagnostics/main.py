import sys
import traceback

import torch

from bitsandbytes.consts import PACKAGE_GITHUB_URL
from bitsandbytes.cuda_specs import get_cuda_specs
from bitsandbytes.diagnostics.cuda import (
    print_cuda_diagnostics,
    print_cuda_runtime_diagnostics,
)
from bitsandbytes.diagnostics.utils import print_dedented, print_header


def sanity_check():
    from bitsandbytes.cextension import lib

    if lib is None:
        print_dedented(
            """
            Couldn't load the bitsandbytes library, likely due to missing binaries.
            Please ensure bitsandbytes is properly installed.

            For source installations, compile the binaries with `cmake -DCOMPUTE_BACKEND=cuda -S .`.
            See the documentation for more details if needed.

            Trying a simple check anyway, but this will likely fail...
            """,
        )

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


def main():
    print_header("")
    print_header("BUG REPORT INFORMATION")
    print_header("")

    print_header("OTHER")
    cuda_specs = get_cuda_specs()
    print("CUDA specs:", cuda_specs)
    if not torch.cuda.is_available():
        print("Torch says CUDA is not available. Possible reasons:")
        print("1. CUDA driver not installed")
        print("2. CUDA not installed")
        print("3. You have multiple conflicting CUDA libraries")
    if cuda_specs:
        print_cuda_diagnostics(cuda_specs)
    print_cuda_runtime_diagnostics()
    print_header("")
    print_header("DEBUG INFO END")
    print_header("")
    print("Checking that the library is importable and CUDA is callable...")
    try:
        sanity_check()
        print("SUCCESS!")
        print("Installation was successful!")
        return
    except ImportError:
        print(
            f"WARNING: {__package__} is currently running as CPU-only!\n"
            "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
            f"If you think that this is so erroneously,\nplease report an issue!",
        )
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
