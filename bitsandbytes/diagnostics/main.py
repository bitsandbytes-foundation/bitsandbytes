import sys
import traceback

import torch

from bitsandbytes.cextension import BNB_BACKEND, HIP_ENVIRONMENT
from bitsandbytes.consts import PACKAGE_GITHUB_URL
from bitsandbytes.cuda_specs import get_cuda_specs
from bitsandbytes.diagnostics.cuda import (
    print_diagnostics,
    print_runtime_diagnostics,
)
from bitsandbytes.diagnostics.utils import print_dedented, print_header


def sanity_check():
    from bitsandbytes.cextension import lib

    if lib is None:
        compute_backend = "cuda" if not HIP_ENVIRONMENT else "hip"
        print_dedented(
            f"""
            Couldn't load the bitsandbytes library, likely due to missing binaries.
            Please ensure bitsandbytes is properly installed.

            For source installations, compile the binaries with `cmake -DCOMPUTE_BACKEND={compute_backend} -S .`.
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
    if HIP_ENVIRONMENT:
        rocm_specs = f" rocm_version_string='{cuda_specs.cuda_version_string}',"
        rocm_specs += f" rocm_version_tuple={cuda_specs.cuda_version_tuple}"
        print(f"{BNB_BACKEND} specs:{rocm_specs}")
    else:
        print(f"{BNB_BACKEND} specs:{cuda_specs}")
    if not torch.cuda.is_available():
        print(f"Torch says {BNB_BACKEND} is not available. Possible reasons:")
        print(f"1. {BNB_BACKEND} driver not installed")
        print(f"2. {BNB_BACKEND} not installed")
        print(f"3. You have multiple conflicting {BNB_BACKEND} libraries")
    if cuda_specs:
        print_diagnostics(cuda_specs)
    print_runtime_diagnostics()
    print_header("")
    print_header("DEBUG INFO END")
    print_header("")
    print(f"Checking that the library is importable and {BNB_BACKEND} is callable...")
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
