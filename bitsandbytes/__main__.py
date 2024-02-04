import glob
import os
import sys
from warnings import warn

import torch

HEADER_WIDTH = 60


def find_dynamic_library(folder, filename):
    for ext in ("so", "dll", "dylib"):
        yield from glob.glob(os.path.join(folder, "**", filename + ext))


def generate_bug_report_information():
    print_header("")
    print_header("BUG REPORT INFORMATION")
    print_header("")
    print('')

    path_sources = [
        ("ANACONDA CUDA PATHS", os.environ.get("CONDA_PREFIX")),
        ("/usr/local CUDA PATHS", "/usr/local"),
        ("CUDA PATHS", os.environ.get("CUDA_PATH")),
        ("WORKING DIRECTORY CUDA PATHS", os.getcwd()),
    ]
    try:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH")
        if ld_library_path:
            for path in set(ld_library_path.strip().split(os.pathsep)):
                path_sources.append((f"LD_LIBRARY_PATH {path} CUDA PATHS", path))
    except Exception as e:
        print(f"Could not parse LD_LIBRARY_PATH: {e}")

    for name, path in path_sources:
        if path and os.path.isdir(path):
            print_header(name)
            print(list(find_dynamic_library(path, '*cuda*')))
            print("")


def print_header(
    txt: str, width: int = HEADER_WIDTH, filler: str = "+"
) -> None:
    txt = f" {txt} " if txt else ""
    print(txt.center(width, filler))


def print_debug_info() -> None:
    from . import PACKAGE_GITHUB_URL
    print(
        "\nAbove we output some debug information. Please provide this info when "
        f"creating an issue via {PACKAGE_GITHUB_URL}/issues/new/choose ...\n"
    )


def main():
    generate_bug_report_information()

    from . import COMPILED_WITH_CUDA
    from .cuda_setup.main import get_compute_capabilities

    print_header("OTHER")
    print(f"COMPILED_WITH_CUDA = {COMPILED_WITH_CUDA}")
    print(f"COMPUTE_CAPABILITIES_PER_GPU = {get_compute_capabilities()}")
    print_header("")
    print_header("DEBUG INFO END")
    print_header("")
    print("Checking that the library is importable and CUDA is callable...")
    print("\nWARNING: Please be sure to sanitize sensitive info from any such env vars!\n")

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
    except ImportError:
        print()
        warn(
            f"WARNING: {__package__} is currently running as CPU-only!\n"
            "Therefore, 8-bit optimizers and GPU quantization are unavailable.\n\n"
            f"If you think that this is so erroneously,\nplease report an issue!"
        )
        print_debug_info()
    except Exception as e:
        print(e)
        print_debug_info()
        sys.exit(1)


if __name__ == "__main__":
    main()
