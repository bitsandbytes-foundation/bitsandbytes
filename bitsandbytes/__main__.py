import os
from os.path import isdir
import shlex
import subprocess
import sys
from typing import Tuple
from warnings import warn

import torch

HEADER_WIDTH = 60


def find_file_recursive(folder, filename):
    import glob
    outs = []
    try:
        for ext in ["so", "dll", "dylib"]:
            out = glob.glob(os.path.join(folder, "**", filename + ext))
            outs.extend(out)
    except Exception as e:
        raise RuntimeError('Error: Something when wrong when trying to find file. {e}')

    return outs


def generate_bug_report_information():
    print_header("")
    print_header("BUG REPORT INFORMATION")
    print_header("")
    print('')

    if 'CONDA_PREFIX' in os.environ:
        paths = find_file_recursive(os.environ['CONDA_PREFIX'], '*cuda*')
        print_header("ANACONDA CUDA PATHS")
        print(paths)
        print('')
    if isdir('/usr/local/'):
        paths = find_file_recursive('/usr/local', '*cuda*')
        print_header("/usr/local CUDA PATHS")
        print(paths)
        print('')
    if 'CUDA_PATH' in os.environ and isdir(os.environ['CUDA_PATH']):
        paths = find_file_recursive(os.environ['CUDA_PATH'], '*cuda*')
        print_header("CUDA PATHS")
        print(paths)
        print('')

    if isdir(os.getcwd()):
        paths = find_file_recursive(os.getcwd(), '*cuda*')
        print_header("WORKING DIRECTORY CUDA PATHS")
        print(paths)
        print('')

    print_header("LD_LIBRARY CUDA PATHS")
    if 'LD_LIBRARY_PATH' in os.environ:
        lib_path = os.environ['LD_LIBRARY_PATH'].strip()
        for path in set(lib_path.split(os.pathsep)):
            try:
                if isdir(path):
                    print_header(f"{path} CUDA PATHS")
                    paths = find_file_recursive(path, '*cuda*')
                    print(paths)
            except Exception as e:
                print(f'Could not read LD_LIBRARY_PATH: {path} ({e})')
    print('')





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


generate_bug_report_information()


from . import COMPILED_WITH_CUDA, PACKAGE_GITHUB_URL
from .cuda_setup.main import get_compute_capabilities

print_header("OTHER")
print(f"COMPILED_WITH_CUDA = {COMPILED_WITH_CUDA}")
print(f"COMPUTE_CAPABILITIES_PER_GPU = {get_compute_capabilities()}")
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
print("\nWARNING: Please be sure to sanitize sensible info from any such env vars!\n")

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
