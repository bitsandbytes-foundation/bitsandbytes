"""
build is dependent on
- compute capability
    - dependent on GPU family
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

alle Binaries packagen

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

from os import environ as env
from pathlib import Path
from typing import Set, Union
from .utils import warn_of_missing_prerequisite, print_err


CUDA_RUNTIME_LIB: str = "libcudart.so"

def tokenize_paths(paths: str) -> Set[Path]:
    return {
        Path(ld_path) for ld_path in paths.split(':')
        if ld_path
    }

def get_cuda_runtime_lib_path(
    # TODO: replace this with logic for all paths in env vars
    LD_LIBRARY_PATH: Union[str, None] = env.get("LD_LIBRARY_PATH")
) -> Union[Path, None]:
    """ # TODO: add doc-string
    """

    if not LD_LIBRARY_PATH:
        warn_of_missing_prerequisite(
            'LD_LIBRARY_PATH is completely missing from environment!'
        )
        return None

    ld_library_paths: Set[Path] = tokenize_paths(LD_LIBRARY_PATH)

    non_existent_directories: Set[Path]  = {
        path for path in ld_library_paths
        if not path.exists()
    }

    if non_existent_directories:
        print_err(
            "WARNING: The following directories listed your path were found to "
            f"be non-existent: {non_existent_directories}"
        )

    cuda_runtime_libs: Set[Path] = {
        path / CUDA_RUNTIME_LIB for path in ld_library_paths
        if (path / CUDA_RUNTIME_LIB).is_file()
    } - non_existent_directories

    if len(cuda_runtime_libs) > 1:
        err_msg = f"Found duplicate {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        raise FileNotFoundError(err_msg)

    elif len(cuda_runtime_libs) < 1:
        err_msg = f"Did not find {CUDA_RUNTIME_LIB} files: {cuda_runtime_libs}.."
        raise FileNotFoundError(err_msg)

    single_cuda_runtime_lib_dir = next(iter(cuda_runtime_libs))
    return ld_library_paths

def evaluate_cuda_setup():
    # - if paths faulty, return meaningful error
    # - else:
    #     - determine CUDA version
    #     - determine capabilities
    #     - based on that set the default path
    pass
