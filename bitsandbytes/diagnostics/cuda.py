import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator

import torch

from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import NONPYTORCH_DOC_URL
from bitsandbytes.cuda_specs import CUDASpecs
from bitsandbytes.diagnostics.utils import print_dedented

CUDART_PATH_PREFERRED_ENVVARS = ("CONDA_PREFIX", "LD_LIBRARY_PATH")

CUDART_PATH_IGNORED_ENVVARS = {
    "DBUS_SESSION_BUS_ADDRESS",  # hardware related
    "GOOGLE_VM_CONFIG_LOCK_FILE",  # GCP: requires elevated permissions, causing problems in VMs and Jupyter notebooks
    "HOME",  # Linux shell default
    "LESSCLOSE",
    "LESSOPEN",  # related to the `less` command
    "MAIL",  # something related to emails
    "OLDPWD",
    "PATH",  # this is for finding binaries, not libraries
    "PWD",  # PWD: this is how the shell keeps track of the current working dir
    "SHELL",  # binary for currently invoked shell
    "SSH_AUTH_SOCK",  # SSH stuff, therefore unrelated
    "SSH_TTY",
    "TMUX",  # Terminal Multiplexer
    "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
    "XDG_GREETER_DATA_DIR",  # XDG: Desktop environment stuff
    "XDG_RUNTIME_DIR",
    "_",  # current Python interpreter
}

CUDA_RUNTIME_LIB_PATTERNS = (
    "cudart64*.dll",  # Windows
    "libcudart*.so*",  # libcudart.so, libcudart.so.11.0, libcudart.so.12.0, libcudart.so.12.1, libcudart.so.12.2 etc.
    "nvcuda*.dll",  # Windows
)

logger = logging.getLogger(__name__)


def find_cuda_libraries_in_path_list(paths_list_candidate: str) -> Iterable[Path]:
    for dir_string in paths_list_candidate.split(os.pathsep):
        if not dir_string:
            continue
        if os.sep not in dir_string:
            continue
        try:
            dir = Path(dir_string)
            try:
                if not dir.exists():
                    logger.warning(f"The directory listed in your path is found to be non-existent: {dir}")
                    continue
            except OSError:  # Assume an esoteric error trying to poke at the directory
                pass
            for lib_pattern in CUDA_RUNTIME_LIB_PATTERNS:
                for pth in dir.glob(lib_pattern):
                    if pth.is_file():
                        yield pth
        except (OSError, PermissionError):
            pass


def is_relevant_candidate_env_var(env_var: str, value: str) -> bool:
    return (
        env_var in CUDART_PATH_PREFERRED_ENVVARS  # is a preferred location
        or (
            os.sep in value  # might contain a path
            and env_var not in CUDART_PATH_IGNORED_ENVVARS  # not ignored
            and "CONDA" not in env_var  # not another conda envvar
            and "BASH_FUNC" not in env_var  # not a bash function defined via envvar
            and "\n" not in value  # likely e.g. a script or something?
        )
    )


def get_potentially_lib_path_containing_env_vars() -> Dict[str, str]:
    return {env_var: value for env_var, value in os.environ.items() if is_relevant_candidate_env_var(env_var, value)}


def find_cudart_libraries() -> Iterator[Path]:
    """
    Searches for a cuda installations, in the following order of priority:
        1. active conda env
        2. LD_LIBRARY_PATH
        3. any other env vars, while ignoring those that
            - are known to be unrelated
            - don't contain the path separator `/`

    If multiple libraries are found in part 3, we optimistically try one,
    while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()

    for envvar in CUDART_PATH_PREFERRED_ENVVARS:
        if envvar in candidate_env_vars:
            directory = candidate_env_vars[envvar]
            yield from find_cuda_libraries_in_path_list(directory)
            candidate_env_vars.pop(envvar)

    for env_var, value in candidate_env_vars.items():
        yield from find_cuda_libraries_in_path_list(value)


def print_cuda_diagnostics(cuda_specs: CUDASpecs) -> None:
    print(
        f"PyTorch settings found: CUDA_VERSION={cuda_specs.cuda_version_string}, "
        f"Highest Compute Capability: {cuda_specs.highest_compute_capability}.",
    )

    binary_path = get_cuda_bnb_library_path(cuda_specs)
    if not binary_path.exists():
        print_dedented(
            f"""
        Library not found: {binary_path}. Maybe you need to compile it from source?
        If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION`,
        for example, `make CUDA_VERSION=113`.

        The CUDA version for the compile might depend on your conda install, if using conda.
        Inspect CUDA version via `conda list | grep cuda`.
        """,
        )

    cuda_major, cuda_minor = cuda_specs.cuda_version_tuple
    if cuda_major < 11:
        print_dedented(
            """
            WARNING: CUDA versions lower than 11 are currently not supported for LLM.int8().
            You will be only to use 8-bit optimizers and quantization routines!
            """,
        )

    print(f"To manually override the PyTorch CUDA version please see: {NONPYTORCH_DOC_URL}")

    # 7.5 is the minimum CC for cublaslt
    if not cuda_specs.has_cublaslt:
        print_dedented(
            """
            WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!
            If you run into issues with 8-bit matmul, you can try 4-bit quantization:
            https://huggingface.co/blog/4bit-transformers-bitsandbytes
            """,
        )

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed


def print_cuda_runtime_diagnostics() -> None:
    cudart_paths = list(find_cudart_libraries())
    if not cudart_paths:
        print("CUDA SETUP: WARNING! CUDA runtime files not found in any environmental path.")
    elif len(cudart_paths) > 1:
        print_dedented(
            f"""
            Found duplicate CUDA runtime files (see below).

            We select the PyTorch default CUDA runtime, which is {torch.version.cuda},
            but this might mismatch with the CUDA version that is needed for bitsandbytes.
            To override this behavior set the `BNB_CUDA_VERSION=<version string, e.g. 122>` environmental variable.

            For example, if you want to use the CUDA version 122,
                BNB_CUDA_VERSION=122 python ...

            OR set the environmental variable in your .bashrc:
                export BNB_CUDA_VERSION=122

            In the case of a manual override, make sure you set LD_LIBRARY_PATH, e.g.
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2,
            """,
        )
        for pth in cudart_paths:
            print(f"* Found CUDA runtime at: {pth}")
