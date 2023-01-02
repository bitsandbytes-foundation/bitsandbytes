import os
from typing import List, NamedTuple

import pytest

import bitsandbytes as bnb
from bitsandbytes.cuda_setup.main import (
    CUDA_RUNTIME_LIB,
    determine_cuda_runtime_lib_path,
    evaluate_cuda_setup,
    extract_candidate_paths,
)

"""
'LD_LIBRARY_PATH': ':/mnt/D/titus/local/cuda-11.1/lib64/'
'CONDA_EXE': '/mnt/D/titus/miniconda/bin/conda'
'LESSCLOSE': '/usr/bin/lesspipe %s %s'
'OLDPWD': '/mnt/D/titus/src'
'CONDA_PREFIX': '/mnt/D/titus/miniconda/envs/8-bit'
'SSH_AUTH_SOCK': '/mnt/D/titus/.ssh/ssh-agent.tim-uw.sock'
'CONDA_PREFIX_1': '/mnt/D/titus/miniconda'
'PWD': '/mnt/D/titus/src/8-bit'
'HOME': '/mnt/D/titus'
'CONDA_PYTHON_EXE': '/mnt/D/titus/miniconda/bin/python'
'CUDA_HOME': '/mnt/D/titus/local/cuda-11.1/'
'TMUX': '/tmp/tmux-1007/default,59286,1'
'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop'
'SSH_TTY': '/dev/pts/0'
'MAIL': '/var/mail/titus'
'SHELL': '/bin/bash'
'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1007/bus'
'XDG_RUNTIME_DIR': '/run/user/1007'
'PATH': '/mnt/D/titus/miniconda/envs/8-bit/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/mnt/D/titus/local/cuda-11.1/bin'
'LESSOPEN': '| /usr/bin/lesspipe %s'
'_': '/mnt/D/titus/miniconda/envs/8-bit/bin/python'
# any that include 'CONDA' that are not 'CONDA_PREFIX'

# we search for
'CUDA_HOME': '/mnt/D/titus/local/cuda-11.1/'
"""


class InputAndExpectedOutput(NamedTuple):
    input: str
    output: str


HAPPY_PATH__LD_LIB_TEST_PATHS: List[InputAndExpectedOutput] = [
    (
        f"some/other/dir:dir/with/{CUDA_RUNTIME_LIB}",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
    (
        f":some/other/dir:dir/with/{CUDA_RUNTIME_LIB}",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
    (
        f"some/other/dir:dir/with/{CUDA_RUNTIME_LIB}:",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
    (
        f"some/other/dir::dir/with/{CUDA_RUNTIME_LIB}",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
    (
        f"dir/with/{CUDA_RUNTIME_LIB}:some/other/dir",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
    (
        f"dir/with/{CUDA_RUNTIME_LIB}:other/dir/libcuda.so",
        f"dir/with/{CUDA_RUNTIME_LIB}",
    ),
]


@pytest.fixture(params=HAPPY_PATH__LD_LIB_TEST_PATHS)
def happy_path_path_string(tmpdir, request):
    for path in extract_candidate_paths(request.param):
        test_dir.mkdir()
        if CUDA_RUNTIME_LIB in path:
            (test_input / CUDA_RUNTIME_LIB).touch()

UNHAPPY_PATH__LD_LIB_TEST_PATHS = [
    f"a/b/c/{CUDA_RUNTIME_LIB}:d/e/f/{CUDA_RUNTIME_LIB}",
    f"a/b/c/{CUDA_RUNTIME_LIB}:d/e/f/{CUDA_RUNTIME_LIB}:g/h/j/{CUDA_RUNTIME_LIB}",
]


def test_full_system():
    ## this only tests the cuda version and not compute capability

    # if CONDA_PREFIX exists, it has priority before all other env variables
    # but it does not contain the library directly, so we need to look at the a sub-folder
    version = ""
    if "CONDA_PREFIX" in os.environ:
        ls_output, err = bnb.utils.execute_and_return(f'ls -l {os.environ["CONDA_PREFIX"]}/lib/libcudart.so')
        major, minor, revision = (ls_output.split(" ")[-1].replace("libcudart.so.", "").split("."))
        version = float(f"{major}.{minor}")

    if version == "" and "LD_LIBRARY_PATH" in os.environ:
        ld_path = os.environ["LD_LIBRARY_PATH"]
        paths = ld_path.split(":")
        version = ""
        for p in paths:
            if "cuda" in p:
                idx = p.rfind("cuda-")
                version = p[idx + 5 : idx + 5 + 4].replace("/", "")
                version = float(version)
                break


    assert version > 0
    binary_name, cudart_path, cuda, cc, cuda_version_string = evaluate_cuda_setup()
    binary_name = binary_name.replace("libbitsandbytes_cuda", "")
    assert binary_name.startswith(str(version).replace(".", ""))
