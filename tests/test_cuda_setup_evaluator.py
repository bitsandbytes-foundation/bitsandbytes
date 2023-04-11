import os
from typing import List, NamedTuple

import pytest

import bitsandbytes as bnb
from bitsandbytes.cuda_setup.main import (
    determine_cuda_runtime_lib_path,
    evaluate_cuda_setup,
    extract_candidate_paths,
)


def test_cuda_full_system():
    ## this only tests the cuda version and not compute capability

    # if CONDA_PREFIX exists, it has priority before all other env variables
    # but it does not contain the library directly, so we need to look at the a sub-folder
    version = ""
    if "CONDA_PREFIX" in os.environ:
        ls_output, err = bnb.utils.execute_and_return(f'ls -l {os.environ["CONDA_PREFIX"]}/lib/libcudart.so.11.0')
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
