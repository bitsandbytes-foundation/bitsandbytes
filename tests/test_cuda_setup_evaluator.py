import os
import pytest
import bitsandbytes as bnb

from typing import List, NamedTuple

from bitsandbytes.cuda_setup import (
    CUDA_RUNTIME_LIB,
    evaluate_cuda_setup,
    get_cuda_runtime_lib_path,
    tokenize_paths,
)


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
    for path in tokenize_paths(request.param):
        test_dir.mkdir()
        if CUDA_RUNTIME_LIB in path:
            (test_input / CUDA_RUNTIME_LIB).touch()


@pytest.mark.parametrize("test_input, expected", HAPPY_PATH__LD_LIB_TEST_PATHS)
def test_get_cuda_runtime_lib_path__happy_path(
    tmp_path, test_input: str, expected: str
):
    for path in tokenize_paths(test_input):
        path.mkdir()
        (path / CUDA_RUNTIME_LIB).touch()
    assert get_cuda_runtime_lib_path(test_input) == expected


UNHAPPY_PATH__LD_LIB_TEST_PATHS = [
    f"a/b/c/{CUDA_RUNTIME_LIB}:d/e/f/{CUDA_RUNTIME_LIB}",
    f"a/b/c/{CUDA_RUNTIME_LIB}:d/e/f/{CUDA_RUNTIME_LIB}:g/h/j/{CUDA_RUNTIME_LIB}",
]


@pytest.mark.parametrize("test_input", UNHAPPY_PATH__LD_LIB_TEST_PATHS)
def test_get_cuda_runtime_lib_path__unhappy_path(tmp_path, test_input: str):
    test_input = tmp_path / test_input
    (test_input / CUDA_RUNTIME_LIB).touch()
    with pytest.raises(FileNotFoundError) as err_info:
        get_cuda_runtime_lib_path(test_input)
    assert all(match in err_info for match in {"duplicate", CUDA_RUNTIME_LIB})


def test_get_cuda_runtime_lib_path__non_existent_dir(capsys, tmp_path):
    existent_dir = tmp_path / "a/b"
    existent_dir.mkdir()
    non_existent_dir = tmp_path / "c/d"  # non-existent dir
    test_input = ":".join([str(existent_dir), str(non_existent_dir)])

    get_cuda_runtime_lib_path(test_input)
    std_err = capsys.readouterr().err

    assert all(match in std_err for match in {"WARNING", "non-existent"})


def test_full_system():
    ## this only tests the cuda version and not compute capability

    # if CONDA_PREFIX exists, it has priority before all other env variables
    # but it does not contain the library directly, so we need to look at the a sub-folder
    version = ''
    if 'CONDA_PREFIX' in os.environ:
        ls_output, err = bnb.utils.execute_and_return(f'ls -l {os.environ["CONDA_PREFIX"]}/lib/libcudart.so')
        major, minor, revision = ls_output.split(' ')[-1].replace('libcudart.so.', '').split('.')
        version = float(f'{major}.{minor}')


    if version == '' and 'LD_LIBRARY_PATH':
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
    binary_name = evaluate_cuda_setup()
    binary_name = binary_name.replace("libbitsandbytes_cuda", "")
    assert binary_name.startswith(str(version).replace(".", ""))
