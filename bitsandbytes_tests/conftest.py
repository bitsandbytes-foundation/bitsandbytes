import gc
import random

import numpy as np
import pytest
import torch


def _set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.mps.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def pytest_runtest_call(item):
    try:
        _set_seed()
        item.runtest()
    except AssertionError as ae:
        if str(ae) == "Torch not compiled with CUDA enabled":
            pytest.skip("Torch not compiled with CUDA enabled")
        raise
    except RuntimeError as re:
        # CUDA-enabled Torch build, but no CUDA-capable device found
        if "Found no NVIDIA driver on your system" in str(re):
            pytest.skip("No NVIDIA driver found")
        raise


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def requires_cuda() -> bool:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pytest.skip("CUDA is required")
    return cuda_available
