import gc
import pytest
import torch


_available_devices = []
if torch.cuda.is_available():
    _available_devices.append("cuda") # Covers both NVIDIA CUDA and AMD ROCm via torch.cuda
if hasattr(torch, "xpu") and torch.xpu.is_available():
    _available_devices.append("xpu")


def pytest_addoption(parser):
    if len(_available_devices) == 0:
        raise ValueError("Found no available devices.")
    parser.addoption(
        "--device",
        action="store",
        default=_available_devices[0],
        help="Specify the device to run tests on: cuda, xpu",
        choices=_available_devices,
    )

@pytest.fixture(scope="session")
def device(request):
    """Yields the device string selected via --device."""
    device_str = request.config.getoption("--device")
    return device_str



def pytest_runtest_call(item):
    try:
        item.runtest()
    except RuntimeError as re:
        if "Found no NVIDIA driver" in str(re) and "cuda" in item.config.getoption("--device"):
             pytest.skip("No NVIDIA driver found for selected CUDA device")
        raise


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()


@pytest.fixture
def requires_gpu(device, scope="session"):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("Test requires CUDA device")
    if device == 'xpu' and not torch.xpu.is_available():
        pytest.skip("Test requires XPU device")
    
    return True


@pytest.fixture
def requires_cuda(target_device):
     if target_device.type != 'cuda':
        pytest.skip("Test requires CUDA device")


@pytest.fixture
def requires_xpu(target_device):
    if target_device.type != 'xpu':
        pytest.skip("Test requires XPU device")
