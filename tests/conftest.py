import pytest

from testutil import get_gpu_devices

@pytest.fixture(scope="session", params=get_gpu_devices())
def gpu_device(request):
    """
    Use this fixture to run the test across all GPU devices supported by the environment.
    """
    return request.param

@pytest.fixture(scope="session", params=[ "cpu" ] + get_gpu_devices())
def device(request):
    """
    Use this fixture to run the test across all compute devices (CPU + GPU) supported by the environment.
    """
    return request.param
