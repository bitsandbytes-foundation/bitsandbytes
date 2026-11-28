import gc
import random

import numpy as np
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-mps-hub",
        action="store_true",
        help="Require MPS, the bitsandbytes-mps Hub kernel, and a passing test that verifies Hub execution.",
    )


def _require_mps_hub():
    from tests.helpers import get_available_devices

    if "mps" not in get_available_devices() or not torch.backends.mps.is_available():
        raise pytest.UsageError("--require-mps-hub requires an available MPS device selected by BNB_TEST_DEVICE.")

    from bitsandbytes.backends.mps import ops

    kernel = ops._get_kernel()
    if kernel is None:
        raise pytest.UsageError(
            "--require-mps-hub could not load kernels-community/bitsandbytes-mps; "
            "macOS 26+ and a compatible kernels package/Hub build are required."
        )
    return kernel


class _MPSHubExecutionCheck:
    def __init__(self):
        self.verified_tests = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed and ("mps_hub_verified", True) in report.user_properties:
            self.verified_tests += 1

    def pytest_sessionfinish(self, session, exitstatus):
        if exitstatus == pytest.ExitCode.OK and not self.verified_tests:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_sep(
                    "=",
                    "ERROR: --require-mps-hub completed without any passing test verifying Hub execution. "
                    "Include test_dequantize_4bit_mps with a supported Hub block size.",
                    red=True,
                )


def pytest_configure(config):
    if config.getoption("--require-mps-hub"):
        _require_mps_hub()
        config.pluginmanager.register(_MPSHubExecutionCheck(), "mps-hub-execution-check")


def pytest_report_header(config):
    if config.getoption("--require-mps-hub"):
        from importlib.metadata import version

        kernel = _require_mps_hub()
        return f"MPS Hub required: torch={torch.__version__}, kernels={version('kernels')}, variant={kernel.__file__}"


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


_teardown_counter = 0


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    global _teardown_counter
    _teardown_counter += 1
    if _teardown_counter % 50 == 0 or nextitem is None:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.empty_cache()


@pytest.fixture(scope="session")
def requires_cuda() -> bool:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pytest.skip("CUDA is required")
    return cuda_available
