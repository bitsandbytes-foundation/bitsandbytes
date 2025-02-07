import pytest

from bitsandbytes.cextension import HIP_ENVIRONMENT, get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs


@pytest.fixture
def cuda120_spec() -> CUDASpecs:
    return CUDASpecs(
        cuda_version_string="120",
        highest_compute_capability=(8, 6),
        cuda_version_tuple=(12, 0),
    )


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm")
def test_get_cuda_bnb_library_path(monkeypatch, cuda120_spec):
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(cuda120_spec).stem == "libbitsandbytes_cuda120"


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm")
def test_get_cuda_bnb_library_path_override(monkeypatch, cuda120_spec, caplog):
    monkeypatch.setenv("BNB_CUDA_VERSION", "110")
    assert get_cuda_bnb_library_path(cuda120_spec).stem == "libbitsandbytes_cuda110"
    assert "BNB_CUDA_VERSION" in caplog.text  # did we get the warning?
