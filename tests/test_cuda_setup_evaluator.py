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


# Simulates torch+rocm7.0 (PyTorch bundled ROCm) on a system with ROCm 7.2
@pytest.fixture
def rocm70_spec() -> CUDASpecs:
    return CUDASpecs(
        cuda_version_string="70",  # from torch.version.hip == "7.0.x"
        highest_compute_capability=(0, 0),  # unused for ROCm library path resolution
        cuda_version_tuple=(7, 0),
    )


@pytest.mark.skipif(not HIP_ENVIRONMENT, reason="this test is only supported on ROCm")
def test_get_rocm_bnb_library_path(monkeypatch, rocm70_spec):
    """Without override, library path uses PyTorch's ROCm 7.0 version."""
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(rocm70_spec).stem == "libbitsandbytes_rocm70"


@pytest.mark.skipif(not HIP_ENVIRONMENT, reason="this test is only supported on ROCm")
def test_get_rocm_bnb_library_path_override(monkeypatch, rocm70_spec, caplog):
    """BNB_ROCM_VERSION=72 overrides to load the ROCm 7.2 library instead of 7.0."""
    monkeypatch.setenv("BNB_ROCM_VERSION", "72")
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(rocm70_spec).stem == "libbitsandbytes_rocm72"
    assert "BNB_ROCM_VERSION" in caplog.text


@pytest.mark.skipif(not HIP_ENVIRONMENT, reason="this test is only supported on ROCm")
def test_get_rocm_bnb_library_path_rejects_cuda_override(monkeypatch, rocm70_spec):
    """BNB_CUDA_VERSION should be rejected on ROCm with a helpful error."""
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    monkeypatch.setenv("BNB_CUDA_VERSION", "72")
    with pytest.raises(RuntimeError, match=r"BNB_CUDA_VERSION.*detected for ROCm"):
        get_cuda_bnb_library_path(rocm70_spec)


@pytest.mark.skipif(not HIP_ENVIRONMENT, reason="this test is only supported on ROCm")
def test_get_rocm_bnb_library_path_rocm_override_takes_priority(monkeypatch, rocm70_spec, caplog):
    """When both are set, BNB_ROCM_VERSION wins if HIP_ENVIRONMENT is True."""
    monkeypatch.setenv("BNB_ROCM_VERSION", "72")
    monkeypatch.setenv("BNB_CUDA_VERSION", "72")
    assert get_cuda_bnb_library_path(rocm70_spec).stem == "libbitsandbytes_rocm72"
    assert "BNB_ROCM_VERSION" in caplog.text
    assert "BNB_CUDA_VERSION" not in caplog.text
