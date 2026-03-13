import pytest

from bitsandbytes.cextension import BNB_BACKEND, get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs


@pytest.fixture
def cuda120_spec() -> CUDASpecs:
    """Simulates torch+cuda12.0 and a representative Ampere-class capability."""
    return CUDASpecs(
        cuda_version_string="120",
        highest_compute_capability=(8, 6),
        cuda_version_tuple=(12, 0),
    )


@pytest.mark.skipif(BNB_BACKEND != "CUDA", reason="this test requires a CUDA backend")
def test_get_cuda_bnb_library_path(monkeypatch, cuda120_spec):
    """Without overrides, library path uses the detected CUDA 12.0 version."""
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(cuda120_spec).stem == "libbitsandbytes_cuda120"


@pytest.mark.skipif(BNB_BACKEND != "CUDA", reason="this test requires a CUDA backend")
def test_get_cuda_bnb_library_path_override(monkeypatch, cuda120_spec, caplog):
    """BNB_CUDA_VERSION=110 overrides path selection to the CUDA 11.0 binary."""
    monkeypatch.setenv("BNB_CUDA_VERSION", "110")
    assert get_cuda_bnb_library_path(cuda120_spec).stem == "libbitsandbytes_cuda110"
    assert "BNB_CUDA_VERSION" in caplog.text  # did we get the warning?


@pytest.mark.skipif(BNB_BACKEND != "CUDA", reason="this test requires a CUDA backend")
def test_get_cuda_bnb_library_path_rejects_rocm_override(monkeypatch, cuda120_spec):
    """BNB_ROCM_VERSION should be rejected on CUDA with a helpful error."""
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    monkeypatch.setenv("BNB_ROCM_VERSION", "72")
    with pytest.raises(RuntimeError, match=r"BNB_ROCM_VERSION.*detected for CUDA"):
        get_cuda_bnb_library_path(cuda120_spec)


@pytest.fixture
def rocm70_spec() -> CUDASpecs:
    """Simulates torch+rocm7.0 (bundled ROCm) when the system ROCm is newer."""
    return CUDASpecs(
        cuda_version_string="70",
        highest_compute_capability=(0, 0),
        cuda_version_tuple=(7, 0),
    )


@pytest.mark.skipif(BNB_BACKEND != "ROCm", reason="this test requires a ROCm backend")
def test_get_rocm_bnb_library_path(monkeypatch, rocm70_spec):
    """Without override, library path uses PyTorch's ROCm 7.0 version."""
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(rocm70_spec).stem == "libbitsandbytes_rocm70"


@pytest.mark.skipif(BNB_BACKEND != "ROCm", reason="this test requires a ROCm backend")
def test_get_rocm_bnb_library_path_override(monkeypatch, rocm70_spec, caplog):
    """BNB_ROCM_VERSION=72 overrides to load the ROCm 7.2 library instead of 7.0."""
    monkeypatch.setenv("BNB_ROCM_VERSION", "72")
    assert "BNB_ROCM_VERSION" in caplog.text


@pytest.mark.skipif(BNB_BACKEND != "ROCm", reason="this test requires a ROCm backend")
def test_get_rocm_bnb_library_path_rejects_cuda_override(monkeypatch, rocm70_spec):
    """BNB_CUDA_VERSION should be rejected on ROCm with a helpful error."""
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    monkeypatch.setenv("BNB_CUDA_VERSION", "120")
    with pytest.raises(RuntimeError, match=r"BNB_CUDA_VERSION.*detected for ROCm"):
        get_cuda_bnb_library_path(rocm70_spec)