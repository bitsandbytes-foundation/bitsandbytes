from pathlib import Path
from unittest.mock import patch

import pytest

from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX
from bitsandbytes.cuda_specs import CUDASpecs


@pytest.fixture
def cuda120_spec() -> CUDASpecs:
    """Simulates torch+cuda12.0 and a representative Ampere-class capability."""
    return CUDASpecs(
        cuda_version_string="120",
        highest_compute_capability=(8, 6),
        cuda_version_tuple=(12, 0),
    )


@pytest.fixture
def rocm70_spec() -> CUDASpecs:
    """Simulates torch+rocm7.0."""
    return CUDASpecs(
        cuda_version_string="70",
        highest_compute_capability=(0, 0),
        cuda_version_tuple=(7, 0),
    )


@pytest.mark.parametrize(
    "spec,fake_libs,hip_version,expected_name,expect_warning",
    [
        # exact match
        (
            CUDASpecs(cuda_version_string="124", highest_compute_capability=(8, 6), cuda_version_tuple=(12, 4)),
            {(12, 4): Path(f"libbitsandbytes_cuda124{DYNAMIC_LIBRARY_SUFFIX}")},
            None,
            f"libbitsandbytes_cuda124{DYNAMIC_LIBRARY_SUFFIX}",
            False,
        ),
        # forward fallback within major: 12.0 -> 12.1
        (
            CUDASpecs(cuda_version_string="120", highest_compute_capability=(8, 6), cuda_version_tuple=(12, 0)),
            {
                (12, 1): Path(f"libbitsandbytes_cuda121{DYNAMIC_LIBRARY_SUFFIX}"),
                (12, 4): Path(f"libbitsandbytes_cuda124{DYNAMIC_LIBRARY_SUFFIX}"),
            },
            None,
            f"libbitsandbytes_cuda121{DYNAMIC_LIBRARY_SUFFIX}",
            True,
        ),
        # backward fallback: 12.9 -> 12.8
        (
            CUDASpecs(cuda_version_string="129", highest_compute_capability=(8, 9), cuda_version_tuple=(12, 9)),
            {
                (12, 4): Path(f"libbitsandbytes_cuda124{DYNAMIC_LIBRARY_SUFFIX}"),
                (12, 8): Path(f"libbitsandbytes_cuda128{DYNAMIC_LIBRARY_SUFFIX}"),
            },
            None,
            f"libbitsandbytes_cuda128{DYNAMIC_LIBRARY_SUFFIX}",
            True,
        ),
        # ROCm double-digit minor: 7.13 -> 7.2
        (
            CUDASpecs(cuda_version_string="713", highest_compute_capability=(0, 0), cuda_version_tuple=(7, 13)),
            {(7, 2): Path(f"libbitsandbytes_rocm72{DYNAMIC_LIBRARY_SUFFIX}")},
            "7.13.0",
            f"libbitsandbytes_rocm72{DYNAMIC_LIBRARY_SUFFIX}",
            True,
        ),
        # no same-major match: 11.8 with only 12.x -> non-existent exact path, no warning
        (
            CUDASpecs(cuda_version_string="118", highest_compute_capability=(7, 5), cuda_version_tuple=(11, 8)),
            {(12, 1): Path("libbitsandbytes_cuda121.so"), (12, 4): Path("libbitsandbytes_cuda124.so")},
            None,
            f"libbitsandbytes_cuda118{DYNAMIC_LIBRARY_SUFFIX}",
            False,
        ),
        # no libs at all -> non-existent exact path, no warning
        (
            CUDASpecs(cuda_version_string="129", highest_compute_capability=(8, 9), cuda_version_tuple=(12, 9)),
            {},
            None,
            f"libbitsandbytes_cuda129{DYNAMIC_LIBRARY_SUFFIX}",
            False,
        ),
    ],
)
def test_version_selection(monkeypatch, caplog, spec, fake_libs, hip_version, expected_name, expect_warning):
    """Library selection: exact match, fallback, no-same-major, no-libs."""
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    is_hip = spec.cuda_version_tuple[0] < 10
    with (
        patch("torch.version.hip", hip_version if is_hip else None),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value=fake_libs),
    ):
        with caplog.at_level("WARNING"):
            result = get_cuda_bnb_library_path(spec)
    assert result.name == expected_name
    if expect_warning:
        assert caplog.text
    else:
        assert not caplog.text


def test_override(monkeypatch, cuda120_spec, caplog):
    """BNB_CUDA_VERSION overrides path selection."""
    monkeypatch.setenv("BNB_CUDA_VERSION", "110")
    with patch("bitsandbytes.cextension._find_cuda_libs", return_value={}):
        with caplog.at_level("WARNING"):
            result = get_cuda_bnb_library_path(cuda120_spec)
    assert result.stem == "libbitsandbytes_cuda110"
    assert "BNB_CUDA_VERSION" in caplog.text


def test_rocm_override(monkeypatch, rocm70_spec, caplog):
    """BNB_ROCM_VERSION overrides path selection."""
    monkeypatch.setenv("BNB_ROCM_VERSION", "72")
    with (
        patch("torch.version.hip", "7.0.0"),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value={}),
    ):
        with caplog.at_level("WARNING"):
            result = get_cuda_bnb_library_path(rocm70_spec)
    assert result.stem == "libbitsandbytes_rocm72"
    assert "BNB_ROCM_VERSION" in caplog.text


def test_override_invalid_format(monkeypatch, cuda120_spec):
    """Override value must be digits only (e.g. '124'), not dotted or alphanumeric."""
    monkeypatch.setenv("BNB_CUDA_VERSION", "12.4")
    with pytest.raises(RuntimeError, match="digits only"):
        get_cuda_bnb_library_path(cuda120_spec)
