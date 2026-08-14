from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX
from bitsandbytes.cuda_specs import CUDASpecs


def specs(version: tuple[int, int]) -> CUDASpecs:
    return CUDASpecs(
        cuda_version_string=f"{version[0]}{version[1]}",
        highest_compute_capability=(0, 0),
        cuda_version_tuple=version,
    )


@pytest.mark.parametrize(
    "backend,backend_version,runtime_version,available,expected,warning",
    [
        # Exact match.
        ("cuda", "12.4", (12, 4), [(12, 4)], (12, 4), False),
        # Same-major fallback to the newest older binary.
        ("cuda", "12.9", (12, 9), [(12, 4), (12, 8)], (12, 8), True),
        # Same-major fallback to the oldest newer binary.
        ("cuda", "12.0", (12, 0), [(12, 1), (12, 4)], (12, 1), True),
        # CUDA does not fall back across major versions.
        ("cuda", "11.8", (11, 8), [(12, 1)], None, False),
        # ROCm same-major fallback with a double-digit minor.
        ("hip", "7.13.0", (7, 13), [(7, 2), (7, 14)], (7, 2), True),
        # ROCm same-major fallback to the newest older binary.
        ("hip", "7.9.0", (7, 9), [(7, 2), (7, 14)], (7, 2), True),
        # ROCm/HIP version-line divergence with an older cross-major fallback.
        ("hip", "7.16.0", (12, 1), [(8, 0), (7, 14)], (8, 0), True),
        # ROCm cross-major fallback to the newest older binary.
        ("hip", "8.0.0", (8, 0), [(7, 14)], (7, 14), True),
        # ROCm cross-major fallback to the oldest newer binary.
        ("hip", "6.4.0", (6, 4), [(7, 0)], (7, 0), True),
        # No packaged libraries returns the requested path without a warning.
        ("hip", "7.14.0", (7, 14), [], None, False),
    ],
)
def test_library_selection(
    backend,
    backend_version,
    runtime_version,
    available,
    expected,
    warning,
    monkeypatch,
    caplog,
):
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    other_backend = "cuda" if backend == "hip" else "hip"
    prefix = "rocm" if backend == "hip" else "cuda"
    paths = {
        version: Path(f"libbitsandbytes_{prefix}{version[0]}{version[1]}{DYNAMIC_LIBRARY_SUFFIX}")
        for version in available
    }
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value=paths),
        caplog.at_level("WARNING"),
    ):
        result = get_cuda_bnb_library_path(specs(runtime_version))

    if expected is None:
        tag = f"{runtime_version[0]}{runtime_version[1]}"
        assert result.name == f"libbitsandbytes_{prefix}{tag}{DYNAMIC_LIBRARY_SUFFIX}"
    else:
        assert result == paths[expected]
    assert bool(caplog.text) is warning


@pytest.mark.parametrize(
    "backend,backend_version,version,override,expected_stem",
    [
        ("hip", "7.0.0", (7, 0), "72", "libbitsandbytes_rocm72"),
        ("hip", "7.0.0", (7, 0), "7.2", "libbitsandbytes_rocm72"),
        ("cuda", "12.0", (12, 0), "128", "libbitsandbytes_cuda128"),
        ("cuda", "12.0", (12, 0), "12.8", "libbitsandbytes_cuda128"),
        ("cuda", "12.0", (12, 0), "12.8.1", "libbitsandbytes_cuda128"),
    ],
)
def test_override_formats(monkeypatch, backend, backend_version, version, override, expected_stem):
    other_backend = "cuda" if backend == "hip" else "hip"
    override_var = "BNB_ROCM_VERSION" if backend == "hip" else "BNB_CUDA_VERSION"
    other_var = "BNB_CUDA_VERSION" if backend == "hip" else "BNB_ROCM_VERSION"
    monkeypatch.setenv(override_var, override)
    monkeypatch.delenv(other_var, raising=False)
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
    ):
        assert get_cuda_bnb_library_path(specs(version)).stem == expected_stem


@pytest.mark.parametrize(
    "backend,backend_version,version",
    [
        ("cuda", "12.0", (12, 0)),
        ("hip", "7.2.0", (7, 2)),
    ],
)
def test_opposite_backend_override_warns(monkeypatch, caplog, backend, backend_version, version):
    other_backend = "hip" if backend == "cuda" else "cuda"
    correct_var = "BNB_CUDA_VERSION" if backend == "cuda" else "BNB_ROCM_VERSION"
    wrong_var = "BNB_ROCM_VERSION" if backend == "cuda" else "BNB_CUDA_VERSION"
    monkeypatch.setenv(wrong_var, "72")
    monkeypatch.delenv(correct_var, raising=False)
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value={}),
        caplog.at_level("WARNING"),
    ):
        get_cuda_bnb_library_path(specs(version))
    assert f"{wrong_var} is ignored" in caplog.text
    assert f"use {correct_var} instead" in caplog.text


@pytest.mark.parametrize(
    "backend,backend_version,version",
    [
        ("hip", "7.0.0", (7, 0)),
        ("cuda", "12.0", (12, 0)),
    ],
)
def test_invalid_override(monkeypatch, backend, backend_version, version):
    other_backend = "cuda" if backend == "hip" else "hip"
    override_var = "BNB_ROCM_VERSION" if backend == "hip" else "BNB_CUDA_VERSION"
    other_var = "BNB_CUDA_VERSION" if backend == "hip" else "BNB_ROCM_VERSION"
    monkeypatch.setenv(override_var, "not-a-version")
    monkeypatch.delenv(other_var, raising=False)
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        pytest.raises(RuntimeError, match="dotted version"),
    ):
        get_cuda_bnb_library_path(specs(version))
