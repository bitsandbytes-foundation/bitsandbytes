import logging
from types import SimpleNamespace

from bitsandbytes import cuda_specs


def _mock_rocm_env(monkeypatch):
    monkeypatch.setattr(cuda_specs.torch.version, "hip", "7.0.2", raising=False)
    monkeypatch.setattr(cuda_specs.platform, "system", lambda: "Linux")


def test_get_rocm_gpu_arch_prefers_detected_value_over_override(monkeypatch):
    _mock_rocm_env(monkeypatch)
    monkeypatch.setenv("BNB_ROCM_GPU_ARCH", "gfx90a")
    monkeypatch.setattr(cuda_specs.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="Name: gfx1100"))

    assert cuda_specs.get_rocm_gpu_arch() == "gfx1100"


def test_get_rocm_gpu_arch_uses_override_after_detection_failure(monkeypatch, caplog):
    _mock_rocm_env(monkeypatch)
    monkeypatch.setenv("BNB_ROCM_GPU_ARCH", "1100")
    monkeypatch.setattr(cuda_specs.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="no match"))

    with caplog.at_level(logging.INFO, logger="bitsandbytes.cuda_specs"):
        assert cuda_specs.get_rocm_gpu_arch() == "gfx1100"

    assert "Install rocminfo if possible" in caplog.text
    assert "BNB_ROCM_GPU_ARCH" in caplog.text


def test_get_rocm_gpu_arch_rejects_invalid_override(monkeypatch, caplog):
    _mock_rocm_env(monkeypatch)
    monkeypatch.setenv("BNB_ROCM_GPU_ARCH", "gfx11-00")
    monkeypatch.setattr(cuda_specs.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="no match"))

    with caplog.at_level(logging.WARNING, logger="bitsandbytes.cuda_specs"):
        assert cuda_specs.get_rocm_gpu_arch() == "unknown"

    assert "Ignoring invalid BNB_ROCM_GPU_ARCH value" in caplog.text
