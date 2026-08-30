import logging
import sys
from types import ModuleType, SimpleNamespace

import bitsandbytes.backends.cpu.ops as cpu_ops


def test_load_gemm_4bit_forward_kernel_requests_cpu_backend(monkeypatch):
    kernels = ModuleType("kernels")
    sentinel = object()
    captured = {}

    def fake_get_kernel(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return SimpleNamespace(gemm_4bit_forward=sentinel)

    kernels.get_kernel = fake_get_kernel
    monkeypatch.setitem(sys.modules, "kernels", kernels)

    kernel = cpu_ops._load_gemm_4bit_forward_kernel()

    assert kernel is sentinel
    assert captured == {
        "repo_id": "kernels-community/quantization-bitsandbytes",
        "kwargs": {"version": 1, "backend": "cpu"},
    }


def test_load_gemm_4bit_forward_kernel_logs_and_falls_back(monkeypatch, caplog):
    kernels = ModuleType("kernels")

    def fake_get_kernel(*args, **kwargs):
        raise FileNotFoundError("missing cpu variant")

    kernels.get_kernel = fake_get_kernel
    monkeypatch.setitem(sys.modules, "kernels", kernels)

    with caplog.at_level(logging.WARNING, logger=cpu_ops.__name__):
        kernel = cpu_ops._load_gemm_4bit_forward_kernel()

    assert kernel is None
    assert "Failed to load CPU gemm_4bit_forward from kernels-community" in caplog.text
    assert "missing cpu variant" in caplog.text
