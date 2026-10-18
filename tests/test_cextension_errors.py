"""Regression tests for the native-library loader's error classification.

These cover failures that only ever surfaced on Windows + ROCm:

* the "no compatible binary" message is classified by a substring that the loader
  never actually raised, so users were told to build a CPU-only library;
* a missing dependency on Windows is reported as ``[WinError 126]``, not as a
  ``dlerror()`` string, so it never reached the dependency guidance;
* that guidance only ever described ``LD_LIBRARY_PATH`` and ``.bashrc``;
* ``os.add_dll_directory`` was never called, and ctypes does not search ``PATH``;
* ``has_imma`` returned the HIP version *string* rather than a bool;
* ``BNB_ROCM_VERSION=7.2`` (the natural spelling) was a hard error.

Everything here is platform-independent: no GPU and no ROCm toolchain required.
"""

from __future__ import annotations

import pytest

from bitsandbytes import cextension
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX
from bitsandbytes.cuda_specs import CUDASpecs


def _spec() -> CUDASpecs:
    """A spec that will not match any packaged library, so resolution is exercised."""
    return CUDASpecs(
        cuda_version_string="72",
        highest_compute_capability=(12, 1),
        cuda_version_tuple=(7, 2),
    )


def _override_var() -> str:
    return "BNB_ROCM_VERSION" if cextension.HIP_ENVIRONMENT else "BNB_CUDA_VERSION"


# --------------------------------------------------------------------------------------
# Missing-dependency detection
# --------------------------------------------------------------------------------------


def test_windows_module_not_found_is_a_missing_dependency():
    assert cextension._is_missing_dependency_error("[WinError 126] The specified module could not be found")


def test_unrelated_error_is_not_a_missing_dependency():
    assert not cextension._is_missing_dependency_error("something else went wrong")


def test_dependency_hint_is_platform_specific(monkeypatch):
    monkeypatch.setattr(cextension.os, "name", "nt")
    windows_hint = cextension._dependency_search_path_hint("ROCm", "7")
    assert "DLL search path" in windows_hint
    assert "ROCM_PATH" in windows_hint
    assert "LD_LIBRARY_PATH" not in windows_hint

    monkeypatch.setattr(cextension.os, "name", "posix")
    monkeypatch.setattr(cextension, "HIP_ENVIRONMENT", True)
    posix_hint = cextension._dependency_search_path_hint("ROCm", "7")
    assert "LD_LIBRARY_PATH" in posix_hint
    assert "ROCM_PATH" not in posix_hint


# --------------------------------------------------------------------------------------
# "No compatible binary" must reach the branch that explains it
# --------------------------------------------------------------------------------------


def test_missing_binary_raises_the_message_the_formatter_keys_on(monkeypatch, tmp_path):
    monkeypatch.setattr(cextension, "get_cuda_specs", _spec)
    monkeypatch.setattr(cextension, "PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(cextension, "_find_cuda_libs", lambda prefix, is_hip: {})

    with pytest.raises(RuntimeError) as excinfo:
        cextension.get_native_library()

    # This is the exact coupling that was broken: the formatter matched
    # "{backend} binary not found" while the loader raised "{backend} binary found at".
    assert f"{cextension.BNB_BACKEND} binary found at" in str(excinfo.value)


def test_missing_binary_does_not_advise_a_cpu_only_build():
    message = (
        f"Original error: No compatible {cextension.BNB_BACKEND} binary found at "
        f"C:/pkg/libbitsandbytes_rocm72{DYNAMIC_LIBRARY_SUFFIX}\n"
    )
    formatted = cextension.ErrorHandlerMockBNBNativeLibrary(message).formatted_error
    assert "COMPILE FROM SOURCE for CPU-only" not in formatted


def test_cpu_library_probe_uses_the_platform_suffix():
    # The old probe hardcoded ".so", so it never matched on Windows. Exercise the
    # formatter directly: a real CPU-load failure also contains a dlerror() string,
    # which routes to the dependency formatter before this branch is reached.
    lib = cextension.ErrorHandlerMockBNBNativeLibrary("boom")
    formatted = lib._format_lib_error_message(
        available_versions=[],
        user_cuda_version="7.2",
        original_error=f"Original error: libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}: failed to load\n",
    )
    assert "Failed to load CPU-only bitsandbytes library" in formatted


# --------------------------------------------------------------------------------------
# Windows DLL search path
# --------------------------------------------------------------------------------------


def test_dll_registration_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(cextension, "_DLL_DIR_HANDLES", [])
    monkeypatch.setattr(cextension.os, "name", "posix")
    cextension._register_windows_dll_directories()
    assert cextension._DLL_DIR_HANDLES == []


def test_dll_registration_finds_the_rocm_runtime(monkeypatch, tmp_path):
    rocm_root = tmp_path / "rocm"
    (rocm_root / "bin").mkdir(parents=True, exist_ok=True)
    (rocm_root / "bin" / "amdhip64_7.dll").write_bytes(b"")

    registered: list[str] = []
    monkeypatch.setattr(cextension, "_DLL_DIR_HANDLES", [])
    monkeypatch.setattr(cextension.os, "name", "nt")
    monkeypatch.setattr(
        cextension.os, "add_dll_directory", lambda path: registered.append(path) or object(), raising=False
    )
    monkeypatch.setattr(cextension.torch, "__file__", str(tmp_path / "torch" / "__init__.py"), raising=False)
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    for var in ("HIP_PATH", "ROCM_HOME", "CONDA_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", "")

    cextension._register_windows_dll_directories()

    assert registered == [str(rocm_root / "bin")]
    assert len(cextension._DLL_DIR_HANDLES) == 1


def test_dll_registration_skips_directories_without_a_runtime(monkeypatch, tmp_path):
    (tmp_path / "bin").mkdir(exist_ok=True)
    registered: list[str] = []
    monkeypatch.setattr(cextension, "_DLL_DIR_HANDLES", [])
    monkeypatch.setattr(cextension.os, "name", "nt")
    monkeypatch.setattr(
        cextension.os, "add_dll_directory", lambda path: registered.append(path) or object(), raising=False
    )
    monkeypatch.setattr(cextension.torch, "__file__", str(tmp_path / "torch" / "__init__.py"), raising=False)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    for var in ("HIP_PATH", "ROCM_HOME", "CONDA_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", "")

    cextension._register_windows_dll_directories()

    assert registered == []


# --------------------------------------------------------------------------------------
# BNB_ROCM_VERSION / BNB_CUDA_VERSION spelling
# --------------------------------------------------------------------------------------


def test_dotted_override_is_accepted(monkeypatch):
    monkeypatch.setenv(_override_var(), "7.2")
    path = cextension.get_cuda_bnb_library_path(_spec())
    prefix = "rocm" if cextension.HIP_ENVIRONMENT else "cuda"
    assert path.name == f"libbitsandbytes_{prefix}72{DYNAMIC_LIBRARY_SUFFIX}"


def test_compact_override_is_still_accepted(monkeypatch):
    monkeypatch.setenv(_override_var(), "72")
    path = cextension.get_cuda_bnb_library_path(_spec())
    prefix = "rocm" if cextension.HIP_ENVIRONMENT else "cuda"
    assert path.name == f"libbitsandbytes_{prefix}72{DYNAMIC_LIBRARY_SUFFIX}"


def test_garbage_override_still_raises(monkeypatch):
    monkeypatch.setenv(_override_var(), "seventy")
    with pytest.raises(RuntimeError, match="digits only"):
        cextension.get_cuda_bnb_library_path(_spec())


# --------------------------------------------------------------------------------------
# ROCm capability detection
# --------------------------------------------------------------------------------------


class _FakeProps:
    gcnArchName = "gfx90a:sramecc+:xnack-"


def test_rocm_arch_uses_device_properties(monkeypatch):
    from bitsandbytes import cuda_specs

    monkeypatch.setattr(cuda_specs.torch.version, "hip", "7.2.0", raising=False)
    monkeypatch.setattr(cuda_specs.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_specs.torch.cuda, "get_device_properties", lambda index=0: _FakeProps())
    assert cuda_specs.get_rocm_gpu_arch() == "gfx90a"


def test_rocm_arch_is_unknown_without_hip(monkeypatch):
    from bitsandbytes import cuda_specs

    monkeypatch.setattr(cuda_specs.torch.version, "hip", None, raising=False)
    assert cuda_specs.get_rocm_gpu_arch() == "unknown"


@pytest.mark.parametrize("arch,expected", [("gfx1201", True), ("gfx942", True), ("gfx906", False), ("gfx900", False)])
def test_has_imma_returns_a_bool(monkeypatch, arch, expected):
    from bitsandbytes import cuda_specs

    monkeypatch.setattr(cuda_specs.torch.version, "hip", "7.2.0", raising=False)
    monkeypatch.setattr(cuda_specs, "get_rocm_gpu_arch", lambda *args, **kwargs: arch)

    result = _spec().has_imma

    assert isinstance(result, bool), "has_imma must not leak the HIP version string"
    assert result is expected
