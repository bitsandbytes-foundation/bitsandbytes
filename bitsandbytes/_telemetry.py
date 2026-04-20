# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Anonymous feature-usage telemetry for bitsandbytes.

Sends one HEAD request per distinct feature per process via
`huggingface_hub.utils.send_telemetry()`. Data lands in the Hugging Face
Hub telemetry index under `path_prefix == "/api/telemetry/bitsandbytes/"`
and informs maintenance and deprecation decisions.

What is collected
    - Session fingerprint (once per process, first feature use):
      bnb version, OS name/version, CPU arch, glibc version, Python/torch
      versions, accelerator vendor/name/arch/count.
    - Per-feature events: feature name plus feature-specific metadata
      (e.g. `quant_type="nf4"`, `bits="8"`, `paged="true"`).

What is NOT collected
    Model names, file paths, parameter shapes, user identifiers, training
    data, gradient values, or any value derived from user input.

Automatically disabled when running under pytest (detected via
`pytest` in `sys.modules` or `PYTEST_CURRENT_TEST` env var) so that test
runs in CI and locally do not pollute the real-usage stream.

Opt-out (any of the following env vars disables all telemetry):
    - BNB_DISABLE_TELEMETRY=1           (bitsandbytes only)
    - HF_HUB_DISABLE_TELEMETRY=1        (all HF libraries)
    - HF_HUB_OFFLINE=1                  (all HF libraries)

End-to-end verification:
    Set `BNB_TELEMETRY_TAG=<some-id>` before importing bitsandbytes and the
    value is attached as `bitsandbytes.tag` on every event. Use this to
    correlate a single run's events in ES.

No-ops silently if `huggingface_hub` is not installed, and never raises.

Keys are namespaced under `bitsandbytes.*` in the resulting
`metadata.bitsandbytes.*` fields so they do not collide with fields logged
by other libraries in the shared telemetry index.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_REPORTED: set[str] = set()
_FINGERPRINT: Optional[dict[str, str]] = None

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _is_pytest() -> bool:
    """Detect whether we are running inside a pytest process.

    Telemetry is suppressed during test runs so that CI and local test
    invocations don't pollute the real-usage stream. Tests that want to
    assert on telemetry behavior monkey-patch this function to return False.
    """
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def _is_disabled() -> bool:
    for var in ("BNB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in _TRUTHY:
            return True
    if _is_pytest():
        return True
    return False


def _os_info() -> tuple[str, str]:
    os_name = platform.system()
    os_name = {"Darwin": "macOS"}.get(os_name, os_name)
    if os_name == "Windows":
        try:
            build = sys.getwindowsversion().build
            os_version = f"11 (build {build})" if build >= 22000 else f"10 (build {build})"
        except Exception:
            os_version = platform.release()
    elif os_name == "macOS":
        os_version = platform.mac_ver()[0] or platform.release()
    else:
        os_version = platform.release()
    return os_name, os_version


def _accel_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        import torch
    except ImportError:
        info["bitsandbytes.accel"] = "unknown"
        return info

    try:
        if torch.cuda.is_available():
            vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
            info["bitsandbytes.accel"] = vendor
            info["bitsandbytes.accel_count"] = str(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            info["bitsandbytes.accel_name"] = props.name
            if vendor == "nvidia":
                info["bitsandbytes.accel_arch"] = f"sm_{props.major}{props.minor}"
            else:
                info["bitsandbytes.accel_arch"] = getattr(props, "gcnArchName", "unknown")
            return info

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["bitsandbytes.accel"] = "xpu"
            info["bitsandbytes.accel_count"] = str(torch.xpu.device_count())
            try:
                info["bitsandbytes.accel_name"] = torch.xpu.get_device_properties(0).name
            except Exception:
                pass
            return info

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["bitsandbytes.accel"] = "mps"
            return info

        if hasattr(torch, "hpu") and torch.hpu.is_available():
            info["bitsandbytes.accel"] = "hpu"
            return info
    except Exception:
        pass

    info["bitsandbytes.accel"] = "cpu"
    return info


def _fingerprint() -> dict[str, str]:
    global _FINGERPRINT
    if _FINGERPRINT is not None:
        return _FINGERPRINT

    try:
        import bitsandbytes

        version = bitsandbytes.__version__
    except Exception:
        version = "unknown"

    os_name, os_version = _os_info()
    info = {
        "bitsandbytes.version": version,
        "bitsandbytes.os": os_name,
        "bitsandbytes.os_version": os_version,
        "bitsandbytes.arch": platform.machine(),
        "bitsandbytes.python": platform.python_version(),
    }
    if os_name == "Linux":
        try:
            libc_name, libc_ver = platform.libc_ver()
            if libc_name:
                info["bitsandbytes.libc"] = f"{libc_name}-{libc_ver}"
        except Exception:
            pass
    try:
        import torch

        info["bitsandbytes.torch"] = torch.__version__
    except ImportError:
        pass

    info.update(_accel_info())

    _FINGERPRINT = info
    return info


def report_feature(feature: str, details: Optional[dict[str, object]] = None) -> None:
    """Report that a bitsandbytes feature was used.

    Fires at most once per `feature` per process. Subsequent calls with the
    same `feature` are O(1) no-ops.

    Args:
        feature: Short feature name. Becomes the final URL path segment:
            `/api/telemetry/bitsandbytes/{feature}` (so it appears as
            `path_filename` in ES queries).
        details: Optional feature-specific key/value metadata. Keys without a
            `bitsandbytes.` prefix are prefixed automatically.
    """
    if feature in _REPORTED:
        return
    _REPORTED.add(feature)

    if _is_disabled():
        return

    try:
        from huggingface_hub.utils import send_telemetry
    except ImportError:
        return

    fingerprint = _fingerprint()
    user_agent = dict(fingerprint)
    user_agent["bitsandbytes.feature"] = feature
    if details:
        for k, v in details.items():
            key = k if k.startswith("bitsandbytes.") else f"bitsandbytes.{k}"
            user_agent[key] = str(v)

    tag = os.environ.get("BNB_TELEMETRY_TAG", "").strip()
    if tag:
        user_agent["bitsandbytes.tag"] = tag

    try:
        send_telemetry(
            topic=f"bitsandbytes/{feature}",
            library_name="bitsandbytes",
            library_version=fingerprint.get("bitsandbytes.version", "unknown"),
            user_agent=user_agent,
        )
    except Exception as e:
        logger.debug("bitsandbytes telemetry send failed: %s", e)


def _reset_for_testing() -> None:
    """Clear module state. Intended for use in test fixtures only."""
    global _FINGERPRINT
    _REPORTED.clear()
    _FINGERPRINT = None
