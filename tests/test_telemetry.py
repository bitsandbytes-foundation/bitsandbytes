# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import sys
from unittest import mock

import pytest

from bitsandbytes import _telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry_state(monkeypatch):
    """Clear telemetry dedup state + opt-out env vars before each test.

    Also bypasses the pytest auto-detection in _is_disabled so the telemetry
    code path is actually exercised. Individual tests that want the
    pytest-disabled behavior override this by restoring _is_pytest.
    """
    for var in ("BNB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE", "BNB_TELEMETRY_TAG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_telemetry, "_is_pytest", lambda: False)
    _telemetry._reset_for_testing()
    yield
    _telemetry._reset_for_testing()


@pytest.fixture
def fake_send():
    """Replace huggingface_hub.utils.send_telemetry with a recording mock."""
    with mock.patch("huggingface_hub.utils.send_telemetry") as m:
        yield m


def test_report_feature_fires_once_per_feature(fake_send):
    for _ in range(10):
        _telemetry.report_feature("linear_4bit", {"quant_type": "nf4"})
    assert fake_send.call_count == 1


def test_distinct_features_each_fire(fake_send):
    _telemetry.report_feature("linear_4bit", {"quant_type": "nf4"})
    _telemetry.report_feature("linear_8bit", {"threshold": 6.0})
    _telemetry.report_feature("optimizer", {"name": "AdamW8bit"})
    assert fake_send.call_count == 3


def test_payload_is_namespaced_under_bitsandbytes(fake_send):
    _telemetry.report_feature("linear_4bit", {"quant_type": "nf4", "blocksize": 64})
    kwargs = fake_send.call_args.kwargs
    assert kwargs["topic"] == "bitsandbytes/linear_4bit"
    assert kwargs["library_name"] == "bitsandbytes"
    ua = kwargs["user_agent"]
    assert ua["bitsandbytes.feature"] == "linear_4bit"
    assert ua["bitsandbytes.quant_type"] == "nf4"
    assert ua["bitsandbytes.blocksize"] == "64"
    # No bare keys leaked
    assert not any(k == "quant_type" or k == "blocksize" for k in ua)


def test_fingerprint_fields_present(fake_send):
    _telemetry.report_feature("linear_4bit")
    ua = fake_send.call_args.kwargs["user_agent"]
    assert "bitsandbytes.version" in ua
    assert "bitsandbytes.os" in ua
    assert "bitsandbytes.arch" in ua
    assert "bitsandbytes.python" in ua
    assert "bitsandbytes.accel" in ua


def test_values_are_stringified(fake_send):
    _telemetry.report_feature("x", {"bits": 8, "paged": True, "blocksize": 64})
    ua = fake_send.call_args.kwargs["user_agent"]
    assert ua["bitsandbytes.bits"] == "8"
    assert ua["bitsandbytes.paged"] == "True"
    assert ua["bitsandbytes.blocksize"] == "64"


def test_already_prefixed_keys_not_double_prefixed(fake_send):
    _telemetry.report_feature("x", {"bitsandbytes.custom": "v"})
    ua = fake_send.call_args.kwargs["user_agent"]
    assert ua["bitsandbytes.custom"] == "v"
    assert "bitsandbytes.bitsandbytes.custom" not in ua


@pytest.mark.parametrize(
    "env_var",
    ["BNB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE"],
)
def test_opt_out_env_vars(fake_send, monkeypatch, env_var):
    monkeypatch.setenv(env_var, "1")
    _telemetry.report_feature("linear_4bit")
    fake_send.assert_not_called()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_accepts_truthy_values(fake_send, monkeypatch, value):
    monkeypatch.setenv("BNB_DISABLE_TELEMETRY", value)
    _telemetry.report_feature("linear_4bit")
    fake_send.assert_not_called()


def test_opt_out_blank_does_not_disable(fake_send, monkeypatch):
    monkeypatch.setenv("BNB_DISABLE_TELEMETRY", "")
    _telemetry.report_feature("linear_4bit")
    assert fake_send.call_count == 1


def test_telemetry_tag_attached_when_set(fake_send, monkeypatch):
    monkeypatch.setenv("BNB_TELEMETRY_TAG", "verify-abc123")
    _telemetry.report_feature("linear_4bit")
    ua = fake_send.call_args.kwargs["user_agent"]
    assert ua["bitsandbytes.tag"] == "verify-abc123"


def test_no_tag_attached_by_default(fake_send):
    _telemetry.report_feature("linear_4bit")
    ua = fake_send.call_args.kwargs["user_agent"]
    assert "bitsandbytes.tag" not in ua


def test_graceful_when_huggingface_hub_missing(monkeypatch):
    """If huggingface_hub is unavailable, report_feature must no-op silently."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub.utils" or name.startswith("huggingface_hub"):
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Ensure any cached submodule is gone so the patched __import__ is consulted
    for mod in list(sys.modules):
        if mod.startswith("huggingface_hub"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Should not raise
    _telemetry.report_feature("linear_4bit")


def test_send_exception_is_swallowed(monkeypatch):
    """A failing send_telemetry must not propagate."""
    with mock.patch("huggingface_hub.utils.send_telemetry", side_effect=RuntimeError("boom")):
        _telemetry.report_feature("linear_4bit")  # must not raise


def test_dedup_persists_even_when_disabled(fake_send, monkeypatch):
    """Disabled calls still burn the dedup slot — otherwise toggling the env
    var mid-process could send a duplicate event for the same feature."""
    monkeypatch.setenv("BNB_DISABLE_TELEMETRY", "1")
    _telemetry.report_feature("linear_4bit")
    fake_send.assert_not_called()

    monkeypatch.delenv("BNB_DISABLE_TELEMETRY")
    _telemetry.report_feature("linear_4bit")
    fake_send.assert_not_called()


def test_pytest_auto_detection_disables_telemetry(fake_send, monkeypatch):
    """Under a real pytest process _is_pytest() returns True and suppresses
    telemetry so tests don't pollute the real-usage stream."""
    monkeypatch.setattr(_telemetry, "_is_pytest", lambda: True)
    _telemetry.report_feature("linear_4bit")
    fake_send.assert_not_called()


def test_is_pytest_detects_current_process(monkeypatch):
    """Sanity check: the helper reports True when we really are in pytest.

    The autouse fixture monkey-patches _is_pytest; `undo()` restores the real
    implementation so we can verify its actual behavior.
    """
    monkeypatch.undo()
    assert _telemetry._is_pytest()
