"""
Regression tests for the optional Gaudi plugin version lookup.

``get_gaudi_sw_version`` runs at import time (``GAUDI_SW_VER``), so a metadata
payload without a ``Version`` key must resolve to ``None`` instead of raising.
"""

from importlib.metadata import PackageNotFoundError

from packaging import version
import pytest

from bitsandbytes.backends import utils


def _metadata_returning(mapping):
    def _get(_name):
        return mapping

    return _get


def test_version_is_none_when_package_is_missing(monkeypatch):
    def missing(_name):
        raise PackageNotFoundError("habana-torch-plugin")

    monkeypatch.setattr(utils, "metadata", missing)
    assert utils.get_gaudi_sw_version() is None


@pytest.mark.parametrize(
    "package_metadata",
    [
        {},
        {"Name": "habana-torch-plugin"},
        {"Version": ""},
    ],
)
def test_version_is_none_when_metadata_has_no_version(monkeypatch, package_metadata):
    monkeypatch.setattr(utils, "metadata", _metadata_returning(package_metadata))
    assert utils.get_gaudi_sw_version() is None


def test_version_is_parsed_from_metadata(monkeypatch):
    monkeypatch.setattr(utils, "metadata", _metadata_returning({"Version": "1.21.0"}))
    assert utils.get_gaudi_sw_version() == version.parse("1.21.0")
