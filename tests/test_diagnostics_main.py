from importlib import metadata as importlib_metadata

from bitsandbytes.diagnostics import main


def test_get_package_version_returns_installed_version(monkeypatch):
    monkeypatch.setattr(main.importlib_metadata, "version", lambda name: "1.2.3")

    assert main.get_package_version("pip") == "1.2.3"


def test_get_package_version_returns_not_found_for_missing_package(monkeypatch):
    def raise_package_not_found(name):
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(main.importlib_metadata, "version", raise_package_not_found)

    assert main.get_package_version("missing-package") == "not found"
