"""Dispatcher shimming the editable layout.

When this repository is used via ``pip install -e .`` the real Python
package lives under ``bitsandbytes/bitsandbytes``.  Importing from the
workspace root (e.g. running scripts from ``.../ai/kernels``) would
otherwise resolve to this outer directory, yielding a namespace module
with no attributes.  Import the inner package eagerly and mirror its
symbols so ``import bitsandbytes`` always behaves the same as the
installed wheel.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_inner: ModuleType = importlib.import_module(".bitsandbytes", __name__)

# Copy dunder metadata expected by consumers.
for _name in ("__all__", "__doc__", "__file__", "__loader__", "__path__", "__spec__", "__version__"):
    if hasattr(_inner, _name):
        globals()[_name] = getattr(_inner, _name)

# Re-export public symbols while leaving dunders alone.
for _name, _value in vars(_inner).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

del _inner, _name, _value, ModuleType, importlib

