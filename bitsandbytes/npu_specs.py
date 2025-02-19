import dataclasses

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


@dataclasses.dataclass(frozen=True)
class NPUSpecs:
    cann_version_string: str


def get_npu_specs():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return NPUSpecs(cann_version_string=torch.version.cann)
    else:
        return None
