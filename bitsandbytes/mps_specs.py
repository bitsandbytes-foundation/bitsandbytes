
from dataclasses import dataclass
from typing import Optional

import platform
import torch


@dataclass(frozen=True)
class MPSSpecs:
    macos_version_string: Optional[str]


def get_mps_specs():
    macos_version_string = None
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if 'macOS' in platform.platform():
            macos_version_string = platform.mac_ver()[0]
        return MPSSpecs(macos_version_string=macos_version_string)
    return None