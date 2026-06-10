import dataclasses
from functools import lru_cache
import logging
import os
import platform
import re
import subprocess
import sys
from typing import Optional

import torch


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: tuple[int, int]

    @property
    def has_imma(self) -> bool:
        return torch.version.hip or self.highest_compute_capability >= (7, 5)


def get_compute_capabilities() -> list[tuple[int, int]]:
    return sorted(torch.cuda.get_device_capability(torch.cuda.device(i)) for i in range(torch.cuda.device_count()))


@lru_cache(None)
def get_cuda_version_tuple() -> Optional[tuple[int, int]]:
    """Get CUDA/HIP version as a tuple of (major, minor)."""
    try:
        if torch.version.cuda:
            version_str = torch.version.cuda
        elif torch.version.hip:
            version_str = torch.version.hip
        else:
            return None

        parts = version_str.split(".")
        if len(parts) >= 2:
            return tuple(map(int, parts[:2]))
        return None
    except (AttributeError, ValueError, IndexError):
        return None


def get_cuda_version_string() -> Optional[str]:
    """Get CUDA/HIP version as a string."""
    version_tuple = get_cuda_version_tuple()
    if version_tuple is None:
        return None
    major, minor = version_tuple
    return f"{major}{minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    """Get CUDA/HIP specifications."""
    if not torch.cuda.is_available():
        return None

    try:
        compute_capabilities = get_compute_capabilities()
        if not compute_capabilities:
            return None

        version_tuple = get_cuda_version_tuple()
        if version_tuple is None:
            return None

        version_string = get_cuda_version_string()
        if version_string is None:
            return None

        return CUDASpecs(
            highest_compute_capability=compute_capabilities[-1],
            cuda_version_string=version_string,
            cuda_version_tuple=version_tuple,
        )
    except Exception:
        return None


def get_rocm_gpu_arch() -> str:
    """Get ROCm GPU architecture."""
    logger = logging.getLogger(__name__)

    if not torch.version.hip:
        return "unknown"

    # Prefer the architecture torch already knows; this needs no subprocess.
    if torch.cuda.is_available():
        try:
            # gcnArchName may include feature flags, e.g. "gfx90a:sramecc+:xnack-".
            return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        except Exception as e:
            logger.debug(f"Could not get ROCm GPU architecture from torch: {e}")

    # Fall back to parsing tool output. On Windows, use hipInfo.exe; on Linux, use rocminfo.
    if platform.system() == "Windows":
        # hipInfo.exe is usually not on PATH: the HIP SDK does not add its bin directory,
        # and AMD's PyTorch wheels for Windows ship hipInfo.exe next to python.exe instead.
        cmds = [
            ["hipinfo.exe"],
            [os.path.join(os.path.dirname(sys.executable), "hipInfo.exe")],
        ]
        arch_pattern = r"gcnArchName:\s+gfx([a-zA-Z\d]+)"
    else:
        cmds = [["rocminfo"]]
        arch_pattern = r"Name:\s+gfx([a-zA-Z\d]+)"

    last_error: Optional[Exception] = None
    for cmd in cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            last_error = e
            continue
        match = re.search(arch_pattern, result.stdout)
        if match:
            return "gfx" + match.group(1)

    if last_error is not None:
        logger.error(f"Could not detect ROCm GPU architecture: {last_error}")
        if torch.cuda.is_available():
            logger.warning(
                """
ROCm GPU architecture detection failed despite ROCm being available.
                """,
            )
    return "unknown"
