import dataclasses
from functools import lru_cache
import logging
import re
import subprocess
import platform
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
    return f"{major * 10 + minor}"


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
    try:
        if torch.version.hip:
            rocminfo_process_name = "rocminfo"
            search_pattern = r"Name:\s+gfx([a-zA-Z\d]+)"
            if platform.system() == "Windows":  
                rocminfo_process_name = "hipinfo"
                search_pattern = r"Name:\s*gfx([a-zA-Z\d]+)"
            result = subprocess.run([rocminfo_process_name], capture_output=True, text=True)
            match = re.search(search_pattern, result.stdout)
            if match:
                print("BNB: gfx"+ match.group(1))
                return "gfx" + match.group(1)
            else:
                print("BNB: gfx unknown")
                return "unknown"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Could not detect ROCm GPU architecture: {e}")
        if torch.cuda.is_available():
            logger.warning(
                """
ROCm GPU architecture detection failed despite ROCm being available.
                """,
            )
        return "unknown"


def get_rocm_warpsize() -> int:
    """Get ROCm warp size."""
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            rocminfo_process_name = "rocminfo"
            search_pattern = r"Wavefront Size:\s+([0-9]{2})\(0x[0-9]{2}\)"
            if platform.system() == "Windows":  
                rocminfo_process_name = "hipinfo"
                search_pattern = r"warpSize:\s*(\d+)"
            result = subprocess.run([rocminfo_process_name], capture_output=True, text=True)
            match = re.search(search_pattern, result.stdout)
            if match:
                print(f"warpSize: {int(match.group(1))}")
                return int(match.group(1))
            else:
                # default to 64 to be safe
                print(f"warpSize: 64")
                return 64
        else:
            # nvidia cards always use 32 warp size
            return 32
    except Exception as e:
        logger.error(f"Could not detect ROCm warp size: {e}. Defaulting to 64. (some 4-bit functions may not work!)")
        if torch.cuda.is_available():
            logger.warning(
                """
ROCm warp size detection failed despite ROCm being available.
                """,
            )
        return 64
