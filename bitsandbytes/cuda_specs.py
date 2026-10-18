import dataclasses
from functools import lru_cache
import logging
from typing import Optional

import torch


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: tuple[int, int]

    @property
    def has_imma(self) -> bool:
        if torch.version.hip:
            # ROCm: int8 matrix cores (MFMA) start at CDNA1 (gfx908). The GCN5 parts
            # (gfx900/gfx906/gfx90c) have none. This previously returned
            # `torch.version.hip` -- a version *string* -- which is always truthy, so
            # every AMD GPU was reported as IMMA-capable.
            return not get_rocm_gpu_arch().startswith(("gfx900", "gfx906", "gfx90c"))
        return self.highest_compute_capability >= (7, 5)


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


def get_rocm_gpu_arch(device_index: int = 0) -> str:
    """Get the gfx architecture of a ROCm device, e.g. 'gfx1201'.

    Reads the device property torch already exposes rather than shelling out to
    hipinfo.exe/rocminfo: the subprocess is not on PATH for pip-installed ROCm SDKs
    (TheRock), its output format differs per platform, and the result is available
    directly from the runtime.
    """
    logger = logging.getLogger(__name__)
    if not torch.version.hip or not torch.cuda.is_available():
        return "unknown"
    try:
        # gcnArchName carries feature suffixes, e.g. 'gfx90a:sramecc+:xnack-'.
        name = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "") or ""
        return name.split(":")[0] or "unknown"
    except Exception as e:
        logger.error(f"Could not detect ROCm GPU architecture: {e}")
        return "unknown"
