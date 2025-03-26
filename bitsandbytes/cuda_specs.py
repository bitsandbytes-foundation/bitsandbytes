import dataclasses
from functools import lru_cache
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
def get_cuda_version_tuple() -> tuple[int, int]:
    if torch.version.cuda:
        return map(int, torch.version.cuda.split(".")[0:2])
    elif torch.version.hip:
        return map(int, torch.version.hip.split(".")[0:2])

    return None


def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f"{major * 10 + minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    if not torch.cuda.is_available():
        return None

    return CUDASpecs(
        highest_compute_capability=(get_compute_capabilities()[-1]),
        cuda_version_string=(get_cuda_version_string()),
        cuda_version_tuple=get_cuda_version_tuple(),
    )
