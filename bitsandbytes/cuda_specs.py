import dataclasses
from typing import List, Optional, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: Tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: Tuple[int, int]

    @property
    def has_cublaslt(self) -> bool:
        return self.highest_compute_capability >= (7, 5)


def get_compute_capabilities() -> List[Tuple[int, int]]:
    return sorted(torch.cuda.get_device_capability(torch.cuda.device(i)) for i in range(torch.cuda.device_count()))


def get_cuda_version_tuple() -> Tuple[int, int]:
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
    major, minor = map(int, torch.version.cuda.split("."))
    return major, minor


def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f"{major}{minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    if not torch.cuda.is_available():
        return None

    return CUDASpecs(
        highest_compute_capability=(get_compute_capabilities()[-1]),
        cuda_version_string=(get_cuda_version_string()),
        cuda_version_tuple=get_cuda_version_tuple(),
    )
