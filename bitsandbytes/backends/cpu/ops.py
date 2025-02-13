from typing import Optional

import torch

from ..._ops import register_kernel


@register_kernel("bitsandbytes::int8_linear_matmul", "cpu")
def _(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    # Naive implementation: perform matmul in fp32
    result = torch.matmul(A.float(), B.float().t()).to(torch.int32)
    if out is not None:
        result = out.copy_(result)
    return result
