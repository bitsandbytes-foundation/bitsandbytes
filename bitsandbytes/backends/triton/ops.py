from collections.abc import Sequence
from typing import Optional

import torch

from . import kernels_4bit, kernels_8bit_quant, kernels_optim

# currently codes unused, kept for reference
# Should be the same for quant/dequant
# from bitsandbytes.functional import get_4bit_type
# _FP4_QUANT_TABLE = get_4bit_type("fp4", device="xpu")
# _NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")
device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
torch_accelerator_module = getattr(torch, device_type, torch.cuda)


def quantize_blockwise(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(A.dtype == torch.float32, lambda: f"A must be float32 on xpu, got {A.dtype}")
    with torch_accelerator_module.device(A.device):
        out, absmax = kernels_8bit_quant.quantize_blockwise_triton(A, code, blocksize)
        return out, absmax.float()


def dequantize_blockwise(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    # torch._check(dtype == torch.float32, lambda: f"dtype must be float32 on xpu, got {dtype}")
    with torch_accelerator_module.device(A.device):
        out = kernels_8bit_quant.dequant_8bit_blockwise(
            A,
            absmax,
            code,
            blocksize,
            dtype=dtype,
        )
    return out


def dequantize_blockwise_inplace(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")

    with torch_accelerator_module.device(A.device):
        kernels_8bit_quant.dequant_8bit_blockwise(
            A,
            absmax,
            code,
            blocksize,
            dtype=dtype,
            out=out,
        )


def quantize_4bit(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()

    # TODO: Support when weight matrix is not divisible by blocksize
    # torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    blocks = -(n // -(blocksize * 2))

    absmax = torch.empty((blocks * 2,), device=A.device, dtype=A.dtype)
    out = torch.empty((n // 2, 1), device=A.device, dtype=torch.uint8)

    with torch_accelerator_module.device(A.device):
        kernels_4bit.quantize_4bit_blockwise_triton(
            A, blocksize, quant_type, blocks, absmax, num_elements=n, quantized_out=out
        )
    packed = out

    if quant_storage != torch.uint8:
        packed = out.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


def dequantize_4bit(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on XPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    # torch._check(
    #     A.dtype == torch.uint8,
    #     lambda: f"Blockwise 4bit dequantization on XPU only supports uint8 storage, got {A.dtype}",
    # )
    # Check if this is fine and fast
    if A.dtype != torch.uint8:
        A = A.squeeze().view(torch.uint8).unsqueeze(1)

    out = torch.empty(shape, dtype=dtype, device=A.device)
    with torch_accelerator_module.device(A.device):
        kernels_4bit.dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)

    return out


def dequantize_4bit_inplace(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    with torch_accelerator_module.device(A.device):
        kernels_4bit.dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def gemv_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    if B.dtype != torch.uint8:
        B = B.squeeze().view(torch.uint8).unsqueeze(1)

    B_dq_triton = torch.empty(shapeB, dtype=A.dtype, device=A.device)

    with torch_accelerator_module.device(A.device):
        kernels_4bit.dequantize_4bit_impl_passing_code(
            B,
            absmax,
            blocksize,
            code,
            dtype=A.dtype,
            out=B_dq_triton,
        )

        return torch.nn.functional.linear(
            A,
            B_dq_triton,
            bias=None,
        )


# optimizer_update_8bit_blockwise_impl = kernels_optim.optimizer_update_8bit_blockwise_pytorch
# optimizer_update_8bit_blockwise_impl = torch.compile(kernels_optim.optimizer_update_8bit_blockwise_pytorch) # 60ms
# optimizer_update_8bit_blockwise_impl = kernels_optim.optimizer_update_8bit_blockwise_triton_quant #2.8ms
# optimizer_update_8bit_blockwise_impl = torch.compile(kernels_optim.optimizer_update_8bit_blockwise_triton_quant) # 2.3ms
optimizer_update_8bit_blockwise_impl = kernels_optim.optimizer_update_8bit_blockwise_impl  # ~0.95ms for adam


def optimizer_update_8bit_blockwise(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: torch.Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: torch.Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    # torch._check(
    #     g.numel() == p.numel(),
    #     lambda: f"g and p must have the same number of elements, got {g.numel()} and {p.numel()}",
    # )
    # compute_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # torch._check(
    #     g.dtype in compute_dtypes,
    #     lambda: f"g must be bfloat16, float16, or float32, got {g.dtype}",
    # )
    # torch._check(
    #     g.dtype == p.dtype,
    #     lambda: f"Expected all tensors to have the same dtype, got g.dtype={g.dtype}, p.dtype={p.dtype}",
    # )
    # torch._check(
    #     state1.dtype == torch.uint8,
    #     lambda: f"state1 must be uint8, got {state1.dtype}",
    # )
    # torch._check(
    #     qmap1.dtype == absmax1.dtype == torch.float32,
    #     lambda: f"Expected qmap1 and absmax1 to be float32, got qmap1.dtype={qmap1.dtype}, absmax1.dtype={absmax1.dtype}",
    # )
    # if state2 is not None:
    #     torch._check(
    #         state2.dtype == torch.uint8,
    #         lambda: f"state2 must be uint8, got {state2.dtype}",
    #     )
    #     torch._check(
    #         qmap2.dtype == absmax2.dtype == torch.float32,
    #         lambda: f"Expected qmap2 and absmax2 to be float32, got qmap2.dtype={qmap2.dtype}, absmax2.dtype={absmax2.dtype}",
    #     )

    with torch_accelerator_module.device(state1.device):
        optimizer_update_8bit_blockwise_impl(
            optimizer_name=optimizer_name,
            g=g,
            p=p,
            state1=state1,
            state2=state2,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            step=step,
            lr=lr,
            qmap1=qmap1,
            qmap2=qmap2,
            absmax1=absmax1,
            absmax2=absmax2,
            weight_decay=weight_decay,
            gnorm_scale=gnorm_scale,
            skip_zeros=skip_zeros,
        )


def optimizer_update_32bit(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    unorm_vec: Optional[torch.Tensor],
    max_unorm: float,
    param_norm: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    step: int,
    lr: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    with torch_accelerator_module.device(state1.device):
        kernels_optim.optimizer_update_32bit_impl(
            optimizer_name=optimizer_name,
            g=g,
            p=p,
            state1=state1,
            state2=state2,
            unorm_vec=unorm_vec,
            max_unorm=max_unorm,
            param_norm=param_norm,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            step=step,
            lr=lr,
            gnorm_scale=gnorm_scale,
            skip_zeros=skip_zeros,
        )
