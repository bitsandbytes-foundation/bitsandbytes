from collections.abc import Sequence
from functools import partial

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


# optimizer_update_8bit_blockwise = kernels_optim.optimizer_update_8bit_blockwise_pytorch
# optimizer_update_8bit_blockwise = torch.compile(kernels_optim.optimizer_update_8bit_blockwise_pytorch) # 60ms
# optimizer_update_8bit_blockwise = kernels_optim.optimizer_update_8bit_blockwise_triton_quant #2.8ms
# optimizer_update_8bit_blockwise = torch.compile(kernels_optim.optimizer_update_8bit_blockwise_triton_quant) # 2.3ms

# adam_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="adam")
# momentum_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="momentum")
# rmsprop_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="rmsprop")
# lion_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="lion")
# adagrad_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="adagrad")
# ademamix_8bit_blockwise_grad = partial(optimizer_update_8bit_blockwise, optimizer_name="ademamix")

# ~0.95ms for adam
update_1state = kernels_optim.optimizer_update_1state_8bit_blockwise
update_2state = kernels_optim.optimizer_update_2state_8bit_blockwise
momentum_8bit_blockwise_grad = partial(update_1state, optimizer_id=kernels_optim.name2optimizer_id["momentum"])
rmsprop_8bit_blockwise_grad = partial(update_1state, optimizer_id=kernels_optim.name2optimizer_id["rmsprop"])
lion_8bit_blockwise_grad = partial(update_1state, optimizer_id=kernels_optim.name2optimizer_id["lion"])
adagrad_8bit_blockwise_grad = partial(update_1state, optimizer_id=kernels_optim.name2optimizer_id["adagrad"])

ademamix_8bit_blockwise_grad = partial(update_2state, optimizer_id=kernels_optim.name2optimizer_id["ademamix"])
adam_8bit_blockwise_grad = partial(update_2state, optimizer_id=kernels_optim.name2optimizer_id["adam"])
