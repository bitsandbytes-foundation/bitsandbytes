import ctypes as ct
from typing import Literal, Optional, Tuple, Union

import torch

try:
    # to support Ascend NPU backend
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bitsandbytes.cextension import lib
from bitsandbytes.functional import (
    get_4bit_type,
    get_ptr,
)
from bitsandbytes.utils import QuantState

from .base import Backend


def assert_on_npu(tensors):
    if not all(t.device.type == "npu" for t in tensors if t is not None):
        raise TypeError(
            "All input tensors to be on NPU, but found some tensors not be on NPU:\n"
            f"{[(t.shape, t.device) if isinstance(t, torch.Tensor) else None for t in tensors]}"
        )
    return True


class NPUBackend(Backend):
    def double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ):
        raise NotImplementedError

    def transform(
        self,
        A: torch.Tensor,
        to_order: str,
        from_order="row",
        out: Optional[torch.Tensor] = None,
        transpose=False,
        state: Optional[Tuple[torch.Size, str]] = None,
        ld=None,
    ):
        raise NotImplementedError

    def igemmlt(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        SA: Tuple[torch.Size, str],
        SB: Tuple[torch.Size, str],
        out: Optional[torch.Tensor] = None,
        Sout: Optional[Tuple[torch.Size, str]] = None,
        dtype=torch.int32,
    ) -> Union[torch.Tensor, Tuple[Optional[Tuple[torch.Tensor, Tuple[torch.Size, str]]]]]:
        raise NotImplementedError

    def mm_dequant(
        self,
        A: torch.Tensor,
        quant_state: Tuple[torch.Size, str],
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        new_row_stats: Optional[torch.Tensor] = None,
        new_col_stats: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "nf4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if quant_type not in ["nf4"]:
            raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")
        if compress_statistics:
            raise NotImplementedError("compress_statistics is not implemented.")
        if blocksize is None:
            blocksize = 128

        prev_device = torch.npu.current_device()
        torch.npu.set_device(A.device)
        if A.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            data = [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
            data = torch.tensor(data, device="npu", dtype=torch.float32).view(1, -1)
            absmax = A.view(-1, blocksize).abs().max(dim=1, keepdim=True).values
            a = A.view(-1, blocksize) / absmax.float()
            diff = torch.abs(a.unsqueeze(-1) - data)
            out = (torch.argmin(diff, dim=-1) + 8) % 16
            out = out.reshape(-1, 2)
            out = (out[:, 0] + out[:, 1] * 16).to(torch.uint8)
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        assert_on_npu([A, absmax, out])
        torch.npu.set_device(prev_device)

        code = get_4bit_type(quant_type, device=A.device)
        state = QuantState(
            absmax=absmax,
            shape=A.shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )

        return out, state

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        quant_type: Literal["fp4", "nf4"] = "nf4",
    ) -> torch.Tensor:
        if blocksize is None:
            blocksize = 128
        supported_blocksizes = [2048, 4096, 1024, 512, 256, 128, 64]
        if blocksize not in supported_blocksizes:
            raise ValueError(
                f"The blockwise of {blocksize} is not supported. Supported values: {supported_blocksizes}"
            )

        if quant_state is None:
            assert absmax is not None and out is not None
            quant_state = QuantState(
                absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type
            )
        else:
            absmax = quant_state.absmax

        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

        n = out.numel()

        prev_device = torch.npu.current_device()
        torch.npu.set_device(A.device)
        assert_on_npu([A, absmax, out])

        if quant_state.quant_type not in ["nf4"]:
            raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")

        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32_nf4(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(n),
                torch.npu.current_stream(),
            )
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16_nf4(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(n),
                torch.npu.current_stream(),
            )
        elif out.dtype == torch.bfloat16:
            # bf16: bf16 -> fp32 -> op -> fp32 -> bf16
            absmax = absmax.to(torch.float32)
            out = out.to(torch.float32)
            lib.cdequantize_blockwise_fp32_nf4(
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(n),
                torch.npu.current_stream(),
            )
            out = out.to(torch.bfloat16)
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        torch.npu.set_device(prev_device)
        is_transposed = True if A.shape[0] == 1 else False

        if is_transposed:
            return out.t()
        else:
            return out

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def dequantize_blockwise(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        code: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 4096,
        nested=False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError

    def optimizer_update_8bit_blockwise(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
        state2: Optional[torch.Tensor],
        beta1: float,
        beta2: float,
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
        raise NotImplementedError

    def optimizer_update_32bit(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
        beta1: float,
        eps: float,
        step: int,
        lr: float,
        state2: Optional[torch.Tensor] = None,
        beta2: float = 0.0,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        unorm_vec: Optional[torch.Tensor] = None,
        max_unorm: float = 0.0,
        skip_zeros=False,
    ) -> None:
        raise NotImplementedError
