import math
from typing import Literal, Optional, Tuple

import torch

from bitsandbytes.functional import get_4bit_type
from bitsandbytes.utils import QuantState

from .base import Backend
from .cpu_xpu_common import (
    INT8_QUANT_TABLE,
    NF4_QUANT_TABLE,
    dequant_8bit,
)

Tensor = torch.Tensor


def assert_on_hpu(tensors):
    on_hpu = True
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        on_hpu &= t.device.type == "hpu"
    if not on_hpu:
        raise TypeError(
            "All input tensors need to be on HPU, but found some tensors to not be on HPU:\n"
            f" {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}"
        )
    return on_hpu


class HPUBackend(Backend):
    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Not yet implemented for HPU backend")

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
        raise NotImplementedError("Not yet implemented for HPU backend")

    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented for HPU backend")

    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented for HPU backend")

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented for HPU backend")

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["nf4"] = "nf4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if blocksize is None:
            blocksize = 64
        assert_on_hpu([A, absmax, out])
        assert quant_storage == torch.uint8, "HPU backend only supports uint8 quant_storage"
        return self.quantize_4bit_impl(A, absmax, out, blocksize, compress_statistics, quant_type)

    def quantize_4bit_impl(
        self,
        A: Tensor,
        absmax: Tensor = None,
        out: Tensor = None,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
    ) -> Tensor:
        if quant_type not in ["nf4", "int8"]:
            raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented for HPU.")
        assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        n = A.numel()
        input_shape = A.shape
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0

        if absmax is None:
            absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)

        if out is None:
            out = torch.zeros(((n + 1) // 2), dtype=torch.uint8, device=A.device)

        rem = n % blocksize
        has_rem = rem > 0

        # Scale tensor to [-1, 1]
        A_reshaped = A.reshape(n)
        A_com = A_reshaped[: n - rem]
        A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
        absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
        scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
        scaled_A = scaled_A.reshape(-1)
        if has_rem:
            absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
            scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
            scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)
        # map [-1, 1] to nf4
        out_uint8 = torch.empty(scaled_A.shape, dtype=torch.uint8, device=A.device)
        if quant_type == "nf4":
            for i in range(len(NF4_QUANT_TABLE)):
                out_uint8[scaled_A > NF4_QUANT_TABLE[i]] = i
        elif quant_type == "int8":
            map = torch.tensor(INT8_QUANT_TABLE, device=scaled_A.device)
            diff = torch.abs(scaled_A.unsqueeze(-1) - map)
            out_uint8 = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device)

        if quant_type == "int8":
            out = out_uint8
            code = torch.Tensor(INT8_QUANT_TABLE).to(A.device)
        else:
            if out_uint8.size(-1) % 2:
                out_uint8 = torch.nn.functional.pad(out_uint8, (0, 1), value=0)
            # To align with HPU dequantize operator
            out[:] = out_uint8[1::2].bitwise_left_shift(4).bitwise_or_(out_uint8[::2])
            code = get_4bit_type(quant_type, device=A.device)

        if compress_statistics:
            offset = absmax.mean()
            absmax -= offset
            qabsmax, state2 = self.quantize_4bit_impl(absmax, blocksize=256, quant_type="int8")
            del absmax
            state = QuantState(
                absmax=qabsmax,
                shape=input_shape,
                dtype=A.dtype,
                blocksize=blocksize,
                code=code,
                quant_type=quant_type,
                offset=offset,
                state2=state2,
            )
        else:
            state = QuantState(
                absmax=absmax,
                shape=input_shape,
                dtype=A.dtype,
                blocksize=blocksize,
                code=code,
                quant_type=quant_type,
            )
        return out, state

    def dequantize_nf4_impl(
        self,
        input: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_state: QuantState,
    ) -> torch.Tensor:
        """
        HPU dequantization function for NF4 quantized tensors.
        """
        assert_on_hpu([input, absmax])
        out_shape = (math.prod(quant_state.shape),)
        out_dq = torch.ops.hpu.dequantize_nf4(
            input, absmax, blocksize, out_shape=out_shape, out_dtype=quant_state.dtype
        )
        output = out_dq.reshape(quant_state.shape).T
        return output

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["nf4"] = "nf4",
    ) -> torch.Tensor:
        if blocksize is None:
            blocksize = 64

        assert_on_hpu([A, absmax, out])
        if quant_state.nested:
            absmax = dequant_8bit(absmax, quant_state.offset, quant_state.state2)
        return self.dequantize_nf4_impl(A, absmax, blocksize, quant_state)

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented for HPU backend")

    def int8_vectorwise_dequant(self, A: torch.Tensor, stats: torch.Tensor):
        raise NotImplementedError("Not yet implemented for HPU backend")

    def int8_vectorwise_quant(self, A: torch.Tensor, threshold=0.0):
        raise NotImplementedError("Not yet implemented for HPU backend")

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
        raise NotImplementedError("Not yet implemented for HPU backend")

    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError("Not yet implemented for HPU backend")

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
        raise NotImplementedError("Not yet implemented for HPU backend")

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
        raise NotImplementedError("Not yet implemented for HPU backend")
