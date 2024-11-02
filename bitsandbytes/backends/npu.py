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
    dtype2bytes,
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
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "nf4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        # if quant_type != "nf4":
        #     raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")
        if compress_statistics:
            raise NotImplementedError("compress_statistics is not implemented.")
        if blocksize is None:
            blocksize = 128

        n, input_shape = A.numel(), A.shape

        if absmax is None:
            blocks = n // blocksize
            blocks += 1 if n % blocksize > 0 else 0
            absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

        if out is None:
            mod = dtype2bytes[quant_storage] * 2
            out = torch.zeros(((n + 1) // mod, 1), dtype=quant_storage, device=A.device)

        assert_on_npu([A, absmax, out])
        if A.dtype == torch.float16:
            if quant_type == "nf4":
                lib.cquantize_blockwise_fp16_nf4(
                    get_ptr(A),
                    get_ptr(absmax),
                    get_ptr(out),
                    ct.c_int32(blocksize),
                    ct.c_int(A.numel()),
                    torch.npu.current_stream(),
                )
            else:
                raise NotImplementedError("Only float16 to nf4 is implemented.")
        else:
            raise NotImplementedError("Only float16 to nf4 is implemented.")

        code = get_4bit_type(quant_type, device=A.device)
        state = QuantState(
            absmax=absmax,
            shape=input_shape,
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
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        raise NotImplementedError

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
