import ctypes as ct
from typing import Literal, Optional, Tuple

import torch

try:
    # to support Ascend NPU backend
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bitsandbytes.cextension import lib
from bitsandbytes.functional import (
    COOSparseTensor,
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


def coo_zeros(rows, cols, rowidx, colidx, values, nnz, device, dtype=torch.half):
    rowidx = rowidx.to(torch.int32)
    colidx = colidx.to(torch.int32)
    values = values.to(device).to(dtype)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def row_col_stats(A, threshold):
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    row_max = torch.zeros(rows, dtype=torch.float32, device="npu")
    col_max = torch.zeros(cols, dtype=torch.float32, device="npu")
    outlier_num = torch.zeros(1, dtype=torch.int32, device="npu")
    lib.cget_col_row_stats(
        get_ptr(A),
        get_ptr(row_max),
        get_ptr(col_max),
        get_ptr(outlier_num),
        ct.c_float(threshold),
        ct.c_int32(rows),
        ct.c_int32(cols),
        torch.npu.current_stream()
    )
    return row_max, col_max, outlier_num


class Int8AB:
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B
        self.device = A.device


class NPUBackend(Backend):
    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        past_device = None
        device = A.device
        assert A.dtype == torch.half
        assert device.type == "npu"

        cols = A.shape[-1]
        if len(A.shape) == 3:
            rows = A.shape[0] * A.shape[1]
        else:
            rows = A.shape[0]

        if past_device != str(A.device):
            torch.npu.set_device(A.device)  # reset context
            past_device = str(A.device)

        row_stats, col_stats, cnt_npu = row_col_stats(A, threshold)

        quant_row = torch.empty((rows, cols), dtype=torch.int8, device=device)
        quant_col = torch.empty((rows, cols), dtype=torch.int8, device=device)
        outliers_row_idx = torch.zeros(rows, dtype=torch.int32, device=device)
        outliers_col_idx = torch.zeros(40 * cols, dtype=torch.int32, device=device) - 1
        outliers_value = torch.empty(0, dtype=torch.float16, device=device)

        lib.cdouble_rowcol_quant(
            get_ptr(A),
            get_ptr(row_stats),
            get_ptr(col_stats),
            get_ptr(quant_row),
            get_ptr(quant_col),
            get_ptr(outliers_row_idx),
            get_ptr(outliers_col_idx),
            get_ptr(outliers_value),
            ct.c_int(cols),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            torch.npu.current_stream()
        )

        colidx_tmp = torch.unique(outliers_col_idx)
        colidx = colidx_tmp[colidx_tmp != -1]

        coo_tensor = None
        if threshold != 0.0:
            coo_tensor = coo_zeros(rows, cols, outliers_row_idx, colidx, outliers_value, cnt_npu, device, dtype=torch.half)

        return quant_row, quant_col, row_stats, col_stats, coo_tensor

    def int8_vectorwise_dequant(self, A, stats):
        return super().int8_vectorwise_dequant(A, stats)

    def int8_vectorwise_quant(
        self,
        A: torch.Tensor,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = A.device
        assert A.dtype == torch.half
        assert device.type == "npu"

        cols = A.shape[-1]
        if len(A.shape) == 3:
            rows = A.shape[0] * A.shape[1]
        else:
            rows = A.shape[0]

        A_no_threshold = None
        if threshold > 0.0:
            zero = torch.tensor(0.0, dtype=torch.half, device=device)
            A_no_threshold = torch.where(A.view(rows, cols).abs() < threshold, A.view(rows, cols), zero)
            row_stats = torch.amax(A_no_threshold.abs(), dim=1, keepdim=True).to(device)
            out_row = torch.round(A_no_threshold * 127.0 / row_stats).to(torch.int8)
        else:
            row_stats = torch.amax(A.view(rows, cols).abs(), dim=1, keepdim=True).to(device)
            out_row = torch.round(A * 127.0 / row_stats).to(torch.int8)

        outlier_cols = None
        if threshold > 0.0:
            # TODO we could improve perf of this
            outliers = A.abs() >= threshold

            if outliers.any():
                outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)

        return out_row, row_stats, outlier_cols

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

    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        return Int8AB(A, B)

    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A, B = A.A, A.B
        out = torch_npu.npu_quant_matmul(
            A,
            B.t(),
            scale=col_stats.float() / 127.0,
            pertoken_scale=row_stats.float().view(-1) / 127.0,
            output_dtype=torch.float16
        )
        return out

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

        total_blocks = A.numel() // blocksize
        chunks = 8 if A.numel() > 2048 * 2048 else 1
        chunksize = (total_blocks + chunks - 1) // chunks

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
            chunks_absmax = []
            chunks_out = []

            for i in range(chunks):
                start = i * chunksize * blocksize
                end = min((i + 1) * chunksize * blocksize, A.numel())
                chunk_data = A.view(-1)[start:end].view(-1, blocksize)

                absmax = chunk_data.abs().max(dim=1, keepdim=True).values
                chunks_absmax.append(absmax)

                a = chunk_data / absmax.float()
                diff = torch.abs(a.unsqueeze(-1) - data)
                out = (torch.argmin(diff, dim=-1) + 8) % 16

                out = out.reshape(-1, 2)
                out = (out[:, 0] + out[:, 1] * 16).to(torch.uint8)
                chunks_out.append(out)

            absmax = torch.cat(chunks_absmax, dim=0)
            out = torch.cat(chunks_out, dim=0)
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
        beta3: float = 0.0,
        alpha: float = 0.0,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        unorm_vec: Optional[torch.Tensor] = None,
        max_unorm: float = 0.0,
        skip_zeros=False,
    ) -> None:
        raise NotImplementedError
