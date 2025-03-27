from typing import Literal, Optional, Tuple

import torch

from bitsandbytes.utils import QuantState

from .base import Backend
from .cpu_xpu_common import (
    dequantize_4bit_impl,
    double_quant_impl,
    gemm_4bit_impl,
    int8_linear_matmul_impl,
    int8_mm_dequant_impl,
    quantize_4bit_impl,
    _ipex_xpu_version_prereq
)
try:
    import intel_extension_for_pytorch as ipex
    ipex_xpu = ipex if ipex._C._has_xpu() else None
except BaseException:
    ipex_xpu = None

Tensor = torch.Tensor


str2optimizer8bit_blockwise = {}
if ipex_xpu is not None and _ipex_xpu_version_prereq(2, 7):
    str2optimizer8bit_blockwise = {
            "adam": (
                ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_fp32,
                ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_fp16,
                ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_bf16,
            ),
        }


def assert_on_xpu(tensors):
    on_xpu = True
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        on_xpu &= t.device.type == "xpu"
    if not on_xpu:
        raise TypeError(
            "All input tensors need to be on XPU, but found some tensors to not be on XPU:\n"
            f" {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}"
        )
    return on_xpu


class XPUBackend(Backend):
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    def device_synchronize(self):
        torch.xpu.synchronize()

    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert_on_xpu([A, col_stats, row_stats, out_col, out_row])
        output = double_quant_impl(A, col_stats, row_stats, out_col, out_row, threshold)
        return output

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
        """
        Transform tensor A to to_order. It is originally designed for CUDA.
        For XPU, it returns the original tensor if transpose=False.
        Otherwise, it returns the transpose of A
        """
        assert_on_xpu([A, out])
        if transpose:
            if out is not None:
                out.copy_(A.T)
            else:
                out = A.T
        else:
            if out is not None:
                out.copy_(A)
            else:
                out = A
        return out, state

    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        assert_on_xpu([A, B])
        output = int8_linear_matmul_impl(A, B, out, dtype)
        return output

    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert_on_xpu([A, row_stats, col_stats, out, bias])
        output = int8_mm_dequant_impl(
            A,
            row_stats,
            col_stats,
            out,
            bias,
            self.mm_dequant_compute_dtype,
            self.mm_dequant_output_dtype,
        )
        return output

    def int8_vectorwise_dequant(self, A, stats):
        return super().int8_vectorwise_dequant(A, stats)

    def int8_vectorwise_quant(self, A: torch.Tensor, threshold=0.0):
        # TODO: We can optimize this as we don't actually need column-wise quant.
        out, _, stats, _, outlier_cols = self.int8_double_quant(A, threshold=threshold)
        return out, stats, outlier_cols

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        assert_on_xpu([A])
        output = A[:, idx].contiguous()
        return output

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if blocksize is None:
            blocksize = 64
        assert_on_xpu([A, absmax, out])
        output = quantize_4bit_impl(A, absmax, out, blocksize, compress_statistics, quant_type, quant_storage)
        return output

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        if blocksize is None:
            blocksize = 64
        assert_on_xpu([A, absmax, out])
        if quant_type == "nf4" and getattr(quant_state, "ipex", False):
            output = torch.ops.torch_ipex.dequantize_4bit(A, "nf4", quant_state.shape, absmax, None, blocksize).t()
        else:
            output = dequantize_4bit_impl(A, quant_state, absmax, out, blocksize, quant_type)

        return output

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        assert_on_xpu([A, B, out])
        if state is None:
            raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")
        output = gemm_4bit_impl(A, B, out, transposed_A, transposed_B, state)
        return output

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
        if ipex_xpu is None or not _ipex_xpu_version_prereq(2, 7):
            raise RuntimeError("Please install intel_extension_for_ipex >= 2.7 for 8bit optimizer backend on XPU device.")

        # void cdequantize_blockwise_fp32(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, cudaStream_t stream)
        if out.dtype == torch.float16:
            ipex.xpu.bitsandbytes.cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, A.numel())
        elif out.dtype == torch.bfloat16:
            ipex.xpu.bitsandbytes.cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, A.numel())
        elif out.dtype == torch.float32:
            ipex.xpu.bitsandbytes.cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, A.numel())
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")
        

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
        optim_func = None
        if ipex_xpu is None or not _ipex_xpu_version_prereq(2, 7):
            raise RuntimeError("Please install intel_extension_for_ipex >= 2.7 for 8bit optimizer backend on XPU device.")

        assert_on_xpu([g, p, state1, state2, qmap1, qmap2, absmax1, absmax2])

        if g.dtype == torch.float32 and state1.dtype == torch.uint8:
            optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
        elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
            optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
        elif (
            g.dtype == torch.bfloat16
            and state1.dtype == torch.uint8
            and len(str2optimizer8bit_blockwise[optimizer_name]) == 3
        ):
            optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
        else:
            raise ValueError(
                f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
            )
        optim_func(
            p,
            g,
            state1,
            state2,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            step,
            lr,
            qmap1,
            qmap2,
            absmax1,
            absmax2,
            weight_decay,
            gnorm_scale,
            skip_zeros,
            g.numel()
        )


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
