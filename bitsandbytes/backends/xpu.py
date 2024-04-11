# For Intel GPU (xpu is the device name for Intel GPU in PyTorch)
import torch
from .cpu import (
    double_quant_common,
    igemmlt_common,
    mm_dequant_common,
)

Tensor = torch.Tensor

def assert_on_xpu(tensors):
    on_xpu = True
    for t in tensors:
        if t is None: continue # NULL pointers are fine
        on_xpu &= (t.device.type == 'xpu')
    if not on_xpu:
        raise TypeError(
            'All input tensors need to be on XPU, but found some tensors to not be on XPU:\n' \
            f' {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}'
        )
    return on_xpu


class XPUBackend:
    mm_dequant_compute_dtype = torch.half
    mm_dequant_output_dtype = torch.half

    @classmethod
    @torch.compile(dynamic=True, options={"fx_graph_cache": True})
    def double_quant(
        cls, A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
    ):
        assert_on_xpu([A, col_stats, row_stats, out_col, out_row])
        return double_quant_common(A, col_stats, row_stats, out_col, out_row)

    @classmethod
    def transform(cls, A, to_order=None, from_order='row', out=None, transpose=False, state=None, ld=None):
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

    @classmethod
    def igemmlt(cls, A, B, SA=None, SB=None, out=None, Sout=None, dtype=torch.int32):
        assert_on_xpu([A, B])
        return igemmlt_common(A, B, SA, SB, out, Sout, dtype)

    @classmethod
    @torch.compile(dynamic=True, options={"fx_graph_cache": True})
    def mm_dequant(
        cls,
        A,
        quant_state,
        row_stats,
        col_stats,
        out=None,
        new_row_stats=None,
        new_col_stats=None,
        bias=None
    ):
        assert_on_xpu([A, row_stats, col_stats, out, bias])
        return mm_dequant_common(
            A,
            quant_state,
            row_stats,
            col_stats,
            out,
            new_row_stats,
            new_col_stats,
            bias,
            cls.mm_dequant_compute_dtype,
            cls.mm_dequant_output_dtype
        )

    @classmethod
    def extract_outliers(cls, A, SA, idx):
        """
        Extract columns of A by idx
        """
        assert_on_xpu([A])
        return A[:, idx].contiguous()

    @classmethod
    def quantize_4bit(
        cls,
        A: Tensor,
        absmax: Tensor = None,
        out: Tensor = None,
        blocksize=64,
        compress_statistics=False,
        quant_type="fp4",
    ) -> Tensor:
        assert False, "quantize_4bit not yet implemented for XPU backend"

    @classmethod
    def dequantize_4bit(
        cls,
        A: Tensor,
        quant_state = None,
        absmax: Tensor = None,
        out: Tensor = None,
        blocksize: int = 64,
        quant_type="fp4",
    ) -> Tensor:
        assert False, "dequantize_4bit not yet implemented for XPU backend"
