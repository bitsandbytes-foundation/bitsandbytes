import torch
from .cpu_xpu_common import (
    double_quant_impl,
    igemmlt_impl,
    mm_dequant_impl,
)


Tensor = torch.Tensor


def assert_on_cpu(tensors):
    on_cpu = True
    for t in tensors:
        if t is None: continue # NULL pointers are fine
        on_cpu &= (t.device.type == 'cpu')
    if not on_cpu:
        raise TypeError(
            'All input tensors need to be on CPU, but found some tensors to not be on CPU:\n' \
            f' {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}'
        )
    return on_cpu


class CPUBackend:
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    @classmethod
    def double_quant(
        cls, A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
    ):
        assert_on_cpu([A, col_stats, row_stats, out_col, out_row])
        return double_quant_impl(A, col_stats, row_stats, out_col, out_row, threshold)

    @classmethod
    def transform(cls, A, to_order=None, from_order='row', out=None, transpose=False, state=None, ld=None):
        """
        Transform tensor A to to_order. It is originally designed for CUDA.
        For CPU, it returns the original tensor if transpose=False.
        Otherwise, it returns the transpose of A
        """
        assert_on_cpu([A, out])
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
        assert_on_cpu([A, B])
        return igemmlt_impl(A, B, SA, SB, out, Sout, dtype)

    @classmethod
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
        assert_on_cpu([A, row_stats, col_stats, out, bias])
        return mm_dequant_impl(
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
        assert_on_cpu([A])
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
        assert False, "quantize_4bit not yet implemented for CPU backend"

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
        assert False, "dequantize_4bit not yet implemented for CPU backend"
