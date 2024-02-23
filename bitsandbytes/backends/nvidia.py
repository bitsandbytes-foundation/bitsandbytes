import ctypes

import torch

from ._base import BackendInterface
from ._helpers import get_ptr, is_on_gpu, post_call, pre_call


class CudaBackend(BackendInterface):
    def check_matmul(
        self, A, B, out, transposed_A, transposed_B, expected_type=torch.int8
    ):
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        if A.dtype != expected_type or B.dtype != expected_type:
            raise TypeError(
                f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}"
            )

        sA = A.shape
        sB = B.shape
        tA = transposed_A
        tB = transposed_B

        correct = True

        if len(sA) == 2 and len(sB) == 2:
            if not tA and not tB and A.shape[1] != B.shape[0]:
                correct = False
            elif tA and not tB and A.shape[0] != B.shape[0]:
                correct = False
            elif tA and tB and A.shape[0] != B.shape[1]:
                correct = False
            elif not tA and tB and A.shape[1] != B.shape[1]:
                correct = False
        elif len(sA) == 3 and len(sB) == 2:
            if not tA and not tB and A.shape[2] != B.shape[0]:
                correct = False
            elif tA and not tB and A.shape[1] != B.shape[0]:
                correct = False
            elif tA and tB and A.shape[1] != B.shape[1]:
                correct = False
            elif not tA and tB and A.shape[2] != B.shape[1]:
                correct = False
        elif len(sA) == 3 and len(sB) == 3:
            if not tA and not tB and A.shape[2] != B.shape[1]:
                correct = False
            elif tA and not tB and A.shape[1] != B.shape[1]:
                correct = False
            elif tA and tB and A.shape[1] != B.shape[2]:
                correct = False
            elif not tA and tB and A.shape[2] != B.shape[2]:
                correct = False

        if out is not None:
            sout = out.shape
            # special case common in backprop
            if not correct and len(sA) == 3 and len(sB) == 3:
                if (
                    sout[0] == sA[2]
                    and sout[1] == sB[2]
                    and sA[0] == sB[0]
                    and sA[1] == sB[1]
                ):
                    correct = True
        else:
            if len(sA) == 2 and len(sB) == 2:
                if not tA and not tB:
                    sout = (sA[0], sB[1])
                elif tA and tB:
                    sout = (sA[1], sB[0])
                elif tA and not tB:
                    sout = (sA[1], sB[1])
                elif not tA and tB:
                    sout = (sA[0], sB[0])
            elif len(sA) == 3 and len(sB) == 2:
                if not tA and not tB:
                    sout = (sA[0], sA[1], sB[1])
                elif tA and tB:
                    sout = (sA[0], sA[2], sB[0])
                elif tA and not tB:
                    sout = (sA[0], sA[2], sB[1])
                elif not tA and tB:
                    sout = (sA[0], sA[1], sB[0])
            elif len(sA) == 3 and len(sB) == 3:
                if not tA and not tB:
                    sout = (sA[0], sA[1], sB[2])
                elif tA and tB:
                    sout = (sA[0], sA[2], sB[1])
                elif tA and not tB:
                    sout = (sA[0], sA[2], sB[2])
                elif not tA and tB:
                    sout = (sA[0], sA[1], sB[1])

        if not correct:
            raise ValueError(
                f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}."
            )

        return sout

    def get_colrow_absmax(
        self, A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
    ):
        assert A.dtype == torch.float16
        device = A.device

        cols = A.shape[-1]
        if len(A.shape) == 3:
            rows = A.shape[0] * A.shape[1]
        else:
            rows = A.shape[0]

        col_tiles = (cols + 255) // 256
        tiled_rows = ((rows + 15) // 16) * 16
        if row_stats is None:
            row_stats = torch.empty((rows,), dtype=torch.float32, device=device).fill_(
                -50000.0
            )
        if col_stats is None:
            col_stats = torch.empty((cols,), dtype=torch.float32, device=device).fill_(
                -50000.0
            )

        if nnz_block_ptr is None and threshold > 0.0:
            nnz_block_ptr = torch.zeros(
                ((tiled_rows * col_tiles) + 1,), dtype=torch.int32, device=device
            )

        ptrA = get_ptr(A)
        ptrRowStats = get_ptr(row_stats)
        ptrColStats = get_ptr(col_stats)
        ptrNnzrows = get_ptr(nnz_block_ptr)
        rows = ctypes.c_int32(rows)
        cols = ctypes.c_int32(cols)

        prev_device = pre_call(A.device)
        is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
        self.lib.cget_col_row_stats(
            ptrA,
            ptrRowStats,
            ptrColStats,
            ptrNnzrows,
            ctypes.c_float(threshold),
            rows,
            cols,
        )
        post_call(prev_device)

        if threshold > 0.0:
            nnz_block_ptr.cumsum_(0)

        return row_stats, col_stats, nnz_block_ptr

    def double_quant(
        self,
        A,
        col_stats=None,
        row_stats=None,
        out_col=None,
        out_row=None,
        threshold=0.0,
    ):
        device = A.device
        assert A.dtype == torch.half
        assert device.type == "cuda"
        prev_device = pre_call(A.device)

        cols = A.shape[-1]
        if len(A.shape) == 3:
            rows = A.shape[0] * A.shape[1]
        else:
            rows = A.shape[0]

        if row_stats is None or col_stats is None:
            row_stats, col_stats, nnz_row_ptr = self.get_colrow_absmax(
                A, threshold=threshold
            )

        if out_col is None:
            out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
        if out_row is None:
            out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)

        coo_tensor = None
        ptrA = get_ptr(A)
        ptrColStats = get_ptr(col_stats)
        ptrRowStats = get_ptr(row_stats)
        ptrOutCol = get_ptr(out_col)
        ptrOutRow = get_ptr(out_row)

        is_on_gpu([A, col_stats, row_stats, out_col, out_row])
        if threshold > 0.0:
            nnz = nnz_row_ptr[-1].item()
            if nnz > 0:
                coo_tensor = self.coo_zeros(
                    A.shape[0], A.shape[1], nnz_row_ptr[-1].item(), device
                )
                ptrRowIdx = get_ptr(coo_tensor.rowidx)
                ptrColIdx = get_ptr(coo_tensor.colidx)
                ptrVal = get_ptr(coo_tensor.values)
                ptrRowPtr = get_ptr(nnz_row_ptr)

                self.lib.cdouble_rowcol_quant(
                    ptrA,
                    ptrRowStats,
                    ptrColStats,
                    ptrOutCol,
                    ptrOutRow,
                    ptrRowIdx,
                    ptrColIdx,
                    ptrVal,
                    ptrRowPtr,
                    ctypes.c_float(threshold),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(cols),
                )
                val, idx = torch.sort(coo_tensor.rowidx)
                coo_tensor.rowidx = val
                coo_tensor.colidx = coo_tensor.colidx[idx]
                coo_tensor.values = coo_tensor.values[idx]
            else:
                self.lib.cdouble_rowcol_quant(
                    ptrA,
                    ptrRowStats,
                    ptrColStats,
                    ptrOutCol,
                    ptrOutRow,
                    None,
                    None,
                    None,
                    None,
                    ctypes.c_float(0.0),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(cols),
                )
        else:
            self.lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                None,
                None,
                None,
                None,
                ctypes.c_float(threshold),
                ctypes.c_int32(rows),
                ctypes.c_int32(cols),
            )
        post_call(prev_device)

        return out_row, out_col, row_stats, col_stats, coo_tensor

    """
    # CUDA specific interface (do not include in general interface):
    'CUBLAS_Context'
    'Cusparse_Context'
    'GlobalPageManager'
    '_mul'
    'arange'
    'dtype2bytes'
    'elementwise_func'
    'fill'
    'get_paged'
    'get_4bit_type'
    'get_ptr'
    'get_special_format_str'
    'get_transform_buffer'
    'get_transform_func'
    'is_on_gpu'
    'nvidia_transform'
    'transform'

    ## Deprecate these:
    'optimizer_update_8bit'
    'dequant_min_max'
    'dequantize'
    'dequantize_no_absmax'
    'igemm'
    'quantize'
    'spmm_coo'
    'spmm_coo_very_sparse'
    'vectorwise_dequant'
    'vectorwise_mm_dequant'
    'vectorwise_quant'
    'CSCSparseTensor'
    'CSRSparseTensor'
    'coo2csc'
    'coo2csr'
    'histogram_scatter_add_2d'
    """
