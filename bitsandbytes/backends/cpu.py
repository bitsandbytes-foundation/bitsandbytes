import torch


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


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def double_quant_common(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    """
    Find absolute max valus of each row/column of a tensor, and symmetrically quantize it to int8.
    If threshold > 0.0, only values <= threshold are counted. All outliers are zeroed out in
    the original tensor and they are kept in COO format: (rows, cols, valus)
    If threashold == 0.0, there are no outliers.
    Args:
        A The tensor to be analyzed and quantized.
        col_stats Absolute max values of each column of A. If it is not None, use the values directly.
            Otherwise, find the values.
        row_stats Absolute max values of each row of A. If it is not None, use the values directly.
            Otherwise, find the values.
        out_col Output buffer for the result quantized per column if it is not None
        out_row Output buffer for the result quantized per row if it is not None
        threshold The threshold for finding outliers if it is > 0.0. Otherwise it has no effect.
    Return:
        A tuple of output quantized per row, output quantized per column, absolute max values of
        each row of A, absolute max values of each column of A, outliers in COO format
    """
    from ..functional import COOSparseTensor
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        assert A.dim() == 2, f"double_quant: Input tensor should be 2d or 3d but got {A.dim()}d"
        rows = A.shape[0]
    A = A.reshape(rows, cols)

    coo_tensor = None

    def get_row_col_stats(A):
        row_stats = torch.max(torch.abs(A), 1).values # absolute max of each row
        col_stats = torch.max(torch.abs(A), 0).values # absolute max of each col
        return row_stats, col_stats

    def quant_to_int8(A, stats):
        return torch.clamp(torch.round(A / stats * 127).to(torch.int8), -128, 127)

    if threshold == 0.0:
        if row_stats is None or col_stats is None:
            row_stats, col_stats = get_row_col_stats(A)
    else:
        outlier_indices = torch.abs(A) > threshold # find outliers
        outlier_coord = outlier_indices.nonzero() # get outlier coordinates
        outlier_rows = outlier_coord[:, 0] # outlier row for COO sparse tensor
        outlier_cols = outlier_coord[:, 1] # outlier column for COO sparse tensor
        outlier_values = A[outlier_indices] # outlier values for COO sparse tensor
        coo_tensor = COOSparseTensor(
            A.shape[0], A.shape[1], outlier_values.numel(), outlier_rows.int(), outlier_cols.int(), outlier_values
        )
        if row_stats is None or col_stats is None:
            A[outlier_indices] = 0 # zero out outliers
            row_stats, col_stats = get_row_col_stats(A)
            A[outlier_indices] = outlier_values # restore outliers for later use

    quant_by_row = quant_to_int8(A, row_stats.unsqueeze(-1))
    quant_by_col = quant_to_int8(A, col_stats.unsqueeze(0))
    if out_row is not None:
        out_row.copy_(quant_by_row)
    else:
        out_row = quant_by_row
    if out_col is not None:
        out_col.copy_(quant_by_col)
    else:
        out_col = quant_by_col
    return out_row, out_col, row_stats, col_stats, coo_tensor


def igemmlt_common(
    A, B, SA=None, SB=None, out=None, Sout=None, dtype=torch.int32
):
    """
    Do GEMMM computation. Data type: int8 * int8 -> int32.
    Args:
        A Activation of linear, data type is int8
        B Weight of linear, data type is int8
        SA Not used for CPU/XPU
        SB Not used for CPU/XPU
        out Specified output tensor if it is not None
        Sout Not used for CPU/XPU but returned as is
        dtype Data type of output
    Return:
        A tuple of GEMM result in dtype and Sout
    """
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    if out is not None:
        assert out.dtype == dtype

    dimsA = A.ndim
    dimsB = B.ndim
    shapeA = A.shape
    shapeB = B.shape
    assert dimsA in [2, 3], 'Only two or three dimensional matrices are supported for argument A'
    assert dimsB == 2, 'Only two dimensional matrices are supported for argument B'

    if dimsA == 2:
        m = shapeA[0]
    elif dimsA == 3:
        m = shapeA[0] * shapeA[1]
    n = shapeB[0]
    k = shapeA[-1]
    assert shapeA[-1] == shapeB[-1], f'Shapes of A and B do not match, got {shapeA} and {shapeB}'
    shapeOut = (shapeA[0], shapeA[1], n) if dimsA == 3 else (m, n)

    # if the tensor is empty, return a transformed empty tensor with the right dimensions
    if shapeA[0] == 0 and dimsA == 2:
        return torch.empty((0, n), device=A.device, dtype=A.dtype)
    elif shapeA[1] == 0 and dimsA == 3:
        return torch.empty(tuple(shapeA[:2] + [n]), device=A.device, dtype=A.dtype)

    A_reshaped = A.reshape(m, k)

    if assert_on_cpu([A_reshaped, B]):
        C = torch._int_mm(A_reshaped, B.T).to(dtype)
    else:
        C = torch.nn.functional.linear(A_reshaped, B).to(dtype)
    if C.ndim != dimsA:
        C = C.reshape(shapeOut)
    if out is not None:
        out.copy_(C)
    else:
        out = C

    return out, Sout


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def mm_dequant_common(
    A,
    quant_state,
    row_stats,
    col_stats,
    out=None,
    new_row_stats=None,
    new_col_stats=None,
    bias=None,
    compute_dtype=torch.float32,
    output_dtype=torch.float32
):
    """
    Dequant and add bias
    out = A_int32 * (scale_A, scale_B) / 127 * 127 + bias
    Args:
        A The output of int8 gemm, whose dtype is int32
        quant_state Not used for CPU
        row_stats Absolute max value of each row of input (A) of gemm
        col_stats Absolute max value of each row of weight (B) of gemm
        out Output buffer
        new_row_stats Not used for CPU/XPU
        new_col_stats Not used for CPU/XPU
        bias Bias of linear
        compute_dtype Data type for computation
        output_dtype Data type for output
    Return:
        The result
    """
    assert A.dtype == torch.int32
    out_shape = A.shape
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])

    A_reshaped = A.reshape(out_shape).to(compute_dtype)
    row_stats = row_stats.reshape(-1).unsqueeze(-1).to(compute_dtype)
    col_stats = col_stats.reshape(-1).unsqueeze(0).to(compute_dtype)
    out = A_reshaped * row_stats * col_stats / (127 * 127)
    if bias is not None:
        out = out + bias.to(compute_dtype)
    out = out.to(output_dtype)
    return out


class CPUBackend:
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    @classmethod
    def double_quant(
        cls, A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
    ):
        assert_on_cpu([A, col_stats, row_stats, out_col, out_row])
        return double_quant_common(A, col_stats, row_stats, out_col, out_row)

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
        return igemmlt_common(A, B, SA, SB, out, Sout, dtype)

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
