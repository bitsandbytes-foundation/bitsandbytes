import torch


class COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        assert rowidx.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class BackendInterface:
    _instance = None

    def __new__(cls, lib=None):
        if cls._instance is None:
            if lib is None:
                raise ValueError(
                    "A 'lib' binary must be provided during the first initialization of BackendInterface."
                )
            cls._instance = super().__new__(cls)
            cls._instance.lib = (
                lib  # Set the binary name during the first and only instantiation
            )
        else:
            if lib is not None:
                raise ValueError(
                    "The BackendInterface singleton has already been initialized with a 'lib' value. Re-initialization with a new 'lib' value is not allowed."
                )
        return cls._instance

    def check_matmul(
        self,
        A,
        B,
        out=None,
        transposed_A=False,
        transposed_B=False,
        expected_type=torch.int8,
    ):
        """
        Checks if the matrix multiplication between A and B can be performed, considering their shapes,
        whether they are transposed, and their data types. It also determines the shape of the output tensor.

        Parameters:
        - A (torch.Tensor): The first matrix in the multiplication.
        - B (torch.Tensor): The second matrix in the multiplication.
        - out (torch.Tensor, optional): The output tensor to store the result of the multiplication. Default is None.
        - transposed_A (bool, optional): Indicates if matrix A is transposed. Default is False.
        - transposed_B (bool, optional): Indicates if matrix B is transposed. Default is False.
        - expected_type (torch.dtype, optional): The expected data type of matrices A and B. Default is torch.int8.

        Returns:
        - tuple: The shape of the output tensor resulting from the matrix multiplication.

        Raises:
        - TypeError: If the data types of A or B do not match the expected type.
        - ValueError: If the dimensions of A and B are not compatible for matrix multiplication.
        """
        raise NotImplementedError

    # 8-bit matmul interface
    def coo_zeros(self, rows, cols, nnz, device, dtype=torch.half):
        rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
        colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
        values = torch.zeros((nnz,), dtype=dtype, device=device)

        return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)

    def get_colrow_absmax(
        self, A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
    ):
        raise NotImplementedError

    def double_quant(
        self,
        A,
        col_stats=None,
        row_stats=None,
        out_col=None,
        out_row=None,
        threshold=0.0,
    ):
        raise NotImplementedError

    def extract_outliers(self, *args, **kwargs):
        raise NotImplementedError

    def igemmlt(self, *args, **kwargs):
        raise NotImplementedError

    def mm_dequant(self, *args, **kwargs):
        raise NotImplementedError

    # k-bit quantization interface
    def create_quant_map(self, interface, quant_name):
        """
        Below functions should be abstracted into a general method
        "create_quant_map(interface, "quant_name")", so we can call e.g.
        create_quant_map(..., quant_name='normal'):
            - 'create_dynamic_map'
            - 'create_fp8_map'
            - 'create_linear_map'
            - 'create_normal_map'
            - 'create_quantile_map'
        """
        raise NotImplementedError

    def estimate_quantiles(self, *args, **kwargs):
        raise NotImplementedError

    def dequantize_blockwise(self, *args, **kwargs):
        raise NotImplementedError

    def quantize_blockwise(self, *args, **kwargs):
        raise NotImplementedError

    # 4-bit matmul interface
    def dequantize_4bit(self, *args, **kwargs):
        raise NotImplementedError

    def quantize_4bit(self, *args, **kwargs):
        raise NotImplementedError

    def gemv_4bit(self, *args, **kwargs):
        raise NotImplementedError

    # 8-bit optimizer interface
    def optimizer_update_32bit(self, *args, **kwargs):
        """This is needed for tests"""
        raise NotImplementedError("Subclasses must implement 'optimizer_update_32bit'.")

    def optimizer_update_8bit_blockwise(self, *args, **kwargs):
        raise NotImplementedError
