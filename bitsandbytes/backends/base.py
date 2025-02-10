from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch

from bitsandbytes.utils import QuantState


class Backend(ABC):
    """Base class for devices backends that will implement their own 8bits and 4bits functions."""

    @abstractmethod
    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

        The statistics are determined both row-wise and column-wise (transposed).

        For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

        <Tip>
        This function is useful for training, but for inference it is advised to use [`int8_vectorwise_quant`] instead.
        This implementation performs additional column-wise transposed calculations which are not optimized.
        </Tip>

        Args:
            A (`torch.Tensor` with dtype `torch.float16`): The input matrix.
            col_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantization scales.
            row_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantization scales.
            out_col (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantized data.
            out_row (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantized data.
            threshold (`float`, *optional*):
                An optional threshold for sparse decomposition of outlier features.

                No outliers are held back when 0.0. Defaults to 0.0.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
            - `torch.Tensor` with dtype `torch.int8`: The row-wise quantized data.
            - `torch.Tensor` with dtype `torch.int8`: The column-wise quantized data.
            - `torch.Tensor` with dtype `torch.float32`: The row-wise quantization scales.
            - `torch.Tensor` with dtype `torch.float32`: The column-wise quantization scales.
            - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
        """
        ...

    @abstractmethod
    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        """Performs an 8-bit integer matrix multiplication.

        A linear transformation is applied such that `out = A @ B.T`. When possible, integer tensor core hardware is
        utilized to accelerate the operation.

        Args:
            A (`torch.Tensor`): The first matrix operand with the data type `torch.int8`.
            B (`torch.Tensor`): The second matrix operand with the data type `torch.int8`.
            out (`torch.Tensor`, *optional*): A pre-allocated tensor used to store the result.
            dtype (`torch.dtype`, *optional*): The expected data type of the output. Defaults to `torch.int32`.

        Raises:
            `NotImplementedError`: The operation is not supported in the current environment.
            `RuntimeError`: Raised when the cannot be completed for any other reason.

        Returns:
            `torch.Tensor`: The result of the operation.
        """
        ...

    @abstractmethod
    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs dequantization on the result of a quantized int8 matrix multiplication.

        Args:
            A (`torch.Tensor` with dtype `torch.int32`): The result of a quantized int8 matrix multiplication.
            row_stats (`torch.Tensor`): The row-wise quantization statistics for the lhs operand of the matrix multiplication.
            col_stats (`torch.Tensor`): The column-wise quantization statistics for the rhs operand of the matrix multiplication.
            out (`torch.Tensor`, *optional*): A pre-allocated tensor to store the output of the operation.
            bias (`torch.Tensor`, *optional*): An optional bias vector to add to the result.

        Returns:
            `torch.Tensor`: The dequantized result with an optional bias, with dtype `torch.float16`.
        """
        ...

    @abstractmethod
    def int8_vectorwise_dequant(self, A: torch.Tensor, stats: torch.Tensor):
        """Dequantizes a tensor with dtype `torch.int8` to `torch.float32`.

        Args:
            A (`torch.Tensor` with dtype `torch.int8`): The quantized int8 tensor.
            stats (`torch.Tensor` with dtype `torch.float32`): The row-wise quantization statistics.

        Returns:
            `torch.Tensor` with dtype `torch.float32`: The dequantized tensor.
        """
        # To dequantize we divide by 127, or multiply by the reciprocal.
        return A * stats.view(-1, 1) * 7.874015718698502e-3

    @abstractmethod
    def int8_vectorwise_quant(self, A: torch.Tensor, threshold=0.0):
        """Quantizes a tensor with dtype `torch.float16` to `torch.int8` in accordance to the `LLM.int8()` algorithm.

        For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

        Args:
            A (`torch.Tensor` with dtype `torch.float16`): The input tensor.
            threshold (`float`, *optional*):
                An optional threshold for sparse decomposition of outlier features.

                No outliers are held back when 0.0. Defaults to 0.0.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
            - `torch.Tensor` with dtype `torch.int8`: The quantized data.
            - `torch.Tensor` with dtype `torch.float32`: The quantization scales.
            - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
        """
        ...

    @abstractmethod
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

    @abstractmethod
    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
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
        """Quantize tensor A in blocks of 4-bit values.

        Quantizes tensor A by dividing it into blocks which are independently quantized.

        Args:
            A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
            absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
            out (`torch.Tensor`, *optional*): A tensor to use to store the result.
            blocksize (`int`, *optional*):
                The size of the blocks. Defaults to 64.
                Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            compress_statistics (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.
            quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.
            quant_storage (`torch.dtype`, *optional*): The dtype of the tensor used to store the result. Defaults to `torch.uint8`.

        Raises:
            ValueError: Raised when the input data type is not supported.

        Returns:
            Tuple[`torch.Tensor`, `QuantState`]: A tuple containing the quantization results.
            - `torch.Tensor`: The quantized tensor with packed 4-bit values.
            - [`QuantState`]: The state object used to undo the quantization.
        """
        ...

    @abstractmethod
    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        """Dequantizes a packed 4-bit quantized tensor.

        The input tensor is dequantized by dividing it into blocks of `blocksize` values.
        The the absolute maximum value within these blocks is used for scaling
        the non-linear dequantization.

        Args:
            A (`torch.Tensor`): The quantized input tensor.
            quant_state ([`QuantState`], *optional*):
                The quantization state as returned by [`quantize_4bit`].
                Required if `absmax` is not provided.
            absmax (`torch.Tensor`, *optional*):
                A tensor containing the scaling values.
                Required if `quant_state` is not provided and ignored otherwise.
            out (`torch.Tensor`, *optional*): A tensor to use to store the result.
            blocksize (`int`, *optional*):
                The size of the blocks. Defaults to 64.
                Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.

        Raises:
            ValueError: Raised when the input data type or blocksize is not supported.

        Returns:
            `torch.Tensor`: The dequantized tensor.
        """
        ...

    @abstractmethod
    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        """Quantize a tensor in blocks of values.

        The input tensor is quantized by dividing it into blocks of `blocksize` values.
        The the absolute maximum value within these blocks is calculated for scaling
        the non-linear quantization.

        Args:
            A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
            code (`torch.Tensor`, *optional*):
                A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
                For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
            absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
            out (`torch.Tensor`, *optional*): A tensor to use to store the result.
            blocksize (`int`, *optional*):
                The size of the blocks. Defaults to 4096.
                Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            nested (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.

        Raises:
            ValueError: Raised when the input data type is not supported.

        Returns:
            `Tuple[torch.Tensor, QuantState]`: A tuple containing the quantization results.
            - `torch.Tensor`: The quantized tensor.
            - [`QuantState`]: The state object used to undo the quantization.
        """
        ...

    @abstractmethod
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
        """Dequantize a tensor in blocks of values.

        The input tensor is dequantized by dividing it into blocks of `blocksize` values.
        The the absolute maximum value within these blocks is used for scaling
        the non-linear dequantization.

        Args:
            A (`torch.Tensor`): The quantized input tensor.
            quant_state ([`QuantState`], *optional*):
                The quantization state as returned by [`quantize_blockwise`].
                Required if `absmax` is not provided.
            absmax (`torch.Tensor`, *optional*):
                A tensor containing the scaling values.
                Required if `quant_state` is not provided and ignored otherwise.
            code (`torch.Tensor`, *optional*):
                A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
                For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
                Ignored when `quant_state` is provided.
            out (`torch.Tensor`, *optional*): A tensor to use to store the result.
            blocksize (`int`, *optional*):
                The size of the blocks. Defaults to 4096.
                Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
                Ignored when `quant_state` is provided.

        Raises:
            ValueError: Raised when the input data type is not supported.

        Returns:
            `torch.Tensor`:
                The dequantized tensor. The datatype is indicated by `quant_state.dtype` and defaults to `torch.float32`.
        """
        ...

    @abstractmethod
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
        """
        Performs an in-place optimizer update with one or two optimizer states.

        Args:
            optimizer_name (`str`): The name of the optimizer, e.g. `adam`
            g (`torch.Tensor`): Gradient tensor.
            p (`torch.Tensor`): Parameter tensor.
            state1 (`torch.Tensor`): Optimizer state 1.
            state2 (`torch.Tensor`, optional): Optimizer state 2.
            beta1 (`float`): Optimizer beta1.
            beta2 (`float`): Optimizer beta2.
            eps (`float`): Optimizer epsilon.
            step (`int`): Current optimizer step.
            lr (`float`): The learning rate.
            qmap1 (`torch.Tensor`): Quantization map for the first state.
            qmap2 (`torch.Tensor`, optional): Quantization map for the second state.
            absmax1 (`torch.Tensor`): Max value for the first state update.
            absmax2 (`torch.Tensor`, optional): Max value for the second state update.
            weight_decay (`float`, optional): Weight decay. Defaults to 0.0.
            gnorm_scale (`float`, optional): The factor to rescale the gradient to the max clip value. Defaults to 1.0.
            skip_zeros (`bool`, optional): Whether to skip zero-valued gradients or not. Defaults to False.
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Performs an in-place optimizer update with one or two optimizer states.

        Universal optimizer update for 32-bit state and 32/16-bit gradients/weights

        Args:
            optimizer_name (`str`): The name of the optimizer, e.g. `adam`
            g (`torch.Tensor`): Gradient tensor.
            p (`torch.Tensor`): Parameter tensor.
            state1 (`torch.Tensor`): Optimizer state 1.
            beta1 (`float`): Optimizer beta1.
            eps (`float`): Optimizer epsilon.
            step (`int`): Current optimizer step.
            lr (`float`): The learning rate.
            state2 (`torch.Tensor`, optional): Optimizer state 2. Defaults to None.
            beta2 (`float`, optional): Optimizer beta2. Defaults to 0.0.
            weight_decay (`float`, optional): Defaults to 0.0.
            gnorm_scale (`float`, optional): The factor to rescale the gradient to the max clip value. Defaults to 1.0.
            unorm_vec (`torch.Tensor`, optional): The tensor for the update norm. Defaults to None.
            max_unorm (`float`, optional): The maximum update norm relative to the weight norm.. Defaults to 0.0.
            skip_zeros (`bool`, optional): Whether to skip zero-valued gradients or not. Defaults to False.
        """
        raise NotImplementedError
