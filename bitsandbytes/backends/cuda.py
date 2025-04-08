import ctypes as ct
from typing import Literal, Optional, Tuple

import torch

from bitsandbytes.cextension import HIP_ENVIRONMENT, lib
from bitsandbytes.functional import (
    CUBLAS_Context,
    _cuda_device_of,
    _get_tensor_stream,
    dequantize_blockwise,
    dtype2bytes,
    get_4bit_type,
    get_colrow_absmax,
    get_ptr,
    get_transform_buffer,
    is_on_gpu,
    nvidia_transform,
    post_call,
    pre_call,
    prod,
    quantize_blockwise,
)
from bitsandbytes.utils import QuantState

from .base import Backend

if lib and lib.compiled_with_cuda:
    """C FUNCTIONS FOR OPTIMIZERS"""
    str2optimizer32bit = {
        "adam": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum32bit_grad_32,
            lib.cmomentum32bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop32bit_grad_32,
            lib.crmsprop32bit_grad_16,
        ),
        "lion": (
            lib.clion32bit_grad_fp32,
            lib.clion32bit_grad_fp16,
            lib.clion32bit_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad32bit_grad_32,
            lib.cadagrad32bit_grad_16,
        ),
        "lamb": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "ademamix": (
            lib.cademamix32bit_grad_fp32,
            lib.cademamix32bit_grad_fp16,
            lib.cademamix32bit_grad_bf16,
        ),
    }

    str2optimizer8bit_blockwise = {
        "adam": (
            lib.cadam_8bit_blockwise_grad_fp32,
            lib.cadam_8bit_blockwise_grad_fp16,
            lib.cadam_8bit_blockwise_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum_8bit_blockwise_grad_fp32,
            lib.cmomentum_8bit_blockwise_grad_fp16,
            lib.cmomentum_8bit_blockwise_grad_bf16,
        ),
        "rmsprop": (
            lib.crmsprop_8bit_blockwise_grad_fp32,
            lib.crmsprop_8bit_blockwise_grad_fp16,
            lib.crmsprop_8bit_blockwise_grad_bf16,
        ),
        "lion": (
            lib.clion_8bit_blockwise_grad_fp32,
            lib.clion_8bit_blockwise_grad_fp16,
            lib.clion_8bit_blockwise_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad_8bit_blockwise_grad_fp32,
            lib.cadagrad_8bit_blockwise_grad_fp16,
            lib.cadagrad_8bit_blockwise_grad_bf16,
        ),
        "ademamix": (
            lib.cademamix_8bit_blockwise_grad_fp32,
            lib.cademamix_8bit_blockwise_grad_fp16,
            lib.cademamix_8bit_blockwise_grad_bf16,
        ),
    }


class CUDABackend(Backend):
    def device_synchronize(self):
        torch.cuda.synchronize()

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
        if HIP_ENVIRONMENT:
            # transform kernel formats (col32/col_turing/col_ampere) are not applicable to ROCm
            # Use nvidia_transform instead
            return nvidia_transform(A, to_order, from_order, out, transpose, state, ld)

        prev_device = pre_call(A.device)
        if state is None:
            state = (A.shape, from_order)
        else:
            from_order = state[1]

        if out is None:
            out, new_state = get_transform_buffer(state[0], A.dtype, A.device, to_order, state[1], transpose)
        else:
            new_state = (state[0], to_order)  # (shape, order)

        shape = state[0]
        if len(shape) == 2:
            dim1 = ct.c_int32(shape[0])
            dim2 = ct.c_int32(shape[1])
        else:
            dim1 = ct.c_int32(shape[0] * shape[1])
            dim2 = ct.c_int32(shape[2])

        is_on_gpu([A, out])
        if to_order == "col32":
            if transpose:
                lib.ctransform_row2col32T(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2col32(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "col_turing":
            if transpose:
                lib.ctransform_row2turingT(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2turing(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "col_ampere":
            if transpose:
                lib.ctransform_row2ampereT(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2ampere(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "row":
            if from_order == "col_turing":
                lib.ctransform_turing2row(get_ptr(A), get_ptr(out), dim1, dim2)
            elif from_order == "col_ampere":
                lib.ctransform_ampere2row(get_ptr(A), get_ptr(out), dim1, dim2)

        else:
            raise NotImplementedError(f"Transform function not implemented: From {from_order} to {to_order}")

        post_call(prev_device)

        return out, new_state

    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # TODO: Optimize/write CUDA kernel for this?

        # Use CUDA kernel for rowwise and COO tensor
        quant_row, row_stats, outlier_cols = self.int8_vectorwise_quant(A, threshold=threshold)

        # PyTorch impl for colwise
        _, col_stats, outlier_mask = get_colrow_absmax(A, threshold=threshold)
        if threshold > 0.0 and outlier_mask is not None:
            A = A.masked_fill(outlier_mask, 0.0)
        quant_col = torch.round(A.mul(127) / col_stats.unsqueeze(0)).to(torch.int8)

        if out_row is not None:
            quant_row = out_row.copy_(quant_row)
        if out_col is not None:
            quant_col = out_col.copy_(quant_col)

        return quant_row, quant_col, row_stats, col_stats.flatten().float(), outlier_cols

    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        #
        # To use the IMMA tensor core kernels without special Turing/Ampere layouts,
        # cublasLt has some rules, namely: A must be transposed, B must not be transposed.
        # The C++ API will calculate `C = A.T @ B` in with A, B, C in col-major.
        # This will typically be used with row-major tensors to efficiently
        # calculate the linear layer with `C = B @ A.T` without any transformations.
        # We will swap A and B in the API invocation, so that we get `C = A @ B.T`.
        #
        # Quick explanation:
        # With row-major A and B tensors, `C = A.T.T @ B.T = A @ B.T`.
        # To get row-major output, `C.T = (A @ B.T).T = B @ A.T`.
        #
        A, B = B, A

        shapeA = A.shape
        shapeB = B.shape

        assert A.dtype == torch.int8
        assert B.dtype == torch.int8
        assert A.ndim == 2, "Only two dimensional matrices are supported for argument B"
        assert B.ndim in [2, 3], "Only two or three dimensional matrices are supported for argument A"
        assert prod(shapeB) > 0, f"Input tensor dimensions need to be > 0: {shapeB}"
        assert out is None or out.dtype == dtype

        shapeC = (*shapeB[:-1], shapeA[0])

        k, m = shapeA
        n = prod(shapeB[:-1])
        lda = shapeA[-1]  # Weights (outputs, inputs)
        ldb = shapeB[-1]  # Activations (batch, tokens, inputs)
        ldc = shapeC[-1]  # Output (batch, tokens, outputs)

        assert (
            lda == ldb
        ), f"int8_linear_matmul only supports B^T @ A. Inner dimensions do not match: B @ A = {shapeB} @ {shapeA}"

        # cuBLASLt does not support int8 matmul with inner dimensions that are not divisible by 4.
        # We'll fall back to a slower fp32 calculation in this circumstance.
        # Fortunately, this should not be very common.
        if lda % 4 != 0:
            result = torch.matmul(B.float(), A.float().t()).to(torch.int32)
            if out is not None:
                result = out.copy_(result)
            return result

        if out is None:
            out = torch.empty(shapeC, device=A.device, dtype=dtype)

        is_on_gpu([A, B, out])

        with _cuda_device_of(A):
            ctx = CUBLAS_Context.get_instance().get_context(A.device)
            ptrA = get_ptr(A)
            ptrB = get_ptr(B)
            ptrC = get_ptr(out)
            ptrRowScale = None
            m = ct.c_int32(m)
            n = ct.c_int32(n)
            k = ct.c_int32(k)
            lda = ct.c_int32(lda)
            ldb = ct.c_int32(ldb)
            ldc = ct.c_int32(ldc)
            stream = _get_tensor_stream(A)

            if dtype == torch.int32:
                has_error = lib.cigemmlt_32(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)
            else:
                has_error = lib.cigemmlt_8(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)

        if has_error == 100:  # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
            raise NotImplementedError("int8_linear_matmul not implemented!")

        if has_error:
            raise RuntimeError(
                f"cublasLt ran into an error!\n"
                f"\t{shapeA=}, {shapeB=}, {shapeC=}\n"
                f"\t{(lda, ldb, ldc)=}\n"
                f"\t{(m, n, k)=}"
            )

        return out

    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert A.dtype == torch.int32

        if bias is not None:
            assert bias.dtype == torch.float16

        if out is None:
            out = torch.empty_like(A, dtype=torch.float16)

        ptrA = get_ptr(A)
        ptrOut = get_ptr(out)
        ptrRowStats = get_ptr(row_stats)
        ptrColStats = get_ptr(col_stats)
        ptrBias = get_ptr(bias)
        numRows = ct.c_int32(prod(A.shape[:-1]))
        numCols = ct.c_int32(A.shape[-1])

        is_on_gpu([A, row_stats, col_stats, out, bias])

        with _cuda_device_of(A):
            lib.cdequant_mm_int32_fp16(
                ptrA, ptrRowStats, ptrColStats, ptrOut, ptrBias, numRows, numCols, _get_tensor_stream(A)
            )

        return out

    def int8_vectorwise_dequant(
        self,
        A: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        # To dequantize we divide by 127, or multiply by the reciprocal.
        return A * stats.view(-1, 1) * 7.874015718698502e-3

    def int8_vectorwise_quant(
        self,
        A: torch.Tensor,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert A.dtype == torch.half
        is_on_gpu([A])

        rows = prod(A.shape[:-1])
        cols = A.shape[-1]

        row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
        out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)

        outlier_cols = None

        if threshold > 0.0:
            # TODO we could improve perf of this
            outliers = A.abs() >= threshold

            if outliers.any():
                outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)

        with _cuda_device_of(A):
            lib.cint8_vector_quant(
                get_ptr(A),
                get_ptr(out_row),
                get_ptr(row_stats),
                ct.c_float(threshold),
                ct.c_int32(rows),
                ct.c_int32(cols),
                _get_tensor_stream(A),
            )

        # Zero out values from outlier columns across all rows.
        # The kernel will handle this for outliers themselves, so we can optimize for rows=1.
        if rows > 1 and outlier_cols is not None:
            out_row[:, outlier_cols] = 0

        return out_row, row_stats, outlier_cols

    def extract_outliers(self, A: torch.Tensor, SA: Tuple[torch.Size, str], idx: torch.Tensor):
        shapeA = SA[0]
        formatA = SA[1]
        if not HIP_ENVIRONMENT:
            assert formatA in ["col_turing", "col_ampere"]
        else:
            # HIP uses col format
            assert formatA in ["col"]
        assert A.device.type == "cuda"

        out = torch.zeros((shapeA[0], idx.numel()), dtype=torch.int8, device=A.device)

        idx_size = ct.c_int32(idx.numel())
        rows = ct.c_int32(shapeA[0])
        cols = ct.c_int32(shapeA[1])
        ptrA = get_ptr(A)
        ptrIdx = get_ptr(idx)
        ptrOut = get_ptr(out)

        prev_device = pre_call(A.device)

        if formatA == "col_turing" or HIP_ENVIRONMENT:
            lib.cextractOutliers_turing(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
        elif formatA == "col_ampere":
            lib.cextractOutliers_ampere(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)

        post_call(prev_device)

        return out

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if blocksize is None:
            # Some AMD GPUs have warpsize 64
            # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
            blocksize = 64 if not HIP_ENVIRONMENT else 128
        if A.device.type != "cuda":
            raise NotImplementedError(f"Device type not supported for FP4 quantization: {A.device.type}")
        if quant_type not in ["fp4", "nf4"]:
            raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")

        n = A.numel()
        input_shape = A.shape

        if absmax is None:
            blocks = n // blocksize
            blocks += 1 if n % blocksize > 0 else 0
            absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

        if out is None:
            mod = dtype2bytes[quant_storage] * 2
            out = torch.zeros(((n + 1) // mod, 1), dtype=quant_storage, device=A.device)

        # Some AMD GPUs have warpsize 64
        # Set min blocksize to 128 (~warpsize 64 in kernel) for HIP
        if not HIP_ENVIRONMENT:
            assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        else:
            assert blocksize in [4096, 2048, 1024, 512, 256, 128]

        is_on_gpu([A, out, absmax])

        with _cuda_device_of(A):
            args = (
                None,
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int32(blocksize),
                ct.c_int(n),
            )

            if A.dtype == torch.bfloat16:
                if quant_type == "fp4":
                    lib.cquantize_blockwise_bf16_fp4(*args)
                else:
                    lib.cquantize_blockwise_bf16_nf4(*args)
            elif A.dtype == torch.float16:
                if quant_type == "fp4":
                    lib.cquantize_blockwise_fp16_fp4(*args)
                else:
                    lib.cquantize_blockwise_fp16_nf4(*args)
            elif A.dtype == torch.float32:
                if quant_type == "fp4":
                    lib.cquantize_blockwise_fp32_fp4(*args)
                else:
                    lib.cquantize_blockwise_fp32_nf4(*args)
            else:
                raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

        code = get_4bit_type(quant_type, device=A.device)

        if compress_statistics:
            offset = absmax.mean()
            absmax -= offset
            qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
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

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        # Some AMD GPUs have warpsize 64
        # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
        if blocksize is None:
            blocksize = 64 if not HIP_ENVIRONMENT else 128
        supported_blocksizes = [2048, 4096, 1024, 512, 256, 128, 64]
        if HIP_ENVIRONMENT:
            supported_blocksizes = supported_blocksizes[:-1]
        if blocksize not in supported_blocksizes:
            raise ValueError(
                f"The blockwise of {blocksize} is not supported. Supported values: {supported_blocksizes}"
            )

        if quant_type not in ["fp4", "nf4"]:
            raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")

        if quant_state is None:
            assert absmax is not None and out is not None

            quant_state = QuantState(
                absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type
            )
        else:
            absmax = quant_state.absmax

        if quant_state.nested:
            absmax = self.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax += quant_state.offset
            if absmax.dtype != torch.float32:
                absmax = absmax.float()

        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

        n = out.numel()

        is_on_gpu([A, absmax, out])
        stream = _get_tensor_stream(A)

        with _cuda_device_of(A):
            args = (
                None,
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(n),
                stream,
            )

            if out.dtype == torch.bfloat16:
                if quant_state.quant_type == "fp4":
                    lib.cdequantize_blockwise_bf16_fp4(*args)
                else:
                    lib.cdequantize_blockwise_bf16_nf4(*args)
            elif out.dtype == torch.float16:
                if quant_state.quant_type == "fp4":
                    lib.cdequantize_blockwise_fp16_fp4(*args)
                else:
                    lib.cdequantize_blockwise_fp16_nf4(*args)
            elif out.dtype == torch.float32:
                if quant_state.quant_type == "fp4":
                    lib.cdequantize_blockwise_fp32_fp4(*args)
                else:
                    lib.cdequantize_blockwise_fp32_nf4(*args)
            else:
                raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")

        if A.shape[0] == 1:  # is transposed, transpose back
            return out.t()

        return out

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ):
        if state is None:
            raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")

        if A.numel() != A.shape[-1]:
            raise ValueError(
                'Dimensions of A are invalid. Must be a vector with the leading dimensions of "1", e.g. [1, 1, 2048]',
            )

        Bshape = state.shape
        bout = Bshape[0]
        absmax = state.absmax
        if state.nested:
            absmax = self.dequantize_blockwise(state.absmax, state.state2)
            absmax += state.offset

        if out is None:
            if len(A.shape) == 3:
                out = torch.empty(size=(A.shape[0], A.shape[1], bout), dtype=A.dtype, device=A.device)
            else:
                out = torch.empty(size=(A.shape[0], bout), dtype=A.dtype, device=A.device)

        n = 1
        m = Bshape[0]
        k = Bshape[1]
        lda = Bshape[0]
        ldc = Bshape[0]
        ldb = (A.shape[-1] + 1) // 2
        is_on_gpu([B, A, out, absmax, state.code])
        m = ct.c_int32(m)
        n = ct.c_int32(n)
        k = ct.c_int32(k)
        lda = ct.c_int32(lda)
        ldb = ct.c_int32(ldb)
        ldc = ct.c_int32(ldc)

        inference_args = [
            m,
            n,
            k,
            get_ptr(A),
            get_ptr(B),
            get_ptr(absmax),
            get_ptr(state.code),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            ct.c_int32(state.blocksize),
            _get_tensor_stream(A),
        ]

        if B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32]:
            if A.dtype == torch.float16:
                lib.cgemm_4bit_inference_naive_fp16(*inference_args)
            elif A.dtype == torch.bfloat16:
                lib.cgemm_4bit_inference_naive_bf16(*inference_args)
            elif A.dtype == torch.float32:
                lib.cgemm_4bit_inference_naive_fp32(*inference_args)
            else:
                raise NotImplementedError(f"Matmul not implemented for data type {A.dtype}")
        else:
            raise NotImplementedError(f"Matmul not implemented for data type {A.dtype}")

        return out

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
        # TODO: Move from bnb.functional
        return dequantize_blockwise(
            A,
            quant_state=quant_state,
            absmax=absmax,
            code=code,
            out=out,
            blocksize=blocksize,
            nested=nested,
        )

    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        # TODO: Move from bnb.functional
        return quantize_blockwise(
            A,
            absmax=absmax,
            code=code,
            out=out,
            blocksize=blocksize,
            nested=nested,
        )

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

        is_on_gpu([g, p, state1, state2, qmap1, qmap2, absmax1, absmax2])
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

        is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])

        with _cuda_device_of(g):
            optim_func(
                get_ptr(p),
                get_ptr(g),
                get_ptr(state1),
                get_ptr(state2),
                ct.c_float(beta1),
                ct.c_float(beta2),
                ct.c_float(beta3),
                ct.c_float(alpha),
                ct.c_float(eps),
                ct.c_int32(step),
                ct.c_float(lr),
                get_ptr(qmap1),
                get_ptr(qmap2),
                get_ptr(absmax1),
                get_ptr(absmax2),
                ct.c_float(weight_decay),
                ct.c_float(gnorm_scale),
                ct.c_bool(skip_zeros),
                ct.c_int32(g.numel()),
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
        param_norm = 0.0
        if max_unorm > 0.0:
            param_norm = torch.norm(p.data.float())

        optim_func = None
        if g.dtype == torch.float32:
            optim_func = str2optimizer32bit[optimizer_name][0]
        elif g.dtype == torch.float16:
            optim_func = str2optimizer32bit[optimizer_name][1]
        elif g.dtype == torch.bfloat16 and len(str2optimizer32bit[optimizer_name]) == 3:
            optim_func = str2optimizer32bit[optimizer_name][2]
        else:
            raise ValueError(
                f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
            )

        is_on_gpu([g, p, state1, state2, unorm_vec])

        with _cuda_device_of(g):
            optim_func(
                get_ptr(g),
                get_ptr(p),
                get_ptr(state1),
                get_ptr(state2),
                get_ptr(unorm_vec),
                ct.c_float(max_unorm),
                ct.c_float(param_norm),
                ct.c_float(beta1),
                ct.c_float(beta2),
                ct.c_float(beta3),
                ct.c_float(alpha),
                ct.c_float(eps),
                ct.c_float(weight_decay),
                ct.c_int32(step),
                ct.c_float(lr),
                ct.c_float(gnorm_scale),
                ct.c_bool(skip_zeros),
                ct.c_int32(g.numel()),
            )
