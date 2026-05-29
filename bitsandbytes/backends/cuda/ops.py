from collections.abc import Sequence
import ctypes as ct
import functools
from math import prod
from typing import Optional
from warnings import warn

import torch

from bitsandbytes.functional import CUBLAS_Context, _cuda_device_of, get_ptr

from ..._ops import register_kernel
from ...cextension import lib
from ..default.ops import _gemm_4bit_default_impl
from ..utils import _get_4bit_code


def _setup_ctypes(names, argtypes, restype=None):
    for name in names:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype


# 4-bit/8-bit dequantize: (code, A, absmax, out, blocksize, numel, stream)
_setup_ctypes(
    [f"cdequantize_blockwise_{d}_{q}" for d in ("fp32", "bf16", "fp16") for q in ("nf4", "fp4")]
    + [f"cdequantize_blockwise_{d}" for d in ("fp32", "bf16", "fp16")],
    [ct.c_void_p] * 4 + [ct.c_int32, ct.c_int32, ct.c_void_p],
)

# 4-bit GEMM: (A, B, absmax, absmax_8bit, absmax_code, absmax_offset, out, bias, M, N, K, blocksize, quant_type, stream)
_setup_ctypes(
    [f"cgemm_4bit_{d}" for d in ("bf16", "fp16", "fp32")],
    [ct.c_void_p] * 8 + [ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_void_p],
)

# 4-bit GEMV: (m, n, k, A, B, absmax, code, out, lda, ldb, ldc, blocksize, stream)
_setup_ctypes(
    [f"cgemm_4bit_inference_naive_{d}" for d in ("bf16", "fp16", "fp32")],
    [ct.c_int32] * 3 + [ct.c_void_p] * 5 + [ct.c_int32] * 3 + [ct.c_int32, ct.c_void_p],
)

# int8 igemm: (ctx, m, n, k, A, B, C, rowscale, lda, ldb, ldc, stream) -> int32
_setup_ctypes(
    ["cigemmlt_32"],
    [ct.c_void_p] + [ct.c_int32] * 3 + [ct.c_void_p] * 4 + [ct.c_int32] * 3 + [ct.c_void_p],
    restype=ct.c_int32,
)

# int8 mm dequant: (A, row_stats, col_stats, out, bias, numRows, numCols, stream)
_setup_ctypes(
    ["cdequant_mm_int32_fp16"],
    [ct.c_void_p] * 5 + [ct.c_int32, ct.c_int32, ct.c_void_p],
)

# int8 vectorwise quant: (A, out, row_stats, threshold, rows, cols, stream)
_setup_ctypes(
    ["cint8_vector_quant"],
    [ct.c_void_p] * 3 + [ct.c_float, ct.c_int32, ct.c_int32, ct.c_void_p],
)

# 4-bit/8-bit blockwise quantize: (code, A, absmax, out, blocksize, n)
_setup_ctypes(
    [f"cquantize_blockwise_{d}_{q}" for d in ("fp32", "bf16", "fp16") for q in ("nf4", "fp4")]
    + [f"cquantize_blockwise_{d}" for d in ("fp32", "bf16", "fp16")],
    [ct.c_void_p] * 4 + [ct.c_int32, ct.c_int32],
)


_get_raw_stream = torch._C._cuda_getCurrentRawStream


@functools.cache
def _gpu_dispatch_props(device_index):
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count, props.major, props.minor


@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _(A: torch.Tensor, B: torch.Tensor):
    out = torch.empty((*A.shape[:-1], B.shape[0]), device=A.device, dtype=torch.int32)
    return _int8_linear_matmul_impl(A, B, out)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    A, B = B, A

    shapeA = A.shape
    shapeB = B.shape

    if A.dtype != torch.int8:
        raise ValueError("B must be int8")
    if B.dtype != torch.int8:
        raise ValueError("A must be int8")
    if A.ndim != 2:
        raise ValueError("Only two dimensional matrices are supported for argument B")
    if B.ndim not in (2, 3):
        raise ValueError("Only two or three dimensional matrices are supported for argument A")
    if prod(shapeB) <= 0:
        raise ValueError(f"Input tensor dimensions need to be > 0: {shapeB}")
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")

    shapeC = (*shapeB[:-1], shapeA[0])
    if out.shape != shapeC:
        raise ValueError(f"Output shape {out.shape} does not match expected shape {shapeC}")

    k, m = shapeA
    n = prod(shapeB[:-1])
    lda = shapeA[-1]  # Weights (outputs, inputs)
    ldb = shapeB[-1]  # Activations (batch, tokens, inputs)
    ldc = shapeC[-1]  # Output (batch, tokens, outputs)

    if lda != ldb:
        raise ValueError(
            f"int8_linear_matmul only supports B^T @ A. Inner dimensions do not match: B @ A = {shapeB} @ {shapeA}"
        )

    # cuBLASLt does not support int8 matmul with inner dimensions that are not divisible by 4.
    # We'll fall back to a slower fp32 calculation in this circumstance.
    # Fortunately, this should not be very common.
    if lda % 4 != 0:
        result = torch.matmul(B.float(), A.float().t()).to(torch.int32)
        return out.copy_(result)

    with _cuda_device_of(A):
        ctx = CUBLAS_Context.get_instance().get_context(A.device)
        has_error = lib.cigemmlt_32(
            ctx,
            m,
            n,
            k,
            A.data_ptr(),
            B.data_ptr(),
            out.data_ptr(),
            None,
            lda,
            ldb,
            ldc,
            _get_raw_stream(A.device.index),
        )

    if has_error:
        if has_error == 100:
            # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`. The HIP backend
            # also returns this when no usable hipBLASLt algo exists for the shape
            # (seen on MI300X for some small-n int8 gemms). Fall back to fp32 — same
            # path used for the `lda % 4 != 0` case above.
            import warnings

            warnings.warn(
                f"int8_linear_matmul has no usable (hip|cu)blasLt algo for shape "
                f"{shapeA=} {shapeB=}; falling back to fp32 matmul.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = torch.matmul(B.float(), A.float().t()).to(torch.int32)
            return out.copy_(result)
        else:
            raise RuntimeError(
                f"cublasLt ran into an error!\n\t{shapeA=}, {shapeB=}, {shapeC=}\n\t{(lda, ldb, ldc)=}\n\t{(m, n, k)=}"
            )

    return out


@register_kernel("bitsandbytes::int8_mm_dequant", "cuda")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if A.dtype != torch.int32:
        raise ValueError(f"A must be int32, got {A.dtype}")
    if row_stats.dtype != torch.float32:
        raise ValueError(f"row_stats must be float32, got {row_stats.dtype}")
    if col_stats.dtype != torch.float32:
        raise ValueError(f"col_stats must be float32, got {col_stats.dtype}")

    # Note: cuda kernel only currently supports fp16 output.
    # We'll later cast to desired dtype if needed.
    out = torch.empty_like(A, dtype=torch.float16)

    # Note: fused bias in the kernel is only supported for fp16
    # TODO(matthewdouglas): Consider supporting bf16 fused bias
    bias_ptr = bias.data_ptr() if bias is not None and bias.dtype == torch.float16 else None

    with _cuda_device_of(A):
        lib.cdequant_mm_int32_fp16(
            A.data_ptr(),
            row_stats.data_ptr(),
            col_stats.data_ptr(),
            out.data_ptr(),
            bias_ptr,
            A.numel() // A.shape[-1],
            A.shape[-1],
            _get_raw_stream(A.device.index),
        )

    # Add bias separately if not fused in kernel
    if bias is not None and bias.dtype != torch.float16:
        out.add_(bias)

    return out.to(dtype or torch.float16)


@register_kernel("bitsandbytes::int8_vectorwise_quant", "cuda")
def _(A: torch.Tensor, threshold=0.0):
    if A.dtype != torch.float16:
        raise ValueError(f"A must be float16, got {A.dtype}")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    rows = A.numel() // A.shape[-1]
    cols = A.shape[-1]

    row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)

    outlier_cols = None

    if threshold > 0.0:
        # TODO we could improve perf of this
        outliers = A.abs() >= threshold

        if outliers.any():
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

    with _cuda_device_of(A):
        lib.cint8_vector_quant(
            A.data_ptr(),
            out_row.data_ptr(),
            row_stats.data_ptr(),
            threshold,
            rows,
            cols,
            _get_raw_stream(A.device.index),
        )

    # Zero out values from outlier columns across all rows.
    # The kernel will handle this for outliers themselves, so we can optimize for rows=1.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    return out_row, row_stats, outlier_cols


@register_kernel("bitsandbytes::int8_double_quant", "cuda")
def _(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Use CUDA kernel for rowwise quant and outlier column detection
    quant_row, row_stats, outlier_cols = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        A,
        threshold=threshold,
    )

    # PyTorch impl for colwise
    col_stats, outlier_mask = _get_col_absmax(A, threshold=threshold)
    if threshold > 0.0 and outlier_mask is not None:
        A = A.masked_fill(outlier_mask, 0.0)
    quant_col = torch.round(A.mul(127.0) / col_stats.unsqueeze(0)).to(torch.int8)

    return quant_row, quant_col, row_stats, col_stats.flatten().float(), outlier_cols


def _get_col_absmax(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not A.is_floating_point():
        raise ValueError(f"A must be a floating point tensor, got {A.dtype}")

    outlier_mask = None

    absA = A.abs().view(-1, A.shape[-1])

    if threshold > 0.0:
        # Filter outliers from stats when enabled
        outlier_mask = absA >= threshold
        absA.masked_fill_(outlier_mask, 0.0)

    # shape [cols]; unsqueeze(0) gives [1,cols]
    col_stats = absA.amax(dim=0, keepdim=False).float()

    return col_stats, outlier_mask


@register_kernel("bitsandbytes::quantize_blockwise", "cuda")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    A = A.contiguous()

    if code.dtype != torch.float32:
        raise ValueError(f"code must be float32, got {code.dtype}")
    if blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError(f"invalid blocksize {blocksize}")

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    if A.dtype == torch.float32:
        fn = lib.cquantize_blockwise_fp32
    elif A.dtype == torch.float16:
        fn = lib.cquantize_blockwise_fp16
    elif A.dtype == torch.bfloat16:
        fn = lib.cquantize_blockwise_bf16
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

    with _cuda_device_of(A):
        fn(
            code.data_ptr(),
            A.data_ptr(),
            absmax.data_ptr(),
            out.data_ptr(),
            blocksize,
            n,
        )

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "cuda")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty_like(A, dtype=dtype)
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_blockwise.out", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    if out.dtype != dtype:
        raise ValueError(f"Expected out.dtype == {dtype}, got {out.dtype}")
    if out.shape != A.shape:
        raise ValueError(f"Expected out.shape == {A.shape}, got {out.shape}")
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)


def _dequantize_blockwise_impl(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    A = A.contiguous()

    if dtype == torch.float32:
        fn = lib.cdequantize_blockwise_fp32
    elif dtype == torch.float16:
        fn = lib.cdequantize_blockwise_fp16
    elif dtype == torch.bfloat16:
        fn = lib.cdequantize_blockwise_bf16
    else:
        raise ValueError(f"Blockwise dequantization only supports 16/32-bit floats, but got {dtype}")

    with _cuda_device_of(A):
        fn(
            code.data_ptr(),
            A.data_ptr(),
            absmax.data_ptr(),
            out.data_ptr(),
            blocksize,
            A.numel(),
            _get_raw_stream(A.device.index),
        )


@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    A = A.contiguous()
    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)

    if A.dtype == torch.bfloat16:
        if quant_type == "fp4":
            fn = lib.cquantize_blockwise_bf16_fp4
        else:
            fn = lib.cquantize_blockwise_bf16_nf4
    elif A.dtype == torch.float16:
        if quant_type == "fp4":
            fn = lib.cquantize_blockwise_fp16_fp4
        else:
            fn = lib.cquantize_blockwise_fp16_nf4
    elif A.dtype == torch.float32:
        if quant_type == "fp4":
            fn = lib.cquantize_blockwise_fp32_fp4
        else:
            fn = lib.cquantize_blockwise_fp32_nf4

    with _cuda_device_of(A):
        fn(
            None,
            A.data_ptr(),
            absmax.data_ptr(),
            out.data_ptr(),
            blocksize,
            n,
        )

    return out, absmax


@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    if out.shape != tuple(shape):
        raise ValueError(f"Expected out.shape == {shape}, got {out.shape}")
    if out.dtype != dtype:
        raise ValueError(f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    A = A.contiguous()

    if dtype == torch.bfloat16:
        if quant_type == "fp4":
            fn = lib.cdequantize_blockwise_bf16_fp4
        else:
            fn = lib.cdequantize_blockwise_bf16_nf4
    elif dtype == torch.float16:
        if quant_type == "fp4":
            fn = lib.cdequantize_blockwise_fp16_fp4
        else:
            fn = lib.cdequantize_blockwise_fp16_nf4
    elif dtype == torch.float32:
        if quant_type == "fp4":
            fn = lib.cdequantize_blockwise_fp32_fp4
        else:
            fn = lib.cdequantize_blockwise_fp32_nf4
    else:
        raise ValueError(f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}")

    with _cuda_device_of(A):
        fn(
            None,
            A.data_ptr(),
            absmax.data_ptr(),
            out.data_ptr(),
            blocksize,
            out.numel(),
            _get_raw_stream(A.device.index),
        )


@register_kernel("bitsandbytes::gemv_4bit", "cuda")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    shape = (*A.shape[:-1], shapeB[0])
    out = torch.empty(shape, device=A.device, dtype=A.dtype)
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)
    return out


@register_kernel("bitsandbytes::gemv_4bit.out", "cuda")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    expected_shape = (*A.shape[:-1], shapeB[0])
    if out.shape != expected_shape:
        raise ValueError(f"Expected out.shape == {expected_shape}, got {out.shape}")
    if out.dtype != A.dtype:
        raise ValueError(f"Expected out.dtype == {A.dtype}, got {out.dtype}")
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    if blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError(f"invalid blocksize {blocksize}")

    # Note: these checks are not strictly necessary, and cost more than they are worth, so they are commented out for now.
    # torch._check(
    #     A.numel() == A.size(-1),
    #     lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}",
    # )
    # torch._check(
    #     A.dtype in [torch.float16, torch.bfloat16, torch.float32],
    #     lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    # )
    # torch._check(
    #     B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
    #     lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    # )
    # torch._check(absmax.dtype == torch.float32, lambda: f"absmax must be float32, got {absmax.dtype}")
    # torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    m = shapeB[0]
    n = 1
    k = shapeB[1]

    lda = m
    ldb = (A.shape[-1] + 1) // 2
    ldc = m

    if A.dtype == torch.float16:
        fn = lib.cgemm_4bit_inference_naive_fp16
    elif A.dtype == torch.bfloat16:
        fn = lib.cgemm_4bit_inference_naive_bf16
    elif A.dtype == torch.float32:
        fn = lib.cgemm_4bit_inference_naive_fp32

    with _cuda_device_of(A):
        fn(
            m,
            n,
            k,
            A.data_ptr(),
            B.data_ptr(),
            absmax.data_ptr(),
            code.data_ptr(),
            out.data_ptr(),
            lda,
            ldb,
            ldc,
            blocksize,
            _get_raw_stream(A.device.index),
        )


@functools.cache
def _gemm_4bit_use_custom(device_index, dtype, M, N, K):
    """Custom kernel vs dequant+F.linear heuristic for M in [5, 1536].

    Per-arch notes (bf16/fp16, M >= 8, large weight):
      sm75 (T4, ~300 GB/s GDDR6):    fp16 MMA only; GDDR makes dequant expensive.
      sm80 (A100, ~2 TB/s HBM2e):    mma.sync; HBM thresholds; K-heavy shapes handled explicitly.
      sm86 (A10, ~600 GB/s GDDR6):   dedicated block; wider M caps than sm89 at medium N.
      sm89 (4090, L40S, GDDR6X):     default fallback; tall-K and large-N get higher M caps.
      sm90 (H100/H200, HBM3/HBM3e):  dequant+linear is much faster; thresholds are tight.
      sm100 (B200/B300, HBM3e):       exits early at top of function.
      sm120 (RTX 5000, GDDR7):        dedicated block; medium-N tiers differ from sm89.
    """
    num_sms, major, minor = _gpu_dispatch_props(device_index)
    n_blocks = (N + 63) // 64

    # fp32 has no MMA kernel; pre-sm75 has no MMA kernel; sm75 has fp16 MMA only.
    # For all of these, custom only wins in the SIMT range (M<8).
    if dtype == torch.float32 or major < 7:
        return M < 8
    if major == 7 and (minor < 5 or dtype != torch.float16):
        return M < 8

    # sm87 and sm110: no calibration data, conservative fallback.
    if (major == 8 and minor == 7) or major == 11:
        return False

    # sm100 (B200/B300): dequant+F.linear is significantly faster than our mma.sync kernel.
    if major == 10:
        if n_blocks >= num_sms * 3:
            return M <= 32
        if n_blocks >= num_sms:
            return False if K >= N else M <= 8
        return False

    is_sm75 = major == 7 and minor == 5
    is_sm80 = major == 8 and minor == 0
    is_sm86 = major == 8 and minor == 6
    is_sm90 = major == 9
    is_sm120 = major == 12 and minor == 0
    is_hbm = is_sm80 or is_sm90  # sm100 already returned above
    tall_k_2xn = K > N * 2

    # Small-weight path (N*K < 4MB): dequant overhead dominates.
    if N * K < 4 * 1024 * 1024:
        if K * 2 < N:
            # Very short K (K < N/2): latency-dominated, custom 3-9x cheaper.
            if is_hbm:
                # Calibrated on A100: custom wins to M=1536 (low wave), M=512 (high wave).
                # Calibrated on H100/H200: custom wins to M=512 (low wave), M=320 (high wave).
                low_wave = n_blocks * 3 < num_sms
                if is_sm80:
                    return M <= (1536 if low_wave else 512)
                return M <= (512 if low_wave else 320)
            if is_sm75:
                # T4: wins require >=3 waves; M cap scales with K depth.
                if n_blocks >= num_sms * 3:
                    return M <= 320
                if K >= 1024:
                    return M <= 64
                if K >= 704:
                    return M <= 96
                return M <= 320
            # sm86/sm89/sm120: well-subscribed wins to M=320; undersubscribed tighter.
            if n_blocks >= num_sms:
                return M <= 320
            return M <= 192 if n_blocks * K > num_sms * 320 else M <= 320
        # K*2 >= N: arch-specific handling at low occupancy.
        quarter_wave = n_blocks * 4 <= num_sms
        if is_sm80 and quarter_wave:
            # A100 <1/4 wave: K>=N loses earlier (K-tiling efficient on HBM2e).
            if K >= N:
                return M <= (32 if n_blocks * 8 <= num_sms else 128)
            return M <= 384
        # T4 <1 wave non-short-K: M>8 routes through occupancy caps below.
        if is_sm75 and n_blocks < num_sms and M > 8:
            return M <= 64
        # General tiers (sm90, sm86, sm89, sm120):
        # GDDR tall-K (K>=N) at <1/4 wave: K-tiling in default impl wins above M=23.
        if quarter_wave:
            return M <= (32 if (K < N or is_hbm) else 23)
        if n_blocks * 2 <= num_sms:
            return M <= 16
        return False  # >=1/2 wave: no validated wins for remaining small-weight shapes

    # Non-small-weight: custom wins up to M=512; dequant+F.linear wins above that.
    if M > 512:
        return False

    # M=5-7: custom SIMT generally wins because dequant cost dominates.
    # Exceptions where K-tiling efficiency or MMA occupancy favors dequant+F.linear:
    #   HBM at M=6-7: tall-K (K>N) at ~3/4 MMA wave.
    #   sm90 square (K==N) at specific occupancy bands: arch-specific crossover.
    if M < 8:
        hbm_m67_thresh = 36 if is_sm90 else 48
        if is_hbm and M >= 6 and n_blocks >= hbm_m67_thresh:
            lt_75pct_wave = n_blocks * 4 < num_sms * 3
            lt_60pct_wave = n_blocks * 5 < num_sms * 3
            # Tall-K: K-tiling in default impl wins when under-subscribed.
            if K > N and lt_75pct_wave:
                return False
            # Square: arch-specific crossover around 0.6 wave.
            # A100 (HBM2e): loses below 0.6 wave. H100/H200 (HBM3/3e): loses above.
            if K == N:
                if is_sm80 and lt_60pct_wave:
                    return False
                if is_sm90 and lt_75pct_wave and not lt_60pct_wave:
                    return False
        return True

    # M in [8, 512]: per-arch tier ladders.

    if is_sm75:
        # fp16 MMA (m16n8k8). GDDR bandwidth makes dequant relatively expensive.
        if n_blocks >= num_sms * 3:
            return M <= (128 if K < N else 64)
        if n_blocks >= num_sms // 2:
            return M <= 64
        return M <= 32

    if is_sm80:
        # mma.sync (m16n8k16). HBM2e thresholds; K-heavy shapes handled explicitly.
        if n_blocks >= num_sms * 3:
            return M <= 128
        if n_blocks >= num_sms:
            return M <= (64 if K < N else 32)
        # Very tall-K (K>=3N) at >1/4 wave: K-tiling in default impl wins at all M.
        # Uses >= to catch K==3N (e.g. N=4096,K=12288 M=9-16: measured regression on A100).
        if K >= N * 3 and n_blocks * 4 > num_sms:
            return False
        # Square (K==N) at 0.5-1 wave: K-tiling wins at ~0.6 wave.
        # n_blocks>=48 excludes small N where SIMT still wins.
        if K == N and n_blocks >= 48 and n_blocks * 5 < num_sms * 3:
            return False
        # <0.5 wave: K<=N custom wins to M=128; K>N default wins above wave threshold.
        if n_blocks * 2 < num_sms:
            if K <= N:
                return M <= 128
            if n_blocks * 3 >= num_sms:
                return False
        # 0.5-1 wave K<N: calibrated on A100 to M=128 (e.g. N=4096,K=1536 M=17-384).
        if n_blocks >= num_sms // 2 and K < N:
            return M <= 128
        return M <= 16

    if is_sm86:
        # ~600-940 GB/s GDDR6/GDDR6X. Dedicated block: sm89 fallback tiers are too
        # loose for 600 GB/s bandwidth and cause regressions at medium N (~N=4096).
        if n_blocks >= num_sms:
            return M <= 128
        if n_blocks >= num_sms // 2:
            return M <= 64
        return M <= 16

    if is_sm90:
        # HBM3/HBM3e. dequant+F.linear (WGMMA path) is significantly faster than our
        # mma.sync kernel; thresholds are calibrated conservatively (H100/H200 share path).
        if n_blocks >= num_sms * 3:
            return M <= 64
        if n_blocks >= num_sms * 2:
            return M <= 48
        if n_blocks >= num_sms:
            return M <= 32
        if n_blocks >= num_sms // 2:
            # Square/tall-K at <3/4 wave: K-tiling too efficient on HBM3e.
            if K >= N and n_blocks * 4 < num_sms * 3:
                return False
            return M <= 16
        return False

    if is_sm120:
        # GDDR7 (~1-1.8 TB/s). Medium-N threshold tiers differ from sm89.
        # sm121 (DGX Spark) has a different bandwidth/SM profile; uses sm89
        # fallback below until validated.
        if n_blocks >= num_sms * 3:
            return M <= 256
        if n_blocks >= num_sms * 2:
            return M <= 128
        # Short-K (K<N) at ~0.8 wave: default impl competitive above M=64.
        if n_blocks * 5 >= num_sms * 4:
            return M <= (96 if K >= N else 64)
        if n_blocks >= num_sms:
            return M <= 64
        if n_blocks >= num_sms // 2:
            # Large-N (n_blocks>=128, N>=8192) with K>=N/2: calibrated on RTX Pro 6000 to M=64.
            return M <= (64 if (K * 2 >= N and n_blocks >= 128) else 8)
        if tall_k_2xn and n_blocks > 64:
            return M <= 16
        return M <= 8

    # Fallback: sm89 (4090, L40S, L4), sm121 (DGX Spark), unrecognized arches.
    # GDDR bandwidth makes dequant relatively expensive so custom wins at higher M.
    if n_blocks >= num_sms * 3:
        return M <= 256
    if n_blocks >= num_sms * 2:
        return M <= 128
    # Near-wave (~0.8x): tall-K and very large N (n_blocks>=200, N>=14336) raise cap to M=128.
    # N=10240 (n_blocks=160) deliberately excluded to avoid regressions there.
    if n_blocks * 5 >= num_sms * 4:
        if tall_k_2xn or n_blocks >= 200:
            return M <= 128
        # Square/tall-K: >=60 SMs wins to M=128; <60 SMs default wins earlier.
        if K >= N:
            return M <= (128 if num_sms >= 60 else 32)
        return M <= 64
    if n_blocks >= num_sms // 2:
        if tall_k_2xn:
            return M <= 64
        if n_blocks >= 64:
            return M <= 8
        return M <= 32
    # Tall-K (K>N) at narrow N (n_blocks<=48): M-driven crossover.
    # K>=3N (e.g. N=2560,K=10240): SIMT wins to M=12. Moderate K>N: M=10.
    if K > N and n_blocks <= 48:
        return M <= (12 if K >= N * 3 else 10)
    return M <= (16 if (tall_k_2xn or n_blocks < 48) else 8)


if torch.version.hip is None:

    @register_kernel("bitsandbytes::gemm_4bit", "cuda")
    def _(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB: Sequence[int],
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        bias: Optional[torch.Tensor] = None,
        absmax_8bit: Optional[torch.Tensor] = None,
        absmax_code: Optional[torch.Tensor] = None,
        absmax_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = A.shape[-1]
        M = A.numel() // K
        N = shapeB[0]

        # M>1536: dequant+F.linear wins (dequant savings negligible at very large batch).
        # M<=4: always custom (custom kernel wins universally at small batch).
        # M in [5, 1536]: shape/arch-dependent; cached per (device, dtype, M, N, K).
        if M > 1536:
            use_custom = False
        elif K % blocksize != 0:
            warn(
                f"inner dimension ({K}) is not aligned for fast kernel "
                f"with blocksize={blocksize}, falling back to slower implementation.",
                UserWarning,
            )
            use_custom = False
        else:
            use_custom = M <= 4 or _gemm_4bit_use_custom(A.device.index, A.dtype, M, N, K)

        if not use_custom:
            if absmax_8bit is not None:
                absmax_dq = torch.empty_like(absmax_8bit, dtype=torch.float32)
                _dequantize_blockwise_impl(absmax_8bit, absmax, absmax_code, 256, torch.float32, out=absmax_dq)
                absmax = absmax_dq + absmax_offset
            B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)
            _dequantize_4bit_impl(B, absmax, blocksize, quant_type, A.dtype, out=B_dq)
            return torch.nn.functional.linear(A, B_dq, bias)

        if K != shapeB[1]:
            raise RuntimeError(f"A inner dim ({K}) does not match weight ({shapeB[1]})")
        if absmax.dtype != torch.float32:
            raise RuntimeError(f"absmax must be float32, got {absmax.dtype}")
        if bias is not None:
            if bias.ndim != 1:
                raise RuntimeError(f"bias must be 1D, got {bias.ndim}D")
            if bias.dtype != A.dtype:
                raise RuntimeError(f"bias dtype ({bias.dtype}) must match A dtype ({A.dtype})")

        quant_type_int = 1 if quant_type == "fp4" else 2

        out = torch.empty((*A.shape[:-1], N), dtype=A.dtype, device=A.device)
        stream = _get_raw_stream(A.device.index)

        if A.dtype == torch.bfloat16:
            fn = lib.cgemm_4bit_bf16
        elif A.dtype == torch.float16:
            fn = lib.cgemm_4bit_fp16
        elif A.dtype == torch.float32:
            fn = lib.cgemm_4bit_fp32
        else:
            raise RuntimeError(f"unsupported dtype {A.dtype}")

        with _cuda_device_of(A):
            fn(
                A.data_ptr(),
                B.data_ptr(),
                absmax.data_ptr(),
                absmax_8bit.data_ptr() if absmax_8bit is not None else None,
                absmax_code.data_ptr() if absmax_code is not None else None,
                absmax_offset.data_ptr() if absmax_offset is not None else None,
                out.data_ptr(),
                bias.data_ptr() if bias is not None else None,
                M,
                N,
                K,
                blocksize,
                quant_type_int,
                stream,
            )

        return out

else:

    @register_kernel("bitsandbytes::gemm_4bit", "cuda")
    def _(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB: Sequence[int],
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        bias: Optional[torch.Tensor] = None,
        absmax_8bit: Optional[torch.Tensor] = None,
        absmax_code: Optional[torch.Tensor] = None,
        absmax_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = A.shape[-1]
        M = A.numel() // K
        N = shapeB[0]

        if M == 1:
            if K % blocksize == 0:
                if absmax_8bit is not None:
                    absmax = (
                        torch.ops.bitsandbytes.dequantize_blockwise.default(
                            absmax_8bit, absmax, absmax_code, 256, torch.float32
                        )
                        + absmax_offset
                    )

                code = _get_4bit_code(quant_type, A.device)
                out = torch.empty((*A.shape[:-1], N), dtype=A.dtype, device=A.device)
                _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)

                if bias is not None:
                    out = out + bias
                return out

            warn(
                f"inner dimension ({K}) is not aligned for fast kernel "
                f"with blocksize={blocksize}, falling back to slower implementation.",
                UserWarning,
            )

        return _gemm_4bit_default_impl(
            A,
            B,
            shapeB,
            absmax,
            blocksize,
            quant_type,
            bias,
            absmax_8bit=absmax_8bit,
            absmax_code=absmax_code,
            absmax_offset=absmax_offset,
        )


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
    "lars": (
        lib.cmomentum32bit_grad_32,
        lib.cmomentum32bit_grad_16,
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


def _optimizer_update_32bit_impl(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    state2: Optional[torch.Tensor],
    unorm_vec: Optional[torch.Tensor],
    max_unorm: float,
    param_norm: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    step: int,
    lr: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    optim_fns = str2optimizer32bit.get(optimizer_name, None)
    if optim_fns is None:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {list(str2optimizer32bit.keys())}"
        )
    if g.dtype == torch.float32:
        optim_func = optim_fns[0]
    elif g.dtype == torch.float16:
        optim_func = optim_fns[1]
    elif g.dtype == torch.bfloat16 and len(optim_fns) == 3:
        optim_func = optim_fns[2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )

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


def _optimizer_update_8bit_blockwise_impl(
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
    weight_decay: float,
    gnorm_scale: float,
    skip_zeros=False,
) -> None:
    # torch._check(
    #     g.numel() == p.numel(),
    #     lambda: f"g and p must have the same number of elements, got {g.numel()} and {p.numel()}",
    # )
    # compute_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # torch._check(
    #     g.dtype in compute_dtypes,
    #     lambda: f"g must be bfloat16, float16, or float32, got {g.dtype}",
    # )
    # torch._check(
    #     g.dtype == p.dtype,
    #     lambda: f"Expected all tensors to have the same dtype, got g.dtype={g.dtype}, p.dtype={p.dtype}",
    # )
    # torch._check(
    #     state1.dtype == torch.uint8,
    #     lambda: f"state1 must be uint8, got {state1.dtype}",
    # )
    # torch._check(
    #     qmap1.dtype == absmax1.dtype == torch.float32,
    #     lambda: f"Expected qmap1 and absmax1 to be float32, got qmap1.dtype={qmap1.dtype}, absmax1.dtype={absmax1.dtype}",
    # )
    # if state2 is not None:
    #     torch._check(
    #         state2.dtype == torch.uint8,
    #         lambda: f"state2 must be uint8, got {state2.dtype}",
    #     )
    #     torch._check(
    #         qmap2.dtype == absmax2.dtype == torch.float32,
    #         lambda: f"Expected qmap2 and absmax2 to be float32, got qmap2.dtype={qmap2.dtype}, absmax2.dtype={absmax2.dtype}",
    #     )
    optimizer_fns = str2optimizer8bit_blockwise.get(optimizer_name)
    if optimizer_fns is None:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. Supported optimizers: {list(str2optimizer8bit_blockwise.keys())}"
        )

    if g.dtype == torch.float32:
        optimizer_fn = optimizer_fns[0]
    elif g.dtype == torch.float16:
        optimizer_fn = optimizer_fns[1]
    elif g.dtype == torch.bfloat16:
        optimizer_fn = optimizer_fns[2]
    else:
        raise ValueError(
            f"Unsupported gradient dtype: {g.dtype}. Supported dtypes: torch.float32, torch.float16, torch.bfloat16"
        )

    with _cuda_device_of(g):
        optimizer_fn(
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


register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "cuda")(_optimizer_update_8bit_blockwise_impl)
register_kernel("bitsandbytes::optimizer_update_32bit", "cuda")(_optimizer_update_32bit_impl)
