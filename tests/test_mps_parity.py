"""CPU-as-oracle parity tests for the MPS backend (Phase 1 of the Apple Silicon Metal port).

For every op the ``mps`` backend currently supports, these tests run the same seeded
inputs through the CPU path (on a source checkout without a native build, the ``cpu``
device resolves to the ``default`` pure-PyTorch backend -- the oracle) and through the
``mps`` path, then assert agreement within documented per-dtype tolerances.

Tolerances (empirically calibrated on torch 2.12.1 / macOS 26.4.1, see
``docs/apple_silicon/MPS_STATUS.md`` for the measured baseline):

- Quantization artifacts (uint8 codes, packed nibbles) must be **bit-exact**: both
  paths share the same fp32 quantization math, and any mismatch means a wrong bucket,
  not a rounding difference.
- ``absmax`` and other fp32 statistics: tight fp32 tolerance.
- Matmul outputs (gemv_4bit / gemm_4bit / int8 matmuls): fp32 tight; fp16/bf16 looser,
  following the per-dtype convention used for CUDA (fp32 1e-5, fp16 1e-2, bf16 4e-2
  absolute), so the same bounds keep working once native Metal kernels replace the
  pure-torch fallbacks in Phase 2+.

The whole module skips when MPS is not available.
"""

import os

import pytest
import torch

import bitsandbytes
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")

# Whether the hand-written Metal quantize_blockwise kernel is built and loaded on this
# machine. Set BNB_MPS_REQUIRE_NATIVE=1 to make the build-verification tests fail loudly
# (rather than skip) when the native library did not load -- used to gate a source build.
try:
    import bitsandbytes.backends.mps.ops as _mps_ops

    _NATIVE_AVAILABLE = _mps_ops._native_available()
except Exception:
    _mps_ops = None
    _NATIVE_AVAILABLE = False

_REQUIRE_NATIVE = os.environ.get("BNB_MPS_REQUIRE_NATIVE") == "1"

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BLOCKSIZES = [64, 128, 256, 512]

# Per-dtype (rtol, atol) for comparing MPS results against the CPU oracle.
# fp32 divergence comes only from accumulation order (measured <= ~8e-6 at K<=256);
# fp16/bf16 get the looser bounds to absorb half-precision rounding.
PARITY_TOLERANCE = {
    torch.float32: (1e-6, 1e-5),
    torch.float16: (1e-3, 1e-2),
    torch.bfloat16: (1e-2, 4e-2),
}


def assert_parity(res_mps: torch.Tensor, res_cpu: torch.Tensor, dtype: torch.dtype):
    """Assert an MPS result matches the CPU oracle within the documented tolerance."""
    assert res_mps.device.type == "mps"
    rtol, atol = PARITY_TOLERANCE[dtype]
    torch.testing.assert_close(res_mps.cpu(), res_cpu, rtol=rtol, atol=atol)


def assert_bit_exact(res_mps: torch.Tensor, res_cpu: torch.Tensor):
    """Quantized codes must match bucket-for-bucket, not just approximately."""
    assert res_mps.device.type == "mps"
    if res_cpu.dtype != torch.uint8:
        res_cpu = res_cpu.view(torch.uint8)
        res_mps = res_mps.view(torch.uint8)
    mismatched = (res_mps.cpu() != res_cpu).sum().item()
    assert mismatched == 0, f"{mismatched}/{res_cpu.numel()} quantized values differ from CPU oracle"


class TestBlockwise8bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_quantize_blockwise(self, dtype, blocksize):
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A.to("mps"), code.to("mps"), blocksize)

        assert q_mps.shape == q_cpu.shape
        assert q_mps.dtype == torch.uint8
        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_blockwise(self, dtype, blocksize):
        # As of Phase 3 there IS an mps registration for dequantize_blockwise: native Metal
        # when built, else the same pure-torch compute the default backend uses.
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        # Quantize once on CPU so both dequant paths see identical inputs.
        q, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        dq_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q, absmax, code, blocksize, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_blockwise(
            q.to("mps"), absmax.to("mps"), code.to("mps"), blocksize, dtype
        )

        assert dq_mps.shape == A.shape
        assert dq_mps.dtype == dtype
        assert_parity(dq_mps, dq_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_roundtrip_reconstruction(self, dtype, blocksize):
        """quantize->dequantize entirely on MPS reconstructs as well as the CPU oracle."""
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        dq_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q_cpu, absmax_cpu, code, blocksize, dtype)

        A_mps = A.to("mps")
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A_mps, code.to("mps"), blocksize)
        dq_mps = torch.ops.bitsandbytes.dequantize_blockwise(q_mps, absmax_mps, code.to("mps"), blocksize, dtype)

        err_cpu = (dq_cpu.float() - A.float()).abs().mean().item()
        err_mps = (dq_mps.cpu().float() - A.float()).abs().mean().item()

        # Dynamic 8-bit reconstruction of randn is ~1e-2 mean abs error; "confident
        # garbage" would be ~1.0. The MPS error must also track the oracle closely.
        assert err_mps < 0.05, f"MPS roundtrip error {err_mps} implausibly high"
        assert err_mps == pytest.approx(err_cpu, rel=0.02)


def _require_native():
    """Skip (or hard-fail under BNB_MPS_REQUIRE_NATIVE=1) when the native lib is absent."""
    if not _NATIVE_AVAILABLE:
        if _REQUIRE_NATIVE:
            pytest.fail(
                "BNB_MPS_REQUIRE_NATIVE=1 but the native MPS library did not load. "
                "Build it: cmake -DCOMPUTE_BACKEND=mps -S . -B . && cmake --build . --config Release"
            )
        pytest.skip("Native MPS library not built (using Hub/pure-torch fallback).")


class TestNativeMetalPath:
    """Native-Metal verification: quant/dequant ops through the hand-written kernels.

    These assert the native path is exercised (not a fallback) and stays bit-exact vs the
    CPU oracle. When the native library is not built, they skip -- unless
    BNB_MPS_REQUIRE_NATIVE=1, which turns the missing library into a hard failure so a
    source-build verification run cannot silently pass on the fallback.
    """

    def test_native_library_loaded(self):
        _require_native()
        from bitsandbytes.cextension import get_mps_library

        assert get_mps_library() is not None

    def test_buffer_contract_guard(self):
        """The data_ptr()-is-the-MTLBuffer guard: passes for a real MPS tensor, rejects
        a bogus pointer and an oversized length. A future torch that breaks the contract
        must be caught here rather than corrupting a Metal dispatch."""
        _require_native()
        import ctypes as ct

        from bitsandbytes.cextension import get_mps_library

        lib = get_mps_library()
        # Re-running the load-time verification must not raise on this torch.
        lib.verify_buffer_contract()

        t = torch.empty(64, dtype=torch.float32, device="mps")
        torch.mps.synchronize()
        check = lib._lib.bnb_mps_check_buffer_contract
        assert check(ct.c_void_p(t.data_ptr()), ct.c_int64(t.numel() * 4)) == 1
        assert check(ct.c_void_p(0), ct.c_int64(0)) == 0  # null pointer rejected
        assert check(ct.c_void_p(t.data_ptr()), ct.c_int64(10**9)) == 0  # oversize rejected

    def test_view_data_ptr_is_base_plus_offset(self):
        """Pins the pointer semantics that force the offset-0 clone in _ensure_native_buffer
        (verified in Phase M4): for an MPS view with storage_offset != 0, data_ptr() is
        base_ptr + storage_offset * itemsize -- raw pointer arithmetic, NOT an id<MTLBuffer>
        (casting it would be a miscast; even objc-probing such an interior pointer
        SIGSEGVs). The base buffer object IS recoverable via untyped_storage().data_ptr(),
        which is the documented recipe should offset binding ever be implemented. If this
        test fails, torch changed the contract -- re-verify _ensure_native_buffer before
        trusting the native ops."""
        _require_native()
        import ctypes as ct

        from bitsandbytes.cextension import get_mps_library

        lib = get_mps_library()
        check = lib._lib.bnb_mps_check_buffer_contract

        base = torch.arange(1024, dtype=torch.float32, device="mps")
        view = base[128:]
        torch.mps.synchronize()

        # A view's data_ptr() is base + offset bytes (raw arithmetic, not an objc object)...
        assert view.storage_offset() == 128
        assert view.data_ptr() - base.data_ptr() == 128 * base.element_size()
        # ...while the storage's data_ptr() is the base allocation, which IS the MTLBuffer.
        storage = view.untyped_storage()
        assert storage.data_ptr() == base.data_ptr()
        assert check(ct.c_void_p(storage.data_ptr()), ct.c_int64(storage.nbytes())) == 1
        # Deliberately NOT calling check() on view.data_ptr(): an interior pointer is not
        # an objc object, and probing it segfaults (uncatchable by @try/@catch) -- which is
        # exactly why _ensure_native_buffer must keep cloning offset != 0 views.

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_quantize_blockwise_native_bit_exact(self, dtype, blocksize):
        if not _NATIVE_AVAILABLE:
            if _REQUIRE_NATIVE:
                pytest.fail("BNB_MPS_REQUIRE_NATIVE=1 but native path unavailable.")
            pytest.skip("Native MPS library not built.")

        torch.manual_seed(1337)
        A = torch.randn(1024, 1024, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        # This dispatches through the native Metal kernel (routing is automatic on mps).
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A.to("mps"), code.to("mps"), blocksize)

        # The kernel mirrors the reference math exactly (fp32 reductions, correctly-rounded
        # division via -fno-fast-math), so codes AND absmax must be bit-exact.
        assert_bit_exact(q_mps, q_cpu)
        assert torch.equal(absmax_mps.cpu(), absmax_cpu)

    @pytest.mark.parametrize("blocksize", [64, 256])
    def test_native_partial_block_bit_exact(self, blocksize):
        """Tail block (numel not divisible by blocksize) is also bit-exact."""
        if not _NATIVE_AVAILABLE:
            if _REQUIRE_NATIVE:
                pytest.fail("BNB_MPS_REQUIRE_NATIVE=1 but native path unavailable.")
            pytest.skip("Native MPS library not built.")

        torch.manual_seed(1337)
        A = torch.randn(7, blocksize - 1, dtype=torch.float32)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A.to("mps"), code.to("mps"), blocksize)

        assert_bit_exact(q_mps, q_cpu)
        assert torch.equal(absmax_mps.cpu(), absmax_cpu)

    # ---- Phase 3: dequantize_blockwise (newly registered on mps), + the 4-bit ops ----

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_blockwise_native(self, dtype, blocksize):
        _require_native()
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)
        q, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        dq_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q, absmax, code, blocksize, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_blockwise(
            q.to("mps"), absmax.to("mps"), code.to("mps"), blocksize, dtype
        )
        # fp32 kernel + a torch .to(dtype) cast reproduces the reference exactly.
        assert dq_mps.dtype == dtype
        assert torch.equal(dq_mps.cpu(), dq_cpu)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.bfloat16], ids=id_formatter("storage"))
    def test_quantize_4bit_native(self, dtype, quant_type, blocksize, storage_dtype):
        _require_native()
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, storage_dtype)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, storage_dtype)

        assert q_mps.dtype == storage_dtype
        assert q_mps.shape == q_cpu.shape
        # Packed nibbles bit-exact (view-as-uint8 avoids NaN!=NaN on the bf16 reinterpret).
        assert_bit_exact(q_mps, q_cpu)
        assert torch.equal(absmax_mps.cpu(), absmax_cpu)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_4bit_native(self, dtype, quant_type, blocksize):
        _require_native()
        torch.manual_seed(1337)
        shape = (256, 256)
        A = torch.randn(shape, dtype=dtype)
        q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, blocksize, quant_type, shape, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(
            q.to("mps"), absmax.to("mps"), blocksize, quant_type, shape, dtype
        )
        assert dq_mps.shape == shape
        assert dq_mps.dtype == dtype
        assert torch.equal(dq_mps.cpu(), dq_cpu)

    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_4bit_native_partial_block_bit_exact(self, quant_type, blocksize):
        """Odd-numel tail (the padding nibble) is bit-exact for quantize AND dequantize."""
        _require_native()
        torch.manual_seed(1337)
        shape = (7, blocksize - 1)  # numel not divisible by blocksize; odd for odd blocksize-1
        A = torch.randn(shape, dtype=torch.float32)

        q_cpu, am_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        q_mps, am_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, torch.uint8)
        assert_bit_exact(q_mps, q_cpu)
        assert torch.equal(am_mps.cpu(), am_cpu)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q_cpu, am_cpu, blocksize, quant_type, shape, torch.float32)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(q_mps, am_mps, blocksize, quant_type, shape, torch.float32)
        assert torch.equal(dq_mps.cpu(), dq_cpu)

    # ---- Phase M2: fused gemv_4bit (dequant + dot product in one Metal kernel) ----

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 256])
    def test_gemv_4bit_native_fused(self, dtype, quant_type, blocksize, monkeypatch):
        """gemv_4bit routes through the fused native Metal kernel (asserted via a spy, not
        assumed) and matches the CPU oracle within the documented per-dtype tolerances."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        calls = []
        orig = _mps_ops._gemv_4bit_native

        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(_mps_ops, "_gemv_4bit_native", spy)

        torch.manual_seed(1337)
        # K=256 matches the size the fp32 tolerance was calibrated at (accumulation-order
        # deviation vs the CPU oracle grows with K; at K=512 a single fp32 element lands at
        # ~1.2e-5, just past the 1e-5 atol -- order noise, not a dequant bug).
        out_features, in_features = 1024, 256
        A = torch.randn(1, 1, in_features, dtype=dtype)
        B = torch.randn(out_features, in_features, dtype=dtype)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, torch.uint8)
        code = F.get_4bit_type(quant_type, device="cpu", blocksize=blocksize)

        out_cpu = torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)
        out_mps = torch.ops.bitsandbytes.gemv_4bit(
            A.to("mps"), B_q.to("mps"), B.shape, absmax.to("mps"), code.to("mps"), blocksize
        )

        assert calls, "gemv_4bit did not route through the fused native Metal kernel"
        assert out_mps.shape == (1, 1, out_features)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    def test_gemv_4bit_unaligned_k_uses_fallback(self, monkeypatch):
        """K % 32 != 0 cannot take the fused kernel (uint4 row loads); it must fall back to
        dequant + F.linear and still be correct."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        def fail_if_called(*args, **kwargs):
            pytest.fail("fused native gemv_4bit must not be used when K % 32 != 0")

        monkeypatch.setattr(_mps_ops, "_gemv_4bit_native", fail_if_called)

        torch.manual_seed(1337)
        out_features, in_features = 128, 80  # K % 32 == 16
        A = torch.randn(1, 1, in_features, dtype=torch.float32)
        B = torch.randn(out_features, in_features, dtype=torch.float32)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, 64, "nf4", torch.uint8)
        code = F.get_4bit_type("nf4", device="cpu", blocksize=64)

        out_cpu = torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, 64)
        out_mps = torch.ops.bitsandbytes.gemv_4bit(
            A.to("mps"), B_q.to("mps"), B.shape, absmax.to("mps"), code.to("mps"), 64
        )
        assert_parity(out_mps, out_cpu, torch.float32)

    # ---- Phase M3: native gemm_4bit (dequant scratch + MPSMatrixMultiplication + bias) ----

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("has_bias", [False, True], ids=id_formatter("has_bias"))
    @pytest.mark.parametrize("compress_statistics", [False, True], ids=id_formatter("compress_statistics"))
    def test_gemm_4bit_native(self, dtype, quant_type, has_bias, compress_statistics, monkeypatch):
        """gemm_4bit routes through the native one-command-buffer Metal path (asserted via a
        spy) and matches the CPU oracle within the documented per-dtype tolerances. All three
        dtypes are native: fp32/fp16 via MPSMatrixMultiplication, bf16 via MPSGraph (which has
        a bf16 matmul where MPSMatrixMultiplication hard-asserts on anything but
        fp32/fp16/int8/int16)."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        calls = []
        orig = _mps_ops._gemm_4bit_native

        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(_mps_ops, "_gemm_4bit_native", spy)

        torch.manual_seed(1337)
        # K=256 matches the size the fp32 tolerance was calibrated at (MPSMatrixMultiplication's
        # accumulation order differs from F.linear's; deviation grows with K).
        N, K, blocksize = 128, 256, 64
        A = torch.randn(2, 2, K, dtype=dtype)
        B = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=dtype) if has_bias else None

        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B, blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(
            B.to("mps"), blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )

        if compress_statistics:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A,
                B_q,
                list(B.shape),
                qs.state2.absmax,
                blocksize,
                quant_type,
                bias=bias,
                absmax_8bit=qs.absmax,
                absmax_code=qs.state2.code,
                absmax_offset=qs.offset,
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.state2.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
                absmax_8bit=qs_mps.absmax,
                absmax_code=qs_mps.state2.code,
                absmax_offset=qs_mps.offset,
            )
        else:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A, B_q, list(B.shape), qs.absmax, blocksize, quant_type, bias=bias
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
            )

        assert calls, "gemm_4bit did not route through the native Metal path"
        assert out_mps.shape == (2, 2, N)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    def test_gemm_4bit_bf16_is_native_and_matches_the_fallback(self, monkeypatch):
        """bf16 takes the native MPSGraph gemm and reproduces the dequant + F.linear fallback
        it replaced.

        That fallback is the sharper oracle here, not the CPU one: it is the exact composition
        bf16 ran before this path existed, on the same device, so any drift is a real change in
        what a QLoRA forward computes. Without bias the two agree BIT-EXACTLY, which is the
        assertion with teeth.

        With bias they can differ. The epilogue kernel computes (bf16)(gemm + bias) on an
        already-rounded gemm result where F.linear rounds once, so the error is one ulp of the
        *pre-add* magnitude -- which, wherever bias largely cancels the gemm result, is several
        ulp of the much smaller output. Relative-to-output tolerances are the wrong shape for
        that, so the biased case is held to the documented CPU parity tolerance instead.
        """
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")
        if not hasattr(_mps_ops._mps_native._lib, "bnb_mps_gemm_4bit_supports_bf16"):
            pytest.skip("dylib predates bf16 gemm support")

        calls = []
        orig = _mps_ops._gemm_4bit_native

        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(_mps_ops, "_gemm_4bit_native", spy)

        torch.manual_seed(1337)
        N, K, blocksize = 128, 256, 64
        A = torch.randn(2, 2, K, dtype=torch.bfloat16)
        B = torch.randn(N, K, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.bfloat16)
        B_q, qs = bitsandbytes.functional.quantize_4bit(B, blocksize=blocksize, quant_type="nf4")
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(B.to("mps"), blocksize=blocksize, quant_type="nf4")

        B_dq = torch.ops.bitsandbytes.dequantize_4bit(
            B_q_mps.view(-1, 1), qs_mps.absmax, blocksize, "nf4", [N, K], torch.bfloat16
        )

        for use_bias in (False, True):
            b_cpu = bias if use_bias else None
            b_mps = bias.to("mps") if use_bias else None
            n_before = len(calls)

            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A, B_q, list(B.shape), qs.absmax, blocksize, "nf4", bias=b_cpu
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"), B_q_mps, list(B.shape), qs_mps.absmax, blocksize, "nf4", bias=b_mps
            )
            assert len(calls) > n_before, f"bf16 gemm_4bit (bias={use_bias}) did not route native"
            assert out_mps.dtype == torch.bfloat16
            assert_parity(out_mps, out_cpu, torch.bfloat16)

            out_fallback = torch.nn.functional.linear(A.to("mps"), B_dq, b_mps)
            if not use_bias:
                assert torch.equal(out_mps, out_fallback), (
                    "bf16 native gemm must reproduce the dequant + F.linear fallback bit-exactly "
                    f"without bias; max deviation {(out_mps.float() - out_fallback.float()).abs().max().item()}"
                )

    # ---- Phase M6: fused gemm_4bit_backward (grad_A) ----

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("compress_statistics", [False, True], ids=id_formatter("compress_statistics"))
    def test_gemm_4bit_backward_native(self, dtype, quant_type, compress_statistics, monkeypatch):
        """grad_A = grad_output @ B_dq routes through the fused native path and matches both
        oracles.

        Two oracles, and the on-device one is the strict half: with no bias in the backward
        there is no epilogue to double-round, so the fused kernel must reproduce the dequant +
        torch.matmul composition it replaced BIT-EXACTLY. Any drift at all is a real change to
        the gradients a QLoRA run computes.
        """
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")
        if not hasattr(_mps_ops._mps_native._lib, "bnb_mps_gemm_4bit_bwd"):
            pytest.skip("dylib predates the fused backward")

        calls = []
        orig = _mps_ops._gemm_4bit_backward_native

        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(_mps_ops, "_gemm_4bit_backward_native", spy)

        torch.manual_seed(1337)
        N, K, blocksize = 128, 256, 64
        grad_output = torch.randn(2, 2, N, dtype=dtype)
        B = torch.randn(N, K, dtype=dtype)

        # Quantize ONCE on cpu and move the state across, rather than quantizing independently on
        # each device. The subject of this test is the matmul, and independent quantization drags
        # in a confound that has nothing to do with it: with compress_statistics the absmax is
        # itself run through `quantize_blockwise`, whose CPU kernel snaps to a 65536-point LUT and
        # so disagrees with the exact MPS kernel on the occasional code (see the known
        # quantize_blockwise failures). One differing code in 512 shifts that block's scale by
        # ~9e-4, which lands in B_dq and blows the fp32 1e-5 tolerance -- a real discrepancy, but
        # an upstream one. Sharing the state makes both sides bit-identical by construction.
        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B, blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )
        B_q_mps = B_q.to("mps")

        if compress_statistics:
            extra_cpu = dict(absmax_8bit=qs.absmax, absmax_code=qs.state2.code, absmax_offset=qs.offset)
            am_cpu = qs.state2.absmax
        else:
            extra_cpu = {}
            am_cpu = qs.absmax
        am_mps = am_cpu.to("mps")
        extra_mps = {k: v.to("mps") for k, v in extra_cpu.items()}

        out_cpu = torch.ops.bitsandbytes.gemm_4bit_backward(
            grad_output, B_q, list(B.shape), am_cpu, blocksize, quant_type, **extra_cpu
        )
        out_mps = torch.ops.bitsandbytes.gemm_4bit_backward(
            grad_output.to("mps"), B_q_mps, list(B.shape), am_mps, blocksize, quant_type, **extra_mps
        )

        assert calls, "gemm_4bit_backward did not route through the native Metal path"
        assert out_mps.shape == (2, 2, K)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

        # ...and bit-exactly against the composition MatMul4Bit.backward used to run inline.
        absmax_mps = am_mps
        if compress_statistics:
            absmax_mps = (
                torch.ops.bitsandbytes.dequantize_blockwise.default(
                    extra_mps["absmax_8bit"], am_mps, extra_mps["absmax_code"], 256, torch.float32
                )
                + extra_mps["absmax_offset"]
            )
        B_dq = torch.ops.bitsandbytes.dequantize_4bit(
            B_q_mps.view(-1, 1), absmax_mps, blocksize, quant_type, [N, K], dtype
        )
        out_fallback = torch.matmul(grad_output.to("mps"), B_dq)
        assert torch.equal(out_mps, out_fallback), (
            "fused backward must reproduce dequant + torch.matmul bit-exactly (no bias epilogue "
            f"exists here to explain a difference); max deviation "
            f"{(out_mps.float() - out_fallback.float()).abs().max().item()}"
        )

    def test_gemm_4bit_backward_unaligned_k_uses_fallback(self, monkeypatch):
        """K % 32 != 0 cannot take the fused backward (uint4 loads in the chunked dequant)."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        def fail_if_called(*args, **kwargs):
            pytest.fail("native gemm_4bit_backward must not be used when K % 32 != 0")

        monkeypatch.setattr(_mps_ops, "_gemm_4bit_backward_native", fail_if_called)

        torch.manual_seed(1337)
        N, K, blocksize = 128, 80, 64  # K % 32 == 16
        grad_output = torch.randn(2, 2, N, dtype=torch.float32)
        B = torch.randn(N, K, dtype=torch.float32)
        B_q, qs = bitsandbytes.functional.quantize_4bit(B, blocksize=blocksize, quant_type="nf4")
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(B.to("mps"), blocksize=blocksize, quant_type="nf4")

        out_cpu = torch.ops.bitsandbytes.gemm_4bit_backward(grad_output, B_q, list(B.shape), qs.absmax, blocksize, "nf4")
        out_mps = torch.ops.bitsandbytes.gemm_4bit_backward(
            grad_output.to("mps"), B_q_mps, list(B.shape), qs_mps.absmax, blocksize, "nf4"
        )
        assert_parity(out_mps, out_cpu, torch.float32)

    def test_linear4bit_bf16_autograd_is_unchanged_by_the_native_paths(self, monkeypatch):
        """End to end: toggling the native bf16 paths must not change what a Linear4bit computes.

        Without bias the whole fwd+bwd is bit-identical between the two arms. With bias only the
        FORWARD differs, by the one-ulp bias-epilogue double rounding documented in §11.5, and
        the gradient merely inherits it -- pinning that the fused backward adds no deviation of
        its own.
        """
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")
        if not hasattr(_mps_ops._mps_native._lib, "bnb_mps_gemm_4bit_bwd"):
            pytest.skip("dylib predates the fused backward")

        def run(disabled, use_bias):
            monkeypatch.setenv("BNB_MPS_DISABLE_BF16_GEMM", disabled)
            torch.manual_seed(0)
            lin = bitsandbytes.nn.Linear4bit(
                256, 256, bias=use_bias, compute_dtype=torch.bfloat16, quant_type="nf4"
            ).to("mps")
            x = torch.randn(8, 256, device="mps", dtype=torch.bfloat16, requires_grad=True)
            y = lin(x)
            (y * y).sum().backward()
            return y.clone(), x.grad.clone()

        y_nat, g_nat = run("0", False)
        y_fb, g_fb = run("1", False)
        assert torch.equal(y_nat, y_fb), "bias-free forward must be bit-identical across the toggle"
        assert torch.equal(g_nat, g_fb), "bias-free gradient must be bit-identical across the toggle"

        y_nat, g_nat = run("0", True)
        y_fb, g_fb = run("1", True)
        one_ulp = g_fb.float().abs().max().item() * 2**-7
        assert (g_nat.float() - g_fb.float()).abs().max().item() <= one_ulp
        assert torch.isfinite(g_nat).all()

    def test_gemm_4bit_unaligned_k_uses_fallback(self, monkeypatch):
        """K % 32 != 0 cannot take the native gemm (uint4 loads in the chunked dequant); it
        must fall back to dequant + F.linear and still be correct."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        def fail_if_called(*args, **kwargs):
            pytest.fail("native gemm_4bit must not be used when K % 32 != 0")

        monkeypatch.setattr(_mps_ops, "_gemm_4bit_native", fail_if_called)

        torch.manual_seed(1337)
        N, K, blocksize = 128, 80, 64  # K % 32 == 16
        A = torch.randn(2, 2, K, dtype=torch.float32)
        B = torch.randn(N, K, dtype=torch.float32)
        B_q, qs = bitsandbytes.functional.quantize_4bit(B, blocksize=blocksize, quant_type="nf4")
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(B.to("mps"), blocksize=blocksize, quant_type="nf4")

        out_cpu = torch.ops.bitsandbytes.gemm_4bit(A, B_q, list(B.shape), qs.absmax, blocksize, "nf4")
        out_mps = torch.ops.bitsandbytes.gemm_4bit(
            A.to("mps"), B_q_mps, list(B.shape), qs_mps.absmax, blocksize, "nf4"
        )
        assert_parity(out_mps, out_cpu, torch.float32)

    def test_sync_discipline_interleave_stress(self):
        """Race stress for the cross-queue sync discipline (Phase M4). The native matmuls
        run on a private MTLCommandQueue, so correctness depends on two syncs:
        (a) torch.mps.synchronize() BEFORE dispatch -- torch's pending writes into A (the
            in-place copy_ below is enqueued on torch's stream) must be materialized
            before the native kernel reads A from the other queue;
        (b) waitUntilCompleted AFTER commit -- out must be complete before torch reads it.
        Every iteration enqueues a heavy chained matmul and makes the in-place write into
        A *depend* on it, so torch's write to A lands late on torch's queue; the native
        kernel on the other queue then reads A. With the pre-sync removed this fails
        30/30 iterations (verified empirically in Phase M4 by no-op'ing
        torch.mps.synchronize); with the discipline intact it must pass every time."""
        _require_native()
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        torch.manual_seed(1337)
        dtype = torch.float16
        N, K, blocksize = 1024, 256, 64
        B = torch.randn(N, K, dtype=dtype)
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, "nf4", torch.uint8)
        code = F.get_4bit_type("nf4", device="cpu", blocksize=blocksize)
        B_q_mps, absmax_mps, code_mps = B_q.to("mps"), absmax.to("mps"), code.to("mps")

        # Long-lived, reused buffers -- every iteration writes into these in place.
        Av_mps = torch.zeros(1, 1, K, dtype=dtype, device="mps")
        Am_mps = torch.zeros(8, K, dtype=dtype, device="mps")
        heavy = torch.randn(2048, 2048, dtype=dtype, device="mps")

        for _ in range(30):
            # torch-op phase: heavy chained GPU work, then in-place writes (dependent on
            # that work, so they land late) into the exact buffers the native kernels are
            # about to read. No explicit sync here -- the op implementations own the
            # sync discipline.
            h = heavy
            for _ in range(4):
                h = h @ heavy * 1e-4
            Av_cpu = torch.randn(1, 1, K, dtype=dtype)
            Am_cpu = torch.randn(8, K, dtype=dtype)
            Av_mps.copy_(Av_cpu.to("mps") + 0 * h[0, :K])
            Am_mps.copy_(Am_cpu.to("mps") + 0 * h[:8, :K])

            # native-op phase: fused gemv + native gemm read A from the private queue.
            out_v_mps = torch.ops.bitsandbytes.gemv_4bit(Av_mps, B_q_mps, B.shape, absmax_mps, code_mps, blocksize)
            out_m_mps = torch.ops.bitsandbytes.gemm_4bit(Am_mps, B_q_mps, list(B.shape), absmax_mps, blocksize, "nf4")

            # torch-read phase: consume the native outputs on torch's queue immediately.
            out_v_cpu = torch.ops.bitsandbytes.gemv_4bit(Av_cpu, B_q, B.shape, absmax, code, blocksize)
            out_m_cpu = torch.ops.bitsandbytes.gemm_4bit(Am_cpu, B_q, list(B.shape), absmax, blocksize, "nf4")
            assert_parity(out_v_mps * 2.0, out_v_cpu * 2.0, dtype)
            assert_parity(out_m_mps * 2.0, out_m_cpu * 2.0, dtype)

    @pytest.mark.parametrize(
        "op",
        ["quantize_blockwise", "dequantize_blockwise", "quantize_4bit", "dequantize_4bit", "gemv_4bit", "gemm_4bit"],
    )
    def test_graceful_fallback_when_native_absent(self, monkeypatch, op):
        """With the native handle forced off, every graduated op still works (pure-torch)."""
        if _mps_ops is None:
            pytest.skip("mps backend ops not importable.")

        monkeypatch.setattr(_mps_ops, "_mps_native", None, raising=False)
        assert _mps_ops._native_available() is False

        torch.manual_seed(1337)
        code = F.create_dynamic_map().to(torch.float32)
        A = torch.randn(256, 256, dtype=torch.float32)

        if op == "quantize_blockwise":
            q_cpu, am_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, 128)
            q_mps, am_mps = torch.ops.bitsandbytes.quantize_blockwise(A.to("mps"), code.to("mps"), 128)
            assert_bit_exact(q_mps, q_cpu)
            assert_parity(am_mps, am_cpu, torch.float32)
        elif op == "dequantize_blockwise":
            q, am = torch.ops.bitsandbytes.quantize_blockwise(A, code, 128)
            d_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q, am, code, 128, torch.float32)
            d_mps = torch.ops.bitsandbytes.dequantize_blockwise(
                q.to("mps"), am.to("mps"), code.to("mps"), 128, torch.float32
            )
            assert_parity(d_mps, d_cpu, torch.float32)
        elif op == "quantize_4bit":
            q_cpu, am_cpu = torch.ops.bitsandbytes.quantize_4bit(A, 64, "nf4", torch.uint8)
            q_mps, am_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), 64, "nf4", torch.uint8)
            assert_bit_exact(q_mps, q_cpu)
            assert_parity(am_mps, am_cpu, torch.float32)
        elif op == "dequantize_4bit":
            q, am = torch.ops.bitsandbytes.quantize_4bit(A, 64, "nf4", torch.uint8)
            d_cpu = torch.ops.bitsandbytes.dequantize_4bit(q, am, 64, "nf4", (256, 256), torch.float32)
            d_mps = torch.ops.bitsandbytes.dequantize_4bit(
                q.to("mps"), am.to("mps"), 64, "nf4", (256, 256), torch.float32
            )
            assert_parity(d_mps, d_cpu, torch.float32)
        elif op == "gemv_4bit":
            Av = torch.randn(1, 1, 256, dtype=torch.float32)
            q, am = torch.ops.bitsandbytes.quantize_4bit(A, 64, "nf4", torch.uint8)
            code4 = F.get_4bit_type("nf4", device="cpu", blocksize=64)
            o_cpu = torch.ops.bitsandbytes.gemv_4bit(Av, q, (256, 256), am, code4, 64)
            o_mps = torch.ops.bitsandbytes.gemv_4bit(
                Av.to("mps"), q.to("mps"), (256, 256), am.to("mps"), code4.to("mps"), 64
            )
            assert_parity(o_mps, o_cpu, torch.float32)
        else:  # gemm_4bit
            # K=64 keeps fp32 accumulation-order noise (F.linear MPS vs CPU) inside the
            # documented atol; at K=256 a single element of this M=4 case lands at ~1.4e-5.
            Am = torch.randn(2, 2, 64, dtype=torch.float32)
            Bm = torch.randn(64, 64, dtype=torch.float32)
            q, am = torch.ops.bitsandbytes.quantize_4bit(Bm, 64, "nf4", torch.uint8)
            o_cpu = torch.ops.bitsandbytes.gemm_4bit(Am, q, [64, 64], am, 64, "nf4")
            o_mps = torch.ops.bitsandbytes.gemm_4bit(Am.to("mps"), q.to("mps"), [64, 64], am.to("mps"), 64, "nf4")
            assert_parity(o_mps, o_cpu, torch.float32)


class Test4bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_quantize_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, torch.uint8)

        assert q_mps.shape == q_cpu.shape
        assert q_mps.dtype == torch.uint8
        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        shape = (256, 256)
        A = torch.randn(shape, dtype=dtype)

        # Quantize once on CPU so both dequant paths see identical inputs.
        q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, blocksize, quant_type, shape, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(
            q.to("mps"), absmax.to("mps"), blocksize, quant_type, shape, dtype
        )

        assert dq_mps.shape == shape
        assert dq_mps.dtype == dtype
        assert_parity(dq_mps, dq_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_roundtrip_reconstruction(self, dtype, quant_type, blocksize):
        """NF4/FP4 quantize->dequantize entirely on MPS reconstructs like the CPU oracle."""
        torch.manual_seed(1337)
        shape = (256, 256)
        A = torch.randn(shape, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q_cpu, absmax_cpu, blocksize, quant_type, shape, dtype)

        A_mps = A.to("mps")
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A_mps, blocksize, quant_type, torch.uint8)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(q_mps, absmax_mps, blocksize, quant_type, shape, dtype)

        err_cpu = (dq_cpu.float() - A.float()).abs().mean().item()
        err_mps = (dq_mps.cpu().float() - A.float()).abs().mean().item()

        # 4-bit reconstruction of randn is ~6e-2 mean abs error; garbage would be ~1.0.
        assert err_mps < 0.15, f"MPS roundtrip error {err_mps} implausibly high"
        assert err_mps == pytest.approx(err_cpu, rel=0.02)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_roundtrip_partial_block(self, dtype, quant_type, blocksize):
        """Roundtrip parity when numel is not divisible by blocksize (tail block path)."""
        torch.manual_seed(1337)
        shape = (7, blocksize - 1)
        A = torch.randn(shape, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, torch.uint8)

        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q_cpu, absmax_cpu, blocksize, quant_type, shape, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(q_mps, absmax_mps, blocksize, quant_type, shape, dtype)

        assert dq_mps.shape == shape
        assert torch.isfinite(dq_mps.cpu()).all()
        assert_parity(dq_mps, dq_cpu, dtype)


class TestMatmul4bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 256])
    def test_gemv_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        out_features, in_features = 1024, 256
        A = torch.randn(1, 1, in_features, dtype=dtype)
        B = torch.randn(out_features, in_features, dtype=dtype)

        # Quantize B once on CPU (quantization is bit-exact across devices).
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, torch.uint8)
        code = F.get_4bit_type(quant_type, device="cpu", blocksize=blocksize)

        out_cpu = torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)
        out_mps = torch.ops.bitsandbytes.gemv_4bit(
            A.to("mps"), B_q.to("mps"), B.shape, absmax.to("mps"), code.to("mps"), blocksize
        )

        assert out_mps.shape == (1, 1, out_features)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("compress_statistics", [False, True], ids=id_formatter("compress_statistics"))
    @pytest.mark.parametrize("has_bias", [False, True], ids=id_formatter("has_bias"))
    def test_gemm_4bit(self, dtype, quant_type, compress_statistics, has_bias):
        torch.manual_seed(1337)
        N, K, blocksize = 64, 64, 64
        A = torch.randn(2, 2, K, dtype=dtype)
        B = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=dtype) if has_bias else None

        # Quantize on each device via the public API; parity of the quantization
        # itself is covered by Test4bitParity.
        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B, blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(
            B.to("mps"), blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )

        if compress_statistics:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A,
                B_q,
                list(B.shape),
                qs.state2.absmax,
                blocksize,
                quant_type,
                bias=bias,
                absmax_8bit=qs.absmax,
                absmax_code=qs.state2.code,
                absmax_offset=qs.offset,
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.state2.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
                absmax_8bit=qs_mps.absmax,
                absmax_code=qs_mps.state2.code,
                absmax_offset=qs_mps.offset,
            )
        else:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A, B_q, list(B.shape), qs.absmax, blocksize, quant_type, bias=bias
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
            )

        assert out_mps.shape == (2, 2, N)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)


class TestInt8Parity:
    """LLM.int8() ops on mps all resolve to the "default" (pure-torch) kernels."""

    def test_int8_linear_matmul(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)

        out_cpu = torch.ops.bitsandbytes.int8_linear_matmul(A, B)
        out_mps = torch.ops.bitsandbytes.int8_linear_matmul(A.to("mps"), B.to("mps"))

        assert out_mps.dtype == torch.int32
        # int32 accumulations of int8 products are exactly representable in fp32
        # at these sizes; results must match exactly.
        assert torch.equal(out_mps.cpu(), out_cpu)

    def test_int8_linear_matmul_out(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)

        out_cpu = torch.empty((10, 30), dtype=torch.int32)
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out_cpu)

        out_mps = torch.empty((10, 30), dtype=torch.int32, device="mps")
        torch.ops.bitsandbytes.int8_linear_matmul.out(A.to("mps"), B.to("mps"), out_mps)

        assert torch.equal(out_mps.cpu(), out_cpu)

    @pytest.mark.parametrize("threshold", [0.0, 6.0])
    def test_int8_vectorwise_quant(self, threshold):
        torch.manual_seed(1337)
        A = torch.randn(10, 20, dtype=torch.float16)
        A[1][0] = 1000.0  # outlier

        q_cpu, stats_cpu, outliers_cpu = torch.ops.bitsandbytes.int8_vectorwise_quant(A.clone(), threshold=threshold)
        q_mps, stats_mps, outliers_mps = torch.ops.bitsandbytes.int8_vectorwise_quant(
            A.clone().to("mps"), threshold=threshold
        )

        assert torch.equal(q_mps.cpu(), q_cpu)
        assert_parity(stats_mps, stats_cpu, torch.float32)
        if threshold > 0.0:
            assert outliers_mps is not None
            assert torch.equal(outliers_mps.cpu(), outliers_cpu)
        else:
            assert outliers_mps is None

    def test_int8_vectorwise_dequant(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        stats = torch.rand(10, dtype=torch.float32) * 5

        out_cpu = torch.ops.bitsandbytes.int8_vectorwise_dequant(A, stats)
        out_mps = torch.ops.bitsandbytes.int8_vectorwise_dequant(A.to("mps"), stats.to("mps"))

        assert_parity(out_mps, out_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    def test_int8_mm_dequant(self, dtype):
        torch.manual_seed(1337)
        A = torch.randint(-1000, 1000, (32, 32), dtype=torch.int32)
        row_stats = torch.rand(32, dtype=torch.float32) * 3
        col_stats = torch.rand(32, dtype=torch.float32) * 3

        out_cpu = torch.ops.bitsandbytes.int8_mm_dequant(A, row_stats, col_stats, dtype=dtype)
        out_mps = torch.ops.bitsandbytes.int8_mm_dequant(
            A.to("mps"), row_stats.to("mps"), col_stats.to("mps"), dtype=dtype
        )

        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    @pytest.mark.parametrize("has_bias", [False, True], ids=id_formatter("has_bias"))
    def test_int8_scaled_mm(self, dtype, has_bias):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)
        row_stats = torch.rand(10, dtype=torch.float32)
        col_stats = torch.rand(30, dtype=torch.float32)
        bias = torch.randn(30, dtype=dtype) if has_bias else None

        out_cpu = torch.ops.bitsandbytes.int8_scaled_mm(A, B, row_stats, col_stats, bias=bias, dtype=dtype)
        out_mps = torch.ops.bitsandbytes.int8_scaled_mm(
            A.to("mps"),
            B.to("mps"),
            row_stats.to("mps"),
            col_stats.to("mps"),
            bias=bias.to("mps") if bias is not None else None,
            dtype=dtype,
        )

        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)


def _run_optimizer_32bit_both_devices(optimizer_name: str, weight_decay: float, steps=(1, 2, 3)):
    torch.manual_seed(1337)
    g = torch.randn(256, dtype=torch.float32)
    p = torch.randn(256, dtype=torch.float32)
    state1 = torch.zeros(256, dtype=torch.float32)
    state2 = torch.zeros(256, dtype=torch.float32) if optimizer_name == "adam" else None

    g_mps = g.to("mps")
    p_mps = p.clone().to("mps")
    state1_mps = state1.clone().to("mps")
    state2_mps = state2.clone().to("mps") if state2 is not None else None

    for step in steps:
        args = (0.0, 0.0, 0.9, 0.999, 0.0, 0.0, 1e-8, weight_decay, step, 1e-3, 1.0)
        torch.ops.bitsandbytes.optimizer_update_32bit(optimizer_name, g, p, state1, state2, None, *args)
        torch.ops.bitsandbytes.optimizer_update_32bit(
            optimizer_name, g_mps, p_mps, state1_mps, state2_mps, None, *args
        )

    return (p, state1, state2), (p_mps, state1_mps, state2_mps)


class TestOptimizerParity:
    @pytest.mark.parametrize("optimizer_name", ["adam", "momentum", "rmsprop", "lion"])
    def test_optimizer_update_32bit(self, optimizer_name):
        # On mps this resolves to the "default" (pure-torch) kernel; the cpu oracle
        # runs the dedicated "cpu" kernel from backends/cpu/ops.py.
        # lion + weight_decay>0 parity is covered separately by
        # test_lion_weight_decay_decoupled_parity below; keep wd=0 here to avoid
        # redundant coverage.
        weight_decay = 0.0 if optimizer_name == "lion" else 0.01
        (p, state1, state2), (p_mps, state1_mps, state2_mps) = _run_optimizer_32bit_both_devices(
            optimizer_name, weight_decay
        )

        assert_parity(p_mps, p, torch.float32)
        assert_parity(state1_mps, state1, torch.float32)
        if state2 is not None:
            assert_parity(state2_mps, state2, torch.float32)

    def test_lion_weight_decay_decoupled_parity(self):
        """Regression: lion + weight_decay agrees across backends on mps.

        This was previously a strict-xfail documenting a real divergence -- the
        'default' kernel (used on mps) applied COUPLED weight decay for lion
        (g += p*wd), while the 'cpu' and CUDA kernels applied DECOUPLED decay
        (p *= 1 - lr*wd), matching the Lion paper. The default backend was the
        outlier; it was fixed to be decoupled (#1992 / #1993), so mps and the cpu
        oracle now match. See docs/apple_silicon/MPS_STATUS.md.
        """
        (p, state1, _), (p_mps, state1_mps, _) = _run_optimizer_32bit_both_devices("lion", weight_decay=0.01)

        assert_parity(p_mps, p, torch.float32)
        assert_parity(state1_mps, state1, torch.float32)


class TestKnownGapsOnMps:
    """Ops with no "mps" and no "default" registration must fail loudly on mps.

    These document the current coverage gaps (see docs/apple_silicon/MPS_STATUS.md).
    If one of these tests starts failing because the op now *works* on mps, an
    implementation has been registered: move the op into the parity tests above and
    update MPS_STATUS.md.
    """

    def test_dequantize_blockwise_out_missing(self):
        A = torch.randint(0, 256, (4096,), dtype=torch.uint8, device="mps")
        code = F.create_dynamic_map().to("mps", torch.float32)
        absmax = torch.rand(16, device="mps")
        out = torch.empty(4096, dtype=torch.float32, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.dequantize_blockwise.out(A, absmax, code, 256, torch.float32, out)

    def test_int8_double_quant_missing(self):
        A = torch.randn(10, 20, dtype=torch.float16, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.int8_double_quant(A)

    def test_optimizer_update_8bit_blockwise_missing(self):
        g = torch.randn(256, device="mps")
        p = torch.randn(256, device="mps")
        state1 = torch.zeros(256, dtype=torch.uint8, device="mps")
        qmap = F.create_dynamic_map(signed=True).to("mps")
        absmax = torch.zeros(1, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.optimizer_update_8bit_blockwise(
                "adam", g, p, state1, None, 0.9, 0.999, 0.0, 0.0, 1e-8, 1, 1e-3, qmap, None, absmax, None, 0.0, 1.0
            )
