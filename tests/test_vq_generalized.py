"""Correctness tests for generalized VQ kernels (all 5 configs).

Tests all VQ configurations:
  - p=2, index_bits=8  (4.00 bits/wt, BS=32, 256-entry codebook)
  - p=2, index_bits=10 (5.00 bits/wt, BS=32, 1024-entry codebook)
  - p=3, index_bits=8  (2.67 bits/wt, BS=48, 256-entry codebook)
  - p=3, index_bits=10 (3.33 bits/wt, BS=48, 1024-entry codebook)
  - p=4, index_bits=8  (2.00 bits/wt, BS=32, 256-entry codebook)

Covers scalar GEMV (M=1-4), MMA (M=5,8,16), dequant+cuBLAS (M=32),
roundtrip quality, and shape correctness.
"""

import pytest
import torch

# All 5 VQ configurations
VQ_CONFIGS = [
    (2, 8),
    (2, 10),
    (3, 8),
    (3, 10),
    (4, 8),
]

# K_dim must be divisible by BS (32 for p=2/4, 48 for p=3).
# N must be divisible by 128.
# These shapes are chosen to be compatible with all configs.
# For p=3 (BS=48): K_dim must be multiple of 48.
# LCM(32, 48) = 96, so K_dim must be multiple of 96 to work for all configs.
# For practical shapes: use multiples of 96 for K_dim.
SHAPES = [
    (2112, 5120),   # close to Qwen3 2048×5120 (padded to multiple of 96)
    (3072, 2048),   # already multiple of both 32 and 96
    (5120, 2048),   # already multiple of both 32 and 96
]

SHAPES_P3 = [
    (2112, 5120),   # 2112 = 44×48
    (3072, 2048),   # 3072 = 64×48
    (4800, 2048),   # 4800 = 100×48
]


def _bs(p):
    return 48 if p == 3 else 32


def _valid_kdim(K_dim, p):
    """Make K_dim valid for given p by rounding up to BS."""
    BS = _bs(p)
    return ((K_dim + BS - 1) // BS) * BS


def prepare_vq(K_dim, N, p, index_bits, dtype=torch.float16):
    """Quantize a weight matrix with VQ codebook. Returns flat and tiled data."""
    from bitsandbytes.functional import create_vq_codebook, quantize_vq, repack_vq

    codebook = create_vq_codebook(p, device="cuda", index_bits=index_bits)
    W = torch.randn(N, K_dim, dtype=dtype, device="cuda")

    packed_flat, absmax_flat, codebook = quantize_vq(W, p=p, codebook=codebook, index_bits=index_bits)
    packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p, index_bits=index_bits)

    return packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, W


def dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits):
    """Dequantize VQ packed data and reshape to [N, K_dim]."""
    from bitsandbytes.functional import dequantize_vq

    n_total = N * K_dim
    W_flat = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=n_total, index_bits=index_bits)
    return W_flat.reshape(N, K_dim)


def assert_close(actual, expected, max_rel_err, label=""):
    """Check relative error between tensors."""
    diff = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().clamp(min=1.0)
    rel_err = (diff / scale).max().item()
    assert rel_err < max_rel_err, (
        f"{label}Max relative error {rel_err:.6f} exceeds threshold {max_rel_err}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQRoundtrip:
    """Test VQ quantize-dequantize roundtrip quality for all configs."""

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_roundtrip_mse(self, p, index_bits):
        """Roundtrip MSE is within expected bounds per config."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq

        BS = _bs(p)
        N, K_dim = 512, _valid_kdim(2048, p)
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda", index_bits=index_bits)

        packed, absmax, _ = quantize_vq(W, p=p, codebook=codebook, index_bits=index_bits)
        n_total = N * K_dim
        W_deq = dequantize_vq(packed, absmax, codebook, p=p, n=n_total, index_bits=index_bits)
        W_deq = W_deq.reshape(N, K_dim)

        mse = ((W.float() - W_deq.float()) ** 2).mean().item()
        orig_var = (W.float() ** 2).mean().item()
        nmse = mse / orig_var

        # Expected NMSE thresholds per config
        thresholds = {
            (2, 8): 0.10,   # 4.0 bits/wt
            (2, 10): 0.05,  # 5.0 bits/wt (more entries = less error)
            (3, 8): 0.15,   # 2.67 bits/wt
            (3, 10): 0.10,  # 3.33 bits/wt
            (4, 8): 0.25,   # 2.0 bits/wt
        }
        threshold = thresholds[(p, index_bits)]
        assert nmse < threshold, (
            f"p={p}, ib={index_bits}: NMSE {nmse:.4f} exceeds threshold {threshold}"
        )

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_roundtrip_shapes(self, p, index_bits):
        """Roundtrip preserves expected output shapes."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq
        from bitsandbytes._ops import _vq_traits

        traits = _vq_traits(p, index_bits)
        BS = traits["BS"]
        N, K_dim = 256, _valid_kdim(512, p)
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda", index_bits=index_bits)

        packed, absmax, _ = quantize_vq(W, p=p, codebook=codebook, index_bits=index_bits)

        n_total = N * K_dim
        num_blocks = -(n_total // -BS)
        assert absmax.shape == (num_blocks,), f"Expected absmax shape ({num_blocks},), got {absmax.shape}"
        assert packed.shape == (num_blocks * traits["WORDS"],), (
            f"Expected packed shape ({num_blocks * traits['WORDS']},), got {packed.shape}"
        )

        W_deq = dequantize_vq(packed, absmax, codebook, p=p, n=n_total, index_bits=index_bits)
        assert W_deq.shape == (n_total,), f"Expected deq shape ({n_total},), got {W_deq.shape}"

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_flat_vs_tiled_dequant(self, p, index_bits):
        """Flat dequant and tiled dequant produce same results."""
        from bitsandbytes.functional import create_vq_codebook, quantize_vq, dequantize_vq, repack_vq

        N, K_dim = 256, _valid_kdim(512, p)
        W = torch.randn(N, K_dim, dtype=torch.float16, device="cuda")
        codebook = create_vq_codebook(p, device="cuda", index_bits=index_bits)

        packed_flat, absmax_flat, _ = quantize_vq(W, p=p, codebook=codebook, index_bits=index_bits)
        packed_tiled, absmax_tiled = repack_vq(packed_flat, absmax_flat, K_dim, N, p=p, index_bits=index_bits)

        n_total = N * K_dim
        W_flat = dequantize_vq(packed_flat, absmax_flat, codebook, p=p, n=n_total, index_bits=index_bits)
        W_tiled = torch.ops.bitsandbytes.dequantize_vq_tiled(
            packed_tiled, codebook, absmax_tiled, p, K_dim, N, torch.float16, index_bits
        )

        assert torch.equal(W_flat, W_tiled), (
            f"p={p}, ib={index_bits}: flat vs tiled dequant mismatch. "
            f"Max diff: {(W_flat.float() - W_tiled.float()).abs().max().item()}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQScalarGemvGeneralized:
    """Test VQ scalar GEMV for all 5 configs."""

    @pytest.mark.parametrize("M", [1, 2, 3, 4])
    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_basic_correctness(self, M, p, index_bits):
        """VQ scalar GEMV matches dequant + matmul reference."""
        K_dim, N = _valid_kdim(2048, p), 512
        packed_flat, absmax_flat, _, _, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p, index_bits,
        )
        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert C.shape == C_ref.shape
        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, M={M}: ")

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_tiled_matches_flat(self, p, index_bits):
        """Tiled GEMV matches flat GEMV."""
        K_dim, N = _valid_kdim(2048, p), 512
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(1, K_dim, dtype=torch.float16, device="cuda")

        C_flat = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p, index_bits,
        )
        C_tiled = torch.ops.bitsandbytes.vq_scalar_gemv_tiled(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, p, index_bits,
        )

        assert torch.equal(C_flat, C_tiled), (
            f"p={p}, ib={index_bits}: flat vs tiled GEMV mismatch. "
            f"Max diff: {(C_flat.float() - C_tiled.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize("K_dim,N", SHAPES)
    @pytest.mark.parametrize("p,index_bits", [(2, 8), (2, 10), (4, 8)])
    def test_shapes_p2_p4(self, K_dim, N, p, index_bits):
        """VQ scalar GEMV works for representative shapes (p=2/4)."""
        K_dim = _valid_kdim(K_dim, p)
        packed_flat, absmax_flat, _, _, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(1, K_dim, dtype=torch.float16, device="cuda")
        C = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p, index_bits,
        )
        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, ({K_dim},{N}): ")

    @pytest.mark.parametrize("K_dim,N", SHAPES_P3)
    @pytest.mark.parametrize("p,index_bits", [(3, 8), (3, 10)])
    def test_shapes_p3(self, K_dim, N, p, index_bits):
        """VQ scalar GEMV works for representative shapes (p=3)."""
        packed_flat, absmax_flat, _, _, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(1, K_dim, dtype=torch.float16, device="cuda")
        C = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p, index_bits,
        )
        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, ({K_dim},{N}): ")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_dtype(self, dtype, p, index_bits):
        """VQ scalar GEMV works with both fp16 and bf16."""
        K_dim, N = _valid_kdim(2048, p), 512
        packed_flat, absmax_flat, _, _, codebook, _ = prepare_vq(K_dim, N, p, index_bits, dtype=dtype)

        A = torch.randn(2, K_dim, dtype=dtype, device="cuda")
        C = torch.ops.bitsandbytes.vq_scalar_gemv(
            A, packed_flat, absmax_flat, codebook, K_dim, N, p, index_bits,
        )

        assert C.dtype == dtype
        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(dtype)
        tol = 0.25 if dtype == torch.bfloat16 else 0.10
        assert_close(C, C_ref, max_rel_err=tol,
                     label=f"dtype={dtype}, p={p}, ib={index_bits}: ")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQMMAGeneralized:
    """Test VQ MMA kernel (vq_gemm_prod) for all 5 configs."""

    @pytest.mark.parametrize("M", [5, 8, 16])
    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_mma_correctness(self, M, p, index_bits):
        """VQ MMA kernel matches dequant + matmul reference."""
        K_dim, N = _valid_kdim(2048, p), 512
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C = torch.ops.bitsandbytes.vq_gemm_prod(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, p, 1, index_bits,
        )

        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert C.shape == C_ref.shape
        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, M={M}: ")

    @pytest.mark.parametrize("K_dim,N", [(3072, 2048)])
    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_mma_large_shape(self, K_dim, N, p, index_bits):
        """VQ MMA kernel on larger shapes."""
        K_dim = _valid_kdim(K_dim, p)
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        M = 8
        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C = torch.ops.bitsandbytes.vq_gemm_prod(
            A, packed_tiled, absmax_tiled, codebook, K_dim, N, p, 1, index_bits,
        )

        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, ({K_dim},{N}): ")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVQLinearDispatchGeneralized:
    """Test vq_linear dispatch across full M range for all configs."""

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32])
    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_vq_linear_dispatch(self, M, p, index_bits):
        """vq_linear correctly dispatches for various M values."""
        from bitsandbytes.functional import vq_linear

        K_dim, N = _valid_kdim(2048, p), 512
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")

        C = vq_linear(A, packed_tiled, absmax_tiled, codebook, p, K_dim, N, index_bits=index_bits)

        W_deq = dequant_ref(packed_flat, absmax_flat, codebook, p, N, K_dim, index_bits)
        C_ref = (A.float() @ W_deq.float().T).to(A.dtype)

        assert C.shape == (M, N)
        assert_close(C, C_ref, max_rel_err=0.10,
                     label=f"p={p}, ib={index_bits}, M={M}: ")

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_vq_linear_preallocated_output(self, p, index_bits):
        """vq_linear uses pre-allocated output correctly."""
        from bitsandbytes.functional import vq_linear

        K_dim, N = _valid_kdim(2048, p), 512
        M = 4  # scalar GEMV path
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")
        C = vq_linear(A, packed_tiled, absmax_tiled, codebook, p, K_dim, N, out=out, index_bits=index_bits)

        assert C.data_ptr() == out.data_ptr(), "vq_linear didn't use pre-allocated output"

        C_ref = vq_linear(A, packed_tiled, absmax_tiled, codebook, p, K_dim, N, index_bits=index_bits)
        assert torch.allclose(C, C_ref, atol=1e-3, rtol=1e-3), (
            f"Pre-allocated output differs: max diff {(C.float()-C_ref.float()).abs().max():.6f}"
        )

    @pytest.mark.parametrize("p,index_bits", VQ_CONFIGS)
    def test_vq_linear_workspace(self, p, index_bits):
        """vq_linear with workspace produces correct results."""
        from bitsandbytes.functional import vq_linear, vq_linear_workspace

        K_dim, N = _valid_kdim(2048, p), 512
        M = 8  # MMA path
        packed_flat, absmax_flat, packed_tiled, absmax_tiled, codebook, _ = prepare_vq(K_dim, N, p, index_bits)

        A = torch.randn(M, K_dim, dtype=torch.float16, device="cuda")
        ws = vq_linear_workspace(M, K_dim, N, p, torch.float16, torch.device("cuda"))
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")

        C = vq_linear(A, packed_tiled, absmax_tiled, codebook, p, K_dim, N,
                       out=out, workspace=ws, index_bits=index_bits)

        C_ref = vq_linear(A, packed_tiled, absmax_tiled, codebook, p, K_dim, N, index_bits=index_bits)
        assert torch.allclose(C, C_ref, atol=1e-3, rtol=1e-3), (
            f"Workspace output differs: max diff {(C.float()-C_ref.float()).abs().max():.6f}"
        )
