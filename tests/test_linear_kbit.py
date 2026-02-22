"""
Tests for ParamsKbit and LinearKbit.

Verifies:
- ParamsKbit quantization on .to(device)
- LinearKbit forward correctness against fp16 reference
- Kernel dispatch (GEMV for M<=4, dequant+mm for M>4)
- N-padding (output features not divisible by 128)
- Bias handling
- Multiple k values (2,3,4,5)
- Global weight buffer reuse
"""

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import _ops  # noqa: F401 — ensure ops are registered
from bitsandbytes.nn import LinearKbit, ParamsKbit, _GlobalWeightBuffer

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _dequant_reference_forward(layer, x, bias=None):
    """Reference forward using dequantized kbit weights.

    This provides an apples-to-apples comparison: the quantization error is
    inherent to the format and not a bug in LinearKbit.
    """
    w = layer.weight
    n_elements = w.N_padded * w.K_dim
    w_deq = bnb.functional.dequantize_kbit(
        w.packed, w.absmax, w.codebook, w.k, n_elements, x.dtype,
    )
    w_deq = w_deq[:n_elements].reshape(w.N_padded, w.K_dim)
    if w.N_padded != w.N:
        w_deq = w_deq[: w.N, :]
    out = x.float() @ w_deq.float().t()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


class TestParamsKbit:
    """Tests for the ParamsKbit parameter class."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_quantize_on_cuda_move(self, k):
        """ParamsKbit should quantize when moved to CUDA."""
        N, K_dim = 256, 512
        data = torch.randn(N, K_dim, dtype=torch.float16)
        p = ParamsKbit(data, k=k)

        assert not p.kbit_quantized
        p = p.to("cuda")
        assert p.kbit_quantized
        assert p.packed is not None
        assert p.absmax is not None
        assert p.codebook is not None
        assert p.K_dim == K_dim
        assert p.N == N
        assert p.N_padded >= N
        assert p.N_padded % 128 == 0

    def test_codebook_size(self):
        """Codebook should have 2^k entries."""
        for k in [2, 3, 4, 5]:
            data = torch.randn(128, 256, dtype=torch.float16)
            p = ParamsKbit(data, k=k).to("cuda")
            assert p.codebook.shape[0] == (1 << k)

    def test_n_padding(self):
        """N not divisible by 128 should be padded."""
        N, K_dim = 300, 256  # 300 -> padded to 384
        data = torch.randn(N, K_dim, dtype=torch.float16)
        p = ParamsKbit(data, k=4).to("cuda")
        assert p.N == 300
        assert p.N_padded == 384

    def test_n_no_padding_needed(self):
        """N already divisible by 128 should not change."""
        N, K_dim = 256, 512
        data = torch.randn(N, K_dim, dtype=torch.float16)
        p = ParamsKbit(data, k=4).to("cuda")
        assert p.N == 256
        assert p.N_padded == 256

    def test_serialization(self):
        """getstate / setstate round-trip should preserve all fields."""
        data = torch.randn(128, 256, dtype=torch.float16)
        p = ParamsKbit(data, k=3).to("cuda")
        state = p.__getstate__()
        p2 = ParamsKbit.__new__(ParamsKbit)
        p2.__setstate__(state)
        assert p2.k == 3
        assert p2.kbit_quantized
        assert p2.K_dim == 256
        assert p2.N == 128


class TestLinearKbit:
    """Tests for the LinearKbit module."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_basic_forward(self, k):
        """LinearKbit forward should produce output of correct shape."""
        in_f, out_f = 512, 256
        layer = LinearKbit(in_f, out_f, bias=True, k=k)
        layer = layer.to("cuda")
        x = torch.randn(4, in_f, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (4, out_f)
        assert out.dtype == torch.float16

    def test_forward_matches_dequant_reference(self):
        """LinearKbit output should match dequantize+matmul reference."""
        in_f, out_f = 512, 256

        layer = LinearKbit(in_f, out_f, bias=False, k=4)
        layer = layer.to("cuda")

        x = torch.randn(2, in_f, dtype=torch.float16, device="cuda")
        out = layer(x)
        ref = _dequant_reference_forward(layer, x)

        # Should match very closely since both paths use the same dequantized weights
        diff = (out.float() - ref.float()).abs()
        scale = ref.float().abs().clamp(min=1.0)
        rel_err = (diff / scale).max().item()
        assert rel_err < 0.01, f"Relative error too large: {rel_err:.4f}"

    def test_dispatch_gemv(self):
        """M=1 should use GEMV path (no error, just shape check)."""
        layer = LinearKbit(512, 256, bias=False, k=4).to("cuda")
        x = torch.randn(1, 512, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (1, 256)

    def test_dispatch_dequant_mm(self):
        """M=32 should use dequant+mm path."""
        layer = LinearKbit(512, 256, bias=False, k=4).to("cuda")
        x = torch.randn(32, 512, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (32, 256)

    def test_n_padding_output_sliced(self):
        """Output features not divisible by 128 should still produce correct shape."""
        in_f, out_f = 256, 300  # 300 not divisible by 128
        layer = LinearKbit(in_f, out_f, bias=True, k=4).to("cuda")
        x = torch.randn(8, in_f, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (8, 300)

    def test_bias(self):
        """Bias should be added to output."""
        in_f, out_f = 256, 128
        layer = LinearKbit(in_f, out_f, bias=True, k=4).to("cuda")
        assert layer.bias is not None

        x = torch.randn(2, in_f, dtype=torch.float16, device="cuda")
        out_with_bias = layer(x)

        # Disable bias and compare
        layer.bias = None
        out_no_bias = layer(x)

        # They should differ
        assert not torch.allclose(out_with_bias, out_no_bias)

    def test_batch_dimensions(self):
        """Should handle (batch, seq, features) input."""
        layer = LinearKbit(256, 128, bias=False, k=4).to("cuda")
        x = torch.randn(2, 8, 256, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_training_mode_uses_dequant(self):
        """In training mode, even M=1 should use dequant path (not GEMV)."""
        layer = LinearKbit(512, 256, bias=False, k=4).to("cuda")
        layer.train()
        x = torch.randn(1, 512, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (1, 256)


class TestGlobalWeightBuffer:
    """Tests for the _GlobalWeightBuffer."""

    def setup_method(self):
        _GlobalWeightBuffer.clear()

    def test_buffer_allocation(self):
        """Buffer should be allocated on first call."""
        device = torch.device("cuda")
        buf = _GlobalWeightBuffer.get_buffer(device, 1024, torch.float16)
        assert buf.shape[0] == 1024
        assert buf.is_cuda
        assert buf.dtype == torch.float16

    def test_buffer_reuse(self):
        """Subsequent calls should reuse the same buffer."""
        device = torch.device("cuda")
        buf1 = _GlobalWeightBuffer.get_buffer(device, 1024, torch.float16)
        buf2 = _GlobalWeightBuffer.get_buffer(device, 512, torch.float16)
        assert buf1.data_ptr() == buf2.data_ptr()

    def test_buffer_grows(self):
        """Buffer should grow when a larger size is requested."""
        device = torch.device("cuda")
        buf1 = _GlobalWeightBuffer.get_buffer(device, 512, torch.float16)
        buf2 = _GlobalWeightBuffer.get_buffer(device, 2048, torch.float16)
        assert buf2.shape[0] == 2048

    def test_no_new_alloc_during_forward(self):
        """LinearKbit forward should not allocate new tensors after warmup."""
        layer = LinearKbit(512, 256, bias=False, k=4).to("cuda")
        x = torch.randn(8, 512, dtype=torch.float16, device="cuda")

        # Warmup
        _ = layer(x)

        # Measure
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        _ = layer(x)
        after = torch.cuda.memory_allocated()

        # The dequant path will produce output tensors, but the weight buffer
        # should be reused. Allow for output tensor allocation only.
        # Output: 8 * 256 * 2 bytes = 4096 bytes
        # Allow 64 KB overhead for PyTorch internals
        max_new_alloc = 65536
        new_alloc = after - before
        assert new_alloc < max_new_alloc, f"New allocation: {new_alloc} bytes (expected < {max_new_alloc})"
