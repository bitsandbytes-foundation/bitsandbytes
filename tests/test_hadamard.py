"""Tests for the Hadamard rotation kernel (hadamard_rotate)."""

import pytest
import torch

from bitsandbytes.functional import hadamard_rotate

BLOCK_SIZES = [32, 64, 128, 256]
FULL_DIMS = [512, 1024, 2048, 4096, 8192]
DTYPES = [torch.float16, torch.bfloat16]


class TestOrthogonality:
    """H(H(x)) ≈ x for plain Hadamard (no signs)."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_double_apply_identity(self, block_size, dtype):
        x = torch.randn(1024, dtype=dtype, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=block_size)
        hadamard_rotate(x, block_size=block_size)
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_double_apply_large(self, block_size, dtype):
        """Test on a larger tensor (32K elements)."""
        x = torch.randn(32768, dtype=dtype, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=block_size)
        hadamard_rotate(x, block_size=block_size)
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)


class TestSignedOrthogonality:
    """Randomized Hadamard: R=H*D is orthogonal (R^T*R=I)."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_signed_inverse(self, block_size, dtype):
        """Verify inv(H*D) = D*H: forward then inverse recovers original."""
        signs = torch.randint(0, 2**31, (block_size // 32,), dtype=torch.int32, device="cuda")
        x = torch.randn(1024, dtype=dtype, device="cuda")
        x_orig = x.clone()

        # Forward: H*D*x
        hadamard_rotate(x, block_size=block_size, signs=signs)

        # Inverse: D*H*x' = first apply H (no signs), then sign flip
        hadamard_rotate(x, block_size=block_size)  # H
        # Apply D (sign flip)
        x_flat = x.view(-1)
        for j in range(block_size // 32):
            word = signs[j].item()
            for bit in range(32):
                if word & (1 << bit):
                    pos = j * 32 + bit
                    x_flat[pos::block_size] *= -1

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)


class TestGEMMEquivalence:
    """H(A) @ H(B)^T ≈ A @ B^T (within quantization tolerance)."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gemm_plain(self, block_size, dtype):
        M, K, N = 4, 256, 8
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B = torch.randn(N, K, dtype=dtype, device="cuda")
        ref = A.float() @ B.float().T

        A_rot = A.clone()
        B_rot = B.clone()
        hadamard_rotate(A_rot, block_size=block_size)
        hadamard_rotate(B_rot, block_size=block_size)
        result = A_rot.float() @ B_rot.float().T

        atol = 0.1 if dtype == torch.bfloat16 else 0.05
        torch.testing.assert_close(result, ref, atol=atol, rtol=0.05)

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gemm_signed(self, block_size, dtype):
        """GEMM equivalence with random sign flips."""
        M, K, N = 4, 256, 8
        signs = torch.randint(0, 2**31, (block_size // 32,), dtype=torch.int32, device="cuda")
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B = torch.randn(N, K, dtype=dtype, device="cuda")
        ref = A.float() @ B.float().T

        A_rot = A.clone()
        B_rot = B.clone()
        hadamard_rotate(A_rot, block_size=block_size, signs=signs)
        hadamard_rotate(B_rot, block_size=block_size, signs=signs)
        result = A_rot.float() @ B_rot.float().T

        atol = 0.1 if dtype == torch.bfloat16 else 0.05
        torch.testing.assert_close(result, ref, atol=atol, rtol=0.05)

    def test_gemm_qwen3_shapes(self):
        """GEMM equivalence on Qwen3-Coder-Next 70B shapes."""
        shapes = [
            (1, 2048, 5120),  # gate/up at M=1
            (4, 5120, 2048),  # down at M=4
            (1, 2048, 4096),  # Q proj
            (4, 4096, 2048),  # O proj
        ]
        for M, K, N in shapes:
            A = torch.randn(M, K, dtype=torch.float16, device="cuda")
            B = torch.randn(N, K, dtype=torch.float16, device="cuda")
            ref = A.float() @ B.float().T

            A_rot = A.clone()
            B_rot = B.clone()
            hadamard_rotate(A_rot, block_size=64)
            hadamard_rotate(B_rot, block_size=64)
            result = A_rot.float() @ B_rot.float().T

            torch.testing.assert_close(result, ref, atol=0.05, rtol=0.05)


class TestEdgeCases:
    """Edge cases: sizes not divisible by block_size, various M values."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_size_not_divisible(self, block_size):
        """When n is not divisible by block_size, the last partial block
        should still be processed (padded with zeros internally)."""
        n = block_size * 3 + 7  # partial block
        x = torch.randn(n, dtype=torch.float16, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=block_size)
        # The rotated values should differ from the original
        assert not torch.allclose(x, x_orig, atol=1e-4)
        # Double-apply should recover the original
        hadamard_rotate(x, block_size=block_size)
        # Full blocks should be exact, partial block may have more error
        full_n = (n // block_size) * block_size
        torch.testing.assert_close(x[:full_n], x_orig[:full_n], atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("n", [32, 64, 128, 256, 512, 1024, 4096])
    def test_various_sizes(self, n):
        x = torch.randn(n, dtype=torch.float16, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=32)
        hadamard_rotate(x, block_size=32)
        torch.testing.assert_close(x, x_orig, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_single_block(self, block_size):
        """Exactly one block."""
        x = torch.randn(block_size, dtype=torch.float16, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=block_size)
        hadamard_rotate(x, block_size=block_size)
        torch.testing.assert_close(x, x_orig, atol=1e-3, rtol=1e-3)

    def test_invalid_block_size(self):
        x = torch.randn(128, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            hadamard_rotate(x, block_size=16)
        with pytest.raises(RuntimeError):
            hadamard_rotate(x, block_size=48)

    def test_invalid_dtype(self):
        x = torch.randn(128, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError):
            hadamard_rotate(x, block_size=32)

    def test_2d_tensor(self):
        """Rotation should work on 2D tensors (flattened internally)."""
        x = torch.randn(8, 64, dtype=torch.float16, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=64)
        hadamard_rotate(x, block_size=64)
        torch.testing.assert_close(x, x_orig, atol=1e-3, rtol=1e-3)


class TestDeterminism:
    """Same input → same output."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_deterministic(self, block_size, dtype):
        x = torch.randn(1024, dtype=dtype, device="cuda")
        a = x.clone()
        b = x.clone()
        hadamard_rotate(a, block_size=block_size)
        hadamard_rotate(b, block_size=block_size)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_deterministic_signed(self, block_size):
        signs = torch.randint(0, 2**31, (block_size // 32,), dtype=torch.int32, device="cuda")
        x = torch.randn(1024, dtype=torch.float16, device="cuda")
        a = x.clone()
        b = x.clone()
        hadamard_rotate(a, block_size=block_size, signs=signs)
        hadamard_rotate(b, block_size=block_size, signs=signs)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


class TestNormPreservation:
    """Hadamard rotation preserves L2 norm (orthogonal transform)."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_norm_preservation(self, block_size, dtype):
        x = torch.randn(block_size * 4, dtype=dtype, device="cuda")
        norm_before = x.float().norm().item()
        hadamard_rotate(x, block_size=block_size)
        norm_after = x.float().norm().item()
        assert abs(norm_after - norm_before) / norm_before < 0.01


# ==================== Full-dimension Hadamard tests ====================


class TestFullDimOrthogonality:
    """H(H(x)) = x for full-dimension Hadamard (block_size=0)."""

    @pytest.mark.parametrize("dim", FULL_DIMS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_double_apply_identity(self, dim, dtype):
        x = torch.randn(1, dim, dtype=dtype, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=0)
        hadamard_rotate(x, block_size=0)
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)

    @pytest.mark.parametrize("dim", FULL_DIMS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multi_row(self, dim, dtype):
        """Full-dimension rotation on a batch of rows."""
        x = torch.randn(8, dim, dtype=dtype, device="cuda")
        x_orig = x.clone()
        hadamard_rotate(x, block_size=0)
        hadamard_rotate(x, block_size=0)
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)


class TestFullDimSignedInverse:
    """Randomized full-dimension Hadamard: inv(H*D) = D*H."""

    @pytest.mark.parametrize("dim", FULL_DIMS[:4])  # skip 8192 for speed
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_signed_inverse(self, dim, dtype):
        signs = torch.randint(0, 2**31, (dim // 32,), dtype=torch.int32, device="cuda")
        x = torch.randn(1, dim, dtype=dtype, device="cuda")
        x_orig = x.clone()

        # Forward: H*D*x
        hadamard_rotate(x, block_size=0, signs=signs)

        # Inverse: D*H*x' = first apply H (no signs), then sign flip
        hadamard_rotate(x, block_size=0)
        x_flat = x.view(-1)
        for j in range(dim // 32):
            word = signs[j].item()
            for bit in range(32):
                if word & (1 << bit):
                    x_flat[j * 32 + bit] *= -1

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(x, x_orig, atol=atol, rtol=atol)


class TestFullDimGEMMEquivalence:
    """H(A) @ H(B)^T = A @ B^T for full-dimension Hadamard."""

    @pytest.mark.parametrize("dim", [512, 1024, 2048])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gemm_plain(self, dim, dtype):
        M, K, N = 4, dim, 8
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B = torch.randn(N, K, dtype=dtype, device="cuda")
        ref = A.float() @ B.float().T

        A_rot = A.clone()
        B_rot = B.clone()
        hadamard_rotate(A_rot, block_size=0)
        hadamard_rotate(B_rot, block_size=0)
        result = A_rot.float() @ B_rot.float().T

        # Larger dims accumulate more rounding error in bf16
        atol = 0.25 if dtype == torch.bfloat16 else 0.05
        torch.testing.assert_close(result, ref, atol=atol, rtol=0.1)

    @pytest.mark.parametrize("dim", [512, 1024, 2048])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gemm_signed(self, dim, dtype):
        """GEMM equivalence with random sign flips (full dimension)."""
        M, K, N = 4, dim, 8
        signs = torch.randint(0, 2**31, (dim // 32,), dtype=torch.int32, device="cuda")
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B = torch.randn(N, K, dtype=dtype, device="cuda")
        ref = A.float() @ B.float().T

        A_rot = A.clone()
        B_rot = B.clone()
        hadamard_rotate(A_rot, block_size=0, signs=signs)
        hadamard_rotate(B_rot, block_size=0, signs=signs)
        result = A_rot.float() @ B_rot.float().T

        atol = 0.25 if dtype == torch.bfloat16 else 0.05
        torch.testing.assert_close(result, ref, atol=atol, rtol=0.1)

    def test_gemm_qwen3_shapes(self):
        """GEMM equivalence on Qwen3-70B power-of-2 shapes with full-dim rotation."""
        shapes = [
            (1, 2048, 4096),  # Q proj
            (4, 4096, 2048),  # O proj
            (1, 2048, 2048),  # square
        ]
        for M, K, N in shapes:
            A = torch.randn(M, K, dtype=torch.float16, device="cuda")
            B = torch.randn(N, K, dtype=torch.float16, device="cuda")
            ref = A.float() @ B.float().T

            A_rot = A.clone()
            B_rot = B.clone()
            hadamard_rotate(A_rot, block_size=0)
            hadamard_rotate(B_rot, block_size=0)
            result = A_rot.float() @ B_rot.float().T

            torch.testing.assert_close(result, ref, atol=0.05, rtol=0.05)


class TestFullDimNormPreservation:
    """Full-dimension Hadamard preserves L2 norm."""

    @pytest.mark.parametrize("dim", FULL_DIMS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_norm_preservation(self, dim, dtype):
        x = torch.randn(4, dim, dtype=dtype, device="cuda")
        norm_before = x.float().norm().item()
        hadamard_rotate(x, block_size=0)
        norm_after = x.float().norm().item()
        assert abs(norm_after - norm_before) / norm_before < 0.01


class TestFullDimDeterminism:
    """Same input -> same output for full-dimension mode."""

    @pytest.mark.parametrize("dim", FULL_DIMS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_deterministic(self, dim, dtype):
        x = torch.randn(1, dim, dtype=dtype, device="cuda")
        a = x.clone()
        b = x.clone()
        hadamard_rotate(a, block_size=0)
        hadamard_rotate(b, block_size=0)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @pytest.mark.parametrize("dim", FULL_DIMS[:4])
    def test_deterministic_signed(self, dim):
        signs = torch.randint(0, 2**31, (dim // 32,), dtype=torch.int32, device="cuda")
        x = torch.randn(1, dim, dtype=torch.float16, device="cuda")
        a = x.clone()
        b = x.clone()
        hadamard_rotate(a, block_size=0, signs=signs)
        hadamard_rotate(b, block_size=0, signs=signs)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
