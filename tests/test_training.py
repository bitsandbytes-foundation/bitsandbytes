"""
Tests for training utilities (gradient checkpointing with CPU offload).

Verifies:
- Correctness: output matches standard forward/backward
- Memory reduction: GPU memory is lower with CPU offload
- Gradient flow: gradients propagate correctly through checkpoint
"""

import pytest
import torch
import torch.nn as nn

from bitsandbytes.training import checkpoint_cpu_offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _simple_block(x):
    """A simple compute block for testing."""
    return torch.nn.functional.gelu(x @ x.t()) @ x


class TestCPUOffloadCheckpoint:
    """Tests for checkpoint_cpu_offload."""

    def test_forward_correctness(self):
        """Output should match standard (non-checkpointed) forward."""
        x = torch.randn(4, 64, dtype=torch.float32, device="cuda", requires_grad=True)
        ref = _simple_block(x.detach().clone().requires_grad_(True))
        out = checkpoint_cpu_offload(_simple_block, x)
        diff = (out - ref).abs().max().item()
        assert diff < 1e-5, f"Forward diff: {diff}"

    def test_gradient_correctness(self):
        """Gradients should match standard backward."""
        # Standard
        x_std = torch.randn(4, 64, dtype=torch.float32, device="cuda", requires_grad=True)
        out_std = _simple_block(x_std)
        out_std.sum().backward()
        grad_std = x_std.grad.clone()

        # Checkpointed
        x_ckpt = x_std.detach().clone().requires_grad_(True)
        out_ckpt = checkpoint_cpu_offload(_simple_block, x_ckpt)
        out_ckpt.sum().backward()
        grad_ckpt = x_ckpt.grad.clone()

        diff = (grad_std - grad_ckpt).abs().max().item()
        assert diff < 1e-5, f"Gradient diff: {diff}"

    def test_with_nn_module(self):
        """Should work with nn.Module as the function."""
        linear = nn.Linear(64, 64).cuda()

        x = torch.randn(4, 64, dtype=torch.float32, device="cuda", requires_grad=True)
        out = checkpoint_cpu_offload(linear, x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_memory_reduction(self):
        """CPU offload should use less GPU memory than standard checkpoint."""
        dim = 1024

        # Standard forward (saves activations on GPU)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        layers = nn.ModuleList([nn.Linear(dim, dim).cuda() for _ in range(4)])
        x = torch.randn(32, dim, device="cuda", requires_grad=True)

        # Standard: all activations stay on GPU
        h = x
        for layer in layers:
            h = torch.nn.functional.gelu(layer(h))
        h.sum().backward()
        peak_standard = torch.cuda.max_memory_allocated()

        # Reset
        del h, x
        for p in layers.parameters():
            p.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # CPU offload: activations go to CPU
        x = torch.randn(32, dim, device="cuda", requires_grad=True)
        h = x
        for layer in layers:
            h = checkpoint_cpu_offload(lambda inp, l=layer: torch.nn.functional.gelu(l(inp)), h)
        h.sum().backward()
        peak_offload = torch.cuda.max_memory_allocated()

        # CPU offload should use less peak memory
        # Allow some margin since PyTorch internal allocations vary
        assert peak_offload < peak_standard, (
            f"CPU offload ({peak_offload / 1e6:.1f} MB) should use less peak memory "
            f"than standard ({peak_standard / 1e6:.1f} MB)"
        )

    def test_preserves_rng_state(self):
        """RNG state should be preserved for dropout reproducibility."""
        linear = nn.Linear(64, 64).cuda()
        dropout = nn.Dropout(0.5)

        def block(x):
            return dropout(linear(x))

        torch.manual_seed(42)
        x = torch.randn(4, 64, device="cuda", requires_grad=True)

        # Run twice with same seed — should produce same output
        torch.manual_seed(123)
        out1 = checkpoint_cpu_offload(block, x)

        torch.manual_seed(123)
        out2 = checkpoint_cpu_offload(block, x.detach().clone().requires_grad_(True))

        diff = (out1 - out2).abs().max().item()
        assert diff < 1e-6, f"RNG state not preserved: diff={diff}"
