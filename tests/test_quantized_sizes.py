"""Validate that quantized tensor size formulas exactly match actual quantize_kbit output.

These formulas are used by the streaming quantizer (two-pass) to compute the
safetensors header before any GPU quantization happens. If the formulas are
wrong, the safetensors file will be corrupted.

The formulas (from _ops.py):
  N_padded = ceil(N / 128) * 128
  n_elements = N_padded * K
  num_blocks = ceil(n_elements / 32)
  packed_numel = num_blocks * k + k      # int32 elements
  absmax_numel = num_blocks + 1          # float32 elements
  codebook_numel = 2^k                   # float32 elements
"""

import pytest
import torch

import bitsandbytes.functional as F


def compute_quantized_sizes(N: int, K: int, k: int) -> dict:
    """Compute quantized tensor sizes for a weight matrix [N, K].

    This is the formula that the streaming quantizer will use.
    """
    N_padded = ((N + 127) // 128) * 128
    n_elements = N_padded * K
    num_blocks = -(n_elements // -32)  # ceil_div

    packed_numel = num_blocks * k + k
    absmax_numel = num_blocks + 1
    codebook_numel = 1 << k

    return {
        "N_padded": N_padded,
        "n_elements": n_elements,
        "num_blocks": num_blocks,
        "packed_numel": packed_numel,
        "absmax_numel": absmax_numel,
        "codebook_numel": codebook_numel,
    }


# Standard N values (multiples of 128)
N_VALUES_STANDARD = [128, 256, 512, 768, 1024, 1536, 2048, 4096, 12288]
# Edge case N values (NOT multiples of 128)
N_VALUES_EDGE = [100, 300, 1000]
# K values
K_VALUES = [128, 512, 1024, 2048, 4096, 5120]
# k values (bit widths)
K_BIT_VALUES = [2, 3, 4, 5]


@pytest.mark.parametrize("k", K_BIT_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("N", N_VALUES_STANDARD + N_VALUES_EDGE)
def test_quantized_sizes_match(N, K, k):
    """Verify formula-predicted sizes match actual quantize_kbit output."""
    predicted = compute_quantized_sizes(N, K, k)
    N_padded = predicted["N_padded"]

    # Create a tensor with the padded size
    A = torch.randn(N_padded * K, device="cuda", dtype=torch.float32)

    # Actually quantize
    packed, absmax, codebook = F.quantize_kbit(A, k=k, absmax_format="fp32")

    # Compare sizes
    assert packed.numel() == predicted["packed_numel"], (
        f"packed size mismatch for N={N}, K={K}, k={k}: "
        f"got {packed.numel()}, expected {predicted['packed_numel']}"
    )
    assert absmax.numel() == predicted["absmax_numel"], (
        f"absmax size mismatch for N={N}, K={K}, k={k}: "
        f"got {absmax.numel()}, expected {predicted['absmax_numel']}"
    )
    assert codebook.numel() == predicted["codebook_numel"], (
        f"codebook size mismatch for N={N}, K={K}, k={k}: "
        f"got {codebook.numel()}, expected {predicted['codebook_numel']}"
    )

    # Verify N_padded is correct
    assert N_padded >= N
    assert N_padded % 128 == 0
    assert N_padded - N < 128


@pytest.mark.parametrize("k", K_BIT_VALUES)
def test_codebook_size(k):
    """Verify codebook has 2^k entries."""
    codebook = F.create_normal_float_codebook(k, device="cuda")
    assert codebook.numel() == (1 << k)


def test_glm47_q_proj_sizes():
    """Verify formula with GLM-4.7 q_proj dimensions (a real-world case)."""
    # GLM-4.7: num_heads=96, head_dim=128 → N=12288, K=5120
    N, K, k = 12288, 5120, 4
    predicted = compute_quantized_sizes(N, K, k)

    assert predicted["N_padded"] == 12288  # already multiple of 128
    assert predicted["n_elements"] == 12288 * 5120
    assert predicted["num_blocks"] == -(12288 * 5120 // -32)
    assert predicted["packed_numel"] == predicted["num_blocks"] * 4 + 4
    assert predicted["codebook_numel"] == 16

    # Verify against actual quantization
    A = torch.randn(predicted["n_elements"], device="cuda", dtype=torch.float32)
    packed, absmax, codebook = F.quantize_kbit(A, k=k, absmax_format="fp32")
    assert packed.numel() == predicted["packed_numel"]
    assert absmax.numel() == predicted["absmax_numel"]


def test_glm47_expert_gate_sizes():
    """Verify formula with GLM-4.7 expert gate_proj dimensions."""
    # GLM-4.7 expert: intermediate=1536, hidden=5120, NF2
    N, K, k = 1536, 5120, 2
    predicted = compute_quantized_sizes(N, K, k)

    assert predicted["N_padded"] == 1536  # already multiple of 128
    assert predicted["codebook_numel"] == 4  # 2^2 = 4 for NF2

    A = torch.randn(predicted["n_elements"], device="cuda", dtype=torch.float32)
    packed, absmax, codebook = F.quantize_kbit(A, k=k, absmax_format="fp32")
    assert packed.numel() == predicted["packed_numel"]
    assert absmax.numel() == predicted["absmax_numel"]
