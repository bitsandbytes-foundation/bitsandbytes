import os
from contextlib import nullcontext
from itertools import product
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.nn.modules import Linear4bit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="this test requires a GPU")
@pytest.mark.parametrize(
    "quant_type, compress_statistics, bias",
    list(product(["nf4", "fp4"], [False, True], [False, True])),
)
def test_linear_serialization(quant_type, compress_statistics, bias):
    original_dtype = torch.float16
    compute_dtype = None
    device = "cuda"
    layer_shape = (300, 400)

    linear = torch.nn.Linear(*layer_shape, dtype=original_dtype)  # original layer

    # Quantizing original layer
    linear_q = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device=device,
    )
    new_weight = bnb.nn.Params4bit(data=linear.weight, requires_grad=False)
    linear_q.weight = new_weight.to(device)
    if bias:
        linear_q.bias.data = linear.bias.data.to(device)

    # saving to state_dict:
    sd = linear_q.state_dict()

    # creating new layer with same params:
    linear_q2 = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device=device,                  # TODO create on meta device to save loading time
    )
    # loading weights from state_dict:
    linear_q2.load_state_dict(sd)    

    # MATCHING
    a, b = linear_q.weight, linear_q2.weight

    assert a.device == b.device
    assert a.dtype == b.dtype
    assert torch.equal(a, b)
    
    q0 = a.quant_state
    q1 = b.quant_state
    for attr in ('code', 'dtype', 'blocksize', 'absmax'):
        c, d = getattr(q0, attr), getattr(q1, attr)
        if isinstance(c, torch.Tensor):
            assert torch.equal(c, d)
        else:
            assert c == d, f"{c} != {d}"

    if q0.state2 is not None:
        for attr in ('code', 'dtype', 'blocksize', 'absmax'):
            c, d = getattr(q0.state2, attr), getattr(q1.state2, attr)
            if isinstance(c, torch.Tensor):
                assert torch.equal(c, d)
            else:
                assert c == d, f"{c} != {d}"

    if bias:
        a, b = linear_q.bias, linear_q2.bias
        assert a.device == b.device
        assert a.dtype == b.dtype
        assert torch.equal(a, b)

    # Forward test
    x = torch.rand(42, layer_shape[0], device=device)
    a = linear_q(x)
    b = linear_q2(x)
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert torch.equal(a, b)

    # Saved size ratio test. Target set for layer_shape == (300, 400) w/ bias
    with TemporaryDirectory() as tmpdir:
        state_path_4bit = os.path.join(tmpdir, "state_4bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")
        torch.save(linear.state_dict(), state_path)
        torch.save(linear_q.state_dict(), state_path_4bit)

        size_orig, size_4 = os.path.getsize(state_path), os.path.getsize(
            state_path_4bit
        )
        size_ratio = size_4 / size_orig
        target_compression = 0.143 if original_dtype == torch.float32 else 0.285
        ratio_error_msg = f"quantized_size {size_4:,} is larger on disk than {target_compression:.2%} of original size {size_orig:,}"
        assert size_ratio < target_compression, ratio_error_msg
