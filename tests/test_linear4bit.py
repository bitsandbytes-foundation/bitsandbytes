import copy
import os
import pickle
import platform
import sys
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.cextension import ROCM_WARP_SIZE_64
from tests.helpers import (
    TRUE_FALSE,
    describe_dtype,
    get_available_devices,
    id_formatter,
    is_supported_on_hpu,
    torch_load_from_buffer,
    torch_save_to_buffer,
)

storage = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_storage", ["uint8", "float16", "bfloat16", "float32"])
@pytest.mark.parametrize("original_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("save_before_forward", TRUE_FALSE, ids=id_formatter("save_before_forward"))
def test_linear_serialization(
    device, quant_type, original_dtype, compress_statistics, bias, quant_storage, save_before_forward
):
    if device == "hpu" and not is_supported_on_hpu(quant_type, original_dtype, storage[quant_storage]):
        pytest.skip("This configuration is not supported on HPU.")

    compute_dtype = None
    layer_shape = (300, 400)

    linear = torch.nn.Linear(*layer_shape, dtype=original_dtype, device="cpu")  # original layer

    # Quantizing original layer
    linear_q = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device="meta",
    )
    new_weight = bnb.nn.Params4bit(data=linear.weight, quant_type=quant_type, requires_grad=False)
    linear_q.weight = new_weight
    if bias:
        linear_q.bias = torch.nn.Parameter(linear.bias)
    linear_q = linear_q.to(device)

    # saving to state_dict:
    sd = linear_q.state_dict()

    # restoring from state_dict:
    bias_data2 = sd.pop("bias", None)
    weight_data2 = sd.pop("weight")
    weight2 = bnb.nn.Params4bit.from_prequantized(quantized_stats=sd, data=weight_data2, device=device)

    # creating new layer with same params:
    linear_q2 = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        device="meta",
    )
    # loading weights from state_dict:
    linear_q2.weight = weight2
    if bias:
        linear_q2.bias = torch.nn.Parameter(bias_data2)
    linear_q2 = linear_q2.to(device)

    # MATCHING
    a, b = linear_q.weight, linear_q2.weight

    # Quantizing original layer with specified quant_storage type
    linear_qs = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=bias,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=storage[quant_storage],
        device="meta",
    )
    linear_qs.weight = bnb.nn.Params4bit(
        data=linear.weight,
        requires_grad=False,
        quant_type=quant_type,
        quant_storage=storage[quant_storage],
    )
    if bias:
        linear_qs.bias = torch.nn.Parameter(linear.bias)
    linear_qs = linear_qs.to(device)

    assert a.device == b.device
    assert a.dtype == b.dtype
    assert torch.equal(a, b)

    q0 = a.quant_state
    q1 = b.quant_state
    for attr in ("code", "dtype", "blocksize", "absmax"):
        c, d = getattr(q0, attr), getattr(q1, attr)
        if isinstance(c, torch.Tensor):
            assert torch.equal(c, d)
        else:
            assert c == d, f"{c} != {d}"

    if q0.state2 is not None:
        for attr in ("code", "dtype", "blocksize", "absmax"):
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

    if save_before_forward:
        bytes_4bit = torch_save_to_buffer(linear_q)

    # Forward test
    x = torch.rand(42, layer_shape[0], device=device)
    a = linear_q(x)
    b = linear_q2(x)
    c = linear_qs(x)
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.device == c.device
    assert a.dtype == c.dtype
    assert torch.equal(a, b)
    assert torch.equal(a, c)

    if not save_before_forward:
        bytes_4bit = torch_save_to_buffer(linear_q)
    linear_q3 = torch_load_from_buffer(bytes_4bit)

    # Test moving to CPU and back to GPU
    if device != "cpu":
        linear_q2.to("cpu")
        linear_q2.to(device)
    d = linear_qs(x)
    assert c.dtype == d.dtype
    assert c.device == d.device
    assert torch.equal(c, d)

    d = linear_q3(x)
    assert c.dtype == d.dtype
    assert c.device == d.device
    assert torch.equal(c, d)

    # Saved size ratio test. Target set for layer_shape == (300, 400) w/ bias
    with TemporaryDirectory() as tmpdir:
        state_path_4bit = os.path.join(tmpdir, "state_4bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")
        torch.save(linear.state_dict(), state_path)
        torch.save(linear_q.state_dict(), state_path_4bit)

        size_orig, size_4 = (
            os.path.getsize(state_path),
            os.path.getsize(state_path_4bit),
        )
        size_ratio = size_4 / size_orig
        target_compression = (
            0.143 if original_dtype == torch.float32 else 0.29
        )  # these numbers get lower as weight shape increases
        ratio_error_msg = (
            f"quantized_size {size_4:,} is larger on disk than {target_compression:.2%} of original size {size_orig:,}"
        )
        assert size_ratio < target_compression, ratio_error_msg


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [32, 64, 128] if not ROCM_WARP_SIZE_64 else [64, 128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_copy_param(device, quant_type, blocksize, compress_statistics):
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    tensor = torch.randn(300, 400)
    param = bnb.nn.Params4bit(
        data=tensor,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        requires_grad=False,
    ).to(device)

    shallow_copy_param = copy.copy(param)
    assert param.quant_state is shallow_copy_param.quant_state
    assert param.data.data_ptr() == shallow_copy_param.data.data_ptr()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_params4bit_torch_chunk_split(device, quant_type):
    """Test that torch.chunk and torch.split preserve Params4bit subclass for FSDP2 compatibility."""
    if device == "hpu" and not is_supported_on_hpu(quant_type, torch.float16, torch.uint8):
        pytest.skip("This configuration is not supported on HPU.")

    if device == "cpu":
        pytest.skip("CPU quantization causes segfault, skipping CPU test")

    original_tensor = torch.randn(8, 4, dtype=torch.float16, device="cpu")

    params4bit = bnb.nn.Params4bit(data=original_tensor, quant_type=quant_type, requires_grad=False)

    if device != "cpu":
        params4bit = params4bit.to(device)

    chunks = torch.chunk(params4bit, 2, dim=0)

    assert isinstance(chunks, tuple), "torch.chunk should return tuple"
    for chunk in chunks:
        assert isinstance(chunk, bnb.nn.Params4bit), "Chunk should preserve Params4bit subclass"
        assert hasattr(chunk, "quant_type"), "Should preserve metadata"
        assert chunk.quant_type == params4bit.quant_type, "Should preserve quant_type value"

    splits = torch.split(params4bit, 2, dim=0)

    assert isinstance(splits, tuple), "torch.split should return tuple"
    assert len(splits) > 0, "Should have at least one split"
    for split in splits:
        assert isinstance(split, bnb.nn.Params4bit), "Split should preserve Params4bit subclass"
        assert hasattr(split, "quant_type"), "Should preserve metadata"
        assert split.quant_type == params4bit.quant_type, "Should preserve quant_type value"


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize(
    "quant_storage",
    [torch.uint8, torch.float16, torch.bfloat16, torch.float32],
    ids=describe_dtype,
)
def test_quant_storage_shard_roundtrip(device, quant_type, quant_storage):
    """Test that quantized weights survive a flatten-chunk-reassemble roundtrip.

    Non-uint8 quant_storage exists so that FSDP can shard quantized tensors
    without splitting packed 4-bit pairs. This test simulates FSDP's
    shard/gather pattern and verifies numerical correctness after reassembly.
    """
    M, K = 256, 128
    A = torch.randn(1, K, dtype=torch.float16, device=device)
    B = torch.randn(M, K, dtype=torch.float16, device=device)

    qB, state = bnb.functional.quantize_4bit(B, quant_type=quant_type, quant_storage=quant_storage)
    ref = bnb.functional.gemv_4bit(A, qB.t(), state=state)

    # Simulate FSDP: flatten, split into shards, reassemble
    flat = qB.flatten()
    n_shards = 4
    shards = flat.chunk(n_shards)
    reassembled = torch.cat(shards).reshape(qB.shape)

    assert reassembled.dtype == qB.dtype
    assert torch.equal(reassembled.view(torch.uint8), qB.view(torch.uint8)), "Bytes changed after shard roundtrip"

    out = bnb.functional.gemv_4bit(A, reassembled.t(), state=state)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [32, 64, 128] if not ROCM_WARP_SIZE_64 else [64, 128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_deepcopy_param(device, quant_type, blocksize, compress_statistics):
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    tensor = torch.randn(300, 400)
    param = bnb.nn.Params4bit(
        data=tensor,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        requires_grad=False,
    ).to(device)
    dict_keys_before = set(param.__dict__.keys())
    copy_param = copy.deepcopy(param)
    dict_keys_after = set(param.__dict__.keys())
    dict_keys_copy = set(copy_param.__dict__.keys())

    assert param.quant_state is not copy_param.quant_state
    assert param.data.data_ptr() != copy_param.data.data_ptr()

    # there was a bug where deepcopy would modify the original object
    assert dict_keys_before == dict_keys_after
    assert dict_keys_before == dict_keys_copy


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [32, 64, 128] if not ROCM_WARP_SIZE_64 else [64, 128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_params4bit_real_serialization(device, quant_type, blocksize, compress_statistics):
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    original_tensor = torch.randn(300, 400)
    original_param = bnb.nn.Params4bit(
        data=original_tensor,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
    )
    dict_keys_before = set(original_param.__dict__.keys())

    original_param.to(device)  # change device to trigger quantization

    serialized_param = pickle.dumps(original_param)
    deserialized_param = pickle.loads(serialized_param)
    dict_keys_after = set(original_param.__dict__.keys())
    dict_keys_deserialized = set(deserialized_param.__dict__.keys())

    assert torch.equal(original_param.data, deserialized_param.data)
    assert original_param.requires_grad == deserialized_param.requires_grad == False
    assert original_param.quant_type == deserialized_param.quant_type
    assert original_param.blocksize == deserialized_param.blocksize
    assert original_param.compress_statistics == deserialized_param.compress_statistics
    assert original_param.quant_state == deserialized_param.quant_state

    # there was a bug where deepcopy would modify the original object
    assert dict_keys_before == dict_keys_after
    assert dict_keys_before == dict_keys_deserialized


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
@pytest.mark.parametrize("fullgraph", TRUE_FALSE, ids=id_formatter("fullgraph"))
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"], ids=id_formatter("mode"))
@pytest.mark.skipif(torch.__version__ < (2, 4), reason="Not supported in torch < 2.4")
@pytest.mark.skipif(
    torch.__version__ < (2, 10) and sys.version_info >= (3, 14), reason="Not supported in Python 3.14 until torch 2.10"
)
def test_linear4bit_torch_compile(device, quant_type, compute_dtype, compress_statistics, bias, fullgraph, mode):
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    if fullgraph and torch.__version__ < (2, 8, 0, "dev"):
        pytest.skip("fullgraph mode requires torch 2.8 or higher")

    if device == "cuda" and platform.system() == "Windows":
        pytest.skip("Triton is not officially supported on Windows")

    # Has a strange regression on Linux aarch64 CPU in torch==2.6.0 when fullgraph=False.
    if (
        not fullgraph
        and device == "cpu"
        and platform.machine() == "aarch64"
        and platform.system() == "Linux"
        and ((2, 7) > torch.__version__ >= (2, 6))
    ):
        pytest.xfail("Regression in torch==2.6.0 on Linux aarch64 CPU")

    dim = 256
    batch_size = 16

    torch.compiler.reset()

    # Create a small network with Linear4bit layers
    net = torch.nn.Sequential(
        *[
            bnb.nn.Linear4bit(
                dim,
                dim,
                bias=bias,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            for _ in range(4)
        ]
    ).to(device)

    # Create input tensor
    x = torch.randn(batch_size, dim, dtype=compute_dtype, device=device)

    # Get reference output before compilation
    with torch.no_grad():
        ref_output = net(x)

    # Compile the model
    compile_backend = "hpu_backend" if device == "hpu" else "inductor"
    compiled_net = torch.compile(net, fullgraph=fullgraph, mode=mode, backend=compile_backend)

    # Get output from compiled model
    with torch.no_grad():
        compiled_output = compiled_net(x)

    # Check outputs match
    assert compiled_output.shape == ref_output.shape
    assert compiled_output.device == ref_output.device
    assert compiled_output.dtype == ref_output.dtype
    torch.testing.assert_close(compiled_output, ref_output)

    # Test with gradients
    x.requires_grad_(True)
    y1 = net(x).sum()
    y1.backward()
    grad_ref = x.grad.clone()

    x.grad = None
    y2 = compiled_net(x).sum()
    y2.backward()
    grad_compiled = x.grad.clone()

    torch.testing.assert_close(grad_compiled, grad_ref)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_params4bit_quant_state_attr_access(device, quant_type, compress_statistics):
    """Test that Params4bit proxies QuantState attributes for FSDP state_dict traversal (#1405).

    PyTorch's FSDP state_dict machinery traverses FQN paths like
    'model.layers.0.weight.absmax' using getattr(). This test verifies
    that Params4bit and QuantState expose the attributes that appear as
    state_dict keys so that _get_fqns() traversal succeeds.
    """
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    layer = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    layer = layer.to(device)
    w = layer.weight

    assert w.quant_state is not None, "quant_state should be set after quantization"

    # Direct QuantState attributes proxied through Params4bit
    assert torch.equal(w.absmax, w.quant_state.absmax)
    assert torch.equal(w.code, w.quant_state.code)

    # "quant_map" is how as_dict() serializes "code" — FSDP uses this key name
    assert torch.equal(w.quant_map, w.quant_state.code)

    # QuantState packed key: as_dict(packed=True) produces "quant_state.bitsandbytes__<type>"
    # FSDP resolves this as getattr(quant_state_obj, "bitsandbytes__<type>")
    packed_attr = f"bitsandbytes__{quant_type}"
    assert hasattr(w.quant_state, packed_attr)
    packed_val = getattr(w.quant_state, packed_attr)
    assert isinstance(packed_val, torch.Tensor)

    # Simulate the full FSDP _get_fqns traversal for all state_dict keys
    state_dict_keys = list(w.quant_state.as_dict(packed=True).keys())
    for key in state_dict_keys:
        # Each key is relative to "weight.", e.g. "absmax" or "quant_state.bitsandbytes__nf4"
        parts = key.split(".")
        obj = w
        for part in parts:
            obj = getattr(obj, part)
        assert obj is not None

    # hasattr should return True for proxied attrs, False for unknown ones
    assert hasattr(w, "absmax")
    assert hasattr(w, "code")
    assert hasattr(w, "quant_map")
    assert not hasattr(w, "nonexistent_attribute")

    # Unknown attributes must still raise AttributeError
    with pytest.raises(AttributeError, match="nonexistent_attribute"):
        _ = w.nonexistent_attribute

    # Verify that normal Params4bit attributes are unaffected by __getattr__
    assert isinstance(w.quant_state, bnb.functional.QuantState)
    assert isinstance(w.bnb_quantized, bool)
    assert w.bnb_quantized is True
