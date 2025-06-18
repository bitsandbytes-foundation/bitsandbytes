import copy
import os
import pickle
import platform
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.cextension import HIP_ENVIRONMENT
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
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
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
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
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
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
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
