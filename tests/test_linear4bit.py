import copy
import os
import pickle
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.cextension import HIP_ENVIRONMENT
from tests.helpers import TRUE_FALSE, get_available_devices, id_formatter, torch_load_from_buffer, torch_save_to_buffer

storage = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@pytest.mark.parametrize(
    "device",
    [d for d in get_available_devices() if not (HIP_ENVIRONMENT and d == "cpu")],
)
@pytest.mark.parametrize("quant_storage", ["uint8", "float16", "bfloat16", "float32"])
@pytest.mark.parametrize("bias", TRUE_FALSE, ids=id_formatter("bias"))
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("save_before_forward", TRUE_FALSE, ids=id_formatter("save_before_forward"))
def test_linear_serialization(device, quant_type, compress_statistics, bias, quant_storage, save_before_forward):
    if device == "cpu":
        if quant_type == "fp4":
            pytest.xfail("FP4 is not supported for CPU")
        if quant_storage != "uint8":
            pytest.xfail("Only uint8 storage is supported for CPU")

    original_dtype = torch.float16
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


@pytest.mark.parametrize(
    "device",
    [d for d in get_available_devices() if not (HIP_ENVIRONMENT and d == "cpu")],
)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_copy_param(device, quant_type, blocksize, compress_statistics):
    if device == "cpu":
        if compress_statistics:
            pytest.skip("Currently segfaults on CPU")
        if quant_type == "fp4":
            pytest.xfail("FP4 not supported on CPU")

    tensor = torch.linspace(1, blocksize, blocksize)
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


@pytest.mark.parametrize(
    "device",
    [d for d in get_available_devices() if not (HIP_ENVIRONMENT and d == "cpu")],
)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_deepcopy_param(device, quant_type, blocksize, compress_statistics):
    if device == "cpu":
        if compress_statistics:
            pytest.skip("Currently segfaults on CPU")
        if quant_type == "fp4":
            pytest.xfail("FP4 not supported on CPU")

    tensor = torch.linspace(1, blocksize, blocksize)
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


@pytest.mark.parametrize(
    "device",
    [d for d in get_available_devices() if not (HIP_ENVIRONMENT and d == "cpu")],
)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [64, 128] if not HIP_ENVIRONMENT else [128])
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
def test_params4bit_real_serialization(device, quant_type, blocksize, compress_statistics):
    if device == "cpu":
        if compress_statistics:
            pytest.skip("Currently segfaults on CPU")
        if quant_type == "fp4":
            pytest.xfail("FP4 not supported on CPU")

    original_tensor = torch.linspace(1, blocksize, blocksize, dtype=torch.float32)
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
