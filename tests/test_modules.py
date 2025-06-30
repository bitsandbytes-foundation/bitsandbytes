import inspect

import pytest
import torch
from torch import nn

import bitsandbytes as bnb
from tests.helpers import get_available_devices, id_formatter, is_supported_on_hpu


class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


class MLP8bit(torch.nn.Module):
    def __init__(self, dim1, dim2, has_fp16_weights=True, threshold=0.0):
        super().__init__()
        self.fc1 = bnb.nn.Linear8bitLt(
            dim1,
            dim2,
            has_fp16_weights=has_fp16_weights,
            threshold=threshold,
        )
        self.fc2 = bnb.nn.Linear8bitLt(
            dim2,
            dim1,
            has_fp16_weights=has_fp16_weights,
            threshold=threshold,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_args():
    args = MockArgs([])
    args.quant_type = "vector"
    args.use_8bit_training = "full"
    args.clip_freq = 9999
    return args


def assert_all_approx_close(a, b, atol=1e-8, rtol=1e-5, count=10):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        print(f"Too many values not close: assert {sumval} < {count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("threshold", [0.0, 3.0], ids=id_formatter("threshold"))
def test_linear8bitlt_inference(device, threshold):
    l1 = bnb.nn.Linear8bitLt(32, 64, threshold=threshold, has_fp16_weights=False).to(device).half()
    assert l1.weight.device.type == device
    assert l1.weight.dtype == torch.int8

    l1.eval()
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device=device).half()
        o1 = l1(b1)
        if i == 1:
            assert l1.state.CB is not None


# TODO: Remove support for training int8 weights
@pytest.mark.parametrize("device", get_available_devices())
def test_linear8bitlt_accumulated_gradient(device):
    if device != "cuda":
        pytest.skip("Only supported on CUDA")

    l1 = torch.nn.Sequential(*[bnb.nn.Linear8bitLt(32, 32).to(device).half() for i in range(2)])
    l2 = torch.nn.Sequential(*[torch.nn.Linear(32, 32).to(device).half() for i in range(2)])
    l1[0].weight.data.copy_(l2[0].weight.data)
    l1[1].weight.data.copy_(l2[1].weight.data)
    l1[0].bias.data.copy_(l2[0].bias.data)
    l1[1].bias.data.copy_(l2[1].bias.data)

    opt1 = bnb.optim.Adam32bit(l1.parameters(), lr=0.001)
    opt2 = bnb.optim.Adam32bit(l2.parameters(), lr=0.001)

    acc_steps = 10

    for i in range(15):
        b1 = torch.randn(16, 8, 32, device=device).half()
        o1 = l1(b1)
        o2 = l2(b1)
        loss1 = o1.mean()
        loss2 = o2.mean()
        loss1.backward()
        loss2.backward()
        if i == 2:
            assert l1[0].state.CB is not None
            assert l1[1].state.CB is not None

        if i > 0 and i % acc_steps == 0:
            opt1.step()
            opt1.zero_grad(True)
            opt2.step()
            opt2.zero_grad(True)
            assert_all_approx_close(l1[0].weight, l2[0].weight, rtol=1.05, atol=0.01, count=2)
            assert_all_approx_close(l1[1].weight, l2[1].weight, rtol=1.05, atol=0.01, count=2)
            # we do this copy because otherwise we have small divergences over time that add up
            l1[0].weight.data.copy_(l2[0].weight.data)
            l1[1].weight.data.copy_(l2[1].weight.data)
            l1[0].bias.data.copy_(l2[0].bias.data)
            l1[1].bias.data.copy_(l2[1].bias.data)
        else:
            assert_all_approx_close(l1[0].weight.grad, l2[0].weight.grad, rtol=1.05, atol=0.04, count=1)
            assert_all_approx_close(l1[1].weight.grad, l2[1].weight.grad, rtol=1.05, atol=0.04, count=1)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("threshold", [0.0, 2.0])
def test_linear8bitlt_no_fp16_weights(device, threshold):
    l1 = (
        bnb.nn.Linear8bitLt(
            32,
            64,
            threshold=threshold,
            has_fp16_weights=False,
        )
        .to(device)
        .half()
    )
    assert l1.weight.dtype == torch.int8

    l1.eval()
    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = l1(b1)
        assert o1.dtype == torch.float16

    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).to(device)
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8

    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None

    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).to(device).half()
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8

    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None

    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).half().to(device)

    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8

    mlp = (
        MLP8bit(
            32,
            64,
            threshold=threshold,
            has_fp16_weights=False,
        )
        .half()
        .to(device)
    )

    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    assert mlp.fc1.weight.device.type == device
    assert mlp.fc2.weight.device.type == device

    mlp = MLP8bit(
        32,
        64,
        threshold=threshold,
        has_fp16_weights=False,
    )
    w1, w2 = mlp.fc1.weight.clone().to(device), mlp.fc2.weight.clone().to(device)  # grab weights before quantization,
    mlp = mlp.to(device).half()  # and this line triggers quantization

    for i in range(4):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None

    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    assert mlp.fc1.weight.device.type == device
    assert mlp.fc2.weight.device.type == device

    b1 = torch.randn(16, 8, 32, device=device, requires_grad=True, dtype=torch.half)
    o1 = mlp(b1)
    assert o1.dtype == torch.float16
    assert o1.requires_grad
    grad_proj = torch.randn_like(o1)

    mlp.zero_grad()
    (o1 * grad_proj).sum().backward()
    grad_ref = grad_proj.flatten(2) @ w2.half() @ w1.half()
    scale = grad_ref.abs().mean()

    torch.testing.assert_close(b1.grad, grad_ref, rtol=0, atol=0.05 * scale)
    idx = torch.isclose(b1.grad, grad_ref, atol=0.01 * scale, rtol=0.1)
    assert (idx == 0).sum().item() <= b1.numel() * 0.005


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "module",
    [
        lambda n_in, n_out, bias=True: bnb.nn.Linear8bitLt(n_in, n_out, bias=bias, has_fp16_weights=False),
        bnb.nn.LinearNF4,
    ],
    ids=["Int8Lt", "NF4"],
)
def test_linear_kbit_fp32_bias(device, module):
    # casts model to fp16 -> int8 automatically
    l1 = module(32, 64).to(device)
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    assert l1.bias.dtype == torch.float32

    for i in range(100):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        # casts bias to fp32
        o1 = l1(b1)
        assert l1.bias.dtype == torch.float16

    # casts model to fp16 -> int8 automatically
    l1 = module(32, 64, bias=False).to(device)
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    assert l1.bias is None

    for i in range(100):
        b1 = torch.randn(16, 8, 32, device=device, dtype=torch.float16)
        o1 = l1(b1)
        assert l1.bias is None


module_dict = {
    "Int8Lt": bnb.nn.Linear8bitLt,
    "4bit": bnb.nn.Linear4bit,
    "FP4": bnb.nn.LinearFP4,
    "NF4": bnb.nn.LinearNF4,
    "FP4+C": lambda d1, d2: bnb.nn.LinearFP4(d1, d2, compress_statistics=True),
    "NF4+C": lambda d1, d2: bnb.nn.LinearNF4(d1, d2, compress_statistics=True),
    "NF4+fp32": lambda d1, d2: bnb.nn.LinearNF4(d1, d2, compute_dtype=torch.float32),
    "NF4+fp16": lambda d1, d2: bnb.nn.LinearNF4(d1, d2, compute_dtype=torch.float16),
    "NF4+bf16": lambda d1, d2: bnb.nn.LinearNF4(d1, d2, compute_dtype=torch.bfloat16),
}


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("module", module_dict.values(), ids=module_dict.keys())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kbit_backprop(device, module, dtype):
    b = 16
    dim1 = 36
    dim2 = 84
    # dim1 = 37
    # dim2 = 83

    ref = nn.Sequential(*[torch.nn.Linear(dim1, dim2), torch.nn.Linear(dim2, 128)])
    torch.nn.init.kaiming_normal_(ref[0].weight)
    torch.nn.init.kaiming_normal_(ref[1].weight)
    ref[1].weight.requires_grad_(False)

    kbit = nn.Sequential(*[torch.nn.Linear(dim1, dim2), module(dim2, 128)])

    if (
        device == "hpu"
        and isinstance(kbit[1], bnb.nn.Linear4bit)
        and not is_supported_on_hpu(kbit[1].weight.quant_type, dtype)
    ):
        pytest.skip("This configuration not supported on HPU")

    kbit[0].weight.detach().copy_(ref[0].weight)
    kbit[1].weight.detach().copy_(ref[1].weight)
    kbit[0].bias.detach().copy_(ref[0].bias)
    kbit[1].bias.detach().copy_(ref[1].bias)
    kbit[1].weight.requires_grad_(False)
    ref = ref.to(device=device, dtype=dtype)
    kbit = kbit.to(device=device, dtype=dtype)
    kbit = kbit.to(device=device, dtype=dtype)

    errs1 = []
    errs2 = []
    relerrs1 = []
    relerrs2 = []
    for i in range(100):
        batch = torch.randn(b, dim1, device=device, dtype=dtype)
        out1 = ref(batch)
        out2 = kbit(batch)
        out1.mean().backward()
        out2.mean().backward()

        grad1 = ref[0].weight.grad
        grad2 = kbit[0].weight.grad
        bgrad1 = ref[0].bias.grad
        bgrad2 = kbit[0].bias.grad

        err1 = (out1 - out2).abs().float()
        err2 = (grad1 - grad2).abs().float()
        relerr1 = err1 / (out1.abs().float() + 1e-9)
        relerr2 = err2 / (grad1.abs().float() + 1e-9)
        errs1.append(err1.mean().item())
        errs2.append(err2.mean().item())
        relerrs1.append(relerr1.mean().item())
        relerrs2.append(relerr2.mean().item())

        if isinstance(module, bnb.nn.Linear8bitLt):
            assert_all_approx_close(grad1, grad2, atol=0.008, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.008, rtol=0.05)
        else:
            assert_all_approx_close(grad1, grad2, atol=0.015, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.02, rtol=0.05)
        ref.zero_grad()
        kbit.zero_grad()

        assert kbit[0].weight.grad is None or kbit[0].weight.grad.sum().item() == 0
        assert kbit[0].weight.grad is None or kbit[0].bias.grad.sum().item() == 0


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("embedding_dim", [64, 65])
@pytest.mark.parametrize("input_shape", [(10,), (10, 10), (10, 10, 10)], ids=str)
@pytest.mark.parametrize(
    "embedding_class,quant_storage",
    [
        (bnb.nn.Embedding8bit, None),
        (bnb.nn.EmbeddingFP4, torch.uint8),
        (bnb.nn.EmbeddingFP4, torch.float32),
        (bnb.nn.EmbeddingNF4, torch.uint8),
        (bnb.nn.EmbeddingNF4, torch.float32),
    ],
    ids=lambda x: x.__name__ if inspect.isclass(x) else str(x),
)
def test_embedding_lossless(device, embedding_class, input_shape, embedding_dim, quant_storage):
    if device == "hpu":
        if embedding_class is bnb.nn.EmbeddingFP4:
            pytest.skip("FP4 is not supported on HPU")
        elif embedding_class is bnb.nn.EmbeddingNF4 and not is_supported_on_hpu("nf4", torch.float32, quant_storage):
            pytest.skip("This configuration is not supported on HPU")

    num_embeddings = 128

    src_weight = (torch.randn((num_embeddings, embedding_dim), dtype=torch.float32) > 0).to(
        torch.float32
    ) * 2 - 1  # Embeddings filled with {-1, 1} values. It should compress losslessly

    emb_base = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        _freeze=True,
        _weight=src_weight,
    )
    if embedding_class is bnb.nn.Embedding8bit:
        e = embedding_class(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    else:
        e = embedding_class(num_embeddings=num_embeddings, embedding_dim=embedding_dim, quant_storage=quant_storage)

    e.load_state_dict(emb_base.state_dict())

    emb_base.to(device)
    e.to(device)

    input_tokens = torch.randint(low=0, high=num_embeddings, size=input_shape, device=device)

    torch.testing.assert_close(
        actual=e(input_tokens),
        expected=emb_base(input_tokens),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("embedding_dim", [64, 65])
@pytest.mark.parametrize("input_shape", [(10,), (10, 10), (10, 10, 10)], ids=str)
@pytest.mark.parametrize(
    "embedding_class,quant_storage",
    [
        (bnb.nn.Embedding8bit, None),
        (bnb.nn.EmbeddingFP4, torch.uint8),
        (bnb.nn.EmbeddingFP4, torch.float32),
        (bnb.nn.EmbeddingNF4, torch.uint8),
        (bnb.nn.EmbeddingNF4, torch.float32),
    ],
    ids=lambda x: x.__name__ if inspect.isclass(x) else str(x),
)
def test_embedding_error(device, embedding_class, input_shape, embedding_dim, quant_storage):
    if device == "hpu":
        if embedding_class is bnb.nn.EmbeddingFP4:
            pytest.skip("FP4 is not supported on HPU")
        elif embedding_class is bnb.nn.EmbeddingNF4 and not is_supported_on_hpu("nf4", torch.float32, quant_storage):
            pytest.skip("This configuration is not supported on HPU")

    is_8bit = embedding_class is bnb.nn.Embedding8bit

    num_embeddings = 128

    src_weight = torch.rand((num_embeddings, embedding_dim), dtype=torch.float32)

    emb_base = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        _freeze=True,
        _weight=src_weight,
    )
    if is_8bit:
        e = embedding_class(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    else:
        e = embedding_class(num_embeddings=num_embeddings, embedding_dim=embedding_dim, quant_storage=quant_storage)

    e.load_state_dict(emb_base.state_dict())

    emb_base.to(device)
    e.to(device)

    input_tokens = torch.randint(low=0, high=num_embeddings, size=input_shape, device=device)

    torch.testing.assert_close(
        actual=e(input_tokens),
        expected=emb_base(input_tokens),
        atol=0.05 if is_8bit else 0.20,
        rtol=0.0,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_4bit_linear_warnings(device):
    dim1 = 64

    with pytest.warns(UserWarning, match=r"inference or training"):
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, quant_type="nf4") for i in range(10)])
        net = net.to(device)
        inp = torch.rand(10, dim1, device=device, dtype=torch.float16)
        net(inp)
    with pytest.warns(UserWarning, match=r"inference."):
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, quant_type="nf4") for i in range(10)])
        net = net.to(device)
        inp = torch.rand(1, dim1, device=device, dtype=torch.float16)
        net(inp)

    with pytest.warns(UserWarning) as record:
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, quant_type="nf4") for i in range(10)])
        net = net.to(device)
        inp = torch.rand(10, dim1, device=device, dtype=torch.float16)
        net(inp)

        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, quant_type="nf4") for i in range(10)])
        net = net.to(device)
        inp = torch.rand(1, dim1, device=device, dtype=torch.float16)
        net(inp)

    assert len(record) == 2


@pytest.mark.parametrize("device", get_available_devices())
def test_4bit_embedding_warnings(device):
    num_embeddings = 128
    default_block_size = 64

    with pytest.warns(UserWarning, match=r"inference."):
        net = bnb.nn.Embedding4bit(
            num_embeddings=num_embeddings, embedding_dim=default_block_size + 1, quant_type="nf4"
        )
        net.to(device)
        inp = torch.randint(low=0, high=num_embeddings, size=(1,), device=device)
        net(inp)


def test_4bit_embedding_weight_fsdp_fix(requires_cuda):
    num_embeddings = 64
    embedding_dim = 32

    module = bnb.nn.Embedding4bit(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    module.cuda()

    module.weight.quant_state = None

    input_tokens = torch.randint(low=0, high=num_embeddings, size=(1,), device="cuda")

    module(input_tokens)

    assert module.weight.quant_state is not None


def test_4bit_linear_weight_fsdp_fix(requires_cuda):
    inp_size = 64
    out_size = 32

    module = bnb.nn.Linear4bit(inp_size, out_size)

    module.cuda()

    module.weight.quant_state = None

    input_tensor = torch.randn((1, inp_size), device="cuda")

    module(input_tensor)

    assert module.weight.quant_state is not None


def test_embedding_not_implemented_error():
    with pytest.raises(NotImplementedError):
        emb = bnb.nn.Embedding4bit(32, 32)
        emb.state_dict()

    with pytest.raises(NotImplementedError):
        emb = bnb.nn.Embedding8bit(32, 32)
        emb.state_dict()
