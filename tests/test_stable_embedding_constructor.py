import copy

import pytest
import torch

from bitsandbytes.nn import StableEmbedding


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_stable_embedding_factory_kwargs_train(device, dtype):
    torch.manual_seed(41)
    embedding = StableEmbedding(19, 8, padding_idx=0, device=device, dtype=dtype)
    expected_dtype = dtype or torch.get_default_dtype()
    assert embedding.weight.device.type == "cpu"
    assert embedding.weight.dtype == expected_dtype
    assert embedding.weight.requires_grad
    assert embedding.norm.weight.device == embedding.weight.device
    assert embedding.norm.weight.dtype == torch.get_default_dtype()

    reference = torch.nn.Embedding(19, 8, padding_idx=0, device=device, dtype=dtype)
    reference.load_state_dict({"weight": embedding.weight.detach().clone()})
    norm = copy.deepcopy(embedding.norm)
    head = torch.nn.Linear(8, 3, dtype=expected_dtype)
    reference_head = copy.deepcopy(head)
    parameters = list(embedding.parameters()) + list(head.parameters())
    reference_parameters = list(reference.parameters()) + list(norm.parameters()) + list(reference_head.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.04, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference_parameters, lr=0.04, momentum=0.9)
    initial_weight = embedding.weight.detach().clone()
    tokens = torch.tensor([[0, 1, 2, 1], [3, 4, 0, 3]])
    targets = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])

    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual = head(embedding(tokens))
        expected = reference_head(norm(reference(tokens).to(norm.weight.dtype)).to(expected_dtype))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(actual.float().reshape(-1, 3), targets.reshape(-1)).backward()
        torch.nn.functional.cross_entropy(expected.float().reshape(-1, 3), targets.reshape(-1)).backward()
        for parameter, reference_parameter in zip(parameters, reference_parameters):
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
        assert torch.count_nonzero(embedding.weight.grad[0]) == 0
        optimizer.step()
        reference_optimizer.step()
        for parameter, reference_parameter in zip(parameters, reference_parameters):
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                reference_optimizer.state[reference_parameter]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
    assert not torch.equal(embedding.weight[1:], initial_weight[1:])
    assert torch.count_nonzero(embedding.weight[0]) == 0


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.bfloat16])
def test_stable_embedding_factory_kwargs_meta(dtype):
    embedding = StableEmbedding(19, 8, device="meta", dtype=dtype)
    assert embedding.weight.device.type == "meta"
    assert embedding.norm.weight.device.type == "meta"
    assert embedding.weight.dtype == (dtype or torch.get_default_dtype())
    assert embedding.weight.requires_grad


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_stable_embedding_pretrained_weight_remains_trainable(device):
    weight = torch.arange(19 * 8, dtype=torch.float64).reshape(19, 8) / 100
    embedding = StableEmbedding(19, 8, _weight=weight.clone(), device=device, dtype=torch.float32)
    reference = torch.nn.Embedding(19, 8, _weight=weight.clone(), device=device, dtype=torch.float32)
    # Factory dtype does not recast an explicitly supplied pretrained weight.
    assert embedding.weight.dtype == reference.weight.dtype == torch.float64
    assert embedding.weight.requires_grad == reference.weight.requires_grad
    embedding(torch.tensor([1, 2, 3])).square().sum().backward()
    assert embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.weight.grad) > 0


@pytest.mark.parametrize(
    "options",
    [
        {"max_norm": 0.5, "norm_type": 1.0},
        {"scale_grad_by_freq": True},
        {"sparse": True},
    ],
)
def test_stable_embedding_constructor_preserves_embedding_options(options):
    torch.manual_seed(17)
    embedding = StableEmbedding(19, 8, padding_idx=-1, device="cpu", dtype=torch.float32, **options)
    reference = torch.nn.Embedding(19, 8, padding_idx=-1, device="cpu", dtype=torch.float32, **options)
    reference.load_state_dict({"weight": embedding.weight.detach().clone()})
    norm = copy.deepcopy(embedding.norm)
    tokens = torch.tensor([1, 1, 2, 18])
    actual = embedding(tokens)
    expected = norm(reference(tokens))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficients = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(embedding.weight, reference.weight, rtol=0, atol=0)
    actual_gradient = embedding.weight.grad
    expected_gradient = reference.weight.grad
    if options.get("sparse"):
        assert actual_gradient.is_sparse
        actual_gradient = actual_gradient.to_dense()
        expected_gradient = expected_gradient.to_dense()
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
