import torch
from torch import nn

from bitsandbytes.utils import replace_linear


class _TrackingLinear(nn.Linear):
    """Stands in for a quantized replacement that needs a step after construction."""

    post_init_calls: list[str] = []

    def post_init(self):
        _TrackingLinear.post_init_calls.append(f"{self.in_features}x{self.out_features}")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.block = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.lm_head = nn.Linear(16, 4)


def test_replace_linear_runs_the_post_processing_hook():
    """`post_processing_function` is documented as a method of the replacement class.

    It used to be looked up on the module being replaced, which is a plain `nn.Linear` and never
    carries it, so `getattr(..., None)` returned `None` and the hook silently never ran.
    """
    _TrackingLinear.post_init_calls.clear()
    model = _Model()

    replace_linear(model, _TrackingLinear, post_processing_function="post_init")

    assert isinstance(model.fc, _TrackingLinear)
    assert isinstance(model.block[0], _TrackingLinear)
    assert not isinstance(model.lm_head, _TrackingLinear)  # skip_modules default
    assert _TrackingLinear.post_init_calls == ["4x8", "8x16"]


def test_replace_linear_without_hook_is_unchanged():
    _TrackingLinear.post_init_calls.clear()
    model = _Model()

    replace_linear(model, _TrackingLinear)

    assert isinstance(model.fc, _TrackingLinear)
    assert _TrackingLinear.post_init_calls == []


def test_replace_linear_tolerates_a_replacement_without_the_hook():
    """A replacement class that does not define the method must not raise."""
    model = _Model()

    replace_linear(model, nn.Linear, post_processing_function="post_init")

    assert isinstance(model.fc, nn.Linear)


def test_replace_linear_copy_weights():
    model = _Model()
    original = model.fc.weight.detach().clone()

    replace_linear(model, _TrackingLinear, copy_weights=True)

    assert torch.equal(model.fc.weight.detach(), original)
