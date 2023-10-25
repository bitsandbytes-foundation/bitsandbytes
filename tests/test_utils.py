import pytest
import torch
from torch.nn import Linear, Module

from bitsandbytes.nn import Linear4bit
from bitsandbytes.utils import replace_linear


class TwoLayerFCModel(Module):

    def __init__(self):
        super().__init__()
        self.fully_connected_1 = Linear(10, 50)
        self.fully_connected_2 = Linear(50, 1)

    def forward(self, x):
        x = self.fully_connected_1(x)
        return self.fully_connected_2(x)


replace_linear_test_params = [
    (Linear, True),
    (Linear, False),
    (Linear4bit, True),
    (Linear4bit, False),
]


@pytest.mark.parametrize("replacement_layer, copy_weights", replace_linear_test_params)
def test_replace_linear_replacement(replacement_layer, copy_weights):
    model = TwoLayerFCModel()
    new_model = replace_linear(model, replacement_layer, copy_weights=copy_weights)

    for name, module in new_model.named_children():
        assert isinstance(
            module, replacement_layer), f"Replacement failed for module: {name}"

        if copy_weights:
            old_module = getattr(model, name)

            assert torch.equal(
                module.weight.data, old_module.weight.data
            ), f"Weights not copied correctly for module: {name}"

            if module.bias is not None and old_module.bias is not None:
                assert torch.equal(
                    module.bias.data, old_module.bias.data
                ), f"Bias not copied correctly for module: {name}"


def test_replace_linear_skip_module():
    model = TwoLayerFCModel()
    replacement_layer = Linear4bit

    new_model = replace_linear(
        model, replacement_layer, skip_modules=['fully_connected_1'])

    assert type(
        new_model.fully_connected_1
    ) is Linear, "Module 'fully_connected_1' should NOT be replaced"

    assert type(
        new_model.fully_connected_2
    ) is replacement_layer, "Module 'fully_connected_2' SHOULD be replaced"
