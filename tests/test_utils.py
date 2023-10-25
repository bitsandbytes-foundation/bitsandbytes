from copy import deepcopy

import pytest
import torch
from torch.nn import BatchNorm1d, Linear, Module, ReLU

from bitsandbytes.nn import Linear4bit
from bitsandbytes.utils import replace_linear


class NoLinearModel(Module):

    def __init__(self):
        super().__init__()
        self.layer1 = ReLU()


class TwoLayerFCModel(Module):

    def __init__(self):
        super().__init__()
        self.fully_connected_1 = Linear(10, 50)
        self.relu = ReLU()
        self.fully_connected_2 = Linear(50, 1)

    def forward(self, x):
        x = self.fully_connected_1(x)
        x = self.relu(x)
        return self.fully_connected_2(x)


class NestedModel(Module):

    def __init__(self):
        super().__init__()
        self.layer1 = Linear(10, 50)
        self.batch_norm = BatchNorm1d(50)
        self.submodel = TwoLayerFCModel()

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm(x)
        return self.submodel(x)


def test_replace_all_linear():
    new_model = replace_linear(TwoLayerFCModel(), Linear4bit)

    assert type(new_model.fully_connected_1) is Linear4bit
    assert type(new_model.relu) is ReLU  # Ensure non-Linear layer remains unchanged
    assert type(new_model.fully_connected_2) is Linear4bit


def test_skip_layer():
    new_model = replace_linear(
        TwoLayerFCModel(), Linear4bit, skip_modules=['fully_connected_1'])

    assert type(new_model.fully_connected_1) is Linear
    assert type(new_model.relu) is ReLU  # Ensure non-Linear layer remains unchanged
    assert type(new_model.fully_connected_2) is Linear4bit


@pytest.mark.parametrize("copy_weights", [False, True])
def test_copy_weights(copy_weights: bool):
    old_model = TwoLayerFCModel()
    new_model = replace_linear(
        deepcopy(old_model), Linear4bit, copy_weights=copy_weights)

    for layer_name in ['fully_connected_1', 'fully_connected_2']:
        old_weight = getattr(old_model, layer_name).weight.data
        new_weight = getattr(new_model, layer_name).weight.data

        assert torch.equal(old_weight, new_weight) == copy_weights


def test_nested_modules():
    new_model = replace_linear(NestedModel(), Linear4bit)

    assert type(new_model.layer1) is Linear4bit
    # Ensure non-Linear layer remains unchanged
    assert type(new_model.batch_norm) is BatchNorm1d

    for _name, module in new_model.submodel.named_children():
        if isinstance(module, Linear):
            assert type(module) is Linear4bit
        else:
            # Ensure non-Linear layers remain unchanged
            assert type(module) is not Linear4bit


def test_no_replacements():
    model = replace_linear(NoLinearModel(), Linear4bit)

    assert type(model.layer1) is ReLU
    assert sum(1 for _ in model.children()) == 1
    assert sum(1 for _ in model.layer1.children()) == 0
