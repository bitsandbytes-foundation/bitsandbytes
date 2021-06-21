import torch
from torch.optim import Optimizer
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from torch._six import container_abcs

class MockArgs(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


class GlobalOptimManager(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group['params']):
                if id(p) in self.pid2config:
                    self.index2config[(group_index, p_index)] = self.pid2config[id(p)]

    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
        '''
        Overrides initial optimizer config for specific parameters.

        The key-values of the optimizer config for the input parameters are overidden
        This can be both, optimizer parameters like "betas", or "lr" or it can be
        8-bit specific paramters like "is_sparse", "optim_bits", "percentile_clipping".

        Parameters
        ----------
        parameters : torch.Tensor or list(torch.Tensors)
            The input parameters.
        key : str
            The hyperparamter to override.
        value : object
            The value for the hyperparamters.
        key_value_dict : dict
            A dictionary with multiple key-values to override.
        '''
        self.uses_config_override = True
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}

        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:self.pid2config[id(p)].update(key_value_dict)
                else: self.pid2config[id(p)] = key_value_dict


class Optimizer8bit(Optimizer):

    def __init__(self, params, defaults):
        super(Optimizer8bit, self).__init__(params, defaults)
        self.mng = GlobalOptimManager.get_instance()
        self.non_castable_tensor_keys = set(
                ['qmap1', 'qmap2',
                 'max1', 'max2',
                 'new_max1', 'new_max2',
                 'state1', 'state2',
                 'gnorm_vec'])

    def __setstate__(self, state):
        super(Optimizer8bit, self).__setstate__(state)

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                return value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        value[k] = v.to(param.device)
                    else:
                        value[k] = cast(param, v)
                return value
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})




