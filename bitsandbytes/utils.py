import shlex
import subprocess
import torch
from typing import Tuple

def outlier_hook(module, input):
    assert isinstance(module, torch.nn.Linear)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)
    if hvalue not in tracer.hvalue2outlier_idx:
        outlier_idx = find_outlier_dims(module.weight)
        tracer.outliers.append(outlier_idx)
        tracer.hvalues.append(hvalue)
        if len(tracer.outliers) > 1:
            # assign the current layer the outlier idx found from the weight
            # of the previous linear layer
            if tracer.outliers[-1].numel() > 0:
                assert tracer.outliers[-1].max() < module.weight.shape[1]
            tracer.hvalue2outlier_idx[hvalue] = tracer.outliers[-1]

        else:
            # first layer, we cannot use the weight for outlier detection
            # we follow a mixed approach:
            # (1) zscore test of std of hidden dimension
            # (2) magnitude > 6 test
            merged = input[0].view(-1, input[0].shape[-1])
            # (1) zscore test of std of hidden dimension
            outlier_idx = find_outlier_dims(merged, reduction_dim=1, zscore=3)
            # (2) magnitude > 6 test
            dims = (torch.abs(input[0])> 6).sum(dim=list(range(len(input[0].shape)-1)))
            outlier_idx2 = torch.where(dims > 0)[0]
            outlier_idx = torch.cat([outlier_idx, outlier_idx2]).unique()
            tracer.hvalue2outlier_idx[hvalue] = outlier_idx
    else:
        for hook in tracer.hooks:
            hook.remove()


class OutlierTracer(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self, model):
        self.last_w = None
        self.current_outlier_dims = None
        self.hvalues = []
        self.outliers = []
        self.hvalue2outlier_idx = {}
        self.initialized = True
        self.hooks = []

        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(outlier_hook))

    def is_initialized(self):
        return getattr(self, 'initialized', False)

    def get_hvalue(self, weight):
        return weight.data.storage().data_ptr()

    def get_outliers(self, weight):
        if not self.is_initialized():
            print('Outlier tracer is not initialized...')
            return None
        hvalue = self.get_hvalue(weight)
        if hvalue in self.hvalue2outlier_idx:
            return self.hvalue2outlier_idx[hvalue]
        else:
            return None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

def find_outlier_dims(weight, reduction_dim=0, zscore=4.0, topk=None, rdm=False):
    if rdm:
        return torch.randint(0, weight.shape[1], size=(topk,), device=weight.device).long()

    m = weight.mean(reduction_dim)
    mm = m.mean()
    mstd = m.std()
    zm = (m-mm)/mstd

    std = weight.std(reduction_dim)
    stdm = std.mean()
    stdstd = std.std()

    zstd = (std-stdm)/stdstd

    if topk is not None:
        val, idx = torch.topk(std.abs(), k=topk, dim=0)
    else:
        idx = torch.where(zstd > zscore)[0]

    return idx

def replace_linear(model, linear_replacement, skip_modules=["lm_head"], copy_weights=False, post_processing_function=None):
    """
    Replace linear modules with a new Linear module.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
        copy_weights (`bool`):
            Copy the weights from the old linear module to the new one
        post_processing_fun_name (`str`):
            A function name of the replacement linear class that is called
            after processing.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights, post_processing_function)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight = old_module.weight
                model._modules[name].bias = old_module.bias

            if post_processing_function is not None:
               func = getattr(module, post_processing_function, None)
               if func is not None: func(module)
    return model



def execute_and_return(command_string: str) -> Tuple[str, str]:
    def _decode(subprocess_err_out_tuple):
        return tuple(
            to_decode.decode("UTF-8").strip()
            for to_decode in subprocess_err_out_tuple
        )

    def execute_and_return_decoded_std_streams(command_string):
        return _decode(
            subprocess.Popen(
                shlex.split(command_string),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
        )

    std_out, std_err = execute_and_return_decoded_std_streams(command_string)
    return std_out, std_err



def replace_linear(model, linear_replacement, skip_modules=["lm_head"], copy_weights=False, post_processing_function=None):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
        copy_weights (`bool`):
            Copy the weights from the old linear module to the new one
        post_processing_fun_name (`str`):
            A function name of the replacement linear class that is called
            after processing.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights, post_processing_function)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight = old_module.weight
                model._modules[name].bias = old_module.bias

            if post_processing_function is not None:
               func = getattr(module, post_processing_function, None)
               if func is not None: func(module)
    return model

