import shlex
import subprocess
from typing import Tuple


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

