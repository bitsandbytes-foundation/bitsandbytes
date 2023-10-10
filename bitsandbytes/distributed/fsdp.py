"""
Fully Sharded Data Parallelism (FSDP) is a technique used to scale deep learning models across multiple GPUs or nodes, enabling training of models that are too large to fit into the memory of a single GPU.

FSDP shards (splits) the model's parameters, gradients, and optimizer state across the data parallel workers. This reduces the memory consumption on each worker, enabling training of models that are too large to fit into the memory of a single node by sharding the model across multiple nodes.

## Requirements for Selective Wrapping:
Note: Look at the last commit, the FSDP test (`tests/test_optim.py`), how it wraps the model.

We need a mechanism that takes a PyTorch model (which can be a nested `torch.nn.Module`) and wrap the largest possible module graph sub-trees based on the below criteria:

- wrap as many layers into one block as possible to reduce the overhead of communication
- only layers with same data types + grad types can be wrapped together

- therefore layers with `requires_grad=False` CANNOT be wrapped with layers with `requires_grad=True`
- `Linear4bit` is considered such special type that can only be wrapped with the same type, `Linear4bit`
- mixed subtrees cannot be wrapped and can be ignored. Instead wrap the smallest wrappable sub-tree
- not all layers need to be wrapped. Layernorms are usually not faster when wrapped. A good rule of thumb is a layer needs to have at least 1M parameters to be worth wrapping
- bias or no bias grants no special considerations

## Custom Auto Wrap Policy:
Custom auto wrap policy function for determining whether to wrap a given module 
with Fully Sharded Data Parallel (FSDP) based on specific criteria.

This function is designed to be used as the `auto_wrap_policy` argument when 
initializing an FSDP wrapper. It follows the API expected by FSDP for auto-wrap 
policies and makes wrapping decisions, but it has a second type of boolean logic baked
in, which makes things confusing. The boolean return value has different meanings,
based on whether the `recurse` parameter is `True` or `False`, see below for further
details.

Parameters:
- module (nn.Module): The module being considered for wrapping.
- recurse (bool): A flag indicating whether the function is being called during 
    the traversal down the module tree. If `True`, the function will always continue 
    the traversal by returning `True`.
- nonwrapped_numel (int): The number of elements in the module that are not yet 
    wrapped. This parameter is not used in the current implementation but is included 
    to fulfill the expected API.
    
Returns:
- bool: A boolean value indicating either (1) whether recursion should continue, if
    called with `recurse=True` or (2) whether the given module should be wrapped, if
    called with `recurse=False`.
    

How the recursion works:
The FSDP wrapper traverses the module tree, starting from the root module, and 
calls this function for each module encountered. The `recurse` parameter indicates 
whether the current call is part of the traversal down the tree. If `recurse` is 
`True`, the function returns `True` to continue the traversal. When reaching a leaf 
module, `recurse` is `False`, and the function makes a decision based on the specific 
criteria whether to wrap the module. This way, the function is recursively called for 
each module in the tree, allowing selective wrapping of the modules. Therefore, the
leaves are wrapped first, and the wrapping propagates up the tree.
"""

from typing import Union, OrderedDict, Iterable, Optional, Set, Type, Dict, Tuple
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.nn import Linear4bit

import functools

from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy


def parameters_all_consistent(params: Iterable[torch.nn.Parameter]) -> bool:
    """
    Check if all parameters in the iterable have the same dtype and requires_grad attributes.
    
    Parameters:
    - params (Iterable[torch.nn.Parameter]): An iterable of PyTorch parameters.
    
    Returns:
    - bool: True if all parameters are consistent, False otherwise.
    """
    params_iter = iter(params)

    try:
        first_param = next(params_iter)
    except StopIteration:
        return True  # No parameters to check

    return all(
        param.dtype == first_param.dtype
        and param.requires_grad == first_param.requires_grad for param in params_iter)


# TODO: still getting "ValueError: Must flatten tensors with uniform `requires_grad`
# when `use_orig_params=False`" when running `pytest tests/test_fsdp.py::test_fsdp_bnb`
# something must still be off with the custom auto wrap policy...
def bnb_fsdp_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    debug: bool = True,
    *args,
    **kwargs,
) -> bool:
    """See the module doc string, section "Custom Auto Wrap Policy" for details..."""
    if debug:  # TODO: remove this and the extraneous comments once this is working
        module_type = type(module).__name__
        print(f"{module_type = }")
        if args: print(f'{args = }')  #noqa: E701
        if kwargs: print(f'{kwargs = }')  #noqa: E701
        print(f'{recurse = }')
        print(f'{nonwrapped_numel = }')
        print(f'{parameters_all_consistent(module.parameters()) = }')

    if recurse:
        # return True to recurse: we recurse until we hit a module w/ consistent params
        return not parameters_all_consistent(module.parameters())
    # if we're not recursing, we're evaluating if the module should be wrapped and
    # therefore return True if the module has consistent params, as we're trying to
    # wrap the largest possible module graph sub-trees based on this criterium
    return parameters_all_consistent(module.parameters())


# TODO: this example policy will be removed later, we still need to integrate the
# min_num_params mechanism, either directly into the bnb_custom_auto_wrap_policy or
# by using the FSDP `_or_policy`.
def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Set[Type[nn.Module]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Set[Type[nn.Module]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None else force_leaf_modules)
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None else exclude_wrap_modules)

    # Keep the argument `min_num_params` for BC for now, but it represents the
    # minimum non-wrapped *numel* before triggering a wrapping
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    if recurse:
        # We should recurse if the module is big enough but not in force_leaf_modules list.
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # If we are not recursing, determine if we should wrap.
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {
    nn.ModuleList, nn.ModuleDict
}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {
    nn.MultiheadAttention
}  # type: ignore[attr-defined]
