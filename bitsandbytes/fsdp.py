"""
Fully Sharded Data Parallelism (FSDP) is a technique used to scale deep learning models across multiple GPUs or nodes, enabling training of models that are too large to fit into the memory of a single GPU.

FSDP shards (splits) the model’s parameters, gradients, and optimizer state across the data parallel workers. This reduces the memory consumption on each worker, enabling training of larger models.

## Selectively Wrapping Data Types:

1. **Why Selective Wrapping:**
   In the context of FSDP, selectively wrapping each data type refers to choosing which layers or parts of the model should be sharded and which should be kept replicated across all GPUs. This decision can significantly affect the memory consumption, communication overhead, and ultimately, the efficiency of the parallel training process.

2. **Balancing Communication and Computation:**
   When a model is sharded, there is a need for communication between GPUs to synchronize gradients and parameters. However, not all layers are equal in terms of computational cost and communication cost. For layers with a high computation-to-communication ratio, sharding might be beneficial, as it reduces the memory footprint without introducing significant overhead. For layers with a low computation-to-communication ratio, the communication overhead might outweigh the benefits of sharding.

3. **Handling Different Data Types:**
   Deep learning models can have various data types, such as float parameters for weights, integer parameters for embeddings, etc. The choice of which data types or layers to shard can be influenced by:
   - **Memory Consumption:** Some layers or data types might consume more memory than others. For example, embedding layers with large vocabularies can be particularly memory-intensive.
   - **Update Frequency:** Layers that are updated less frequently might benefit less from sharding, as the communication overhead might dominate.
   - **Layer Type:** Batch normalization layers, for instance, require synchronization of running statistics, making them less suitable for sharding.

4. **Granularity of Sharding:**
   The granularity at which sharding is applied can also be an important consideration. Sharding at a finer granularity (e.g., per layer) can lead to better memory efficiency but might increase communication overhead. On the other hand, sharding at a coarser granularity (e.g., per module) might reduce communication overhead but be less memory-efficient.

5. **Strategies for Selective Wrapping:**
   - **Profile-Based:** By profiling the model’s memory consumption and communication requirements, one can make informed decisions on which layers to shard.
   - **Heuristic-Based:** Applying heuristics based on layer types, data types, and model architecture can also guide the selective wrapping process.

6. **Implementation Details:**
   In FSDP, selectively wrapping a layer can be done by wrapping the corresponding module with the `FullyShardedDataParallel` wrapper.

   Layers that are not wrapped with FSDP will remain replicated across all GPUs.

## Requirements for Selective Wrapping:
Note: Look at the last commit, the FSDP test (`tests/test_optim.py`), how it wraps the model.

We need a mechanism that takes a PyTorch model (which can be a nested `torch.nn.Module`) and wrap its layers differently based on two criteria:

1. **For `Linear4bit` Layers:** The layers that are represented with 4-bit precision should be wrapped separately, and these layers should have `requires_grad` set to `False`. This implies that these 4-bit layers are not going to be updated during training (i.e., they are frozen).

2. **For Different Data Types:** The layers should also be wrapped separately based on their data types, with the exception of layers with `dtype` equal to `torch.float64`. This suggests that the wrapping should be done mainly for layers with trainable parameters (`requires_grad=True`).
"""
import os
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from pathlib import Path

from bitsandbytes.nn import Linear4bit


def is_four_bit_frozen_layer(module: nn.Module) -> bool:
    """Check if the module is a 4-bit frozen layer."""
    return isinstance(module, Linear4bit) and all(
        param.requires_grad is False for param in module.parameters())


def should_wrap_based_on_dtype(module: nn.Module) -> bool:
    """Check if the module should be wrapped based on data type and requires_grad."""
    return any(
        param.dtype != torch.float64 and param.requires_grad
        for param in module.parameters())


def is_leaf_module(module: nn.Module) -> bool:
    """Check if the module is a leaf module."""
    return len(list(module.children())) == 0


def custom_auto_wrap_policy(
        module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    """
    Custom auto wrap policy function for determining whether to wrap a given module 
    with Fully Sharded Data Parallel (FSDP) based on specific criteria.
    
    This function is designed to be used as the `auto_wrap_policy` argument when 
    initializing an FSDP wrapper. It follows the API expected by FSDP for auto-wrap 
    policies and makes wrapping decisions based on whether the module is a 4-bit frozen 
    layer or if it should be wrapped based on its data type.
    
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
    if recurse:
        return True  # always continue the traversal

    if not is_leaf_module(module):
        return False

    return is_four_bit_frozen_layer(module) or should_wrap_based_on_dtype(module)


def distributed_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

    model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
    )
    torch.manual_seed(1337)

    FSDP(
        deepcopy(model_4bit),
        cpu_offload=CPUOffload(offload_params=True),
        auto_wrap_policy=custom_auto_wrap_policy,
    )


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    # Launch a separate process for each GPU
    mp.spawn(distributed_worker, args=(world_size, ), nprocs=world_size, join=True)

# if __name__ == "__main__":
#     """
#     NOTE: make any tiny model from scratch like this:
#     https://huggingface.co/stas/tiny-wmt19-en-de/blob/main/fsmt-make-tiny-model.py
#     or search for "tiny" in the HuggingFace model hub.
#     """
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from copy import deepcopy
#     from pathlib import Path
#     import os
#     import torch.distributed as dist

#     model_name = "HuggingFaceH4/tiny-random-LlamaForCausalLM"

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model_4bit = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         load_in_4bit=True,
#         # max_memory={0: "600MB", 1: "1GB"}, # for multi-gpu, using accelerate
#     )
#
#     if False:
#         models_dir = Path.home() / 'src/models'
#         tiny_random_dir = models_dir / "tiny-random-LlamaForCausalLM"
#         tiny_random_dir.mkdir(parents=True, exist_ok=True)

#         model_4bit.to("cpu")
#         model_4bit.save_pretrained(tiny_random_dir)
#         tokenizer.save_pretrained(tiny_random_dir)
#         model_4bit.push_to_hub("tiny-random-LlamaForCausalLM")
#         tokenizer.push_to_hub("tiny-random-LlamaForCausalLM")
