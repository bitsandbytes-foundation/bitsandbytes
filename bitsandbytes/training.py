"""Training utilities for kbit QLoRA.

Provides gradient checkpointing with CPU offload for reducing GPU memory
during QLoRA fine-tuning.
"""

from typing import Any

import torch


class _CPUOffloadCheckpointFunction(torch.autograd.Function):
    """Gradient checkpoint that offloads activations to CPU during forward.

    Forward: copies activations to CPU asynchronously, frees GPU copy.
    Backward: copies activations back from CPU, recomputes the forward pass.

    This saves GPU memory at the cost of CPU→GPU bandwidth during backward.
    Non-blocking transfers overlap with GPU compute when possible.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # Save RNG state if requested
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.random.get_rng_state()
            ctx.had_cuda = torch.cuda._initialized
            if ctx.had_cuda:
                ctx.fwd_gpu_state = torch.cuda.get_rng_state()

        # Save inputs to CPU (async)
        ctx.cpu_inputs = []
        ctx.input_requires_grad = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                ctx.input_requires_grad.append(arg.requires_grad)
                # Async copy to CPU, pin memory for faster D2H transfer
                cpu_tensor = torch.empty(
                    arg.shape, dtype=arg.dtype, device="cpu", pin_memory=True,
                )
                cpu_tensor.copy_(arg, non_blocking=True)
                ctx.cpu_inputs.append(cpu_tensor)
            else:
                ctx.input_requires_grad.append(None)
                ctx.cpu_inputs.append(arg)

        # Run the function
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Restore inputs from CPU (async)
        inputs = []
        for cpu_input, req_grad in zip(ctx.cpu_inputs, ctx.input_requires_grad):
            if isinstance(cpu_input, torch.Tensor):
                # Async copy back to GPU
                gpu_tensor = cpu_input.to("cuda", non_blocking=True)
                if req_grad:
                    gpu_tensor.requires_grad_(True)
                inputs.append(gpu_tensor)
            else:
                inputs.append(cpu_input)

        # Synchronize to ensure transfers are complete
        torch.cuda.current_stream().synchronize()

        # Restore RNG state and recompute forward
        if ctx.preserve_rng_state:
            rng_devices = []
            if ctx.had_cuda:
                rng_devices.append("cuda")
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                torch.random.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda:
                    torch.cuda.set_rng_state(ctx.fwd_gpu_state)
                with torch.enable_grad():
                    outputs = ctx.run_function(*inputs)
        else:
            with torch.enable_grad():
                outputs = ctx.run_function(*inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Use backward() to accumulate gradients into all leaf parameters
        # (not just inputs). This is needed when the checkpointed function
        # is an nn.Module with trainable parameters.
        torch.autograd.backward(outputs, grad_outputs)

        # Collect input gradients
        result = [None, None]  # for run_function and preserve_rng_state
        for inp, req_grad in zip(inputs, ctx.input_requires_grad):
            if isinstance(inp, torch.Tensor) and req_grad:
                result.append(inp.grad if inp.grad is not None else torch.zeros_like(inp))
            else:
                result.append(None)

        # Free CPU copies
        ctx.cpu_inputs = None

        return tuple(result)


def checkpoint_cpu_offload(
    function: Any,
    *args: Any,
    preserve_rng_state: bool = True,
) -> Any:
    """Gradient checkpoint with CPU offload.

    Like ``torch.utils.checkpoint.checkpoint`` but offloads saved activations
    to CPU during forward to reduce GPU memory.  Activations are copied back
    from CPU asynchronously during backward.

    Args:
        function: The function to checkpoint.
        *args: Arguments to the function. Tensors will be offloaded.
        preserve_rng_state: Preserve and restore RNG state during recompute.

    Returns:
        Output of the function.
    """
    return _CPUOffloadCheckpointFunction.apply(function, preserve_rng_state, *args)
