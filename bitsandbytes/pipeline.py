"""1F1B Pipeline Parallelism Engine.

Custom implementation of one-forward-one-backward pipeline schedule for
training large models across multiple stages. Each stage processes a
subset of model layers.

Supports:
- Single-process mode: all stages on one GPU (for testing)
- Multi-process NCCL mode: one stage per GPU, activation transfer via isend/irecv

The 1F1B schedule minimizes peak activation memory by interleaving forward
and backward passes, keeping at most (num_stages) micro-batches in flight.
"""

import torch
import torch.nn as nn


def generate_1f1b_schedule(num_stages, num_micro_batches):
    """Generate the 1F1B (one-forward-one-backward) pipeline schedule.

    The schedule for each stage consists of:
    1. Warmup phase: (num_stages - 1 - stage_id) forward passes
    2. Steady state: alternating backward+forward (non-last) or forward+backward (last)
    3. Cooldown phase: remaining backward passes

    Args:
        num_stages: Number of pipeline stages.
        num_micro_batches: Number of micro-batches per training step.
            Must be >= num_stages.

    Returns:
        List of lists: schedule[stage_id] = [(op, micro_batch_id), ...]
        where op is 'F' (forward) or 'B' (backward).
    """
    assert num_micro_batches >= num_stages, (
        f"Need at least {num_stages} micro-batches for {num_stages} stages, "
        f"got {num_micro_batches}"
    )

    S = num_stages
    M = num_micro_batches
    schedules = [[] for _ in range(S)]

    for s in range(S):
        warmup_forwards = S - 1 - s
        is_last_stage = s == S - 1

        # Warmup: forward-only passes to fill the pipeline
        for m in range(warmup_forwards):
            schedules[s].append(("F", m))

        # Steady state: interleave F and B
        f_idx = warmup_forwards
        b_idx = 0
        num_steady = M - warmup_forwards

        for _ in range(num_steady):
            if is_last_stage:
                # Last stage: F then B (must receive activation before backward)
                schedules[s].append(("F", f_idx))
                f_idx += 1
                schedules[s].append(("B", b_idx))
                b_idx += 1
            else:
                # Non-last stages: B then F (drain before filling)
                schedules[s].append(("B", b_idx))
                b_idx += 1
                schedules[s].append(("F", f_idx))
                f_idx += 1

        # Cooldown: remaining backward passes
        while b_idx < M:
            schedules[s].append(("B", b_idx))
            b_idx += 1

    return schedules


class PipelineEngine:
    """1F1B pipeline parallelism engine.

    Splits a model into pipeline stages and executes them using the 1F1B
    schedule. Supports single-process mode for testing and multi-process
    NCCL mode for multi-GPU training.

    The model must be provided as a list of stage callables. Each stage
    takes a hidden state tensor and returns the next hidden state.
    The last stage should include the loss computation.

    Args:
        stage_modules: List of nn.Module instances, one per stage.
            Each module's forward takes (hidden_states,) and returns hidden_states.
        loss_fn: Loss function taking (last_stage_output, labels) -> scalar loss.
            Used only at the last stage. If None, the last stage must return the loss.
        num_micro_batches: Number of micro-batches per training step.
        device: Device for all stages (single-process mode).
    """

    def __init__(
        self,
        stage_modules: list[nn.Module],
        loss_fn=None,
        num_micro_batches: int = 4,
        device: torch.device = None,
    ):
        self.stage_modules = stage_modules
        self.loss_fn = loss_fn
        self.num_stages = len(stage_modules)
        self.num_micro_batches = num_micro_batches
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.schedule = generate_1f1b_schedule(self.num_stages, num_micro_batches)

    def step(self, micro_batch_inputs, micro_batch_labels=None):
        """Run one training step with 1F1B schedule (single-process mode).

        Executes all stages sequentially in a single process, following the
        1F1B schedule for correct ordering. At each schedule index:
        - Forward operations are processed left-to-right (stage 0 first)
        - Backward operations are processed right-to-left (last stage first)

        This respects data dependencies: forward outputs flow left-to-right,
        backward gradients flow right-to-left.

        Args:
            micro_batch_inputs: List of M input tensors, one per micro-batch.
            micro_batch_labels: List of M label tensors. Required if loss_fn is set.

        Returns:
            dict with:
                loss: Average loss across micro-batches (float).
                losses: List of per-micro-batch losses.
        """
        S = self.num_stages
        M = self.num_micro_batches

        assert len(micro_batch_inputs) == M, (
            f"Expected {M} micro-batch inputs, got {len(micro_batch_inputs)}"
        )

        # Storage for intermediate activations
        # fwd_inputs[s][m] = input tensor to stage s for micro-batch m (requires_grad)
        # fwd_outputs[s][m] = output tensor from stage s for micro-batch m
        fwd_inputs = [[None] * M for _ in range(S)]
        fwd_outputs = [[None] * M for _ in range(S)]
        losses = [None] * M
        grad_inputs = [[None] * M for _ in range(S)]  # gradients from backward

        # Execute the 1F1B schedule with proper dependency ordering
        max_ops = max(len(sched) for sched in self.schedule)

        for op_idx in range(max_ops):
            # Collect operations at this schedule index
            forward_ops = []
            backward_ops = []
            for s in range(S):
                if op_idx >= len(self.schedule[s]):
                    continue
                op, m = self.schedule[s][op_idx]
                if op == "F":
                    forward_ops.append((s, m))
                else:
                    backward_ops.append((s, m))

            # Process forward operations left-to-right (stage 0 first)
            for s, m in sorted(forward_ops, key=lambda x: x[0]):
                self._forward_step(s, m, micro_batch_inputs, micro_batch_labels,
                                   fwd_inputs, fwd_outputs, losses)

            # Process backward operations right-to-left (last stage first)
            for s, m in sorted(backward_ops, key=lambda x: -x[0]):
                self._backward_step(s, m, fwd_inputs, fwd_outputs, losses,
                                    grad_inputs)

        # Compute average loss
        valid_losses = [l.item() for l in losses if l is not None]
        avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0

        return {
            "loss": avg_loss,
            "losses": valid_losses,
        }

    def _forward_step(self, stage, micro_batch, inputs, labels,
                      fwd_inputs, fwd_outputs, losses):
        """Execute one forward step for a stage and micro-batch."""
        S = self.num_stages

        # Get input
        if stage == 0:
            # First stage: use the micro-batch input directly
            inp = inputs[micro_batch]
        else:
            # Get output from previous stage (detached for pipeline boundary)
            inp = fwd_outputs[stage - 1][micro_batch].detach()

        # Enable gradient tracking at stage boundaries
        inp = inp.requires_grad_(True)
        fwd_inputs[stage][micro_batch] = inp

        # Run forward through this stage's layers
        output = self.stage_modules[stage](inp)
        fwd_outputs[stage][micro_batch] = output

        # Last stage: compute loss
        if stage == S - 1 and self.loss_fn is not None and labels is not None:
            loss = self.loss_fn(output, labels[micro_batch])
            losses[micro_batch] = loss

    def _backward_step(self, stage, micro_batch, fwd_inputs, fwd_outputs,
                       losses, grad_inputs):
        """Execute one backward step for a stage and micro-batch."""
        S = self.num_stages

        output = fwd_outputs[stage][micro_batch]
        inp = fwd_inputs[stage][micro_batch]

        if stage == S - 1:
            # Last stage: backward from loss
            if losses[micro_batch] is not None:
                # Scale loss by 1/M for gradient accumulation
                scaled_loss = losses[micro_batch] / self.num_micro_batches
                scaled_loss.backward(retain_graph=False)
            else:
                # If no loss_fn, backward on output directly
                output.backward(
                    torch.ones_like(output) / self.num_micro_batches,
                    retain_graph=False,
                )
        else:
            # Non-last stage: backward using gradient from next stage
            grad_from_next = grad_inputs[stage + 1][micro_batch]
            if grad_from_next is not None:
                output.backward(grad_from_next, retain_graph=False)

        # Save input gradient for the previous stage
        if inp.grad is not None:
            grad_inputs[stage][micro_batch] = inp.grad.detach()

    def parameters(self):
        """Return all trainable parameters across all stages."""
        for stage_module in self.stage_modules:
            yield from stage_module.parameters()

    @staticmethod
    def split_model_layers(layers, num_stages):
        """Split a list of layers evenly across stages.

        Args:
            layers: List of nn.Module layers.
            num_stages: Number of pipeline stages.

        Returns:
            List of lists: stage_layers[stage_id] = [layer1, layer2, ...]
        """
        n = len(layers)
        assert n >= num_stages, (
            f"Cannot split {n} layers into {num_stages} stages"
        )

        # Even split with remainder going to earlier stages
        base = n // num_stages
        remainder = n % num_stages

        stage_layers = []
        idx = 0
        for s in range(num_stages):
            count = base + (1 if s < remainder else 0)
            stage_layers.append(layers[idx:idx + count])
            idx += count

        return stage_layers


class CheckpointedStage(nn.Module):
    """Pipeline stage with gradient checkpointing and optional CPU offload.

    Wraps a stage module's forward with checkpoint_cpu_offload, so that
    intermediate activations within the stage are offloaded to CPU during
    forward and reloaded+recomputed during backward. Stage boundary
    activations (input/output tensors) stay on GPU — they're managed by
    the PipelineEngine for inter-stage communication.

    Args:
        stage_module: The stage module to wrap.
        cpu_offload: If True, use checkpoint_cpu_offload (offloads to CPU).
            If False, use torch.utils.checkpoint (GPU-only recomputation).
    """

    def __init__(self, stage_module, cpu_offload=True):
        super().__init__()
        self.stage_module = stage_module
        self.cpu_offload = cpu_offload

    def forward(self, x):
        if self.training:
            if self.cpu_offload:
                from bitsandbytes.training import checkpoint_cpu_offload
                return checkpoint_cpu_offload(self.stage_module, x)
            else:
                return torch.utils.checkpoint.checkpoint(
                    self.stage_module, x, use_reentrant=False,
                )
        return self.stage_module(x)


class PipelineCheckpointer:
    """Wraps pipeline stages with gradient checkpointing.

    Provides a static method to wrap each stage module with
    CheckpointedStage. Stage boundary activations (passed between stages)
    remain on GPU for pipeline communication; only internal layer
    activations are checkpointed.

    Usage:
        stages = [SequentialStage(layers[:2]), SequentialStage(layers[2:])]
        stages = PipelineCheckpointer.wrap_stages(stages, cpu_offload=True)
        engine = PipelineEngine(stages, loss_fn=loss_fn, ...)
    """

    @staticmethod
    def wrap_stages(stage_modules, cpu_offload=True):
        """Wrap each stage with gradient checkpointing.

        Args:
            stage_modules: List of nn.Module stage modules.
            cpu_offload: If True, offload activations to CPU. If False,
                use standard gradient checkpointing (GPU recomputation only).

        Returns:
            List of CheckpointedStage modules.
        """
        return [CheckpointedStage(s, cpu_offload=cpu_offload) for s in stage_modules]


class SequentialStage(nn.Module):
    """A pipeline stage that sequentially runs a list of layers.

    Simple wrapper that takes a list of nn.Module layers and runs them
    in sequence. Used as the default stage module when splitting a model.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
