"""Distributed pipeline parallelism test.

Run with: torchrun --nproc_per_node=2 tests/test_distributed_pipeline.py

Verifies that the distributed pipeline engine produces the same
gradients as single-process training with gradient accumulation.
"""

import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from bitsandbytes.pipeline import DistributedPipelineEngine, SequentialStage


class SimpleLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


def run_test():
    # Use gloo for point-to-point ops; NCCL send/recv can fail on single-GPU
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"Requires 2 processes, got {world_size}"

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    dim = 32
    M = 4
    batch = 4

    # Create layers with shared seeds so all ranks have the same initial weights
    torch.manual_seed(42)
    all_layers = [SimpleLayer(dim) for _ in range(4)]

    if rank == 0:
        my_layers = all_layers[:2]
    else:
        my_layers = all_layers[2:]

    my_stage = SequentialStage(my_layers).to(device)
    my_stage.zero_grad()

    # Create identical inputs/labels on all ranks
    torch.manual_seed(123)
    micro_inputs = [torch.randn(batch, dim) for _ in range(M)]
    micro_labels = [torch.randn(batch, dim) for _ in range(M)]

    loss_fn = lambda out, labels: (out - labels).pow(2).mean()

    # Run distributed pipeline
    engine = DistributedPipelineEngine(
        stage_module=my_stage,
        rank=rank,
        world_size=world_size,
        loss_fn=loss_fn,
        num_micro_batches=M,
        hidden_shape=(batch, dim),
        dtype=torch.float32,
    )

    result = engine.step(
        micro_batch_inputs=micro_inputs if rank == 0 else None,
        micro_batch_labels=micro_labels if rank == world_size - 1 else None,
    )

    # Collect per-layer gradients
    pipe_grads = {}
    for i, layer in enumerate(my_layers):
        layer_idx = i + (2 if rank == 1 else 0)
        if layer.linear.weight.grad is not None:
            pipe_grads[layer_idx] = layer.linear.weight.grad.clone()

    # Exchange gradients and loss: rank 1 sends to rank 0 (CPU for gloo)
    if rank == 0:
        for layer_idx in [2, 3]:
            buf = torch.empty(dim, dim)  # CPU tensor for gloo
            dist.recv(buf, src=1, tag=layer_idx)
            pipe_grads[layer_idx] = buf.to(device)
        # Receive loss from last rank
        loss_buf = torch.empty(1)
        dist.recv(loss_buf, src=world_size - 1, tag=100)
        pipeline_loss = loss_buf.item()
    else:
        for layer_idx in [2, 3]:
            dist.send(pipe_grads[layer_idx].cpu(), dst=0, tag=layer_idx)
        # Send loss to rank 0
        dist.send(torch.tensor([result["loss"]]), dst=0, tag=100)
        pipeline_loss = result["loss"]

    # Rank 0 computes reference and checks
    if rank == 0:
        torch.manual_seed(42)
        ref_layers = [SimpleLayer(dim).to(device) for _ in range(4)]
        for ref in ref_layers:
            ref.zero_grad()

        for m in range(M):
            x = micro_inputs[m].to(device)
            for ref in ref_layers:
                x = ref(x)
            loss = loss_fn(x, micro_labels[m].to(device)) / M
            loss.backward()

        ref_grads = [ref.linear.weight.grad.clone() for ref in ref_layers]

        all_pass = True
        for i in range(4):
            ref_g = ref_grads[i]
            pipe_g = pipe_grads.get(i)
            if pipe_g is None:
                print(f"FAIL: Layer {i} — no gradient")
                all_pass = False
            elif not torch.allclose(ref_g, pipe_g, atol=1e-5, rtol=1e-5):
                max_diff = (ref_g - pipe_g).abs().max().item()
                print(f"FAIL: Layer {i} — max diff: {max_diff:.2e}")
                all_pass = False
            else:
                print(f"PASS: Layer {i} — gradients match")

        print(f"\nPipeline loss: {pipeline_loss:.6f}")
        print(f"Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

        if not all_pass:
            sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_test()
