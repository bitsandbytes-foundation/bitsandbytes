"""Tests for 1F1B pipeline parallelism engine.

Verifies:
- Schedule generation correctness (all micro-batches covered, order valid)
- Single-process pipeline execution matches single-device training
- Gradient accumulation across micro-batches is correct
- Multi-stage pipeline produces same results as single stage
"""

import pytest
import torch
import torch.nn as nn

from bitsandbytes.pipeline import (
    PipelineEngine,
    SequentialStage,
    generate_1f1b_schedule,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ─── Schedule Generation Tests ────────────────────────────────────────────

class TestScheduleGeneration:

    def test_basic_schedule(self):
        """2 stages, 4 micro-batches — basic 1F1B schedule."""
        schedule = generate_1f1b_schedule(2, 4)

        assert len(schedule) == 2
        # Each stage should have 2*M operations (M forwards + M backwards)
        for s in range(2):
            ops = schedule[s]
            forwards = [m for op, m in ops if op == "F"]
            backwards = [m for op, m in ops if op == "B"]
            assert len(forwards) == 4, f"Stage {s}: expected 4 forwards, got {len(forwards)}"
            assert len(backwards) == 4, f"Stage {s}: expected 4 backwards, got {len(backwards)}"

    def test_all_micro_batches_covered(self):
        """Every micro-batch should have exactly one F and one B per stage."""
        for S in [2, 3, 4]:
            for M in [S, S + 1, S + 3, 8]:
                schedule = generate_1f1b_schedule(S, M)
                for s in range(S):
                    f_set = {m for op, m in schedule[s] if op == "F"}
                    b_set = {m for op, m in schedule[s] if op == "B"}
                    expected = set(range(M))
                    assert f_set == expected, (
                        f"S={S}, M={M}, stage {s}: forward set {f_set} != {expected}"
                    )
                    assert b_set == expected, (
                        f"S={S}, M={M}, stage {s}: backward set {b_set} != {expected}"
                    )

    def test_forward_before_backward(self):
        """For each micro-batch, forward should come before backward."""
        for S in [2, 3, 4]:
            for M in [S, S + 2, 8]:
                schedule = generate_1f1b_schedule(S, M)
                for s in range(S):
                    for m in range(M):
                        f_pos = next(i for i, (op, mb) in enumerate(schedule[s])
                                     if op == "F" and mb == m)
                        b_pos = next(i for i, (op, mb) in enumerate(schedule[s])
                                     if op == "B" and mb == m)
                        assert f_pos < b_pos, (
                            f"S={S}, M={M}, stage {s}, mb {m}: "
                            f"F at {f_pos}, B at {b_pos}"
                        )

    def test_warmup_counts(self):
        """Non-last stages should have (S-1-s) warmup forwards.

        The last stage has 0 warmup forwards in theory, but its schedule starts
        with one steady-state F (can't backward without a forward first), so
        consecutive F count at the start is max(1, S-1-s).
        """
        S, M = 4, 8
        schedule = generate_1f1b_schedule(S, M)

        for s in range(S):
            # Count consecutive forwards at the start
            warmup = 0
            for op, _ in schedule[s]:
                if op == "F":
                    warmup += 1
                else:
                    break

            if s < S - 1:
                expected = S - 1 - s
            else:
                expected = 1  # last stage: 0 warmup, but 1 steady F at start
            assert warmup == expected, (
                f"Stage {s}: expected {expected} consecutive forwards at start, got {warmup}"
            )

    def test_bounded_in_flight(self):
        """At most num_stages micro-batches should be in flight per stage."""
        for S in [2, 3, 4]:
            for M in [S, S + 2, 8]:
                schedule = generate_1f1b_schedule(S, M)
                for s in range(S):
                    in_flight = 0
                    max_in_flight = 0
                    for op, _ in schedule[s]:
                        if op == "F":
                            in_flight += 1
                        else:
                            in_flight -= 1
                        max_in_flight = max(max_in_flight, in_flight)
                    assert max_in_flight <= S, (
                        f"S={S}, M={M}, stage {s}: max in-flight {max_in_flight} > {S}"
                    )

    def test_minimum_micro_batches(self):
        """Should require M >= S."""
        with pytest.raises(AssertionError):
            generate_1f1b_schedule(4, 3)


# ─── Pipeline Engine Tests ────────────────────────────────────────────────

class SimpleLayer(nn.Module):
    """Simple linear layer for testing."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestPipelineEngine:

    @pytest.fixture
    def simple_model_setup(self):
        """Create a simple 4-layer model for testing."""
        dim = 32
        torch.manual_seed(42)

        layers = [SimpleLayer(dim).cuda() for _ in range(4)]

        return {
            "layers": layers,
            "dim": dim,
        }

    def test_pipeline_runs(self, simple_model_setup):
        """Pipeline should run without errors."""
        s = simple_model_setup
        dim = s["dim"]

        # Split into 2 stages
        stage0 = SequentialStage(s["layers"][:2]).cuda()
        stage1 = SequentialStage(s["layers"][2:]).cuda()

        engine = PipelineEngine(
            stage_modules=[stage0, stage1],
            loss_fn=lambda out, labels: (out - labels).pow(2).mean(),
            num_micro_batches=4,
        )

        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(4)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(4)]

        result = engine.step(micro_inputs, micro_labels)

        assert "loss" in result
        assert result["loss"] > 0
        assert len(result["losses"]) == 4

    def test_gradient_matches_single_device(self):
        """Pipeline gradients should match single-device gradient accumulation.

        This is the core correctness test: run the same model with the same
        inputs in pipeline mode and in single-device accumulated mode,
        and verify the gradients match.
        """
        dim = 32
        M = 4  # micro-batches
        torch.manual_seed(42)

        # Create model layers (will be shared between pipeline and reference)
        layer0 = SimpleLayer(dim).cuda()
        layer1 = SimpleLayer(dim).cuda()
        layer2 = SimpleLayer(dim).cuda()
        layer3 = SimpleLayer(dim).cuda()

        # Create inputs and labels
        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        # --- Reference: single-device gradient accumulation ---
        ref_layers = [
            SimpleLayer(dim).cuda(), SimpleLayer(dim).cuda(),
            SimpleLayer(dim).cuda(), SimpleLayer(dim).cuda(),
        ]
        # Copy weights
        for ref, orig in zip(ref_layers, [layer0, layer1, layer2, layer3]):
            ref.linear.weight.data.copy_(orig.linear.weight.data)

        # Forward + backward for all micro-batches, accumulate gradients
        for ref in ref_layers:
            ref.zero_grad()

        for m in range(M):
            x = micro_inputs[m]
            for ref in ref_layers:
                x = ref(x)
            loss = loss_fn(x, micro_labels[m]) / M  # Scale by 1/M for accumulation
            loss.backward()

        ref_grads = [ref.linear.weight.grad.clone() for ref in ref_layers]

        # --- Pipeline: 2 stages ---
        # Reset gradients on original layers
        for layer in [layer0, layer1, layer2, layer3]:
            layer.zero_grad()

        stage0 = SequentialStage([layer0, layer1]).cuda()
        stage1 = SequentialStage([layer2, layer3]).cuda()

        engine = PipelineEngine(
            stage_modules=[stage0, stage1],
            loss_fn=loss_fn,
            num_micro_batches=M,
        )

        result = engine.step(micro_inputs, micro_labels)

        # Compare gradients
        pipeline_grads = [
            layer0.linear.weight.grad,
            layer1.linear.weight.grad,
            layer2.linear.weight.grad,
            layer3.linear.weight.grad,
        ]

        for i, (ref_g, pipe_g) in enumerate(zip(ref_grads, pipeline_grads)):
            assert pipe_g is not None, f"Layer {i}: no gradient from pipeline"
            torch.testing.assert_close(
                ref_g, pipe_g,
                atol=1e-5, rtol=1e-5,
                msg=f"Layer {i}: gradient mismatch",
            )

    def test_loss_matches_single_device(self):
        """Pipeline loss should match single-device computation."""
        dim = 32
        M = 4
        torch.manual_seed(42)

        layers = [SimpleLayer(dim).cuda() for _ in range(4)]

        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        # Reference losses
        ref_losses = []
        for m in range(M):
            x = micro_inputs[m]
            for layer in layers:
                x = layer(x)
            loss = loss_fn(x, micro_labels[m])
            ref_losses.append(loss.item())

        # Pipeline
        stage0 = SequentialStage(layers[:2]).cuda()
        stage1 = SequentialStage(layers[2:]).cuda()

        engine = PipelineEngine(
            stage_modules=[stage0, stage1],
            loss_fn=loss_fn,
            num_micro_batches=M,
        )

        result = engine.step(micro_inputs, micro_labels)

        for i, (ref_l, pipe_l) in enumerate(zip(ref_losses, result["losses"])):
            assert abs(ref_l - pipe_l) < 1e-5, (
                f"Micro-batch {i}: ref loss {ref_l:.6f} vs pipeline loss {pipe_l:.6f}"
            )

    def test_three_stages(self):
        """Pipeline should work with 3 stages."""
        dim = 32
        M = 6
        torch.manual_seed(42)

        layers = [SimpleLayer(dim).cuda() for _ in range(6)]

        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        # Reference
        ref_grads = {}
        for layer in layers:
            layer.zero_grad()
        for m in range(M):
            x = micro_inputs[m]
            for layer in layers:
                x = layer(x)
            loss = loss_fn(x, micro_labels[m]) / M
            loss.backward()
        for i, layer in enumerate(layers):
            ref_grads[i] = layer.linear.weight.grad.clone()

        # Pipeline with 3 stages
        for layer in layers:
            layer.zero_grad()
        stages = [
            SequentialStage(layers[0:2]).cuda(),
            SequentialStage(layers[2:4]).cuda(),
            SequentialStage(layers[4:6]).cuda(),
        ]
        engine = PipelineEngine(
            stage_modules=stages, loss_fn=loss_fn, num_micro_batches=M,
        )
        result = engine.step(micro_inputs, micro_labels)

        for i, layer in enumerate(layers):
            assert layer.linear.weight.grad is not None, f"Layer {i}: no gradient"
            torch.testing.assert_close(
                ref_grads[i], layer.linear.weight.grad,
                atol=1e-5, rtol=1e-5,
                msg=f"Layer {i}: gradient mismatch (3 stages)",
            )

    def test_four_stages(self):
        """Pipeline should work with 4 stages."""
        dim = 32
        M = 8
        torch.manual_seed(42)

        layers = [SimpleLayer(dim).cuda() for _ in range(4)]

        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        # Reference
        for layer in layers:
            layer.zero_grad()
        for m in range(M):
            x = micro_inputs[m]
            for layer in layers:
                x = layer(x)
            loss = loss_fn(x, micro_labels[m]) / M
            loss.backward()
        ref_grads = [layer.linear.weight.grad.clone() for layer in layers]

        # Pipeline: 4 stages (1 layer each)
        for layer in layers:
            layer.zero_grad()
        stages = [SequentialStage([layer]).cuda() for layer in layers]
        engine = PipelineEngine(
            stage_modules=stages, loss_fn=loss_fn, num_micro_batches=M,
        )
        result = engine.step(micro_inputs, micro_labels)

        for i, layer in enumerate(layers):
            assert layer.linear.weight.grad is not None, f"Layer {i}: no gradient"
            torch.testing.assert_close(
                ref_grads[i], layer.linear.weight.grad,
                atol=1e-5, rtol=1e-5,
                msg=f"Layer {i}: gradient mismatch (4 stages)",
            )

    def test_split_model_layers(self):
        """split_model_layers should evenly distribute layers."""
        layers = list(range(7))

        splits = PipelineEngine.split_model_layers(layers, 3)
        assert len(splits) == 3
        assert splits == [[0, 1, 2], [3, 4], [5, 6]]

        splits = PipelineEngine.split_model_layers(layers, 2)
        assert len(splits) == 2
        assert splits == [[0, 1, 2, 3], [4, 5, 6]]

        splits = PipelineEngine.split_model_layers(layers, 7)
        assert len(splits) == 7
        assert all(len(s) == 1 for s in splits)

    def test_nonlinear_model(self):
        """Pipeline should work with nonlinear layers (ReLU, etc.)."""
        dim = 32
        M = 4
        torch.manual_seed(42)

        # Model with ReLU activations
        class ReLULayer(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = nn.Linear(d, d, bias=True)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        layers = [ReLULayer(dim).cuda() for _ in range(4)]

        micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
        micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        # Reference
        for layer in layers:
            layer.zero_grad()
        for m in range(M):
            x = micro_inputs[m]
            for layer in layers:
                x = layer(x)
            loss = loss_fn(x, micro_labels[m]) / M
            loss.backward()
        ref_grads = [layer.linear.weight.grad.clone() for layer in layers]

        # Pipeline
        for layer in layers:
            layer.zero_grad()
        stages = [
            SequentialStage(layers[:2]).cuda(),
            SequentialStage(layers[2:]).cuda(),
        ]
        engine = PipelineEngine(
            stage_modules=stages, loss_fn=loss_fn, num_micro_batches=M,
        )
        result = engine.step(micro_inputs, micro_labels)

        for i, layer in enumerate(layers):
            torch.testing.assert_close(
                ref_grads[i], layer.linear.weight.grad,
                atol=1e-5, rtol=1e-5,
                msg=f"ReLU layer {i}: gradient mismatch",
            )

    def test_multiple_steps(self):
        """Multiple training steps should accumulate correctly."""
        dim = 32
        M = 2
        torch.manual_seed(42)

        layers = [SimpleLayer(dim).cuda() for _ in range(2)]

        loss_fn = lambda out, labels: (out - labels).pow(2).mean()

        stage0 = SequentialStage([layers[0]]).cuda()
        stage1 = SequentialStage([layers[1]]).cuda()

        engine = PipelineEngine(
            stage_modules=[stage0, stage1],
            loss_fn=loss_fn,
            num_micro_batches=M,
        )

        # Run two steps
        for _ in range(2):
            for layer in layers:
                layer.zero_grad()

            micro_inputs = [torch.randn(4, dim, device="cuda") for _ in range(M)]
            micro_labels = [torch.randn(4, dim, device="cuda") for _ in range(M)]

            result = engine.step(micro_inputs, micro_labels)
            assert result["loss"] > 0

            # All parameters should have gradients
            for layer in layers:
                assert layer.linear.weight.grad is not None
                assert (layer.linear.weight.grad != 0).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
