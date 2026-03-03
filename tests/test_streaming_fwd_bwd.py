"""Tests for separated forward_streaming / backward_streaming API.

Verifies gradient correctness against a non-streaming reference model,
and tests gradient accumulation and training convergence.
"""

import os
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_pair():
    """Create matching non-streaming and streaming models from same checkpoint."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from bitsandbytes.checkpoint import save_lora, save_quantized
    from bitsandbytes.kbit_lora import KbitLoraModel

    config = LlamaConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config).to(torch.float16).cuda()

    kbit = KbitLoraModel(
        model,
        lora_r=4,
        lora_alpha=8.0,
        k=4,
        attn_chunk_size=64,
        mlp_chunk_size=64,
        ce_chunk_size=256,
        compute_dtype=torch.bfloat16,
    )

    tmpdir = tempfile.mkdtemp()
    quant_path = os.path.join(tmpdir, "quant.safetensors")
    lora_path = os.path.join(tmpdir, "lora.safetensors")
    save_quantized(kbit, quant_path)
    save_lora(kbit, lora_path)

    # Non-streaming reference (standard autograd works correctly)
    non_streaming = KbitLoraModel.from_quantized(
        quant_path,
        lora_r=4,
        lora_alpha=8.0,
        attn_chunk_size=64,
        mlp_chunk_size=64,
        ce_chunk_size=256,
        compute_dtype=torch.bfloat16,
        weight_streaming=False,
        lora_checkpoint=lora_path,
    )

    # Streaming model
    streaming = KbitLoraModel.from_quantized(
        quant_path,
        lora_r=4,
        lora_alpha=8.0,
        attn_chunk_size=64,
        mlp_chunk_size=64,
        ce_chunk_size=256,
        compute_dtype=torch.bfloat16,
        weight_streaming=True,
        lora_checkpoint=lora_path,
    )

    return non_streaming, streaming, tmpdir


@pytest.fixture(scope="module")
def model_pair():
    non_streaming, streaming, tmpdir = _make_model_pair()
    yield non_streaming, streaming
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


class TestForwardBackwardSeparation:
    def test_gradient_match(self, model_pair):
        """Streaming gradients must match non-streaming standard forward+backward."""
        non_streaming, streaming = model_pair
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        # ─── Reference: non-streaming forward() + loss.backward() ───
        non_streaming.train()
        for p in non_streaming.get_trainable_parameters():
            p.grad = None

        result = non_streaming(input_ids, labels=labels)
        result["loss"].backward()

        grads_ref = {}
        for name, p in non_streaming._lora_params.items():
            if p.grad is not None:
                grads_ref[name] = p.grad.clone()
        for name, p in non_streaming._norm_weights.items():
            if p.grad is not None:
                grads_ref[f"norm_{name}"] = p.grad.clone()

        loss_ref = result["loss"].detach()

        # ─── Streaming: forward_streaming + backward_streaming ───
        for p in streaming.get_trainable_parameters():
            p.grad = None

        loss_stream, ctx = streaming.forward_streaming(input_ids, labels)
        streaming.backward_streaming(ctx)

        grads_stream = {}
        for name, p in streaming._lora_params.items():
            if p.grad is not None:
                grads_stream[name] = p.grad.clone()
        for name, p in streaming._norm_weights.items():
            if p.grad is not None:
                grads_stream[f"norm_{name}"] = p.grad.clone()

        # Compare losses
        assert torch.allclose(loss_ref, loss_stream, atol=1e-5), (
            f"Loss mismatch: {loss_ref.item()} vs {loss_stream.item()}"
        )

        # Compare gradients
        assert set(grads_ref.keys()) == set(grads_stream.keys()), (
            f"Gradient key mismatch: {set(grads_ref) - set(grads_stream)} vs {set(grads_stream) - set(grads_ref)}"
        )

        for name in grads_ref:
            assert torch.allclose(grads_ref[name], grads_stream[name], atol=1e-5, rtol=1e-4), (
                f"Gradient mismatch for {name}: max diff {(grads_ref[name] - grads_stream[name]).abs().max().item()}"
            )

    def test_loss_curve_match(self, model_pair):
        """Loss curves must match between non-streaming and streaming over 20 steps."""
        non_streaming, streaming = model_pair
        lr = 1e-3

        # Set both models to same initial state
        for (n1, p1), (n2, p2) in zip(non_streaming._lora_params.items(), streaming._lora_params.items()):
            torch.manual_seed(42)
            val = torch.randn_like(p1.data) * 0.01
            p1.data.copy_(val)
            p2.data.copy_(val)
        for (n1, p1), (n2, p2) in zip(non_streaming._norm_weights.items(), streaming._norm_weights.items()):
            p1.data.fill_(1.0)
            p2.data.fill_(1.0)

        losses_ref = []
        losses_stream = []

        for step in range(20):
            torch.manual_seed(step + 1000)
            input_ids = torch.randint(0, 100, (1, 32), device="cuda")
            labels = input_ids.clone()

            # Non-streaming
            non_streaming.train()
            for p in non_streaming.get_trainable_parameters():
                p.grad = None
            result = non_streaming(input_ids, labels=labels)
            result["loss"].backward()
            losses_ref.append(result["loss"].item())
            for p in non_streaming.get_trainable_parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)

            # Streaming
            for p in streaming.get_trainable_parameters():
                p.grad = None
            loss_s, ctx = streaming.forward_streaming(input_ids, labels)
            streaming.backward_streaming(ctx)
            losses_stream.append(loss_s.item())
            for p in streaming.get_trainable_parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)

        # Losses should match at each step
        for i, (lr_val, ls_val) in enumerate(zip(losses_ref, losses_stream)):
            if lr_val == 0:
                continue
            rel_diff = abs(lr_val - ls_val) / abs(lr_val)
            assert rel_diff < 0.05, (
                f"Step {i}: ref loss {lr_val:.6f} vs stream loss {ls_val:.6f} (rel diff {rel_diff:.4f})"
            )

    def test_context_freed_after_backward(self, model_pair):
        """backward_streaming should free the context's checkpoint memory."""
        _, streaming = model_pair
        input_ids = torch.randint(0, 100, (1, 32), device="cuda")
        labels = input_ids.clone()

        for p in streaming.get_trainable_parameters():
            p.grad = None

        _, ctx = streaming.forward_streaming(input_ids, labels)
        assert len(ctx.checkpoints) > 0

        streaming.backward_streaming(ctx)
        assert len(ctx.checkpoints) == 0
        assert ctx.hidden_final is None
        assert ctx.grad_from_loss is None

    def test_gradient_accumulation(self, model_pair):
        """Multiple forward_streaming + backward_streaming calls should accumulate gradients."""
        _, streaming = model_pair

        for p in streaming.get_trainable_parameters():
            p.grad = None

        # Two micro-batches
        for _ in range(2):
            input_ids = torch.randint(0, 100, (1, 32), device="cuda")
            labels = input_ids.clone()

            _, ctx = streaming.forward_streaming(input_ids, labels)
            streaming.backward_streaming(ctx)

        # At least some parameters should have gradients
        has_grad = False
        for p in streaming.get_trainable_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients after 2 micro-batches"
