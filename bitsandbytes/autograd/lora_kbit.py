"""Fused LoRA autograd functions for kbit-quantized weights.

These custom autograd functions exploit LoRA's low-rank structure for
bracket-optimized matrix chains, reducing FLOPs and memory compared to
naive per-projection backward passes.

All functions operate on kbit-quantized weights (k=2-5, blocksize 32)
via ``dequantize_kbit``.  The base weight gradient is never computed
(frozen weights).

Convention (matching PEFT):
    - W: base weight, shape [N, K]  (out_features × in_features)
    - A: lora_A weight, shape [r, K] (rank × in_features)
    - B: lora_B weight, shape [N, r] (out_features × rank)
    - s: scaling factor (typically lora_alpha / r)
    - X: input activation, shape [M, K] (batch × in_features)

Forward:  out = X @ W^T + (X @ A^T @ B^T) * s
"""

import torch

import bitsandbytes.functional as F


class LoRA_W_Kbit(torch.autograd.Function):
    """Single linear projection with LoRA on kbit-quantized weight.

    Forward:  out = X @ W_deq^T + (X @ A^T @ B^T) * s
    Backward: grad_A, grad_B, grad_X (no gradient for base weight)

    Bracket optimization for backward:
        grad_out @ B produces [M, r] (small), then chained with A or X.
    """

    @staticmethod
    def forward(
        ctx,
        X,         # [M, K]
        packed,    # int32, kbit packed weight
        absmax,    # float32, per-block absmax
        codebook,  # float32, 2^k entries
        A,         # [r, K] lora_A weight
        B,         # [N, r] lora_B weight
        s,         # scalar scaling factor
        k,         # bit width
        K_dim,     # reduction dimension
        N_padded,  # padded output dimension
        N,         # original output dimension
        compute_dtype,
    ):
        # Dequantize base weight
        n_elements = N_padded * K_dim
        w_deq = F.dequantize_kbit(packed, absmax, codebook, k, n_elements, compute_dtype)
        W = w_deq[:n_elements].reshape(N_padded, K_dim)[:N, :]  # [N, K]

        # Base matmul + LoRA contribution
        out = X @ W.t()                            # [M, N]
        lora_out = (X @ A.t()) @ B.t()             # [M, r] @ [r, N] = [M, N]
        out = out + lora_out * s

        # Save for backward
        ctx.save_for_backward(X, A, B, packed, absmax, codebook)
        ctx.s = s
        ctx.k = k
        ctx.K_dim = K_dim
        ctx.N_padded = N_padded
        ctx.N = N
        ctx.compute_dtype = compute_dtype

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for X, A, B.

        Key shapes (r << K, r << N):
            grad_output: [M, N]
            gB = grad_output @ B  -> [M, r]  (small intermediate)
        """
        X, A, B, packed, absmax, codebook = ctx.saved_tensors
        s = ctx.s
        grad_X = grad_A = grad_B = None

        if ctx.needs_input_grad[4]:  # grad_A
            # dL/dA = s * (grad_output @ B)^T @ X  = s * B^T @ grad_output^T @ X  [r, K]
            gB = grad_output @ B                    # [M, r] — bracket optimized
            grad_A = (gB.t() @ X) * s               # [r, M] @ [M, K] = [r, K]

        if ctx.needs_input_grad[5]:  # grad_B
            # dL/dB = s * grad_output^T @ (X @ A^T) = s * grad_output^T @ Z  [N, r]
            Z = X @ A.t()                           # [M, r]
            grad_B = (grad_output.t() @ Z) * s      # [N, M] @ [M, r] = [N, r]

        if ctx.needs_input_grad[0]:  # grad_X
            # dL/dX = grad_output @ W_deq + s * grad_output @ B @ A  [M, K]
            n_elements = ctx.N_padded * ctx.K_dim
            w_deq = F.dequantize_kbit(
                packed, absmax, codebook, ctx.k, n_elements, ctx.compute_dtype,
            )
            W = w_deq[:n_elements].reshape(ctx.N_padded, ctx.K_dim)[:ctx.N, :]
            grad_X = grad_output @ W                # [M, N] @ [N, K] = [M, K]
            if gB is None:
                gB = grad_output @ B
            grad_X = grad_X + (gB @ A) * s          # [M, r] @ [r, K] = [M, K]

        # No gradient for: packed, absmax, codebook, s, k, K_dim, N_padded, N, compute_dtype
        return grad_X, None, None, None, grad_A, grad_B, None, None, None, None, None, None


class LoRA_QKV_Kbit(torch.autograd.Function):
    """Fused Q+K+V projections with LoRA on kbit-quantized weights.

    Computes three projections in one call:
        Q = X @ W_q^T + (X @ A_q^T @ B_q^T) * s_q
        K = X @ W_k^T + (X @ A_k^T @ B_k^T) * s_k
        V = X @ W_v^T + (X @ A_v^T @ B_v^T) * s_v

    Combined grad_X = grad_X_q + grad_X_k + grad_X_v (accumulated in-place).
    """

    @staticmethod
    def forward(
        ctx,
        X,          # [M, K]
        # Q projection
        packed_q, absmax_q, codebook_q, A_q, B_q, s_q,
        # K projection
        packed_k, absmax_k, codebook_k, A_k, B_k, s_k,
        # V projection
        packed_v, absmax_v, codebook_v, A_v, B_v, s_v,
        # Shared params
        k, K_dim, N_padded, N, compute_dtype,
    ):
        n_elements = N_padded * K_dim

        results = []
        for packed, absmax, codebook, A, B, s in [
            (packed_q, absmax_q, codebook_q, A_q, B_q, s_q),
            (packed_k, absmax_k, codebook_k, A_k, B_k, s_k),
            (packed_v, absmax_v, codebook_v, A_v, B_v, s_v),
        ]:
            w_deq = F.dequantize_kbit(packed, absmax, codebook, k, n_elements, compute_dtype)
            W = w_deq[:n_elements].reshape(N_padded, K_dim)[:N, :]
            out = X @ W.t() + (X @ A.t()) @ B.t() * s
            results.append(out)

        ctx.save_for_backward(
            X,
            packed_q, absmax_q, codebook_q, A_q, B_q,
            packed_k, absmax_k, codebook_k, A_k, B_k,
            packed_v, absmax_v, codebook_v, A_v, B_v,
        )
        ctx.s_q, ctx.s_k, ctx.s_v = s_q, s_k, s_v
        ctx.k = k
        ctx.K_dim = K_dim
        ctx.N_padded = N_padded
        ctx.N = N
        ctx.compute_dtype = compute_dtype

        return results[0], results[1], results[2]

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        (
            X,
            packed_q, absmax_q, codebook_q, A_q, B_q,
            packed_k, absmax_k, codebook_k, A_k, B_k,
            packed_v, absmax_v, codebook_v, A_v, B_v,
        ) = ctx.saved_tensors

        n_elements = ctx.N_padded * ctx.K_dim
        grad_X = torch.zeros_like(X) if ctx.needs_input_grad[0] else None

        all_grad_A = [None, None, None]
        all_grad_B = [None, None, None]

        projections = [
            (grad_q, packed_q, absmax_q, codebook_q, A_q, B_q, ctx.s_q, 4, 5),
            (grad_k, packed_k, absmax_k, codebook_k, A_k, B_k, ctx.s_k, 10, 11),
            (grad_v, packed_v, absmax_v, codebook_v, A_v, B_v, ctx.s_v, 16, 17),
        ]

        for idx, (grad_out, packed, absmax, codebook, A, B, s, a_idx, b_idx) in enumerate(projections):
            gB = grad_out @ B  # [M, r]

            if ctx.needs_input_grad[a_idx]:
                all_grad_A[idx] = (gB.t() @ X) * s

            if ctx.needs_input_grad[b_idx]:
                Z = X @ A.t()
                all_grad_B[idx] = (grad_out.t() @ Z) * s

            if grad_X is not None:
                w_deq = F.dequantize_kbit(
                    packed, absmax, codebook, ctx.k, n_elements, ctx.compute_dtype,
                )
                W = w_deq[:n_elements].reshape(ctx.N_padded, ctx.K_dim)[:ctx.N, :]
                grad_X += grad_out @ W + (gB @ A) * s

        # Return: X, packed_q, absmax_q, codebook_q, A_q, B_q, s_q,
        #         packed_k, absmax_k, codebook_k, A_k, B_k, s_k,
        #         packed_v, absmax_v, codebook_v, A_v, B_v, s_v,
        #         k, K_dim, N_padded, N, compute_dtype
        return (
            grad_X,
            None, None, None, all_grad_A[0], all_grad_B[0], None,
            None, None, None, all_grad_A[1], all_grad_B[1], None,
            None, None, None, all_grad_A[2], all_grad_B[2], None,
            None, None, None, None, None,
        )


class LoRA_MLP_Kbit(torch.autograd.Function):
    """Fused gate+up+down MLP with LoRA on kbit-quantized weights and SwiGLU.

    Forward:
        e = X @ W_gate^T + (X @ A_gate^T @ B_gate^T) * s_gate   # gate
        g = X @ W_up^T   + (X @ A_up^T   @ B_up^T)   * s_up     # up
        h = silu(e) * g                                           # SwiGLU
        out = h @ W_down^T + (h @ A_down^T @ B_down^T) * s_down  # down

    Backward computes 6 adapter gradients + grad_X with bracket optimization.
    """

    @staticmethod
    def forward(
        ctx,
        X,           # [M, K]
        # Gate projection
        packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
        # Up projection
        packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
        # Down projection
        packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
        # Shared params
        k, K_dim_in, N_hidden, N_hidden_padded,
        K_dim_hidden, N_out, N_out_padded,
        compute_dtype,
    ):
        n_gate = N_hidden_padded * K_dim_in
        n_down = N_out_padded * K_dim_hidden

        # Gate projection
        w_deq = F.dequantize_kbit(packed_gate, absmax_gate, codebook_gate, k, n_gate, compute_dtype)
        W_gate = w_deq[:n_gate].reshape(N_hidden_padded, K_dim_in)[:N_hidden, :]
        e = X @ W_gate.t() + (X @ A_gate.t()) @ B_gate.t() * s_gate

        # Up projection
        w_deq = F.dequantize_kbit(packed_up, absmax_up, codebook_up, k, n_gate, compute_dtype)
        W_up = w_deq[:n_gate].reshape(N_hidden_padded, K_dim_in)[:N_hidden, :]
        g = X @ W_up.t() + (X @ A_up.t()) @ B_up.t() * s_up

        # SwiGLU activation
        sig_e = torch.sigmoid(e)
        silu_e = e * sig_e
        h = silu_e * g

        # Down projection
        w_deq = F.dequantize_kbit(packed_down, absmax_down, codebook_down, k, n_down, compute_dtype)
        W_down = w_deq[:n_down].reshape(N_out_padded, K_dim_hidden)[:N_out, :]
        out = h @ W_down.t() + (h @ A_down.t()) @ B_down.t() * s_down

        ctx.save_for_backward(
            X, e, sig_e, g, h,
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate,
            packed_up, absmax_up, codebook_up, A_up, B_up,
            packed_down, absmax_down, codebook_down, A_down, B_down,
        )
        ctx.s_gate = s_gate
        ctx.s_up = s_up
        ctx.s_down = s_down
        ctx.k = k
        ctx.K_dim_in = K_dim_in
        ctx.N_hidden = N_hidden
        ctx.N_hidden_padded = N_hidden_padded
        ctx.K_dim_hidden = K_dim_hidden
        ctx.N_out = N_out
        ctx.N_out_padded = N_out_padded
        ctx.compute_dtype = compute_dtype

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            X, e, sig_e, g, h,
            packed_gate, absmax_gate, codebook_gate, A_gate, B_gate,
            packed_up, absmax_up, codebook_up, A_up, B_up,
            packed_down, absmax_down, codebook_down, A_down, B_down,
        ) = ctx.saved_tensors

        # --- Down projection backward ---
        n_down = ctx.N_out_padded * ctx.K_dim_hidden
        w_deq = F.dequantize_kbit(
            packed_down, absmax_down, codebook_down, ctx.k, n_down, ctx.compute_dtype,
        )
        W_down = w_deq[:n_down].reshape(ctx.N_out_padded, ctx.K_dim_hidden)[:ctx.N_out, :]

        # grad_h = grad_output @ W_down + s_down * grad_output @ B_down @ A_down
        gB_down = grad_output @ B_down                          # [M, r]
        grad_h = grad_output @ W_down + (gB_down @ A_down) * ctx.s_down  # [M, K_hidden]

        grad_A_down = (gB_down.t() @ h) * ctx.s_down           # [r, K_hidden]
        Z_down = h @ A_down.t()                                 # [M, r]
        grad_B_down = (grad_output.t() @ Z_down) * ctx.s_down  # [N_out, r]

        # --- SwiGLU backward ---
        # h = silu(e) * g, where silu(e) = e * sigmoid(e)
        # dh/de = g * (sigmoid(e) + e * sigmoid(e) * (1 - sigmoid(e)))
        #       = g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
        # dh/dg = silu(e)
        silu_e = e * sig_e
        grad_e = grad_h * g * sig_e * (1.0 + e * (1.0 - sig_e))
        grad_g = grad_h * silu_e

        # --- Gate projection backward ---
        n_gate = ctx.N_hidden_padded * ctx.K_dim_in
        w_deq = F.dequantize_kbit(
            packed_gate, absmax_gate, codebook_gate, ctx.k, n_gate, ctx.compute_dtype,
        )
        W_gate = w_deq[:n_gate].reshape(ctx.N_hidden_padded, ctx.K_dim_in)[:ctx.N_hidden, :]

        gB_gate = grad_e @ B_gate                                # [M, r]
        grad_A_gate = (gB_gate.t() @ X) * ctx.s_gate            # [r, K_in]
        Z_gate = X @ A_gate.t()                                  # [M, r]
        grad_B_gate = (grad_e.t() @ Z_gate) * ctx.s_gate        # [N_hidden, r]

        grad_X = grad_e @ W_gate + (gB_gate @ A_gate) * ctx.s_gate  # [M, K_in]

        # --- Up projection backward ---
        w_deq = F.dequantize_kbit(
            packed_up, absmax_up, codebook_up, ctx.k, n_gate, ctx.compute_dtype,
        )
        W_up = w_deq[:n_gate].reshape(ctx.N_hidden_padded, ctx.K_dim_in)[:ctx.N_hidden, :]

        gB_up = grad_g @ B_up                                    # [M, r]
        grad_A_up = (gB_up.t() @ X) * ctx.s_up                  # [r, K_in]
        Z_up = X @ A_up.t()                                      # [M, r]
        grad_B_up = (grad_g.t() @ Z_up) * ctx.s_up              # [N_hidden, r]

        grad_X = grad_X + grad_g @ W_up + (gB_up @ A_up) * ctx.s_up

        # Return order matches forward args:
        # X, packed_gate, absmax_gate, codebook_gate, A_gate, B_gate, s_gate,
        #    packed_up, absmax_up, codebook_up, A_up, B_up, s_up,
        #    packed_down, absmax_down, codebook_down, A_down, B_down, s_down,
        #    k, K_dim_in, N_hidden, N_hidden_padded,
        #    K_dim_hidden, N_out, N_out_padded, compute_dtype
        return (
            grad_X,
            None, None, None, grad_A_gate, grad_B_gate, None,
            None, None, None, grad_A_up, grad_B_up, None,
            None, None, None, grad_A_down, grad_B_down, None,
            None, None, None, None,
            None, None, None, None,
        )
