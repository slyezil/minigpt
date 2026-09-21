import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")

        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32)
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        angles = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len {self.max_seq_len}."
            )

        cos = self.cos[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = self.sin[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.empty_like(x)

        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.scale = self.head_dim ** -0.5

        # Q, K and V projections.
        #
        # Each projection:
        #   [B, T, d_model] -> [B, T, d_model]
        #
        # We deliberately use bias=False because our parameter-count
        # calculation assumes bias-free linear layers.
        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        # After all heads have been combined, this projects the result
        # back into the model's embedding space.
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.context_length,
        )

        # Lower-triangular causal mask.
        #
        # Example for T=4:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        causal_mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # ---------------------------------------------------------
        # 1. Create Query, Key and Value
        # ---------------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Current shape:
        #
        # [B, T, d_model]
        #
        # Example:
        # [2, 512, 768]

        # ---------------------------------------------------------
        # 2. Split d_model into attention heads
        # ---------------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        # Change:
        #
        # [B, T, H, D]
        #
        # into:
        #
        # [B, H, T, D]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # For MiniGPT:
        #
        # [B, 12, T, 64]

        # ---------------------------------------------------------
        # 3. Add positional information using RoPE
        # ---------------------------------------------------------

        q = self.rope(q)
        k = self.rope(k)

        # Notice:
        #
        # Q -> RoPE
        # K -> RoPE
        # V -> unchanged

        # ---------------------------------------------------------
        # 4. Calculate attention scores
        # ---------------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores * self.scale

        # scores shape:
        #
        # [B, H, T, T]

        # ---------------------------------------------------------
        # 5. Apply causal mask
        # ---------------------------------------------------------

        mask = self.causal_mask[
            :seq_len,
            :seq_len,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        # ---------------------------------------------------------
        # 6. Convert scores to probabilities
        # ---------------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # ---------------------------------------------------------
        # 7. Weighted combination of Values
        # ---------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        # Shape:
        #
        # [B, H, T, D]

        # ---------------------------------------------------------
        # 8. Put all heads back together
        # ---------------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        # Back to:
        #
        # [B, T, d_model]

        # ---------------------------------------------------------
        # 9. Output projection
        # ---------------------------------------------------------

        return self.out_proj(attention_output)