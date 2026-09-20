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