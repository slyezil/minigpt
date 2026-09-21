import torch
import torch.nn as nn

from src.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return self.weight * x * rms

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.up_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.down_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.nn.functional.silu(
            self.gate_proj(x)
        )

        value = self.up_proj(x)

        return self.down_proj(
            gate * value
        )

class TransformerBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.mlp_norm = RMSNorm(
            config.d_model
        )

        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Attention sublayer + residual
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward sublayer + residual
        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # ---------------------------------------------------------
        # 1. Token embeddings
        #
        # Token ID:
        #     42
        #
        # becomes:
        #     vector of 768 numbers
        # ---------------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # ---------------------------------------------------------
        # 2. Transformer stack
        #
        # For our real model:
        #     12 Transformer blocks
        # ---------------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # ---------------------------------------------------------
        # 3. Final normalization
        # ---------------------------------------------------------

        self.final_norm = RMSNorm(
            config.d_model
        )

        # ---------------------------------------------------------
        # 4. Language-model output head
        #
        # 768-dimensional token representation
        #              ↓
        #         16,000 logits
        # ---------------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        # Initialize the whole network first.
        self.apply(self._init_weights) 

        # ---------------------------------------------------------
        # 5. Weight tying
        #
        # Input embedding matrix and output projection matrix
        # share the SAME Parameter object.
        # ---------------------------------------------------------

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len = tokens.shape

        # ---------------------------------------------------------
        # Guard against sequences longer than RoPE/context support.
        # ---------------------------------------------------------

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length "
                f"{self.config.context_length}."
            )

        # ---------------------------------------------------------
        # Token IDs -> embeddings
        #
        # [B, T]
        #    ↓
        # [B, T, d_model]
        # ---------------------------------------------------------

        x = self.token_embedding(tokens)

        # ---------------------------------------------------------
        # Pass through all Transformer blocks
        # ---------------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # ---------------------------------------------------------
        # Final RMSNorm
        # ---------------------------------------------------------

        x = self.final_norm(x)

        # ---------------------------------------------------------
        # Convert every token representation into vocabulary logits
        #
        # [B, T, 768]
        #       ↓
        # [B, T, 16000]
        # ---------------------------------------------------------

        logits = self.lm_head(x)

        return logits
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )