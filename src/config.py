from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_000
    context_length: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2048

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def estimated_parameter_count(self) -> int:
        token_embedding = self.vocab_size * self.d_model

        attention_per_layer = (
            4 * self.d_model * self.d_model
        )

        swiglu_per_layer = (
            3 * self.d_model * self.d_ff
        )

        rmsnorm_per_layer = 2 * self.d_model

        transformer_blocks = self.n_layers * (
            attention_per_layer
            + swiglu_per_layer
            + rmsnorm_per_layer
        )

        final_rmsnorm = self.d_model

        return (
            token_embedding
            + transformer_blocks
            + final_rmsnorm
        )