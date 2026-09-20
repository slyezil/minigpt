import torch

from src.attention import RotaryEmbedding
from src.attention import CasualSelfAttention
from src.config import ModelConfig


def test_rope_preserves_shape():
    rope = RotaryEmbedding(dim=64, max_seq_len=512)

    x = torch.randn(2, 12, 16, 64)

    output = rope(x)

    assert output.shape == x.shape


def test_rope_position_zero_is_unchanged():
    rope = RotaryEmbedding(dim=4, max_seq_len=8)

    x = torch.tensor(
        [[[
            [1.0, 2.0, 3.0, 4.0]
        ]]]
    )

    output = rope(x)

    assert torch.allclose(output, x, atol=1e-6)


def test_rope_preserves_vector_norm():
    rope = RotaryEmbedding(dim=64, max_seq_len=512)

    x = torch.randn(2, 12, 32, 64)

    output = rope(x)

    original_norm = torch.linalg.vector_norm(x, dim=-1)
    rotated_norm = torch.linalg.vector_norm(output, dim=-1)

    assert torch.allclose(
        original_norm,
        rotated_norm,
        atol=1e-5,
    )
def test_attention_preserves_shape():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    attention = CausalSelfAttention(config)

    x = torch.randn(2, 8, 64)

    output = attention(x)

    assert output.shape == x.shape


def test_attention_has_expected_parameter_count():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    attention = CausalSelfAttention(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in attention.parameters()
    )

    expected = 4 * config.d_model * config.d_model

    assert parameter_count == expected


def test_attention_is_causal():
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    attention = CausalSelfAttention(config)
    attention.eval()

    x1 = torch.randn(1, 6, 64)

    x2 = x1.clone()

    # Change only token at position 4.
    x2[:, 4, :] = torch.randn(64) * 100

    output1 = attention(x1)
    output2 = attention(x2)

    # Positions 0-3 must NOT be affected by a future token.
    assert torch.allclose(
        output1[:, :4, :],
        output2[:, :4, :],
        atol=1e-5,
    )

    # Position 4 should change because its own input changed.
    assert not torch.allclose(
        output1[:, 4, :],
        output2[:, 4, :],
    )