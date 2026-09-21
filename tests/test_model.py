import torch

from src.model import RMSNorm,SwiGLU, TransformerBlock,MiniGPT
from src.config import ModelConfig

def test_rmsnorm_preserves_shape():
    norm = RMSNorm(d_model=768)

    x = torch.randn(2, 16, 768)
    output = norm(x)

    assert output.shape == x.shape


def test_rmsnorm_matches_manual_calculation():
    norm = RMSNorm(d_model=4, eps=1e-6)

    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ]
    )

    output = norm(x)

    rms = torch.rsqrt(
        x.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )

    expected = x * rms

    assert torch.allclose(output, expected, atol=1e-5)
def test_swiglu_preserves_model_dimension():
    mlp = SwiGLU(
        d_model=64,
        d_ff=128,
    )

    x = torch.randn(2, 16, 64)

    output = mlp(x)

    assert output.shape == x.shape


def test_swiglu_has_expected_parameter_count():
    mlp = SwiGLU(
        d_model=64,
        d_ff=128,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in mlp.parameters()
    )

    expected = 3 * 64 * 128

    assert parameter_count == expected

def test_transformer_block_preserves_shape():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    block = TransformerBlock(config)

    x = torch.randn(2, 8, 64)

    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_residual_connection():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    block = TransformerBlock(config)

    # Make attention and MLP output exactly zero.
    for parameter in block.attention.parameters():
        parameter.data.zero_()

    for parameter in block.mlp.parameters():
        parameter.data.zero_()

    x = torch.randn(2, 8, 64)

    output = block(x)

    # If both sublayers produce zero,
    # residual paths should leave x unchanged.
    assert torch.allclose(
        output,
        x,
        atol=1e-6,
    )


def test_transformer_block_parameter_count():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    block = TransformerBlock(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in block.parameters()
    )

    attention = (
        4
        * config.d_model
        * config.d_model
    )

    swiglu = (
        3
        * config.d_model
        * config.d_ff
    )

    rmsnorm = (
        2
        * config.d_model
    )

    expected = (
        attention
        + swiglu
        + rmsnorm
    )

    assert parameter_count == expected

def test_minigpt_output_shape():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    model = MiniGPT(config)

    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    logits = model(tokens)

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )


def test_minigpt_ties_embedding_and_lm_head_weights():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    model = MiniGPT(config)

    assert (
        model.token_embedding.weight
        is model.lm_head.weight
    )


def test_minigpt_parameter_count_matches_config():
    config = ModelConfig()

    model = MiniGPT(config)

    actual = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert actual == config.estimated_parameter_count


def test_minigpt_rejects_sequence_longer_than_context():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    model = MiniGPT(config)

    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, 17),
    )

    try:
        model(tokens)
    except ValueError as exc:
        assert "context length" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected sequence longer than context_length "
            "to raise ValueError."
        )

def test_minigpt_uses_small_embedding_initialization():
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    model = MiniGPT(config)

    std = model.token_embedding.weight.std().item()

    assert 0.015 < std < 0.025