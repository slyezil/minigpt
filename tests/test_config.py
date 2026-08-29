from src.config import ModelConfig


def test_default_model_config():
    config = ModelConfig()

    assert config.vocab_size == 16_000
    assert config.context_length == 512
    assert config.d_model == 768
    assert config.n_layers == 12
    assert config.n_heads == 12
    assert config.d_ff == 2048


def test_head_dimension():
    config = ModelConfig()

    assert config.head_dim == 64


def test_parameter_estimate():
    config = ModelConfig()

    assert config.estimated_parameter_count == 97_241_856