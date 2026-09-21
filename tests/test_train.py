import torch

from src.config import ModelConfig
from src.model import MiniGPT
from src.train import next_token_loss, train_step


def test_train_step_updates_model_parameters():
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    inputs = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    before = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    loss = train_step(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
    )

    after = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, after)