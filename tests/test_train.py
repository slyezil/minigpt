import torch

from src.config import ModelConfig
from src.model import MiniGPT
from src.train import next_token_loss, train_step
import math

from src.train import get_learning_rate


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

def test_learning_rate_starts_at_zero():
    lr = get_learning_rate(
        step=0,
        warmup_steps=100,
        total_steps=1000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    assert lr == 0.0


def test_learning_rate_reaches_max_after_warmup():
    lr = get_learning_rate(
        step=100,
        warmup_steps=100,
        total_steps=1000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    assert math.isclose(
        lr,
        3e-4,
        rel_tol=1e-6,
    )


def test_learning_rate_decays_to_minimum():
    lr = get_learning_rate(
        step=1000,
        warmup_steps=100,
        total_steps=1000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    assert math.isclose(
        lr,
        3e-5,
        rel_tol=1e-6,
    )


def test_learning_rate_is_between_min_and_max_during_decay():
    lr = get_learning_rate(
        step=550,
        warmup_steps=100,
        total_steps=1000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    assert 3e-5 < lr < 3e-4