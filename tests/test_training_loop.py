import math

import torch

from src.config import ModelConfig
from src.model import MiniGPT
from src.train import (
    evaluate,
    train_accumulated_step,
)


class FakeDataset:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
    ):
        # +1 because targets are shifted by one token.
        tokens = torch.randint(
            0,
            self.vocab_size,
            (
                batch_size,
                self.context_length + 1,
            ),
            device=device,
        )

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        return inputs, targets


def make_tiny_model():
    config = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )

    return config, MiniGPT(config)


def test_accumulated_step_updates_parameters():
    torch.manual_seed(42)

    config, model = make_tiny_model()

    device = torch.device("cpu")

    model.to(device)

    dataset = FakeDataset(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    before = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    loss = train_accumulated_step(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
        batch_size=2,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
    )

    after = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    assert math.isfinite(loss)

    assert not torch.equal(
        before,
        after,
    )


def test_evaluate_returns_finite_loss():
    torch.manual_seed(42)

    config, model = make_tiny_model()

    device = torch.device("cpu")

    model.to(device)

    dataset = FakeDataset(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
    )

    loss = evaluate(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=2,
        eval_batches=3,
    )

    assert math.isfinite(loss)


def test_evaluate_does_not_create_gradients():
    config, model = make_tiny_model()

    device = torch.device("cpu")

    model.to(device)

    dataset = FakeDataset(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
    )

    for parameter in model.parameters():
        parameter.grad = None

    evaluate(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=2,
        eval_batches=2,
    )

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )