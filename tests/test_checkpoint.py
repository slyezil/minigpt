from pathlib import Path

import torch

from src.config import ModelConfig
from src.model import MiniGPT
from src.train import (
    load_checkpoint,
    save_checkpoint,
)


def make_tiny_model():
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

    return model, optimizer


def test_checkpoint_file_is_created(
    tmp_path: Path,
):
    model, optimizer = make_tiny_model()

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=123,
        tokens_seen=456_789,
    )

    assert checkpoint_path.exists()


def test_checkpoint_restores_model_parameters(
    tmp_path: Path,
):
    torch.manual_seed(42)

    model, optimizer = make_tiny_model()

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    original = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=123,
        tokens_seen=456_789,
    )

    with torch.no_grad():
        model.token_embedding.weight.zero_()

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    restored = (
        model.token_embedding.weight
        .detach()
        .clone()
    )

    assert torch.equal(
        original,
        restored,
    )


def test_checkpoint_restores_training_state(
    tmp_path: Path,
):
    model, optimizer = make_tiny_model()

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=321,
        tokens_seen=9_876_543,
    )

    state = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert state["step"] == 321
    assert (
        state["tokens_seen"]
        == 9_876_543
    )