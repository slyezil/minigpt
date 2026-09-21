from pathlib import Path

import numpy as np
import torch

from src.dataset import (
    TokenDataset,
    write_token_file,
)
from src.tokenizer import train_tokenizer


def test_write_token_file_creates_uint16_binary_file(
    tmp_path: Path,
):
    texts = [
        "Once upon a time there was a cat.",
        "The cat was happy.",
    ] * 20

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    output_path = tmp_path / "tokens.bin"

    token_count = write_token_file(
        texts=texts,
        tokenizer=tokenizer,
        output_path=output_path,
    )

    assert output_path.exists()

    data = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="r",
    )

    assert len(data) == token_count
    assert token_count > 0


def test_write_token_file_adds_eos_between_documents(
    tmp_path: Path,
):
    texts = [
        "The cat sat down.",
        "The dog ran away.",
    ]

    tokenizer = train_tokenizer(
        texts=texts * 20,
        vocab_size=100,
    )

    output_path = tmp_path / "tokens.bin"

    write_token_file(
        texts=texts,
        tokenizer=tokenizer,
        output_path=output_path,
    )

    data = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="r",
    )

    eos_id = tokenizer.token_to_id("<eos>")

    assert eos_id is not None

    eos_count = int(
        np.count_nonzero(data == eos_id)
    )

    assert eos_count == len(texts)


def test_token_dataset_returns_input_and_shifted_target(
    tmp_path: Path,
):
    token_ids = np.arange(
        100,
        dtype=np.uint16,
    )

    output_path = tmp_path / "tokens.bin"

    token_ids.tofile(output_path)

    dataset = TokenDataset(
        path=output_path,
        context_length=8,
    )

    inputs, targets = dataset.get_batch(
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)

    assert torch.equal(
        inputs[:, 1:],
        targets[:, :-1],
    )


def test_token_dataset_reports_token_count(
    tmp_path: Path,
):
    token_ids = np.arange(
        200,
        dtype=np.uint16,
    )

    output_path = tmp_path / "tokens.bin"

    token_ids.tofile(output_path)

    dataset = TokenDataset(
        path=output_path,
        context_length=16,
    )

    assert dataset.token_count == 200