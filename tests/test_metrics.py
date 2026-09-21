import csv
from pathlib import Path

from src.train import log_metrics


def test_log_metrics_creates_csv(
    tmp_path: Path,
):
    path = tmp_path / "training.csv"

    log_metrics(
        path=path,
        step=10,
        tokens_seen=163_840,
        train_loss=8.5,
        val_loss=8.7,
        learning_rate=3e-4,
        tokens_per_second=5000.0,
        elapsed_seconds=30.0,
    )

    assert path.exists()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1
    assert rows[0]["step"] == "10"


def test_log_metrics_appends_rows(
    tmp_path: Path,
):
    path = tmp_path / "training.csv"

    for step in [10, 20]:
        log_metrics(
            path=path,
            step=step,
            tokens_seen=step * 100,
            train_loss=8.0,
            val_loss=8.1,
            learning_rate=3e-4,
            tokens_per_second=5000.0,
            elapsed_seconds=30.0,
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 2