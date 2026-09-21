import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
import math
import csv


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.size(-1)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(inputs)

    loss = next_token_loss(
        logits,
        targets,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )

    optimizer.step()

    return float(loss.detach().item())


def get_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:

    # -----------------------------
    # Linear warmup
    # -----------------------------
    if step < warmup_steps:
        return max_lr * (
            step / warmup_steps
        )

    # -----------------------------
    # Training finished
    # -----------------------------
    if step >= total_steps:
        return min_lr

    # -----------------------------
    # Cosine decay
    # -----------------------------
    decay_ratio = (
        step - warmup_steps
    ) / (
        total_steps - warmup_steps
    )

    coefficient = 0.5 * (
        1.0
        + math.cos(
            math.pi * decay_ratio
        )
    )

    return (
        min_lr
        + coefficient
        * (max_lr - min_lr)
    )

def _autocast_context(
    device: torch.device,
):
    if device.type == "xpu":
        return torch.autocast(
            device_type="xpu",
            dtype=torch.bfloat16,
        )

    return nullcontext()

def train_accumulated_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    device: torch.device,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = 0.0

    for _ in range(
        gradient_accumulation_steps
    ):
        inputs, targets = dataset.get_batch(
            batch_size=batch_size,
            device=device,
        )

        with _autocast_context(device):
            logits = model(inputs)

            loss = next_token_loss(
                logits,
                targets,
            )

        accumulated_loss += float(
            loss.detach().item()
        )

        # Critical:
        #
        # We divide by accumulation count so that
        # four accumulated micro-batches don't make
        # the gradients four times larger.
        scaled_loss = (
            loss
            / gradient_accumulation_steps
        )

        scaled_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )

    optimizer.step()

    return (
        accumulated_loss
        / gradient_accumulation_steps
    )

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
) -> float:
    model.eval()

    total_loss = 0.0

    for _ in range(eval_batches):
        inputs, targets = dataset.get_batch(
            batch_size=batch_size,
            device=device,
        )

        with _autocast_context(device):
            logits = model(inputs)

            loss = next_token_loss(
                logits,
                targets,
            )

        total_loss += float(
            loss.item()
        )

    return total_loss / eval_batches

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    return {
        "step": checkpoint["step"],
        "tokens_seen": (
            checkpoint["tokens_seen"]
        ),
    }

def log_metrics(
    path: str | Path,
    step: int,
    tokens_seen: int,
    train_loss: float,
    val_loss: float | None,
    learning_rate: float,
    tokens_per_second: float,
    elapsed_seconds: float,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = path.exists()

    fieldnames = [
        "step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "learning_rate",
        "tokens_per_second",
        "elapsed_seconds",
    ]

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": train_loss,
                "val_loss": (
                    ""
                    if val_loss is None
                    else val_loss
                ),
                "learning_rate": (
                    learning_rate
                ),
                "tokens_per_second": (
                    tokens_per_second
                ),
                "elapsed_seconds": (
                    elapsed_seconds
                ),
            }
        )