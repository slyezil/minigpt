import argparse
import csv
import math
import time

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.dataset import TokenDataset
from src.model import MiniGPT


# ============================================================
# Loss
# ============================================================


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.size(-1)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


# ============================================================
# Simple one-step training helper
# ============================================================


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

    return float(
        loss.detach().item()
    )


# ============================================================
# Mixed precision
# ============================================================


def _autocast_context(
    device: torch.device,
):
    if device.type == "xpu":
        return torch.autocast(
            device_type="xpu",
            dtype=torch.bfloat16,
        )

    return nullcontext()


# ============================================================
# Gradient-accumulated training step
# ============================================================


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


# ============================================================
# Validation
# ============================================================


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


# ============================================================
# Learning-rate schedule
# ============================================================


def get_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:

    # Linear warmup.
    if step < warmup_steps:
        return max_lr * (
            step / warmup_steps
        )

    # Training completed.
    if step >= total_steps:
        return min_lr

    # Cosine decay.
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


# ============================================================
# Checkpoints
# ============================================================


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


# ============================================================
# CSV metrics
# ============================================================


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


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MiniGPT-97M on TinyStories."
        )
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path(
            "data/train.bin"
        ),
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path(
            "data/validation.bin"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "checkpoints"
        ),
    )

    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path(
            "checkpoints/training.csv"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--total-tokens",
        type=int,
        default=150_000_000,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main training loop
# ============================================================


def main() -> None:
    args = parse_args()

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is not available."
        )

    device = torch.device("xpu")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.xpu.manual_seed_all(
        args.seed
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    config = ModelConfig()

    train_dataset = TokenDataset(
        path=args.train_data,
        context_length=(
            config.context_length
        ),
    )

    validation_dataset = TokenDataset(
        path=args.validation_data,
        context_length=(
            config.context_length
        ),
    )

    model = MiniGPT(
        config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(
            args.beta1,
            args.beta2,
        ),
        weight_decay=(
            args.weight_decay
        ),
    )

    # --------------------------------------------------------
    # Training-budget calculations
    # --------------------------------------------------------

    tokens_per_micro_batch = (
        args.batch_size
        * config.context_length
    )

    tokens_per_update = (
        tokens_per_micro_batch
        * args.gradient_accumulation
    )

    total_steps = math.ceil(
        args.total_tokens
        / tokens_per_update
    )

    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    start_step = 0
    tokens_seen = 0

    if args.resume is not None:
        print(
            f"Loading checkpoint: "
            f"{args.resume}"
        )

        state = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        start_step = state["step"]
        tokens_seen = state[
            "tokens_seen"
        ]

    # --------------------------------------------------------
    # Run information
    # --------------------------------------------------------

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "========================================"
    )
    print(
        "        MiniGPT-97M Training"
    )
    print(
        "========================================"
    )

    print(
        f"Device:               "
        f"{torch.xpu.get_device_name(0)}"
    )

    print(
        f"Parameters:           "
        f"{parameter_count:,}"
    )

    print(
        f"Training tokens:      "
        f"{train_dataset.token_count:,}"
    )

    print(
        f"Validation tokens:    "
        f"{validation_dataset.token_count:,}"
    )

    print(
        f"Context length:       "
        f"{config.context_length}"
    )

    print(
        f"Micro batch size:     "
        f"{args.batch_size}"
    )

    print(
        f"Gradient accumulation:"
        f" {args.gradient_accumulation}"
    )

    print(
        f"Tokens/update:        "
        f"{tokens_per_update:,}"
    )

    print(
        f"Target tokens:        "
        f"{args.total_tokens:,}"
    )

    print(
        f"Optimizer updates:    "
        f"{total_steps:,}"
    )

    print(
        f"Starting step:        "
        f"{start_step:,}"
    )

    print(
        "Precision:            "
        "BF16 autocast"
    )

    print(
        "========================================"
    )

    if start_step >= total_steps:
        print(
            "Training target has already "
            "been reached."
        )
        return

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    print()
    print(
        "Running initial validation..."
    )

    initial_val_loss = evaluate(
        model=model,
        dataset=validation_dataset,
        device=device,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
    )

    print(
        f"Initial validation loss: "
        f"{initial_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Timers / rolling metrics
    # --------------------------------------------------------

    run_start_time = time.perf_counter()

    interval_start_time = (
        time.perf_counter()
    )

    interval_start_tokens = tokens_seen

    rolling_loss = 0.0
    rolling_steps = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    try:
        for step in range(
            start_step + 1,
            total_steps + 1,
        ):
            learning_rate = (
                get_learning_rate(
                    step=step,
                    warmup_steps=(
                        args.warmup_steps
                    ),
                    total_steps=total_steps,
                    max_lr=args.max_lr,
                    min_lr=args.min_lr,
                )
            )

            for param_group in (
                optimizer.param_groups
            ):
                param_group[
                    "lr"
                ] = learning_rate

            train_loss = (
                train_accumulated_step(
                    model=model,
                    optimizer=optimizer,
                    dataset=train_dataset,
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                    gradient_accumulation_steps=(
                        args.gradient_accumulation
                    ),
                    max_grad_norm=(
                        args.max_grad_norm
                    ),
                )
            )

            tokens_seen += (
                tokens_per_update
            )

            rolling_loss += train_loss
            rolling_steps += 1

            # --------------------------------------------
            # Detect catastrophic numerical failure.
            # --------------------------------------------

            if not math.isfinite(
                train_loss
            ):
                emergency_path = (
                    args.checkpoint_dir
                    / (
                        "checkpoint-"
                        f"nonfinite-{step}.pt"
                    )
                )

                save_checkpoint(
                    path=emergency_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_seen=tokens_seen,
                )

                raise RuntimeError(
                    "Non-finite training loss "
                    f"detected at step {step}: "
                    f"{train_loss}"
                )

            # --------------------------------------------
            # Logging
            # --------------------------------------------

            should_log = (
                step
                % args.log_interval
                == 0
                or step == 1
            )

            should_evaluate = (
                step
                % args.eval_interval
                == 0
                or step == total_steps
            )

            val_loss = None

            if should_log:
                torch.xpu.synchronize()

                now = time.perf_counter()

                interval_tokens = (
                    tokens_seen
                    - interval_start_tokens
                )

                interval_seconds = (
                    now
                    - interval_start_time
                )

                tokens_per_second = (
                    interval_tokens
                    / interval_seconds
                )

                average_train_loss = (
                    rolling_loss
                    / rolling_steps
                )

            # --------------------------------------------
            # Validation
            # --------------------------------------------

            if should_evaluate:
                print(
                    f"\nEvaluating at "
                    f"step {step:,}..."
                )

                val_loss = evaluate(
                    model=model,
                    dataset=(
                        validation_dataset
                    ),
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                    eval_batches=(
                        args.eval_batches
                    ),
                )

            # --------------------------------------------
            # Print + CSV
            # --------------------------------------------

            if should_log:
                elapsed_seconds = (
                    time.perf_counter()
                    - run_start_time
                )

                log_metrics(
                    path=args.metrics_path,
                    step=step,
                    tokens_seen=(
                        tokens_seen
                    ),
                    train_loss=(
                        average_train_loss
                    ),
                    val_loss=val_loss,
                    learning_rate=(
                        learning_rate
                    ),
                    tokens_per_second=(
                        tokens_per_second
                    ),
                    elapsed_seconds=(
                        elapsed_seconds
                    ),
                )

                message = (
                    f"step {step:>5,}"
                    f"/{total_steps:,}"
                    f" | tokens "
                    f"{tokens_seen / 1e6:>7.2f}M"
                    f" | loss "
                    f"{average_train_loss:>7.4f}"
                    f" | lr "
                    f"{learning_rate:.2e}"
                    f" | "
                    f"{tokens_per_second:>7,.0f}"
                    f" tok/s"
                )

                if val_loss is not None:
                    message += (
                        f" | val "
                        f"{val_loss:.4f}"
                    )

                print(message)

                rolling_loss = 0.0
                rolling_steps = 0

                interval_start_tokens = (
                    tokens_seen
                )

                interval_start_time = (
                    time.perf_counter()
                )

            # --------------------------------------------
            # Checkpoint
            # --------------------------------------------

            if (
                step
                % args.checkpoint_interval
                == 0
            ):
                checkpoint_path = (
                    args.checkpoint_dir
                    / (
                        f"checkpoint-"
                        f"{step:05d}.pt"
                    )
                )

                print(
                    f"Saving checkpoint: "
                    f"{checkpoint_path}"
                )

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_seen=tokens_seen,
                )

                # Exclude checkpoint writing from
                # subsequent throughput calculation.
                interval_start_time = (
                    time.perf_counter()
                )

                interval_start_tokens = (
                    tokens_seen
                )

    except KeyboardInterrupt:
        print()
        print(
            "Training interrupted."
        )

        interrupted_path = (
            args.checkpoint_dir
            / "checkpoint-interrupted.pt"
        )

        print(
            "Saving emergency checkpoint "
            f"to {interrupted_path}"
        )

        save_checkpoint(
            path=interrupted_path,
            model=model,
            optimizer=optimizer,
            step=step,
            tokens_seen=tokens_seen,
        )

        raise

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    final_path = (
        args.checkpoint_dir
        / "checkpoint-final.pt"
    )

    print()
    print(
        f"Saving final checkpoint: "
        f"{final_path}"
    )

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        tokens_seen=tokens_seen,
    )

    total_elapsed = (
        time.perf_counter()
        - run_start_time
    )

    print()
    print(
        "========================================"
    )
    print(
        "Training complete"
    )
    print(
        "========================================"
    )

    print(
        f"Optimizer updates: "
        f"{total_steps:,}"
    )

    print(
        f"Tokens seen:       "
        f"{tokens_seen:,}"
    )

    print(
        f"Wall time:         "
        f"{total_elapsed / 3600:.2f} hours"
    )

    print(
        f"Final checkpoint:  "
        f"{final_path}"
    )

    print(
        f"Metrics:           "
        f"{args.metrics_path}"
    )


if __name__ == "__main__":
    main()