import argparse
import time

import torch

from src.config import ModelConfig
from src.dataset import TokenDataset
from src.model import MiniGPT
from src.train import next_token_loss


def benchmark_batch_size(
    model: MiniGPT,
    dataset: TokenDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    context_length: int,
    warmup_steps: int = 2,
    benchmark_steps: int = 5,
) -> float:
    model.train()

    total_steps = warmup_steps + benchmark_steps
    measured_time = 0.0

    for step in range(total_steps):
        inputs, targets = dataset.get_batch(
            batch_size=batch_size,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        if step >= warmup_steps:
            torch.xpu.synchronize()
            start = time.perf_counter()

        # BF16 compute on the Intel Arc GPU.
        with torch.autocast(
            device_type="xpu",
            dtype=torch.bfloat16,
        ):
            logits = model(inputs)

            loss = next_token_loss(
                logits,
                targets,
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step >= warmup_steps:
            torch.xpu.synchronize()

            measured_time += (
                time.perf_counter() - start
            )

    tokens_processed = (
        benchmark_steps
        * batch_size
        * context_length
    )

    tokens_per_second = (
        tokens_processed / measured_time
    )

    return tokens_per_second


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/train.bin",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is not available."
        )

    device = torch.device("xpu")

    config = ModelConfig()

    print()
    print("MiniGPT Arc Benchmark")
    print("=====================")
    print(
        f"Device:        "
        f"{torch.xpu.get_device_name(0)}"
    )
    print(
        f"Parameters:    "
        f"{config.estimated_parameter_count:,}"
    )
    print(
        f"Context:       "
        f"{config.context_length}"
    )
    print(
        f"Precision:     BF16 autocast"
    )
    print()

    dataset = TokenDataset(
        path=args.data,
        context_length=config.context_length,
    )

    results = []

    for batch_size in args.batch_sizes:
        print(
            f"Testing batch size "
            f"{batch_size}..."
        )

        try:
            # Create a fresh model for every batch-size test
            # so each run starts under comparable conditions.
            model = MiniGPT(
                config
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                weight_decay=0.1,
            )

            tokens_per_second = (
                benchmark_batch_size(
                    model=model,
                    dataset=dataset,
                    optimizer=optimizer,
                    device=device,
                    batch_size=batch_size,
                    context_length=(
                        config.context_length
                    ),
                    warmup_steps=(
                        args.warmup_steps
                    ),
                    benchmark_steps=(
                        args.benchmark_steps
                    ),
                )
            )

            results.append(
                (
                    batch_size,
                    tokens_per_second,
                )
            )

            print(
                f"  {tokens_per_second:,.0f} "
                f"tokens/sec"
            )

            del optimizer
            del model

            torch.xpu.empty_cache()

        except RuntimeError as exc:
            message = str(exc).lower()

            if (
                "out of memory" in message
                or "memory" in message
            ):
                print(
                    "  OUT OF MEMORY"
                )

                torch.xpu.empty_cache()

                break

            raise

    print()
    print("Results")
    print("=======")

    for (
        batch_size,
        tokens_per_second,
    ) in results:
        tokens_in_8_hours = (
            tokens_per_second
            * 60
            * 60
            * 8
        )

        print(
            f"Batch {batch_size:>2}: "
            f"{tokens_per_second:>8,.0f} "
            f"tok/s | "
            f"~{tokens_in_8_hours / 1_000_000:,.1f}M "
            f"tokens / 8h"
        )


if __name__ == "__main__":
    main()