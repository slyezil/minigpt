import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.benchmark import (
    benchmark_naive_generation,
    print_benchmark,
)

from src.generate import load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/checkpoint-final.pt"
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "tokenizer/tokenizer.json"
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is unavailable."
        )

    device = torch.device("xpu")

    print()
    print("Loading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(args.tokenizer)
    )

    print("Loading model...")

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    encoded = tokenizer.encode(
        args.prompt
    )

    prompt_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    print()
    print("=" * 60)
    print("MiniGPT Baseline V0")
    print("=" * 60)

    print(
        f"Device:          "
        f"{torch.xpu.get_device_name(0)}"
    )

    print(
        f"Parameters:      "
        f"{parameter_count:,}"
    )

    print(
        f"Prompt:          "
        f"{args.prompt!r}"
    )

    print(
        f"Prompt tokens:   "
        f"{len(encoded.ids)}"
    )

    print(
        f"Generate tokens: "
        f"{args.max_new_tokens}"
    )

    print("=" * 60)

    result = benchmark_naive_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=3,
        use_autocast=True,
    )

    print_benchmark(result)

    print()
    print("Per-token latency:")
    print("-" * 60)

    for index, latency in enumerate(
        result.token_latencies_ms,
        start=1,
    ):
        print(
            f"Token {index:02d}: "
            f"{latency:8.2f} ms"
        )
    generated_text = tokenizer.decode(
        result.output_ids[0].tolist(),
        skip_special_tokens=True,
    )

    print("Generated text:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


if __name__ == "__main__":
    main()