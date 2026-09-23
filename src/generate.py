import argparse
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.config import ModelConfig
from src.model import MiniGPT


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> MiniGPT:
    config = ModelConfig()

    model = MiniGPT(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:

    if temperature <= 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    logits = logits / temperature

    if top_k > 0:
        top_k = min(
            top_k,
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            top_k,
        )

        threshold = values[:, -1:]

        logits = logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str, float]:

    config = model.config

    encoded = tokenizer.encode(prompt)

    token_ids = encoded.ids

    if not token_ids:
        raise ValueError(
            "Prompt produced no tokens."
        )

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    generated = 0

    torch.xpu.synchronize()
    start = time.perf_counter()

    for _ in range(max_new_tokens):

        # Our model supports at most 512 tokens.
        model_input = tokens[
            :,
            -config.context_length:
        ]

        with torch.autocast(
            device_type="xpu",
            dtype=torch.bfloat16,
        ):
            logits = model(model_input)

        # We only care about the prediction
        # at the final position.
        next_token_logits = logits[
            :,
            -1,
            :
        ]

        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

        generated += 1

        if (
            eos_id is not None
            and next_token.item() == eos_id
        ):
            break

    torch.xpu.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    output = tokenizer.decode(
        tokens[0].tolist(),
        skip_special_tokens=True,
    )

    tokens_per_second = (
        generated / elapsed
        if elapsed > 0
        else 0.0
    )

    return output, tokens_per_second


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
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is unavailable."
        )

    device = torch.device("xpu")

    tokenizer = Tokenizer.from_file(
        str(args.tokenizer)
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print()
    print(
        f"Device: {torch.xpu.get_device_name(0)}"
    )
    print(
        f"Prompt: {args.prompt!r}"
    )
    print()

    text, tokens_per_second = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(text)

    print()
    print(
        f"Generation speed: "
        f"{tokens_per_second:.1f} tok/s"
    )


if __name__ == "__main__":
    main()