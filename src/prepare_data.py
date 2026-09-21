import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from src.dataset import write_token_file
from src.tokenizer import train_tokenizer


DATASET_NAME = "roneneldan/TinyStories"


def iter_texts(
    dataset: Iterable[dict],
    limit: int | None,
) -> Iterator[str]:
    count = 0

    for row in dataset:
        text = row.get("text")

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        yield text

        count += 1

        if (
            limit is not None
            and count >= limit
        ):
            break


def load_tinystories_split(
    split: str,
):
    return load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )


def train_and_save_tokenizer(
    output_path: Path,
    vocab_size: int,
    document_limit: int,
) -> Tokenizer:
    print(
        f"Training tokenizer on up to "
        f"{document_limit:,} TinyStories documents..."
    )

    dataset = load_tinystories_split(
        "train"
    )

    texts = iter_texts(
        dataset,
        limit=document_limit,
    )

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=vocab_size,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(output_path)
    )

    print(
        f"Tokenizer saved to: {output_path}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    return tokenizer


def load_saved_tokenizer(
    path: Path,
) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {path}"
        )

    return Tokenizer.from_file(
        str(path)
    )


def prepare_split(
    tokenizer: Tokenizer,
    split: str,
    output_path: Path,
    document_limit: int,
) -> int:
    print(
        f"Preparing TinyStories '{split}' "
        f"split..."
    )

    dataset = load_tinystories_split(
        split
    )

    texts = iter_texts(
        dataset,
        limit=document_limit,
    )

    token_count = write_token_file(
        texts=texts,
        tokenizer=tokenizer,
        output_path=output_path,
    )

    print(
        f"{split}: wrote "
        f"{token_count:,} tokens "
        f"to {output_path}"
    )

    return token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TinyStories for MiniGPT."
        )
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16_000,
    )

    parser.add_argument(
        "--tokenizer-docs",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--train-docs",
        type=int,
        default=250_000,
    )

    parser.add_argument(
        "--validation-docs",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(
            "tokenizer/tokenizer.json"
        ),
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(
            "data/train.bin"
        ),
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path(
            "data/validation.bin"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = train_and_save_tokenizer(
        output_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        document_limit=args.tokenizer_docs,
    )

    train_tokens = prepare_split(
        tokenizer=tokenizer,
        split="train",
        output_path=args.train_path,
        document_limit=args.train_docs,
    )

    validation_tokens = prepare_split(
        tokenizer=tokenizer,
        split="validation",
        output_path=args.validation_path,
        document_limit=args.validation_docs,
    )

    print()
    print("Preparation complete.")
    print(
        f"Train tokens:      "
        f"{train_tokens:,}"
    )
    print(
        f"Validation tokens: "
        f"{validation_tokens:,}"
    )


if __name__ == "__main__":
    main()