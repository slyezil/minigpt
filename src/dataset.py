from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


TOKEN_DTYPE = np.uint16


def write_token_file(
    texts: Iterable[str],
    tokenizer: Tokenizer,
    output_path: str | Path,
) -> int:
    """
    Tokenize documents and write the resulting token IDs
    to a compact uint16 binary file.

    An <eos> token is appended after every document.

    Returns the number of token IDs written.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    if eos_id is None:
        raise ValueError(
            "Tokenizer does not contain an <eos> token."
        )

    token_count = 0

    with output_path.open("wb") as file:
        for text in texts:
            encoded = tokenizer.encode(text)

            token_ids = (
                encoded.ids
                + [eos_id]
            )

            if token_ids:
                max_token_id = max(token_ids)

                if max_token_id > np.iinfo(
                    TOKEN_DTYPE
                ).max:
                    raise ValueError(
                        f"Token ID {max_token_id} "
                        "does not fit in uint16."
                    )

            array = np.asarray(
                token_ids,
                dtype=TOKEN_DTYPE,
            )

            array.tofile(file)

            token_count += len(array)

    return token_count


class TokenDataset:
    def __init__(
        self,
        path: str | Path,
        context_length: int,
    ):
        self.path = Path(path)
        self.context_length = context_length

        if not self.path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.path}"
            )

        self.data = np.memmap(
            self.path,
            dtype=TOKEN_DTYPE,
            mode="r",
        )

        self.token_count = len(self.data)

        minimum_tokens = context_length + 1

        if self.token_count < minimum_tokens:
            raise ValueError(
                f"Dataset contains {self.token_count} tokens, "
                f"but at least {minimum_tokens} are required."
            )

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        max_start = (
            self.token_count
            - self.context_length
            - 1
        )

        starts = np.random.randint(
            low=0,
            high=max_start + 1,
            size=batch_size,
        )

        inputs = np.stack(
            [
                self.data[
                    start:
                    start + self.context_length
                ]
                for start in starts
            ]
        )

        targets = np.stack(
            [
                self.data[
                    start + 1:
                    start + self.context_length + 1
                ]
                for start in starts
            ]
        )

        inputs = torch.from_numpy(
            inputs.astype(
                np.int64,
                copy=True,
            )
        )

        targets = torch.from_numpy(
            targets.astype(
                np.int64,
                copy=True,
            )
        )

        return (
            inputs.to(device),
            targets.to(device),
        )