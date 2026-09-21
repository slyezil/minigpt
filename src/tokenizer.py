from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


def build_tokenizer() -> Tokenizer:
    """
    Create an untrained byte-level BPE tokenizer.

    The tokenizer operates at the byte level before applying BPE,
    which means it can represent arbitrary UTF-8 text while still
    learning common subword units.
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
    )

    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 16_000,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer from an iterable of strings.

    Parameters
    ----------
    texts:
        Training text samples.

    vocab_size:
        Maximum number of vocabulary entries, including
        the special tokens.

    Returns
    -------
    Tokenizer
        A trained Hugging Face Tokenizers tokenizer.
    """

    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            "vocab_size must be greater than the number "
            f"of special tokens ({len(SPECIAL_TOKENS)})."
        )

    tokenizer = build_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
    )

    return tokenizer