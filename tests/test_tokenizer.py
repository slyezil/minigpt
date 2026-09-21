from src.tokenizer import (
    SPECIAL_TOKENS,
    train_tokenizer,
)


def test_tokenizer_has_required_special_tokens():
    texts = [
        "Once upon a time there was a little cat.",
        "The cat lived in a small house.",
    ]

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)

        assert token_id is not None, (
            f"Special token {token!r} is missing "
            "from the tokenizer vocabulary."
        )


def test_tokenizer_can_encode_text():
    texts = [
        "Once upon a time there was a little cat.",
        "The little cat liked to play in the garden.",
        "One day the cat met a friendly dog.",
    ] * 20

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    encoded = tokenizer.encode(
        "The little cat liked to play."
    )

    assert len(encoded.ids) > 0

    assert all(
        isinstance(token_id, int)
        for token_id in encoded.ids
    )


def test_tokenizer_can_encode_and_decode_text():
    texts = [
        "Once upon a time there was a little cat.",
        "The little cat liked to play in the garden.",
        "One day the cat met a friendly dog.",
    ] * 20

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    text = "The little cat liked to play."

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(
        encoded.ids,
        skip_special_tokens=True,
    )

    assert len(encoded.ids) > 0

    assert "cat" in decoded.lower()

    assert "play" in decoded.lower()


def test_tokenizer_respects_requested_vocab_size():
    texts = [
        "Once upon a time there was a little girl.",
        "She lived in a small house near the forest.",
        "The forest was full of animals and trees.",
        "A friendly rabbit lived near the river.",
        "The little girl loved playing with the animals.",
        "One sunny morning she walked into the forest.",
    ] * 100

    requested_vocab_size = 100

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=requested_vocab_size,
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    assert actual_vocab_size <= requested_vocab_size


def test_tokenizer_produces_valid_token_ids():
    texts = [
        "The dragon lived on top of the mountain.",
        "A young boy visited the mountain.",
        "The dragon and the boy became friends.",
    ] * 50

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    encoded = tokenizer.encode(
        "The dragon became friends with the boy."
    )

    vocab_size = tokenizer.get_vocab_size()

    assert len(encoded.ids) > 0

    for token_id in encoded.ids:
        assert 0 <= token_id < vocab_size


def test_special_token_ids_are_unique():
    texts = [
        "Once upon a time there was a small village.",
        "Many people lived happily in the village.",
    ] * 20

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    special_token_ids = [
        tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    ]

    assert len(special_token_ids) == len(
        set(special_token_ids)
    )


def test_unknown_token_exists():
    texts = [
        "Once upon a time there was a cat.",
        "The cat liked sleeping near the window.",
    ] * 20

    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=100,
    )

    unk_id = tokenizer.token_to_id("<unk>")

    assert unk_id is not None