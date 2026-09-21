from itertools import islice

from src.prepare_data import iter_texts


class FakeDataset:
    def __iter__(self):
        yield {"text": "Story one."}
        yield {"text": ""}
        yield {"text": "   "}
        yield {"text": "Story two."}
        yield {"text": None}
        yield {"text": "Story three."}


def test_iter_texts_skips_empty_documents():
    texts = list(
        iter_texts(
            FakeDataset(),
            limit=None,
        )
    )

    assert texts == [
        "Story one.",
        "Story two.",
        "Story three.",
    ]


def test_iter_texts_respects_limit():
    texts = list(
        iter_texts(
            FakeDataset(),
            limit=2,
        )
    )

    assert texts == [
        "Story one.",
        "Story two.",
    ]