"""P0/P1 data pipeline guardrails — split sizes, nonempty splits."""

import random

import tiktoken

from minigpt.data import prepare_agnews_data, prepare_data


def _tokenizer():
    return tiktoken.get_encoding("gpt2")


class TestSplitSizes:
    """P0/P1: Splits must be nonempty and sum to total tokens."""

    def test_prepare_agnews_data_produces_nonempty_splits(self):
        samples = [
            (1, "World news headline A", "Description of world event A"),
            (1, "World news headline B", "Description of world event B"),
            (2, "Sports headline A", "Description of sports event A"),
            (2, "Sports headline B", "Description of sports event B"),
            (3, "Business headline A", "Description of business event A"),
            (3, "Business headline B", "Description of business event B"),
            (4, "SciTech headline A", "Description of science event A"),
            (4, "SciTech headline B", "Description of science event B"),
        ]
        enc = _tokenizer()
        result = prepare_agnews_data(
            samples, enc,
            id_categories=[1, 2], ood_categories=[3, 4],
            val_fraction=0.1, test_fraction=0.1, seed=42,
        )
        assert len(result["train"]) > 0
        assert len(result["val"]) > 0
        assert len(result["test_id"]) > 0
        assert result["test_ood"] is not None
        assert len(result["test_ood"]) > 0

    def test_shakespeare_splits_sum_to_total(self):
        enc = _tokenizer()
        text = "Hello world. " * 500  # ~1500 tokens
        result = prepare_data(text, enc, val_fraction=0.1, test_fraction=0.1)

        total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        all_tokens = enc.encode_ordinary(text)
        assert total == len(all_tokens), (
            f"Split sum ({total}) != total tokens ({len(all_tokens)})"
        )
        assert result["test_ood"] is None

    def test_agnews_id_splits_sum_to_id_total(self):
        enc = _tokenizer()
        samples = [
            (1, f"Title {i}", f"Description for article number {i}") for i in range(100)
        ] + [
            (3, f"OOD Title {i}", f"OOD description for article number {i}") for i in range(50)
        ]
        result = prepare_agnews_data(
            samples, enc,
            id_categories=[1], ood_categories=[3],
            val_fraction=0.1, test_fraction=0.1, seed=42,
        )

        id_total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        # Reconstruct ID tokens to verify
        id_articles = [f"{t} {d}" for cat, t, d in samples if cat == 1]
        rng = random.Random(42)
        rng.shuffle(id_articles)
        id_tokens = enc.encode_ordinary("\n\n".join(id_articles))
        assert id_total == len(id_tokens), (
            f"ID split sum ({id_total}) != ID total tokens ({len(id_tokens)})"
        )
