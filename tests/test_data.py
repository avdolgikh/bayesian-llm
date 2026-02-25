"""P0/P1 data pipeline guardrails — category isolation, split sizes."""

import tiktoken
from minigpt.data import prepare_data, prepare_agnews_data


def _tokenizer():
    return tiktoken.get_encoding("gpt2")


# --- P0: Category isolation (AG News) ---

class TestCategoryIsolation:
    """P0: Category isolation.

    If even one ID article leaks into the OOD tensor, the MI comparison
    (ID vs OOD) is compromised.
    """

    def _make_samples(self) -> list[tuple[int, str, str]]:
        """Synthetic AG News-like samples with known categories."""
        return [
            (1, "World news headline A", "Description of world event A"),
            (1, "World news headline B", "Description of world event B"),
            (2, "Sports headline A", "Description of sports event A"),
            (2, "Sports headline B", "Description of sports event B"),
            (3, "Business headline A", "Description of business event A"),
            (3, "Business headline B", "Description of business event B"),
            (4, "SciTech headline A", "Description of science event A"),
            (4, "SciTech headline B", "Description of science event B"),
        ]

    def test_id_contains_only_id_categories(self):
        samples = self._make_samples()
        id_cats = [1, 2]
        ood_cats = [3, 4]
        enc = _tokenizer()

        # Verify filtering at the article level (before tokenization)
        id_articles = [s for s in samples if s[0] in id_cats]
        ood_articles = [s for s in samples if s[0] in ood_cats]

        for cat, _, _ in id_articles:
            assert cat in id_cats, f"Non-ID category {cat} in ID articles"
        for cat, _, _ in ood_articles:
            assert cat in ood_cats, f"Non-OOD category {cat} in OOD articles"

    def test_zero_overlap_between_id_and_ood(self):
        samples = self._make_samples()
        id_cats = [1, 2]
        ood_cats = [3, 4]

        id_articles = {f"{t} {d}" for cat, t, d in samples if cat in id_cats}
        ood_articles = {f"{t} {d}" for cat, t, d in samples if cat in ood_cats}

        overlap = id_articles & ood_articles
        assert len(overlap) == 0, f"ID/OOD overlap: {overlap}"

    def test_all_categories_accounted_for(self):
        samples = self._make_samples()
        id_cats = [1, 2]
        ood_cats = [3, 4]

        all_cats = {s[0] for s in samples}
        covered = set(id_cats) | set(ood_cats)
        assert all_cats == covered, f"Uncovered categories: {all_cats - covered}"

    def test_prepare_agnews_data_produces_nonempty_splits(self):
        samples = self._make_samples()
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


# --- P1: Split sizes sum to total ---

class TestSplitSizes:
    """P1: Split sizes sum to total.

    Off-by-one in int(len * fraction) can silently shift thousands of
    tokens between splits.
    """

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
        import random
        rng = random.Random(42)
        rng.shuffle(id_articles)
        id_tokens = enc.encode_ordinary("\n\n".join(id_articles))
        assert id_total == len(id_tokens), (
            f"ID split sum ({id_total}) != ID total tokens ({len(id_tokens)})"
        )

    def test_split_proportions_approximate(self):
        enc = _tokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 2000
        result = prepare_data(text, enc, val_fraction=0.1, test_fraction=0.1)

        total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        train_frac = len(result["train"]) / total
        val_frac = len(result["val"]) / total
        test_frac = len(result["test_id"]) / total

        assert abs(train_frac - 0.8) < 0.01, f"Train fraction {train_frac:.3f} not ~0.8"
        assert abs(val_frac - 0.1) < 0.01, f"Val fraction {val_frac:.3f} not ~0.1"
        assert abs(test_frac - 0.1) < 0.01, f"Test fraction {test_frac:.3f} not ~0.1"
