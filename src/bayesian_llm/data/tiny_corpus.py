from __future__ import annotations

from dataclasses import dataclass


TINY_CORPUS = (
    "Bayesian methods help language models express uncertainty.\n"
    "A small transformer can be trained on a tiny corpus to validate the loop.\n"
    "We start with a deterministic baseline, then add Bayesian layers later.\n"
    "Uncertainty metrics include calibration error, NLL, Brier score, and OOD tests.\n"
    "Keep experiments small, fast, and easy to reproduce.\n"
)


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def build_tokenizer(text: str) -> CharTokenizer:
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = list(vocab)
    return CharTokenizer(stoi=stoi, itos=itos)


def load_corpus() -> str:
    return TINY_CORPUS
