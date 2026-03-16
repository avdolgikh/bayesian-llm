import copy
import random
import sys
import types

import pytest
import torch

import minigpt.data as data_module
from minigpt.config import DEFAULT_CONFIG, validate_config
from minigpt.data import PILE_DOMAIN_NAMES


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [sum(ord(ch) for ch in token) % 10007 for token in text.split()]

    def encode_ordinary(self, text: str) -> list[int]:
        self.calls.append(text)
        return self._encode(text)


def make_pile_config(
    id_domains: list[str] | None = None,
    ood_domains: list[str] | None = None,
    id_tokens: int = 120,
    ood_tokens: int = 45,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1337,
) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"].update(
        {
            "dataset": "pile",
            "pile_id_domains": id_domains if id_domains is not None else ["wikipedia_en"],
            "pile_ood_domains": ood_domains if ood_domains is not None else ["arxiv"],
            "pile_id_tokens": id_tokens,
            "pile_ood_tokens": ood_tokens,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
        }
    )
    cfg["train"].update({"seed": seed, "device": "cpu"})
    return cfg


def make_partial_pile_config(data_overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"]["dataset"] = "pile"
    if data_overrides:
        cfg["data"].update(data_overrides)
    return cfg


def fake_pile_docs(
    domain_name: str, n_docs: int = 6, words_per_doc: int = 24,
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for doc_idx in range(n_docs):
        words = [
            f"{domain_name}_doc{doc_idx}_word{w}" for w in range(words_per_doc)
        ]
        parts = [f"{domain_name}_header_{doc_idx}", *words,
                 f"{domain_name}_tail_{doc_idx}"]
        text = " ".join(parts)
        docs.append({"text": text})
    return docs


def expected_domain_tokens(domain_name: str, seed: int, limit: int) -> torch.Tensor:
    docs = [doc["text"] for doc in fake_pile_docs(domain_name)]
    rng = random.Random(seed)
    rng.shuffle(docs)
    tokens: list[int] = []
    for text in docs:
        tokens.extend(RecordingTokenizer._encode(text))
        if len(tokens) >= limit:
            break
    return torch.tensor(tokens[:limit], dtype=torch.long)


def expected_id_splits(
    id_domains: list[str],
    seed: int,
    id_tokens: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    display_names = [PILE_DOMAIN_NAMES[d] for d in id_domains]
    tensors = [expected_domain_tokens(name, seed, id_tokens) for name in display_names]
    all_tokens = torch.cat(tensors)
    gen = torch.Generator()
    gen.manual_seed(seed)
    all_tokens = all_tokens[torch.randperm(len(all_tokens), generator=gen)]
    train_end = int(len(all_tokens) * (1 - val_fraction - test_fraction))
    val_end = int(len(all_tokens) * (1 - test_fraction))
    return all_tokens[:train_end], all_tokens[train_end:val_end], all_tokens[val_end:]


@pytest.fixture
def tokenizer() -> RecordingTokenizer:
    return RecordingTokenizer()


@pytest.fixture
def pile_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def mock_pile_dataset(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        calls.append(
            {"path": path, "name": name, "split": split, "streaming": streaming}
        )
        docs = fake_pile_docs(name or "unknown")
        return iter(docs) if streaming else docs

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return calls


class TestPileDomainLoading:
    def test_pile_requests_huggingface_streaming_for_each_domain(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange"],
            ood_domains=["arxiv", "freelaw"],
            id_tokens=60,
            ood_tokens=30,
        )

        data_module.load_pile_data(cfg, tokenizer)

        assert mock_pile_dataset == [
            {
                "path": "ArmelR/the-pile-splitted",
                "name": "Wikipedia (en)",
                "split": "train",
                "streaming": True,
            },
            {
                "path": "ArmelR/the-pile-splitted",
                "name": "StackExchange",
                "split": "train",
                "streaming": True,
            },
            {
                "path": "ArmelR/the-pile-splitted",
                "name": "ArXiv",
                "split": "train",
                "streaming": True,
            },
            {
                "path": "ArmelR/the-pile-splitted",
                "name": "FreeLaw",
                "split": "train",
                "streaming": True,
            },
        ]

    def test_pile_tokenizes_each_document_with_the_passed_tokenizer(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(id_domains=["wikipedia_en"], ood_domains=["arxiv"], id_tokens=50)

        data_module.load_pile_data(cfg, tokenizer)

        assert tokenizer.calls, "Expected encode_ordinary() to be called at least once"
        assert tokenizer.calls[0].startswith("Wikipedia (en)_header_")


class TestPileIdSplits:
    def test_pile_returns_required_id_keys(self, tokenizer, pile_data_dir, mock_pile_dataset):
        cfg = make_pile_config(id_tokens=70, ood_tokens=25)

        result = data_module.load_pile_data(cfg, tokenizer)

        assert set(result) >= {"train", "val", "test_id", "test_ood_arxiv"}
        assert result["train"].dtype == torch.long
        assert result["val"].dtype == torch.long
        assert result["test_id"].dtype == torch.long
        assert result["train"].ndim == 1
        assert result["val"].ndim == 1
        assert result["test_id"].ndim == 1
        assert len(result["train"]) > 0
        assert len(result["val"]) > 0
        assert len(result["test_id"]) > 0

    def test_pile_id_split_token_counts_match_target_with_rounding(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange"],
            id_tokens=55,
            val_fraction=0.2,
            test_fraction=0.2,
        )

        result = data_module.load_pile_data(cfg, tokenizer)

        total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        assert total == 110

    def test_pile_id_splits_follow_contiguous_domain_ordered_concatenation(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange"],
            ood_domains=["arxiv"],
            id_tokens=45,
            seed=2026,
            val_fraction=0.2,
            test_fraction=0.2,
        )

        result = data_module.load_pile_data(cfg, tokenizer)
        expected_train, expected_val, expected_test = expected_id_splits(
            cfg["data"]["pile_id_domains"],
            seed=2026,
            id_tokens=45,
            val_fraction=0.2,
            test_fraction=0.2,
        )

        assert torch.equal(result["train"], expected_train)
        assert torch.equal(result["val"], expected_val)
        assert torch.equal(result["test_id"], expected_test)


class TestPileOodTensors:
    @pytest.mark.parametrize(
        "ood_domains",
        [
            ["arxiv"],
            ["arxiv", "freelaw"],
            ["arxiv", "freelaw", "pubmed_abstracts"],
        ],
    )
    def test_pile_returns_one_test_key_per_ood_domain(
        self,
        ood_domains,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(ood_domains=ood_domains, id_tokens=50, ood_tokens=20)

        result = data_module.load_pile_data(cfg, tokenizer)

        expected_keys = {"train", "val", "test_id", *[f"test_ood_{name}" for name in ood_domains]}
        assert set(result) == expected_keys

    def test_pile_ood_tensors_match_expected_domain_tokens(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en"],
            ood_domains=["arxiv", "freelaw"],
            id_tokens=40,
            ood_tokens=25,
            seed=42,
        )

        result = data_module.load_pile_data(cfg, tokenizer)

        expected_arxiv = expected_domain_tokens("ArXiv", seed=42, limit=25)
        expected_freelaw = expected_domain_tokens("FreeLaw", seed=42, limit=25)
        assert torch.equal(result["test_ood_arxiv"], expected_arxiv)
        assert torch.equal(result["test_ood_freelaw"], expected_freelaw)


class TestPileTokenCountControl:
    def test_pile_id_domains_respect_id_token_limit(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange"],
            id_tokens=37,
            ood_tokens=15,
        )

        result = data_module.load_pile_data(cfg, tokenizer)

        assert len(result["train"]) + len(result["val"]) + len(result["test_id"]) == 74

    def test_pile_ood_domains_respect_ood_token_limit(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(ood_domains=["arxiv", "freelaw"], id_tokens=40, ood_tokens=17)

        result = data_module.load_pile_data(cfg, tokenizer)

        assert len(result["test_ood_arxiv"]) == 17
        assert len(result["test_ood_freelaw"]) == 17

    def test_pile_uses_all_available_tokens_when_domain_is_shorter_than_target(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
        monkeypatch,
    ):
        def short_dataset(path, name=None, split=None, streaming=False):
            docs = fake_pile_docs(name or "unknown", n_docs=1, words_per_doc=5)
            return iter(docs) if streaming else docs

        fake_module = types.SimpleNamespace(load_dataset=short_dataset)
        monkeypatch.setitem(sys.modules, "datasets", fake_module)
        cfg = make_pile_config(id_tokens=1000, ood_tokens=1000)

        result = data_module.load_pile_data(cfg, tokenizer)
        id_total = len(result["train"]) + len(result["val"]) + len(result["test_id"])

        assert id_total < 1000
        assert len(result["test_ood_arxiv"]) < 1000


class TestPileCaching:
    def test_pile_creates_cache_files_after_first_load(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange"],
            ood_domains=["arxiv"],
            id_tokens=33,
            ood_tokens=11,
        )

        data_module.load_pile_data(cfg, tokenizer)

        assert (pile_data_dir / "pile" / "wikipedia_en_33.pt").exists()
        assert (pile_data_dir / "pile" / "stackexchange_33.pt").exists()
        assert (pile_data_dir / "pile" / "arxiv_11.pt").exists()

    def test_pile_second_load_uses_cache_without_redownload(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
        monkeypatch,
    ):
        cfg = make_pile_config(id_tokens=44, ood_tokens=12)

        data_module.load_pile_data(cfg, tokenizer)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("datasets.load_dataset should not be called on cache hit")

        fake_module = types.SimpleNamespace(load_dataset=fail_if_called)
        monkeypatch.setitem(sys.modules, "datasets", fake_module)
        data_module.load_pile_data(cfg, tokenizer)

    def test_pile_different_id_token_target_invalidates_cache(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg_40 = make_pile_config(id_tokens=40, ood_tokens=10)
        cfg_55 = make_pile_config(id_tokens=55, ood_tokens=10)

        data_module.load_pile_data(cfg_40, tokenizer)
        data_module.load_pile_data(cfg_55, tokenizer)

        assert (pile_data_dir / "pile" / "wikipedia_en_40.pt").exists()
        assert (pile_data_dir / "pile" / "wikipedia_en_55.pt").exists()


class TestPileConfigSurface:
    def test_default_config_defines_pile_token_defaults(self):
        assert DEFAULT_CONFIG["data"]["pile_id_tokens"] == 100_000_000
        assert DEFAULT_CONFIG["data"]["pile_ood_tokens"] == 10_000_000

    def test_load_pile_data_uses_default_token_counts_when_keys_are_absent(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_partial_pile_config(
            {
                "pile_id_domains": ["wikipedia_en"],
                "pile_ood_domains": ["arxiv"],
                "pile_id_tokens": 50,
                "pile_ood_tokens": 20,
            }
        )
        # Remove explicit keys to test that the function falls back to defaults
        del cfg["data"]["pile_id_tokens"]
        del cfg["data"]["pile_ood_tokens"]

        result = data_module.load_pile_data(cfg, tokenizer)

        # With defaults (100M/10M) but mock has only ~156 tokens per domain,
        # function should use all available tokens (B-4: no error on short domain).
        id_total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        assert id_total > 0
        assert len(result["test_ood_arxiv"]) > 0

    def test_explicit_pile_token_counts_override_defaults(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(id_tokens=28, ood_tokens=9)

        result = data_module.load_pile_data(cfg, tokenizer)

        id_total = len(result["train"]) + len(result["val"]) + len(result["test_id"])
        assert id_total == 28
        assert len(result["test_ood_arxiv"]) == 9


class TestPileDispatcher:
    def test_load_dataset_dispatches_pile_to_load_pile_data(
        self, monkeypatch, tokenizer,
    ):
        cfg = make_pile_config()
        sentinel = {
            "train": torch.tensor([1]),
            "val": torch.tensor([2]),
            "test_id": torch.tensor([3]),
        }

        def fake_load_pile_data(arg_cfg, arg_tokenizer):
            assert arg_cfg is cfg
            assert arg_tokenizer is tokenizer
            return sentinel

        monkeypatch.setattr(data_module, "load_pile_data", fake_load_pile_data, raising=False)

        result = data_module.load_dataset(cfg, tokenizer)

        assert result is sentinel

    def test_load_dataset_keeps_existing_dataset_paths_after_pile_support(
        self, monkeypatch, tokenizer,
    ):
        calls: list[str] = []

        monkeypatch.setattr(data_module, "load_shakespeare", lambda: "hello world")
        monkeypatch.setattr(
            data_module,
            "prepare_data",
            lambda text, enc, val_fraction, test_fraction: calls.append("tiny") or {"tiny": True},
        )
        monkeypatch.setattr(data_module, "load_agnews", lambda: [("sample",)])
        monkeypatch.setattr(
            data_module,
            "prepare_agnews_data",
            lambda samples, enc, id_categories, ood_categories, val_fraction, test_fraction, seed: (
                calls.append("agnews") or {"agnews": True}
            ),
        )
        monkeypatch.setattr(
            data_module,
            "load_pile_data",
            lambda cfg, enc: calls.append("pile") or {"pile": True},
            raising=False,
        )

        tiny_cfg = {"data": {"dataset": "tinyshakespeare"}, "train": {}}
        assert data_module.load_dataset(tiny_cfg, tokenizer) == {
            "tiny": True
        }
        assert data_module.load_dataset(make_pile_config(), tokenizer) == {"pile": True}
        assert data_module.load_dataset(
            {"data": {"dataset": "agnews"}, "train": {"seed": 1337}},
            tokenizer,
        ) == {"agnews": True}
        assert calls == ["tiny", "pile", "agnews"]


class TestPileDomainNameMapping:
    def test_pile_accepts_supported_domain_names(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "stackexchange", "hackernews", "pile_cc"],
            ood_domains=["arxiv", "freelaw", "pubmed_abstracts", "gutenberg"],
            id_tokens=20,
            ood_tokens=10,
        )

        result = data_module.load_pile_data(cfg, tokenizer)

        assert "test_ood_arxiv" in result
        assert "test_ood_freelaw" in result
        assert "test_ood_pubmed_abstracts" in result
        assert "test_ood_gutenberg" in result

    def test_pile_invalid_domain_name_raises_helpful_value_error(self):
        cfg = make_pile_config(id_domains=["wikipedia_en"], ood_domains=["definitely_not_real"])

        with pytest.raises(ValueError, match="valid.*wikipedia_en.*pubmed_abstracts"):
            validate_config(cfg)


class TestPileReproducibility:
    def test_pile_same_seed_produces_identical_outputs(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg_a = make_pile_config(id_tokens=36, ood_tokens=14, seed=77)
        cfg_b = make_pile_config(id_tokens=36, ood_tokens=14, seed=77)

        result_a = data_module.load_pile_data(cfg_a, tokenizer)
        result_b = data_module.load_pile_data(cfg_b, tokenizer)

        assert torch.equal(result_a["train"], result_b["train"])
        assert torch.equal(result_a["val"], result_b["val"])
        assert torch.equal(result_a["test_id"], result_b["test_id"])
        assert torch.equal(result_a["test_ood_arxiv"], result_b["test_ood_arxiv"])

    def test_pile_different_seed_changes_id_split_order(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
    ):
        cfg_a = make_pile_config(id_tokens=36, ood_tokens=14, seed=11)
        cfg_b = make_pile_config(id_tokens=36, ood_tokens=14, seed=12)

        result_a = data_module.load_pile_data(cfg_a, tokenizer)
        result_b = data_module.load_pile_data(cfg_b, tokenizer)

        assert not torch.equal(result_a["train"], result_b["train"])


class TestPileProgressReporting:
    def test_pile_prints_domain_progress_and_summary(
        self,
        tokenizer,
        pile_data_dir,
        mock_pile_dataset,
        capsys,
    ):
        cfg = make_pile_config(id_tokens=25, ood_tokens=10)

        data_module.load_pile_data(cfg, tokenizer)
        captured = capsys.readouterr()

        assert "wikipedia_en" in captured.out
        assert "arxiv" in captured.out
        assert "tokens" in captured.out
        assert str(pile_data_dir / "pile") in captured.out


class TestPileValidation:
    def test_validate_config_rejects_empty_pile_id_domains(self):
        cfg = make_pile_config(id_domains=[], ood_domains=["arxiv"])

        with pytest.raises(ValueError, match="pile_id_domains"):
            validate_config(cfg)

    def test_validate_config_rejects_empty_pile_ood_domains(self):
        cfg = make_pile_config(id_domains=["wikipedia_en"], ood_domains=[])

        with pytest.raises(ValueError, match="pile_ood_domains"):
            validate_config(cfg)

    def test_validate_config_rejects_overlapping_pile_domains(self):
        cfg = make_pile_config(
            id_domains=["wikipedia_en", "arxiv"],
            ood_domains=["arxiv", "freelaw"],
        )

        with pytest.raises(ValueError, match="overlap|arxiv"):
            validate_config(cfg)

    def test_validate_config_rejects_non_positive_pile_id_tokens(self):
        cfg = make_pile_config(id_tokens=0, ood_tokens=10)

        with pytest.raises(ValueError, match="pile_id_tokens"):
            validate_config(cfg)

    def test_validate_config_rejects_non_positive_pile_ood_tokens(self):
        cfg = make_pile_config(id_tokens=10, ood_tokens=0)

        with pytest.raises(ValueError, match="pile_ood_tokens"):
            validate_config(cfg)

    def test_validate_config_accepts_valid_pile_config_with_defaults(self):
        cfg = make_partial_pile_config(
            {
                "pile_id_domains": ["wikipedia_en"],
                "pile_ood_domains": ["arxiv"],
            }
        )

        assert DEFAULT_CONFIG["data"]["pile_id_tokens"] == 100_000_000
        assert DEFAULT_CONFIG["data"]["pile_ood_tokens"] == 10_000_000
        validate_config(cfg)

