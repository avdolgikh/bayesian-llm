import csv
import os
import random
import urllib.request
from pathlib import Path

import tiktoken
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path("data")

AGNEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

CATEGORY_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_shakespeare() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        print(f"Downloading TinyShakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print(f"Done ({os.path.getsize(path):,} bytes).")
    return path.read_text(encoding="utf-8")


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def prepare_data(
    text: str,
    enc: tiktoken.Encoding,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, torch.Tensor | None]:
    tokens = enc.encode_ordinary(text)
    data = torch.tensor(tokens, dtype=torch.long)
    train_end = int(len(data) * (1 - val_fraction - test_fraction))
    val_end = int(len(data) * (1 - test_fraction))
    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test_id": data[val_end:],
        "test_ood": None,
    }


# --------------- AG News ---------------

def load_agnews() -> list[tuple[int, str, str]]:
    """Download AG News CSVs and return list of (category, title, description)."""
    agnews_dir = DATA_DIR / "agnews"
    agnews_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[int, str, str]] = []
    for split, url in AGNEWS_URLS.items():
        path = agnews_dir / f"{split}.csv"
        if not path.exists():
            print(f"Downloading AG News {split} to {path} ...")
            urllib.request.urlretrieve(url, path)
            print(f"Done ({os.path.getsize(path):,} bytes).")
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                category = int(row[0])
                title = row[1].strip()
                description = row[2].strip()
                samples.append((category, title, description))

    print(f"AG News: {len(samples):,} total samples across {len(AGNEWS_URLS)} splits")
    return samples


def prepare_agnews_data(
    samples: list[tuple[int, str, str]],
    tokenizer: tiktoken.Encoding,
    id_categories: list[int],
    ood_categories: list[int],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1337,
) -> dict[str, torch.Tensor | None]:
    """Filter by category, shuffle, tokenize, split into train/val/test_id/test_ood."""
    id_articles = [f"{title} {desc}" for cat, title, desc in samples
                   if cat in id_categories]
    ood_articles = [f"{title} {desc}" for cat, title, desc in samples
                    if cat in ood_categories]

    id_names = [CATEGORY_NAMES[c] for c in id_categories]
    ood_names = [CATEGORY_NAMES[c] for c in ood_categories]
    print(f"ID categories: {id_names} ({len(id_articles):,} articles)")
    print(f"OOD categories: {ood_names} ({len(ood_articles):,} articles)")

    # Shuffle ID articles (seeded for reproducibility)
    rng = random.Random(seed)
    rng.shuffle(id_articles)

    # Tokenize ID text
    id_text = "\n\n".join(id_articles)
    id_tokens = tokenizer.encode_ordinary(id_text)
    id_data = torch.tensor(id_tokens, dtype=torch.long)
    train_end = int(len(id_data) * (1 - val_fraction - test_fraction))
    val_end = int(len(id_data) * (1 - test_fraction))
    train_data = id_data[:train_end]
    val_data = id_data[train_end:val_end]
    test_id = id_data[val_end:]

    # Tokenize OOD text
    test_ood = None
    if ood_articles:
        rng_ood = random.Random(seed + 1)
        rng_ood.shuffle(ood_articles)
        ood_text = "\n\n".join(ood_articles)
        ood_tokens = tokenizer.encode_ordinary(ood_text)
        test_ood = torch.tensor(ood_tokens, dtype=torch.long)

    return {"train": train_data, "val": val_data, "test_id": test_id, "test_ood": test_ood}


# --------------- Dataset dispatcher ---------------

def load_dataset(cfg: dict, tokenizer: tiktoken.Encoding) -> dict[str, torch.Tensor | None]:
    """Load and prepare dataset based on config. Returns dict with train/val/test_id/test_ood tensors."""
    dataset = cfg["data"]["dataset"]
    val_fraction = cfg["data"].get("val_fraction", 0.1)
    test_fraction = cfg["data"].get("test_fraction", 0.1)

    if dataset == "tinyshakespeare":
        text = load_shakespeare()
        return prepare_data(text, tokenizer, val_fraction, test_fraction)

    elif dataset == "agnews":
        samples = load_agnews()
        id_cats = cfg["data"].get("id_categories", [1, 2])
        ood_cats = cfg["data"].get("ood_categories", [3, 4])
        seed = cfg["train"].get("seed", 1337)
        return prepare_agnews_data(samples, tokenizer, id_cats, ood_cats,
                                   val_fraction, test_fraction, seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'tinyshakespeare' or 'agnews'.")
