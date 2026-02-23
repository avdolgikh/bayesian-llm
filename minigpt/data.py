import os
import urllib.request
from pathlib import Path

import tiktoken
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path("data")


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
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = enc.encode_ordinary(text)
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]
