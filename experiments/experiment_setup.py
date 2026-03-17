"""Shared experiment setup: CLI parsing, config, data, model, device."""

import argparse

import torch

from minigpt.config import (
    DEFAULT_CONFIG,
    apply_overrides,
    build_gpt_config,
    deep_merge,
    load_yaml,
    validate_config,
)
from minigpt.data import get_tokenizer, load_dataset
from minigpt.model import MiniGPT


def parse_base_args(description, add_extra_args_fn=None):
    """Parse CLI args (--config, --set, --no-mlflow) and build config dict.

    Args:
        add_extra_args_fn: optional callable(parser) to add extra arguments.

    Returns (args, cfg).
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="key=value",
    )
    ap.add_argument("--no-mlflow", action="store_true")
    if add_extra_args_fn:
        add_extra_args_fn(ap)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg = deep_merge(cfg, load_yaml(args.config))
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    validate_config(cfg)
    torch.manual_seed(cfg["train"]["seed"])

    return args, cfg


def setup_data(cfg):
    """Load tokenizer + dataset, print summary. Returns (tokenizer, data dict)."""
    tokenizer = get_tokenizer()
    data = load_dataset(cfg, tokenizer)
    print(f"BPE vocab size: {tokenizer.n_vocab}")
    print(f"Train tokens: {len(data['train']):,}  Val tokens: {len(data['val']):,}")
    print(f"Test ID tokens: {len(data['test_id']):,}")
    if "test_ood" in data and data["test_ood"] is not None:
        print(f"Test OOD tokens: {len(data['test_ood']):,}")
    else:
        ood_keys = [k for k in data if k.startswith("test_ood_")]
        for k in sorted(ood_keys):
            print(f"{k} tokens: {len(data[k]):,}")
    return tokenizer, data


def setup_model(cfg, tokenizer):
    """Create model. Returns (model, gpt_config, n_params)."""
    gpt_config = build_gpt_config(cfg, vocab_size=tokenizer.n_vocab)
    model = MiniGPT(gpt_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    return model, gpt_config, n_params


def resolve_device(cfg):
    """Resolve 'auto' to actual device. Returns torch.device."""
    device_str = cfg["train"]["device"]
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    return torch.device(device_str)
