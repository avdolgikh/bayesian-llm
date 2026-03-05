"""YAML config ↔ dataclass bridge for miniGPT experiments."""

import copy
import json
from pathlib import Path

import yaml

from minigpt.layers import BayesConfig
from minigpt.model import GPTConfig
from minigpt.train import TrainConfig

DEFAULT_CONFIG: dict = {
    "experiment": {
        "name": "a0-baseline",
        "run_name": "a0-baseline",
        "mlflow_uri": "sqlite:///mlflow.db",
    },
    "data": {
        "dataset": "tinyshakespeare",
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "id_categories": [1, 2],
        "ood_categories": [3, 4],
    },
    "model": {
        "block_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "dropout": 0.2,
        "bias": True,
        "bayes_head": {
            "enabled": False,
            "prior_std": 1.0,
            "init_rho": -1.0,
        },
        "bayes_ffn": {
            "enabled": False,
            "prior_std": 1.0,
            "init_rho": -1.0,
        },
        "bayes_attn_v": {
            "enabled": False,
            "prior_std": 1.0,
            "init_rho": -1.0,
        },
    },
    "train": {
        "steps": 2000,
        "batch_size": 64,
        "block_size": 256,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "warmup_steps": 200,
        "min_lr": 1e-5,
        "grad_clip": 1.0,
        "eval_interval": 200,
        "eval_iters": 20,
        "checkpoint_interval": 500,
        "checkpoint_dir": "data/checkpoints",
        "gradient_accumulation_steps": 1,
        "kl_weight": 0.0,
        "kl_annealing_steps": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "seed": 1337,
        "device": "auto",
    },
    "eval": {
        "sample_tokens": 200,
        "temperature": 0.8,
        "num_samples": 30,
        "n_perplexity_batches": 20,
        "qualitative_prompts_per_category": 5,
        "qualitative_max_new_tokens": 100,
        "qualitative_seed": 42,
    },
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _coerce_type(value: str):
    """Best-effort cast from string to int / float / bool / None / JSON list."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    # JSON list/object (e.g. "[1,3]")
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides like ``train.lr=1e-3`` to *cfg* in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, val = item.split("=", 1)
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _coerce_type(val)
    return cfg


def _build_bayes_config(d: dict) -> BayesConfig:
    """Build a BayesConfig from a config sub-dict."""
    return BayesConfig(
        enabled=d.get("enabled", False),
        prior_std=d.get("prior_std", 1.0),
        init_rho=d.get("init_rho", -5.0),
    )


def build_gpt_config(cfg: dict, vocab_size: int) -> GPTConfig:
    """Construct a GPTConfig from the merged config dict.

    ``vocab_size`` comes from the tokenizer, never from the config file.
    """
    m = cfg["model"]
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
        bias=m["bias"],
        bayes_head=_build_bayes_config(m.get("bayes_head", {})),
        bayes_ffn=_build_bayes_config(m.get("bayes_ffn", {})),
        bayes_attn_v=_build_bayes_config(m.get("bayes_attn_v", {})),
    )


def build_train_config(cfg: dict) -> TrainConfig:
    t = cfg["train"]
    return TrainConfig(
        steps=t["steps"],
        batch_size=t["batch_size"],
        block_size=t["block_size"],
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        warmup_steps=t["warmup_steps"],
        min_lr=t["min_lr"],
        grad_clip=t["grad_clip"],
        eval_interval=t["eval_interval"],
        eval_iters=t["eval_iters"],
        checkpoint_interval=t["checkpoint_interval"],
        checkpoint_dir=t.get("checkpoint_dir", "data/checkpoints"),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        kl_annealing_steps=t.get("kl_annealing_steps", 0),
        adam_beta1=t.get("adam_beta1", 0.9),
        adam_beta2=t.get("adam_beta2", 0.95),
        seed=t["seed"],
        device=t["device"],
    )


def validate_config(cfg: dict) -> None:
    """Sanity-check the merged config. Raises ValueError on problems."""
    train_bs = cfg["train"]["block_size"]
    model_bs = cfg["model"]["block_size"]
    if train_bs > model_bs:
        raise ValueError(
            f"train.block_size ({train_bs}) must be <= model.block_size ({model_bs})"
        )
    if cfg["train"]["warmup_steps"] >= cfg["train"]["steps"]:
        raise ValueError("train.warmup_steps must be < train.steps")
    kl_weight = cfg["train"]["kl_weight"]
    if kl_weight < 0:
        raise ValueError(f"train.kl_weight must be >= 0, got {kl_weight}")
    bayes_enabled = any(
        cfg["model"][key]["enabled"] for key in ("bayes_head", "bayes_ffn", "bayes_attn_v")
    )
    if bayes_enabled and kl_weight <= 0:
        raise ValueError("train.kl_weight must be > 0 when any Bayesian component is enabled")
    n_embd = cfg["model"]["n_embd"]
    n_head = cfg["model"]["n_head"]
    if n_embd % n_head != 0:
        raise ValueError(f"model.n_embd ({n_embd}) must be divisible by model.n_head ({n_head})")
    val_frac = cfg["data"].get("val_fraction", 0.1)
    test_frac = cfg["data"].get("test_fraction", 0.1)
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_fraction ({val_frac}) + test_fraction ({test_frac}) must be < 1.0"
        )


def config_to_flat_params(cfg: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested config dict for MLflow ``log_params``."""
    flat: dict[str, str] = {}
    for key, val in cfg.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(config_to_flat_params(val, prefix=f"{full_key}."))
        else:
            flat[full_key] = str(val)
    return flat
