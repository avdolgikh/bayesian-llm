"""C-milestone-specific configuration and policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from minigpt.config import DEFAULT_CONFIG, apply_dict_overrides, deep_merge, validate_config

OOD_DOMAINS = ("arxiv", "freelaw", "pubmed_abstracts")
MAX_RUNS = 3
DEFAULT_RUN_ESTIMATE_HOURS = 6.0

C0_TEMPLATE = {
    "data": {
        "dataset": "pile",
        "pile_id_domains": ["wikipedia_en", "stackexchange"],
        "pile_ood_domains": list(OOD_DOMAINS),
    },
    "model": {
        "n_layer": 16,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.1,
        "bayes_head": {"enabled": False},
        "bayes_ffn": {"enabled": False},
        "bayes_attn_v": {"enabled": False},
    },
    "train": {
        "steps": 100_000,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "lr": 6e-4,
        "warmup_steps": 4_000,
        "eval_interval": 1_000,
        "checkpoint_interval": 5_000,
        "checkpoint_dir": "data/checkpoints/c0",
        "kl_weight": 0.0,
        "kl_annealing_steps": 0,
    },
}

C1_TEMPLATE = deep_merge(
    C0_TEMPLATE,
    {
        "model": {
            "bayes_ffn": {"enabled": True, "prior_std": 1.0, "init_rho": -2.0},
        },
        "train": {
            "checkpoint_dir": "data/checkpoints/c1",
            "kl_weight": 0.1,
            "kl_annealing_steps": 5_000,
        },
    },
)

C2_TEMPLATE = deep_merge(
    C0_TEMPLATE,
    {
        "posthoc_method": "laplace",
        "train": {"checkpoint_dir": "data/checkpoints/c2", "steps": 0},
        "laplace": {
            "selection_mode": "ffn",
            "damping": 1.0,
            "sample_scale": 1.0,
            "n_curvature_batches": 30,
        },
    },
)

C3_PHASE2_TEMPLATE = deep_merge(
    C0_TEMPLATE,
    {
        "data": {
            "dataset": "pile",
            "pile_id_domains": ["hackernews"],
            "pile_ood_domains": list(OOD_DOMAINS),
        },
        "lora": {
            "rank": 16,
            "alpha": 32.0,
            "target": "ffn",
            "prior_std": 0.2,
            "init_g": 0.1,
        },
        "train": {
            "steps": 10_000,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "checkpoint_dir": "data/checkpoints/c3",
            # LoRA fine-tuning HPs — NOT inherited from C0 pretraining:
            "lr": 3.0e-4,       # C0 had 6e-4 (pretraining); 3e-4 is standard for LoRA
            "warmup_steps": 500,  # C0 had 4000 (40% of 10K = too long); 5% is standard
            "weight_decay": 0.0,  # KL already regularizes; B2 used 0.0
            "eval_interval": 500,  # Shorter runs need more frequent eval
            # KL scaling: 1.31M Bayesian params / 80M train tokens = 0.016
            # vs B2's 163K / 2M = 0.082. kl_weight=1.0 matches B2's effective pressure.
            "kl_weight": 1.0,
            "kl_annealing_steps": 1_000,
        },
    },
)

C4_TFB_TEMPLATE = deep_merge(
    C3_PHASE2_TEMPLATE,
    {
        "posthoc_method": "tfb",
        "train": {
            "checkpoint_dir": "data/checkpoints/c4_tfb",
            "steps": 0,
            "kl_weight": 0.0,
            "kl_annealing_steps": 0,
        },
        "tfb": {"epsilon": 0.1, "n_search_samples": 10, "n_anchor_batches": 20},
    },
)

C4_LAP_TEMPLATE = deep_merge(
    C3_PHASE2_TEMPLATE,
    {
        "posthoc_method": "laplace",
        "train": {
            "checkpoint_dir": "data/checkpoints/c4_lap",
            "steps": 0,
            "kl_weight": 0.0,
            "kl_annealing_steps": 0,
        },
        "laplace": {
            "selection_mode": "lora",
            "damping": 1.0,
            "sample_scale": 1.0,
            "n_curvature_batches": 30,
        },
    },
)

MILESTONE_TEMPLATES = {
    "c0": C0_TEMPLATE,
    "c1": C1_TEMPLATE,
    "c2": C2_TEMPLATE,
    "c3_phase2": C3_PHASE2_TEMPLATE,
    "c4_tfb": C4_TFB_TEMPLATE,
    "c4_lap": C4_LAP_TEMPLATE,
}

TUNABLE_KNOBS = {
    "c0": {
        "train.lr": (1.0e-5, 1.0e-3),
        "train.steps": (50_000, 300_000),
        "train.warmup_steps": (500, 10_000),
        "model.dropout": (0.0, 0.5),
    },
    "c1": {
        "model.bayes_ffn.init_rho": (-5.0, 0.0),
        "model.bayes_ffn.prior_std": (0.1, 5.0),
        "train.kl_weight": (0.01, 2.0),
        "train.lr": (1.0e-5, 1.0e-3),
        "train.steps": (1, 300_000),
    },
    "c3": {
        "lora.rank": (4, 64),
        "lora.init_g": (0.01, 1.0),
        "lora.prior_std": (0.05, 2.0),
        "train.kl_weight": (0.01, 2.0),
        "train.lr": (1.0e-5, 1.0e-3),
    },
}

FOUR_L_RESULTS = {
    "Variational full": 1.43,
    "BLoB LoRA": 1.13,
    "TFB (post-hoc LoRA)": 1.10,
    "Laplace full": 1.00,
    "Laplace LoRA": 1.00,
}


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    apply_dict_overrides(updated, overrides or {})
    return updated


def build_milestone_config(milestone: str, overrides: dict | None = None) -> dict:
    if milestone not in MILESTONE_TEMPLATES:
        raise ValueError(f"Unknown milestone: {milestone}")
    cfg = deep_merge(DEFAULT_CONFIG, MILESTONE_TEMPLATES[milestone])
    cfg["experiment"]["name"] = milestone
    cfg["experiment"]["run_name"] = milestone
    cfg = _apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def check_gate(milestone: str, result: dict) -> bool:
    if milestone == "c0":
        test_id_ppl = result.get("test_id_ppl", float("inf"))
        test_ood_ppl = result.get("test_ood_ppl") or {}
        return test_id_ppl < 80 and any(value > 2 * test_id_ppl for value in test_ood_ppl.values())
    if milestone == "c1":
        return result.get("mi_ratio_mean", 0.0) > 1.2
    if milestone == "c3":
        return result.get("mi_ratio_mean", 0.0) > 1.05
    if milestone == "c4_tfb":
        return True
    return False


def parse_agent_response(raw: str) -> dict:
    import re

    def _extract(d: dict) -> dict:
        return {
            "diagnosis": d.get("diagnosis", ""),
            "reasoning": d.get("reasoning", ""),
            "adjustment": d.get("adjustment", {}),
        }

    # Strategy 1: raw is direct JSON with expected keys
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "diagnosis" in payload:
            return _extract(payload)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract fenced JSON from text (```json ... ```)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    for candidate in fenced:
        try:
            d = json.loads(candidate)
            if isinstance(d, dict) and "diagnosis" in d:
                return _extract(d)
        except json.JSONDecodeError:
            continue

    # Strategy 3: find balanced { } regions in raw text
    for m in re.finditer(r"\{", raw):
        depth, i = 0, m.start()
        for j in range(i, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(raw[i : j + 1])
                        if isinstance(d, dict) and "diagnosis" in d:
                            return _extract(d)
                    except json.JSONDecodeError:
                        pass
                    break

    return {"diagnosis": "", "reasoning": "", "adjustment": {}}


def gate_description_for(milestone: str) -> str:
    if milestone == "c0":
        return "test_id_ppl < 80 and any test_ood_ppl > 2 * test_id_ppl"
    if milestone == "c1":
        return "mi_ratio_mean > 1.2"
    if milestone == "c3":
        return "mi_ratio_mean > 1.05"
    return "record result"


def dependency_for(milestone: str) -> str | None:
    if milestone == "c2":
        return "c0"
    if milestone in {"c4_tfb", "c4_lap"}:
        return "c3"
    return None


def milestone_key_for(milestone: str) -> str:
    return "c3_phase2" if milestone == "c3" else milestone


def config_path_for(milestone_key: str, repo_root: Path) -> Path:
    """Return the conventional YAML config path for a milestone."""
    return repo_root / "configs" / f"{milestone_key}.yaml"


def knob_family_for(milestone: str) -> str:
    return "c3" if milestone == "c3" else milestone


def needs_phase1(milestone: str) -> bool:
    return milestone == "c3"


def needs_mi_eval(milestone: str) -> bool:
    return milestone != "c0"


def reuse_dependency_checkpoint_for(milestone: str) -> bool:
    return milestone in {"c2", "c4_tfb", "c4_lap"}


def record_only_for(milestone: str) -> bool:
    return milestone in {"c2", "c4_tfb", "c4_lap"}


def max_runs_for(milestone: str) -> int:
    if milestone in {"c2", "c4_tfb", "c4_lap"}:
        return 1
    return MAX_RUNS


def should_early_abort(milestone: str, result: dict) -> bool:
    if milestone not in {"c2", "c4_lap"}:
        return False
    curvature_mean = result.get("curvature_mean")
    mi_ratio_mean = result.get("mi_ratio_mean", 0.0)
    if curvature_mean is None:
        return False
    return curvature_mean < 1.0e-4 and mi_ratio_mean < 1.02


def _load_result(path: Path) -> float | str:
    if not path.exists():
        return "pending"
    state = json.loads(path.read_text())
    if state.get("status") != "completed" or not state.get("runs"):
        return "pending"
    accepted_run = state.get("accepted_run") or 1
    run = state["runs"][accepted_run - 1]
    result = run.get("result", {})
    if "mi_ratio_mean" in result:
        return result["mi_ratio_mean"]
    return "pending"


def comparison_payload(state_dir: Path) -> dict[str, Any]:
    return {
        "title": "Cross-Scale Comparison",
        "ood_domains": list(OOD_DOMAINS),
        "four_layer_reference": FOUR_L_RESULTS,
        "sixteen_layer_results": {
            "C1": _load_result(state_dir / "c1.json"),
            "C2": _load_result(state_dir / "c2.json"),
            "C3": _load_result(state_dir / "c3.json"),
            "C4-TFB": _load_result(state_dir / "c4_tfb.json"),
            "C4-LAP": _load_result(state_dir / "c4_lap.json"),
        },
    }


def comparison_report(state_dir: Path) -> str:
    payload = comparison_payload(state_dir)
    sixteen_layer = payload["sixteen_layer_results"]

    def _fmt(val: float | str) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.2f}x"
        return str(val)

    lines = [
        "=== Cross-Scale Comparison ===",
        "",
        "Method               | 4L MI ratio | 16L MI ratio",
        "---------------------|-------------|-------------",
        f"Variational full     | 1.43x       | {_fmt(sixteen_layer['C1'])}",
        f"BLoB LoRA            | 1.13x       | {_fmt(sixteen_layer['C3'])}",
        f"TFB (post-hoc LoRA)  | 1.10x       | {_fmt(sixteen_layer['C4-TFB'])}",
        f"Laplace full         | 1.00x       | {_fmt(sixteen_layer['C2'])}",
        f"Laplace LoRA         | 1.00x       | {_fmt(sixteen_layer['C4-LAP'])}",
        "",
    ]
    return "\n".join(lines)
