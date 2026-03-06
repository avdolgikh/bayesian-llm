"""Shared MLflow logging helpers for all milestones."""

import json
from contextlib import nullcontext

import mlflow
import torch

from minigpt.config import config_to_flat_params


def mlflow_context(cfg, use_mlflow):
    """Setup MLflow tracking and return run context manager."""
    if use_mlflow:
        mlflow.set_tracking_uri(
            cfg["experiment"].get("mlflow_uri", "sqlite:///mlflow.db"),
        )
        mlflow.set_experiment(cfg["experiment"]["name"])
        return mlflow.start_run(run_name=cfg["experiment"]["run_name"])
    return nullcontext()


def log_common_mlflow(cfg, tokenizer, n_params, milestone, **extra_params):
    """Log common params and tags to active MLflow run."""
    flat = config_to_flat_params(cfg)
    flat["vocab_size"] = str(tokenizer.n_vocab)
    flat["n_params"] = str(n_params)
    flat["tokenizer"] = "gpt2-bpe"
    flat.update({k: str(v) for k, v in extra_params.items()})
    mlflow.log_params(flat)

    mlflow.set_tag("dataset", cfg["data"]["dataset"])
    mlflow.set_tag("milestone", milestone)
    if torch.cuda.is_available():
        mlflow.set_tag("gpu", torch.cuda.get_device_name())


def log_train_meta_mlflow(train_meta):
    """Log training meta to active MLflow run."""
    mlflow.log_params({
        "best_val_loss": f"{train_meta['best_val_loss']:.4f}",
        "best_val_step": str(int(train_meta["best_val_step"])),
        "train_time_sec": f"{train_meta['train_time_sec']:.1f}",
    })


def log_perplexity_mlflow(ppl_results):
    """Log perplexity results to active MLflow run."""
    if ppl_results.get("val_ppl") is not None:
        mlflow.log_param(
            "final_val_perplexity", f"{ppl_results['val_ppl']:.2f}",
        )
    mlflow.log_param("test_id_perplexity", f"{ppl_results['test_id_ppl']:.2f}")
    if ppl_results.get("test_ood_ppl") is not None:
        mlflow.log_param(
            "test_ood_perplexity", f"{ppl_results['test_ood_ppl']:.2f}",
        )


def log_mi_mlflow(mi_id, mi_ood=None, mi_ratio=None):
    """Log MI metrics to active MLflow run."""
    mlflow.log_params({
        "mi_id_mean": f"{mi_id['mi_mean']:.6f}",
        "flip_rate_id": f"{mi_id['flip_rate']:.4f}",
    })
    if mi_ood is not None:
        mlflow.log_params({
            "mi_ood_mean": f"{mi_ood['mi_mean']:.6f}",
            "flip_rate_ood": f"{mi_ood['flip_rate']:.4f}",
        })
    if mi_ratio is not None:
        mlflow.log_param("mi_ood_id_ratio", f"{mi_ratio:.2f}")


def log_qualitative_mlflow(report, qual_results):
    """Log qualitative artifacts to active MLflow run."""
    mlflow.log_text(report, "qualitative_eval.txt")
    mlflow.log_text(
        json.dumps(qual_results, indent=2), "qualitative_metrics.json",
    )
