"""Shared evaluation utilities: perplexity, MI suite, qualitative eval."""

import random

import tiktoken
import torch

from minigpt.data import CATEGORY_NAMES, load_agnews
from minigpt.evaluate import compute_perplexity
from minigpt.model import MiniGPT
from minigpt.uncertainty import score_sequence


def select_prompts(
    samples: list[tuple[int, str, str]],
    categories: list[int],
    n_per_category: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Pick n article openings per category for qualitative eval."""
    rng = random.Random(seed)
    prompts = []
    for cat in categories:
        cat_samples = [(title, desc) for c, title, desc in samples if c == cat]
        rng.shuffle(cat_samples)
        for title, desc in cat_samples[:n_per_category]:
            text = f"{title} {desc}"
            words = text.split()[:60]
            prompts.append({
                "category": CATEGORY_NAMES[cat],
                "category_id": cat,
                "text": " ".join(words),
            })
    return prompts


def run_qualitative_eval(
    model: MiniGPT,
    tokenizer: tiktoken.Encoding,
    prompts: list[dict],
    id_categories: list[int],
    device: torch.device,
    n_samples: int = 20,
    max_new_tokens: int = 100,
    score_fn=None,
    generate: bool = True,
) -> tuple[str, list[dict]]:
    """Score (and optionally generate) prompts, return (text report, results).

    Args:
        score_fn: callable(model, token_ids, device, n_samples) -> dict
            with per-token tensors (mi, flip_rate, predictive_entropy, ...).
            Defaults to score_sequence from uncertainty.py.
        generate: if True, generate text continuations (A-series default).
    """
    if score_fn is None:
        score_fn = score_sequence

    lines = ["=" * 70, "QUALITATIVE EVALUATION", "=" * 70, ""]
    results = []

    for p in prompts:
        split = "ID" if p["category_id"] in id_categories else "OOD"
        tokens = tokenizer.encode_ordinary(p["text"])
        if generate:
            max_prompt_len = model.config.block_size - max_new_tokens
        else:
            max_prompt_len = model.config.block_size
        tokens = tokens[:max_prompt_len]

        prompt_ids = torch.tensor(tokens, dtype=torch.long, device=device)

        continuation = None
        if generate:
            idx = prompt_ids.unsqueeze(0)
            generated_ids = model.generate(
                idx, max_new_tokens=max_new_tokens, use_mean=True,
            )
            continuation_ids = generated_ids[0, len(tokens):].tolist()
            continuation = tokenizer.decode(continuation_ids, errors="replace")

        if len(prompt_ids) > 1:
            metrics = score_fn(model, prompt_ids, device, n_samples)
            prompt_mi = metrics["mi"].mean().item()
            prompt_flip = metrics["flip_rate"].mean().item()
            prompt_pred_ent = metrics["predictive_entropy"].mean().item()
        else:
            prompt_mi = 0.0
            prompt_flip = 0.0
            prompt_pred_ent = 0.0

        prompt_text = tokenizer.decode(tokens, errors="replace")
        lines.append(f"[{split} — {p['category']}] \"{prompt_text[:100]}...\"")
        if continuation is not None:
            lines.append(f"  -> \"{continuation[:150]}...\"")
        lines.append(
            f"  -> MI: {prompt_mi:.4f}  |  Flip rate: {prompt_flip:.3f}"
            f"  |  Pred entropy: {prompt_pred_ent:.3f}"
        )
        lines.append("")

        results.append({
            "category": p["category"],
            "split": split,
            "prompt_mi": prompt_mi,
            "flip_rate": prompt_flip,
            "predictive_entropy": prompt_pred_ent,
        })

    id_mis = [r["prompt_mi"] for r in results if r["split"] == "ID"]
    ood_mis = [r["prompt_mi"] for r in results if r["split"] == "OOD"]
    if id_mis and ood_mis:
        avg_id = sum(id_mis) / len(id_mis)
        avg_ood = sum(ood_mis) / len(ood_mis)
        lines.append("-" * 40)
        ratio = avg_ood / max(avg_id, 1e-10)
        lines.append(
            f"Avg MI — ID: {avg_id:.4f}  OOD: {avg_ood:.4f}  Ratio: {ratio:.2f}x"
        )
        lines.append("")

    return "\n".join(lines), results


def eval_perplexity_suite(model, cfg, test_id, test_ood, device, n_batches,
                          val_data=None):
    """Evaluate perplexity on ID, OOD, and optionally val data.

    Caller wraps in appropriate context (e.g. use_mean_weights) if needed.
    Returns dict: test_id_ppl, test_ood_ppl, and optionally val_ppl.
    """
    block_size = cfg["train"]["block_size"]
    batch_size = cfg["train"]["batch_size"]
    results = {}

    if val_data is not None:
        results["val_ppl"] = compute_perplexity(
            model, val_data, block_size, batch_size, device, n_batches=n_batches,
        )
        print(f"Val perplexity: {results['val_ppl']:.2f}")

    results["test_id_ppl"] = compute_perplexity(
        model, test_id, block_size, batch_size, device, n_batches=n_batches,
    )
    print(f"Test ID perplexity: {results['test_id_ppl']:.2f}")

    results["test_ood_ppl"] = None
    if test_ood is not None:
        results["test_ood_ppl"] = compute_perplexity(
            model, test_ood, block_size, batch_size, device,
            n_batches=n_batches,
        )
        print(f"Test OOD perplexity: {results['test_ood_ppl']:.2f}")

    return results


def eval_mi_suite(mi_fn, model, cfg, test_id, test_ood, device,
                  n_samples, n_batches, **kwargs):
    """Run MI evaluation (ID + OOD). Returns (mi_id, mi_ood, mi_ratio)."""
    block_size = cfg["train"]["block_size"]
    batch_size = cfg["train"]["batch_size"]

    mi_id = mi_fn(
        model, test_id, block_size, batch_size, device,
        n_samples=n_samples, n_batches=n_batches, **kwargs,
    )
    print(f"  ID — MI: {mi_id['mi_mean']:.4f}  "
          f"Pred entropy: {mi_id['predictive_entropy_mean']:.4f}  "
          f"Flip rate: {mi_id['flip_rate']:.4f}")

    mi_ood = None
    mi_ratio = None
    if test_ood is not None:
        mi_ood = mi_fn(
            model, test_ood, block_size, batch_size, device,
            n_samples=n_samples, n_batches=n_batches, **kwargs,
        )
        mi_ratio = mi_ood["mi_mean"] / max(mi_id["mi_mean"], 1e-10)
        print(f"  OOD — MI: {mi_ood['mi_mean']:.4f}  "
              f"Pred entropy: {mi_ood['predictive_entropy_mean']:.4f}  "
              f"Flip rate: {mi_ood['flip_rate']:.4f}")
        print(f"  MI ratio (OOD/ID): {mi_ratio:.2f}x")

    return mi_id, mi_ood, mi_ratio


def run_qualitative_suite(model, tokenizer, cfg, device, n_samples,
                          score_fn=None, generate=True):
    """Load AG News, select prompts, run qualitative eval. Returns (report, results)."""
    eval_cfg = cfg["eval"]
    agnews_samples = load_agnews()
    id_cats = cfg["data"]["id_categories"]
    ood_cats = cfg["data"]["ood_categories"]
    prompts = select_prompts(
        agnews_samples, id_cats + ood_cats,
        n_per_category=eval_cfg.get("qualitative_prompts_per_category", 5),
        seed=eval_cfg.get("qualitative_seed", 42),
    )
    return run_qualitative_eval(
        model, tokenizer, prompts, id_cats, device,
        n_samples=n_samples, score_fn=score_fn, generate=generate,
        max_new_tokens=eval_cfg.get("qualitative_max_new_tokens", 100),
    )
