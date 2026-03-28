"""D3: Production inference benchmarks — latency, VRAM, throughput, AUROC vs N.

Benchmarks five configurations:
  C0 Deterministic  — single forward pass (baseline)
  C3 Mean-Weights   — BLoB LoRA with mu weights (D2 production path)
  C1 Full Var. MC   — all FFN weights sampled (N passes)
  C3 LoRA MC        — only LoRA params sampled (N passes)
  C4-TFB MC         — TFB sampling + apply_sampled_params (N passes)

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --n-sequences 100 --latency-repeats 20
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c_milestones import OOD_DOMAINS, build_milestone_config
from minigpt.config import build_gpt_config, build_lora_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.laplace import apply_sampled_params
from minigpt.layers import use_mean_weights
from minigpt.lora import inject_lora
from minigpt.model import MiniGPT
from minigpt.tfb import load_tfb_state, sample_tfb_params
from minigpt.train import load_checkpoint
from minigpt.uncertainty import aggregate_sequence_scores, auroc

CKPT_DIR = Path("data/checkpoints")
BLOCK_SIZE = 256
N_VALUES = [1, 3, 5, 10, 20]

# Method registry: key -> (label, milestone_key, method_type)
METHOD_DEFS = [
    ("c0",      "C0 Deterministic", "deterministic"),
    ("c3_mean", "C3 Mean-Weights",  "mean_weights"),
    ("c1",      "C1 Full Var. MC",  "variational"),
    ("c3",      "C3 LoRA MC",       "variational"),
    ("c4_tfb",  "C4-TFB MC",        "tfb"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(method_key: str, device: torch.device):
    """Load model + optional post-hoc state. Returns (model, state_or_None)."""
    tokenizer = get_tokenizer()
    state = None

    if method_key == "c0":
        cfg = build_milestone_config("c0")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c0/ckpt_best.pt", model)

    elif method_key == "c1":
        cfg = build_milestone_config("c1")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c1/ckpt_best.pt", model)

    elif method_key in ("c3", "c3_mean"):
        cfg = build_milestone_config("c3_phase2")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        model = inject_lora(model, build_lora_config(cfg), bayesian=True)
        load_checkpoint(CKPT_DIR / "c3/ckpt_best.pt", model)

    elif method_key == "c4_tfb":
        cfg = build_milestone_config("c4_tfb")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        model = inject_lora(model, build_lora_config(cfg), bayesian=False)
        blob_sd = torch.load(
            CKPT_DIR / "c3/ckpt_best.pt", weights_only=False,
        )["model_state_dict"]
        target_sd = model.state_dict()
        mapped = {}
        for key in target_sd:
            if key in blob_sd:
                mapped[key] = blob_sd[key]
            elif "lora_A" in key:
                mapped[key] = blob_sd.get(
                    key.replace(".lora_A", ".lora_A_mu"), target_sd[key],
                )
            else:
                mapped[key] = target_sd[key]
        model.load_state_dict(mapped)
        state = load_tfb_state(
            CKPT_DIR / "c4_tfb/tfb_state.pt", map_location=device,
        )
    else:
        raise ValueError(f"Unknown method key: {method_key}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {method_key}: {n_params:,} params")
    return model.to(device).eval(), state


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_sequences(data: torch.Tensor, block_size: int, n: int):
    max_seqs = (len(data) - 1) // block_size
    n = min(n, max_seqs)
    seqs = []
    for i in range(n):
        s = i * block_size
        seqs.append((data[s : s + block_size], data[s + 1 : s + block_size + 1]))
    return seqs


def load_eval_data(n_sequences: int):
    cfg = build_milestone_config("c0")
    tokenizer = get_tokenizer()
    data = load_pile_data(cfg, tokenizer)
    test_id = data["test_id"]
    ood_parts = [data[f"test_ood_{d}"] for d in OOD_DOMAINS if f"test_ood_{d}" in data]
    test_ood = torch.cat(ood_parts) if ood_parts else data.get("test_ood")
    id_seqs = _extract_sequences(test_id, BLOCK_SIZE, n_sequences)
    ood_seqs = _extract_sequences(test_ood, BLOCK_SIZE, n_sequences)
    print(f"Loaded {len(id_seqs)} ID + {len(ood_seqs)} OOD sequences "
          f"(block_size={BLOCK_SIZE})")
    return id_seqs, ood_seqs


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_n(model, x_dev, n_samples, device, method_type, state, seq_idx=0):
    """Run N forward passes, return (p_bar, mi, max_prob_unc) tensors."""
    eps = 1e-10
    seq_len = x_dev.size(1)
    vocab_size = model.config.vocab_size
    use_amp = device.type == "cuda"

    p_sum = torch.zeros(seq_len, vocab_size, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)

    ctx = use_mean_weights(model) if method_type == "mean_weights" else nullcontext()
    with ctx:
        for s in range(n_samples):
            if method_type == "tfb":
                seed = seq_idx * n_samples + s
                sampled = sample_tfb_params(state, seed=seed)
                with apply_sampled_params(model, sampled), torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp,
                ):
                    logits, _ = model(x_dev)
            else:
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp,
                ):
                    logits, _ = model(x_dev)

            probs = F.softmax(logits[0].float(), dim=-1)
            p_sum.add_(probs)
            entropy_sum.add_(-(probs * torch.log(probs + eps)).sum(dim=-1))

    p_bar = p_sum / n_samples
    pred_ent = -(p_bar * torch.log(p_bar + eps)).sum(dim=-1)
    mi = pred_ent - entropy_sum / n_samples
    max_prob_unc = 1.0 - p_bar.max(dim=-1).values
    return mi.cpu(), max_prob_unc.cpu()


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(
    model, device, method_type, state,
    n_samples: int, seq_len: int = BLOCK_SIZE,
    n_warmup: int = 5, n_measure: int = 20,
) -> tuple[float, float]:
    """Measure avg latency (ms) and peak VRAM (MB) for one sequence scoring."""
    x_dev = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    use_amp = device.type == "cuda"

    def _run():
        ctx = use_mean_weights(model) if method_type == "mean_weights" else nullcontext()
        with ctx:
            for s in range(n_samples):
                if method_type == "tfb":
                    sampled = sample_tfb_params(state, seed=s)
                    with apply_sampled_params(model, sampled), torch.amp.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp,
                    ):
                        model(x_dev)
                else:
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp,
                    ):
                        model(x_dev)

    for _ in range(n_warmup):
        _run()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_measure):
            _run()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        t0 = time.perf_counter()
        for _ in range(n_measure):
            _run()
        total_ms = (time.perf_counter() - t0) * 1000
        peak_vram = 0.0

    return total_ms / n_measure, peak_vram


# ---------------------------------------------------------------------------
# AUROC vs N
# ---------------------------------------------------------------------------

def compute_auroc_at_n(
    model, device, method_type, state,
    n_samples: int, id_seqs: list, ood_seqs: list,
) -> dict[str, float]:
    """Score ID/OOD sequences with N samples, return AUROC for MI and max-prob."""
    id_mi, id_mp = [], []
    ood_mi, ood_mp = [], []

    for i, (x, y) in enumerate(id_seqs):
        mi, mp = _forward_n(
            model, x.unsqueeze(0).to(device), n_samples, device,
            method_type, state, seq_idx=i,
        )
        id_mi.append(aggregate_sequence_scores(mi))
        id_mp.append(aggregate_sequence_scores(mp))

    for i, (x, y) in enumerate(ood_seqs):
        mi, mp = _forward_n(
            model, x.unsqueeze(0).to(device), n_samples, device,
            method_type, state, seq_idx=len(id_seqs) + i,
        )
        ood_mi.append(aggregate_sequence_scores(mi))
        ood_mp.append(aggregate_sequence_scores(mp))

    n_id, n_ood = len(id_mi), len(ood_mi)
    labels = torch.cat([torch.zeros(n_id), torch.ones(n_ood)])

    mi_scores = torch.tensor(id_mi + ood_mi)
    mp_scores = torch.tensor(id_mp + ood_mp)

    return {
        "auroc_mi": auroc(mi_scores, labels) if method_type != "deterministic" else 0.0,
        "auroc_max_prob": auroc(mp_scores, labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="D3: Production inference benchmarks")
    p.add_argument("--n-sequences", type=int, default=200,
                    help="Test sequences per split for AUROC (default: 200)")
    p.add_argument("--latency-repeats", type=int, default=20,
                    help="Measurement repeats for latency (default: 20)")
    p.add_argument("--skip-auroc", action="store_true",
                    help="Skip AUROC-vs-N (latency only)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Part 1: Latency / VRAM / Throughput ----
    print("\n" + "=" * 70)
    print("PART 1: Latency / VRAM / Throughput")
    print("=" * 70)

    latency_results = []  # (label, n, latency_ms, vram_mb, throughput)
    baseline_latency = None

    for method_key, label, method_type in METHOD_DEFS:
        print(f"\n--- {label} ---")
        model, state = _load_model(method_key, device)

        # For deterministic/mean_weights, only N=1 matters
        ns = [1] if method_type in ("deterministic", "mean_weights") else N_VALUES

        for n in ns:
            lat_ms, vram_mb = benchmark_latency(
                model, device, method_type, state,
                n_samples=n, n_measure=args.latency_repeats,
            )
            throughput = 1000.0 / lat_ms if lat_ms > 0 else 0.0

            if baseline_latency is None:
                baseline_latency = lat_ms

            overhead = lat_ms / baseline_latency
            latency_results.append((label, n, lat_ms, vram_mb, throughput, overhead))
            print(f"  N={n:>2}: {lat_ms:>7.1f} ms  |  {vram_mb:>7.0f} MB  |  "
                  f"{throughput:>5.1f} seq/s  |  {overhead:>5.2f}x")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print latency table
    print("\n## Production Inference Table\n")
    print("| Method              |  N | Latency (ms) | Overhead | Throughput | VRAM (MB) |")
    print("|---------------------|----|--------------|----------|------------|-----------|")
    for label, n, lat, vram, tput, overhead in latency_results:
        print(f"| {label:<19} | {n:>2} | {lat:>12.1f} | {overhead:>6.2f}x  "
              f"| {tput:>7.1f}/s  | {vram:>9.0f} |")

    # ---- Part 2: AUROC vs N ----
    if args.skip_auroc:
        print("\nSkipping AUROC-vs-N (--skip-auroc)")
        return

    print("\n" + "=" * 70)
    print("PART 2: AUROC vs N (quality-cost tradeoff)")
    print("=" * 70)

    id_seqs, ood_seqs = load_eval_data(args.n_sequences)

    auroc_results = []  # (label, n, auroc_mi, auroc_mp)

    for method_key, label, method_type in METHOD_DEFS:
        print(f"\n--- {label} ---")
        model, state = _load_model(method_key, device)

        ns = [1] if method_type in ("deterministic", "mean_weights") else N_VALUES

        for n in ns:
            t0 = time.time()
            r = compute_auroc_at_n(
                model, device, method_type, state,
                n_samples=n, id_seqs=id_seqs, ood_seqs=ood_seqs,
            )
            elapsed = time.time() - t0
            auroc_results.append((label, n, r["auroc_mi"], r["auroc_max_prob"]))

            print(f"  N={n:>2}: AUROC(MI)={r['auroc_mi']:.3f}  "
                  f"AUROC(MaxProb)={r['auroc_max_prob']:.3f}  ({elapsed:.0f}s)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print AUROC table
    print("\n## AUROC vs N (Quality-Cost Tradeoff)\n")
    print("| Method              |  N | AUROC (MI) | AUROC (MaxProb) |")
    print("|---------------------|----|------------|-----------------|")
    for label, n, auroc_mi, auroc_mp in auroc_results:
        mi_str = f"{auroc_mi:.3f}" if auroc_mi > 0 else "--"
        print(f"| {label:<19} | {n:>2} | {mi_str:>10} | {auroc_mp:>15.3f} |")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find C3 LoRA MC at N=5 for the key D3 gate check
    c3_n5 = [r for r in latency_results if r[0] == "C3 LoRA MC" and r[1] == 5]
    c3_auroc_n5 = [r for r in auroc_results if r[0] == "C3 LoRA MC" and r[1] == 5]
    if c3_n5:
        lat = c3_n5[0]
        print(f"C3 LoRA MC (N=5): {lat[2]:.1f} ms, {lat[5]:.2f}x overhead, "
              f"{lat[3]:.0f} MB VRAM")
        overhead_ok = lat[5] < 2.0
        print(f"  Overhead gate (<2x): {'PASS' if overhead_ok else 'FAIL'}")
    if c3_auroc_n5:
        a = c3_auroc_n5[0]
        print(f"  AUROC(MI) = {a[2]:.3f}")
        auroc_ok = a[2] > 0.80
        print(f"  AUROC gate (>0.80): {'PASS' if auroc_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
