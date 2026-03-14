"""Profile GPU memory and throughput for C-milestone model configurations.

Tests the 16L/8H/512d miniGPT (and optionally other sizes) across:
  - Deterministic baseline
  - Bayesian FFN (A2-equivalent)
  - BLoB LoRA (B2-equivalent)

Sweeps batch sizes until OOM, reports peak VRAM and throughput.
Idempotent — safe to run repeatedly.

Usage:
  python scripts/profile_c_gpu.py
  python scripts/profile_c_gpu.py --n-layer 12 --n-embd 384
  python scripts/profile_c_gpu.py --seq-len 512
  python scripts/profile_c_gpu.py --skip-bayesian
  python scripts/profile_c_gpu.py --batch-sizes 4,8,16,32
"""

import argparse
import gc
import sys
import time

import tiktoken
import torch

from minigpt.layers import BayesConfig
from minigpt.lora import LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.train import _configure_optimizer, get_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer_stub(weight_decay=0.1, lr=3e-4):
    """Minimal object satisfying _configure_optimizer's interface."""
    return type("C", (), {
        "weight_decay": weight_decay,
        "lr": lr,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
    })()


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _profile_one(model, optimizer, data, block_size, batch_size, device,
                 n_warmup=3, n_steps=10):
    """Forward+backward+step, return stats or None on OOM."""
    scaler = torch.amp.GradScaler()

    def step():
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
            kl = model.kl_loss()
            total = loss + kl * 1e-6
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Warmup
    try:
        for _ in range(n_warmup):
            step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        _clear_gpu()
        return None

    # Measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    tok_per_step = batch_size * block_size
    return {
        "batch_size": batch_size,
        "tok_per_step": tok_per_step,
        "tok_per_sec": tok_per_step * n_steps / elapsed,
        "ms_per_step": elapsed / n_steps * 1000,
        "peak_mb": peak_mb,
        "vram_pct": peak_mb / total_vram * 100,
    }


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _gpt_cfg(n_layer, n_head, n_embd, block_size, vocab_size,
             bayes_ffn=False, bayes_attn_v=False):
    no_bayes = BayesConfig(enabled=False)
    ffn_cfg = BayesConfig(enabled=True, prior_std=1.0, init_rho=-2.0) if bayes_ffn else no_bayes
    attn_v_cfg = (BayesConfig(enabled=True, prior_std=1.0, init_rho=-2.0)
                  if bayes_attn_v else no_bayes)
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.2,
        bias=True,
        bayes_head=no_bayes,
        bayes_ffn=ffn_cfg,
        bayes_attn_v=attn_v_cfg,
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(label, model_factory, data, block_size, batch_sizes, device):
    """Run batch-size sweep, printing results as they come."""
    header = (f"{'Batch':>6} | {'Tok/step':>10} | {'Tok/sec':>10} | "
              f"{'ms/step':>8} | {'Peak MB':>8} | {'VRAM%':>6}")
    print(f"\n{'=' * 74}")
    print(f"  {label}")
    print(f"{'=' * 74}")
    print(header)
    print("-" * 74)

    results = []
    for bs in batch_sizes:
        _clear_gpu()
        try:
            model = model_factory()
            model.to(device)
            model.train()
        except torch.cuda.OutOfMemoryError:
            _clear_gpu()
            print(f"{bs:>6} |  MODEL OOM — skipping remaining batch sizes")
            break

        optimizer = _configure_optimizer(model, _make_optimizer_stub())
        total, trainable = _count_params(model)
        r = _profile_one(model, optimizer, data, block_size, bs, device)
        del model, optimizer
        _clear_gpu()

        if r is None:
            print(f"{bs:>6} |  OOM")
            break

        results.append(r)
        print(f"{r['batch_size']:>6} | {r['tok_per_step']:>10,} | "
              f"{r['tok_per_sec']:>10,.0f} | {r['ms_per_step']:>8.1f} | "
              f"{r['peak_mb']:>8.0f} | {r['vram_pct']:>5.1f}%")

    return results, total, trainable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile C-milestone GPU memory")
    parser.add_argument("--n-layer", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (block_size)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank for BLoB sweep")
    parser.add_argument("--skip-bayesian", action="store_true",
                        help="Only test deterministic model")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip BLoB LoRA sweep")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**2:.0f} MB VRAM)")
    print(f"Model: {args.n_layer}L/{args.n_head}H/{args.n_embd}d, seq={args.seq_len}")

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    data = torch.randint(0, vocab_size, (1_000_000,))

    all_results = {}

    # ---- 1. Deterministic ----
    def det_factory():
        cfg = _gpt_cfg(args.n_layer, args.n_head, args.n_embd,
                        args.seq_len, vocab_size)
        return MiniGPT(cfg)

    results, total, trainable = run_sweep(
        f"Deterministic ({args.n_layer}L/{args.n_head}H/{args.n_embd}d)",
        det_factory, data, args.seq_len, batch_sizes, device,
    )
    print(f"  Params: {total:,} total, {trainable:,} trainable")
    all_results["deterministic"] = results

    if not args.skip_bayesian:
        # ---- 2. Bayesian FFN (A2-equivalent) ----
        def bayes_ffn_factory():
            cfg = _gpt_cfg(args.n_layer, args.n_head, args.n_embd,
                            args.seq_len, vocab_size, bayes_ffn=True)
            return MiniGPT(cfg)

        results, total, trainable = run_sweep(
            f"Bayesian FFN ({args.n_layer}L/{args.n_head}H/{args.n_embd}d)",
            bayes_ffn_factory, data, args.seq_len, batch_sizes, device,
        )
        print(f"  Params: {total:,} total, {trainable:,} trainable")
        all_results["bayesian_ffn"] = results

        # ---- 3. Bayesian FFN + Attn-V (A3-equivalent) ----
        def bayes_full_factory():
            cfg = _gpt_cfg(args.n_layer, args.n_head, args.n_embd,
                            args.seq_len, vocab_size,
                            bayes_ffn=True, bayes_attn_v=True)
            return MiniGPT(cfg)

        results, total, trainable = run_sweep(
            f"Bayesian FFN+AttnV ({args.n_layer}L/{args.n_head}H/{args.n_embd}d)",
            bayes_full_factory, data, args.seq_len, batch_sizes, device,
        )
        print(f"  Params: {total:,} total, {trainable:,} trainable")
        all_results["bayesian_ffn_attn_v"] = results

    if not args.skip_lora and not args.skip_bayesian:
        # ---- 4. BLoB LoRA (B2-equivalent) ----
        lora_cfg = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_rank * 2.0,
            target="ffn",
            prior_std=0.2,
            init_g=0.1,
        )

        def blob_lora_factory():
            cfg = _gpt_cfg(args.n_layer, args.n_head, args.n_embd,
                            args.seq_len, vocab_size)
            model = MiniGPT(cfg)
            inject_lora(model, lora_cfg, bayesian=True)
            return model

        results, total, trainable = run_sweep(
            f"BLoB LoRA rank={args.lora_rank} ({args.n_layer}L/{args.n_head}H/{args.n_embd}d)",
            blob_lora_factory, data, args.seq_len, batch_sizes, device,
        )
        print(f"  Params: {total:,} total, {trainable:,} trainable")
        all_results["blob_lora"] = results

        # ---- 5. Deterministic LoRA (B3-equivalent base) ----
        def det_lora_factory():
            cfg = _gpt_cfg(args.n_layer, args.n_head, args.n_embd,
                            args.seq_len, vocab_size)
            model = MiniGPT(cfg)
            inject_lora(model, lora_cfg, bayesian=False)
            return model

        results, total, trainable = run_sweep(
            f"Det LoRA rank={args.lora_rank} ({args.n_layer}L/{args.n_head}H/{args.n_embd}d)",
            det_lora_factory, data, args.seq_len, batch_sizes, device,
        )
        print(f"  Params: {total:,} total, {trainable:,} trainable")
        all_results["det_lora"] = results

    # ---- Summary ----
    print("\n")
    print("=" * 74)
    print("  SUMMARY — Recommended batch sizes (< 80% VRAM)")
    print("=" * 74)
    for name, results in all_results.items():
        if not results:
            print(f"  {name:25s} : no results (all OOM)")
            continue
        safe = [r for r in results if r["vram_pct"] < 80]
        best = max(safe, key=lambda r: r["tok_per_sec"]) if safe else results[0]
        max_bs = max(results, key=lambda r: r["batch_size"])
        print(f"  {name:25s} : best_bs={best['batch_size']:>3}, "
              f"{best['tok_per_sec']:>8,.0f} tok/s, "
              f"{best['peak_mb']:>6.0f} MB ({best['vram_pct']:.0f}%), "
              f"max_bs={max_bs['batch_size']}")

    # Effective batch size recommendations
    target_effective = 32
    print(f"\n  For effective batch_size={target_effective}:")
    for name, results in all_results.items():
        if not results:
            continue
        safe = [r for r in results if r["vram_pct"] < 80]
        if not safe:
            print(f"    {name:25s} : no safe batch size found")
            continue
        best = max(safe, key=lambda r: r["batch_size"])
        bs = best["batch_size"]
        accum = max(1, target_effective // bs)
        actual_eff = bs * accum
        print(f"    {name:25s} : batch_size={bs}, "
              f"grad_accum={accum}, effective={actual_eff}")


if __name__ == "__main__":
    main()
