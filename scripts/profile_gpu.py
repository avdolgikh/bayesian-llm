"""Profile GPU utilization and throughput at different batch sizes.

Tests fp32 and AMP (float16), finds the largest batch that fits in VRAM,
and reports throughput to guide batch_size + AMP decisions.

Usage: python scripts/profile_gpu.py [--config configs/a1_agnews.yaml]
"""

import argparse
import gc
import sys
import time

import torch

from minigpt.config import DEFAULT_CONFIG, build_gpt_config, deep_merge, load_yaml
from minigpt.model import MiniGPT
from minigpt.train import _configure_optimizer, get_batch


def profile(model, optimizer, data, block_size, batch_size, device, use_amp=False,
            n_warmup=5, n_steps=20):
    """Run forward+backward+step passes and measure throughput + VRAM."""
    tokens_per_step = batch_size * block_size
    scaler = torch.amp.GradScaler() if use_amp else None

    def run_step():
        x, y = get_batch(data, block_size, batch_size, device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
                kl = model.kl_loss()
                total_loss = loss + kl * 1e-6
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            kl = model.kl_loss()
            total_loss = loss + kl * 1e-6
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Warmup
    try:
        for _ in range(n_warmup):
            run_step()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    # Measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_steps):
        run_step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    tokens_per_sec = (tokens_per_step * n_steps) / elapsed
    ms_per_step = (elapsed / n_steps) * 1000

    return {
        "batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_step": ms_per_step,
        "peak_vram_mb": peak_mb,
        "total_vram_mb": total_vram_mb,
        "vram_pct": peak_mb / total_vram_mb * 100,
    }


def run_sweep(label, gpt_cfg, data, block_size, batch_sizes, device, use_amp):
    """Fresh model for each batch size to avoid stale optimizer/memory."""
    print(f"\n--- {label} ---")
    print(f"{'Batch':>6} | {'Tok/step':>10} | {'Tok/sec':>10} | {'ms/step':>8} | "
          f"{'VRAM MB':>8} | {'VRAM %':>7}")
    print("-" * 74)

    results = []
    for bs in batch_sizes:
        # Fresh model + optimizer each time to get clean memory measurement
        torch.cuda.empty_cache()
        gc.collect()
        model = MiniGPT(gpt_cfg).to(device)
        model.train()
        optimizer = _configure_optimizer(model, type("C", (), {
            "weight_decay": 0.1, "lr": 3e-4, "adam_beta1": 0.9, "adam_beta2": 0.95
        })())

        result = profile(model, optimizer, data, block_size, bs, device, use_amp=use_amp)
        del model, optimizer

        if result is None:
            print(f"{bs:>6} |        OOM")
            break

        results.append(result)
        print(
            f"{bs:>6} | {result['tokens_per_step']:>10,} | {result['tokens_per_sec']:>10,.0f} | "
            f"{result['ms_per_step']:>8.1f} | {result['peak_vram_mb']:>8.0f} | "
            f"{result['vram_pct']:>6.1f}%"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Build config
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        yaml_cfg = load_yaml(args.config)
        cfg = deep_merge(cfg, yaml_cfg)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(0).name
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"GPU: {gpu_name} ({total_vram:.0f} MB)")
    print(f"Model: {cfg['model']['n_layer']}L/{cfg['model']['n_head']}H/{cfg['model']['n_embd']}d")
    print(f"Block size: {cfg['train']['block_size']}")
    bayes_head = cfg.get("model", {}).get("bayes_head", {}).get("enabled", False)
    bayes_ffn = cfg.get("model", {}).get("bayes_ffn", {}).get("enabled", False)
    print(f"Bayesian head: {bayes_head}, FFN: {bayes_ffn}")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    gpt_cfg = build_gpt_config(cfg, vocab_size=enc.n_vocab)

    # Count params
    tmp = MiniGPT(gpt_cfg)
    n_params = sum(p.numel() for p in tmp.parameters())
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")
    del tmp

    # Synthetic data
    block_size = cfg["train"]["block_size"]
    data = torch.randint(0, enc.n_vocab, (500_000,))

    batch_sizes = [32, 64, 96, 128, 192, 256, 384, 512]

    # FP32
    fp32_results = run_sweep("FP32", gpt_cfg, data, block_size, batch_sizes, device, use_amp=False)

    # AMP
    amp_results = run_sweep(
        "AMP (float16)", gpt_cfg, data, block_size, batch_sizes, device, use_amp=True
    )

    # Summary
    print("\n=== SUMMARY ===")
    best_fp32 = max(fp32_results, key=lambda r: r["tokens_per_sec"]) if fp32_results else None
    best_amp = max(amp_results, key=lambda r: r["tokens_per_sec"]) if amp_results else None
    if best_fp32:
        print(f"Best FP32: batch={best_fp32['batch_size']}, "
              f"{best_fp32['tokens_per_sec']:,.0f} tok/s, "
              f"{best_fp32['vram_pct']:.0f}% VRAM")
    if best_amp:
        print(f"Best AMP:  batch={best_amp['batch_size']}, "
              f"{best_amp['tokens_per_sec']:,.0f} tok/s, "
              f"{best_amp['vram_pct']:.0f}% VRAM")
    if best_fp32 and best_amp:
        gain = best_amp["tokens_per_sec"] / best_fp32["tokens_per_sec"]
        print(f"AMP speedup: {gain:.2f}x")


if __name__ == "__main__":
    main()
