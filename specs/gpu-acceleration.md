# GPU Acceleration Spec

**Date:** 2026-02-25
**Goal:** Reduce A1 training time by properly utilizing RTX 4070 VRAM (12 GB).

## Problem

The A1 Bayesian model has **41.8M parameters** (vs A0's ~16M) because:
- Weight tying is disabled when `lm_head` is Bayesian (can't share deterministic `token_emb` weights with stochastic `lm_head`)
- `BayesianLinear` stores both μ and ρ per weight, doubling `lm_head` param count

With the current config (`batch_size=64`, fp32), peak VRAM allocation is **14.9 GB** — exceeding the RTX 4070's 12.3 GB physical VRAM. PyTorch does **not** crash with an OOM error. Instead, CUDA on Windows silently spills pages to system RAM over PCIe, causing a **10x throughput collapse** with no warning.

The A0 reference on the same GPU ran at ~70K tok/sec. A1 at `batch_size=64` runs at ~8K tok/sec — not because Bayesian layers are expensive, but because of silent VRAM overflow.

## Profiling Results

Script: `scripts/profile_gpu.py` — measures throughput and peak VRAM at varying batch sizes. Fresh model + optimizer per batch size to avoid stale memory.

**Hardware:** NVIDIA GeForce RTX 4070 (12,282 MB VRAM)
**Model:** 4L/4H/256d, block_size=256, Bayesian lm_head, 41,822,464 params (159.5 MB fp32)

### FP32

| Batch | Tok/step | Tok/sec   | ms/step | VRAM MB | VRAM % | Note                        |
|------:|---------:|----------:|--------:|--------:|-------:|-----------------------------|
|    32 |    8,192 |    82,258 |    99.6 |   7,818 |  63.7% | **Fits — max throughput**   |
|    64 |   16,384 |     8,227 | 1,991.5 |  14,939 | 121.6% | Silent overflow → **10x slower** |
|    96 |   24,576 |     7,500 | 3,276.9 |  22,053 | 179.6% | Deep swap                   |
|   128 |   32,768 |    14,616 | 2,241.9 |  29,168 | 237.5% | Deep swap                   |
|   192 |      — |       OOM |     — |     — |    — | Hard OOM                    |

### AMP (float16 autocast + GradScaler)

| Batch | Tok/step | Tok/sec   | ms/step | VRAM MB | VRAM % | Note                        |
|------:|---------:|----------:|--------:|--------:|-------:|-----------------------------|
|    32 |    8,192 |   110,815 |    73.9 |   5,939 |  48.4% | **Best throughput overall** |
|    64 |   16,384 |    32,637 |   502.0 |  11,193 |  91.1% | Fits, but saturated         |
|    96 |   24,576 |    17,580 | 1,398.0 |  16,447 | 133.9% | Overflow begins             |
|   128 |   32,768 |    13,163 | 2,489.5 |  21,700 | 176.7% | Swap                        |
|   192 |   49,152 |    16,929 | 2,903.5 |  32,211 | 262.3% | Deep swap                   |
|   256 |      — |       OOM |     — |     — |    — | Hard OOM                    |

### Summary

| Config              | Tok/sec   | VRAM % | vs current (B=64 fp32) |
|---------------------|----------:|-------:|------------------------:|
| Current (B=64 fp32) |     8,227 | 121.6% |                   1.0x |
| B=32 fp32           |    82,258 |  63.7% |               **10.0x** |
| B=32 AMP            |   110,815 |  48.4% |               **13.5x** |
| B=64 AMP            |    32,637 |  91.1% |                   4.0x |

**Key insight:** `batch_size=32` already saturates the GPU's compute units for this model size. Going larger provides no throughput gain — it only overflows VRAM.

## Implementation Plan

Three changes, in priority order:

### 1. AMP (Mixed Precision) — 1.35x speedup

Add `torch.amp.autocast` + `GradScaler` to the training loop. The forward pass (embeddings, attention, FFN, Bayesian lm_head, KL) runs in float16; optimizer step stays in float32 (master weights).

**Why it's safe:** All operations in the model (matmuls, softplus, softmax, KL divergence) are numerically stable in float16. GradScaler handles underflow in gradients.

**Files to change:**

#### `minigpt/train.py`

In `train()`:
```python
# After optimizer creation (~line 170)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler() if use_amp else None

# Training loop body (replace lines 211-233):
x, y = get_batch(...)
with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    _, ce_loss = model(x, y)
    if is_bayesian:
        kl = model.kl_loss()
        loss = ce_loss + kl_scale * kl
    else:
        loss = ce_loss

optimizer.zero_grad(set_to_none=True)
if scaler is not None:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
```

In `estimate_loss()`:
```python
# Wrap forward pass in autocast
with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    _, ce_loss = model(x, y)
```

#### `minigpt/uncertainty.py`

In `_stream_metrics()` and `compute_uncertainty_metrics()`:
```python
# Wrap lm_head calls + softmax in autocast
with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    logits = model.lm_head(h)
    probs = F.softmax(logits, dim=-1).float()  # back to fp32 for entropy accumulation
```

#### Config

No YAML change needed — AMP is auto-detected from device type. Could add `train.use_amp: true` as an opt-out mechanism, but default-on for CUDA is simpler and matches PyTorch conventions.

### 2. Batch Size: 64 → 32

One-line YAML change:

```yaml
# configs/a1_agnews.yaml
train:
  batch_size: 32    # was 64 — overflowed 12 GB VRAM silently
```

With AMP, `batch_size=32` uses only 48% VRAM and delivers maximum throughput (110K tok/sec). There is headroom to revisit this when the model grows in A2.

**Training dynamics note:** Halving batch size means noisier gradient estimates. For this small model with a forgiving loss landscape, this is fine — SGD noise is often beneficial for generalization. The effective learning rate scales with batch size, but our cosine schedule + warmup already handles this. No LR adjustment needed.

### 3. Gradient Accumulation (infrastructure for A2+)

Add `gradient_accumulation_steps` to `TrainConfig` and the training loop. This decouples "effective batch size" (how many tokens inform one optimizer update) from "micro-batch size" (how many tokens fit in VRAM at once).

Not strictly needed for A1 (`batch_size=32` is fine), but the infrastructure is cheap and will be essential for A2 (Bayesian FFN layers → even more params → tighter VRAM).

**Files to change:**

#### `minigpt/train.py`

```python
@dataclass
class TrainConfig:
    ...
    gradient_accumulation_steps: int = 1  # new field

# In train():
accum_steps = cfg.gradient_accumulation_steps

for step in range(start_step, cfg.steps + 1):
    # LR update (unchanged)
    ...

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(accum_steps):
        x, y = get_batch(...)
        with torch.amp.autocast(...):
            _, ce_loss = model(x, y)
            if is_bayesian:
                kl = model.kl_loss()
                loss = ce_loss + kl_scale * kl
            else:
                loss = ce_loss
            loss = loss / accum_steps  # scale so gradients average correctly

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Optimizer step (once per outer step)
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
```

**KL handling with accumulation:** KL is a property of the weight distribution, not the data batch. It should be added once per optimizer step, not once per micro-batch. Adjust by only adding KL on the last micro-step, or dividing KL by `accum_steps` to compensate.

#### Config

```yaml
# configs/a1_agnews.yaml
train:
  batch_size: 32
  gradient_accumulation_steps: 1  # effective batch = 32 * 1 = 32
```

For A2, could set `gradient_accumulation_steps: 2` to get effective batch 64 with micro-batch 32.

## What NOT to Do

- **PEFT (LoRA, adapters):** Not applicable — we train from scratch, not fine-tuning a pretrained model.
- **Gradient checkpointing:** Trades compute for activation memory. Our bottleneck is parameter memory (42M params × fp32 × 4 for param+grad+2 Adam states = 670 MB), not activation memory. Gradient checkpointing wouldn't help.
- **torch.compile:** Worth testing later but not a priority. The model is small enough that kernel fusion gains are modest. The main bottleneck (VRAM overflow) is solved by AMP + correct batch size.
- **Reduce `num_samples` (MC samples for uncertainty):** This saves eval time, not training time. Keep at 30 for now — reducing it is an eval-quality tradeoff, not a GPU utilization question.

## Expected Impact

| Metric                | Current (B=64, fp32)  | After (B=32, AMP)     |
|-----------------------|----------------------:|-----------------------:|
| Throughput            | ~8K tok/s             | ~111K tok/s            |
| VRAM usage            | 122% (overflow)       | 48%                    |
| 5K steps wall time    | ~55 min (estimated)   | ~4 min (estimated)     |
| Tokens per step       | 16,384                | 8,192                  |
| Total tokens (5K steps)| 82M                  | 41M                    |

Note: halving batch size halves total tokens seen. If convergence requires the same total tokens, double steps to 10K. Even at 10K steps, wall time is ~8 min vs ~55 min — still a **7x improvement**.

## Implementation Order

1. Add AMP to `train.py` (training loop + `estimate_loss`)
2. Add AMP to `uncertainty.py` (`_stream_metrics` + `compute_uncertainty_metrics`)
3. Change `batch_size: 64 → 32` in `configs/a1_agnews.yaml`
4. Add `gradient_accumulation_steps` to `TrainConfig` + training loop
5. Add `gradient_accumulation_steps: 1` to config YAML
6. Run profiling script to verify (expect ~110K tok/s, <50% VRAM)
7. Run A1 training, compare to A0 reference
