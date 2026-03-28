import math

import tiktoken
import torch

from minigpt.model import MiniGPT
from minigpt.train import get_batch


@torch.no_grad()
def compute_perplexity(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    n_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def compute_perplexity_mc(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    n_samples: int = 20,
    n_batches: int = 20,
) -> float:
    """Compute MC-averaged perplexity: average loss over N weight samples per batch.

    For each batch, runs N forward passes with independently sampled weights
    (via reparameterization trick), averages the cross-entropy losses, then
    computes perplexity = exp(mean averaged loss).

    For deterministic models, all N samples produce identical loss (degenerates).
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        batch_loss = 0.0
        for _ in range(n_samples):
            _, loss = model(x, y)
            batch_loss += loss.item()
        losses.append(batch_loss / n_samples)
    model.train()
    return math.exp(sum(losses) / len(losses))


def generate_text(
    model: MiniGPT,
    enc: tiktoken.Encoding,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    device = next(model.parameters()).device
    if prompt:
        tokens = enc.encode_ordinary(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        # Start with newline token
        idx = torch.tensor([[enc.encode_ordinary("\n")[0]]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return enc.decode(out[0].tolist(), errors="replace")


def evaluate(
    model: MiniGPT,
    val_data: torch.Tensor,
    enc: tiktoken.Encoding,
    block_size: int,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    n_perplexity_batches: int = 20,
) -> dict:
    ppl = compute_perplexity(
        model, val_data, block_size, batch_size, device, n_batches=n_perplexity_batches,
    )
    sample = generate_text(model, enc, max_new_tokens=max_new_tokens, temperature=temperature)
    print(f"\nVal perplexity: {ppl:.2f}")
    import sys
    enc_name = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    safe = sample.encode(enc_name, errors="replace").decode(enc_name)
    print(f"\n=== generated sample ===\n{safe}\n========================")
    return {"perplexity": ppl, "sample": sample}
