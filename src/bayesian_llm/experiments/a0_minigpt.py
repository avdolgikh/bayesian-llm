from __future__ import annotations

import argparse
import time

import torch

from bayesian_llm.data import build_tokenizer, load_corpus
from bayesian_llm.model import BayesConfig, GPTConfig, MiniGPT


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    max_start = data.size(0) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Corpus too small for block_size={block_size}. "
            f"Need length > {block_size + 1}, got {data.size(0)}."
        )
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="A0 mini-GPT baseline.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-tokens", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    text = load_corpus()
    min_len = 10 * (args.block_size + 1)
    if len(text) < min_len:
        repeats = (min_len // len(text)) + 1
        text = text * repeats
    tokenizer = build_tokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        bayes=BayesConfig(enabled=False),
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            train_loss = estimate_loss(
                model, train_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            val_loss = estimate_loss(
                model, val_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            elapsed = time.time() - start
            print(
                f"step {step:4d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s"
            )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=args.sample_tokens)
    print("\n=== sample ===")
    print(tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
