"""P2: Reproducibility — same seed produces identical results."""

import torch

from minigpt.layers import BayesConfig
from minigpt.model import GPTConfig, MiniGPT
from minigpt.train import get_batch


class TestReproducibility:
    """P2: Same config + same seed -> identical loss."""

    def _run_steps(self, seed: int, n_steps: int = 3) -> list[float]:
        torch.manual_seed(seed)
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.1,
            bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        model.train()
        device = torch.device("cpu")

        data = torch.randint(0, 256, (2000,))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(n_steps):
            x, y = get_batch(data, block_size=32, batch_size=4, device=device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test_same_seed_identical_losses(self):
        losses_a = self._run_steps(seed=42)
        losses_b = self._run_steps(seed=42)
        for i, (a, b) in enumerate(zip(losses_a, losses_b)):
            assert a == b, f"Step {i}: loss {a} != {b} with same seed"
