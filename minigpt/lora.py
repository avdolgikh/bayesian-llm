"""B2: BLoB-style Bayesian LoRA.

BLoBLoRALinear: Bayesian Low-Rank Adaptation by Backpropagation.
- Bayesianizes A matrix only; B stays deterministic (BLoB Theorem 3.1).
- G^2 variance parameterization: sigma_ij = G_ij^2.
- KL reduces to cheap diagonal KL in A-space (BLoB Theorem 3.2).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from minigpt.layers import BayesianModule


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target: str = "ffn"
    prior_std: float = 0.2
    init_g: float = 0.05


class BLoBLoRALinear(BayesianModule):
    """Wraps a frozen nn.Linear with a BLoB Bayesian LoRA adapter.

    Forward: y = W0*x + (alpha/rank) * B * A_sample * x
      where A_sample = M + G^2 * eps,  eps ~ N(0, I)
    KL: diagonal KL between q(A) = N(M, diag(G^4)) and p(A) = N(0, sigma_p^2 * I)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        prior_std: float,
        init_g: float,
    ) -> None:
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Store frozen base weight
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.prior_std = prior_std

        # B: deterministic, trainable. Init zeros so adapter starts at identity.
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # A posterior mean: trainable. Init Kaiming uniform (standard LoRA).
        self.lora_A_mu = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A_mu, a=5 ** 0.5)

        # A variance param: sigma = G^2. Init U(init_g/sqrt(2), init_g).
        self.lora_A_g = nn.Parameter(torch.empty(rank, in_features))
        nn.init.uniform_(self.lora_A_g, init_g / 2 ** 0.5, init_g)

        # Sampling state (same protocol as BayesianLinear)
        self._frozen_weight: torch.Tensor | None = None  # cached A sample
        self._use_mean: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mean:
            return self.mean_forward(x)
        if self._frozen_weight is not None:
            a_sample = self._frozen_weight
        else:
            eps = torch.randn_like(self.lora_A_mu)
            a_sample = self.lora_A_mu + self.lora_A_g ** 2 * eps
        base_out = self.base_linear(x)
        lora_out = (x @ a_sample.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def mean_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using posterior mean only — deterministic eval."""
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A_mu.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def freeze_sample(self) -> None:
        """Sample A once and cache for coherent multi-token MC scoring."""
        eps = torch.randn_like(self.lora_A_mu)
        self._frozen_weight = self.lora_A_mu + self.lora_A_g ** 2 * eps

    def unfreeze_sample(self) -> None:
        """Clear cached A — resume fresh sampling each forward call."""
        self._frozen_weight = None

    def kl_loss(self) -> torch.Tensor:
        """Diagonal KL: q(A) = N(M, diag(G^4)) vs p(A) = N(0, sigma_p^2 * I).

        KL = sum[ (M^2 + sigma^2) / (2*sigma_p^2) - log(sigma/sigma_p) - 0.5 ]
        where sigma = G^2.
        """
        sigma = (self.lora_A_g ** 2).clamp(min=1e-8)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma ** 2 + self.lora_A_mu ** 2) / (2 * prior_var)
            - 0.5
        )
        return kl.sum()


def inject_lora(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """Inject BLoB LoRA adapters into FFN layers of a MiniGPT model.

    1. Freezes all base model parameters.
    2. Replaces MLP.fc and MLP.proj in every block with BLoBLoRALinear.
    3. Only LoRA parameters retain requires_grad=True.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    if lora_config.target == "ffn":
        for block in model.blocks:
            block.mlp.fc = BLoBLoRALinear(
                block.mlp.fc.linear,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                prior_std=lora_config.prior_std,
                init_g=lora_config.init_g,
            )
            block.mlp.proj = BLoBLoRALinear(
                block.mlp.proj.linear,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                prior_std=lora_config.prior_std,
                init_g=lora_config.init_g,
            )

    return model
