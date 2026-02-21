from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BayesConfig:
    enabled: bool = False
    prior_std: float = 1.0
    kl_weight: float = 1.0


class BayesianModule(nn.Module):
    def kl_loss(self) -> torch.Tensor:
        device = next(self.parameters(), torch.tensor(0.0)).device
        return torch.tensor(0.0, device=device)


class DeterministicLinear(BayesianModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BayesianLinear(BayesianModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)
        nn.init.constant_(self.weight_rho, -5.0)
        if self.bias_mu is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def _sample(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            bias = None
        return F.linear(x, weight, bias)

    def _kl_normal(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        prior_var = self.prior_std**2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma**2 + mu**2) / (2 * prior_var)
            - 0.5
        )
        return kl.sum()

    def kl_loss(self) -> torch.Tensor:
        kl = self._kl_normal(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            kl = kl + self._kl_normal(self.bias_mu, self.bias_rho)
        return kl


def make_linear(
    in_features: int,
    out_features: int,
    bayes: BayesConfig,
    bias: bool = True,
) -> BayesianModule:
    if bayes.enabled:
        return BayesianLinear(in_features, out_features, prior_std=bayes.prior_std, bias=bias)
    return DeterministicLinear(in_features, out_features, bias=bias)


def sum_kl_loss(module: nn.Module) -> torch.Tensor:
    device = next(module.parameters(), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    for child in module.modules():
        if isinstance(child, BayesianModule):
            total = total + child.kl_loss()
    return total
