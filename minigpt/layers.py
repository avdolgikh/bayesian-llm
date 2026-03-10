from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BayesConfig:
    enabled: bool = False
    prior_std: float = 1.0
    init_rho: float = -1.0


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
        init_rho: float = -1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.init_rho = init_rho

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self._frozen_weight: torch.Tensor | None = None
        self._frozen_bias: torch.Tensor | None = None
        self._use_mean: bool = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)
        nn.init.constant_(self.weight_rho, self.init_rho)
        if self.bias_mu is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, self.init_rho)

    def _sample(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mean:
            return self.mean_forward(x)
        if self._frozen_weight is not None:
            return F.linear(x, self._frozen_weight, self._frozen_bias)
        weight = self._sample(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            bias = None
        return F.linear(x, weight, bias)

    def mean_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using posterior mean (mu) only — deterministic."""
        return F.linear(x, self.weight_mu, self.bias_mu)

    def mean_forward_with_variance(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mu weights + closed-form per-output variance.

        Returns (output, variance) where variance[..., k] = sum_j sigma_kj^2 * x_j^2.
        """
        output = F.linear(x, self.weight_mu, self.bias_mu)
        sigma = F.softplus(self.weight_rho)
        variance = F.linear(x**2, sigma**2)
        return output, variance

    def freeze_sample(self) -> None:
        """Sample weights once and cache for subsequent forward calls."""
        self._frozen_weight = self._sample(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            self._frozen_bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            self._frozen_bias = None

    def unfreeze_sample(self) -> None:
        """Clear cached weights — resume fresh sampling on every forward call."""
        self._frozen_weight = None
        self._frozen_bias = None

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
        return BayesianLinear(
            in_features, out_features,
            prior_std=bayes.prior_std, init_rho=bayes.init_rho, bias=bias,
        )
    return DeterministicLinear(in_features, out_features, bias=bias)


def sum_kl_loss(module: nn.Module) -> torch.Tensor:
    device = next(module.parameters(), torch.tensor(0.0)).device
    total = torch.tensor(0.0, device=device)
    for child in module.modules():
        if isinstance(child, BayesianModule):
            total = total + child.kl_loss()
    return total


def sigma_summary(model: nn.Module) -> dict[str, float]:
    """Compute aggregate sigma statistics across all stochastic Bayesian layers.

    Collects softplus(rho) from BayesianLinear and G^2 from BLoBLoRALinear.
    Returns dict with keys: sigma_mean, sigma_std, sigma_min, sigma_max,
    sigma_median, sigma_p5, sigma_p25, sigma_p75, sigma_p95.
    Returns empty dict if no stochastic layers found.
    """
    all_sigmas = []
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            sigma = F.softplus(module.weight_rho).detach()
            all_sigmas.append(sigma.flatten())
        elif hasattr(module, "lora_A_g"):  # BLoBLoRALinear: sigma = G^2
            sigma = (module.lora_A_g ** 2).detach()
            all_sigmas.append(sigma.flatten())
    if not all_sigmas:
        return {}
    combined = torch.cat(all_sigmas)
    pcts = torch.tensor([0.05, 0.25, 0.75, 0.95], device=combined.device)
    quantiles = torch.quantile(combined.float(), pcts)
    return {
        "sigma_mean": combined.mean().item(),
        "sigma_std": combined.std().item(),
        "sigma_min": combined.min().item(),
        "sigma_max": combined.max().item(),
        "sigma_median": combined.median().item(),
        "sigma_p5": quantiles[0].item(),
        "sigma_p25": quantiles[1].item(),
        "sigma_p75": quantiles[2].item(),
        "sigma_p95": quantiles[3].item(),
    }


@contextmanager
def frozen_bayesian_sample(module: nn.Module):
    """Context manager: freeze all stochastic layers for coherent generation.

    Works with BayesianLinear and any module that implements freeze_sample()
    (e.g. BLoBLoRALinear).
    """
    stochastic_layers = [m for m in module.modules() if hasattr(m, "freeze_sample")]
    for layer in stochastic_layers:
        layer.freeze_sample()
    try:
        yield
    finally:
        for layer in stochastic_layers:
            layer.unfreeze_sample()


@contextmanager
def use_mean_weights(module: nn.Module):
    """Context manager: use mean weights for all stochastic layers.

    Works with BayesianLinear and any module that has a _use_mean flag
    (e.g. BLoBLoRALinear).
    """
    stochastic_layers = [m for m in module.modules() if hasattr(m, "_use_mean")]
    for layer in stochastic_layers:
        layer._use_mean = True
    try:
        yield
    finally:
        for layer in stochastic_layers:
            layer._use_mean = False
