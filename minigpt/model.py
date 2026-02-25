from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from minigpt.layers import (
    BayesConfig,
    frozen_bayesian_sample,
    make_linear,
    sum_kl_loss,
    use_mean_weights,
)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    bayes: BayesConfig = field(default_factory=BayesConfig)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, bayes: BayesConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.qkv = make_linear(config.n_embd, 3 * config.n_embd, bayes, bias=config.bias)
        self.proj = make_linear(config.n_embd, config.n_embd, bayes, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embd = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(embd, dim=2)
        q = q.view(bsz, seq_len, self.config.n_head, self.head_size).transpose(1, 2)
        k = k.view(bsz, seq_len, self.config.n_head, self.head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, self.config.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_size**0.5)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, embd)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, bayes: BayesConfig) -> None:
        super().__init__()
        self.fc = make_linear(config.n_embd, 4 * config.n_embd, bayes, bias=config.bias)
        self.proj = make_linear(4 * config.n_embd, config.n_embd, bayes, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, bayes: BayesConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, bayes)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, bayes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks: always deterministic (A1 — only lm_head is Bayesian)
        no_bayes = BayesConfig(enabled=False)
        self.blocks = nn.ModuleList([Block(config, no_bayes) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head: gets the real bayes config
        self.lm_head = make_linear(config.n_embd, config.vocab_size, config.bayes, bias=False)

        # Weight tying (GPT-2 style) — only when lm_head is deterministic
        if not config.bayes.enabled:
            self.lm_head.linear.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_body(self, idx: torch.Tensor) -> torch.Tensor:
        """Run transformer body up to (but not including) lm_head. Returns hidden states."""
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size
        pos = torch.arange(0, seq_len, device=idx.device)

        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)[None, :, :]
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.forward_body(idx)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def kl_loss(self) -> torch.Tensor:
        return sum_kl_loss(self)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        use_mean: bool = False,
    ) -> torch.Tensor:
        ctx = use_mean_weights(self) if use_mean else _nullcontext()
        with ctx:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def frozen_bayesian_sample(self):
        """Context manager: freeze all BayesianLinear layers for coherent generation."""
        return frozen_bayesian_sample(self)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
