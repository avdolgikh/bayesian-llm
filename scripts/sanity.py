#!/usr/bin/env python
"""Trust checks: one command, one PASS / FAIL / SKIP table (spec S9, i2-trust-checks.md).

Each check's right answer comes from outside the code under test: math, saved results,
sklearn or a brute-force pair loop. CPU only, except T1 (GPU, opt-in). Exit code 0 only
when no check FAILs; a SKIP is always printed with its reason. Run it from the repo with the
dev dependencies installed: T8 imports the S1 builder fixture from tests/test_eval_rebuild.py.

    python scripts/sanity.py                 # T2-T9 on the CPU
    python scripts/sanity.py --only T2,T6    # a subset
    python scripts/sanity.py --with-gpu      # also T1: re-score the old D1 blocks (~76 GPU min)
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import io
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import yaml

PLANT_ENV = "SANITY_PLANT_ERROR"  # any value: T2 runs the scorer with a wrong log epsilon (T9)
PLANTED_EPS = 0.1  # the scorer's PROB_EPS is 1e-10
TOL_ZERO, TOL_G, TOL_COV, TOL_SCALE, TOL_AUROC, TOL_REPRO = 1e-6, 1e-4, 0.05, 0.05, 1e-9, 0.01
TOL_CORR = 0.9  # T5: per-entry correlation of the precision with the Hessian diagonal
MI_TWO = math.log(2) + 0.9 * math.log(0.9) + 0.1 * math.log(0.1)  # log 2 - H(0.9, 0.1) nats
GATE_BLOCKS, GATE_SEEDS, GATE_N_SE = 4, 8, 2.0  # T7: noise allowance = GATE_N_SE x SE
CPU = torch.device("cpu")


@dataclass
class Result:
    check: str
    proves: str
    status: str  # PASS, FAIL or SKIP
    reason: str


@dataclass
class GateCase:
    label: str
    config: Path  # the refit YAML: base checkpoint, model shape, data, budget
    state: Path
    expect_within: bool | None  # None: reported only


def _verdict(check: str, proves: str, ok: bool, reason: str) -> Result:
    return Result(check, proves, "PASS" if ok else "FAIL", reason)


def _tiny_gpt(vocab: int, block: int, n_embd: int, seed: int, dropout: float = 0.0):
    from minigpt.model import GPTConfig, MiniGPT
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=1,
                             n_embd=n_embd, dropout=dropout)).eval()


def check_t1(with_gpu: bool) -> Result:
    proves = "the scorer with v1_legacy samplers reproduces the 7 saved D1 AUROCs"
    cfg_path = REPO / "data/i2/repro/i2_eval_repro.yaml"
    if not with_gpu:
        return Result("T1", proves, "SKIP", "GPU check, opt-in: run with --with-gpu (~76 GPU min)")
    if not torch.cuda.is_available() or not cfg_path.exists():
        return Result("T1", proves, "SKIP", f"no CUDA device or no {cfg_path}")
    with tempfile.TemporaryDirectory(prefix="sanity_t1_", ignore_cleanup_errors=True) as tmp:
        return _check_t1_in(Path(tmp), cfg_path, proves)


def _check_t1_in(out: Path, cfg_path: Path, proves: str) -> Result:
    from sklearn.metrics import roc_auc_score
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["scoring"]["out_dir"] = str(out)
    (out / "repro.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for script, sets in (("eval_c_checkpoints.py", "c0,c1,c3,c4_tfb,c4_lap,c2"),
                         ("eval_mc_dropout.py", "mc_dropout")):
        subprocess.run([sys.executable, str(REPO / "scripts" / script), "--eval-config",
                        str(out / "repro.yaml"), "--eval-set", "legacy_d1", "--run-tag", "sanity",
                        "--device", "cuda", "--score-set", sets], check=True, cwd=REPO)
    old = {**torch.load(REPO / "data/d1_scores.pt", weights_only=False),
           **torch.load(REPO / "data/mc_dropout_scores.pt", weights_only=False)}
    diffs = {}
    for s in ("c0", "mc_dropout", "c1", "c3", "c4_tfb", "c4_lap", "c2"):
        new = torch.load(out / f"{s}__legacy_d1__sanity.pt", weights_only=True)
        bi = new["block_index"].numpy()
        y = np.array([d != "stackexchange" for d in new["domain"]], dtype=int)
        new_key, old_key = ("blk_maxprob_unc", "max_prob_unc") if s == "c0" else ("blk_mi", "mi")
        saved = roc_auc_score(old[s]["labels"].numpy().astype(int)[bi],
                              old[s][old_key].double().numpy()[bi])
        diffs[s] = round(roc_auc_score(y, new[new_key].double().numpy()) - saved, 4)
    return _verdict("T1", proves, all(abs(d) <= TOL_REPRO for d in diffs.values()),
                    f"new - saved AUROC: {diffs} (tolerance {TOL_REPRO})")


class _LogitModel(torch.nn.Module):
    """Predicts the same token distribution, softmax(logits), at every position."""

    def __init__(self, vocab: int):
        super().__init__()
        self.config = argparse.Namespace(vocab_size=vocab)
        self.logits = torch.nn.Parameter(torch.zeros(vocab))

    def forward(self, x):
        return self.logits.expand(*x.shape, -1).clone(), None  # a new tensor, as a real model


def _score(model, tok: torch.Tensor, n: int, sampler, plant: bool) -> dict:
    """The real scorer, with `sampler` as its parameter sampler (so apply_sampled_params
    runs) and, if `plant`, a wrong log epsilon planted in the scorer module (T9)."""
    from unittest import mock

    import eval_c_checkpoints as ecc
    with mock.patch.dict(ecc.PARAM_SAMPLERS, {"sanity": sampler}), \
            mock.patch.object(ecc, "PROB_EPS", PLANTED_EPS if plant else ecc.PROB_EPS):
        return ecc.score_block_batch(model, tok[:, :-1], tok[:, 1:], n, CPU, "sanity", None,
                                     seed_b=0, amp=False)


def two_sample_mi(plant: bool) -> dict[str, float]:
    """MI when the two samples give the token distributions (0.9, 0.1) and (0.1, 0.9)."""
    from minigpt.uncertainty import mc_metrics_single
    two = torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log()
    out = _score(_LogitModel(2), torch.zeros(1, 5, dtype=torch.long), 2,
                 lambda state, seed: {"logits": two[seed % 2]}, plant)
    unc = mc_metrics_single(lambda s: two[s].expand(1, 4, 2), 2, 4, 2, CPU)["mi"]
    return {"scorer": out["tok_mi"].mean().item(), "uncertainty": unc.mean().item()}


def check_t2(plant: bool | None = None) -> Result:
    from eval_c_checkpoints import realized_token_scores

    from minigpt.uncertainty import mc_metrics_single
    plant = bool(os.environ.get(PLANT_ENV)) if plant is None else plant
    model = _tiny_gpt(vocab=64, block=16, n_embd=32, seed=0)
    name = "blocks.0.mlp.fc.linear.weight"
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.token_emb.weight.mul_(50.0)  # peaked, far-from-uniform predictions
        w = dict(model.named_parameters())[name]
        fixed = {name: w + 0.1 * torch.randn(w.shape, generator=gen)}  # the same every draw
    tok = torch.randint(0, 64, (3, 17), generator=gen)
    out = _score(model, tok, 8, lambda state, seed: fixed, plant)
    logits = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(2)) * 5.0
    mi_raw = mc_metrics_single(lambda s: logits, 8, 16, 64, CPU)["mi"]
    mi = max(out["tok_mi"].abs().max().item(), mi_raw.abs().max().item())
    g = realized_token_scores(out["logp_real"])["G"].abs().max().item()
    two = two_sample_mi(plant)
    err = max(abs(v - MI_TWO) for v in two.values())
    return _verdict("T2", "MI = G = 0 for identical samples; MI = log 2 - H(0.9) for two (math)",
                    mi < TOL_ZERO and g < TOL_ZERO and err < TOL_G,
                    f"identical: max |MI| {mi:.1e}, max |G| {g:.1e}; two: MI scorer "
                    f"{two['scorer']:.5f}, uncertainty {two['uncertainty']:.5f}, "
                    f"by hand {MI_TWO:.5f}")


def check_t3() -> Result:
    from eval_c_checkpoints import realized_token_scores

    from minigpt.uncertainty import realized_token_gap
    lp = torch.tensor([[math.log(0.9)], [math.log(0.1)]])
    expected = math.log(0.5) - (math.log(0.9) + math.log(0.1)) / 2  # 0.5108 nats
    got = [realized_token_scores(lp)["G"].item(), realized_token_gap(lp)["G"].item()]
    return _verdict("T3", "g for samples giving 0.9 and 0.1 is 0.511 nats (by hand)",
                    max(abs(v - expected) for v in got) < TOL_G,
                    f"scorer {got[0]:.5f}, uncertainty {got[1]:.5f}, by hand {expected:.5f}")


def tfb_cov_error(sampler_version: str, n_draws: int = 8192) -> float:
    """Relative Frobenius error of cov(B dA) against sigma^2 U U^T (the external review's check)."""
    from minigpt.tfb import TFBState, sample_tfb_params
    B = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, -1.0]])
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    state = TFBState(sigma_q=0.1, svd_cache={"layer": (U, S, Vh)},
                     a_map={"layer.lora_A": torch.zeros(2, 1)}, param_names=["layer.lora_A"],
                     epsilon=0.1, anchor_loss=1.0, sampler_version=sampler_version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # v1_legacy warns on every draw
        draws = torch.stack([(B @ sample_tfb_params(state, seed=s)["layer.lora_A"]).flatten()
                             for s in range(n_draws)]).double().numpy()
    expected = (0.1 ** 2 * U @ U.T).double().numpy()
    return float(np.linalg.norm(np.cov(draws, rowvar=False) - expected) / np.linalg.norm(expected))


def check_t4() -> Result:
    v2, v1 = tfb_cov_error("v2"), tfb_cov_error("v1_legacy")
    return _verdict("T4", "TFB noise B dA has covariance sigma^2 U U^T; v1_legacy fails it",
                    v2 < TOL_COV <= v1, f"error v2 {v2:.1%}, v1_legacy {v1:.1%} (8,192 draws)")


def laplace_scale_ratios(tokens_per_seq: int | None = None) -> dict[str, float]:
    return {k: r for k, (r, _) in laplace_hessian_check(tokens_per_seq).items()}


@functools.cache  # deterministic; the tests and check_t5 share one run
def laplace_hessian_check(tokens_per_seq: int | None = None) -> dict[str, tuple[float, float]]:
    """Per group: (sum of (v2 precision - prior) / sum of the exact Hessian diagonal, and the
    per-entry correlation of the two, so an error that moves curvature between entries shows).

    The tiny model is a bigram (no position embedding, zero attention output) and its data
    is sampled from itself, so the empirical Fisher and the Hessian agree in expectation.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from minigpt.laplace import fit_laplace, posterior_std, scale_laplace_state, select_params
    from minigpt.train import get_batch
    V, T, n_batches, bsz, prior = 8, 16, 256, 16, 1e-12
    model = _tiny_gpt(vocab=V, block=T, n_embd=4, seed=0)
    with torch.no_grad():
        model.pos_emb.weight.zero_()
        model.token_emb.weight.normal_(0.0, 0.5)
        for b in model.blocks:
            b.attn.proj.linear.weight.zero_()
            b.attn.proj.linear.bias.zero_()
            b.mlp.fc.linear.weight.normal_(0.0, 0.5)
            b.mlp.proj.linear.weight.normal_(0.0, 0.3)
        probs = torch.softmax(model(torch.arange(V).unsqueeze(1))[0][:, 0], dim=-1)
    gen, stream = torch.Generator().manual_seed(3), [0]
    for _ in range(30_000):
        stream.append(int(torch.multinomial(probs[stream[-1]], 1, generator=gen)))
    data = torch.tensor(stream)
    sel = {**select_params(model, "head"), **select_params(model, "ffn")}
    torch.manual_seed(0)
    v2 = scale_laplace_state(fit_laplace(model, data, T, bsz, sel, n_batches, damping=0.0),
                             n_data_seqs=n_batches * bsz, tokens_per_seq=tokens_per_seq or T,
                             prior_prec=prior)
    torch.manual_seed(0)  # the same windows that fit_laplace drew
    x, y = (torch.cat(t) for t in zip(*[get_batch(data, T, bsz, CPU) for _ in range(n_batches)]))
    with sdpa_kernel(SDPBackend.MATH):  # the fused CPU kernel has no second derivative
        nll = torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, V), y.reshape(-1),
                                                reduction="sum")
    grads = torch.autograd.grad(nll, list(sel.values()), create_graph=True)
    std, out = posterior_std(v2), {}
    for (name, p), g in zip(sel.items(), grads):
        flat = g.reshape(-1)
        h = np.array([torch.autograd.grad(flat[i], p, retain_graph=True)[0].reshape(-1)[i].item()
                      for i in range(flat.numel())])
        prec = (std[name] ** -2 - prior).reshape(-1).double().numpy()
        out[name] = (float(prec.sum() / h.sum()), float(np.corrcoef(prec, h)[0, 1]))
    return out


def check_t5() -> Result:
    res = laplace_hessian_check()
    worst = max((r for r, _ in res.values()), key=lambda r: abs(r - 1.0))
    corr = min(c for _, c in res.values())
    return _verdict("T5", "Laplace precision N T^2 F_hat = exact Hessian diagonal (autograd)",
                    all(abs(r - 1.0) < TOL_SCALE and c > TOL_CORR for r, c in res.values()),
                    f"{len(res)} parameter groups, worst sum ratio {worst:.3f}, lowest "
                    f"per-entry correlation {corr:.3f}")


def brute_force_auroc(scores, labels, weights) -> float:
    """Weighted AUROC by the pair loop: P(score OOD > score ID), ties count one half."""
    s, y, w = (np.asarray(a, dtype=np.float64) for a in (scores, labels, weights))
    diff = s[y == 1][:, None] - s[y == 0][None, :]
    pair_w = w[y == 1][:, None] * w[y == 0][None, :]
    return float((pair_w * ((diff > 0) + 0.5 * (diff == 0))).sum() / pair_w.sum())


def check_t6() -> Result:
    from sklearn.metrics import roc_auc_score

    from minigpt.uncertainty import _WeightedROC, auroc
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.normal(size=400) + 0.7 * y, 1)  # rounding makes many ties
    w = 1.0 / rng.integers(1, 4, 400)  # 1 / k_d
    ref, ref_1 = brute_force_auroc(s, y, w), brute_force_auroc(s, y, np.ones(400))
    fpr, tpr = _WeightedROC(s, y).curves(w[None, :])
    errs = {"auroc": auroc(s, y, sample_weight=w) - ref, "unweighted": auroc(s, y) - ref_1,
            "bootstrap path": _WeightedROC.auroc(fpr, tpr)[0] - ref,
            "sklearn": roc_auc_score(y, s, sample_weight=w) - ref}
    err = max(abs(v) for v in errs.values())
    return _verdict("T6", "weighted AUROC = brute-force pair loop and sklearn (with ties)",
                    err < TOL_AUROC, f"max |diff| {err:.1e} over {', '.join(errs)}")


def within_budget(delta_nll: float, ell0: float, budget_rel: float, se: float) -> bool:
    """The refit rule, |Delta-NLL| <= budget x l_0, plus GATE_N_SE standard errors of this
    measurement for sampling noise. No other margin."""
    return abs(delta_nll) <= budget_rel * ell0 + GATE_N_SE * se


def _budget(cfg: dict) -> float:
    """TFB: epsilon_rel. Laplace: rho_TFB from the TFB record, read and checked by the refit's
    own rule (if absent: its ceiling epsilon_rel from the TFB YAML), times 1 + match_tol."""
    if cfg["method"] == "tfb":
        return cfg["tfb"]["epsilon_rel"]
    from minigpt.posthoc_refit import _read_tfb_budget
    rec = REPO / cfg["laplace"]["tfb_record_path"]
    if rec.exists():
        rho = _read_tfb_budget(rec)["rho_tfb"]
    else:
        tfb = yaml.safe_load((REPO / "configs/i2_posthoc_c4_tfb.yaml").read_text("utf-8"))
        rho = tfb["tfb"]["epsilon_rel"]
    return rho * (1.0 + cfg["laplace"]["match_tol"])


def _familiar_batches(cfg: dict, pile_dir: Path | None = None) -> list:
    """GATE_BLOCKS windows from the start of the base model's own ID val slice, split from the
    real cache lengths the way load_pile_data joins the domains."""
    d, T = cfg["data"], cfg["model"]["block_size"]
    pile_dir = pile_dir or REPO / "data/pile"
    caches = [torch.load(pile_dir / f"{dom}_{d['pile_id_tokens']}.pt", mmap=True,
                         weights_only=True) for dom in d["pile_id_domains"]]
    total = sum(len(t) for t in caches)
    start = int(total * (1 - d["val_fraction"] - d["test_fraction"]))
    if start + 4 * T * (GATE_BLOCKS - 1) + T + 1 > int(total * (1 - d["test_fraction"])):
        raise ValueError("the ID val slice is too short for the T7 blocks")

    def tokens(a: int, b: int) -> torch.Tensor:  # [a, b) of the joined caches
        parts, off = [], 0
        for t in caches:
            if max(a - off, 0) < min(b - off, len(t)):
                parts.append(t[max(a - off, 0):min(b - off, len(t))])
            off += len(t)
        return torch.cat(parts).long()

    w = torch.stack([tokens(start + 4 * T * i, start + 4 * T * i + T + 1)
                     for i in range(GATE_BLOCKS)])
    return [(w[i:i + 2, :-1], w[i:i + 2, 1:]) for i in range(0, GATE_BLOCKS, 2)]


def check_t7(check: str, cases: list[GateCase], proves: str = "") -> Result:
    proves = proves or "sampled weights keep familiar-text NLL inside the method's budget"
    cfgs = {c.label: yaml.safe_load(c.config.read_text(encoding="utf-8")) for c in cases}
    for cfg in cfgs.values():
        cfg["base"]["base_checkpoint"] = str(REPO / cfg["base"]["base_checkpoint"])
    missing = [p.as_posix().removeprefix(REPO.as_posix() + "/") for c in cases
               for p in (c.state, Path(cfgs[c.label]["base"]["base_checkpoint"])) if not p.exists()]
    if missing:
        return Result(check, proves, "SKIP", "stored states or bases absent: " + ", ".join(missing))
    from minigpt.laplace import load_laplace_state, sample_laplace_params
    from minigpt.posthoc_refit import load_posthoc_base, measure_delta_nll
    from minigpt.tfb import load_tfb_state, sample_tfb_params
    ok, notes = True, []
    for c in cases:
        cfg, tfb = cfgs[c.label], cfgs[c.label]["method"] == "tfb"
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model, _ = load_posthoc_base(cfg, device=CPU)
            state = (load_tfb_state if tfb else load_laplace_state)(c.state, map_location="cpu")
            sampler = sample_tfb_params if tfb else sample_laplace_params
            m = measure_delta_nll(model, lambda s: sampler(state, seed=s),
                                  _familiar_batches(cfg), range(GATE_SEEDS))
        budget = _budget(cfg)
        within = within_budget(m["delta_nll"], m["ell0"], budget, m["se"])
        ok &= c.expect_within is None or within == c.expect_within
        notes.append(f"{c.label}: NLL {m['ell0']:.2f}->{m['mean_nll']:.2f} (SE {m['se']:.3f}), "
                     f"{'inside' if within else 'over'} budget {budget:.4f}"
                     + (" (reported, not judged)" if c.expect_within is None else ""))
    return _verdict(check, proves, ok, f"{GATE_BLOCKS} val blocks, {GATE_SEEDS} draws, rule "
                    f"|dNLL| <= budget x l0 + {GATE_N_SE:g} SE: " + "; ".join(notes))


def _gate_cases() -> dict[str, list[GateCase]]:
    old, new = [], []
    for cell, state, expect in (("c2", "laplace_state.pt", False),
                                ("c4_lap", "laplace_state.pt", False),
                                ("c4_tfb", "tfb_state.pt", None)):
        config = REPO / f"configs/i2_posthoc_{cell}.yaml"
        out_dir = yaml.safe_load(config.read_text(encoding="utf-8"))["out_dir"]
        old.append(GateCase(f"{cell} pre-fix", config, REPO / "data/checkpoints" / cell / state,
                            expect))
        new.append(GateCase(f"{cell} refit", config, REPO / out_dir / state, True))
    return {"T7a": old, "T7b": new}


def _t8_checkpoints(tmp: Path, train: torch.Tensor) -> tuple[Path, Path]:
    """A tiny byte-level base trained 150 steps, and a BLoB checkpoint with a rotated B."""
    from minigpt.lora import LoRAConfig, inject_lora
    base = _tiny_gpt(vocab=256, block=256, n_embd=16, seed=0, dropout=0.1).train()
    opt = torch.optim.AdamW(base.parameters(), lr=3e-3)
    for _ in range(150):
        w = torch.stack([train[i:i + 257] for i in torch.randint(0, len(train) - 257, (8,))])
        opt.zero_grad()
        base(w[:, :-1], w[:, 1:].contiguous())[1].backward()
        opt.step()
    blob = inject_lora(copy.deepcopy(base.eval()), LoRAConfig(rank=4, alpha=8.0, target="ffn"))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for mod in (m for m in blob.modules() if hasattr(m, "lora_A_mu")):
            u, _ = torch.linalg.qr(torch.randn(mod.lora_B.shape[0], 4, generator=g))
            mod.lora_B.copy_(u @ torch.diag(torch.logspace(0, -0.5, 4)))
            mod.lora_A_mu.copy_(torch.randn(mod.lora_A_mu.shape, generator=g) * 0.1)
    for model, name in ((base, "base.pt"), (blob, "blob.pt")):
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, tmp / name)
    return tmp / "base.pt", tmp / "blob.pt"


def check_t8() -> Result:
    proves = "the real builder's manifest feeds run_refit for C4-TFB, C4-LAP and C2"
    sys.path.insert(0, str(REPO / "tests"))
    from test_eval_rebuild import ID_CACHE, OOD_CACHE, Fixture  # the S1 builder fixture

    with tempfile.TemporaryDirectory(prefix="sanity_t8_", ignore_cleanup_errors=True) as tmp:
        return _check_t8_in(Path(tmp), proves, Fixture, ID_CACHE, OOD_CACHE)


def _check_t8_in(tmp: Path, proves: str, Fixture, ID_CACHE: int, OOD_CACHE: int) -> Result:
    from minigpt import data as data_mod
    from minigpt.laplace import fit_laplace, save_laplace_state, select_params
    from minigpt.posthoc_refit import load_posthoc_base, load_refit_data, run_refit
    fx = Fixture(tmp)
    report = fx.build()
    if report["status"] != "PASS":
        return Result("T8", proves, "FAIL", f"builder: {report['failures']}")
    hn = torch.load(tmp / "pile" / f"hackernews_{ID_CACHE}.pt", weights_only=True)
    base, blob = _t8_checkpoints(tmp, hn[: int(0.8 * ID_CACHE)])
    old_dir, data_mod.DATA_DIR, done = data_mod.DATA_DIR, tmp, {}
    try:
        for cell, ckpt, domains, fit_domain in (
                ("c4_tfb", blob, ["hackernews"], "hackernews"),
                ("c4_lap", blob, ["hackernews"], "hackernews"),
                ("c2", base, ["wikipedia_en", "stackexchange"], "stackexchange")):
            cfg = yaml.safe_load((REPO / f"configs/i2_posthoc_{cell}.yaml").read_text("utf-8"))
            cfg.update(device="cpu", autocast="none", out_dir=str(tmp / "out" / cell))
            cfg["base"]["base_checkpoint"] = str(ckpt)
            cfg["model"].update(n_layer=2, n_head=1, n_embd=16)
            cfg["data"].update(pile_id_domains=domains, pile_id_tokens=ID_CACHE,
                               pile_ood_tokens=OOD_CACHE)
            cfg["fit"].update(manifest_path=str(tmp / "eval_i2" / "manifest.jsonl"),
                              fit_domain=fit_domain, n_fit_blocks=4, batch_size=4)
            cfg.get("lora", {}).update(rank=4, alpha=8.0)
            with contextlib.redirect_stdout(io.StringIO()):
                if cfg["method"] == "tfb":
                    cfg["tfb"].update(n_search_samples=4, n_delta_samples=4)
                else:
                    lap = cfg["laplace"]
                    lap.update(n_curvature_batches=2, curvature_batch_size=4, n_delta_samples=4,
                               n_data_seqs=int(0.8 * ID_CACHE * len(domains)) // 256,
                               bisect_max_steps=4, reference_state_path=str(tmp / f"{cell}.pt"),
                               tfb_record_path=str(tmp / "out/c4_tfb/fit_record.json"))
                    model, _ = load_posthoc_base(cfg)
                    torch.manual_seed(lap["curvature_seed"])
                    save_laplace_state(fit_laplace(
                        model, load_refit_data(cfg).train, 256, 4,
                        select_params(model, lap["selection_mode"]), 2, damping=1.0),
                        lap["reference_state_path"])
                res = run_refit(cfg)
            state = tmp / "out" / cell / f"{cfg['method']}_state.pt"
            done[cell] = (res.exit_code, res.record_path.exists() and state.exists())
    except Exception as exc:  # what this check exists to catch (defect D8)
        return Result("T8", proves, "FAIL", f"{type(exc).__name__}: {exc}"[:200])
    finally:
        data_mod.DATA_DIR = old_dir
    return _verdict("T8", proves, all(c == 0 and w for c, w in done.values()),
                    ", ".join(f"{k} exit {c}, files {'yes' if w else 'NO'}"
                              for k, (c, w) in done.items()))


def check_t9() -> Result:
    r = check_t2(plant=True)
    return _verdict("T9", "a bug planted in the scorer (log epsilon 0.1) turns T2 into FAIL",
                    r.status == "FAIL", f"planted T2: {r.status}, {r.reason}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trust checks (spec S9): one PASS/FAIL/SKIP table.")
    ap.add_argument("--with-gpu", action="store_true", help="also run T1 on the GPU")
    ap.add_argument("--only", default="", help="comma list of checks, e.g. T2,T6,T7")
    args = ap.parse_args(argv)
    gates = _gate_cases()
    checks = {"T1": lambda: check_t1(args.with_gpu), "T2": check_t2, "T3": check_t3,
              "T4": check_t4, "T5": check_t5, "T6": check_t6,
              "T7a": lambda: check_t7("T7a", gates["T7a"], "the gate rejects the pre-fix "
                                      "C2 and C4-LAP states (NLL far over budget)"),
              "T7b": lambda: check_t7("T7b", gates["T7b"], "every refit (v2) state keeps "
                                      "familiar-text NLL inside its budget"),
              "T8": check_t8, "T9": check_t9}
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    rows, t0 = [], time.time()
    for name, fn in checks.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        t = time.time()
        try:
            r = fn()
        except Exception as exc:
            r = Result(name, "-", "FAIL", f"{type(exc).__name__}: {exc}"[:200])
        r.reason += f" [{time.time() - t:.1f}s]"
        rows.append(r)
    head = Result("check", "what it proves", "status", "reason")
    wid = [max(len(getattr(r, k)) for r in [head, *rows]) for k in ("check", "proves", "status")]
    for r in [head, *rows]:
        print(f"{r.check:<{wid[0]}} | {r.proves:<{wid[1]}} | {r.status:<{wid[2]}} | {r.reason}")
    failed = [r.check for r in rows if r.status == "FAIL"]
    print(f"{len(rows)} checks in {time.time() - t0:.0f}s: "
          + (f"FAIL in {', '.join(failed)}" if failed else "no failures"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
