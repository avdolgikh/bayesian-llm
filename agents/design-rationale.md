# Design Rationale

Why specific technical decisions were made at each milestone. For results and numbers, see `report.md`.

---

## Bayesian Layer Strategy

1. Output head (A1) — simplest, proves pipeline
2. FFN layers (A2) — strongest epistemic signal (FFN stores factual knowledge)
3. Attention V (A3) — closed (negative, adds noise without improving discrimination)

---

## A1: Sigma window and output head ceiling

Optimal sigma ~ 0.1-0.3. Below -> posterior collapse. Above -> uniform noise. Output head ceiling at MI ratio 1.2-1.4x (vocabulary-level uncertainty only, factual knowledge lives in FFN).

## A2: Why init_rho=-2

init_rho=-3: sigma frozen at init. init_rho=-2: sigma spread [0.036, 0.966], model differentiates which weights need certainty. FFN detects topic-level uncertainty (Business ~ Sci/Tech), not vocabulary-level (A1: Sci/Tech >> Business). 4.2M Bayesian params outperform A1's 25.7M.

## B1: Why diagonal Laplace fails for LMs

Curvature at convergence is flat (~1e-5). With damping=1.0: posterior variance ~ 1.0 everywhere -> isotropic perturbations -> MI ratio 1.00x. Both identity-curvature (ICLA) and per-sample Fisher produce the same failure. ICLA works for image classification (10-200 classes) but fails for LM (50K vocab).

## B2: Category-split design for BLoB LoRA

Pretrain World -> LoRA fine-tune Sports -> OOD Business/Sci-Tech. Ensures OOD is truly unseen by both base and adapter. Key HPs: rank=16, alpha=32, lr=3e-4, init_g=0.1, prior_std=0.2.

## B3: Why SVD-structured variance works and curvature doesn't

TFB uses SVD of B matrix: Omega_ij = sigma_q / S_i (directions with large singular values get small variance). Diagonal Laplace uses loss curvature (flat at convergence). SVD captures geometric structure of LoRA subspace. Curvature carries no directional information.

## C: Why LoRA scales better than full-weight

Hypothesis: LoRA's rank-16 subspace constrains posteriors to meaningful directions rather than spreading uncertainty across all parameters. As model size grows, this constraint becomes an advantage. Full-weight variational must learn meaningful sigma for millions of parameters — harder at scale.

## C: Infrastructure fixes

- Token-level shuffle in Pile loader destroyed sequential structure -> removed
- Residual projection scaling for deep models (`std / sqrt(2*n_layer)`) -> added
- Flash Attention (`F.scaled_dot_product_attention`) -> replaced manual attention
- Patience early-stop + per-step NaN detection -> added to train loop
- Agent subprocess: max-turns 5, structured_output envelope, stdin prompt delivery, UTF-8 encoding
- torch.quantile overflow for >16M elements -> random subsample
