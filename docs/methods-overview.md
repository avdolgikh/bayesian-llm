# Bayesian Methods for LLM Uncertainty: A Plain-English Overview

This page explains the main ways to estimate **epistemic uncertainty** in language models by treating
the weights as uncertain. It groups dozens of papers into a few method families, compares them on
simple dimensions, and says which ones this project tests. It is written for students and engineers,
and covers papers found up to September 2026. Every paper claim has a reference in
[section 9](#9-references).

Contents:
[1. The problem](#1-the-problem-in-five-lines) ·
[2. The big picture](#2-the-big-picture) ·
[3. Method cards](#3-method-cards) ·
[4. Non-Bayesian competitors](#4-non-bayesian-competitors) ·
[5. Timeline](#5-timeline-how-the-field-moved) ·
[6. Existing comparisons](#6-existing-comparisons-and-where-this-project-fits) ·
[7. Open problems](#7-open-problems) ·
[8. Two talks](#8-two-talks-that-shaped-this-project) ·
[9. References](#9-references)

---

## 1. The problem in five lines

1. A language model predicts the next token and always gives probabilities, even on text unlike anything it has read.
2. **Aleatoric uncertainty** comes from the text itself: "My favourite colour is ..." has many good answers. More data does not remove it.
3. **Epistemic uncertainty** comes from the model: it has not seen enough data like this, for example a page from a rare legal field. More data would reduce it.
4. One probability mixes both kinds. The methods on this page try to pull out the epistemic part.
5. Their shared trick: keep many plausible versions of the weights, not one, and measure how much the versions disagree.

**Everyday picture.** Ask five doctors about a patient. If all five say "A or B, 50/50", the case is
truly unclear (aleatoric). If each one confidently names a different disease, the case is outside
their experience (epistemic).

### Words used on this page

| Word | Meaning |
|---|---|
| Weights | The numbers inside the network, learned in training |
| Posterior | A spread over weight values ("these values are plausible, those are not"), instead of one fixed value per weight |
| Bell curve (Gaussian) | A spread centred on one value, where values far from it are unlikely |
| Covariance | How the spreads of different weights move together |
| Forward pass | One run of the model over a text. Most methods here need N passes, one per weight version |
| Hidden state | The model's internal vector for each token |
| FFN | The feed-forward layers inside each transformer block. They hold about two-thirds of the weights inside the blocks (about half of all weights in Pythia-410M) |
| LoRA | Low-Rank Adaptation: freeze the big model and train two thin matrices, B and A, whose product is the weight change. Only a tiny share of weights is trained |
| Rank (of LoRA) | How many thin directions the adapter has, often 8 |
| Fine-tuning | Extra training of a pretrained model on new data |
| Post-hoc | Added after training is finished |
| ID / OOD | In-distribution (like the training data) / out-of-distribution (unlike it) |
| Perplexity | How surprised the model is by a text; lower means less surprised |
| Calibration | Whether stated confidence matches accuracy: "80% sure" should be right 80% of the time. ECE (expected calibration error) measures the gap |
| NLL | Negative log-likelihood: how surprised the model is by the right answer; lower is better |
| AUROC | The chance that a random OOD text gets a higher uncertainty score than a random ID text. 0.5 is a coin flip, 1.0 is perfect |
| Winogrande-S | A small commonsense multiple-choice test used in many of these papers |
| Surface cues | How a text looks (symbols, length, markup) rather than what it says |

### How this project turns weight versions into a score

Most methods in section 3 differ mainly in **how they produce the N weight versions**. This project
scores all of them the same way. The main score is the realized-token disagreement *g*: how much the
versions disagree about the token that actually comes next. Next to it we report the classic
disagreement score, mutual information (MI):

$$\mathrm{MI} = H\Big[\tfrac{1}{N}\textstyle\sum_{i} p_i\Big] - \tfrac{1}{N}\textstyle\sum_{i} H[p_i]$$

In words: $p_i$ is the next-token prediction of weight version $i$, and $H$ is entropy (how spread out a
prediction is). The first term is the uncertainty of the averaged prediction (total). The second is
the average uncertainty of each version on its own (aleatoric). The difference, MI, is the
disagreement between versions (epistemic). If all versions say the same thing, MI is 0, even when
each version is unsure. Both scores, and the metrics used to judge them, are in the
[metrics guide](metrics-guide.md). Most method papers themselves report calibration (ECE, NLL) of the
averaged prediction, not MI.

---

## 2. The big picture

### Table A: when and where the uncertainty lives

| Method | Family | When the spread is added | Which weights are uncertain | First appeared | This project |
|---|---|---|---|---|---|
| MC dropout | Random masks | At test time (needs dropout layers) | Units with dropout | 2015 (ICML 2016) | Compared |
| Bayes by Backprop | Variational | During training | All, or chosen layers (e.g. FFN) | 2015 | Compared (FFN weights) |
| IVON | Variational | During training (it is the optimizer) | All | 2024 | Not compared |
| Laplace | Laplace | After training | All, chosen layers, or last layer | 1992; revived late 2010s; easy since 2021 | Compared (FFN weights, diagonal curvature) |
| Laplace-LoRA | Laplace | After fine-tuning | LoRA adapters (A and B) | 2023 | Diagonal version on matrix A only |
| BLoB | Variational LoRA | During fine-tuning | LoRA matrix A | 2024 | Compared |
| ScalaBL | Variational LoRA | During fine-tuning | A few numbers per adapter (a subspace) | 2025 | Maybe, if time allows |
| Bayesian-LoRA (2026) | Variational LoRA | During fine-tuning | LoRA factors, correlated | 2026 | Not compared |
| TFB | Training-free | After fine-tuning, no gradient steps | LoRA adapters | 2024 | Compared |
| Deep ensemble | Ensemble | Training M models | All | 2016 (NeurIPS 2017) | Not compared (too costly) |
| LoRA ensemble | Ensemble | Fine-tuning M adapters | LoRA adapters | 2023 | Compared (3 adapters) |
| SWAG / SWAG-LoRA | Trajectory | End of training | All / LoRA | 2019 / 2024 | Not compared |
| Bayesian last layer | Variational or Laplace | During or after training | Output layer only | Older idea; VBLL 2024 | Not compared |

### Table B: what it costs and what the spread looks like

| Method | Extra training | Test-time passes | Extra memory | Shape of the weight spread |
|---|---|---|---|---|
| MC dropout | None, if the model was trained with dropout | N | None | Random on/off masks |
| Bayes by Backprop | Retrain with a new loss; twice the parameters to learn | N | A standard deviation per weight | Independent bell curve per weight |
| IVON | About the cost of AdamW | N | Variance kept in optimizer state | Independent bell curve per weight |
| Laplace | One pass over data to measure curvature | N, or 1 pass plus gradients per output (linearized; practical only with few outputs, such as answer letters) | Curvature per weight or per layer block | Bell curve around the trained weights |
| Laplace-LoRA | One pass over data | N, or linearized (as above) | Curvature of the adapter only | Bell curve around the trained adapter |
| BLoB | About one LoRA fine-tune | N | A standard deviation per entry of A | Independent bell curve per entry of A |
| ScalaBL | About one LoRA fine-tune | N | About 1,000 numbers at 7B | Bell curve over a tiny subspace |
| Bayesian-LoRA | About 1.2x a LoRA fine-tune | N | About 0.42M numbers | Correlated, low-dimensional |
| TFB | A short search for one noise size | N | One noise size | Same-size noise in every direction the adapter can change |
| Deep ensemble | M full trainings | M | M full models | M separate points |
| LoRA ensemble | M adapter fine-tunes | M | M small adapters | M separate points |
| SWAG | A few extra epochs, recording snapshots | N | Mean, diagonal and a few directions | Bell curve fitted to snapshots |
| Bayesian last layer | About 1x | 1 backbone pass | Covariance of the output layer | Bell curve over the output layer |

Cost figures for ScalaBL and Bayesian-LoRA are the authors' own (abstracts of their papers).
N is usually 3 to 20. Methods with N or M passes cost about N or M times the compute of one normal
forward pass. Batching can hide part of the wall-clock time on a GPU that is not fully busy. Methods
whose randomness sits only in the last layer (last-layer TFB, Bayesian last layers) can share the rest
of the pass. Methods with learned means (Bayes by Backprop, IVON, BLoB) can also run one pass at the
mean weights for normal use, but that pass gives no uncertainty score.

### The family tree

```
Weight-space (Bayesian) uncertainty
|
+-- Learn the spread while training (variational inference)
|     +-- full weights:  Bayes by Backprop (2015) -> IVON (2024)
|     +-- LoRA adapters: BLoB (2024) -> ScalaBL (2025) -> Bayesian-LoRA (2026);
|                        IVON-LoRA (2024); DALorRA (2026)
|
+-- Fit the spread after training from curvature (Laplace)
|     +-- full weights:  MacKay (1992) -> Laplace Redux (2021)
|     +-- LoRA adapters: Laplace-LoRA (2023)
|
+-- Set the spread by a simple rule, no training
|     +-- MC dropout (2015): random masks
|     +-- TFB (2024): largest noise that keeps performance (loss or accuracy)
|     +-- TokUR (2025): fixed noise
|
+-- Collect several solutions
      +-- independent runs: deep ensembles (2016) -> LoRA ensembles (2023)
      +-- one training path: SWA (2018) -> SWAG (2019) -> SWAG-LoRA (2024)
```

Arrows show time order, not direct descent.

### Rough guide: which family fits which situation

This is our judgement, not a measured result.

| Your situation | Families that fit |
|---|---|
| You can train or retrain the whole model | Variational full weights (Bayes by Backprop, IVON), deep ensemble |
| You fine-tune with LoRA anyway | BLoB, ScalaBL, LoRA ensemble |
| You already have a trained adapter and no training budget | TFB, Laplace-LoRA |
| The model was trained with dropout | MC dropout (free to try) |
| You need one forward pass at test time | Bayesian last layer, or a non-Bayesian score (section 4) |

---

## 3. Method cards

Each card has the same parts: idea, analogy, strengths, caveats, when and why it appeared, key papers,
and whether this project compares it. The "when and why" lines are our reading of the history, except
where a paper's own words are cited. Most methods produce N weight versions (M for ensembles), which
this project scores with *g* and MI (section 1).

### 3.1 MC dropout

- **Idea.** Dropout switches off random neurons during training. Gal and Ghahramani showed that leaving it on at test time is an approximate Bayesian method. Each random mask is one weight version.
- **Analogy.** Ask a team the same question several times, each time with a few random members absent. If the answer keeps changing, the knowledge is fragile.
- **Strengths.** No extra training if the model already uses dropout. A few lines of code. It is a baseline in most Bayesian LoRA papers (Yang 2023; Wang 2024; Shi 2024; Samplawski 2026).
- **Caveats.**
  - The size of the spread is set by the dropout rate, which was chosen to fight overfitting, not to measure uncertainty.
  - In the BLoB paper, MC dropout barely improved the calibration of a fine-tuned 7B model. In the SWAG-LoRA paper's out-of-distribution test, it scored about the same as the plain model.
  - Many modern LLMs are trained without dropout, so there is nothing to switch on. Dropout can be added inside a LoRA adapter instead (BayesLoRA).
- **When and why.** 2015-2016. Dropout was already in almost every network, so this gave uncertainty "for free" from models people already had.
- **Key papers.** Gal and Ghahramani 2016; Doyle 2025 (dropout in LoRA adapters).
- **This project.** Compared. If the chosen model was trained without dropout (as Pythia was), dropout goes inside a small adapter.

### 3.2 Variational inference on full weights (Bayes by Backprop, IVON)

- **Idea.** Every weight gets a mean and a standard deviation, both learned. The loss has two parts: fit the data, and stay close to a simple starting belief (the prior). Where data is plentiful, the spread shrinks.
- **Analogy.** Instead of one number in pen per weight, write a number plus an error bar. Training shrinks the error bars that the data pins down.
- **Strengths.** Principled. It learns where to be unsure. The mean weights form a normal model for everyday use, and sampling is only needed when a score is wanted.
- **Caveats.**
  - Twice the parameters to learn (mean and spread), noisier training, and the model must be trained with this loss from the start or retrained.
  - Each weight is usually treated as independent, which ignores how weights work together.
  - The learned spread can be very narrow, so samples barely disagree.
  - Memory: at 1B parameters, a mean, a spread and optimizer state for every weight do not fit on a typical 12-24 GB consumer GPU. So it is often limited to chosen layers.
- **IVON, the modern version.** An optimizer that reads the spread from its own running curvature estimate, at close to AdamW cost. It trained GPT-2 models (125M to 773M) from scratch with slightly lower perplexity than AdamW at the mean weights (Shen 2024). Daheim 2025 uses IVON posteriors inside text generation.
- **When and why.** Graves 2011 and Blundell 2015 (Bayes by Backprop) made it trainable with ordinary backpropagation. IVON (2024) made it cheap enough for large networks.
- **Key papers.** Graves 2011; Blundell 2015; Wen 2018 (Flipout, cheaper sampling); Shen 2024 (IVON); Daheim 2025.
- **This project.** Compared, on the FFN weights. IVON is not compared.

### 3.3 Laplace approximation (classic, Laplace Redux, Laplace-LoRA)

- **Idea.** Train normally. Then measure how sharply the loss curves around the final weights. A sharp curve means the data pins the weight down, so its spread is small. A flat curve means many values are equally good, so its spread is large. Put a bell curve there.
- **Analogy.** A ball at the bottom of a valley. In a narrow valley it cannot move far. In a wide, flat valley many positions are almost as good.
- **Options.** Curvature can be stored per weight (diagonal: cheap, ignores links between weights), per layer block (KFAC, "Kronecker-factored": a compact approximation for each layer; better, more memory), or for the last layer only. Predictions come from sampling N weight versions, or from a linearized model (the network is replaced by a straight-line approximation around the trained weights), which often works better.
- **Strengths.** No retraining. Works on any trained model with one pass over some data. Laplace-LoRA (KFAC curvature on all adapter layers, linearized predictions) cut the calibration error on Winogrande-S from 31.2% to 2.1% for Llama2-7B in its best setting (7.8% in its setup that tunes on a validation set), with small overhead. Fitting only the last layer reached 22.8% (Yang 2023).
- **Caveats.**
  - The prior strength (how wide the bell starts) must be tuned, and results are sensitive to it.
  - The diagonal version gave no consistent calibration gains in Laplace-LoRA's own appendix.
  - Memory grows fast with model size: Laplace-LoRA ran out of memory at 32B on an 80 GB GPU (Samplawski 2025). The Bayesian Adaptation Gym (BAG) benchmark found Laplace erratic and memory-hungry (Samplawski 2026).
  - Easy to get the scale wrong: the curvature must be summed over the whole training set, or the noise is far too large. Our own pilot had this bug; it is fixed.
- **When and why.** MacKay 1992: networks were small enough to compute curvature. It faded when networks grew. It returned in the late 2010s with cheap curvature approximations, and Laplace Redux (2021) made it easy with a library. Laplace-LoRA (2023) applied it to LoRA, where the few trainable weights make curvature affordable.
- **Key papers.** MacKay 1992; Daxberger 2021 (Laplace Redux); Yang 2023 (Laplace-LoRA); Yang 2024 (Bayesian reward models); Miani 2024 (a low-memory curvature score).
- **This project.** Compared twice, both with diagonal curvature: on the FFN weights, and on the LoRA adapter's A matrix. The published Laplace-LoRA uses KFAC curvature over A and B with a linearized prediction; that version is not compared here.

### 3.4 Bayesian LoRA, trained (BLoB, ScalaBL, Bayesian-LoRA and relatives)

- **Idea.** Keep the pretrained model fixed. Make the small LoRA adapter random instead of fixed, and learn its spread during fine-tuning with variational inference (section 3.2).
- **Analogy.** Instead of making every brick of a building uncertain, make only the new extension uncertain.

| Variant | What is random | Extra numbers | Model sizes in the paper | Venue |
|---|---|---|---|---|
| BLoB (Wang 2024) | Matrix A, one bell curve per entry; B is trained as a single value (no spread) | One spread per entry of A | 7B | NeurIPS 2024 |
| ScalaBL (Samplawski 2025) | One random number per rank direction (r per adapted layer, r = LoRA rank). Each number scales one direction of the adapter | About 1,000 | 7B-32B | UAI 2025 |
| Bayesian-LoRA (Lin 2026) | A correlated spread over the adapter, built from a few small matrices (ideas borrowed from Gaussian processes) | About 0.42M | 7B-30B | Preprint |
| IVON-LoRA (Cong 2024) | A and B, spread from the IVON optimizer | None (optimizer state) | 7B | NeurIPS 2024 workshop |
| DALorRA (Zhang 2026) | An on/off switch for each rank direction of the adapter | Hundreds | 7B-8B | EMNLP 2026 (to appear) |

- **How BLoB stays cheap.** It uses Flipout sampling (Wen 2018) and picks a prior so that the extra loss term (the pull toward the prior) only involves matrix A. ScalaBL goes further: an SVD (singular value decomposition, a standard way to split a matrix into independent directions) of the adapter gives r fixed directions, and only r numbers per layer are random.
- **Strengths.** Affordable on large models. Papers report big calibration gains on multiple-choice questions: BLoB cut the calibration error on Winogrande-S from 29.8% to 9.4% for Llama2-7B (Wang 2024). The mean adapter can be merged into the model for normal use.
- **Caveats.**
  - The base model stays fixed, so the spread covers only what the adapter changes. Gaps in pretraining knowledge show up only indirectly.
  - Most papers test calibration on multiple-choice questions with 7B-8B models. Out-of-distribution detection and free text are rare.
  - The spread can be small: Bayesian-LoRA's metrics barely changed between 1 and 10 samples.
  - Small models can fail: in BAG, BLoB fell to chance accuracy on one task with the smallest model (0.6B).
  - Numbers do not transfer between papers. BLoB's calibration error on Winogrande-S is reported anywhere from 5.8% to 11.2% across eight papers (BLoB, TFB, ScalaBL, C-LoRA, Bayesian-LoRA, DALorRA, PoLAR-VBLL, BAG), depending on base model and setup.
  - C-LoRA (Rahmati 2025) looks similar but predicts the noise from the input. Its authors say it models aleatoric uncertainty, so it is not a weight posterior.
- **When and why.** 2023-2026. LLMs became too large for full Bayesian methods, LoRA (2021) became the default way to fine-tune, and fine-tuned LLMs are often overconfident on small datasets (as the BLoB and Laplace-LoRA abstracts say). ScalaBL appeared because earlier Bayesian LoRA methods need many extra parameters, which makes them hard to scale (its abstract says so).
- **Key papers.** Wang 2024 (BLoB); Samplawski 2025 (ScalaBL); Lin 2026 (Bayesian-LoRA); Cong 2024 and Cong 2025 (IVON-LoRA); Zhang 2026 (DALorRA); Marszałek 2025 (a tiny projected subspace).
- **This project.** BLoB is compared. ScalaBL maybe, if time allows. The others are not compared.

### 3.5 Training-free Bayesian LoRA (TFB) and fixed-noise methods

- **Idea.** Take a LoRA adapter that is already trained. The same weight change can be written with many pairs of B and A, so TFB first uses an SVD (section 3.4) to pick one standard pair. It then adds noise scaled so that the weight change gets the same size of noise in every direction the adapter can change. One number, σ_q, sets that size. TFB searches for the largest σ_q that keeps performance on a small anchor dataset within a small relative margin (the paper uses 0.3% for loss and 1% for accuracy). No gradient steps.
- **Analogy.** Turn up a "wobble" knob until the model starts to fail on examples it knows, then stop. The allowed wobble is the uncertainty.
- **Theory.** The TFB paper shows that, under mild conditions, this search is equivalent to a form of variational inference.
- **Strengths.** No training. Works on any trained LoRA adapter. The fit is a short search. A last-layer-only version is much faster at test time: 182 s versus 1,114 s for full TFB and 118 s for the plain model, at 10 samples (Shi 2024).
- **Caveats.**
  - One number controls everything. The shape of the spread is fixed, not learned.
  - Results depend on the anchor set and the margin.
  - Details matter: the SVD rotation and the relative margin. Our own pilot's first version missed both; it is fixed.
  - The TFB paper applies it to plainly trained adapters and to the mean of a BLoB-trained adapter, with calibration gains on both. This project starts TFB from a plainly trained adapter, so that its result is not mixed with BLoB's training.
- **TokUR, a close relative.** Adds fixed-size noise to the attention weights of the base model, with no fitting, and splits each generated token's uncertainty into parts. It has no out-of-distribution test, and in its own ablation, length-normalized scores lose most of the gain (Zhang 2025).
- **When and why.** Late 2024. Many LoRA checkpoints already existed, and trained Bayesian LoRA needs extra, complex training. TFB offers uncertainty with no training.
- **Key papers.** Shi 2024 (TFB); Zhang 2025 (TokUR).
- **This project.** TFB is compared. TokUR is not.

### 3.6 Ensembles (deep ensembles, LoRA ensembles)

- **Idea.** Train M models from different random starts. They end up in different good solutions. Their disagreement is the epistemic signal.
- **Analogy.** A panel of independent experts.
- **Is it Bayesian?** Not by derivation. It is usually read as a rough sample of plausible weights, and it is the standard reference that Bayesian methods must beat.
- **Strengths.** Simple, and the standard reference. In SWAG-LoRA's out-of-distribution test, the deep ensemble was best on 3 of 4 test sets (Onal 2024). Members can sit in quite different solutions, which single-bell-curve methods cannot capture.
- **Caveats.**
  - M times the training and M times the test cost. For full LLMs this is out of reach.
  - Weaker on LoRA calibration: in BAG's calibration table, a LoRA ensemble improved calibration much less than BLoB (Samplawski 2026).
  - LoRA ensembles share one frozen base, so all members share its gaps. For reward models, ensembles that differ in pretraining beat ensembles that differ only in fine-tuning (Eisenstein 2023).
  - Few members give noisy estimates, though 5 LoRA members were comparable to 20 in Balabanov 2024.
- **When and why.** 2016-2017: Bayesian networks were hard to implement and slow to train; ensembles offered a simple, parallel alternative with good uncertainty (the paper's abstract says so). 2023-2024: LoRA ensembles, because only small adapters need training.
- **Key papers.** Lakshminarayanan 2017; Malinin 2021 (ensemble uncertainty for generated sequences); Wang 2023 and Balabanov 2024 (LoRA ensembles); Eisenstein 2023; Dold 2026 (curves through LoRA solutions).
- **This project.** A LoRA ensemble of 3 adapters is compared (BLoB's paper also uses 3 LoRAs as its ensemble baseline). A full deep ensemble is not.

### 3.7 SWAG (and SWAG-LoRA)

- **Idea.** Near the end of training, record weight snapshots along the optimizer's path. Fit a bell curve to them: a mean, a per-weight spread and a few main directions. Sample from it.
- **Analogy.** Watch where a hiker wanders on the valley floor. The area covered is the set of plausible weights.
- **Strengths.** Cheap: a few extra epochs. Captures some links between weights. Combining several SWAG models (MultiSWAG) improves calibration.
- **Caveats.** Depends on the learning-rate schedule: snapshots taken while the rate decays sit too close together. In SWAG-LoRA, a single SWAG model detected out-of-distribution questions worse than the plain model.
- **When and why.** SWA (2018) showed that averaging weights along the training path helps. SWAG (2019) added a covariance to get uncertainty. SWAG-LoRA (2024) moved it to adapters.
- **Key papers.** Izmailov 2018; Maddox 2019; Talman 2023; Onal 2024.
- **This project.** Not compared.

### 3.8 Other families, in brief

| Family | One-line idea | Why it is not a main subject here |
|---|---|---|
| Bayesian last layer (VBLL, PoLAR-VBLL) | Only the output layer is uncertain, so the backbone runs once | A 50k-token output layer is hard to make Bayesian; covers one layer only |
| Epistemic neural networks (epinets) | A small extra network on top outputs the uncertainty | Needs its own training design |
| Sampling (MCMC) | Draw weights by a random walk; the reference method in theory | Rarely run on LLMs; a 2026 position paper argues its cost is now close to optimization (Sommer 2026) |
| Input-conditioned noise (C-LoRA) | Noise predicted from the input | Models aleatoric uncertainty by its authors' own framing |

---

## 4. Non-Bayesian competitors

These methods estimate LLM uncertainty without a weight posterior. They matter because a weight
posterior costs N passes. It is only worth that cost if it beats these scores, or adds to them.
This project runs four of them as baselines next to every method: perplexity, token entropy, max
probability and hidden-state distance. The others are listed for context and are not run.

| Method | What it does | Cost | Role here and notes |
|---|---|---|---|
| Probability scores: perplexity, token entropy, max probability, G-NLL | Read the model's own output probabilities. G-NLL scores the greedy answer's likelihood (Aichberger 2024) | 1 pass | **Run as baselines**, except G-NLL, which scores a generated answer. Free. They mix both kinds of uncertainty, but are hard to beat in practice |
| Semantic entropy | Sample several answers, group them by meaning with an entailment model (a model that judges whether two answers mean the same thing), take the entropy over groups (Kuhn 2023; Farquhar 2024) | 5-20 generations plus a second model | **Not run:** it scores generated answers. It measures total uncertainty over meanings. Its Nature paper says: "We do not distinguish between aleatoric and epistemic uncertainty" |
| "To Believe or Not to Believe" iterative prompting | Ask again with earlier answers in the prompt. If new answers bend toward old ones, the uncertainty is epistemic (Abbasi Yadkori 2024) | Many calls per question | **Not run:** it needs an instruction-tuned model. It treats added context as a Bayesian update, which Falck 2024 questions |
| Hidden-state distance (Mahalanobis, relative Mahalanobis) | Distance from a text's hidden state to the training data's hidden states (Lee 2018; Ren 2021) | 1 pass plus a fit | **Run as a baseline.** Very strong on domain shift: near perfect with pretrained models in Uppaal 2023. Beat MC dropout (scored by perplexity, not disagreement) on near shift in Ren 2022 |
| Probes and learned heads (P(IK), semantic entropy probes, knowable/unknowable probes) | A small classifier on hidden states predicts "the model knows this" (Kadavath 2022; Kossen 2024; Ahdritz 2024; Kapoor 2024) | 1 pass; needs labels or a teacher | **Not run:** it needs correctness labels, which this version does not score. Cheap at test time; at best as good as its labels |
| Conformal methods | Turn any score into answer sets or abstentions with a statistical guarantee, using held-out calibration data (Quach 2023; Mohri and Hashimoto 2024; Abbasi Yadkori 2024, conformal abstention) | Calibration data | **Not run:** they wrap a score; they do not estimate epistemic uncertainty by themselves |
| Verbalized confidence | Ask the model how sure it is | 1 call | **Not run:** it needs an instruction-tuned model. Known to be overconfident (Xiong 2023) |
| LM-Polygraph (library and benchmark) | Many of the methods above behind one interface, with a benchmark on QA, translation and summarization (Fadeeva 2023) | n/a | **Not used.** A widely used open-source toolkit. Its benchmark runs no weight-posterior method and calls Bayesian approaches "often difficult to implement" (Vashurin 2024) |

---

## 5. Timeline: how the field moved

The "why then" column is our reading of the history, not a claim made by the papers.

| Year | What appeared | Why then |
|---|---|---|
| 1992 | Laplace for neural networks (MacKay) | Networks were small enough to compute curvature |
| 2011-2015 | Practical variational inference (Graves), Bayes by Backprop (Blundell) | Backpropagation-friendly sampling made Bayesian training possible with standard tools |
| 2015-2016 | MC dropout (Gal and Ghahramani) | Dropout was everywhere, so uncertainty came "for free" |
| 2016-2017 | Deep ensembles (Lakshminarayanan) | Bayesian training was hard and slow; ensembles were simple, parallel and strong |
| 2018-2019 | Flipout; SWA and SWAG | Cheaper sampling; uncertainty from the training path |
| 2021 | Laplace Redux; LoRA | A library made Laplace easy; fine-tuning with small adapters |
| 2022-2023 | LLM boom; semantic entropy; LM-Polygraph; Laplace-LoRA; LoRA ensembles | Hallucination became a public problem. Output-level scores grew fast. Bayesian work moved to adapters |
| 2024 | BLoB, IVON, SWAG-LoRA, TFB; position paper on Bayesian deep learning (Papamarkou 2024) | Fine-tuned LLMs are overconfident; the position paper argues for posteriors on selected parts of a model |
| 2025 | ScalaBL (up to 32B), TokUR, C-LoRA, uncertainty-aware decoding | Bayesian LoRA scales up; token-level weight-space scores for reasoning (TokUR) |
| 2026 | Bayesian-LoRA, DALorRA, PoLAR-VBLL, the BAG benchmark; theory papers that question MI | A crowded field needs a common benchmark; the meaning of "epistemic" is debated |

**Why the 2023-2026 wave?** Three things met. (1) Full-weight Bayesian methods need twice the
parameters, huge curvature matrices or M full models, which is out of reach at billions of weights.
(2) LoRA shrank the trained part of a model to a tiny share of the weights, where Bayesian math is
affordable. (3) Fine-tuned LLMs are often overconfident, above all on small datasets, which gave the
motivation.

---

## 6. Existing comparisons and where this project fits

| Work | What it compares | Models and tasks | What it does not cover |
|---|---|---|---|
| Bayesian Adaptation Gym, BAG (Samplawski 2026) | 9 methods on LoRA adapters: BLoB, ScalaBL, TFB, Laplace-LoRA, MC dropout, ensemble, temperature scaling and others | Qwen3 0.6B-14B and vision-language models; multiple choice, visual QA, distribution shift, active learning | Full-weight posteriors; free text; out-of-distribution detection AUROC |
| The Bayesian LoRA method papers (BLoB, TFB, ScalaBL, Bayesian-LoRA, DALorRA, PoLAR-VBLL) | Each compares several LoRA methods | Mostly 7B-8B models (Llama, Qwen); commonsense multiple choice | Mostly the same gaps (Bayesian-LoRA adds token-level scores on WikiText-2); numbers differ across papers |
| SWAG-LoRA (Onal 2024) | SWAG, MultiSWAG, MC dropout, deep ensemble | Llama-2-7B; multiple choice, with OOD sets | No variational method; OOD scored by entropy only |
| Uncertainty-aware decoding (Daheim 2025) | IVON and ensemble posteriors inside decoding | Translation, summarization | No post-hoc method; full-weight and LoRA on different models |
| LM-Polygraph benchmark (Vashurin 2024); The Origins of Stochasticity (Ou 2026) | About 30 and 21 methods, none with a weight posterior | QA, translation, summarization, reasoning | No weight-space posterior |
| Uncertainty disentanglement benchmark (Mucsányi 2024) | Ensembles, MC dropout, Laplace and others | Image classifiers | Not language models |

BAG's main findings: temperature scaling (a one-number fix of the output probabilities) is competitive
with the Bayesian methods on calibration, BLoB, TFB and ScalaBL look most promising, and Laplace is
erratic and memory-hungry.

**Where this project fits.** It is a self-contained comparison, not a replacement for the work above.

- Seven methods on one pretrained backbone: MC dropout, variational FFN weights, diagonal Laplace on FFN weights, BLoB, TFB, diagonal Laplace on LoRA and a 3-adapter LoRA ensemble (ScalaBL if time allows). Full-weight and LoRA posteriors sit side by side on the same model.
- The planned models are pretrained open models with public training data, so we know which text they saw. The chosen models are Pythia-410M for all methods, then Pythia-1.4B, trained on The Pile; Pythia's exact training data and order were released.
- The main test scores human-written free text: held-out documents from the kind of text the model saw, against text from sources outside its training data.
- Cheap baselines, sanity checks and the lessons from an earlier 76M pilot are in the [metrics guide](metrics-guide.md) (sections 5 and 7).
- Cost per method is reported next to its score.
- Scope: the project measures epistemic uncertainty; what to do with a high score (warn, abstain) is the deployer's choice. Scoring the model's own generated answers, and answer correctness, come later.

---

## 7. Open problems

1. **What "epistemic" means is disputed.** MI is one definition among several that conflict (Kirchhof 2025). The split of total uncertainty into MI plus the rest behaves oddly in some cases (Wimmer 2022). Recent theory argues MI is not the "reducible by more data" kind (Young 2026), and that uncertainty about the weights can stay even when the function the network computes is fully pinned down by data (Rügamer 2026).
2. **The two kinds are hard to separate in practice.** On image classifiers, aleatoric and epistemic estimates are highly correlated (Mucsányi 2024).
3. **The signal may shrink as models grow.** Epistemic uncertainty collapses in large networks (Kirsch 2024; Fellaji 2024). Larger LLMs give lower uncertainty estimates (Ou 2026).
4. **Out-of-distribution detection is only a proxy.** It can rank epistemic methods wrongly (Paplhám 2026). A detector can also win on surface cues, which is why this project adds a character-count check.
5. **Cost.** N or M passes at test time; Laplace-LoRA out of memory at 32B (Samplawski 2025); full ensembles out of reach.
6. **Calibration is not epistemic detection.** Most Bayesian LoRA papers report calibration on multiple choice, where a simple temperature fix is competitive (BAG).
7. **Generated text is rarely scored.** Few weight-space works score the model's own outputs (Malinin 2021; Zhang 2025; Daheim 2025).
8. **Wrong model assumptions.** Ghahramani remarks, about weather models, that "sometimes your model assumptions are wrong. That is also a form of uncertainty." Our reading: a spread over weights inside a fixed architecture cannot express that the architecture itself is wrong.

---

## 8. Two talks that shaped this project

These are the speakers' views, not established results. Quotes are from the auto-generated transcripts.

**"The ChatGPT Paradox: Impressive Yet Incomplete"** (Machine Learning Street Talk,
<https://www.youtube.com/watch?v=7bmhjt1cpRs>). The guest is Thomas G. Dietterich, identified from the roles he describes.

- An LLM is "a statistical model of a knowledge base", not a knowledge base. So it answers frequent questions well and rare ones badly.
- In his view, today's LLM uncertainty scores read the output probabilities, which reflect noise (aleatoric), so nobody measures epistemic uncertainty. (Our note: LM-Polygraph's distance-based methods are a partial exception.)
- The two classic epistemic tools are hard to use on LLMs: distance to training data (almost no one releases training data) and ensembles ("we can't say train 10 of them and have them vote").
- Metrics should show the cost of abstaining: how much must be refused to reach, say, 95% correct answers.

**"The mathematics of AI uncertainty"** (Google DeepMind: The Podcast,
<https://www.youtube.com/watch?v=tBjgCj_dGZM>). The guest is Zoubin Ghahramani, a co-author of MC dropout.

- Intelligence means choosing well under uncertainty, so a system must represent its uncertainty, update it and act on it.
- LLM confidence follows how often words appear together in the data. He calls that "faking it". A confidence stated in words is also just more generated text.
- Exact Bayesian inference is too slow, but "we have decades of ideas on how to approximate these methods really efficiently", and compute now makes them worth trying.
- He does not use the word "epistemic". He contrasts a coin flip (aleatoric) with an uncertain belief, where you collect more information. His closing line: "I would rather have an AI system that knows when it doesn't know than an AI system that is arrogant and overconfident."

**What we took from them.** Use a model whose training data is known, fit distance baselines on it,
add a surface-cue check, and report cost next to every score.

---

## 9. References

Grouped by family. Every arXiv id was checked on arxiv.org. Venues are given where we could confirm them.

**Foundations, positions and data**
- MacKay 1992. A Practical Bayesian Framework for Backpropagation Networks. Neural Computation 4(3):448-472.
- Graves 2011. Practical Variational Inference for Neural Networks. NeurIPS 2011.
- Hu et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). ICLR 2022.
- Ghahramani 2015. Probabilistic machine learning and artificial intelligence. Nature 521:452-459. doi:10.1038/nature14541.
- Papamarkou et al. 2024. Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI. [arXiv:2402.00809](https://arxiv.org/abs/2402.00809). ICML 2024.
- Kirchhof et al. 2025. Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents. [arXiv:2505.22655](https://arxiv.org/abs/2505.22655). ICML 2025.
- Gao et al. 2020. The Pile: An 800GB Dataset of Diverse Text for Language Modeling. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
- Biderman et al. 2023. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).

**MC dropout**
- Gal and Ghahramani 2016. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142). ICML 2016.
- Doyle 2025. BayesLoRA: Task-Specific Uncertainty in Low-Rank Adapters. [arXiv:2506.22809v1](https://arxiv.org/abs/2506.22809v1). (Later versions, with more authors and a different method, are titled "Learning Adapter Rank via Symmetry Breaking".)

**Variational inference on full weights**
- Blundell et al. 2015. Weight Uncertainty in Neural Networks. [arXiv:1505.05424](https://arxiv.org/abs/1505.05424). ICML 2015.
- Wen et al. 2018. Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. [arXiv:1803.04386](https://arxiv.org/abs/1803.04386). ICLR 2018.
- Shen et al. 2024. Variational Learning is Effective for Large Deep Networks (IVON). [arXiv:2402.17641](https://arxiv.org/abs/2402.17641). ICML 2024.
- Daheim et al. 2025. Uncertainty-Aware Decoding with Minimum Bayes Risk. [arXiv:2503.05318](https://arxiv.org/abs/2503.05318). ICLR 2025.

**Laplace approximation**
- Daxberger et al. 2021. Laplace Redux: Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806). NeurIPS 2021.
- Yang et al. 2023. Bayesian Low-rank Adaptation for Large Language Models (Laplace-LoRA). [arXiv:2308.13111](https://arxiv.org/abs/2308.13111). ICLR 2024.
- Yang et al. 2024. Bayesian Reward Models for LLM Alignment. [arXiv:2402.13210](https://arxiv.org/abs/2402.13210).
- Miani et al. 2024. Sketched Lanczos uncertainty score: a low-memory summary of the Fisher information. [arXiv:2409.15008](https://arxiv.org/abs/2409.15008).

**Bayesian LoRA, trained**
- Wang et al. 2024. BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675). NeurIPS 2024.
- Samplawski et al. 2025. Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference (ScalaBL). [arXiv:2506.21408](https://arxiv.org/abs/2506.21408). UAI 2025.
- Lin et al. 2026. Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models. [arXiv:2601.21003](https://arxiv.org/abs/2601.21003).
- Cong et al. 2024. Variational Low-Rank Adaptation Using IVON. [arXiv:2411.04421](https://arxiv.org/abs/2411.04421). NeurIPS 2024 workshop.
- Cong et al. 2025. Improving LoRA with Variational Learning. [arXiv:2506.14280](https://arxiv.org/abs/2506.14280).
- Zhang et al. 2026. Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty Estimation (DALorRA). [arXiv:2607.02182](https://arxiv.org/abs/2607.02182). EMNLP 2026 (to appear).
- Marszałek et al. 2025. Minimal Ranks, Maximum Confidence: Parameter-efficient Uncertainty Quantification for LoRA. [arXiv:2502.12122](https://arxiv.org/abs/2502.12122). Findings of EMNLP 2025.
- Rahmati et al. 2025. C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models. [arXiv:2505.17773](https://arxiv.org/abs/2505.17773). NeurIPS 2025.

**Training-free**
- Shi et al. 2024. Training-Free Bayesianization for Low-Rank Adapters of Large Language Models (TFB). [arXiv:2412.05723](https://arxiv.org/abs/2412.05723). NeurIPS 2025.
- Zhang et al. 2025. TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning. [arXiv:2505.11737](https://arxiv.org/abs/2505.11737). ICLR 2026.

**Ensembles**
- Lakshminarayanan et al. 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474). NeurIPS 2017.
- Malinin and Gales 2021. Uncertainty Estimation in Autoregressive Structured Prediction. [arXiv:2002.07650](https://arxiv.org/abs/2002.07650). ICLR 2021.
- Wang et al. 2023. LoRA ensembles for large language model fine-tuning. [arXiv:2310.00035](https://arxiv.org/abs/2310.00035).
- Balabanov and Linander 2024. Uncertainty quantification in fine-tuned LLMs using LoRA ensembles. [arXiv:2402.12264](https://arxiv.org/abs/2402.12264). ICLR 2025 workshop.
- Eisenstein et al. 2023. Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking. [arXiv:2312.09244](https://arxiv.org/abs/2312.09244). CoLM 2024.
- Dold et al. 2026. On the Construction and Implications of Low-Loss Valleys in LoRA-based Bayesian Inference. [arXiv:2605.29580](https://arxiv.org/abs/2605.29580).

**SWAG**
- Izmailov et al. 2018. Averaging Weights Leads to Wider Optima and Better Generalization (SWA). [arXiv:1803.05407](https://arxiv.org/abs/1803.05407).
- Maddox et al. 2019. A Simple Baseline for Bayesian Uncertainty in Deep Learning (SWAG). [arXiv:1902.02476](https://arxiv.org/abs/1902.02476). NeurIPS 2019.
- Talman et al. 2023. Uncertainty-Aware Natural Language Inference with Stochastic Weight Averaging. [arXiv:2304.04726](https://arxiv.org/abs/2304.04726). NoDaLiDa 2023.
- Onal et al. 2024. Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of Large Language Models (SWAG-LoRA). [arXiv:2405.03425](https://arxiv.org/abs/2405.03425).

**Other weight-space families**
- Harrison et al. 2024. Variational Bayesian Last Layers (VBLL). [arXiv:2404.11599](https://arxiv.org/abs/2404.11599). ICLR 2024.
- Xiang et al. 2026. Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters (PoLAR-VBLL). [arXiv:2604.03388](https://arxiv.org/abs/2604.03388).
- Osband et al. 2022. Fine-Tuning Language Models via Epistemic Neural Networks. [arXiv:2211.01568](https://arxiv.org/abs/2211.01568).
- Sommer and Rügamer 2026. Position: The Time for Sampling Is Now! Charting a New Course for Bayesian Deep Learning. [arXiv:2605.21765](https://arxiv.org/abs/2605.21765). ICML 2026.

**Non-Bayesian competitors**
- Kuhn et al. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664). ICLR 2023.
- Farquhar et al. 2024. Detecting hallucinations in large language models using semantic entropy. Nature 630:625-630. doi:10.1038/s41586-024-07421-0.
- Abbasi Yadkori et al. 2024. To Believe or Not to Believe Your LLM. [arXiv:2406.02543](https://arxiv.org/abs/2406.02543).
- Falck et al. 2024. Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective. [arXiv:2406.00793](https://arxiv.org/abs/2406.00793). ICML 2024.
- Aichberger et al. 2024. Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure (G-NLL). [arXiv:2412.15176](https://arxiv.org/abs/2412.15176). ICLR 2026.
- Kadavath et al. 2022. Language Models (Mostly) Know What They Know (P(IK)). [arXiv:2207.05221](https://arxiv.org/abs/2207.05221).
- Kossen et al. 2024. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. [arXiv:2406.15927](https://arxiv.org/abs/2406.15927).
- Ahdritz et al. 2024. Distinguishing the Knowable from the Unknowable with Language Models. [arXiv:2402.03563](https://arxiv.org/abs/2402.03563). ICML 2024.
- Kapoor et al. 2024. Large Language Models Must Be Taught to Know What They Don't Know. [arXiv:2406.08391](https://arxiv.org/abs/2406.08391). NeurIPS 2024.
- Lee et al. 2018. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks (Mahalanobis). [arXiv:1807.03888](https://arxiv.org/abs/1807.03888). NeurIPS 2018.
- Ren et al. 2021. A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. [arXiv:2106.09022](https://arxiv.org/abs/2106.09022).
- Ren et al. 2022. Out-of-Distribution Detection and Selective Generation for Conditional Language Models. [arXiv:2209.15558](https://arxiv.org/abs/2209.15558). ICLR 2023.
- Uppaal et al. 2023. Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection. [arXiv:2305.13282](https://arxiv.org/abs/2305.13282). ACL 2023.
- Xiong et al. 2023. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. [arXiv:2306.13063](https://arxiv.org/abs/2306.13063). ICLR 2024.
- Quach et al. 2023. Conformal Language Modeling. [arXiv:2306.10193](https://arxiv.org/abs/2306.10193). ICLR 2024.
- Mohri and Hashimoto 2024. Language Models with Conformal Factuality Guarantees. [arXiv:2402.10978](https://arxiv.org/abs/2402.10978).
- Abbasi Yadkori et al. 2024. Mitigating LLM Hallucinations via Conformal Abstention. [arXiv:2405.01563](https://arxiv.org/abs/2405.01563).
- Fadeeva et al. 2023. LM-Polygraph: Uncertainty Estimation for Language Models. [arXiv:2311.07383](https://arxiv.org/abs/2311.07383). EMNLP 2023.
- Vashurin et al. 2024. Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph. [arXiv:2406.15627](https://arxiv.org/abs/2406.15627). TACL 2025.

**Benchmarks**
- Samplawski et al. 2026. Bayesian Adaptation Gym: A Benchmark for the Bayesian Low-Rank Adaptation of Multi-Modal Language Models (BAG). [arXiv:2606.22188](https://arxiv.org/abs/2606.22188). UAI 2026.
- Ou et al. 2026. The Origins of Stochasticity: Comprehensive Investigations on Uncertainty Quantification for Large Language Models. [arXiv:2606.22792](https://arxiv.org/abs/2606.22792).
- Mucsányi et al. 2024. Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks. [arXiv:2402.19460](https://arxiv.org/abs/2402.19460).

**Open problems**
- Wimmer et al. 2022. Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures? [arXiv:2209.03302](https://arxiv.org/abs/2209.03302). UAI 2023.
- Kirsch 2024. (Implicit) Ensembles of Ensembles: Epistemic Uncertainty Collapse in Large Models. [arXiv:2409.02628](https://arxiv.org/abs/2409.02628). TMLR.
- Fellaji and Pennerath 2024. The Epistemic Uncertainty Hole: an issue of Bayesian Neural Networks. [arXiv:2407.01985](https://arxiv.org/abs/2407.01985).
- Young 2026. Epistemic Uncertainty Is Not the Reducible Kind. [arXiv:2606.12646](https://arxiv.org/abs/2606.12646).
- Rügamer 2026. On the Epistemic Uncertainty of Overparametrized Neural Networks. [arXiv:2605.25234](https://arxiv.org/abs/2605.25234). ICML 2026.
- Paplhám et al. 2026. Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning. [arXiv:2607.14817](https://arxiv.org/abs/2607.14817).
