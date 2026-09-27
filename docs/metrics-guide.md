# Metrics Guide

How this project measures epistemic uncertainty in language models, and how it judges whether a measure works.
Written for students and engineers. Every metric is defined the first time it appears. This guide is the plan for
the next version of the project; its results are not in yet. The earlier version was a pilot: a small model (76M
parameters) trained from scratch on part of The Pile. The results in the README and the paper come from that pilot
and use some metrics dropped here. Section 7 lists what changed and why.

## At a glance

- **Main score (Section 2):** the realized-token disagreement, called *g*. For each token of a text, the weight
  samples say how likely the actual next token is, and *g* measures how much they disagree. A text's score is the mean.
- **Companion score (Section 2):** mutual information (MI), the classic disagreement measure over the whole vocabulary.
- **Main test (Section 3):** AUROC for telling familiar from unfamiliar text, reported separately per unfamiliar source.
- **Honest error bars (Section 4):** bootstrap by document, paired comparisons, Holm correction, rules fixed in advance.
- **Bars to beat (Section 5):** the plain model's perplexity, a hidden-state distance, a character count, matched noise.
- **Price tags (Section 6):** model quality and calibration, and cost (time, memory, number of samples).
- **Later (Section 8):** scoring the model's own answers, and a test with facts seen a known number of times.

## 1. Two kinds of uncertainty

A model can be unsure for two different reasons.

- **Aleatoric uncertainty** comes from the data itself. The text is ambiguous, and more training data would not
  help. Example: "My favourite colour is ___". Many answers are fine, so even a perfect model spreads its bets.
- **Epistemic uncertainty** comes from the model. It has not seen enough data like this, and more such data would
  help. Example: a model trained only on news articles is asked to read computer code.

An everyday picture: a coin flip is aleatoric; even a perfect physicist says 50/50. A weather forecaster who moved
to a new city last week has epistemic uncertainty; after a year of local data, the forecasts get better.

The project asks: when a language model reads or writes text, how much of its uncertainty is epistemic? It compares
seven methods that keep many plausible versions of the model's weights instead of one. Six are **Bayesian**: MC
dropout, variational inference on the feed-forward (FFN) weights, Laplace on the full weights, and three on LoRA (a
small adapter trained on top of frozen weights): BLoB (trained), TFB (noise added to a trained LoRA, no extra
training) and Laplace-LoRA. The seventh, an ensemble of 3 LoRAs, is not Bayesian by derivation but is a common
reference in Bayesian-LoRA papers [Wang 2024]. The [methods overview](methods-overview.md) explains them all. The
planned models are pretrained open models whose training data is public, so we know what text they saw. The
chosen models are Pythia-410M and Pythia-1.4B [Biderman 2023], trained on The Pile. Every method is scored on the same texts.

One principle shapes the metrics: **we measure; the people who deploy the model decide.** What to do with a high
score (warn, abstain, ask a human) is their choice. So the first test of a measure is whether it tracks what the
model has not seen. Whether it predicts wrong answers is a second, later check (Section 8).

## 2. The scores: what each method outputs

### Weight samples

A normal model has one set of weights. A Bayesian method gives a spread of plausible weight sets (a **posterior**).
We draw N of them (for example N = 20), called **weight samples**, and feed the same text through each one. At every
position, each sample gives a probability for every possible next token. If the samples agree, the model's knowledge
is solid there. If they disagree, the training data did not pin the weights down. That disagreement is the epistemic
signal.

- **Entropy** measures how spread out a probability distribution is. It is 0 when one token gets probability 1, and
  largest when all tokens are equally likely. With natural logs the unit is the **nat** (1 nat = 1.44 bits).
- The **averaged prediction** $\bar p$ is the mean of the N samples' distributions.

### The four per-token scores

| Score | In words | What it captures |
|---|---|---|
| Predictive entropy (total uncertainty, TU) | Entropy of the averaged prediction | Everything: epistemic plus aleatoric |
| Expected entropy (aleatoric part, AU) | Average of each sample's own entropy | Uncertainty left even if we knew the right weights |
| Mutual information (MI) | TU minus AU | How much the samples disagree about the next token |
| Realized-token disagreement (*g*) | Disagreement about the token that actually comes next | The same idea, focused on one token |

**Mutual information.**

$$\mathrm{MI}_t = H[\bar p_t] - \frac{1}{N}\sum_{s=1}^{N} H[p_{s,t}]$$

In words: at position *t*, the entropy of the average minus the average of the entropies. It is never negative, and
it is 0 when all samples give the same distribution. This is the classic BALD score [Houlsby 2011]. Malinin and
Gales use it as "knowledge uncertainty" for sequence models [Malinin 2021].

Worked example with two samples and only two possible tokens (values in nats):

| Case | Sample A | Sample B | TU | AU | MI |
|---|---|---|---|---|---|
| Confident, agree | [0.9, 0.1] | [0.9, 0.1] | 0.33 | 0.33 | 0 |
| Unsure, agree | [0.5, 0.5] | [0.5, 0.5] | 0.69 | 0.69 | 0 |
| Confident, disagree | [0.9, 0.1] | [0.1, 0.9] | 0.69 | 0.33 | 0.37 |

Row 2 is pure aleatoric uncertainty: both samples say "it's a coin flip". Row 3 is epistemic: each sample is sure,
but they are sure of different things. TU is 0.69 in both rows. Only the split into AU and MI tells them apart.

**Realized-token disagreement (the main score).** When we score existing text, we know which token actually comes
next. Call it $y_t$. Each sample *s* gives it a log-probability $\ell_{s,t} = \log p_s(y_t)$.

$$g_t = \log\Big(\frac{1}{N}\sum_{s=1}^{N} p_s(y_t)\Big) - \frac{1}{N}\sum_{s=1}^{N}\ell_{s,t}$$

In words: the log of the average probability, minus the average of the log-probabilities. This gap is never
negative (a math fact called Jensen's inequality), so it is also called the Jensen gap. It is 0 when all samples give
the actual token the same probability. For a small spread, *g* is about half the variance of the log-probabilities.

Worked example: sample A gives the actual token 0.9, sample B gives it 0.1. The log of the average is
log 0.5 = -0.69. The average of the logs is (-0.11 - 2.30) / 2 = -1.20. So *g* = -0.69 + 1.20 = 0.51 nats.
If both samples gave 0.1, *g* would be 0: the model is surprised, but its versions agree. Surprise is not disagreement.

Why *g* is the main score:
- It needs only N numbers per token (the probability of the actual token), not N full distributions over a
  vocabulary of about 50,000 tokens. It is cheap to store and simple to compute once the N forward passes are done
  (the passes themselves cost the same as for MI).
- Like MI, it works for every method (variational, Laplace, dropout, ensembles), and it applies unchanged when,
  later, we score the model's own generated answers.
- It has a known relative: on tokens sampled from the averaged model, its average equals the "reverse mutual
  information" of [Malinin 2021]. On human-written text, and on greedy answers later, the tokens are not sampled
  that way, so here *g* is its own score, not an estimate of reverse MI.

MI is reported next to *g* for every method, because MI is the standard score in the literature.

### From tokens to one score per text

Up to a few fixed-length windows (for example 256 tokens) are taken from each document. A window's score is the
plain average of its per-token scores. A document counts once in total: if it gives 3 windows, each gets weight 1/3.

- The averaging rule is fixed in advance. A plain mean can dilute a few highly uncertain tokens, and some tokens
  matter more than others [Duan 2023]. Other rules (maximum, mean of the top 10%) may appear as labelled extras.
- The average of token MI is not the MI of the whole sequence [Malinin 2021]. We call it an average per-token score.
- **Raw values do not compare across methods.** The size of MI or *g* depends on each method's free settings (noise
  size, dropout rate, prior strength). One method can give ten times larger values than another and still rank texts
  worse. So only how well each score ranks texts is compared (Section 3).

## 3. How we judge a score

### The test

We build two piles of human-written text:

- **Familiar text (in-distribution, ID):** documents from the sources the model trained on, but from a held-out
  part, so the model should not have seen them. Web text repeats, so they are checked for near-copies of training text.
- **Unfamiliar text (out-of-distribution, OOD):** documents from sources the model never trained on.

A good epistemic score is higher on unfamiliar text. This is the standard first test for an epistemic measure
[Malinin 2021; Ielanskyi 2025]. A score that cannot flag unseen sources is unlikely to track missing knowledge.

It is not the whole story. The label is coarse: an unfamiliar document still holds many common words the model
knows well. Unfamiliar text can also simply *look* different (Section 5). One recent paper argues that this kind of
test can even rank epistemic methods in the wrong order [Paplhám 2026]. Hence the baselines, and more tests later.

### The test metrics

**AUROC (main).** Pick one random familiar text and one random unfamiliar text. AUROC is the chance that the
unfamiliar one gets the higher score (ties count half). 1.0 is perfect, 0.5 is a coin flip, and below 0.5 the score
points the wrong way. Worked example: familiar scores {1, 2, 3}, unfamiliar scores {2.5, 4, 5}. Of the 9 pairs, the
unfamiliar text wins 8 (only 2.5 vs 3 loses). AUROC = 8/9 = 0.89.

Why it is the main metric: it needs no threshold, only the ranking matters (not the scale of the score), and it does
not depend on the ratio of familiar to unfamiliar texts (its error bar does depend on how many documents there are).
It is the standard metric in out-of-distribution detection papers [Hendrycks 2017; Ren 2022].

**FPR at 95% TPR (FPR95).** Set the threshold so that 95% of unfamiliar texts are flagged (true positive rate,
TPR = 95%). FPR95 is the share of familiar texts flagged by mistake (false positive rate). Lower is better. Example:
with 100 texts of each kind, the threshold that catches 95 unfamiliar texts also flags 30 familiar ones: FPR95 = 30%.
Caveat: it depends on the few hardest unfamiliar texts, so it is noisy and always gets an error bar.

**TPR at a low false-alarm rate (for example, TPR at 5% FPR).** The reverse question: if only 1 in 20 familiar texts
may be flagged by mistake, what share of unfamiliar texts do we catch? This is the view of a deployer who cannot
annoy users with false alarms. Example: with 100 familiar texts, the threshold is set so only 5 are flagged. If it
catches 60 of 100 unfamiliar texts, TPR at 5% FPR = 60%.

**AUPRC (area under the precision-recall curve).** Precision is the share of flagged texts that are really
unfamiliar. When unfamiliar text is rare, a score can have a high AUROC and still low precision. Example: 5 of 100
texts are unfamiliar. A detector flags 10 texts and catches all 5, but its precision is only 50%. With equal-size
piles, AUPRC mostly repeats AUROC. So it is reported only at a realistic mix (for example 5% unfamiliar), if at all.

### One result per source, never pooled

Each unfamiliar source is reported against the familiar set on its own. Sources differ in difficulty. Pooling them
lets an easy source (for example text full of math symbols) hide a failure on a hard one (plain prose close to the
training text). Which detector wins depends on the type of shift [Arora 2021].

## 4. Honest error bars

### Bootstrap by document

A number from one test set could be luck. The **bootstrap** estimates how much it could move: draw a new test set
from the old one, with replacement (familiar and unfamiliar documents are drawn separately, so each pile keeps its
size), recompute the AUROC, repeat many times (for example 10,000), and take the middle 95% of the results. That
range is the 95% **confidence interval (CI)**.

The key detail: whole **documents** are resampled, not windows. Windows from one document are alike (same topic,
author and format), so 50 windows from one document are closer to one piece of evidence than to 50. A window-level
bootstrap gives error bars far too narrow. Lesson from the earlier version: the unfamiliar set came from only a
handful of documents, and the bootstrap treated every window as independent, so the error bars looked tight but were
not honest. In this version each source will have many documents (the target is about a thousand).

### Paired comparisons

To claim "method A beats method B", checking whether their two intervals overlap is the wrong test: they can
overlap even when one method is clearly better. Instead, every bootstrap round computes the difference A minus B on
the same resampled documents. The CI of that difference is the test. Both methods see the same easy and hard
documents, so that shared noise cancels out.

A **minimum meaningful difference** is set in advance (for example 0.02 AUROC). A smaller gap can come from random
weight sampling or tiny run-to-run differences on the GPU, so it counts as a tie. Where training is cheap, it is
repeated with a few seeds (for example 3) and the spread is reported: one run says nothing about training luck.

### Many comparisons: the Holm correction

Seven methods give 21 pairs, times several sources. At the usual 5% level, about 1 in 20 tests of a true tie looks
"significant" by chance. (A p-value is the chance of a result at least this large if there were no real difference.)
The **Holm correction** guards against this, within each family of related tests. Sort the m p-values from smallest
to largest. Multiply the smallest by m, the next by m - 1, and so on. Stop at the first result above 0.05: it and all
later ones fail. Example with m = 3 and p-values 0.01, 0.03, 0.04: 0.01 × 3 = 0.03, a pass. 0.03 × 2 = 0.06, a
fail, so we stop: the second and third results both fail. Without the correction, all three would pass.

### Fix the rules before looking

Before any test text is scored, we write down the main score, the main metric, the comparisons, the margin, the
correction, the decision rules and the wording of a negative result. We store a fingerprint (hash) of these settings,
and the analysis code will refuse to run if they change after that. This is **pre-registration**. Why: if you choose
the score, the threshold or the averaging rule after seeing test results, you can find a "win" by chance. Fitting
and tuning (noise size, prior strength, dropout rate) use separate held-out texts, never the test texts. The one
exception is the combined-score test (Section 5), which cross-validates on test documents.

## 5. Baselines and sanity checks

A high AUROC does not prove that a score measures what the model has not seen. The unfamiliar text may just look
different. So every method's disagreement score is compared with cheaper scores that use no weight samples at all.

| Baseline | What it is | Cost | Why it is here |
|---|---|---|---|
| Perplexity (NLL, negative log-likelihood) | How surprised the plain model is: the average of -log p(actual token). Perplexity = exp(NLL). | 1 pass | The main bar to beat. A model is often simply more surprised by unfamiliar text. |
| Token entropy | Average entropy of the plain model's next-token distribution | 1 pass | Total uncertainty, with no split |
| Max-probability | 1 minus the probability of the top token, averaged | 1 pass | The classic out-of-distribution baseline [Hendrycks 2017] |
| Hidden-state distance | How far the model's internal vectors (hidden states) for this text are from those of training text. It uses the Mahalanobis distance, which scales each direction by how much training texts vary along it. The "relative" version subtracts the same distance measured against a broad background set of texts. | 1 pass, plus a one-off fit | From [Lee 2018; Ren 2021]. Far beat perplexity and MC dropout on a summarization shift [Ren 2022]; near perfect on many domain-shift pairs with pretrained models [Uppaal 2023] |
| Character count | Simple character shares, such as backslashes, dollar signs, digits or punctuation. No model at all. | Free | Tests whether the task is just "the text looks different" |
| Random weight noise of matched size | Random noise from a bell curve (Gaussian) on the same weights, sized so it raises NLL on held-out familiar text by about as much (within a fixed tolerance) as the Bayesian method does | N passes | Tests whether the *shape* of the Bayesian uncertainty matters, or any wobble would do |

Lesson from the earlier version: a simple character count separated familiar from unfamiliar text about as well as
the Bayesian methods did. The benchmark was partly detecting markup, not missing knowledge. So surface baselines are
required in this version.

### The key question: does the Bayesian score add anything?

Beating each baseline one by one is not enough. The real question is: **does the Bayesian score add information
beyond what the plain model already tells us?** The **combined-score test** answers it:

1. Fit a logistic regression (a simple linear classifier) that predicts "unfamiliar" from all one-pass features
   (perplexity, entropy, max-probability, distances, character counts). Measure its AUROC.
2. Fit it again with the Bayesian score added. Measure the AUROC again.
3. The gain is the difference. The model is fitted on some documents and scored on others (cross-validation by
   document), and the gain gets a paired bootstrap CI.

| Verdict (fixed in advance) | Rule |
|---|---|
| Adds | The gain is at least the margin (for example 0.02) and significant after Holm |
| No useful gain | Even the top of the gain's CI is below the margin |
| Ceiling | The one-pass features alone already score above 1 minus the margin (0.98), so nothing can add the margin |
| Inconclusive | Anything else |

Caveat: this test uses the labels of test documents. It measures information, not a detector anyone could deploy.

### Matched texts and entropy checks

- **Surface-matched bins.** Group texts into bins by the character-count score (or by perplexity), and compute
  AUROC inside each bin. A score that still separates texts that look alike is not just a surface detector.
- **Markup removed.** If an unfamiliar source has obvious markup (for example LaTeX), a copy without it is scored too.
- **TU, AU, MI and *g* side by side.** If the epistemic part separates the piles and total uncertainty does not,
  that is evidence for an epistemic signal. Malinin and Gales saw this on translation: with German input to an
  English-to-German model, total entropy gave AUROC 0.408 (worse than a coin flip) and MI 0.761 [Malinin 2021].
- **Entropy bins and correlation.** The split of TU into AU and MI has known weaknesses [Wimmer 2022], and on image
  models the aleatoric and epistemic estimates are highly correlated [Mucsányi 2024]. So each method also reports
  AUROC within bins of total entropy, and the rank correlation between MI and expected entropy (do two scores put
  texts in the same order? 1 means the same order, 0 means unrelated).

## 6. Side metrics: the price of going Bayesian

These do not measure epistemic uncertainty. They show what a method costs.

### Model quality and calibration

| Metric | Plain meaning | Tiny example | Good value |
|---|---|---|---|
| NLL and perplexity | Average -log p(actual token) under the averaged prediction. Perplexity = exp(NLL): "how many tokens the model is choosing between". | Perplexity 14 means as unsure as picking uniformly among 14 tokens | Lower; close to the plain model |
| ECE (expected calibration error) | Group tokens by the model's confidence in its top guess. In each group, compare the confidence with how often the top guess is right. Average the gaps, weighting each group by how many tokens it holds [Guo 2017]. | Tokens with confidence near 0.8 are right 70% of the time: a 0.1 gap in that group | Lower; 0 is perfect |
| Brier score | Squared error between the predicted probabilities and the truth (1 for the actual token, 0 for the rest) | Probabilities [0.7, 0.2, 0.1], actual = first: 0.3² + 0.2² + 0.1² = 0.14 | Lower; 0 is perfect |

Why these matter:
- **Sanity gate.** If the weight samples make familiar-text NLL worse than the plain model by more than a limit fixed
  in advance, the method is broken, and its "uncertainty" is just damage. Each method must pass this check before its
  other metrics are read. Lesson from the earlier version: implementation bugs in two after-training methods made them
  wreck the model. The bugs are fixed in the code, and this gate catches that kind of failure early.
- **Confound check.** A worse model can show more disagreement for reasons that have nothing to do with missing
  data. Perplexity is reported next to every AUROC.
- **Comparability.** Bayesian-LoRA papers report NLL and ECE [Yang 2023; Wang 2024; Shi 2024], so readers can place
  our numbers. Calibration is computed on familiar and unfamiliar text, with document-bootstrap CIs.

### Cost

| Metric | What it tells a practitioner |
|---|---|
| Time per text | Milliseconds to score one window with N samples, same GPU and batch size; also as a ratio to one plain pass |
| Peak GPU memory | Whether the method fits on your card |
| Training or fitting time | One-off cost: one adapter training run for dropout (if the model was trained without dropout, as Pythia was, dropout goes inside a small LoRA adapter); a short search or fit for after-training methods; full training runs for variational methods and ensembles |
| Extra stored parameters | Disk and memory for the Bayesian part |
| Number of weight samples N | How AUROC grows with N, so you can pick the cheapest N that works |

For the N curve we report the share of the above-chance AUROC that is kept, $(\mathrm{AUROC}_N - 0.5) / (\mathrm{AUROC}_{N_{\max}} - 0.5)$.
In words: how much of the useful signal is left with fewer samples. A plain ratio of AUROCs would count the free 0.5
that any score gets. With N = 1, MI and *g* are exactly 0, so their AUROC is 0.5.

## 7. What changed from the earlier version

| Earlier version | This version | Why |
|---|---|---|
| MI ratio: mean MI on unfamiliar text divided by mean MI on familiar text | Dropped | Not a standard metric: we found no paper that uses it. It compares averages and ignores the spread. It changes with each method's free settings (noise size, dropout rate) in ways that do not reflect ranking quality; AUROC depends only on the ranking. A ratio near 1 cannot tell a broken model (huge MI everywhere) from one whose samples barely differ (tiny MI everywhere). A ratio above 1 can hide a large overlap between the two piles. AUROC replaces it. |
| Flip rate: share of samples whose top token differs from the most common one | Dropped | It looks only at the top token and ignores probabilities. On flat predictions the top token flips after tiny changes, so it mixes in aleatoric uncertainty. Not standard. |
| AURC over all tokens pooled: error rate against the share of tokens kept, from next-token correctness | Dropped for now | The plain model has no disagreement score (MI is always 0), so its AURC was just its error rate and ranked nothing. Next-token "correctness" on human text is also a weak label: often many next words are fine. Selective metrics return at the answer level (Section 8). |
| AUPRC with equal-size piles | Realistic mix, or dropped | With equal piles it repeats AUROC |
| Bootstrap over windows as if they were independent | Bootstrap over documents | Windows from one document are not independent |
| Overlapping intervals read as "no difference" | Paired differences with Holm correction | Overlap is not a valid test |
| One unfamiliar source, from a handful of documents | Several sources, many documents each, reported separately | Tiny samples and pooling both mislead |
| Max-probability and entropy as the only baselines | Full baseline set, character counts, matched noise, combined-score test | A character count matched the methods |
| Some settings checked on the texts later used for results | Rules fixed in advance; fits on separate held-out texts; fresh test documents | Choosing after looking inflates results |
| "N samples keep X% of the signal" as a raw AUROC ratio | Share of the above-chance AUROC | The raw ratio counts the 0.5 that comes free |
| Calibration on familiar text only, without error bars | Familiar and unfamiliar text, with CIs | Differences between methods were small; without CIs they cannot be read |
| No check that a posterior is healthy | Sanity gate on familiar-text NLL | Catches broken weight sampling before any result is read |

## 8. Later, not in this version

This version scores human-written text. The research question also covers text the model generates.

**Scoring the model's own answers.** Generate an answer once with the average weights. Rescore its tokens with N weight
samples in one batched pass, and compute *g* or MI per token. The formulas stay the same; only the source of the
tokens and the ground truth change. Answer-level metrics:

- **Correctness AUROC:** does the score rank wrong answers above right ones? The standard test for generated
  answers [Kuhn 2023; Vashurin 2024].
- **PRR (prediction rejection ratio):** reject the most uncertain answers and see how much average quality improves.
  0 means no better than rejecting at random; 1 means as good as an oracle that knows which answers are wrong
  [Malinin 2021; Fadeeva 2023].
- **Accuracy at a fixed coverage** (for example, on the 80% of answers the model is most sure of), and answer ECE.

Known caveats: right/wrong labels made by word overlap with a reference answer, or by one LLM judge (another language
model that grades the answer), are noisy. The choice of a single judge adds AUROC noise (standard deviation up to
0.04), as large as typical gaps between methods [Ielanskyi 2025]. Answer length can drive both the score and the label
[Santilli 2025]. Tasks with exact-match answers avoid most of this.

**The known-exposure test (the strongest ground truth).** Teach the model made-up facts, each seen 0, 1, 10 or 100
times in training, then ask about them. A good epistemic score is high for facts seen 0 times and falls as exposure
grows. An aleatoric control is an attribute redrawn at random at every mention, so no amount of data helps. There
the epistemic score should stay low, even though total uncertainty is high. Metrics: the rank correlation between
score and exposure count, and the AUROC of unseen facts against the control. Related work studies fact learning with
synthetic biographies [Allen-Zhu 2023], and finds that answer accuracy tracks how many pretraining documents mention
a fact [Kandpal 2022]. A known risk: for a never-seen name, all weight samples may agree on the same "average guess".
The test would reveal that, which is a finding in itself.

## 9. Known limits of these measures

A weight posterior cannot flag errors that come from the model's design itself. Epistemic signal can shrink as
models grow [Kirsch 2024], so two model sizes are tested. Researchers also disagree on what "epistemic" means
[Kirchhof 2025]; this project uses a practical definition: uncertainty that should be higher on text the model has not
seen. More in the [methods overview](methods-overview.md#7-open-problems).

## 10. Summary

| Metric | Question it answers | Good value | Role |
|---|---|---|---|
| *g* (realized-token disagreement), per text | Do the weight samples disagree about the actual text? | Higher on unfamiliar text | Main score |
| MI | Do the weight samples disagree about the next token, over the whole vocabulary? | Higher on unfamiliar text | Companion score |
| AUROC, per source | How well does the score rank unfamiliar text above familiar text? | 1.0 (0.5 is chance) | Main test |
| FPR95; TPR at 5% FPR | How many familiar texts are wrongly flagged; how many unfamiliar texts are caught? | 0%; 100% | Secondary |
| AUPRC at a realistic mix | Are flagged texts really unfamiliar when unfamiliar text is rare? | 1.0 | Optional |
| Document-bootstrap CI | How much could the number move with other documents? | Narrow and honest | Every number |
| Paired difference with Holm | Is method A really better than method B? | Difference at least the margin (0.02), significant after Holm | Every claim |
| Combined-score gain | Does the Bayesian score add beyond one-pass scores? | At least the margin, significant | Key question |
| Baselines: perplexity, entropy, max-probability, distance, character count, matched noise | Could something cheaper do the same? | The Bayesian score beats them | Required |
| NLL, perplexity, ECE, Brier; time, memory, training time, N | What does going Bayesian cost in quality, calibration and compute? | Close to the plain model; low | Side |
| Correctness AUROC, PRR; rank correlation with exposure count | Does the score flag wrong answers? Does it fall as a fact is seen more often? | High; strongly negative | Later |

## References

- [Allen-Zhu 2023] Allen-Zhu and Li. Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. [arXiv:2309.14316](https://arxiv.org/abs/2309.14316)
- [Arora 2021] Arora et al. Types of Out-of-Distribution Texts and How to Detect Them. [arXiv:2109.06827](https://arxiv.org/abs/2109.06827)
- [Biderman 2023] Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- [Duan 2023] Duan et al. Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models. [arXiv:2307.01379](https://arxiv.org/abs/2307.01379)
- [Fadeeva 2023] Fadeeva et al. LM-Polygraph: Uncertainty Estimation for Language Models. EMNLP 2023. [arXiv:2311.07383](https://arxiv.org/abs/2311.07383)
- [Guo 2017] Guo et al. On Calibration of Modern Neural Networks. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- [Hendrycks 2017] Hendrycks and Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- [Houlsby 2011] Houlsby et al. Bayesian Active Learning for Classification and Preference Learning. [arXiv:1112.5745](https://arxiv.org/abs/1112.5745)
- [Ielanskyi 2025] Ielanskyi et al. Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation. [arXiv:2510.02279](https://arxiv.org/abs/2510.02279)
- [Kandpal 2022] Kandpal et al. Large Language Models Struggle to Learn Long-Tail Knowledge. ICML 2023. [arXiv:2211.08411](https://arxiv.org/abs/2211.08411)
- [Kirchhof 2025] Kirchhof et al. Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents. ICML 2025. [arXiv:2505.22655](https://arxiv.org/abs/2505.22655)
- [Kirsch 2024] Kirsch. (Implicit) Ensembles of Ensembles: Epistemic Uncertainty Collapse in Large Models. TMLR. [arXiv:2409.02628](https://arxiv.org/abs/2409.02628)
- [Kuhn 2023] Kuhn, Gal and Farquhar. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- [Lee 2018] Lee et al. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- [Malinin 2021] Malinin and Gales. Uncertainty Estimation in Autoregressive Structured Prediction. ICLR 2021. [arXiv:2002.07650](https://arxiv.org/abs/2002.07650)
- [Mucsányi 2024] Mucsányi, Kirchhof and Oh. Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks. [arXiv:2402.19460](https://arxiv.org/abs/2402.19460)
- [Paplhám 2026] Paplhám et al. Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning. [arXiv:2607.14817](https://arxiv.org/abs/2607.14817)
- [Ren 2021] Ren et al. A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. [arXiv:2106.09022](https://arxiv.org/abs/2106.09022)
- [Ren 2022] Ren et al. Out-of-Distribution Detection and Selective Generation for Conditional Language Models. ICLR 2023. [arXiv:2209.15558](https://arxiv.org/abs/2209.15558)
- [Santilli 2025] Santilli et al. Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results. [arXiv:2504.13677](https://arxiv.org/abs/2504.13677)
- [Shi 2024] Shi et al. Training-Free Bayesianization for Low-Rank Adapters of Large Language Models (TFB). [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- [Uppaal 2023] Uppaal et al. Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection. ACL 2023. [arXiv:2305.13282](https://arxiv.org/abs/2305.13282)
- [Vashurin 2024] Vashurin et al. Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph. TACL 2025. [arXiv:2406.15627](https://arxiv.org/abs/2406.15627)
- [Wang 2024] Wang et al. BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models. NeurIPS 2024. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- [Wimmer 2022] Wimmer et al. Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures? [arXiv:2209.03302](https://arxiv.org/abs/2209.03302)
- [Yang 2023] Yang et al. Bayesian Low-rank Adaptation for Large Language Models (Laplace-LoRA). ICLR 2024. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
