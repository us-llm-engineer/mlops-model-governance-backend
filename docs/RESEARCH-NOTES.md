# Research grounding for the live-test pipeline

Eleven papers, selected and read specifically to answer questions the live-test pipeline actually
needed answered — not a general literature survey. Each entry below states the concrete design
decision it drove, or the explicit limitation it exposed.

## Training and evaluation methodology under distribution shift

- **Wild-Time** ([arXiv:2211.14238](https://arxiv.org/abs/2211.14238)) — multi-seed benchmarking
  under temporal distribution shift (3 seeds). Every experiment in this pipeline is a single-seed
  run; the numbers describe one seed on one commit, not a statistically robust estimate, and every
  write-up says so explicitly rather than implying otherwise.
- **TabReD** ([arXiv:2406.19380](https://arxiv.org/abs/2406.19380)) — shows random splits on
  temporally-ordered tabular data let a model exploit leakage, producing misleadingly optimistic
  scores (near-100% on one benchmark via pure leakage). Every trainer in `live_tests/train.py`
  splits by time/order (`Time`, `Elevation`, calendar date) and reports the held-out *future*
  score, never a random split.
- **TableShift** ([arXiv:2312.07577](https://arxiv.org/abs/2312.07577)) — tabular
  distribution-shift benchmark methodology; informs treating Covertype's elevation-ordered split as
  a genuine distribution shift rather than an arbitrary partition (the EDA's elevation-by-class
  chart shows the classes really do separate by elevation).

## Drift detection and retraining

- **A Framework for Evaluating Concept Drift Detection** (arXiv:2606.07789) — PSI is unreliable
  below roughly 200-300 samples (up to 100% false-alarm rate on driftless data at small window
  sizes); KS is stable across sizes. Every drift check in this pipeline uses windows of roughly 300
  samples for exactly this reason.
- **When Drift Detectors Cry Wolf** (arXiv:2607.17336) — alert-fatigue/false-alarm dynamics in
  drift monitoring; drove an explicit attack design testing whether a detector re-checking a small
  window repeatedly produces false alarms on genuinely driftless data.
- **Multi-Criteria Automated MLOps Pipeline** (arXiv:2512.11541) — aggregate/multivariate drift
  tests can completely miss a real, localized single-feature shift (measured true-positive-rate of
  0.00 in one reported case). Every scenario here runs one named, single-feature drift monitor per
  feature that matters, never one bundled multivariate check.

## Text-classifier and decision-theoretic evaluation

- **Azimuth: Systematic Error Analysis for Text Classification**
  ([arXiv:2212.08216](https://arxiv.org/abs/2212.08216)) — a token-saliency technique (feature
  weight x TF-IDF/IDF score) for diagnosing a linear text classifier's errors, directly computable
  without gradient or embedding infrastructure. Applied to the AG News classifier as per-class
  top-token saliency (`stats.py`'s `token_saliency_stats`), explicitly scoped as inspired by
  Azimuth's saliency technique rather than a reimplementation of its full tool (no
  dataset-shift/embedding/behavioral analysis, which would need sentence-transformer and vector-search
  infrastructure this pipeline doesn't carry).
- **Does the evaluation stand up to evaluation? A first-principle approach to the evaluation of
  classifiers** ([arXiv:2302.12006](https://arxiv.org/abs/2302.12006)) — proves that macro-F1 (and
  F1 generally) is not decision-theory-compliant: it cannot be written as a linear combination of
  confusion-matrix cells with fixed, prediction-independent utility coefficients, which can cause a
  model with a *higher* F1 to be worse in real deployment terms than one with a lower F1. This is a
  genuine, cited limitation of every macro-F1 policy gate in this pipeline (AG News, Covertype),
  stated here rather than left implicit.

## Tree-ensemble diagnostics: calibration, heteroscedasticity, and feature attribution

- **The Hidden Cost of Resampling: How Imbalance Correction Degrades Probability Calibration in
  Tree Ensembles** (arXiv:2606.29720) — resampling (random undersampling especially) can
  substantially distort a tree ensemble's predicted probabilities (Expected Calibration Error
  rising from a baseline of roughly 0.05 to 0.19 in one reported comparison), and recommends always
  reporting a calibration metric (ECE, Brier score) alongside a discrimination metric (AUPRC,
  ROC-AUC), since the latter are insensitive to probability-scale distortion. The fraud classifier
  here does not resample its training data, so the specific failure mode doesn't directly apply —
  but it is exactly why the calibration curve is paired with numeric ECE and Brier score rather
  than left as a plot alone.
- **Distributional Gradient Boosting Machines** ([arXiv:2204.00778](https://arxiv.org/abs/2204.00778))
  — proposes modeling the full conditional distribution of a regression target (via a distinct tree
  ensemble per distributional parameter) rather than a single point prediction, to properly capture
  heteroscedasticity. This pipeline's regressors (taxi, Jena) use plain residual diagnostics plus a
  simpler, sklearn-native three-quantile approach (independent `loss="quantile"` fits at the 5th,
  50th, and 95th percentiles) — explicitly *not* the same guarantee: three independent fits are not
  guaranteed non-crossing the way a single distributional model is, so the measured crossing rate is
  reported directly (2.6% taxi, 0.3% Jena) rather than assumed to be zero.
- **SHAP for additively modeled features in a boosted trees model**
  ([arXiv:2207.14490](https://arxiv.org/abs/2207.14490)) — proves SHAP dependence plots equal
  partial-dependence plots (up to a vertical shift) *only* when a tree ensemble is additive in that
  feature (no tree splits jointly on it and another feature — i.e. depth-1 stumps or explicit
  interaction constraints). Every boosted model in this pipeline uses depth 6-8 with no interaction
  constraints, so this equivalence does not hold here; permutation importance (the feature-attribution
  method actually shipped) is architecturally distinct from SHAP — data-shuffling plus a global
  metric drop, versus tree-structure traversal plus a local additive decomposition — and is described
  as such, not as a SHAP substitute.

## What has no published answer

No paper read for this pipeline gives a promotion-refusal threshold for any specific metric —
every `min_metric`/`max_metric` gate value in the policy layer (AUPRC ≥ 0.5, macro-F1 ≥ 0.8, taxi
MAE ≤ 3.0) is a stated engineering choice, not a number drawn from the literature.
