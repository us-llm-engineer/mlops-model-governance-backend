# Improvement-experiment history

Three stages of investigating and improving the two weakest-scoring models from the live-test
statistics pass (`fraud` and `covertype` — see the main [README](../README.md) and
[docs/RESEARCH-NOTES.md](../docs/RESEARCH-NOTES.md)). Each stage's code is preserved unmodified:
[`improve_legacy.py`](improve_legacy.py) (stage 1), [`improve_legacy2.py`](improve_legacy2.py)
(stage 2), [`improve_current.py`](improve_current.py) (stage 3, current). None of the three
touches the original live-test trainers — every number here is a separate benchmarking experiment,
not a change to the deployed pipeline.

## Stage 1 (`improve_legacy.py`) — hyperparameter tuning + class weighting

**Fraud**: `class_weight="balanced"` + a validated `max_depth` sweep (time-forward
train/validation/test, validation carved from the end of the train window only).

| Metric | Baseline | Tuned |
|---|---|---|
| Test AUPRC | 0.3920 | **0.7203** |
| ECE | 0.0019 | 0.0896 (47x worse) |
| Brier score | 0.0012 | 0.0096 (8x worse) |

| Baseline calibration | Tuned calibration |
|---|---|
| ![Fraud baseline calibration](figures/fraud_baseline_calibration.png) | ![Fraud tuned calibration](figures/fraud_tuned_calibration.png) |

A real, large AUPRC gain, at a real, disclosed calibration cost (Liu et al. 2026's exact predicted
tradeoff for resampling/reweighting under severe imbalance — cited in RESEARCH-NOTES.md).

**Covertype**: same `class_weight="balanced"` approach, `max_depth` validated the same way.

| Metric | Baseline | Tuned |
|---|---|---|
| Test macro-F1 | 0.2822 | **0.4051** |
| Krummholz recall | 0.0173 | 0.0015 (worse) |
| Krummholz precision | 0.264 | 0.969 |

| Baseline confusion matrix | Tuned confusion matrix |
|---|---|
| ![Covertype baseline confusion](figures/covertype_baseline_confusion.png) | ![Covertype tuned confusion](figures/covertype_tuned_confusion.png) |

The aggregate metric improved by getting better at the classes with full train/test elevation
overlap — Krummholz (zero overlap between its 103 training rows and 20,407 test rows) got *worse*,
not better, as the model became more conservative overall.

## Stage 2 (`improve_legacy2.py`) — ensemble resampling (CLIMB-inspired)

CLIMB (arXiv:2505.17451, NeurIPS 2025) shows ensemble-resampling methods sometimes beat plain
gradient boosting by large margins on hard multiclass problems, via a different mechanism than
class weighting (each tree's bootstrap sample is itself rebalanced). Tested
`imblearn.ensemble.BalancedRandomForestClassifier` on Covertype:

| Metric | Baseline | Tuned (stage 1) | Balanced-RF ensemble |
|---|---|---|---|
| Test macro-F1 | 0.2822 | 0.4051 | 0.2994 |
| Krummholz recall | 0.0173 | 0.0015 | **0.0834** |
| Krummholz F1 | 0.032 | 0.003 | **0.143** |

![Balanced-RF ensemble confusion matrix](figures/covertype_ensemble_confusion.png)

The first method that made real progress on Krummholz specifically — a genuine tradeoff against
stage 1's aggregate-optimizing result, not a strict improvement on it.

## Stage 3 (`improve_current.py`) — Britsch-inspired binary reduction + weight sweep

Britsch, Gagunashvili & Schmelling (arXiv:1011.6224) is the one paper confirmed to use the real
UCI Covertype dataset (Cottonwood/Willow as a binary "signal" class). Their central technique
(more majority data + SMOTE, under a random split) doesn't transfer — their paper explicitly
assumes full train/test feature overlap, which Krummholz does not have. Two structural ideas *do*
transfer: collapsing to one-vs-rest binary (concentrating model capacity), and scanning a cost/
weight parameter to trace a full precision/recall frontier rather than trusting one heuristic
choice.

| Weight | HGB recall | HGB precision | HGB F1 |
|---|---|---|---|
| 1 | 0.065 | 0.310 | 0.108 |
| 5 | 0.015 | 0.236 | 0.029 |
| **20** | **0.135** | **0.451** | **0.208** |
| 100 | 0.0005 | 0.588 | 0.001 |
| 300 | 0.001 | 0.647 | 0.001 |
| 564 (`class_weight="balanced"`'s own ratio) | 0.0017 | 0.630 | 0.003 |
| 1000 | 0.0021 | 0.538 | 0.004 |

The same sweep, run against `BalancedRandomForestClassifier` (stage 2's bagging/undersampling
mechanism) instead of `HistGradientBoosting`, on the same binary Krummholz-vs-rest reduction:

| Weight | BalancedRF recall | BalancedRF precision | BalancedRF F1 |
|---|---|---|---|
| **1** | **0.083** | 0.434 | 0.139 |
| 5 | 0.053 | 0.796 | 0.100 |
| 20 | 0.020 | 0.930 | 0.038 |
| 100 | 0.006 | 0.962 | 0.012 |
| 300 | 0.004 | 1.000 | 0.008 |
| 564 | 0.001 | 1.000 | 0.003 |
| 1000 | 0.002 | 1.000 | 0.004 |

![Krummholz precision/recall sweep](figures/covertype_krummholz_pr_curve.png)

**The best Krummholz result across every method tried** — 7.8x the baseline recall — found only
by sweeping the weight parameter rather than using `class_weight="balanced"` directly, which lands
at one of the *worst* points in the same sweep. That win is specific to the boosting mechanism:
BalancedRF's own best recall under the same binary reduction (0.083 at w=1) exactly ties its
*multiclass* ensemble recall already reported in stage 2 (0.0834) — i.e. binary reduction did
nothing for the bagging/undersampling mechanism, and increasing BalancedRF's weight only drove its
precision toward 1.000 while collapsing recall toward zero. Capacity concentration via binary
reduction helped boosting specifically, not resampling-based bagging in general. Full sweep data
(both families, 7 weights each, val + test splits) is `../krummholz_binary_sweep.json` (raw JSON,
not curated); see [docs/RESEARCH-NOTES.md](../docs/RESEARCH-NOTES.md) for the honest caveat that
validation-based weight selection is itself uninformative for this class (the val window recreates
the same zero-overlap problem one level down between train and validation).

## EuroSAT: CNN reproduction, attention fusion, and in-training diagnostics

Replaces the question "can a CNN beat the deployed PCA(50)+LogisticRegression EuroSAT classifier"
with two real, trained, evaluated answers: a plain baseline CNN, and a toned-down reproduction of
a balanced multi-task attention architecture (arXiv:2510.15527, CoordAttn + SE fused per block via
a learnable `sigmoid(alpha)` gate). Code: [`eurosat_cnn.py`](eurosat_cnn.py) (architecture,
training, all interpretability functions) and
[`eurosat_test_in_training_viz.py`](eurosat_test_in_training_viz.py) (12 tests against the real
in-training-diagnostic classes, synthetic data with known-correct expected outputs, all passing).
Full paper grounding, including two honest self-corrections found during this work:
[docs/RESEARCH-NOTES.md](../docs/RESEARCH-NOTES.md#improvement-experiment-3-eurosat-cnn-reproduction-attention-fusion-and-in-training-diagnostics).

### Results

| Model | Test macro-F1 | Epochs | Notes |
|---|---|---|---|
| Deployed PCA(50)+LogisticRegression | 0.3989 | — | unchanged, `live_tests/train.py` |
| Baseline CNN | **0.9536** | 30 | 3-block conv-BN-ReLU-maxpool, well-converged |
| Balanced-attention CNN | 0.5522 | 17 | CoordAttn+SE, an intentionally short "quick report" run — under-converged, currently below the simpler baseline CNN, reported as-is |

### The BatchNorm/heavy-augmentation incident, found, diagnosed, and defended against

The attention model's heavy training-time augmentation (rotation, color jitter, blur, random
erasing) corrupted its BatchNorm running statistics relative to clean evaluation images — the
model looked healthy on training data (train_acc climbing normally) while scoring far worse under
real (`model.eval()`) evaluation than under `model.train()` batch statistics, on the *identical*
weights and batch. Confirmed causally (train-mode vs. eval-mode accuracy on one fixed batch, only
the mode changed), not assumed. Fixed by BatchNorm recalibration — a 9.4s forward-only pass over
clean training data, no weight changes — which alone took the attention model from 0.2418 (clean
validation, pre-recalibration) to 0.5522 (real test set, post-recalibration).

That incident directly motivated the in-training diagnostic below, so the same class of problem
gets caught live next time instead of only after a separate post-hoc evaluation:

- **`WeightTrajectoryTracker`** — 3 raw weight scalars from the first conv layer, recorded every
  training step (~4,500 points over the 17-epoch run), reproducing In situ TensorView's
  (arXiv:1806.07382) actual mechanism, not a coarser per-epoch summary.
- A periodic clean-validation evaluation (every few epochs, `eval()` mode on held-out clean data)
  is the mechanism that actually caught the BatchNorm mismatch live, via the widening train/
  clean-validation accuracy gap below.

| | |
|---|---|
| ![Training curves + live train/val gap](figures/eurosat_training_curves.png)<br><sub>Train loss/accuracy climb normally while clean-validation accuracy stays flat — the gap (0.33→0.47) is the BatchNorm mismatch, caught live this time.</sub> | ![Weight trajectory, TensorView style](figures/eurosat_weight_trajectory_3d.png)<br><sub>3 raw weight scalars over ~4,500 steps, colored by time — the trajectory tightens as training converges, not a frozen/diverging path.</sub> |

### EuroSAT: post-training interpretability

| | |
|---|---|
| ![Baseline first-layer filters](figures/eurosat_baseline_filters.png)<br><sub>Every learned 3x3 first-conv filter as an RGB patch. Correctly rendered, but a real limitation found by inspection: a 3x3 kernel is too small to show the edge-detector structure Zeiler & Fergus's own diagnostic (built on AlexNet's 11x11 filters) depends on — see RESEARCH-NOTES.md for the full correction.</sub> | ![Baseline weight histograms](figures/eurosat_baseline_weight_hist.png)<br><sub>Per-layer weight distributions — the architecture-appropriate health check for a 3x3-kernel network (no collapsed/saturated layers).</sub> |
| ![Attention model first-layer filters](figures/eurosat_attn_filters.png)<br><sub>Same visualization, attention model's stem layer.</sub> | ![Attention model weight histograms](figures/eurosat_attn_weight_hist.png)<br><sub>Attention model's per-layer weight distributions, post BN-recalibration.</sub> |
| ![Grad-CAM, baseline model, classes 1-5](figures/eurosat_gradcam_baseline_part1.png)<br><sub>Grad-CAM, 2 examples per class (AnnualCrop-Industrial). Caveat: 64x64 input gives an 8x8 last-conv feature map, smaller than anything validated in the original Grad-CAM paper (7x14x14 on 224x224) — an honest extrapolation.</sub> | ![Grad-CAM, baseline model, classes 6-10](figures/eurosat_gradcam_baseline_part2.png)<br><sub>Grad-CAM, 2 examples per class (Pasture-SeaLake), same caveat as above.</sub> |
| ![Occlusion sensitivity, baseline model](figures/eurosat_occlusion_sensitivity.png)<br><sub>Zeiler & Fergus-style causal check — predicted-class probability as a grey patch sweeps the image.</sub> | |

### EuroSAT: dataset understanding (independent of any trained model)

| | |
|---|---|
| ![Class prototypes](figures/eurosat_class_prototypes.png)<br><sub>Per-class mean image — the typical visual signature per class, independent of any one example.</sub> | ![Raw-pixel PCA embedding](figures/eurosat_raw_pixel_pca.png)<br><sub>PCA(2) on raw normalized pixels — the same feature representation the deployed PCA+LogReg baseline uses, before any CNN touches the data.</sub> |
| ![Confused-pair gallery](figures/eurosat_confused_pairs_gallery.png)<br><sub>Top-3 most-confused class pairs on the attention model's confusion matrix (top: Forest/SeaLake, 771 confusions) — differs from the reproduction paper's own most-confused pair, most plausibly an under-convergence artifact given the attention model's macro-F1 (0.55) vs. the baseline's (0.95), reported honestly rather than presented as a literature match.</sub> | |

### Hardware, checkpoints, and raw statistics

Trained on **NVIDIA Tesla T4** and **NVIDIA L4** (both via Google Colab; no local CUDA in this
environment). The final 17-epoch attention-model run measured 247s (4.1 min) training + 9.4s
BatchNorm recalibration on an L4.

All 5 checkpoints (baseline; the final recalibrated attention model; its pre-recalibration raw
version; and the two checkpoints from the original BatchNorm-mismatch incident, kept per this
project's own "never delete, migrate as legacy" discipline) are published via
[this shared Google Drive folder](https://drive.google.com/drive/folders/1onzEVzL6x5wMEhrklJMoWm7QOW8NssTd?usp=sharing)
rather than committed into git history (no LFS in this repo; keeps the tree lean). Raw statistics
(full per-epoch history, all ~4,500 weight-trajectory points, checkpoint metadata) are in
[`eurosat_stats/`](eurosat_stats/) as plain JSON.

## Summary: what actually worked

| Problem | What helped | What didn't |
|---|---|---|
| Fraud AUPRC | Class weighting + tuning (+84% relative) | — |
| Fraud calibration | — | Class weighting (47x worse ECE, a real cost of the AUPRC gain) |
| Covertype aggregate macro-F1 | Class weighting + tuning (+43% relative) | — |
| Covertype Krummholz recall specifically | Binary reduction + swept weight (+680% relative) | Multiclass class weighting (made it worse); ensemble resampling alone (smaller gain); `class_weight="balanced"` as a single heuristic (one of the worst points in the sweep) |
| EuroSAT test macro-F1 | A CNN at all (+139% relative, baseline CNN vs. deployed PCA+LogReg) | The more complex attention architecture, at only 17 epochs (0.5522, below the simpler baseline — needs a longer run, not a re-architecture) |
