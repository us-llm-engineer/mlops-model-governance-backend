"""Targeted improvement experiments for the three weakest-scoring live-test models (fraud,
covertype, eurosat), grounded in the direct-data root-cause investigation this session ran before
writing any of this code. These are separate trainer variants, not edits to train.py's real
trainers: the three HTTP-driven scenarios and their already-published experiment write-ups depend
on train.py's exact current behavior, so it stays byte-for-byte unchanged. This module's results
are new, additive benchmarking evidence.

Root causes (verified against real data, not assumed):
- fraud: no hyperparameter tuning, no class weighting despite 0.17% positives.
- covertype: Krummholz has 103/406,708 training rows, all at elevation 2868-3128m, while every
  real test-set Krummholz row sits at >=3129m -- a rarity-plus-zero-overlap extrapolation problem,
  not simply an imbalance problem. Class weighting is tried and reported honestly either way.
- eurosat: PCA(50) on flattened pixels discards spatial structure a CNN can exploit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                              classification_report, f1_score)

from live_tests.stats import _calibration_stats, classifier_stats


def train_fraud_classifier_tuned(df: pd.DataFrame, split_frac: float = 0.7, val_frac: float = 0.15,
                                  seed: int = 0):
    """Time-forward train/validation/test split -- validation carved from the END of the train
    window only, never touching the test window while choosing hyperparameters (preserves the
    time-forward discipline TabReD motivates). class_weight='balanced' plus a small max_depth
    sweep, each candidate scored on the validation window by AUPRC; the winner is refit on
    train+validation and scored once on the untouched test window."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]
    features = [c for c in df.columns if c != "Class"]

    best = None
    for max_depth in (4, 6, 8, 10):
        clf = HistGradientBoostingClassifier(
            max_depth=max_depth, class_weight="balanced", early_stopping=True,
            scoring="average_precision", validation_fraction=0.15, n_iter_no_change=15,
            max_iter=300, random_state=seed)
        clf.fit(train[features], train["Class"])
        val_auprc = average_precision_score(val["Class"], clf.predict_proba(val[features])[:, 1])
        if best is None or val_auprc > best[0]:
            best = (val_auprc, max_depth)
    best_val_auprc, best_depth = best

    final = HistGradientBoostingClassifier(
        max_depth=best_depth, class_weight="balanced", early_stopping=True,
        scoring="average_precision", validation_fraction=0.15, n_iter_no_change=15,
        max_iter=300, random_state=seed)
    final.fit(trainval[features], trainval["Class"])
    train_auprc = average_precision_score(trainval["Class"], final.predict_proba(trainval[features])[:, 1])
    test_proba = final.predict_proba(test[features])[:, 1]
    test_auprc = average_precision_score(test["Class"], test_proba)

    return {
        "name": "fraud-tuned", "best_max_depth": best_depth, "val_auprc": float(best_val_auprc),
        "train_auprc": float(train_auprc), "test_auprc": float(test_auprc),
        "n_train": len(trainval), "n_test": len(test),
    }, final, trainval, test, test_proba


def train_covertype_classifier_tuned(df: pd.DataFrame, split_frac: float = 0.7, val_frac: float = 0.15,
                                      seed: int = 0):
    """Same elevation-ordered split as the real trainer. class_weight='balanced' targets
    Krummholz's severe rarity (103/406,708 rows). max_depth is chosen via a validation window
    carved from the END of the train window only (never touching the test window), matching
    TableShift's own finding that ID and OOD accuracy are correlated (rho=0.81) -- a properly
    tuned model should generalize better even under shift, though TableShift's own large-scale
    study also found no robustness method (DRO/IRM/Group DRO/domain-adversarial) reliably beats
    plain gradient boosting, and often trades away in-distribution performance for no real OOD
    gain -- so this experiment stays within plain, tuned gradient boosting rather than reaching
    for those methods. Reported honestly against whether Krummholz recall specifically improves
    on a class with zero train/test elevation overlap, not assumed to fix it just because the
    aggregate macro-F1 moves."""
    df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]

    best = None
    for max_depth in (4, 6, 8, 10, 12):
        clf = HistGradientBoostingClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed)
        clf.fit(train[features], train["Cover_Type"])
        val_f1 = f1_score(val["Cover_Type"], clf.predict(val[features]), average="macro")
        if best is None or val_f1 > best[0]:
            best = (val_f1, max_depth)
    best_val_f1, best_depth = best

    final = HistGradientBoostingClassifier(max_depth=best_depth, class_weight="balanced", random_state=seed)
    final.fit(trainval[features], trainval["Cover_Type"])
    y_pred_train = final.predict(trainval[features])
    y_pred_test = final.predict(test[features])
    train_f1 = f1_score(trainval["Cover_Type"], y_pred_train, average="macro")
    test_f1 = f1_score(test["Cover_Type"], y_pred_test, average="macro")

    return {
        "name": "covertype-tuned", "best_max_depth": best_depth, "val_macro_f1": float(best_val_f1),
        "train_macro_f1": float(train_f1), "test_macro_f1": float(test_f1),
        "n_train": len(trainval), "n_test": len(test),
    }, final, trainval, test, y_pred_test


def train_covertype_classifier_ensemble(df: pd.DataFrame, split_frac: float = 0.7, val_frac: float = 0.15,
                                         seed: int = 0):
    """CLIMB (arXiv:2505.17451, NeurIPS 2025)'s large-scale benchmark shows ensemble-resampling
    methods (self-paced ensemble, balanced bagging/RF, EasyEnsemble) sometimes beat plain GBDTs by
    large margins on hard multiclass problems -- a different mechanism than class_weight (each
    tree's bootstrap sample is itself rebalanced, not just the loss). Britsch et al. (arXiv:1011.6224)
    directly used this exact Covertype dataset (Cottonwood/Willow as the rare class) and found more
    majority-class data plus SMOTE-augmented minority data improves ROC results -- but on a random
    split, where the rare class fully overlaps train/test in feature space.

    Stated theoretical expectation, checked (not assumed) below: SMOTE-family oversampling generates
    synthetic minority points by interpolating between EXISTING minority examples -- it cannot
    produce a synthetic Krummholz point with Elevation >= 3129m when none of the 103 real training
    Krummholz rows exceed 3128m (interpolation stays within the training convex hull). So this
    experiment is expected to behave like class_weight="balanced" -- move the aggregate metric via
    the classes with real train/test overlap, not fix Krummholz's zero-overlap extrapolation -- and
    is reported honestly against that expectation rather than assumed to confirm it."""
    from imblearn.ensemble import BalancedRandomForestClassifier

    df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]

    best = None
    for n_estimators in (100, 300):
        clf = BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=None,
                                              sampling_strategy="all", replacement=True,
                                              bootstrap=True, random_state=seed, n_jobs=-1)
        clf.fit(train[features], train["Cover_Type"])
        val_f1 = f1_score(val["Cover_Type"], clf.predict(val[features]), average="macro")
        if best is None or val_f1 > best[0]:
            best = (val_f1, n_estimators)
    best_val_f1, best_n = best

    final = BalancedRandomForestClassifier(n_estimators=best_n, max_depth=None, sampling_strategy="all",
                                            replacement=True, bootstrap=True, random_state=seed, n_jobs=-1)
    final.fit(trainval[features], trainval["Cover_Type"])
    y_pred_test = final.predict(test[features])
    train_f1 = f1_score(trainval["Cover_Type"], final.predict(trainval[features]), average="macro")
    test_f1 = f1_score(test["Cover_Type"], y_pred_test, average="macro")

    return {
        "name": "covertype-balanced-rf-ensemble", "best_n_estimators": best_n,
        "val_macro_f1": float(best_val_f1), "train_macro_f1": float(train_f1),
        "test_macro_f1": float(test_f1), "n_train": len(trainval), "n_test": len(test),
    }, final, trainval, test, y_pred_test


def train_covertype_krummholz_binary(df: pd.DataFrame, split_frac: float = 0.7, val_frac: float = 0.15,
                                      seed: int = 0,
                                      weights: tuple = (1, 5, 20, 100, 300, 564, 1000)):
    """Britsch et al. (arXiv:1011.6224)-inspired: collapse the 7-class problem to a binary
    Krummholz-vs-rest target (matching their class-4-as-signal reduction) and scan a class-weight
    parameter to trace a precision/recall frontier (matching their 'scan the cost parameter x to
    produce the ROC curve', Table 2) rather than reporting a single operating point. w=564 is
    class_weight='balanced''s own implied ratio for this class (406708/(2*103*3.5) order of
    magnitude) and is the reference point against which the rest of the sweep is judged.

    Explicitly NOT expected to fix the underlying zero-elevation-overlap extrapolation problem --
    Britsch et al.'s own paper never addresses that regime (RESEARCH-NOTES.md, britsch-q6). This
    tests two independent mechanisms instead: capacity concentration (binary reduction) and ensemble
    bagging diversity (BalancedRandomForestClassifier), each swept, reported honestly against
    whether either beats the already-found Balanced-RF-ensemble multiclass recall (0.083)."""
    from imblearn.ensemble import BalancedRandomForestClassifier

    df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]

    y_train = (train["Cover_Type"] == 7).astype(int)
    y_val = (val["Cover_Type"] == 7).astype(int)
    y_trainval = (trainval["Cover_Type"] == 7).astype(int)
    y_test = (test["Cover_Type"] == 7).astype(int)

    def _metrics(y_true, y_pred):
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1])
        return {"precision": rep["1"]["precision"], "recall": rep["1"]["recall"], "f1": rep["1"]["f1-score"]}

    sweep = {"hgb": [], "balanced_rf": []}
    for w in weights:
        hgb = HistGradientBoostingClassifier(class_weight={0: 1, 1: w}, random_state=seed)
        hgb.fit(train[features], y_train)
        val_m = _metrics(y_val, hgb.predict(val[features]))
        hgb_final = HistGradientBoostingClassifier(class_weight={0: 1, 1: w}, random_state=seed)
        hgb_final.fit(trainval[features], y_trainval)
        test_m = _metrics(y_test, hgb_final.predict(test[features]))
        sweep["hgb"].append({"weight": w, "val": val_m, "test": test_m})

        brf = BalancedRandomForestClassifier(n_estimators=100, sampling_strategy={0: min(len(train) - 1, int(y_train.sum() * w)), 1: int(y_train.sum())},
                                              replacement=True, bootstrap=True, random_state=seed, n_jobs=-1)
        try:
            brf.fit(train[features], y_train)
            val_m2 = _metrics(y_val, brf.predict(val[features]))
            brf_final = BalancedRandomForestClassifier(n_estimators=100, sampling_strategy={0: min(len(trainval) - 1, int(y_trainval.sum() * w)), 1: int(y_trainval.sum())},
                                                        replacement=True, bootstrap=True, random_state=seed, n_jobs=-1)
            brf_final.fit(trainval[features], y_trainval)
            test_m2 = _metrics(y_test, brf_final.predict(test[features]))
        except Exception as e:
            val_m2 = test_m2 = {"precision": None, "recall": None, "f1": None, "error": str(e)}
        sweep["balanced_rf"].append({"weight": w, "val": val_m2, "test": test_m2})

    return sweep, features


def train_covertype_krummholz_binary_monotonic(df: pd.DataFrame, split_frac: float = 0.7,
                                                 val_frac: float = 0.15, seed: int = 0,
                                                 weights: tuple = (1, 5, 20, 100, 300, 564, 1000)):
    """Stage 4: same binary Krummholz-vs-rest reduction and weight sweep as
    train_covertype_krummholz_binary, but with a monotonic increasing constraint on Elevation in
    the HGB model (sklearn HistGradientBoostingClassifier's monotonic_cst, +1 = higher Elevation
    can only push P(Krummholz) up or flat, never down).

    Honest grounding: this is NOT a technique from a paper that solves the zero-elevation-overlap
    extrapolation problem -- no such paper was found. It's a different, real mechanism (Koklev
    2026, arXiv:2512.17945, 'What's the Price of Monotonicity?') that studies monotonic
    constraints as structural regularization on GBTs, finding costs are often small-to-negligible
    on large datasets with few constrained features, and that constraints can act as variance
    reduction when correctly aligned with the true relationship. That paper's own experiments are
    all binary credit-PD classification, NOT elevation extrapolation or Covertype -- it never
    tests this scenario. The hypothesis here is this project's own: Krummholz is ecologically an
    extreme-high-elevation cover type (above treeline), so "higher elevation -> P(Krummholz) does
    not decrease" is a real, defensible domain-knowledge constraint (matching the paper's own
    protocol: "prioritize constraining only features with clear, defensible monotonic
    relationships"). Whether this constraint actually helps a model extrapolate to elevations
    never seen in training (rather than just regularizing within the training range) is an
    empirical question this sweep answers directly, not something assumed from the citation.
    BalancedRandomForestClassifier has no monotonic-constraint mechanism, so this stage is
    HGB-only, compared against stage 3's unconstrained HGB sweep at the same weights."""
    df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    elevation_idx = features.index("Elevation")
    monotonic_cst = [1 if i == elevation_idx else 0 for i in range(len(features))]

    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]

    y_train = (train["Cover_Type"] == 7).astype(int)
    y_val = (val["Cover_Type"] == 7).astype(int)
    y_trainval = (trainval["Cover_Type"] == 7).astype(int)
    y_test = (test["Cover_Type"] == 7).astype(int)

    def _metrics(y_true, y_pred):
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1])
        return {"precision": rep["1"]["precision"], "recall": rep["1"]["recall"], "f1": rep["1"]["f1-score"]}

    sweep = []
    for w in weights:
        hgb = HistGradientBoostingClassifier(class_weight={0: 1, 1: w}, monotonic_cst=monotonic_cst,
                                              random_state=seed)
        hgb.fit(train[features], y_train)
        val_m = _metrics(y_val, hgb.predict(val[features]))
        hgb_final = HistGradientBoostingClassifier(class_weight={0: 1, 1: w}, monotonic_cst=monotonic_cst,
                                                     random_state=seed)
        hgb_final.fit(trainval[features], y_trainval)
        test_m = _metrics(y_test, hgb_final.predict(test[features]))
        sweep.append({"weight": w, "val": val_m, "test": test_m})

    return sweep, features


def compare_covertype_krummholz_binary_monotonic(data_dir: Path, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from live_tests.eda import CAT

    df = pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")
    unconstrained_sweep, _ = train_covertype_krummholz_binary(df)
    monotonic_sweep, _ = train_covertype_krummholz_binary_monotonic(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "krummholz_binary_monotonic_sweep.json").write_text(
        json.dumps({"unconstrained_hgb": unconstrained_sweep["hgb"], "monotonic_hgb": monotonic_sweep},
                    indent=2, default=str))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pts_source, color in [("unconstrained HGB", unconstrained_sweep["hgb"], CAT[0]),
                                     ("monotonic HGB (Elevation +1)", monotonic_sweep, CAT[2])]:
        pts = [(p["test"]["recall"], p["test"]["precision"], p["weight"]) for p in pts_source
               if p["test"]["recall"] is not None]
        if pts:
            xs, ys, ws = zip(*pts)
            ax.plot(xs, ys, marker="o", color=color, label=name)
            for x, y, w in zip(xs, ys, ws):
                ax.annotate(str(w), (x, y), fontsize=7)
    ax.set_xlabel("recall (Krummholz)"); ax.set_ylabel("precision (Krummholz)")
    ax.set_title("Krummholz-vs-rest: monotonic-Elevation constraint vs. unconstrained")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "krummholz_monotonic_pr_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    best_uncon = max(unconstrained_sweep["hgb"], key=lambda p: p["test"]["recall"] or 0)
    best_mono = max(monotonic_sweep, key=lambda p: p["test"]["recall"] or 0)
    return {
        "unconstrained_hgb_sweep": unconstrained_sweep["hgb"],
        "monotonic_hgb_sweep": monotonic_sweep,
        "best_unconstrained": best_uncon,
        "best_monotonic": best_mono,
    }


def compare_covertype_krummholz_binary(data_dir: Path, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from live_tests.eda import CAT

    df = pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")
    sweep, _ = train_covertype_krummholz_binary(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "krummholz_binary_sweep.json").write_text(json.dumps(sweep, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, color) in enumerate([("hgb", CAT[0]), ("balanced_rf", CAT[1])]):
        pts = [(p["test"]["recall"], p["test"]["precision"], p["weight"]) for p in sweep[name]
               if p["test"]["recall"] is not None]
        if pts:
            xs, ys, ws = zip(*pts)
            ax.plot(xs, ys, marker="o", color=color, label=name)
            for x, y, w in zip(xs, ys, ws):
                ax.annotate(str(w), (x, y), fontsize=7)
    ax.set_xlabel("recall (Krummholz)"); ax.set_ylabel("precision (Krummholz)")
    ax.set_title("Krummholz-vs-rest binary: precision/recall sweep over class weight")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "krummholz_pr_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return sweep


def compare_fraud(data_dir: Path, out_dir: Path) -> dict:
    from live_tests import train as train_mod

    df = pd.read_parquet(data_dir / "datasets" / "fraud" / "creditcard.parquet")
    baseline_model, baseline_est, baseline_train, baseline_test = train_mod.train_fraud_classifier(df)
    baseline_features = [c for c in baseline_test.columns if c != "Class"]
    baseline_proba = baseline_est.predict_proba(baseline_test[baseline_features])[:, 1]
    baseline_cal = _calibration_stats(baseline_test["Class"], baseline_proba, out_dir / "fraud_baseline", "fraud-baseline")

    tuned_summary, tuned_est, tuned_train, tuned_test, tuned_proba = train_fraud_classifier_tuned(df)
    tuned_cal = _calibration_stats(tuned_test["Class"], tuned_proba, out_dir / "fraud_tuned", "fraud-tuned")

    return {
        "baseline": {"train_auprc": baseline_model.train_metric, "test_auprc": baseline_model.test_metric,
                     "ece": baseline_cal["ece"], "brier_score": baseline_cal["brier_score"]},
        "tuned": {**tuned_summary, "ece": tuned_cal["ece"], "brier_score": tuned_cal["brier_score"]},
    }


def compare_covertype(data_dir: Path, out_dir: Path) -> dict:
    from live_tests import train as train_mod

    names = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
             5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
    df = pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")

    baseline_model, baseline_est, baseline_train, baseline_test = train_mod.train_covertype_classifier(df)
    baseline_features = [c for c in baseline_test.columns if c != "Cover_Type"]
    baseline_pred = baseline_est.predict(baseline_test[baseline_features])
    baseline_report = classifier_stats(baseline_test["Cover_Type"], baseline_pred,
                                        [names[t] for t in sorted(names)], out_dir / "covertype_baseline",
                                        "covertype-baseline", labels=sorted(names))["classification_report"]

    tuned_summary, tuned_est, tuned_train, tuned_test, tuned_pred = train_covertype_classifier_tuned(df)
    tuned_report = classifier_stats(tuned_test["Cover_Type"], tuned_pred, [names[t] for t in sorted(names)],
                                     out_dir / "covertype_tuned", "covertype-tuned",
                                     labels=sorted(names))["classification_report"]

    ens_summary, ens_est, ens_train, ens_test, ens_pred = train_covertype_classifier_ensemble(df)
    ens_report = classifier_stats(ens_test["Cover_Type"], ens_pred, [names[t] for t in sorted(names)],
                                   out_dir / "covertype_ensemble", "covertype-balanced-rf-ensemble",
                                   labels=sorted(names))["classification_report"]

    return {
        "baseline": {"train_macro_f1": baseline_model.train_metric, "test_macro_f1": baseline_model.test_metric,
                     "krummholz": baseline_report["Krummholz"]},
        "tuned": {**tuned_summary, "krummholz": tuned_report["Krummholz"]},
        "balanced_rf_ensemble": {**ens_summary, "krummholz": ens_report["Krummholz"]},
    }


def train_covertype_capacity_sweep(df: pd.DataFrame, split_frac: float = 0.7, val_frac: float = 0.15,
                                    seed: int = 0, max_iters: tuple = (100, 300, 600),
                                    learning_rates: tuple = (0.1, 0.05), random_split: bool = False,
                                    class_weight: str | None = "balanced",
                                    early_stoppings: tuple = ("auto", False)) -> dict:
    """Stage 5: isolates the capacity axis (max_iter, learning_rate) that stage 1's tuning sweep
    never touched -- stage 1 only swept max_depth, leaving max_iter at sklearn's default of 100 and
    learning_rate at 0.1 on a 581K-row dataset. Published guidance is that lr=0.1/max_iter=100 is
    too aggressive and that lowering the learning rate while raising the round count tends to
    improve generalization, so this sweeps max_iter x learning_rate with max_depth fixed at 8 (the
    deployed trainer's value) to isolate that one axis.

    random_split=True is a CONTROL CONDITION, not an alternate headline result: it exists to test
    whether this codebase's own trainer, given the same capacity sweep, can reach the ~94-95%
    accuracy that published modern Covertype baselines report on a conventional random split. That
    lets us separate two different questions -- "is the model undertrained" vs. "is the
    elevation-ordered protocol simply hard" -- rather than conflating them. The deployed pipeline is
    deliberately elevation-split (a genuine covariate-shift benchmark), so the random-split number
    here is a diagnostic reference point only. It does not fix, improve, or supersede the
    elevation-ordered result; both are reported honestly side by side.

    `class_weight` is exposed as its own axis because it is a REAL confound in that comparison:
    the published ~94-95% figures are accuracy on unweighted models, while this project's whole
    Covertype line uses class_weight='balanced' (which deliberately trades overall accuracy to
    help the rare classes, Krummholz especially). Comparing a balanced-weighted model's accuracy
    against an unweighted published number would understate this trainer and could produce the
    false conclusion "our model is undertrained" when the real difference is the weighting choice.
    So the control is run at class_weight=None for the like-for-like accuracy comparison, and the
    balanced variant is run too, which isolates weighting from split.

    `early_stoppings` exists because of a measured discovery, not a hypothesis: sklearn's
    HistGradientBoostingClassifier auto-enables early stopping once n_samples > 10_000, so on this
    345,701-row training window a model asked for max_iter=600 actually trains only 57 iterations
    (verified directly via `n_iter_`; 7.3s vs 62.8s wall for the same nominal max_iter). That means
    every prior Covertype experiment here -- the deployed trainer and stage 1's max_depth sweep
    included -- has been silently capped at ~57 rounds, and max_iter was never the live lever it
    appeared to be. Worse for THIS protocol specifically: early stopping selects its cutoff on an
    internal RANDOM validation subset, which under an elevation-ordered split is drawn from the
    same elevation band as training, so it measures in-distribution convergence and is blind to the
    shifted test window it is implicitly deciding capacity for. Sweeping early_stopping explicitly
    is therefore the honest way to test the capacity question; leaving it at the default would have
    made every max_iter value collapse to the same ~57-iteration model."""
    if random_split:
        order = np.random.default_rng(seed).permutation(len(df))
        df = df.iloc[order].reset_index(drop=True)
    else:
        df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    cut_test = int(len(df) * split_frac)
    trainval, test = df.iloc[:cut_test], df.iloc[cut_test:]
    cut_val = int(len(trainval) * (1 - val_frac))
    train, val = trainval.iloc[:cut_val], trainval.iloc[cut_val:]

    grid = []
    best = None
    for max_iter in max_iters:
        for learning_rate in learning_rates:
            for early_stopping in early_stoppings:
                clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                                      max_depth=8, class_weight=class_weight,
                                                      early_stopping=early_stopping,
                                                      random_state=seed)
                clf.fit(train[features], train["Cover_Type"])
                val_f1 = f1_score(val["Cover_Type"], clf.predict(val[features]), average="macro")
                grid.append({"max_iter": max_iter, "learning_rate": learning_rate,
                             "early_stopping": early_stopping,
                             "actual_n_iter": int(clf.n_iter_),
                             "val_macro_f1": float(val_f1)})
                if best is None or val_f1 > best[0]:
                    best = (val_f1, max_iter, learning_rate, early_stopping)
    best_val_f1, best_max_iter, best_learning_rate, best_early_stopping = best

    final = HistGradientBoostingClassifier(max_iter=best_max_iter, learning_rate=best_learning_rate,
                                            max_depth=8, class_weight=class_weight,
                                            early_stopping=best_early_stopping, random_state=seed)
    final.fit(trainval[features], trainval["Cover_Type"])
    y_pred_train = final.predict(trainval[features])
    y_pred_test = final.predict(test[features])
    train_macro_f1 = f1_score(trainval["Cover_Type"], y_pred_train, average="macro")
    train_accuracy = accuracy_score(trainval["Cover_Type"], y_pred_train)
    test_macro_f1 = f1_score(test["Cover_Type"], y_pred_test, average="macro")
    test_accuracy = accuracy_score(test["Cover_Type"], y_pred_test)

    return {
        "name": ("covertype-capacity-random" if random_split else "covertype-capacity-elevation")
                 + ("-unweighted" if class_weight is None else "-balanced"),
        "random_split": random_split,
        "class_weight": class_weight,
        "grid": grid,
        "best_max_iter": best_max_iter, "best_learning_rate": best_learning_rate,
        "best_early_stopping": best_early_stopping, "final_actual_n_iter": int(final.n_iter_),
        "best_val_macro_f1": float(best_val_f1),
        "train_macro_f1": float(train_macro_f1), "train_accuracy": float(train_accuracy),
        "test_macro_f1": float(test_macro_f1), "test_accuracy": float(test_accuracy),
        "n_train": len(trainval), "n_test": len(test),
    }


def compare_covertype_capacity(data_dir: Path, out_dir: Path) -> dict:
    """Stage 5 driver: three conditions, designed so each pairwise comparison isolates exactly one
    variable rather than confounding two at once.

      1. elevation split, class_weight='balanced'  -- the real deployed-protocol result
      2. random split,    class_weight='balanced'  -- (2 vs 1) isolates the SPLIT, weighting held
      3. random split,    class_weight=None        -- (3 vs 2) isolates the WEIGHTING, split held

    Condition 3 is the only one directly comparable to the published ~94-95% accuracy Covertype
    baselines, which are unweighted accuracy on a random split. Conditions 2 and 3 are diagnostic
    controls, not headline results for the deployed pipeline (which is deliberately elevation-split
    to benchmark covariate shift). No plotting -- results are tabular by design."""
    df = pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")
    # Only condition 1 is a real result, so only it gets the full grid. Conditions 2 and 3 are
    # reference points ("can this trainer reach the published ~94-95%?"), which one good config
    # answers -- sweeping them too was ~25 wasted fits with no extra information.
    control_kwargs = dict(max_iters=(600,), learning_rates=(0.1,), early_stoppings=(False,))
    elevation_balanced = train_covertype_capacity_sweep(df, random_split=False, class_weight="balanced")
    random_balanced = train_covertype_capacity_sweep(df, random_split=True, class_weight="balanced",
                                                       **control_kwargs)
    random_unweighted = train_covertype_capacity_sweep(df, random_split=True, class_weight=None,
                                                        **control_kwargs)

    result = {
        "elevation_split_balanced": elevation_balanced,
        "random_split_balanced_control": random_balanced,
        "random_split_unweighted_control": random_unweighted,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "covertype_capacity_sweep.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["fraud", "covertype"])
    a = ap.parse_args()
    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if "fraud" in a.only:
        t0 = time.time()
        summary["fraud"] = compare_fraud(data_dir, out_dir)
        summary["fraud"]["seconds"] = round(time.time() - t0, 2)
        print("fraud:", json.dumps(summary["fraud"], indent=2))
    if "covertype" in a.only:
        t0 = time.time()
        summary["covertype"] = compare_covertype(data_dir, out_dir)
        summary["covertype"]["seconds"] = round(time.time() - t0, 2)
        print("covertype:", json.dumps(summary["covertype"], indent=2))
    if "covertype-krummholz-binary" in a.only:
        t0 = time.time()
        summary["covertype_krummholz_binary"] = compare_covertype_krummholz_binary(data_dir, out_dir)
        summary["covertype_krummholz_binary_seconds"] = round(time.time() - t0, 2)
        print("covertype_krummholz_binary:", json.dumps(summary["covertype_krummholz_binary"], indent=2))
    if "covertype-krummholz-monotonic" in a.only:
        t0 = time.time()
        summary["covertype_krummholz_monotonic"] = compare_covertype_krummholz_binary_monotonic(data_dir, out_dir)
        summary["covertype_krummholz_monotonic_seconds"] = round(time.time() - t0, 2)
        print("covertype_krummholz_monotonic:", json.dumps(summary["covertype_krummholz_monotonic"], indent=2))
    if "covertype-capacity" in a.only:
        t0 = time.time()
        summary["covertype_capacity"] = compare_covertype_capacity(data_dir, out_dir)
        summary["covertype_capacity_seconds"] = round(time.time() - t0, 2)
        print("covertype_capacity:", json.dumps(summary["covertype_capacity"], indent=2))

    (out_dir / "IMPROVEMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
