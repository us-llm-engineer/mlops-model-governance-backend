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
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

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

    (out_dir / "IMPROVEMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
