"""Post-training statistical evidence for the six live-test models: confusion matrices, per-class
metrics, calibration (fraud only), residual diagnostics (regressors), and permutation feature
importance (tabular models only). Mirrors train.py/eda.py's one-function-per-dataset pattern and
reuses eda.py's plotting style (savefig, CAT, SEQ_BLUE, TEXT_PRIMARY) rather than redefining it.

Design decisions (my own, not research numbers unless cited):
- Calibration curve is fraud-only: fraud's AUPRC is a probability-ranking metric the policy gate
  actually consumes; agnews/covertype/eurosat gate on macro-F1, so nothing downstream reads their
  calibration, and true multiclass calibration would mean 4/7/10 subplots per model.
- Permutation importance uses a uniform PERM_IMPORTANCE_SAMPLE_CAP=100_000 rows, N_REPEATS=5, seed
  0 across all four tabular models (fraud, taxi, jena, covertype) -- same convention as
  s2_city_demand.py's TAXI_MAE_MAX, a stated design threshold, not a research number.
- agnews (20k-dim TF-IDF) and eurosat (50 PCA components) get no permutation importance: neither
  has interpretable named features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

from live_tests.eda import CAT, SEQ_BLUE, TEXT_PRIMARY, savefig
from live_tests.train import eurosat_image_to_vector

PERM_IMPORTANCE_SAMPLE_CAP = 100_000
N_REPEATS = 5
SEED = 0


def classifier_stats(y_true, y_pred, class_names: list, out_dir: Path, dataset: str, labels: list = None) -> dict:
    """Confusion matrix heatmap + classification_report JSON + per-class F1 bar chart."""
    import matplotlib.pyplot as plt

    labels = labels if labels is not None else list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 2, 1.1 * len(class_names) + 2))
    im = ax.imshow(cm, cmap=SEQ_BLUE)
    fig.colorbar(im, ax=ax, label="count")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else TEXT_PRIMARY)
    ax.set_title(f"{dataset}: confusion matrix (held-out)")
    savefig(fig, out_dir, "confusion_matrix", f"Held-out confusion matrix for the {dataset} classifier.", [])

    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                    output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 2, 3.5))
    f1s = [report[c]["f1-score"] for c in class_names]
    ax.bar(class_names, f1s, color=(CAT * 2)[:len(class_names)])
    ax.set_ylim(0, 1); ax.set_ylabel("F1 score")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{dataset}: per-class F1 (held-out)")
    savefig(fig, out_dir, "per_class_f1", f"Per-class F1 for the {dataset} classifier's held-out set.", [])

    return {"confusion_matrix": cm.tolist(), "classification_report": report}


def regressor_stats(y_true, y_pred, bucket: pd.Series, bucket_name: str, out_dir: Path, dataset: str) -> dict:
    """Residual distribution histogram + error-by-bucket bar chart."""
    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true) - np.asarray(y_pred)
    quantiles = {str(q): float(np.quantile(residuals, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(residuals, bins=60, color=CAT[0])
    ax.axvline(0, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_xlabel("residual (actual - predicted)"); ax.set_ylabel("count")
    ax.set_title(f"{dataset}: residual distribution (held-out)")
    savefig(fig, out_dir, "residual_distribution",
            f"Residual histogram for the {dataset} regressor's held-out set.", [])

    mae_by_bucket = pd.Series(np.abs(residuals)).groupby(pd.Series(bucket).reset_index(drop=True)).mean().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([str(b) for b in mae_by_bucket.index], mae_by_bucket.values, color=CAT[0])
    ax.set_xlabel(bucket_name); ax.set_ylabel("MAE")
    ax.set_title(f"{dataset}: error by {bucket_name} (held-out)")
    savefig(fig, out_dir, "error_by_bucket", f"MAE broken down by {bucket_name} for the {dataset} regressor.", [])

    return {
        "residual_mean": float(residuals.mean()), "residual_std": float(residuals.std()),
        "residual_quantiles": quantiles,
        "mae_by_bucket": {str(k): float(v) for k, v in mae_by_bucket.items()},
    }


def permutation_importance_stats(estimator, X: pd.DataFrame, y, scoring: str, out_dir: Path, dataset: str,
                                  seed: int = SEED) -> dict:
    """Permutation importance on a capped, seeded sample. Uniform PERM_IMPORTANCE_SAMPLE_CAP/N_REPEATS
    across all four tabular models -- my own design choice, not a research number."""
    import matplotlib.pyplot as plt

    if len(X) > PERM_IMPORTANCE_SAMPLE_CAP:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=PERM_IMPORTANCE_SAMPLE_CAP, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
    else:
        X_sample, y_sample = X, y

    result = permutation_importance(estimator, X_sample, y_sample, scoring=scoring, n_repeats=N_REPEATS,
                                     random_state=seed, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1]
    features = list(X.columns)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1.5))
    ax.barh([features[i] for i in order][::-1], result.importances_mean[order][::-1],
            xerr=result.importances_std[order][::-1], color=CAT[0])
    ax.set_xlabel(f"decrease in {scoring} when permuted")
    ax.set_title(f"{dataset}: permutation feature importance (n={len(X_sample):,}, {N_REPEATS} repeats)")
    savefig(fig, out_dir, "permutation_importance",
            f"Permutation importance for the {dataset} model, capped at {PERM_IMPORTANCE_SAMPLE_CAP:,} rows.", [])

    stats = {
        "scoring": scoring, "n_sample": len(X_sample), "n_repeats": N_REPEATS, "seed": seed,
        "importances_mean": {features[i]: float(result.importances_mean[i]) for i in range(len(features))},
        "importances_std": {features[i]: float(result.importances_std[i]) for i in range(len(features))},
    }
    (out_dir / "permutation_importance.json").write_text(json.dumps(stats, indent=2))
    return stats


def training_curve_boosting(estimator, X_tr, y_tr, out_dir: Path, dataset: str) -> dict:
    """Diagnostic-only refit (same class/params as the real trainer, early_stopping added) to expose
    train_score_/validation_score_ per boosting iteration -- sklearn only populates these when
    early_stopping=True, which the real trainer does not use. A separate instance: the real
    trainer's fit and reported metric are completely untouched by this."""
    import matplotlib.pyplot as plt

    params = {**estimator.get_params(), "early_stopping": True, "validation_fraction": 0.1,
              "n_iter_no_change": 10}
    diag = type(estimator)(**params)
    diag.fit(X_tr, y_tr)

    train_score, val_score = diag.train_score_, diag.validation_score_
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(train_score, color=CAT[0], label="train")
    ax.plot(val_score, color=CAT[1], label="validation (10% held out)")
    ax.set_xlabel("boosting iteration"); ax.set_ylabel("score")
    ax.legend(frameon=False)
    ax.set_title(f"{dataset}: training curve (diagnostic refit, early stopping)")
    savefig(fig, out_dir, "training_curve",
            f"Diagnostic refit with early stopping enabled to expose the training curve for "
            f"{dataset}; the reported metric above comes from the unmodified trainer.", [])

    stats = {"train_score": train_score.tolist(), "validation_score": val_score.tolist(),
             "n_iter": int(diag.n_iter_)}
    (out_dir / "training_curve.json").write_text(json.dumps(stats, indent=2))
    return stats


def training_curve_logreg(X_tr, y_tr, X_val, y_val, lr_params: dict, out_dir: Path, dataset: str,
                           step: int = 10, note: str = "") -> dict:
    """Diagnostic-only warm_start loop over increasing max_iter checkpoints, tracking train/held-out
    log-loss -- a separate LogisticRegression instance from the real trainer's fitted pipeline, so
    the reported classifier is untouched."""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    max_iter_total = lr_params.get("max_iter", 100)
    checkpoints = list(range(step, max_iter_total + 1, step))
    if not checkpoints or checkpoints[-1] != max_iter_total:
        checkpoints.append(max_iter_total)

    clf = LogisticRegression(**{**lr_params, "warm_start": True, "max_iter": step})
    train_losses, val_losses, prev = [], [], 0
    for cp in checkpoints:
        clf.max_iter = cp - prev
        clf.fit(X_tr, y_tr)
        prev = cp
        train_losses.append(float(log_loss(y_tr, clf.predict_proba(X_tr))))
        val_losses.append(float(log_loss(y_val, clf.predict_proba(X_val))))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(checkpoints, train_losses, color=CAT[0], marker="o", label="train")
    ax.plot(checkpoints, val_losses, color=CAT[1], marker="o", label="held-out")
    ax.set_xlabel("cumulative solver iterations"); ax.set_ylabel("log loss")
    ax.legend(frameon=False)
    ax.set_title(f"{dataset}: training curve (diagnostic warm-start refit)")
    caption = (f"Diagnostic warm-start refit tracking log-loss vs. solver iterations for {dataset}; "
               "the reported classifier is the unmodified trainer's own fit.")
    if note:
        caption += " " + note
    savefig(fig, out_dir, "training_curve", caption, [])

    stats = {"checkpoints": checkpoints, "train_log_loss": train_losses, "val_log_loss": val_losses}
    if note:
        stats["limitation"] = note
    (out_dir / "training_curve.json").write_text(json.dumps(stats, indent=2))
    return stats


def quantile_bands(X_tr, y_tr, X_test, y_test, out_dir: Path, dataset: str,
                    quantiles: tuple = (0.05, 0.5, 0.95), seed: int = SEED) -> dict:
    """Three independent HistGradientBoostingRegressor(loss='quantile') fits -- inspired by but
    explicitly NOT DGBM (treediag-q4/q5): DGBM's single distributional model guarantees non-crossing
    quantiles, our three independent fits do not, so the crossing rate is measured and reported
    honestly rather than assumed away."""
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingRegressor

    preds = {}
    for q in quantiles:
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=6, random_state=seed)
        reg.fit(X_tr, y_tr)
        preds[q] = reg.predict(X_test)

    lo, mid, hi = preds[quantiles[0]], preds[quantiles[1]], preds[quantiles[-1]]
    crossing_rate = float(np.mean((lo > mid) | (mid > hi)))

    rng = np.random.default_rng(seed)
    y_test_arr = np.asarray(y_test)
    idx = rng.choice(len(y_test_arr), size=min(300, len(y_test_arr)), replace=False)
    order = idx[np.argsort(y_test_arr[idx])]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x, lo[order], hi[order], color=CAT[0], alpha=0.25,
                     label=f"{quantiles[0]}-{quantiles[-1]} band")
    ax.plot(x, mid[order], color=CAT[0], linewidth=1.2, label=f"{quantiles[1]} prediction")
    ax.scatter(x, y_test_arr[order], color=TEXT_PRIMARY, s=6, alpha=0.5, label="actual")
    ax.set_xlabel(f"held-out sample (sorted by actual, n={len(order)})"); ax.set_ylabel("target")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{dataset}: quantile prediction bands (3 independent fits)")
    savefig(fig, out_dir, "quantile_bands",
            f"Three independent quantile-loss fits ({quantiles}) for {dataset} -- inspired by but "
            f"not DGBM's single distributional model; crossing rate {crossing_rate:.3%}, reported "
            "rather than assumed zero.", [])

    stats = {"quantiles": list(quantiles), "crossing_rate": crossing_rate,
             "quantile_means": {str(q): float(np.mean(p)) for q, p in preds.items()}}
    (out_dir / "quantile_bands.json").write_text(json.dumps(stats, indent=2))
    return stats


def token_saliency_stats(pipeline, class_names: list, out_dir: Path, dataset: str, top_n: int = 8) -> dict:
    """Azimuth-inspired per-class token saliency = LogisticRegression coefficient x the vectorizer's
    IDF weight (evalmeth-q2: directly computable for a linear TF-IDF model, no gradient needed) --
    explicitly NOT the full Azimuth tool (no dataset-shift/embedding/behavioral analysis, which would
    need sentence-transformers+faiss)."""
    import matplotlib.pyplot as plt

    vectorizer, clf = pipeline[0], pipeline[-1]
    vocab = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    coef = clf.coef_
    if coef.shape[0] == 1:
        coef = np.vstack([-coef[0], coef[0]])

    per_class = {}
    fig, axes = plt.subplots(1, len(class_names), figsize=(3.2 * len(class_names), 3.2), sharey=True)
    axes = [axes] if len(class_names) == 1 else axes
    for i, name in enumerate(class_names):
        saliency = coef[i] * idf
        top_idx = np.argsort(saliency)[::-1][:top_n]
        top_tokens = [(str(vocab[j]), float(saliency[j])) for j in top_idx]
        per_class[name] = top_tokens
        axes[i].barh([t for t, _ in top_tokens][::-1], [v for _, v in top_tokens][::-1],
                     color=CAT[i % len(CAT)])
        axes[i].set_title(name, fontsize=10)
    fig.suptitle(f"{dataset}: top-{top_n} salient tokens per class (Azimuth-inspired: weight x idf)")
    savefig(fig, out_dir, "token_saliency",
            f"Azimuth-inspired token saliency (coefficient x idf) per class for {dataset} -- not the "
            "full Azimuth tool (no dataset-shift/embedding/behavioral analysis).", [])

    (out_dir / "token_saliency.json").write_text(json.dumps(per_class, indent=2))
    return per_class


def _calibration_stats(y_true, y_proba, out_dir: Path, dataset: str) -> dict:
    """Quantile-binned calibration curve, plus ECE + Brier Score (treediag-q3, Liu 2026: 'always
    report calibration alongside discrimination', since ROC-AUC/PR-AUC are insensitive to
    probability-scale distortions). Uniform bins would concentrate almost all mass in one bin given
    fraud's 0.17% positive rate, so quantile binning is used for both the curve and ECE -- my own
    design choice for this severely imbalanced case."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    bin_edges = np.quantile(y_proba_arr, np.linspace(0, 1, 11))
    bin_edges[-1] += 1e-9
    bin_ids = np.digitize(y_proba_arr, bin_edges[1:-1])
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_proba_arr)) * abs(y_true_arr[mask].mean() - y_proba_arr[mask].mean())
    brier = float(brier_score_loss(y_true, y_proba))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_PRIMARY, linestyle="--", linewidth=1, label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", color=CAT[0], linewidth=1.5, label=dataset)
    ax.set_xlabel("mean predicted probability (quantile bin)"); ax.set_ylabel("observed fraction positive")
    ax.legend(frameon=False)
    ax.set_title(f"{dataset}: calibration curve (held-out, quantile bins)")
    savefig(fig, out_dir, "calibration_curve",
            f"Quantile-binned calibration curve for {dataset}. ECE={ece:.4f}, Brier={brier:.4f} "
            "(reported alongside the curve per treediag-q3/Liu 2026, since ranking metrics alone "
            "are insensitive to probability-scale distortions).", [])

    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(), "n_bins": 10,
            "strategy": "quantile", "ece": float(ece), "brier_score": brier}


def stats_fraud(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    features = [c for c in test_df.columns if c != "Class"]
    y_true = test_df["Class"]
    y_proba = estimator.predict_proba(test_df[features])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    out = {}
    out["classifier"] = classifier_stats(y_true, y_pred, ["legitimate", "fraud"], out_dir, "fraud")
    out["calibration"] = _calibration_stats(y_true, y_proba, out_dir, "fraud")
    out["permutation_importance"] = permutation_importance_stats(
        estimator, test_df[features], y_true, "average_precision", out_dir, "fraud")
    out["training_curve"] = training_curve_boosting(estimator, train_df[features], train_df["Class"],
                                                      out_dir, "fraud")
    return out


def stats_agnews(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    """train_df/test_df here are window_a/window_b -- both drawn from the held-out test set, not the
    real ~120k-article training set (train_agnews_classifier's return contract doesn't expose it).
    The training-curve diagnostic below is honestly captioned with this limitation."""
    classes = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    y_true = test_df["label"]
    y_pred = estimator.predict(test_df["text"])

    out = {"classifier": classifier_stats(y_true, y_pred, list(classes.values()), out_dir, "agnews",
                                           labels=list(classes.keys()))}
    out["token_saliency"] = token_saliency_stats(estimator, list(classes.values()), out_dir, "agnews")
    vectorizer, clf = estimator[0], estimator[-1]
    out["training_curve"] = training_curve_logreg(
        vectorizer.transform(train_df["text"]), train_df["label"],
        vectorizer.transform(test_df["text"]), test_df["label"], clf.get_params(), out_dir, "agnews",
        note="Trained on window_a (a held-out window), not the real ~120k-article training set, "
             "which train_agnews_classifier's return contract does not expose -- shows solver "
             "convergence on a similarly-distributed sample, not the actual training trajectory.")
    return out


def stats_taxi(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    features = ["passenger_count", "trip_distance", "RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
    y_true = test_df["tip_amount"]
    y_pred = estimator.predict(test_df[features])

    out = {"regressor": regressor_stats(y_true, y_pred, test_df["month"], "month", out_dir, "taxi")}
    out["permutation_importance"] = permutation_importance_stats(
        estimator, test_df[features], y_true, "neg_mean_absolute_error", out_dir, "taxi")
    out["training_curve"] = training_curve_boosting(estimator, train_df[features], train_df["tip_amount"],
                                                      out_dir, "taxi")
    out["quantile_bands"] = quantile_bands(train_df[features], train_df["tip_amount"],
                                            test_df[features], y_true, out_dir, "taxi")
    return out


def stats_jena(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    features = ["p (mbar)", "rh (%)", "wv (m/s)", "hour", "doy"]
    y_true = test_df["T (degC)"]
    y_pred = estimator.predict(test_df[features])
    month = test_df["Date Time"].dt.month

    out = {"regressor": regressor_stats(y_true, y_pred, month, "month", out_dir, "jena")}
    out["permutation_importance"] = permutation_importance_stats(
        estimator, test_df[features], y_true, "neg_mean_absolute_error", out_dir, "jena")
    out["training_curve"] = training_curve_boosting(estimator, train_df[features], train_df["T (degC)"],
                                                      out_dir, "jena")
    out["quantile_bands"] = quantile_bands(train_df[features], train_df["T (degC)"],
                                            test_df[features], y_true, out_dir, "jena")
    return out


def stats_covertype(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    names = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
             5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
    features = [c for c in test_df.columns if c != "Cover_Type"]
    y_true = test_df["Cover_Type"]
    y_pred = estimator.predict(test_df[features])
    labels = sorted(names)

    out = {"classifier": classifier_stats(y_true, y_pred, [names[t] for t in labels], out_dir, "covertype",
                                           labels=labels)}
    out["permutation_importance"] = permutation_importance_stats(
        estimator, test_df[features], y_true, "f1_macro", out_dir, "covertype")
    out["training_curve"] = training_curve_boosting(estimator, train_df[features], train_df["Cover_Type"],
                                                      out_dir, "covertype")
    return out


def stats_eurosat(model, estimator, train_df, test_df, out_dir: Path) -> dict:
    pca, clf = estimator
    X_train = np.stack(train_df["image"].map(eurosat_image_to_vector).to_numpy())
    X_test = np.stack(test_df["image"].map(eurosat_image_to_vector).to_numpy())
    y_true = test_df["label"]
    y_pred = clf.predict(pca.transform(X_test))
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    class_names = [str(i) for i in range(n_classes)]

    out = {"classifier": classifier_stats(y_true, y_pred, class_names, out_dir, "eurosat",
                                           labels=list(range(n_classes)))}
    out["training_curve"] = training_curve_logreg(
        pca.transform(X_train), train_df["label"], pca.transform(X_test), test_df["label"],
        clf.get_params(), out_dir, "eurosat")
    return out


STATS = {"fraud": stats_fraud, "agnews": stats_agnews, "taxi": stats_taxi, "jena": stats_jena,
         "covertype": stats_covertype, "eurosat": stats_eurosat}
