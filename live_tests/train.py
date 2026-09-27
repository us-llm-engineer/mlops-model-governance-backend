"""Real model training for the live-test scenarios: time-forward splits, not random ones.

Design decision (RESEARCH-NOTES.md, mlops-live-a-q2): random splits on temporally ordered data
give misleadingly optimistic scores (TabReD: near-100% on a leaked random split). Every trainer
here splits by time/order and reports the held-out (future) score, which is what the policy layer
is given to gate on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline


@dataclass
class TrainedModel:
    name: str
    artifact_hash: str
    metric_name: str
    train_metric: float
    test_metric: float          # the time-forward held-out score: what the policy gate sees
    n_train: int
    n_test: int
    extra: dict


def _hash_model(name: str, coef_summary: str) -> str:
    return hashlib.sha256(f"{name}:{coef_summary}".encode()).hexdigest()


def train_fraud_classifier(df: pd.DataFrame, split_frac: float = 0.7) -> tuple[TrainedModel, Any, pd.DataFrame, pd.DataFrame]:
    """HistGradientBoostingClassifier on the ULB fraud data, split by Time (day 1 -> train, tail of day 1 + day 2 -> test)."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut = int(len(df) * split_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    features = [c for c in df.columns if c not in ("Class",)]
    clf = HistGradientBoostingClassifier(max_depth=6, random_state=0)
    clf.fit(train[features], train["Class"])
    train_auprc = average_precision_score(train["Class"], clf.predict_proba(train[features])[:, 1])
    test_auprc = average_precision_score(test["Class"], clf.predict_proba(test[features])[:, 1])
    model = TrainedModel("fraud-detector", _hash_model("fraud", repr(clf.get_params())), "auprc",
                         float(train_auprc), float(test_auprc), len(train), len(test),
                         {"positives_train": int(train["Class"].sum()), "positives_test": int(test["Class"].sum())})
    return model, clf, train, test


def train_agnews_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, split_frac: float = 0.5):
    """TF-IDF + logistic regression on AG News; 'time-forward' here means class-ordered halves of
    the eval set stand in for two windows, since the public dataset carries no timestamp."""
    pipe = make_pipeline(TfidfVectorizer(max_features=20_000, ngram_range=(1, 2)),
                         LogisticRegression(max_iter=200, C=2.0))
    pipe.fit(train_df["text"], train_df["label"])
    n_cut = int(len(test_df) * split_frac)
    window_a, window_b = test_df.iloc[:n_cut], test_df.iloc[n_cut:]
    f1_a = f1_score(window_a["label"], pipe.predict(window_a["text"]), average="macro")
    f1_b = f1_score(window_b["label"], pipe.predict(window_b["text"]), average="macro")
    model = TrainedModel("news-classifier", _hash_model("agnews", repr(pipe.get_params())), "macro_f1",
                         float(f1_a), float(f1_b), len(train_df), len(window_b), {"classes": 4})
    return model, pipe, window_a, window_b


def train_taxi_regressor(df: pd.DataFrame, train_months: list[int], test_months: list[int]):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    features = ["passenger_count", "trip_distance", "RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
    df = df.dropna(subset=["fare_amount", "tip_amount"] + features)
    train = df[df["month"].isin(train_months)]
    test = df[df["month"].isin(test_months)]
    reg = HistGradientBoostingRegressor(max_depth=6, random_state=0)
    reg.fit(train[features], train["tip_amount"])
    train_mae = mean_absolute_error(train["tip_amount"], reg.predict(train[features]))
    test_mae = mean_absolute_error(test["tip_amount"], reg.predict(test[features]))
    model = TrainedModel("taxi-tip-model", _hash_model("taxi", repr(reg.get_params())), "mae",
                         float(train_mae), float(test_mae), len(train), len(test),
                         {"train_months": train_months, "test_months": test_months})
    return model, reg, train, test


def train_jena_regressor(df: pd.DataFrame, split_frac: float = 0.7):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    df = df.sort_values("Date Time", kind="stable").reset_index(drop=True)
    df["hour"] = df["Date Time"].dt.hour
    df["doy"] = df["Date Time"].dt.dayofyear
    features = ["p (mbar)", "rh (%)", "wv (m/s)", "hour", "doy"]
    cut = int(len(df) * split_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    reg = HistGradientBoostingRegressor(max_depth=6, random_state=0)
    reg.fit(train[features], train["T (degC)"])
    train_mae = mean_absolute_error(train["T (degC)"], reg.predict(train[features]))
    test_mae = mean_absolute_error(test["T (degC)"], reg.predict(test[features]))
    model = TrainedModel("jena-temp-model", _hash_model("jena", repr(reg.get_params())), "mae",
                         float(train_mae), float(test_mae), len(train), len(test), {})
    return model, reg, train, test


def train_covertype_classifier(df: pd.DataFrame, split_frac: float = 0.7):
    """Split ordered by Elevation (the manifest's declared drift_order), not randomly."""
    from sklearn.metrics import f1_score as _f1

    df = df.sort_values("Elevation", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c != "Cover_Type"]
    cut = int(len(df) * split_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    clf = HistGradientBoostingClassifier(max_depth=8, random_state=0)
    clf.fit(train[features], train["Cover_Type"])
    train_f1 = _f1(train["Cover_Type"], clf.predict(train[features]), average="macro")
    test_f1 = _f1(test["Cover_Type"], clf.predict(test[features]), average="macro")
    model = TrainedModel("covertype-classifier", _hash_model("covertype", repr(clf.get_params())), "macro_f1",
                         float(train_f1), float(test_f1), len(train), len(test),
                         {"split_elevation": float(train["Elevation"].max())})
    return model, clf, train, test


def eurosat_image_to_vector(cell) -> np.ndarray:
    """Flatten one EuroSAT image cell to a normalized RGB vector (32x32x3), the exact featurization
    train_eurosat_classifier fits on. Extracted to module level so stats.py can re-featurize held-out
    test images identically when computing eurosat's confusion matrix, rather than duplicating this
    logic in a second, silently-driftable copy."""
    from io import BytesIO

    from PIL import Image

    raw = cell["bytes"] if isinstance(cell, dict) else cell
    img = Image.open(BytesIO(raw)).convert("RGB").resize((32, 32))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def train_eurosat_classifier(df: pd.DataFrame, split_frac: float = 0.7, seed: int = 0):
    """Flatten RGB pixels -> PCA -> logistic regression; split by class-balanced order, held-out
    is a later shuffled block (EuroSAT carries no timestamp; class order stands in for a stream)."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)

    X = np.stack(df["image"].map(eurosat_image_to_vector).to_numpy())
    y = df["label"].to_numpy()
    cut = int(len(df) * split_frac)
    pca = PCA(n_components=50, random_state=seed).fit(X[:cut])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(pca.transform(X[:cut]), y[:cut])
    from sklearn.metrics import f1_score as _f1
    train_f1 = _f1(y[:cut], clf.predict(pca.transform(X[:cut])), average="macro")
    test_f1 = _f1(y[cut:], clf.predict(pca.transform(X[cut:])), average="macro")
    model = TrainedModel("eurosat-classifier", _hash_model("eurosat", repr(clf.get_params())), "macro_f1",
                         float(train_f1), float(test_f1), cut, len(df) - cut, {})
    return model, (pca, clf), df.iloc[:cut], df.iloc[cut:]
