"""Data-understanding visualizations for the six live-test datasets.

Internal engineering EDA artifacts, not a branded product surface, but the dataviz skill's
non-negotiables still apply: one axis, sequential = one hue light->dark for magnitude,
categorical hues in the validated fixed order for identity, a legend for >=2 series, thin
clean marks, no rainbow colormaps. Charts per dataset are chosen by what a reader needs to
trust the scenario's design decisions (class imbalance, the time-forward split point, the
feature used for drift injection), not a fixed template -- 3 for the simplest, up to 5 where
the dataset carries more analytic weight (taxi: largest, sampled, multi-stratum).

Usage: python -m live_tests.eda --data <data dir> --out <data dir>/eda [--only fraud agnews ...]
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Validated categorical order (dataviz skill, references/palette.md), light mode.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#eaf1fb", "#2a78d6", "#0d2c52"])
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e3e2dc"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def savefig(fig, out_dir: Path, name: str, caption: str, captions: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    captions.append((name, caption))
    print(f"  wrote {p.name}")


def write_index(out_dir: Path, dataset: str, captions: list):
    lines = [f"# {dataset}: data-understanding charts\n"]
    for name, cap in captions:
        lines.append(f"## {name}.png\n{cap}\n")
    (out_dir / "README.md").write_text("\n".join(lines))


def write_stats(out_dir: Path, dataset: str, stats: dict):
    """Persist the numeric summary each eda_* already computes for chart titles/captions, so a
    later consumer (README curation, collect_stats.py's RUN_SUMMARY) can read real numbers instead
    of re-deriving them from an image."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    def flatten(prefix, value, rows):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
        else:
            rows.append((prefix, value))

    rows: list = []
    flatten("", stats, rows)
    lines = [f"# {dataset}: numeric summary\n", "| key | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    (out_dir / "STATS.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- fraud (4 charts)
def eda_fraud(data_dir: Path, out: Path):
    df = pd.read_parquet(data_dir / "datasets" / "fraud" / "creditcard.parquet").sort_values("Time")
    cap = []

    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df["Class"].value_counts().sort_index()
    ax.bar(["legitimate", "fraud"], counts.values, color=[CAT[0], CAT[7]], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("transactions (log scale)")
    for x, v in enumerate(counts.values):
        ax.text(x, v * 1.15, f"{v:,} ({v/len(df):.3%})", ha="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_title("Class balance: fraud is 0.17% of transactions")
    savefig(fig, out, "01_class_balance", "Log-scale bar of legitimate vs fraud counts. Justifies using AUPRC "
            "rather than accuracy as the policy-gated metric (accuracy on this split is ~99.8% regardless of "
            "model quality).", cap)

    cut = int(len(df) * 0.7)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df["Time"].iloc[:cut] / 3600, bins=60, color=CAT[0], alpha=0.85, label="train")
    ax.hist(df["Time"].iloc[cut:] / 3600, bins=60, color=CAT[1], alpha=0.85, label="test (future)")
    ax.axvline(df["Time"].iloc[cut] / 3600, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_xlabel("hours since first transaction (spans ~2 days)")
    ax.set_ylabel("transactions")
    ax.legend(frameon=False)
    ax.set_title("Time-forward split point used for training")
    savefig(fig, out, "02_time_split", "Transaction volume over the full ~48h window with the 70/30 time-forward "
            "cutoff marked. Shows the split is a real temporal boundary, not a random shuffle (RESEARCH-NOTES "
            "mlops-live-a-q2).", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["V14"].iloc[:cut], bins=60, color=CAT[0], alpha=0.85, density=True, label="train reference")
    ax.hist(df["V14"].iloc[cut:], bins=60, color=CAT[1], alpha=0.85, density=True, label="test window")
    ax.set_xlabel("V14 (PCA feature used for the drift check)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title("V14: train vs. future window (natural, unshifted)")
    savefig(fig, out, "03_v14_drift_feature", "Distribution of V14, the feature the S1 driver runs through "
            "/v1/drift/check. The natural train/future split is nearly unchanged; the scenario additionally "
            "injects a +8.0 shift to produce a real, detectable drift.", cap)

    fig, ax = plt.subplots(figsize=(5, 4))
    for cls, color, label in [(0, CAT[0], "legitimate"), (1, CAT[7], "fraud")]:
        sub = df[df["Class"] == cls]
        ax.scatter(sub["V14"], np.log1p(sub["Amount"]), s=6, alpha=0.35, color=color, label=label)
    ax.set_xlabel("V14"); ax.set_ylabel("log1p(Amount)")
    ax.legend(frameon=False)
    ax.set_title("V14 vs. Amount by class: separability")
    savefig(fig, out, "04_v14_amount_by_class", "Scatter of the drift feature against transaction amount, colored "
            "by class. Shows fraud is not a trivial amount-threshold problem, motivating a real classifier.", cap)
    write_index(out, "fraud", cap)
    write_stats(out, "fraud", {
        "n_rows": len(df),
        "class_counts": {"legitimate": int(counts[0]), "fraud": int(counts[1])},
        "fraud_pct": float(counts[1] / len(df)),
        "split_frac": 0.7, "split_hour": float(df["Time"].iloc[cut] / 3600),
        "v14_train_mean": float(df["V14"].iloc[:cut].mean()), "v14_train_std": float(df["V14"].iloc[:cut].std()),
        "v14_test_mean": float(df["V14"].iloc[cut:].mean()), "v14_test_std": float(df["V14"].iloc[cut:].std()),
    })


# ---------------------------------------------------------------- agnews (3 charts)
def eda_agnews(data_dir: Path, out: Path):
    d = data_dir / "datasets" / "agnews"
    train, test = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
    classes = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    cap = []

    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = train["label"].map(classes).value_counts().reindex(classes.values())
    ax.bar(counts.index, counts.values, color=CAT[:4])
    ax.set_ylabel("articles")
    ax.set_title("AG News train: class balance (exactly balanced, 30k each)")
    savefig(fig, out, "01_class_balance", "Article counts per class in the training split. Confirms the dataset "
            "is exactly balanced, so macro-F1 differences across windows reflect model behavior, not label skew.", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    lengths = train["text"].str.split().map(len)
    for i, (lbl, name) in enumerate(classes.items()):
        ax.hist(lengths[train["label"] == lbl], bins=30, range=(0, 60), histtype="step",
                color=CAT[i], linewidth=1.8, label=name)
    ax.set_xlabel("words per article"); ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Article length distribution by class")
    savefig(fig, out, "02_length_by_class", "Word-count histograms per class (outline, not filled, so all four "
            "overlap legibly). Sci/Tech and Business articles run slightly longer.", cap)

    from collections import Counter
    import re
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    stop = {"the", "a", "an", "of", "to", "in", "on", "and", "for", "is", "was", "with", "at", "by", "as", "that",
            "it", "its", "be", "are", "has", "have", "his", "her", "he", "she", "after", "new", "said"}
    for i, (lbl, name) in enumerate(classes.items()):
        words = re.findall(r"[a-zA-Z']+", " ".join(train[train["label"] == lbl]["text"].sample(4000, random_state=0)).lower())
        top = Counter(w for w in words if w not in stop and len(w) > 2).most_common(8)
        axes[i].barh([w for w, _ in top][::-1], [n for _, n in top][::-1], color=CAT[i])
        axes[i].set_title(name, fontsize=10)
    fig.suptitle("Most frequent non-stopword tokens per class (4,000-article sample)")
    savefig(fig, out, "03_top_tokens_by_class", "Top-8 tokens per class. Shows the vocabulary signal a TF-IDF "
            "classifier relies on, and why the four classes are linearly separable in bag-of-words space.", cap)
    write_index(out, "agnews", cap)
    write_stats(out, "agnews", {
        "n_train": len(train), "n_test": len(test),
        "class_counts_train": {name: int(counts[name]) for name in classes.values()},
        "word_length_mean": float(lengths.mean()), "word_length_median": float(lengths.median()),
    })


# ---------------------------------------------------------------- jena (3 charts)
def eda_jena(data_dir: Path, out: Path):
    df = pd.read_parquet(data_dir / "datasets" / "jena" / "jena_climate.parquet").sort_values("Date Time")
    cap = []

    fig, ax = plt.subplots(figsize=(9, 3.2))
    daily = df.set_index("Date Time")["T (degC)"].resample("1D").mean()
    ax.plot(daily.index, daily.values, color=CAT[0], linewidth=1.0)
    cut = df["Date Time"].iloc[int(len(df) * 0.7)]
    ax.axvline(cut, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_ylabel("daily mean T (degC)")
    ax.set_title("Jena temperature, 2009-2016, with the 70/30 time-forward split")
    savefig(fig, out, "01_temperature_over_time", "Daily-mean temperature across the full 8-year record with the "
            "train/test cutoff marked, showing the split lands mid-series with full seasonal cycles on both sides.", cap)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    monthly = df.assign(month=df["Date Time"].dt.month).groupby("month")["T (degC)"]
    bp = ax.boxplot([monthly.get_group(m) for m in range(1, 13)], patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(CAT[0]); patch.set_alpha(0.55)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax.set_ylabel("T (degC)")
    ax.set_title("Seasonal temperature distribution by month")
    savefig(fig, out, "02_seasonal_boxplot", "Month-by-month temperature spread across all 8 years. Establishes "
            "January vs. July as a real, large seasonal shift usable for a drift-check demonstration.", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    h = ax.hist2d(df["p (mbar)"], df["T (degC)"], bins=60, cmap=SEQ_BLUE)
    fig.colorbar(h[3], ax=ax, label="observations")
    ax.set_xlabel("pressure (mbar)"); ax.set_ylabel("T (degC)")
    ax.set_title("Pressure vs. temperature (one sequential hue = density)")
    savefig(fig, out, "03_pressure_vs_temp_density", "2D density of pressure against temperature, using the single "
            "sequential blue ramp for magnitude (never a rainbow). Motivates including pressure as a regressor "
            "feature.", cap)
    write_index(out, "jena", cap)
    write_stats(out, "jena", {
        "n_rows": len(df), "split_frac": 0.7, "split_date": str(cut),
        "monthly_temp_mean_c": {int(m): float(monthly.get_group(m).mean()) for m in range(1, 13)},
        "monthly_temp_std_c": {int(m): float(monthly.get_group(m).std()) for m in range(1, 13)},
    })


# ---------------------------------------------------------------- covertype (4 charts)
def eda_covertype(data_dir: Path, out: Path):
    df = pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")
    names = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
             5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
    cap = []

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = df["Cover_Type"].value_counts().reindex(sorted(names)).rename(index=names)
    ax.bar(range(7), counts.values, color=CAT[:7])
    ax.set_xticks(range(7)); ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cells")
    ax.set_title("Cover type class balance (7 classes, real imbalance)")
    savefig(fig, out, "01_class_balance", "Counts per forest cover type. Motivates macro-F1 (not accuracy) as the "
            "policy-gated metric, since two classes dominate the 581k cells.", cap)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = sorted(names)
    bp = ax.boxplot([df.loc[df["Cover_Type"] == t, "Elevation"] for t in order], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], CAT[:7]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_xticklabels([names[t] for t in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation by cover type: why an elevation-ordered split is a real shift")
    savefig(fig, out, "02_elevation_by_class", "Elevation distribution per class. Classes separate strongly by "
            "elevation, so splitting the data by Elevation (this scenario's declared drift_order) produces a "
            "genuine class-mix shift between train and test, not an arbitrary one.", cap)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    cut = df.sort_values("Elevation")["Elevation"].iloc[int(len(df) * 0.7)]
    ax.hist(df["Elevation"], bins=80, color=CAT[0])
    ax.axvline(cut, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel("cells")
    ax.set_title("Elevation histogram with the 70/30 split point")
    savefig(fig, out, "03_elevation_split", "Full elevation histogram with the split point used to build the "
            "training/held-out sets and, separately, the reference/window pair for the drift check.", cap)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, t in enumerate(order):
        sub = df[df["Cover_Type"] == t].sample(min(2000, (df["Cover_Type"] == t).sum()), random_state=0)
        ax.scatter(sub["Elevation"], sub["Slope"], s=4, alpha=0.3, color=CAT[i], label=names[t])
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel("Slope (deg)")
    ax.legend(frameon=False, fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("Elevation vs. Slope by class (subsampled)")
    savefig(fig, out, "04_elevation_slope_by_class", "Elevation against slope, colored by class, subsampled for "
            "legibility. Shows the two features jointly carry more class signal than elevation alone.", cap)
    write_index(out, "covertype", cap)
    write_stats(out, "covertype", {
        "n_rows": len(df), "split_frac": 0.7, "split_elevation_m": float(cut),
        "class_counts": {names[t]: int(counts[names[t]]) for t in order},
        "elevation_by_class_m": {
            names[t]: {
                "mean": float(df.loc[df["Cover_Type"] == t, "Elevation"].mean()),
                "median": float(df.loc[df["Cover_Type"] == t, "Elevation"].median()),
                "min": float(df.loc[df["Cover_Type"] == t, "Elevation"].min()),
                "max": float(df.loc[df["Cover_Type"] == t, "Elevation"].max()),
            } for t in order
        },
    })


# ---------------------------------------------------------------- eurosat (3 charts)
def eda_eurosat(data_dir: Path, out: Path):
    from PIL import Image
    df = pd.read_parquet(data_dir / "datasets" / "eurosat" / "eurosat_rgb.parquet")
    label_names = {i: n for i, n in enumerate(sorted(df["image_id"].str.extract(r"^([A-Za-z]+)")[0].dropna().unique()))} \
        if False else None
    cap = []

    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts = df["label"].value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values, color=(CAT * 2)[:len(counts)])
    ax.set_xlabel("label id"); ax.set_ylabel("images")
    ax.set_title(f"EuroSAT RGB train split: class balance ({len(counts)} classes)")
    savefig(fig, out, "01_class_balance", "Image counts per label id in the 16,200-image train split (60% of the "
            "full 27,000-image EuroSAT-RGB). Roughly balanced, 1,195-1,863 images per class.", cap)

    def to_img(cell):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(raw)).convert("RGB")

    rng = np.random.default_rng(0)
    n_classes = df["label"].nunique()
    fig, axes = plt.subplots(2, n_classes, figsize=(1.35 * n_classes, 2.9))
    for lbl in range(n_classes):
        rows = df[df["label"] == lbl].sample(2, random_state=int(rng.integers(0, 1_000_000)))
        for r, (_, row) in enumerate(rows.iterrows()):
            axes[r, lbl].imshow(to_img(row["image"]))
            axes[r, lbl].axis("off")
            if r == 0:
                axes[r, lbl].set_title(str(lbl), fontsize=9)
    fig.suptitle("Two sample tiles per class label")
    savefig(fig, out, "02_sample_tiles_by_class", "A 2xN grid of real 64x64 tiles, two per class. The 'understand "
            "the input' chart for an image dataset: what the classifier actually sees.", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sample = df.sample(min(800, len(df)), random_state=0)
    means = np.stack([np.asarray(to_img(c), dtype=np.float32).mean(axis=(0, 1)) for c in sample["image"]])
    for i, (ch, color) in enumerate(zip(["R", "G", "B"], [CAT[7], CAT[2], CAT[0]])):
        ax.hist(means[:, i], bins=40, histtype="step", linewidth=1.8, color=color, label=ch)
    ax.set_xlabel("mean channel intensity (0-255)"); ax.set_ylabel("images")
    ax.legend(frameon=False)
    ax.set_title("Per-image mean RGB intensity (800-image sample)")
    savefig(fig, out, "03_mean_rgb_intensity", "Distribution of each image's mean R/G/B value. Motivates the "
            "0-1 pixel scaling used before PCA in the trainer, and shows the three channels are not degenerate.", cap)
    write_index(out, "eurosat", cap)
    write_stats(out, "eurosat", {
        "n_rows": len(df), "n_classes": len(counts),
        "class_counts": {int(i): int(v) for i, v in counts.items()},
        "rgb_mean_intensity": {ch: float(means[:, i].mean()) for i, ch in enumerate(["R", "G", "B"])},
        "rgb_std_intensity": {ch: float(means[:, i].std()) for i, ch in enumerate(["R", "G", "B"])},
    })


# ---------------------------------------------------------------- taxi (5 charts, written separately once sampled)
def eda_taxi(data_dir: Path, out: Path):
    df = pd.read_parquet(data_dir / "datasets" / "taxi" / "yellow_2019_sample.parquet")
    cap = []

    fig, ax = plt.subplots(figsize=(8, 3.5))
    by_month = df["month"].value_counts().sort_index()
    ax.bar(by_month.index, by_month.values, color=CAT[0])
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month (2019)"); ax.set_ylabel("sampled trips")
    ax.set_title("Sampled trip volume by month (stratified sample)")
    savefig(fig, out, "01_trips_by_month", "Sampled-trip counts per month. Confirms the stratified sampler kept "
            "each month proportionally represented after reducing the >1 GB raw source under the 1 GB cap.", cap)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pt_names = {1: "credit card", 2: "cash", 3: "no charge", 4: "dispute", 5: "unknown", -1: "missing"}
    jan = df[df["month"] == 1]["payment_type"].map(pt_names).fillna("other").value_counts()
    jul = df[df["month"] == 7]["payment_type"].map(pt_names).fillna("other").value_counts()
    idx = sorted(set(jan.index) | set(jul.index))
    x = np.arange(len(idx)); w = 0.35
    ax.bar(x - w / 2, [jan.get(i, 0) for i in idx], w, color=CAT[0], label="January")
    ax.bar(x + w / 2, [jul.get(i, 0) for i in idx], w, color=CAT[1], label="July")
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("sampled trips")
    ax.legend(frameon=False)
    ax.set_title("Payment-type mix: January vs. July (the drift axis used in S2)")
    savefig(fig, out, "02_payment_type_jan_vs_jul", "Payment-type distribution for the two months compared in the "
            "S2 drift check. Grouped bars (paired categories), not a dual axis.", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["trip_distance"].clip(0, 20), bins=60, color=CAT[0])
    ax.set_xlabel("trip distance (miles, clipped at 20)"); ax.set_ylabel("sampled trips")
    ax.set_title("Trip distance distribution")
    savefig(fig, out, "03_trip_distance", "Trip-distance histogram, clipped for readability. Right-skewed as "
            "expected for a metro taxi market; motivates a tree-based regressor over a linear one.", cap)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    tip_pct = (df["tip_amount"] / df["fare_amount"].replace(0, np.nan)).clip(0, 0.6) * 100
    ax.hist(tip_pct.dropna(), bins=60, color=CAT[0])
    ax.set_xlabel("tip as % of fare (clipped at 60%)"); ax.set_ylabel("sampled trips")
    ax.set_title("Tip percentage distribution (the regression target's shape)")
    savefig(fig, out, "04_tip_percentage", "Distribution of tip_amount/fare_amount, the quantity the S2 taxi "
            "model is trained to predict via tip_amount. The large spike near 0 (cash trips, unrecorded tips) "
            "is a real property of the target, not sampling noise.", cap)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist2d(df["month"], df["trip_distance"].clip(0, 20), bins=[12, 40], cmap=SEQ_BLUE)
    ax.set_xlabel("month"); ax.set_ylabel("trip distance (mi, clipped)")
    ax.set_title("Trip distance density by month (one sequential hue)")
    savefig(fig, out, "05_distance_by_month_density", "2D density of trip distance across the year, single blue "
            "ramp for magnitude. Surfaces any month-to-month shift in typical trip length beyond payment type.", cap)
    write_index(out, "taxi", cap)
    write_stats(out, "taxi", {
        "n_rows": len(df),
        "trips_by_month": {int(m): int(v) for m, v in by_month.items()},
        "payment_mix_january": {str(k): int(v) for k, v in jan.items()},
        "payment_mix_july": {str(k): int(v) for k, v in jul.items()},
        "trip_distance_quantiles_mi": {q: float(df["trip_distance"].quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
        "tip_pct_quantiles": {q: float(tip_pct.dropna().quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    })


DATASETS = {"fraud": eda_fraud, "agnews": eda_agnews, "jena": eda_jena,
            "covertype": eda_covertype, "eurosat": eda_eurosat, "taxi": eda_taxi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=list(DATASETS))
    a = ap.parse_args()
    data_dir, out_dir = Path(a.data), Path(a.out)
    for name in a.only:
        print(f"== {name}")
        DATASETS[name](data_dir, out_dir / name)


if __name__ == "__main__":
    main()
