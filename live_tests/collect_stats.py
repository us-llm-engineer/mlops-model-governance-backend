"""Drive eda.py + train.py + stats.py for one or more datasets in a single pass, and accumulate
one top-level RUN_SUMMARY.json/.md -- the artifact a later curation step (README rewrite) should
read first. No HTTP server is involved here, so this does not use harness.History/LiveServer; a
plain dict+JSON accumulator matches prepare_data.py's own manifest-writing pattern.

Each dataset's parquet is read twice (once inside eda_*, once inside train_*) rather than threaded
through both -- total prepared-dataset size is ~336 MB, so this costs low single-digit seconds
total, not worth reworking working code for.

Usage: python -m live_tests.collect_stats --data <dir> --out <dir> [--only fraud taxi ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from live_tests import eda, stats, train
from live_tests.harness import git_commit

TRAINERS = {
    "fraud": lambda data_dir: train.train_fraud_classifier(
        pd.read_parquet(data_dir / "datasets" / "fraud" / "creditcard.parquet")),
    "agnews": lambda data_dir: train.train_agnews_classifier(
        pd.read_parquet(data_dir / "datasets" / "agnews" / "train.parquet"),
        pd.read_parquet(data_dir / "datasets" / "agnews" / "test.parquet")),
    "taxi": lambda data_dir: train.train_taxi_regressor(
        pd.read_parquet(data_dir / "datasets" / "taxi" / "yellow_2019_sample.parquet"),
        train_months=list(range(1, 11)), test_months=[11, 12]),
    "jena": lambda data_dir: train.train_jena_regressor(
        pd.read_parquet(data_dir / "datasets" / "jena" / "jena_climate.parquet")),
    "covertype": lambda data_dir: train.train_covertype_classifier(
        pd.read_parquet(data_dir / "datasets" / "covertype" / "covertype.parquet")),
    "eurosat": lambda data_dir: train.train_eurosat_classifier(
        pd.read_parquet(data_dir / "datasets" / "eurosat" / "eurosat_rgb.parquet")),
}


def package_versions() -> dict:
    versions = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "matplotlib", "PIL", "pyarrow"]:
        try:
            mod = __import__({"scikit-learn": "sklearn", "PIL": "PIL"}.get(pkg, pkg))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def run_one(name: str, data_dir: Path, out_dir: Path) -> dict:
    print(f"== {name}")
    entry = {"dataset": name}

    t0 = time.time()
    eda.DATASETS[name](data_dir, out_dir / "eda" / name)
    entry["eda_seconds"] = round(time.time() - t0, 2)

    t0 = time.time()
    model, estimator, train_df, test_df = TRAINERS[name](data_dir)
    entry["train_seconds"] = round(time.time() - t0, 2)
    entry["metric_name"] = model.metric_name
    entry["train_metric"] = model.train_metric
    entry["test_metric"] = model.test_metric
    entry["n_train"] = model.n_train
    entry["n_test"] = model.n_test

    t0 = time.time()
    (out_dir / "stats" / name).mkdir(parents=True, exist_ok=True)
    stats.STATS[name](model, estimator, train_df, test_df, out_dir / "stats" / name)
    entry["stats_seconds"] = round(time.time() - t0, 2)

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=list(TRAINERS))
    a = ap.parse_args()
    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "datasets": {},
    }
    for name in a.only:
        summary["datasets"][name] = run_one(name, data_dir, out_dir)

    (out_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [f"# Run summary\n", f"commit: `{summary['git_commit']}`\n",
             "| package | version |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["package_versions"].items()]
    lines += ["", "| dataset | metric | train | test | n_train | n_test | eda_s | train_s | stats_s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, e in summary["datasets"].items():
        lines.append(f"| {name} | {e['metric_name']} | {e['train_metric']:.4f} | {e['test_metric']:.4f} | "
                      f"{e['n_train']:,} | {e['n_test']:,} | {e['eda_seconds']} | {e['train_seconds']} | "
                      f"{e['stats_seconds']} |")
    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines))
    print(f"\nwrote {out_dir / 'RUN_SUMMARY.json'} and RUN_SUMMARY.md")


if __name__ == "__main__":
    main()
