"""Prepare the six real datasets into <out>/datasets/<name>/{data files, manifest.json}.

Raw files come from fetch_raw.sh (local cache). Anything over the 1 GB cap is stratified-randomly sampled
(live_tests.sampling, seeded). Row counts are asserted against the published dataset sizes so a truncated
download fails loudly instead of producing a plausible smaller dataset.

Usage: python -m live_tests.prepare_data --cache <raw-download cache dir> --out <prepared-data dir>
       [--only fraud agnews taxi jena covertype eurosat] [--seed 20260927] [--taxi-fraction 0.15]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from live_tests.sampling import sample_manifest, stratified_sample  # noqa: E402

CAP_BYTES = 1_000_000_000

SOURCES = {
    "fraud": "https://www.openml.org/data/download/1673544/phpKo8OWT (OpenML 1597, ULB credit card fraud)",
    "agnews": "https://huggingface.co/datasets/fancyzhx/ag_news",
    "taxi": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-MM.parquet (NYC TLC yellow taxi 2019)",
    "jena": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip (Max Planck Jena)",
    "covertype": "https://archive.ics.uci.edu/static/public/31/covertype.zip (UCI Covertype)",
    "eurosat": "https://huggingface.co/datasets/timm/eurosat-rgb (EuroSAT RGB)",
}
EXPECTED_ROWS = {"fraud": 284_807, "agnews_train": 120_000, "agnews_test": 7_600, "jena": 420_551, "covertype": 581_012}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_dataset(out_dir: Path, name: str, frames: dict, kind: str, raw_files: list, extra: dict) -> dict:
    d = out_dir / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for fname, df in frames.items():
        p = d / fname
        df.to_parquet(p, index=False)
        files[fname] = {"rows": int(len(df)), "columns": list(map(str, df.columns)), "bytes": p.stat().st_size,
                        "sha256": sha256_file(p)}
    manifest = {
        "name": name, "kind": kind, "source": SOURCES[name.split("_")[0]] if name.split("_")[0] in SOURCES else SOURCES[name],
        "raw": [{"file": str(f.name), "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in raw_files],
        "files": files, "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    total = sum(v["bytes"] for v in files.values())
    assert total < CAP_BYTES, f"{name}: prepared size {total} exceeds the 1 GB cap"
    print(f"{name}: {sum(v['rows'] for v in files.values()):,} rows, {total/1e6:.1f} MB prepared")
    return manifest


def prep_fraud(cache: Path, out: Path, seed: int):
    from scipy.io import arff
    raw = cache / "fraud" / "creditcard.arff"
    data, _ = arff.loadarff(str(raw))
    df = pd.DataFrame(data)
    df["Class"] = df["Class"].map(lambda b: int(b.decode() if isinstance(b, bytes) else b)).astype("int8")
    assert len(df) == EXPECTED_ROWS["fraud"], len(df)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)        # keep temporal order (two days)
    write_dataset(out, "fraud", {"creditcard.parquet": df}, "tabular", [raw],
                  {"target": "Class", "time_column": "Time", "class_counts": df["Class"].value_counts().to_dict()})


def prep_agnews(cache: Path, out: Path, seed: int):
    tr, te = pd.read_parquet(cache / "agnews" / "train.parquet"), pd.read_parquet(cache / "agnews" / "test.parquet")
    assert len(tr) == EXPECTED_ROWS["agnews_train"] and len(te) == EXPECTED_ROWS["agnews_test"], (len(tr), len(te))
    write_dataset(out, "agnews", {"train.parquet": tr, "test.parquet": te}, "text",
                  [cache / "agnews" / "train.parquet", cache / "agnews" / "test.parquet"],
                  {"target": "label", "text_column": "text", "classes": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}})


TAXI_COLS = ["tpep_pickup_datetime", "passenger_count", "trip_distance", "RatecodeID", "PULocationID",
             "DOLocationID", "payment_type", "fare_amount", "tip_amount", "total_amount"]


def prep_taxi(cache: Path, out: Path, seed: int, fraction: float):
    files = [cache / "taxi" / f"yellow_tripdata_2019-{m:02d}.parquet" for m in range(1, 13)]
    raw_bytes = sum(f.stat().st_size for f in files)
    parts, per_stratum_in, per_stratum_out, rows_in = [], {}, {}, 0
    for i, f in enumerate(files, start=1):
        df = pd.read_parquet(f, columns=TAXI_COLS)
        df = df[(df["tpep_pickup_datetime"].dt.year == 2019) & (df["tpep_pickup_datetime"].dt.month == i)].copy()
        df["month"] = np.int8(i)
        df["payment_type"] = df["payment_type"].fillna(-1).astype("int16")
        rows_in += len(df)
        sample = stratified_sample(df, ["month", "payment_type"], n=int(round(len(df) * fraction)), seed=seed + i)
        m = sample_manifest(df, sample, ["month", "payment_type"], seed + i, f.name)
        per_stratum_in.update(m["strata_in"])
        per_stratum_out.update(m["strata_out"])
        parts.append(sample)
        print(f"  taxi 2019-{i:02d}: {len(df):,} -> {len(sample):,}")
    sampled = pd.concat(parts, ignore_index=True)
    sampling = {"method": "stratified_random_proportional", "strata_cols": ["month", "payment_type"], "seed": seed,
                "fraction": fraction, "raw_total_bytes": raw_bytes, "rows_in": rows_in, "rows_out": int(len(sampled)),
                "reason": "raw source is over the 1 GB cap", "strata_in": per_stratum_in, "strata_out": per_stratum_out}
    assert raw_bytes > CAP_BYTES, "taxi raw is expected to exceed the 1 GB cap (that is why it is sampled)"
    write_dataset(out, "taxi", {"yellow_2019_sample.parquet": sampled}, "tabular_events", files,
                  {"time_column": "tpep_pickup_datetime", "target": "tip_amount / trip demand", "sampling": sampling})


def prep_jena(cache: Path, out: Path, seed: int):
    raw = cache / "jena" / "jena_climate_2009_2016.csv.zip"
    with zipfile.ZipFile(raw) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    assert len(df) == EXPECTED_ROWS["jena"], len(df)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df.columns = [c.strip() for c in df.columns]
    write_dataset(out, "jena", {"jena_climate.parquet": df}, "time_series", [raw],
                  {"time_column": "Date Time", "target": "T (degC)", "interval": "10 minutes"})


COVER_COLS = (["Elevation", "Aspect", "Slope", "HD_Hydrology", "VD_Hydrology", "HD_Roadways", "Hillshade_9am",
               "Hillshade_Noon", "Hillshade_3pm", "HD_Fire_Points"] + [f"Wilderness_Area{i}" for i in range(1, 5)]
              + [f"Soil_Type{i}" for i in range(1, 41)] + ["Cover_Type"])


def prep_covertype(cache: Path, out: Path, seed: int):
    raw = cache / "covertype" / "covertype.zip"
    with zipfile.ZipFile(raw) as z:
        member = next(n for n in z.namelist() if n.endswith(".data.gz") or n.endswith("covtype.data"))
        blob = z.read(member)
    text = gzip.decompress(blob) if member.endswith(".gz") else blob
    df = pd.read_csv(io.BytesIO(text), header=None, names=COVER_COLS)
    assert len(df) == EXPECTED_ROWS["covertype"], len(df)
    write_dataset(out, "covertype", {"covertype.parquet": df}, "tabular", [raw],
                  {"target": "Cover_Type", "drift_order": "Elevation (standard covertype stream ordering)"})


def prep_eurosat(cache: Path, out: Path, seed: int):
    # This parquet is the "train" split (60%) of the 27,000-image EuroSAT-RGB, not the full set;
    # confirmed via HEAD (55,255,643 bytes) and by class balance across the standard 10 labels.
    raw = cache / "eurosat" / "train.parquet"
    df = pd.read_parquet(raw)
    print("  eurosat columns:", list(df.columns), len(df))
    assert len(df) == 16_200 and df["label"].nunique() == 10, (len(df), df["label"].nunique())
    write_dataset(out, "eurosat", {"eurosat_rgb.parquet": df}, "images", [raw],
                  {"target": "label", "image_column": next((c for c in df.columns if "image" in c.lower()), None)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/data/mlops-live-test-cache"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["agnews", "jena", "covertype", "eurosat", "taxi", "fraud"])
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--taxi-fraction", type=float, default=0.15)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.only:
        t0 = time.time()
        fn = {"fraud": prep_fraud, "agnews": prep_agnews, "jena": prep_jena, "covertype": prep_covertype,
              "eurosat": prep_eurosat}.get(name)
        if name == "taxi":
            prep_taxi(cache, out, a.seed, a.taxi_fraction)
        else:
            fn(cache, out, a.seed)
        print(f"  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
