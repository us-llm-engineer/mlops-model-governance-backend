"""Seeded stratified random sampling, used to bring a source larger than 1 GB under the cap."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

METHOD = "stratified_random_proportional"


def allocate(sizes: Dict[str, int], target: int) -> Dict[str, int]:
    """Proportional allocation with largest remainders; every non-empty stratum keeps >= 1 row when target allows."""
    total = sum(sizes.values())
    if target >= total:
        return dict(sizes)
    exact = {k: n * target / total for k, n in sizes.items()}
    alloc = {k: min(sizes[k], max(1 if sizes[k] > 0 and target >= len(sizes) else 0, math.floor(v))) for k, v in exact.items()}
    diff = target - sum(alloc.values())
    if diff > 0:
        order = sorted(sizes, key=lambda k: (-(exact[k] - math.floor(exact[k])), k))
        while diff > 0:
            progressed = False
            for k in order:
                if diff == 0:
                    break
                if alloc[k] < sizes[k]:
                    alloc[k] += 1
                    diff -= 1
                    progressed = True
            if not progressed:
                break
    elif diff < 0:
        order = sorted(sizes, key=lambda k: (-alloc[k], k))
        while diff < 0:
            for k in order:
                if diff == 0:
                    break
                if alloc[k] > 1:
                    alloc[k] -= 1
                    diff += 1
    return alloc


def _labels(df: pd.DataFrame, strata_cols: List[str]) -> pd.Series:
    parts = [df[c].astype("object").where(df[c].notna(), "<NA>").astype(str) for c in strata_cols]
    label = parts[0]
    for p in parts[1:]:
        label = label + "|" + p
    return label


def stratified_sample(df: pd.DataFrame, strata_cols: List[str], n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    labels = _labels(df, strata_cols)
    sizes = labels.value_counts().to_dict()
    alloc = allocate(sizes, n)
    rng = np.random.default_rng(seed)
    positions = pd.Series(np.arange(len(df)))
    picked = []
    for key in sorted(sizes):
        k = alloc[key]
        if k <= 0:
            continue
        pool = positions[(labels == key).to_numpy()].to_numpy()
        picked.append(rng.choice(pool, size=k, replace=False))
    idx = np.sort(np.concatenate(picked))
    return df.iloc[idx]


def sample_manifest(df_in: pd.DataFrame, df_out: pd.DataFrame, strata_cols: List[str], seed: int, source: str) -> dict:
    return {
        "method": METHOD,
        "source": source,
        "seed": seed,
        "strata_cols": list(strata_cols),
        "rows_in": int(len(df_in)),
        "rows_out": int(len(df_out)),
        "strata_in": {k: int(v) for k, v in _labels(df_in, strata_cols).value_counts().sort_index().items()},
        "strata_out": {k: int(v) for k, v in _labels(df_out, strata_cols).value_counts().sort_index().items()},
    }
