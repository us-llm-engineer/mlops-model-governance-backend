"""Stratified random sampler used to bring a >1 GB source under the size cap.

Written before the implementation. Contract: proportional allocation per stratum (largest remainder),
at least one row from every non-empty stratum, deterministic for a seed, never more rows than asked,
and a manifest that records what was done so a run is replayable from {seed, commit}.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from live_tests.sampling import allocate, stratified_sample, sample_manifest


def frame(sizes, seed=0):
    rng = np.random.default_rng(seed)
    parts = [pd.DataFrame({"k": k, "v": rng.normal(size=n)}) for k, n in sizes.items()]
    return pd.concat(parts, ignore_index=True)


def test_allocate_is_proportional_and_sums_to_target():
    a = allocate({"a": 700, "b": 200, "c": 100}, 100)
    assert a == {"a": 70, "b": 20, "c": 10}
    assert sum(allocate({"a": 333, "b": 333, "c": 334}, 100).values()) == 100


def test_allocate_largest_remainder_and_min_one():
    a = allocate({"big": 10_000, "tiny": 3}, 50)
    assert a["tiny"] >= 1 and sum(a.values()) == 50
    assert all(a[k] <= n for k, n in {"big": 10_000, "tiny": 3}.items())


def test_allocate_never_exceeds_stratum_size_and_returns_all_when_target_ge_total():
    assert allocate({"a": 5, "b": 7}, 100) == {"a": 5, "b": 7}
    a = allocate({"a": 5, "b": 1000}, 600)
    assert a["a"] <= 5 and sum(a.values()) == 600


def test_sample_keeps_stratum_proportions_within_one_row():
    df = frame({"x": 9_000, "y": 900, "z": 100})
    out = stratified_sample(df, ["k"], n=1_000, seed=7)
    assert len(out) == 1_000
    counts = out["k"].value_counts().to_dict()
    assert abs(counts["x"] - 900) <= 1 and abs(counts["y"] - 90) <= 1 and abs(counts["z"] - 10) <= 1


def test_rare_stratum_is_never_dropped():
    df = frame({"common": 100_000, "rare": 17})
    out = stratified_sample(df, ["k"], n=100, seed=1)
    assert (out["k"] == "rare").sum() >= 1


def test_deterministic_for_a_seed_and_different_across_seeds():
    df = frame({"x": 5_000, "y": 5_000})
    a, b = stratified_sample(df, ["k"], 500, seed=3), stratified_sample(df, ["k"], 500, seed=3)
    c = stratified_sample(df, ["k"], 500, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a["v"].equals(c["v"])


def test_multi_column_strata_and_nan_stratum_kept():
    df = pd.DataFrame({"m": [1] * 600 + [2] * 400, "p": ["cash"] * 500 + ["card"] * 100 + [None] * 400})
    out = stratified_sample(df, ["m", "p"], n=100, seed=0)
    assert len(out) == 100
    assert out["p"].isna().sum() >= 1                       # NaN payment type is its own stratum


def test_no_sampling_when_n_at_least_len():
    df = frame({"x": 10, "y": 5})
    assert len(stratified_sample(df, ["k"], n=100, seed=0)) == 15


def test_sample_rows_come_from_the_input_unchanged():
    df = frame({"x": 1_000})
    out = stratified_sample(df, ["k"], n=100, seed=0)
    assert set(out["v"]).issubset(set(df["v"]))


def test_manifest_records_what_was_done():
    df = frame({"x": 900, "y": 100})
    out = stratified_sample(df, ["k"], n=100, seed=5)
    m = sample_manifest(df, out, ["k"], seed=5, source="unit-test")
    assert m["rows_in"] == 1_000 and m["rows_out"] == 100 and m["seed"] == 5
    assert m["strata_cols"] == ["k"] and m["source"] == "unit-test"
    assert m["strata_in"] == {"x": 900, "y": 100} and m["strata_out"] == {"x": 90, "y": 10}
    assert m["method"] == "stratified_random_proportional"
