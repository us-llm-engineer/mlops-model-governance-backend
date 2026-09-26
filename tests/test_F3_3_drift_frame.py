"""F3.3 slim suite: mlops.ext.drift_frame.DriftWindowFrame (pandas 3.0.6).

Frozen BEFORE the module exists. Contract: the API contract section 3.

Design decisions this suite locks down (the contract text leaves these to the
implementer; this suite makes them concrete and testable):

- `DriftWindowFrame(pairs)` wraps a DataFrame with columns "ts", "value" built
  from a sequence of (ts, value) pairs. Constructor validates: >= 30 rows, all
  ts/value finite (no NaN/inf/-inf), ts strictly non-decreasing (ties allowed,
  decreases are not) -- else `mlops.kernel.ValidationFailed`.
- `.rolling_psi(reference, window_size=200, bins=10) -> pandas.Series`: for
  each row, the PSI of the trailing `window_size`-row window of "value" against
  `reference`, computed by calling `mlops.drift.psi_from_samples` internally
  (the oracle used below) -- not a hand-rolled reimplementation. The first
  `window_size - 1` entries are NaN (pandas' own rolling convention).
  `window_size` must be an int in [1, 5000], not a bool, else ValidationFailed.
  Calling `.rolling_psi(...)` also attaches the result onto the frame's
  internal table as a column named "psi", so a later `.export_records()`
  reflects it (this is the hook this suite uses to prove NaN never leaks out
  of `.export_records()`).
- `.to_daily_summary() -> pandas.DataFrame`: groupby on the calendar date of
  "ts" (ts are epoch-second floats), columns count/mean/std/min/max of "value".
- `.export_records() -> list[dict]`: `to_dict("records")` with all NaN/NaT
  replaced by None, so `json.dumps(...)` on the result never raises.
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

import pandas as pd

from mlops.ext.drift_frame import DriftWindowFrame
from mlops.kernel import ValidationFailed
from mlops.drift import psi_from_samples

NAN, INF = float("nan"), float("inf")
DAY = 86400.0


def _pairs(n, start_ts=0.0, step=1.0, value_fn=None):
    """n rows of (ts, value), ts strictly increasing by `step`."""
    if value_fn is None:
        value_fn = lambda i: float(i % 7) + 0.1 * i
    return [(start_ts + i * step, value_fn(i)) for i in range(n)]


def _gauss_like(n, seed_offset=0.0):
    """Deterministic pseudo-random-ish finite floats, no external RNG needed
    for reproducibility across environments."""
    return [math.sin(i * 0.37 + seed_offset) * 5.0 + 10.0 for i in range(n)]


# ---------------------------------------------------------------- construction


def test_construction_exactly_30_rows_ok():
    pairs = _pairs(30)
    f = DriftWindowFrame(pairs)
    assert len(f.export_records()) == 30


def test_construction_29_rows_raises():
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(_pairs(29))


def test_construction_empty_raises():
    with pytest.raises(ValidationFailed):
        DriftWindowFrame([])


def test_construction_nan_value_raises():
    pairs = _pairs(30)
    pairs[5] = (pairs[5][0], NAN)
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(pairs)


def test_construction_inf_value_raises():
    pairs = _pairs(30)
    pairs[10] = (pairs[10][0], INF)
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(pairs)


def test_construction_neg_inf_value_raises():
    pairs = _pairs(30)
    pairs[0] = (pairs[0][0], -INF)
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(pairs)


def test_construction_nan_ts_raises():
    pairs = _pairs(30)
    pairs[3] = (NAN, pairs[3][1])
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(pairs)


def test_construction_decreasing_ts_raises():
    pairs = _pairs(30)
    pairs[15] = (pairs[14][0] - 1.0, pairs[15][1])
    with pytest.raises(ValidationFailed):
        DriftWindowFrame(pairs)


def test_construction_ties_in_ts_allowed():
    pairs = _pairs(30)
    pairs[10] = (pairs[9][0], pairs[10][1])  # equal, not decreasing
    f = DriftWindowFrame(pairs)
    assert len(f.export_records()) == 30


# ---------------------------------------------------------------- rolling_psi


def test_rolling_psi_returns_series_correct_length():
    n = 60
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: _gauss_like(n)[i]))
    reference = _gauss_like(200, seed_offset=1.0)
    result = f.rolling_psi(reference, window_size=30, bins=10)
    assert isinstance(result, pd.Series)
    assert len(result) == n


def test_rolling_psi_leading_nan_count():
    n = 60
    window_size = 30
    values = _gauss_like(n)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=1.0)
    result = f.rolling_psi(reference, window_size=window_size, bins=10)
    result_list = list(result)
    for i in range(window_size - 1):
        assert math.isnan(result_list[i]), f"row {i} expected NaN"
    for i in range(window_size - 1, n):
        assert not math.isnan(result_list[i]), f"row {i} unexpectedly NaN"


def test_rolling_psi_values_match_oracle_default_bins():
    n = 60
    window_size = 30
    values = _gauss_like(n)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=1.0)
    result = list(f.rolling_psi(reference, window_size=window_size, bins=10))
    for i in range(window_size - 1, n):
        window_slice = values[i - window_size + 1 : i + 1]
        expected = psi_from_samples(reference, window_slice, bins=10)
        assert result[i] == pytest.approx(expected, abs=1e-9), f"row {i} mismatch"


def test_rolling_psi_values_match_oracle_custom_bins():
    n = 50
    window_size = 30
    values = _gauss_like(n, seed_offset=2.0)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=3.0)
    result = list(f.rolling_psi(reference, window_size=window_size, bins=5))
    for i in range(window_size - 1, n):
        window_slice = values[i - window_size + 1 : i + 1]
        expected = psi_from_samples(reference, window_slice, bins=5)
        assert result[i] == pytest.approx(expected, abs=1e-9), f"row {i} mismatch"


def test_rolling_psi_window_size_zero_raises():
    f = DriftWindowFrame(_pairs(40))
    with pytest.raises(ValidationFailed):
        f.rolling_psi(_gauss_like(200), window_size=0)


def test_rolling_psi_window_size_negative_raises():
    f = DriftWindowFrame(_pairs(40))
    with pytest.raises(ValidationFailed):
        f.rolling_psi(_gauss_like(200), window_size=-5)


def test_rolling_psi_window_size_bool_raises():
    f = DriftWindowFrame(_pairs(40))
    with pytest.raises(ValidationFailed):
        f.rolling_psi(_gauss_like(200), window_size=True)


def test_rolling_psi_window_size_too_large_raises():
    f = DriftWindowFrame(_pairs(40))
    with pytest.raises(ValidationFailed):
        f.rolling_psi(_gauss_like(200), window_size=5001)


def test_rolling_psi_window_size_boundary_5000_accepted():
    n = 5000
    values = _gauss_like(n, seed_offset=4.0)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=5.0)
    result = list(f.rolling_psi(reference, window_size=5000, bins=10))
    assert len(result) == n
    for i in range(4999):
        assert math.isnan(result[i])
    assert not math.isnan(result[4999])
    assert isinstance(result[4999], float)


# ---------------------------------------------------------------- to_daily_summary


def _day_rows(day_index, values):
    start = day_index * DAY
    return [(start + h * 3600.0, v) for h, v in enumerate(values)]


def test_to_daily_summary_groups_correctly():
    day0_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    day1_vals = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]
    day2_vals = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0]

    pairs = _day_rows(0, day0_vals) + _day_rows(1, day1_vals) + _day_rows(2, day2_vals)
    f = DriftWindowFrame(pairs)
    summary = f.to_daily_summary()

    assert isinstance(summary, pd.DataFrame)
    for col in ("count", "mean", "std", "min", "max"):
        assert col in summary.columns

    assert len(summary) == 3

    expected = [day0_vals, day1_vals, day2_vals]
    for i, vals in enumerate(expected):
        row = summary.iloc[i]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample std (ddof=1)
        std = math.sqrt(var)
        assert int(row["count"]) == n
        assert row["mean"] == pytest.approx(mean, abs=1e-9)
        assert row["std"] == pytest.approx(std, abs=1e-9)
        assert row["min"] == pytest.approx(min(vals), abs=1e-9)
        assert row["max"] == pytest.approx(max(vals), abs=1e-9)


def test_to_daily_summary_no_hand_rolled_missing_columns():
    # 30 rows, 1000s apart (30_000s total), safely inside one calendar day.
    pairs = [(i * 1000.0, float(i)) for i in range(30)]
    f = DriftWindowFrame(pairs)
    summary = f.to_daily_summary()
    assert set(["count", "mean", "std", "min", "max"]).issubset(set(summary.columns))
    assert len(summary) == 1


# ---------------------------------------------------------------- export_records


def test_export_records_basic_shape():
    n = 30
    values = list(range(n))
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: float(values[i])))
    records = f.export_records()
    assert isinstance(records, list)
    assert len(records) == n
    for r in records:
        assert isinstance(r, dict)
        assert "ts" in r
        assert "value" in r


def test_export_records_no_nan_after_rolling_psi_leading_window():
    n = 40
    window_size = 30
    values = _gauss_like(n, seed_offset=6.0)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=7.0)
    f.rolling_psi(reference, window_size=window_size, bins=10)

    records = f.export_records()
    assert len(records) == n

    # rows before the window filled must have had NaN psi, now replaced by None
    for i in range(window_size - 1):
        assert "psi" in records[i]
        assert records[i]["psi"] is None

    # rows after the window filled must carry a real float, not None/NaN
    for i in range(window_size - 1, n):
        val = records[i]["psi"]
        assert val is not None
        assert isinstance(val, float)
        assert not math.isnan(val)

    for r in records:
        for v in r.values():
            if isinstance(v, float):
                assert not math.isnan(v)


def test_export_records_json_dumps_safe_after_rolling_psi():
    n = 40
    window_size = 35
    values = _gauss_like(n, seed_offset=8.0)
    f = DriftWindowFrame(_pairs(n, value_fn=lambda i: values[i]))
    reference = _gauss_like(200, seed_offset=9.0)
    f.rolling_psi(reference, window_size=window_size, bins=10)

    records = f.export_records()
    dumped = json.dumps(records)  # must not raise
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert len(reloaded) == n
