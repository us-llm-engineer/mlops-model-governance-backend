"""F3.4 slim suite: mlops.ext.calibration_sk (scikit-learn 1.9.1).

Frozen BEFORE the module exists. Contract: the API contract section 4.

Design decisions this suite locks down (the contract text leaves these to the
implementer; this suite makes them concrete and testable):

- `IsotonicCrossCheck.fit(x, y)`: >= 10 points, len(x) == len(y), every y in
  [0.0, 1.0] inclusive, else `mlops.kernel.ValidationFailed`.
- `IsotonicCrossCheck.predict(x) -> list[float]`: always in [0, 1], even for x
  far outside the fitted range (belt-and-braces clamp on top of sklearn's own
  `out_of_bounds="clip"`).
- `PlattCrossCheck.fit(x, y_binary)` / `.predict_proba_positive(x) -> list[float]`:
  always in [0, 1].
- `compare_to_calibrator(existing_result, x, y_binary) -> dict` with EXACTLY
  the keys "isotonic", "platt", "existing_agrees_within": "isotonic" is the
  list[float] `IsotonicCrossCheck().fit(x, y_binary).predict(x)` would produce
  (same length as x, same values), "platt" is the analogous
  `PlattCrossCheck().fit(x, y_binary).predict_proba_positive(x)` result, and
  "existing_agrees_within" is a non-negative float (a mean-absolute-difference
  style agreement score against `existing_result`, shape mirroring
  `mlops.policy_calibration.evaluate()`'s output dict).
- Import hygiene: `import mlops.ext.calibration_sk` (or `from mlops.ext import
  calibration_sk`) must NOT import sklearn eagerly; sklearn is only imported
  once a function on the module is actually called. Verified via a clean
  subprocess so earlier tests in this same file (which do import sklearn
  transitively) cannot contaminate the check.
"""
import json
import math
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import ValidationFailed

EXEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")


def _monotone_xy(n=40):
    """Strictly monotone-increasing-in-expectation synthetic dataset:
    y is a clean, noiseless non-decreasing step function of x, so isotonic
    regression should recover it almost exactly."""
    x = [float(i) for i in range(n)]
    y = [0.0 if i < n // 2 else 1.0 for i in range(n)]
    return x, y


def _mixed_xy(n=20):
    x = [float(i) - 10.0 for i in range(n)]
    y = [0.0, 1.0] * (n // 2)
    return x, y


def _existing_result_like_evaluate():
    """Shape mirrors mlops.policy_calibration.evaluate()'s return dict."""
    return {
        "n": 20,
        "missed": 2,
        "false_alarms": 1,
        "blocked_correctly": 10,
        "missed_detection_rate": 0.1,
        "false_alarm_rate": 0.05,
    }


# ---------------------------------------------------------------- IsotonicCrossCheck.fit


def test_isotonic_fit_exactly_10_points_ok():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x = [float(i) for i in range(10)]
    y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    check = IsotonicCrossCheck()
    check.fit(x, y)  # must not raise


def test_isotonic_fit_9_points_raises():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x = [float(i) for i in range(9)]
    y = [0.0] * 4 + [1.0] * 5
    with pytest.raises(ValidationFailed):
        IsotonicCrossCheck().fit(x, y)


def test_isotonic_fit_length_mismatch_raises():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x = [float(i) for i in range(10)]
    y = [0.0] * 9
    with pytest.raises(ValidationFailed):
        IsotonicCrossCheck().fit(x, y)


def test_isotonic_fit_y_above_one_raises():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy()
    y = list(y)
    y[0] = 1.5
    with pytest.raises(ValidationFailed):
        IsotonicCrossCheck().fit(x, y)


def test_isotonic_fit_y_below_zero_raises():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy()
    y = list(y)
    y[-1] = -0.2
    with pytest.raises(ValidationFailed):
        IsotonicCrossCheck().fit(x, y)


def test_isotonic_fit_y_boundary_0_and_1_allowed():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x = [float(i) for i in range(10)]
    y = [0.0] * 5 + [1.0] * 5
    IsotonicCrossCheck().fit(x, y)  # must not raise


# ---------------------------------------------------------------- IsotonicCrossCheck.predict


def test_isotonic_predict_basic_in_range():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy()
    check = IsotonicCrossCheck()
    check.fit(x, y)
    preds = check.predict(x)
    assert isinstance(preds, list)
    assert len(preds) == len(x)
    for p in preds:
        assert 0.0 <= p <= 1.0


def test_isotonic_predict_out_of_range_x_still_bounded():
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy()
    check = IsotonicCrossCheck()
    check.fit(x, y)
    far = [-1e9, -1e6, 1e6, 1e9]
    preds = check.predict(far)
    assert len(preds) == len(far)
    for p in preds:
        assert 0.0 <= p <= 1.0
        assert not math.isnan(p)


def test_isotonic_predict_clamps_even_if_underlying_model_misbehaves():
    """Belt-and-braces requirement from the contract: out_of_bounds='clip'
    alone is not trusted; the wrapper must clamp explicitly. Proven here by
    forcing the underlying sklearn model to hand back out-of-range values
    (patched at the class level, so this holds regardless of internal
    attribute names) and checking the wrapper still normalizes to [0, 1]."""
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy()
    check = IsotonicCrossCheck()
    check.fit(x, y)

    bad_output = np.array([-0.3, 1.7, float("nan"), 2.0, -5.0])
    with patch("sklearn.isotonic.IsotonicRegression.predict", return_value=bad_output):
        preds = check.predict([0.0, 1.0, 2.0, 3.0, 4.0])

    assert len(preds) == len(bad_output)
    for p in preds:
        assert isinstance(p, float)
        assert not math.isnan(p)
        assert 0.0 <= p <= 1.0, f"unclamped value leaked: {p}"


def test_isotonic_monotonicity_property():
    """The defining property of isotonic regression: predictions are
    non-decreasing in x for a monotone-increasing-in-expectation dataset."""
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _monotone_xy(n=60)
    check = IsotonicCrossCheck()
    check.fit(x, y)

    probe_x = [float(i) for i in range(-5, 65)]
    preds = check.predict(probe_x)
    for a, b in zip(preds, preds[1:]):
        assert b >= a - 1e-12, f"predictions must be non-decreasing: {a} then {b}"


# ---------------------------------------------------------------- PlattCrossCheck


def test_platt_predict_proba_in_range_basic():
    from mlops.ext.calibration_sk import PlattCrossCheck

    x, y = _mixed_xy()
    check = PlattCrossCheck()
    check.fit(x, y)
    preds = check.predict_proba_positive(x)
    assert isinstance(preds, list)
    assert len(preds) == len(x)
    for p in preds:
        assert 0.0 <= p <= 1.0


def test_platt_predict_proba_out_of_range_extreme_still_bounded():
    from mlops.ext.calibration_sk import PlattCrossCheck

    x, y = _mixed_xy()
    check = PlattCrossCheck()
    check.fit(x, y)
    far = [-1e9, -1e6, 1e6, 1e9]
    preds = check.predict_proba_positive(far)
    assert len(preds) == len(far)
    for p in preds:
        assert 0.0 <= p <= 1.0
        assert not math.isnan(p)


def test_platt_predict_proba_clamps_even_if_underlying_model_misbehaves():
    """Same belt-and-braces requirement as isotonic: force LogisticRegression's
    predict_proba to hand back out-of-range values and confirm the wrapper
    still normalizes to [0, 1]."""
    from mlops.ext.calibration_sk import PlattCrossCheck

    x, y = _mixed_xy()
    check = PlattCrossCheck()
    check.fit(x, y)

    bad_output = np.array(
        [[0.5, -0.3], [0.5, 1.7], [0.5, float("nan")], [0.5, 2.0], [0.5, -5.0]]
    )
    with patch("sklearn.linear_model.LogisticRegression.predict_proba", return_value=bad_output):
        preds = check.predict_proba_positive([0.0, 1.0, 2.0, 3.0, 4.0])

    assert len(preds) == len(bad_output)
    for p in preds:
        assert isinstance(p, float)
        assert not math.isnan(p)
        assert 0.0 <= p <= 1.0, f"unclamped value leaked: {p}"


def test_platt_predict_returns_same_length_as_input():
    from mlops.ext.calibration_sk import PlattCrossCheck

    x, y = _mixed_xy(n=30)
    check = PlattCrossCheck()
    check.fit(x, y)
    subset = x[:5]
    preds = check.predict_proba_positive(subset)
    assert len(preds) == 5


# ---------------------------------------------------------------- compare_to_calibrator


def test_compare_to_calibrator_returns_exactly_three_keys():
    from mlops.ext.calibration_sk import compare_to_calibrator

    x, y_binary = _mixed_xy()
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"isotonic", "platt", "existing_agrees_within"}


def test_compare_to_calibrator_isotonic_matches_direct_call():
    from mlops.ext.calibration_sk import IsotonicCrossCheck, compare_to_calibrator

    x, y_binary = _mixed_xy()
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)

    direct = IsotonicCrossCheck()
    direct.fit(x, [float(v) for v in y_binary])
    expected = direct.predict(x)

    assert isinstance(result["isotonic"], list)
    assert len(result["isotonic"]) == len(x)
    for got, exp in zip(result["isotonic"], expected):
        assert got == pytest.approx(exp, abs=1e-9)


def test_compare_to_calibrator_platt_matches_direct_call():
    from mlops.ext.calibration_sk import PlattCrossCheck, compare_to_calibrator

    x, y_binary = _mixed_xy()
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)

    direct = PlattCrossCheck()
    direct.fit(x, y_binary)
    expected = direct.predict_proba_positive(x)

    assert isinstance(result["platt"], list)
    assert len(result["platt"]) == len(x)
    for got, exp in zip(result["platt"], expected):
        assert got == pytest.approx(exp, abs=1e-9)


def test_compare_to_calibrator_existing_agrees_within_nonnegative_float():
    from mlops.ext.calibration_sk import compare_to_calibrator

    x, y_binary = _mixed_xy()
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)
    val = result["existing_agrees_within"]
    assert isinstance(val, float)
    assert val >= 0.0
    assert not math.isnan(val)


def test_compare_to_calibrator_all_values_bounded_0_1():
    from mlops.ext.calibration_sk import compare_to_calibrator

    x, y_binary = _mixed_xy(n=40)
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)
    for p in result["isotonic"]:
        assert 0.0 <= p <= 1.0
    for p in result["platt"]:
        assert 0.0 <= p <= 1.0


def test_compare_to_calibrator_result_is_json_serializable():
    from mlops.ext.calibration_sk import compare_to_calibrator

    x, y_binary = _mixed_xy()
    result = compare_to_calibrator(_existing_result_like_evaluate(), x, y_binary)
    dumped = json.dumps(result)  # must not raise
    assert isinstance(dumped, str)


# ---------------------------------------------------------------- import hygiene (lazy sklearn)


def test_lazy_import_sklearn_not_loaded_until_called():
    """Run in a clean subprocess: importing the module must not pull in
    sklearn, but calling one of its functions must."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {EXEC_DIR!r})\n"
        "from mlops.ext import calibration_sk\n"
        "assert 'sklearn' not in sys.modules, 'sklearn imported eagerly at module import time'\n"
        "check = calibration_sk.IsotonicCrossCheck()\n"
        "check.fit([float(i) for i in range(10)], [0.0]*5 + [1.0]*5)\n"
        "assert 'sklearn' in sys.modules, 'sklearn was never imported despite being used'\n"
        "print('LAZY_IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "LAZY_IMPORT_OK" in proc.stdout


def test_lazy_import_platt_path_also_triggers_sklearn():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {EXEC_DIR!r})\n"
        "from mlops.ext import calibration_sk\n"
        "assert 'sklearn' not in sys.modules\n"
        "check = calibration_sk.PlattCrossCheck()\n"
        "check.fit([float(i) - 10.0 for i in range(20)], [0, 1] * 10)\n"
        "assert 'sklearn' in sys.modules\n"
        "print('LAZY_IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "LAZY_IMPORT_OK" in proc.stdout
