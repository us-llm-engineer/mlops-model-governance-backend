"""F1.1 slim suite: mlops.stats_scipy (mann_whitney_u_scipy, bootstrap_ci,
wilson_interval, slo_breach_pvalue, ks_crosscheck).

Frozen BEFORE mlops.stats_scipy exists (the API contract
section 1). Hand-computed / scipy-cross-checked oracles only; no wall-clock
budgets, no sleeps.
"""
import math
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from scipy import stats as _scipy_stats  # noqa: E402

from mlops.stats_scipy import (  # noqa: E402
    bootstrap_ci,
    ks_crosscheck,
    mann_whitney_u_scipy,
    slo_breach_pvalue,
    wilson_interval,
)

BAD = (ValueError, TypeError)


# --------------------------------------------------------------- mann-whitney

def test_no_ties_matches_hand_oracle_and_scipy_to_1e12():
    # sample_a = [1, 2], sample_b = [3, 4]: fully separated, no ties.
    # Hand oracle (same formula as the old mlops.metrics implementation):
    n1, n2 = 2, 2
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (0.0 - mean_u) / std_u
    expected_p = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    u, p = mann_whitney_u_scipy([1.0, 2.0], [3.0, 4.0])
    assert u == 0.0
    assert p == pytest.approx(expected_p, abs=1e-12)

    scipy_result = _scipy_stats.mannwhitneyu(
        [1.0, 2.0], [3.0, 4.0], alternative="less", method="asymptotic", use_continuity=False
    )
    assert u == pytest.approx(float(scipy_result.statistic), abs=1e-12)
    assert p == pytest.approx(float(scipy_result.pvalue), abs=1e-12)
    assert type(u) is float and type(p) is float


def test_tie_corrected_variance_matches_scipy_asymptotic():
    a = [1.0, 2.0, 2.0, 3.0]
    b = [2.0, 3.0, 4.0, 5.0]
    u, p = mann_whitney_u_scipy(a, b)

    scipy_result = _scipy_stats.mannwhitneyu(
        a, b, alternative="less", method="asymptotic", use_continuity=False
    )
    assert u == pytest.approx(float(scipy_result.statistic), abs=1e-12)
    assert p == pytest.approx(float(scipy_result.pvalue), abs=1e-12)


def test_all_tied_returns_u_and_one_as_plain_floats():
    u, p = mann_whitney_u_scipy([5.0, 5.0], [5.0, 5.0])
    assert u == 2.0  # n1 * n2 / 2
    assert p == 1.0
    assert type(u) is float and type(p) is float


def test_mann_whitney_n_lt_2_raises():
    with pytest.raises(BAD):
        mann_whitney_u_scipy([1.0], [1.0, 2.0])


@pytest.mark.parametrize(
    "bad_sample",
    [
        [True, 2.0],
        ["a", 2.0],
        [None, 2.0],
        [float("nan"), 2.0],
        [float("inf"), 2.0],
        [[1.0], 2.0],
    ],
    ids=["bool", "str", "none", "nan", "inf", "nested_list"],
)
def test_mann_whitney_bad_input_values_raise(bad_sample):
    with pytest.raises(BAD):
        mann_whitney_u_scipy(bad_sample, [1.0, 2.0])


# ---------------------------------------------------------------- bootstrap

VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]


def test_bootstrap_ci_deterministic_same_seed():
    r1 = bootstrap_ci(VALUES, statistic="mean", seed=7, n_resamples=500)
    r2 = bootstrap_ci(VALUES, statistic="mean", seed=7, n_resamples=500)
    assert r1 == r2


def test_bootstrap_ci_different_seed_can_differ():
    r1 = bootstrap_ci(VALUES, statistic="mean", seed=0, n_resamples=2000)
    r2 = bootstrap_ci(VALUES, statistic="mean", seed=1, n_resamples=2000)
    assert r1 != r2


def test_bootstrap_ci_bounds_ordered_for_mean():
    low, high = bootstrap_ci(VALUES, statistic="mean", seed=0, n_resamples=500)
    assert low <= high
    assert isinstance(low, float) and isinstance(high, float)


def test_bootstrap_ci_statistic_median_bounds_ordered():
    low, high = bootstrap_ci(VALUES, statistic="median", seed=0, n_resamples=500)
    assert low <= high


def test_bootstrap_ci_n_lt_2_raises():
    with pytest.raises(BAD):
        bootstrap_ci([1.0], seed=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"statistic": "bogus"},
        {"confidence": True},
        {"confidence": 1.5},
        {"n_resamples": True},
        {"n_resamples": 50},
        {"seed": True},
    ],
    ids=[
        "bad_statistic",
        "confidence_bool",
        "confidence_out_of_range",
        "n_resamples_bool",
        "n_resamples_out_of_range",
        "seed_bool",
    ],
)
def test_bootstrap_ci_invalid_params_raise(kwargs):
    base = {"statistic": "mean", "confidence": 0.95, "n_resamples": 500, "seed": 0}
    base.update(kwargs)
    with pytest.raises(BAD):
        bootstrap_ci(VALUES, **base)


# ------------------------------------------------------------- wilson_interval

def test_wilson_interval_hand_value():
    low, high = wilson_interval(3, 20)
    assert low == pytest.approx(0.0524, abs=5e-4)
    assert high == pytest.approx(0.3604, abs=5e-4)


@pytest.mark.parametrize(
    "successes,trials",
    [
        (25, 20),  # k > n
        (-1, 20),  # k < 0
        (0, 0),  # n = 0
        (3.5, 20),  # float k
        (True, 20),  # bool accepted by scipy but must be rejected
    ],
    ids=["k_gt_n", "k_lt_0", "n_is_0", "float_k", "bool_k"],
)
def test_wilson_interval_traps_raise(successes, trials):
    with pytest.raises(BAD):
        wilson_interval(successes, trials)


# ---------------------------------------------------------- slo_breach_pvalue

def test_slo_breach_pvalue_hand_value_and_monotonicity():
    p_9 = slo_breach_pvalue(9, 100, 0.05)
    assert p_9 == pytest.approx(0.0631, abs=5e-4)

    p_20 = slo_breach_pvalue(20, 100, 0.05)
    assert p_20 < p_9  # more bad observations -> smaller (more significant) p-value


# ------------------------------------------------------------- ks_crosscheck

def test_ks_crosscheck_identical_disjoint_keys_and_types():
    identical = ks_crosscheck([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert identical["statistic"] == 0.0
    assert identical["pvalue"] == 1.0

    disjoint = ks_crosscheck([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert disjoint["statistic"] == 1.0

    assert set(disjoint.keys()) == {"statistic", "pvalue", "n_a", "n_b"}
    assert type(disjoint["statistic"]) is float
    assert type(disjoint["pvalue"]) is float
    assert type(disjoint["n_a"]) is int and disjoint["n_a"] == 3
    assert type(disjoint["n_b"]) is int and disjoint["n_b"] == 3


# ----------------------------------------------------------------- packaging

def test_import_mlops_avoids_heavy_deps_and_stats_scipy_imports():
    exec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
    script = (
        "import sys\n"
        "import mlops\n"
        "banned = {'opentelemetry', 'fastapi', 'prometheus_client'}\n"
        "loaded = banned & {m.split('.')[0] for m in sys.modules}\n"
        "assert not loaded, loaded\n"
        "import mlops.stats_scipy\n"
        "assert hasattr(mlops.stats_scipy, 'mann_whitney_u_scipy')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=exec_dir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
