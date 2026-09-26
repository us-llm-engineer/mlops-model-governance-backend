"""Statistical functions using scipy: Mann-Whitney U, bootstrap CI, Wilson interval, SLO breach p-value, KS crosscheck."""
import math
import numpy as np
from scipy import stats as scipy_stats


def _validate_finite_reals(sample, name="sample"):
    """Validate that sample is a list/tuple/ndarray of real finite numbers."""
    # First, check raw input for bools, strings, None before any conversion
    if isinstance(sample, (list, tuple)):
        for val in sample:
            if isinstance(val, (bool, np.bool_)):
                raise ValueError(f"{name} contains bool")
            if isinstance(val, str):
                raise ValueError(f"{name} contains str")
            if val is None:
                raise ValueError(f"{name} contains None")
            if isinstance(val, list):
                raise ValueError(f"{name} contains non-numeric value")
        arr = np.asarray(sample, dtype=float)
    elif isinstance(sample, np.ndarray):
        if sample.dtype == bool or sample.dtype == np.bool_:
            raise ValueError(f"{name} contains bool")
        arr = sample.astype(float)
    else:
        raise ValueError(f"{name} must be a list, tuple, or ndarray")

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")

    if len(arr) == 0:
        raise ValueError(f"{name} cannot be empty")

    # Check converted values for NaN and inf
    for val in arr.flat:
        if math.isnan(float(val)):
            raise ValueError(f"{name} contains NaN")
        if math.isinf(float(val)):
            raise ValueError(f"{name} contains inf")

    return arr.astype(float)


def mann_whitney_u_scipy(sample_a, sample_b):
    """One-sided Mann-Whitney U test: H1 'a stochastically LESS than b'."""
    a = _validate_finite_reals(sample_a, "sample_a")
    b = _validate_finite_reals(sample_b, "sample_b")

    if len(a) < 2:
        raise ValueError("requires at least 2 observations per group")
    if len(b) < 2:
        raise ValueError("requires at least 2 observations per group")

    result = scipy_stats.mannwhitneyu(
        a, b, alternative="less", method="asymptotic", use_continuity=False
    )

    u = float(result.statistic)
    p = float(result.pvalue)

    if math.isnan(p):
        p = 1.0

    return u, p


def bootstrap_ci(values, *, statistic="mean", confidence=0.95, n_resamples=2000, seed=0):
    """Bootstrap confidence interval for mean or median."""
    if not isinstance(statistic, str) or statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    if not isinstance(confidence, float) or isinstance(confidence, bool):
        raise ValueError("confidence must be a float")
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be between 0 and 1")

    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool):
        raise ValueError("n_resamples must be an int")
    if n_resamples < 100 or n_resamples > 20000:
        raise ValueError("n_resamples must be between 100 and 20000")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")

    arr = _validate_finite_reals(values, "values")

    if len(arr) < 2:
        raise ValueError("requires at least 2 observations")

    statistic_func = np.mean if statistic == "mean" else np.median

    result = scipy_stats.bootstrap(
        (arr,),
        statistic_func,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )

    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)

    return low, high


def wilson_interval(successes, trials, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if not isinstance(successes, int) or isinstance(successes, bool):
        raise ValueError("successes must be an int (not bool)")
    if not isinstance(trials, int) or isinstance(trials, bool):
        raise ValueError("trials must be an int (not bool)")

    if trials < 1:
        raise ValueError("trials must be >= 1")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")

    result = scipy_stats.binomtest(successes, trials).proportion_ci(
        confidence_level=confidence, method="wilson"
    )

    low = float(result.low)
    high = float(result.high)

    return low, high


def slo_breach_pvalue(bad, total, target_error_rate):
    """P-value for SLO breach: evidence that error rate exceeds target."""
    if not isinstance(bad, int) or isinstance(bad, bool):
        raise ValueError("bad must be an int (not bool)")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("total must be an int (not bool)")

    if not isinstance(target_error_rate, float) or isinstance(target_error_rate, bool):
        raise ValueError("target_error_rate must be a float (not bool)")
    if target_error_rate <= 0 or target_error_rate >= 1:
        raise ValueError("target_error_rate must be between 0 and 1")

    if total < 1:
        raise ValueError("total must be >= 1")
    if bad < 0 or bad > total:
        raise ValueError("bad must be between 0 and total")

    result = scipy_stats.binomtest(bad, total, target_error_rate, alternative="greater")

    return float(result.pvalue)


def ks_crosscheck(a, b):
    """Kolmogorov-Smirnov test for two samples."""
    arr_a = _validate_finite_reals(a, "a")
    arr_b = _validate_finite_reals(b, "b")

    if len(arr_a) < 2:
        raise ValueError("a requires at least 2 observations")
    if len(arr_b) < 2:
        raise ValueError("b requires at least 2 observations")

    result = scipy_stats.ks_2samp(arr_a, arr_b)

    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_a": int(len(arr_a)),
        "n_b": int(len(arr_b)),
    }
