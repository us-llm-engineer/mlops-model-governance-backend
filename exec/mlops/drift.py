"""
S3.2 Drift monitoring: PSI, KS, KL/JS, entropy, kappa, DriftMonitor.

Frozen test: tests/test_S3_2_drift.py
"""

import dataclasses
import math
from collections import deque
from typing import List

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from mlops.kernel import Conflict

__all__ = [
    "psi",
    "psi_from_samples",
    "ks_statistic",
    "kl_divergence",
    "js_divergence",
    "predictive_entropy",
    "cohens_kappa",
    "classify_drift",
    "DriftReport",
    "DriftMonitor",
    "drift_context",
]


# Constants
PSI_WARN = 0.1
PSI_ALERT = 0.25
KS_ALERT = 0.1


def _is_valid_number(x):
    """Check if x is a real finite number (not bool, NaN, inf, str, None)."""
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def _validate_distribution(probs, name="probs"):
    """Validate that probs is a valid probability distribution.

    Raises ValueError if invalid.
    """
    if not isinstance(probs, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")

    if len(probs) < 2:
        raise ValueError(f"{name} must have at least 2 elements")

    for i, p in enumerate(probs):
        if not _is_valid_number(p):
            raise ValueError(f"{name}[{i}] is not a valid finite number")
        if p < 0:
            raise ValueError(f"{name}[{i}] is negative")

    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} sums to {total}, not 1.0")


def psi(expected: List[float], actual: List[float], eps: float = 1e-6) -> float:
    """Population Stability Index (PSI).

    Both are bin proportions of equal length >= 2, non-negative finite,
    summing to 1 within 1e-6. PSI = sum (a-e) ln(a/e) with eps floor
    for zero bins.

    Args:
        expected: Expected distribution proportions
        actual: Actual distribution proportions
        eps: Floor for zero bins to avoid log(0)

    Returns:
        PSI value (>= 0)

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_distribution(expected, "expected")
    _validate_distribution(actual, "actual")

    if len(expected) != len(actual):
        raise ValueError("expected and actual must have equal length")

    psi_val = 0.0
    for e, a in zip(expected, actual):
        e_floored = max(e, eps)
        a_floored = max(a, eps)
        psi_val += (a_floored - e_floored) * math.log(a_floored / e_floored)

    return psi_val


def psi_from_samples(expected_samples: List[float], actual_samples: List[float],
                     bins: int = 10) -> float:
    """PSI computed from raw samples using quantile-based binning.

    Bins are determined by quantiles of the expected samples.

    Args:
        expected_samples: Reference sample values (max 1,000,000)
        actual_samples: Current sample values (max 1,000,000)
        bins: Number of bins (must be 2 <= bins <= 1000)

    Returns:
        PSI value

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate bins FIRST (before allocating)
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 2 or bins > 1000:
        raise ValueError("bins must be in range [2, 1000]")

    # Validate samples BEFORE allocating numpy arrays
    for name, samples in [("expected_samples", expected_samples),
                          ("actual_samples", actual_samples)]:
        if not isinstance(samples, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple")
        if len(samples) < 30:
            raise ValueError(f"{name} must have >= 30 samples")
        if len(samples) > 1000000:
            raise ValueError(f"{name} must have <= 1000000 samples")
        for i, v in enumerate(samples):
            if not _is_valid_number(v):
                raise ValueError(f"{name}[{i}] is not a valid finite number")

    # Create bin edges from expected samples quantiles
    expected_arr = np.array(expected_samples, dtype=float)
    actual_arr = np.array(actual_samples, dtype=float)

    # Get unique quantile boundaries from expected samples
    # Using percentiles to get bin edges
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected_arr, quantiles)

    # Remove duplicates but keep first and last
    bin_edges = np.unique(bin_edges)

    # Histogram both samples
    exp_counts, _ = np.histogram(expected_arr, bins=bin_edges)
    act_counts, _ = np.histogram(actual_arr, bins=bin_edges)

    # Convert to proportions
    exp_proportions = exp_counts / np.sum(exp_counts)
    act_proportions = act_counts / np.sum(act_counts)

    # Compute PSI with eps floor
    eps = 1e-6
    psi_val = 0.0
    for e, a in zip(exp_proportions, act_proportions):
        e_floored = max(e, eps)
        a_floored = max(a, eps)
        psi_val += (a_floored - e_floored) * math.log(a_floored / e_floored)

    return float(psi_val)


def ks_statistic(a: List[float], b: List[float]) -> float:
    """Kolmogorov-Smirnov statistic: max |F_a - F_b|.

    Two-sample test using exact empirical CDFs.

    Args:
        a: First sample (>= 2 values)
        b: Second sample (>= 2 values)

    Returns:
        KS statistic (0 to 1)

    Raises:
        ValueError: If inputs are invalid
    """
    for name, sample in [("a", a), ("b", b)]:
        if not isinstance(sample, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple")
        if len(sample) < 2:
            raise ValueError(f"{name} must have >= 2 samples")
        for i, v in enumerate(sample):
            if not _is_valid_number(v):
                raise ValueError(f"{name}[{i}] is not a valid finite number")

    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    result = stats.ks_2samp(a_arr, b_arr)
    return float(result.statistic)


def kl_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    """KL divergence D(p||q) = sum p_i ln(p_i / q_i).

    Both are probability distributions.

    Args:
        p: First distribution (reference)
        q: Second distribution
        eps: Floor for zero bins to avoid infinity

    Returns:
        KL divergence (>= 0)

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")

    if len(p) != len(q):
        raise ValueError("p and q must have equal length")

    # Apply eps floor to avoid inf
    p_arr = np.array([max(pi, eps) for pi in p])
    q_arr = np.array([max(qi, eps) for qi in q])

    # Use scipy entropy with base e (natural log)
    kl_val = stats.entropy(p_arr, q_arr)
    return float(kl_val)


def js_divergence(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence: symmetric version of KL.

    JS is symmetric and bounded [0, ln(2)].

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        JS divergence (0 to ln(2))

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")

    if len(p) != len(q):
        raise ValueError("p and q must have equal length")

    # jensenshannon returns the DISTANCE (sqrt of divergence)
    # We need to square it to get the divergence
    p_arr = np.array(p)
    q_arr = np.array(q)

    distance = jensenshannon(p_arr, q_arr)
    divergence = distance ** 2

    return float(divergence)


def predictive_entropy(probs: List[float]) -> float:
    """Shannon entropy: -sum p_i ln(p_i).

    Measures uncertainty in a probability distribution.

    Args:
        probs: Probability distribution

    Returns:
        Entropy value (>= 0)

    Raises:
        ValueError: If input is invalid
    """
    _validate_distribution(probs, "probs")

    # Apply eps floor to avoid log(0)
    eps = 1e-12
    probs_arr = np.array([max(pi, eps) for pi in probs])

    # Use scipy entropy with no second argument for Shannon entropy
    entropy_val = stats.entropy(probs_arr)

    return float(entropy_val)


def cohens_kappa(labels_a: List[int], labels_b: List[int]) -> float:
    """Cohen's kappa: measure of inter-rater agreement.

    kappa = (p_o - p_e) / (1 - p_e)
    where p_o is observed agreement and p_e is expected agreement.

    Args:
        labels_a: First set of labels
        labels_b: Second set of labels

    Returns:
        Kappa value (-1 to 1, or 1.0 when pe=1)

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(labels_a, (list, tuple)):
        raise ValueError("labels_a must be a list or tuple")
    if not isinstance(labels_b, (list, tuple)):
        raise ValueError("labels_b must be a list or tuple")

    if len(labels_a) < 1:
        raise ValueError("labels_a must have >= 1 element")
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have equal length")

    # Compute observed agreement
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = matches / len(labels_a)

    # Compute expected agreement
    # Count occurrences of each label in both sequences
    unique_labels = set(labels_a) | set(labels_b)
    pe = 0.0
    for label in unique_labels:
        count_a = sum(1 for x in labels_a if x == label)
        count_b = sum(1 for x in labels_b if x == label)
        pe += (count_a / len(labels_a)) * (count_b / len(labels_b))

    # Compute kappa
    if pe >= 1.0:
        # Perfect expected agreement
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return float(kappa)


def classify_drift(psi_value: float, ks_value: float) -> str:
    """Classify drift level based on PSI and KS statistics.

    Levels: PSI_WARN = 0.1, PSI_ALERT = 0.25, KS_ALERT = 0.1
    Alert if psi > 0.25 or ks > 0.1; else warn if psi > 0.1; else ok.
    (Exact boundaries are NOT the higher level.)

    Args:
        psi_value: Population Stability Index
        ks_value: Kolmogorov-Smirnov statistic

    Returns:
        "ok", "warn", or "alert"

    Raises:
        ValueError: If inputs are invalid
    """
    if not _is_valid_number(psi_value):
        raise ValueError("psi_value must be a valid finite number")
    if not _is_valid_number(ks_value):
        raise ValueError("ks_value must be a valid finite number")

    if psi_value < 0:
        raise ValueError("psi_value must be non-negative")
    if ks_value < 0:
        raise ValueError("ks_value must be non-negative")

    # Alert if psi > PSI_ALERT or ks > KS_ALERT (strict >)
    if psi_value > PSI_ALERT or ks_value > KS_ALERT:
        return "alert"

    # Warn if psi > PSI_WARN (strict >)
    if psi_value > PSI_WARN:
        return "warn"

    return "ok"


@dataclasses.dataclass(frozen=True)
class DriftReport:
    """Frozen dataclass for drift monitoring report.

    The psi field is the raw PSI value. The psi_adjusted field applies
    bias correction for sampling variation: psi_adjusted = max(0, psi_raw - bias),
    where bias = (bins-1)*(1/n_ref + 1/n_win). The course threshold PSI 0.1/0.25
    assumes large samples; bias correction prevents false alarms at small n.

    The ks field is the raw KS statistic. The ks_pvalue field is the
    significance level from ks_2samp; only ks values with pvalue < 0.01
    are considered statistically significant. KS gating prevents false alarms.

    The level field is computed from psi_adjusted (bias-corrected) and
    ks_gated (p-value filtered) values.
    """
    level: str  # "ok", "warn", or "alert"
    psi: float
    ks: float
    n: int  # number of observations
    window_full: bool
    ks_pvalue: float = 1.0  # default: not significant
    psi_adjusted: float = 0.0  # default: bias-corrected PSI


class DriftMonitor:
    """Monitor data drift using a sliding window of observations.

    Compares incoming observations against a reference distribution.
    """

    def __init__(self, reference: List[float], window_size: int = 200, bins: int = 10):
        """Initialize drift monitor.

        Args:
            reference: Reference sample (>= 30 finite values)
            window_size: Max observations to keep (30..100000)
            bins: Number of histogram bins (>= 2)

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate reference
        if not isinstance(reference, (list, tuple)):
            raise ValueError("reference must be a list or tuple")
        if len(reference) < 30:
            raise ValueError("reference must have >= 30 samples")
        for i, v in enumerate(reference):
            if not _is_valid_number(v):
                raise ValueError(f"reference[{i}] is not a valid finite number")

        # Validate window_size
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError("window_size must be an integer")
        if window_size < 30 or window_size > 100000:
            raise ValueError("window_size must be in [30, 100000]")

        # Validate bins
        if not isinstance(bins, int) or isinstance(bins, bool):
            raise ValueError("bins must be an integer")
        if bins < 2 or bins > 1000:
            raise ValueError("bins must be in range [2, 1000]")

        self._reference = list(reference)
        self._window_size = window_size
        self._bins = bins
        self._observations = deque(maxlen=window_size)

    def observe(self, value: float) -> None:
        """Record a new observation.

        Args:
            value: Finite real number

        Raises:
            ValueError: If value is invalid
        """
        if not _is_valid_number(value):
            raise ValueError("observation must be a valid finite number")

        self._observations.append(value)

    def check(self) -> DriftReport:
        """Compute drift metrics against the reference.

        PSI bias correction: bias = (bins-1)*(1/n_ref + 1/n_window),
        psi_adjusted = max(0, psi_raw - bias). The course threshold PSI 0.1/0.25
        assumes large samples; bias correction prevents false alarms.

        KS significance gating: only KS with p-value < 0.01 is passed to
        classify_drift; otherwise 0.0 to avoid false alarms at small sample sizes.
        The course threshold KS > 0.1 also assumes large samples.

        Returns:
            DriftReport with level, psi (raw), psi_adjusted, ks (raw),
            ks_pvalue, n, window_full

        Raises:
            Conflict: If fewer than 30 observations have been recorded
        """
        if len(self._observations) < 30:
            raise Conflict(f"Need >= 30 observations, have {len(self._observations)}")

        obs_list = list(self._observations)

        # Compute PSI (raw)
        psi_raw = psi_from_samples(self._reference, obs_list, self._bins)

        # Compute PSI bias correction
        n_ref = len(self._reference)
        n_win = len(obs_list)
        bias = (self._bins - 1) * (1.0 / n_ref + 1.0 / n_win)
        psi_adjusted = max(0.0, psi_raw - bias)

        # Compute KS statistic and p-value
        ref_arr = np.array(self._reference, dtype=float)
        obs_arr = np.array(obs_list, dtype=float)
        res = stats.ks_2samp(ref_arr, obs_arr)
        ks_raw = res.statistic
        ks_pvalue = res.pvalue

        # Gate KS by significance: only pass ks to classify_drift if p < 0.01
        ks_gated = ks_raw if ks_pvalue < 0.01 else 0.0

        # Classify drift using bias-corrected PSI and gated KS
        level = classify_drift(psi_adjusted, ks_gated)

        # Check if window is full
        window_full = len(self._observations) == self._window_size

        return DriftReport(
            level=level,
            psi=psi_raw,
            ks=ks_raw,
            n=len(self._observations),
            window_full=window_full,
            ks_pvalue=ks_pvalue,
            psi_adjusted=psi_adjusted
        )


def drift_context(report: DriftReport) -> dict:
    """Convert DriftReport to context dict for policy decisions.

    Args:
        report: DriftReport from monitor.check()

    Returns:
        Dict with drift_level_code (0=ok, 1=warn, 2=alert),
        drift_psi, and drift_ks
    """
    level_code = {"ok": 0, "warn": 1, "alert": 2}[report.level]

    return {
        "drift_level_code": level_code,
        "drift_psi": report.psi,
        "drift_ks": report.ks,
    }
