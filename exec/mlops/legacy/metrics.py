"""
Canary metrics collection and statistical judgement (Claims C21, C22).

Maps to requirement R2.3 (Kubernetes Serving & Traffic Management) and is
lifted from the frozen reference implementations in ``tests/R2_3_S1.py``
(C21: latency metrics + one-sided Mann-Whitney U test) and
``tests/R2_3_S2.py`` (C22: per-request timestamped metrics).

UNIFICATION (deliberate)
------------------------
The two source suites define two incompatible ``MetricsCollector`` shapes:

* R2_3_S1: ``record(track, latency_ms)`` stores a bare float, no timestamp.
* R2_3_S2: ``record(track, latency_ms, ts)`` stores a timestamped
  ``RequestMetric`` and exposes ``timestamps()``.

Both are unified here into a single collector whose ``record`` takes an
optional ``ts`` (defaulting to ``time.time()``) and always stores a
``RequestMetric``. ``latencies()`` returns the raw floats exactly as the S1
suite expects, while ``timestamps()`` serves the S2 suite. This preserves the
observable behaviour of both suites without a compatibility shim.

Standard library only (math, time, dataclasses).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RequestMetric:
    """A single recorded request latency with its observation timestamp."""

    latency_ms: float
    timestamp: float


@dataclass
class CanaryMetrics:
    """Paired latency series for the canary and stable traffic tracks."""

    canary_series: list = field(default_factory=list)
    stable_series: list = field(default_factory=list)


def _rank(values):
    """Return tie-corrected average ranks for ``values`` (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u(sample_a, sample_b):
    """One-sided Mann-Whitney U test.

    Alternative hypothesis: ``sample_a`` is stochastically LESS than
    ``sample_b`` (e.g. canary latency lower than stable latency). Returns
    ``(u_statistic, p_value)``; a small p_value means we can reject
    ``H0: a >= b``, i.e. conclude ``a < b``. Uses scipy for accurate p-value.
    """
    from mlops.stats_scipy import mann_whitney_u_scipy
    return mann_whitney_u_scipy(sample_a, sample_b)


def _mann_whitney_u_reference(sample_a, sample_b):
    """One-sided Mann-Whitney U test.

    Alternative hypothesis: ``sample_a`` is stochastically LESS than
    ``sample_b`` (e.g. canary latency lower than stable latency). Returns
    ``(u_statistic, p_value)``; a small p_value means we can reject
    ``H0: a >= b``, i.e. conclude ``a < b``. Uses a normal approximation
    with tie-corrected ranks (no scipy).
    """
    n1, n2 = len(sample_a), len(sample_b)
    if n1 < 2 or n2 < 2:
        raise ValueError(
            "Mann-Whitney U test requires at least 2 observations per group"
        )
    combined = list(sample_a) + list(sample_b)
    ranks = _rank(combined)
    rank_sum_a = sum(ranks[:n1])
    u1 = rank_sum_a - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_u == 0:
        return u1, 1.0
    z = (u1 - mean_u) / std_u
    p_value = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return u1, p_value


class MetricsCollector:
    """Collect per-request latency metrics for the canary and stable tracks.

    Unified across R2_3_S1 (float series) and R2_3_S2 (timestamped series):
    every record is stored as a :class:`RequestMetric`, and ``ts`` defaults to
    the current wall-clock time when omitted.
    """

    def __init__(self):
        self._series: Dict[str, List[RequestMetric]] = {"canary": [], "stable": []}

    def record(self, track: str, latency_ms: float, ts: float = None) -> None:
        """Append a latency observation for ``track`` (timestamp optional)."""
        if ts is None:
            ts = time.time()
        self._series[track].append(
            RequestMetric(latency_ms=latency_ms, timestamp=ts)
        )

    def latencies(self, track: str) -> List[float]:
        """Return the recorded latencies for ``track`` in observation order."""
        return [m.latency_ms for m in self._series[track]]

    def timestamps(self, track: str) -> List[float]:
        """Return the recorded timestamps for ``track`` in observation order."""
        return [m.timestamp for m in self._series[track]]


class StatisticalJudge:
    """Promote/rollback decision from a one-sided Mann-Whitney U test."""

    ALPHA = 0.05

    def judge(self, canary_samples, stable_samples) -> dict:
        """Return U statistic, p-value and the promotion decision."""
        u, p_value = mann_whitney_u(canary_samples, stable_samples)
        return {
            "u_statistic": u,
            "p_value": p_value,
            "promote": p_value < self.ALPHA,
        }


__all__ = [
    "RequestMetric",
    "CanaryMetrics",
    "MetricsCollector",
    "StatisticalJudge",
    "mann_whitney_u",
]
