"""Online/offline feature parity checking and promotion gating (Project-2 X1.3).

Implements the ``mlops/feature_parity.py`` surface of the X1 API contract
(plan A):

* :class:`ParityReport` -- the result of one parity comparison.
* :func:`check_parity` -- compare online vs offline feature values against a
  tolerance with a minimum-pairs floor.
* :class:`ParityGate` -- fail-closed promotion gate over a mapping of
  feature-version id -> :class:`ParityReport`.

Standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

try:  # Reuse the project's unified decision record when available.
    from mlops.gates import GateDecision
except Exception:  # pragma: no cover - standalone fallback

    @dataclass
    class GateDecision:  # type: ignore[no-redef]
        """Minimal GateDecision-like fallback (fields mirror mlops.gates)."""

        rule_name: str
        passed: bool
        reason: str
        predicates: list = field(default_factory=list)
        timestamp: str = ""


__all__ = ["ParityReport", "check_parity", "ParityGate"]


@dataclass
class ParityReport:
    """Outcome of a parity comparison between online and offline values."""

    status: str = ""
    pairs: int = 0
    mismatches: List[str] = field(default_factory=list)
    max_abs_diff: float = 0.0


def check_parity(
    pairs: Sequence[Tuple[str, float, float]],
    tolerance: float,
    min_pairs: int,
) -> ParityReport:
    """Compare ``(entity, online_value, offline_value)`` pairs.

    Fewer than ``min_pairs`` pairs yields ``insufficient_data`` (never
    ``pass``); otherwise the status is ``fail`` if any absolute difference
    exceeds ``tolerance`` (equality passes), else ``pass``. ``max_abs_diff``
    is the largest absolute difference over all supplied pairs.

    Claim C76: Fails closed (status="fail") if any value is NaN, infinite, or
    non-numeric. This is never reported as ``pass``.

    Raises ``ValueError`` if ``tolerance`` is not a finite non-negative real
    number, or if ``min_pairs`` < 1 (bool rejected).
    """
    # Validate tolerance: must be numeric, finite, non-negative, and not bool
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a finite non-negative real number")
    if math.isnan(tolerance) or math.isinf(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative real number")

    # Validate min_pairs: must be >= 1 (empty comparison must never pass)
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
        raise ValueError("min_pairs must be >= 1")

    n = len(pairs)
    if n < min_pairs:
        return ParityReport(
            status="insufficient_data", pairs=n, mismatches=[], max_abs_diff=0.0
        )

    mismatches: List[str] = []
    max_abs_diff = 0.0
    for entity, online, offline in pairs:
        # Fail closed: only real int/float (not bool, str, None) are numeric
        # bool is a subclass of int, so explicitly exclude it
        if (isinstance(online, bool) or isinstance(offline, bool) or
            online is None or offline is None or
            isinstance(online, str) or isinstance(offline, str)):
            mismatches.append(entity)
            continue

        # Must be int or float
        if not isinstance(online, (int, float)) or not isinstance(offline, (int, float)):
            mismatches.append(entity)
            continue

        # Check for NaN or inf (C76: fail closed on these)
        try:
            online_float = float(online)
            offline_float = float(offline)
            if math.isnan(online_float) or math.isnan(offline_float):
                mismatches.append(entity)
                continue
            if math.isinf(online_float) or math.isinf(offline_float):
                mismatches.append(entity)
                continue

            diff = abs(online_float - offline_float)
            if diff > tolerance:
                mismatches.append(entity)
            if diff > max_abs_diff:
                max_abs_diff = diff
        except (TypeError, ValueError):
            # Shouldn't reach here due to type checks, but fail closed anyway
            mismatches.append(entity)

    return ParityReport(
        status="fail" if mismatches else "pass",
        pairs=n,
        mismatches=mismatches,
        max_abs_diff=max_abs_diff,
    )


class ParityGate:
    """Fail-closed promotion gate over per-version parity reports."""

    RULE_NAME = "feature_parity_v1"

    def __init__(self, reports: Dict[str, ParityReport]) -> None:
        self.reports = dict(reports)

    def allow_promotion(self, version_ids: List[str]) -> GateDecision:
        """Return a decision allowing promotion only if every version passes.

        An empty ``version_ids`` list is blocked (nothing to vouch for).
        Missing, failing, or insufficient reports block and name the first
        offending version together with its status (or ``missing``).
        Reports for versions not listed are ignored.
        """
        if not version_ids:
            return GateDecision(
                rule_name=self.RULE_NAME,
                passed=False,
                reason="No feature versions supplied to vouch for",
                predicates=[],
                timestamp=self._timestamp(),
            )

        for version_id in version_ids:
            report = self.reports.get(version_id)
            if report is None or report.status != "pass":
                status = report.status if report is not None else "missing"
                return GateDecision(
                    rule_name=self.RULE_NAME,
                    passed=False,
                    reason=f"feature parity {status} for {version_id}",
                    predicates=[],
                    timestamp=self._timestamp(),
                )

        return GateDecision(
            rule_name=self.RULE_NAME,
            passed=True,
            reason="All feature versions passed parity",
            predicates=[],
            timestamp=self._timestamp(),
        )

    @staticmethod
    def _timestamp() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
