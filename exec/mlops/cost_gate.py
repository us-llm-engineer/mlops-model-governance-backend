"""
Cost governance gates (Claims X3.2): budget admission, reconciliation and
cross-source double-count detection.

Maps the X3 API contract's ``mlops/cost_gate.py`` surface onto the reused
``GateDecision``/``now_iso`` shape from :mod:`mlops.gates` and the ledger
primitives from :mod:`mlops.cost_ledger`:

* ``budget_gate`` admits only when the unallocated spend fraction ``U`` is at
  or below the budget ``P`` (inclusive boundary).
* ``reconcile`` compares the internal ledger total against an external billed
  amount within an absolute tolerance.
* ``detect_double_counted`` flags cost rows with an identical
  ``(resource, window_start, window_end)`` fingerprint reported by two
  different sources.
* ``exclude_source`` drops every row from one source, so removing the
  duplicate contributor reduces the total by exactly the overlap.

All money values are :class:`decimal.Decimal`. Standard library only.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Tuple

from .gates import GateDecision, now_iso
from .cost_ledger import CostRow, unallocated_fraction, total_spend

__all__ = ["budget_gate", "reconcile", "detect_double_counted", "exclude_source"]


def budget_gate(
    rows: List[CostRow],
    budget: Decimal,
    fallbacks: List[str],
    audit=None,
) -> GateDecision:
    """Admit ``rows`` when their unallocated fraction is within ``budget``.

    The comparison is inclusive: ``U == budget`` passes. When ``audit`` is
    given, exactly one entry is appended recording the allow/deny decision.
    """
    u = unallocated_fraction(rows, fallbacks)
    passed = u <= budget
    decision = GateDecision(
        rule_name="cost_budget",
        passed=passed,
        reason=f"unallocated fraction {u} {'<=' if passed else '>'} budget {budget}",
        predicates=[
            {"predicate": "unallocated_fraction", "value": str(u)},
            {"predicate": "budget", "value": str(budget)},
            {"predicate": "u<=budget", "result": passed},
        ],
        timestamp=now_iso(),
    )
    if audit is not None:
        audit.append(
            "system",
            "cost.budget_gate",
            "ledger",
            "allow" if passed else "deny",
        )
    return decision


def reconcile(
    ledger_total: Decimal,
    billed_amount: Decimal,
    tolerance: Decimal,
    audit=None,
) -> dict:
    """Compare the ledger total to a billed amount within an absolute tolerance.

    A delta whose absolute value equals the tolerance is *not* flagged; only
    ``abs(delta) > tolerance`` flags. Exactly one audit entry is appended when
    ``audit`` is supplied.
    """
    delta = ledger_total - billed_amount
    flagged = abs(delta) > tolerance
    if audit is not None:
        audit.append(
            "system",
            "cost.reconcile",
            "ledger",
            "deny" if flagged else "allow",
        )
    return {"delta": delta, "flagged": flagged}


def detect_double_counted(
    rows: List[CostRow],
) -> List[Tuple[CostRow, CostRow]]:
    """Return every cross-source row pair sharing a fingerprint.

    Two rows match when they come from different sources but agree on
    ``(resource, window_start, window_end)``. Pairs preserve input order
    (``i < j``), and same-source duplicates are ignored entirely.
    """
    pairs: List[Tuple[CostRow, CostRow]] = []
    for i, a in enumerate(rows):
        fingerprint = (a.resource, a.window_start, a.window_end)
        for b in rows[i + 1 :]:
            if a.source == b.source:
                continue
            if fingerprint == (b.resource, b.window_start, b.window_end):
                pairs.append((a, b))
    return pairs


def exclude_source(rows: List[CostRow], source: str) -> List[CostRow]:
    """Return the rows whose source differs from ``source``, order preserved."""
    return [row for row in rows if row.source != source]
