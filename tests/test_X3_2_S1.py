"""X3.2-S1: C65-C66 budget-gate and reconciliation decision boundaries."""
import os
import sys
from decimal import Decimal


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.cost_gate import budget_gate, reconcile
from mlops.cost_ledger import CostRow


def _row(owner, amount, *, labels=None):
    return CostRow(
        source="billing-export",
        resource="job-1",
        owner=owner,
        amount=Decimal(amount),
        window_start=0,
        window_end=60,
        labels=labels or {},
    )


# X3.2-S1-C65-01: U below the budget is admitted.
def test_budget_gate_passes_when_unallocated_fraction_is_below_budget():
    decision = budget_gate(
        [_row("team-a", "8.00"), _row(None, "2.00")],
        budget=Decimal("0.21"),
        fallbacks=[],
    )

    assert decision.passed is True


# X3.2-S1-C65-02: the documented U == P boundary is inclusive.
def test_budget_gate_passes_at_inclusive_unallocated_budget_boundary():
    decision = budget_gate(
        [_row("team-a", "3.00"), _row(None, "1.00")],
        budget=Decimal("0.25"),
        fallbacks=[],
    )

    assert decision.passed is True


# X3.2-S1-C65-03: any amount over P is denied, even by one Decimal cent.
def test_budget_gate_fails_when_unallocated_fraction_is_just_above_budget():
    decision = budget_gate(
        [_row("team-a", "74.00"), _row(None, "26.00")],
        budget=Decimal("0.25"),
        fallbacks=[],
    )

    assert decision.passed is False


# X3.2-S1-C65-04: a fallback attribution removes that spend from U.
def test_budget_gate_treats_fallback_attributed_spend_as_allocated():
    decision = budget_gate(
        [_row(None, "10.00", labels={"team": "team-a"})],
        budget=Decimal("0"),
        fallbacks=["team"],
    )

    assert decision.passed is True


# X3.2-S1-C66-01: exact billed and ledger totals have a zero, unflagged delta.
def test_reconcile_exact_match_returns_zero_signed_delta_without_flag():
    result = reconcile(Decimal("12.50"), Decimal("12.50"), Decimal("0"))

    assert result["delta"] == Decimal("0.00")
    assert result["flagged"] is False


# X3.2-S1-C66-02: equality at the absolute tolerance is not flagged.
def test_reconcile_tolerance_boundary_is_not_flagged_and_keeps_signed_delta():
    result = reconcile(Decimal("10.05"), Decimal("10.00"), Decimal("0.05"))

    assert result["delta"] == Decimal("0.05")
    assert result["flagged"] is False


# X3.2-S1-C66-03: a positive delta above tolerance is flagged.
def test_reconcile_flags_positive_delta_just_above_tolerance():
    result = reconcile(Decimal("10.051"), Decimal("10.00"), Decimal("0.05"))

    assert result["delta"] == Decimal("0.051")
    assert result["flagged"] is True


# X3.2-S1-C66-04: absolute tolerance applies equally to under-ledger deltas.
def test_reconcile_flags_negative_delta_just_beyond_tolerance():
    result = reconcile(Decimal("9.949"), Decimal("10.00"), Decimal("0.05"))

    assert result["delta"] == Decimal("-0.051")
    assert result["flagged"] is True
