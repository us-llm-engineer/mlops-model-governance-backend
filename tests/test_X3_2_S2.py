"""X3.2-S2: C67 overlap exclusion and C68 chained cost-operation audits."""
import os
import sys
from decimal import Decimal


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.cost_gate import (
    budget_gate,
    detect_double_counted,
    exclude_source,
    reconcile,
)
from mlops.cost_ledger import CostRow, total_spend


SECRET = b"x3-2-s2-secret"


def _row(source, resource, amount, start=0, end=60, owner="team-a"):
    return CostRow(
        source=source,
        resource=resource,
        owner=owner,
        amount=Decimal(amount),
        window_start=start,
        window_end=end,
    )


# X3.2-S2-C67-01: same resource/window across distinct sources is overlap.
def test_detect_double_counted_finds_cross_source_matching_fingerprint():
    aws = _row("aws", "worker-7", "11.00")
    gateway = _row("gateway", "worker-7", "11.00")

    assert detect_double_counted([aws, gateway]) == [(aws, gateway)]


# X3.2-S2-C67-02: a duplicate inside one source does not cross a source boundary.
def test_detect_double_counted_ignores_matching_rows_from_same_source():
    rows = [_row("aws", "worker-7", "11.00"), _row("aws", "worker-7", "11.00")]

    assert detect_double_counted(rows) == []


# X3.2-S2-C67-03: either window endpoint changing creates a different fingerprint.
def test_detect_double_counted_requires_the_entire_time_window_to_match():
    base = _row("aws", "worker-7", "11.00", start=0, end=60)
    shifted_start = _row("gateway", "worker-7", "11.00", start=1, end=60)
    shifted_end = _row("reseller", "worker-7", "11.00", start=0, end=61)

    assert detect_double_counted([base, shifted_start, shifted_end]) == []


# X3.2-S2-C67-04: three independent sources yield every cross-source pair.
def test_detect_double_counted_reports_all_pairs_for_three_sources():
    aws = _row("aws", "worker-7", "11.00")
    gateway = _row("gateway", "worker-7", "11.00")
    reseller = _row("reseller", "worker-7", "11.00")

    pairs = detect_double_counted([aws, gateway, reseller])

    assert len(pairs) == 3
    assert {frozenset((left.source, right.source)) for left, right in pairs} == {
        frozenset(("aws", "gateway")),
        frozenset(("aws", "reseller")),
        frozenset(("gateway", "reseller")),
    }


# X3.2-S2-C67-05: excluding one overlapping source removes exactly its overlap spend.
def test_excluding_overlapping_source_reduces_total_by_exact_overlap_amount():
    aws_overlap = _row("aws", "worker-7", "11.00")
    gateway_overlap = _row("gateway", "worker-7", "11.00")
    independent = _row("aws", "worker-8", "4.00")
    rows = [aws_overlap, gateway_overlap, independent]

    assert detect_double_counted(rows) == [(aws_overlap, gateway_overlap)]
    without_gateway = exclude_source(rows, "gateway")
    assert total_spend(rows) - total_spend(without_gateway) == Decimal("11.00")
    assert total_spend(without_gateway) == Decimal("15.00")


# X3.2-S2-C68-01: each budget-gate invocation creates one decision-specific audit entry.
def test_budget_gate_appends_one_audit_entry_for_each_allow_and_deny_call():
    audit = ChainedAuditStore(SECRET)

    budget_gate([_row("aws", "allocated", "1.00")], Decimal("0"), [], audit)
    budget_gate([_row("aws", "unallocated", "1.00", owner=None)], Decimal("0"), [], audit)

    entries = audit.export()
    assert [(entry["action"], entry["decision"]) for entry in entries] == [
        ("cost.budget_gate", "allow"),
        ("cost.budget_gate", "deny"),
    ]


# X3.2-S2-C68-02: each reconciliation call contributes precisely one audit decision.
def test_reconcile_appends_one_audit_entry_for_each_within_and_over_tolerance_call():
    audit = ChainedAuditStore(SECRET)

    reconcile(Decimal("10.00"), Decimal("10.00"), Decimal("0"), audit)
    reconcile(Decimal("10.01"), Decimal("10.00"), Decimal("0"), audit)

    entries = audit.export()
    assert [entry["action"] for entry in entries] == ["cost.reconcile", "cost.reconcile"]
    assert [entry["decision"] for entry in entries] == ["allow", "deny"]


# X3.2-S2-C68-03: a ledger, gate, reconciliation cycle has a verifiable chain.
def test_full_cost_gate_and_reconciliation_audit_chain_verifies_end_to_end():
    audit = ChainedAuditStore(SECRET)
    rows = [_row("aws", "allocated", "8.00"), _row("aws", "unallocated", "2.00", owner=None)]

    budget_gate(rows, Decimal("0.20"), [], audit)
    reconcile(total_spend(rows), Decimal("10.00"), Decimal("0"), audit)

    export = audit.export()
    assert [entry["action"] for entry in export] == ["cost.budget_gate", "cost.reconcile"]
    assert ChainVerifier(SECRET).verify_export(export) is True
