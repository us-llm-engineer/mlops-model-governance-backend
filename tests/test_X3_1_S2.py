"""X3.1-S2 (C62, C63): deterministic attribution and tolerant namespace tagging."""
import os
import sys
from decimal import Decimal

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.cost_ledger import (
    CostRow,
    attribute,
    fallback_only_spend,
    tag_namespace,
    unallocated_fraction,
)


def row(**overrides):
    values = {
        "source": "cloud-bill",
        "resource": "gpu-train-01",
        "owner": None,
        "amount": Decimal("10.00"),
        "window_start": 100,
        "window_end": 200,
        "labels": {},
        "quantity": None,
    }
    values.update(overrides)
    return CostRow(**values)


# X3.1-S2-01 / C62: a truthy explicit owner always wins over every fallback label.
def test_attribute_prefers_owner_over_populated_fallbacks():
    cost = row(owner="owner-team", labels={"team": "fallback-team", "project": "project-a"})
    assert attribute(cost, ["team", "project"]) == "owner-team"


# X3.1-S2-02 / C62: fallback precedence follows caller order and skips empty candidates.
def test_attribute_selects_first_truthy_fallback_in_order():
    cost = row(labels={"owner_tag": "", "team": "analytics", "project": "proj-7"})
    assert attribute(cost, ["owner_tag", "team", "project"]) == "analytics"


# X3.1-S2-03 / C62 invalid-attribution boundary: no usable label is visibly unallocated.
def test_attribute_returns_unallocated_when_owner_and_fallbacks_are_empty_or_missing():
    cost = row(owner="", labels={"team": "", "project": None})
    assert attribute(cost, ["team", "project", "absent"]) == "<unallocated>"


# X3.1-S2-04 / C62: U is the exact unallocated spend fraction, not a row count fraction.
def test_unallocated_fraction_is_amount_weighted_and_decimal_exact():
    rows = [
        row(amount=Decimal("1.00"), owner="known"),
        row(amount=Decimal("3.00"), labels={}),
    ]
    result = unallocated_fraction(rows, ["team"])
    assert isinstance(result, Decimal)
    assert result == Decimal("0.75")


# X3.1-S2-05 / C62 boundary: an empty ledger reports zero unallocated fraction.
def test_unallocated_fraction_empty_ledger_is_decimal_zero():
    result = unallocated_fraction([], ["team"])
    assert result == Decimal("0")
    assert isinstance(result, Decimal)


# X3.1-S2-06 / C62: F counts fallback-attributed spend only, excluding explicit owners/unallocated rows.
def test_fallback_only_spend_excludes_owned_and_unallocated_amounts():
    rows = [
        row(amount=Decimal("2.00"), owner="owner", labels={"team": "fallback"}),
        row(amount=Decimal("3.25"), labels={"team": "fallback"}),
        row(amount=Decimal("4.50"), labels={}),
    ]
    assert fallback_only_spend(rows, ["team"]) == Decimal("3.25")


# X3.1-S2-07 / C63: an attributed owner matching RBAC namespace is marked known without altering row.
def test_tag_namespace_marks_known_attributed_namespace_and_preserves_row_identity():
    cost = row(owner=None, labels={"team": "platform"})
    tagged = tag_namespace(cost, {"platform", "research"})
    assert tagged == {"row": cost, "namespace_known": True}
    assert tagged["row"] is cost


# X3.1-S2-08 / C63 invalid namespace: unknown attribution is ledgered and flagged, never rejected.
def test_tag_namespace_tolerates_unknown_namespace_without_rewriting_row():
    cost = row(owner="external-vendor", labels={"team": "platform"})
    tagged = tag_namespace(cost, {"platform"})
    assert tagged["row"] is cost
    assert tagged["namespace_known"] is False
    assert cost.owner == "external-vendor"
