"""X3.1-S1 (C61, C64): immutable-input ledger rows and Decimal-exact totals."""
import os
import sys
from copy import deepcopy
from decimal import Decimal

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.cost_ledger import CostRow, total_spend


def row(**overrides):
    values = {
        "source": "cloud-bill",
        "resource": "gpu-train-01",
        "owner": "ml-platform",
        "amount": Decimal("12.34"),
        "window_start": 1_700_000_000,
        "window_end": 1_700_003_600,
        "labels": {"team": "ml-platform", "environment": "prod"},
        "quantity": 2.5,
    }
    values.update(overrides)
    return CostRow(**values)


# X3.1-S1-01 / C61: a row retains every documented cost-object field verbatim.
def test_cost_row_retains_documented_fields_and_optional_quantity():
    cost = row()
    assert (
        cost.source,
        cost.resource,
        cost.owner,
        cost.amount,
        cost.window_start,
        cost.window_end,
        cost.labels,
        cost.quantity,
    ) == (
        "cloud-bill",
        "gpu-train-01",
        "ml-platform",
        Decimal("12.34"),
        1_700_000_000,
        1_700_003_600,
        {"team": "ml-platform", "environment": "prod"},
        2.5,
    )


# X3.1-S1-02 / C61 boundary: quantity is genuinely optional, not replaced with zero.
def test_cost_row_allows_absent_optional_quantity():
    cost = row(quantity=None)
    assert cost.quantity is None
    assert cost.amount == Decimal("12.34")


# X3.1-S1-03 / C61: adding a row and reading its total must not rewrite that row or labels.
def test_ledger_operations_do_not_mutate_the_row_or_its_labels():
    cost = row()
    before = deepcopy(cost)
    ledger = []
    ledger.append(cost)
    assert total_spend(ledger) == Decimal("12.34")
    assert cost == before
    assert cost.labels == before.labels


# X3.1-S1-04 / C61 boundary: distinct source/window rows remain independently usable.
def test_total_spend_accepts_rows_from_distinct_sources_and_windows():
    rows = [
        row(source="aws", amount=Decimal("1.20"), window_start=10, window_end=20),
        row(source="saas", amount=Decimal("3.45"), window_start=20, window_end=30),
    ]
    assert total_spend(rows) == Decimal("4.65")


# X3.1-S1-05 / C64: totals are Decimal values, never a float-derived monetary result.
def test_total_spend_returns_decimal_for_decimal_rows():
    result = total_spend([row(amount=Decimal("0.10")), row(amount=Decimal("0.20"))])
    assert isinstance(result, Decimal)
    assert result == Decimal("0.30")


# X3.1-S1-06 / C64 precision: a familiar binary-float trap must sum bit-for-bit in Decimal.
def test_total_spend_avoids_binary_float_drift_for_tenths():
    rows = [row(amount=Decimal("0.10")) for _ in range(10)]
    assert total_spend(rows) == Decimal("1.00")


# X3.1-S1-07 / C64 precision: heterogeneous sub-cent values preserve the hand-computed total.
def test_total_spend_preserves_mixed_decimal_scales_exactly():
    rows = [
        row(amount=Decimal("0.0001")),
        row(amount=Decimal("2.34567")),
        row(amount=Decimal("1000000.00002")),
    ]
    assert total_spend(rows) == Decimal("1000002.34579")


# X3.1-S1-08 / C64 boundary: an empty ledger has an exact zero total, not a division/type error.
def test_total_spend_empty_ledger_is_decimal_zero():
    result = total_spend([])
    assert result == Decimal("0")
    assert isinstance(result, Decimal)
