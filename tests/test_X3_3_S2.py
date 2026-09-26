"""X3.3-S2: C71/C72 burn-rate arithmetic and tenant budget-alert lifecycle."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.gates import GateDecision
from mlops.slo_guard import BudgetAlertTracker, burn_rate


def _tracker(threshold=2.0, consecutive_windows=3):
    audit = ChainedAuditStore(secret=b"x3-3-s2-secret")
    return audit, BudgetAlertTracker(audit, threshold, consecutive_windows)


def _denied_gate():
    return GateDecision(
        rule_name="cost_budget",
        passed=False,
        reason="unallocated fraction exceeds budget",
        predicates=[],
        timestamp="2026-01-01T00:00:00+00:00",
    )


# X3.3-S2-01 / C71: burn is error-budget consumption, not simply the raw
# failure rate; this non-integer result catches denominator mistakes.
def test_burn_rate_uses_target_error_budget_with_precision():
    assert burn_rate(observed_success_rate=0.875, target_success_rate=0.9) == pytest.approx(1.25)


# X3.3-S2-02 / C71: a target with no error budget is invalid rather than a
# division-by-zero result or an infinite burn rate.
def test_burn_rate_rejects_target_success_rate_of_one():
    with pytest.raises(ValueError):
        burn_rate(observed_success_rate=0.99, target_success_rate=1.0)


# X3.3-S2-03 / C72: threshold comparison is strict, and only the Nth
# consecutive above-threshold window activates the alert.
def test_alert_activates_only_on_nth_strict_threshold_breach():
    audit, tracker = _tracker(threshold=2.0, consecutive_windows=2)

    assert tracker.record_window("tenant-a", burn=2.0) is False
    assert tracker.record_window("tenant-a", burn=2.01) is False
    assert tracker.record_window("tenant-a", burn=2.01) is True

    alerts = [e for e in audit.export() if e["action"] == "slo.budget_alert"]
    assert len(alerts) == 1
    assert alerts[0]["resource"] == "tenant-a"


# X3.3-S2-04 / C72: a raised SLO alert is a correctly chained audit effect,
# not merely an in-memory flag.
def test_activated_alert_is_recorded_in_a_verifiable_audit_chain():
    audit, tracker = _tracker(consecutive_windows=2)

    assert tracker.record_window("tenant-a", burn=2.1) is False
    assert tracker.record_window("tenant-a", burn=2.1) is True

    export = audit.export()
    assert [(e["action"], e["decision"]) for e in export] == [("slo.budget_alert", "alert")]
    assert ChainVerifier(secret=b"x3-3-s2-secret").verify_export(export) is True


# X3.3-S2-05 / C72: continued breach after activation keeps the flag active
# but never emits one audit alert per window.
def test_sustained_breach_is_deduplicated_while_alert_remains_active():
    audit, tracker = _tracker(consecutive_windows=2)

    assert tracker.record_window("tenant-a", burn=3.0) is False
    assert tracker.record_window("tenant-a", burn=3.0) is True
    assert tracker.record_window("tenant-a", burn=3.0) is True
    assert tracker.record_window("tenant-a", burn=3.0) is True

    assert [e["action"] for e in audit.export()] == ["slo.budget_alert"]


# X3.3-S2-06 / C72: a recovered window clears an active alert and emits the
# separately specified clear event.
def test_recovery_clears_active_alert_and_records_clear_once():
    audit, tracker = _tracker(consecutive_windows=2)
    tracker.record_window("tenant-a", burn=3.0)
    assert tracker.record_window("tenant-a", burn=3.0) is True

    assert tracker.record_window("tenant-a", burn=2.0) is False
    assert [e["action"] for e in audit.export()] == [
        "slo.budget_alert",
        "slo.budget_alert_cleared",
    ]


# X3.3-S2-07 / C72: recovery resets the consecutive counter, so a new alert
# needs a full new N-window breach sequence.
def test_recovery_requires_a_full_new_breach_sequence_before_realerting():
    audit, tracker = _tracker(consecutive_windows=2)
    tracker.record_window("tenant-a", burn=3.0)
    assert tracker.record_window("tenant-a", burn=3.0) is True
    assert tracker.record_window("tenant-a", burn=1.0) is False

    assert tracker.record_window("tenant-a", burn=3.0) is False
    assert tracker.record_window("tenant-a", burn=3.0) is True
    assert [e["action"] for e in audit.export()] == [
        "slo.budget_alert",
        "slo.budget_alert_cleared",
        "slo.budget_alert",
    ]


# X3.3-S2-08 / C72: a failed cost gate produces its own tenant alert even
# when the tenant already has an active burn-rate alert.
def test_cost_alert_is_independent_of_an_active_burn_rate_alert():
    audit, tracker = _tracker(consecutive_windows=2)
    tracker.record_window("tenant-a", burn=3.0)
    assert tracker.record_window("tenant-a", burn=3.0) is True

    tracker.record_cost_alert("tenant-a", _denied_gate())

    assert [(e["action"], e["resource"]) for e in audit.export()] == [
        ("slo.budget_alert", "tenant-a"),
        ("cost.budget_alert", "tenant-a"),
    ]
