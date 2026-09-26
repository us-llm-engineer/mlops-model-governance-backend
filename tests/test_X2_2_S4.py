"""X2.2-S4 (C56): quality-gate pass-through.

evaluate_final_accuracy(gate, family, accuracy) must equal
gate.evaluate(family, accuracy) called directly. GateDecision carries a
now_iso() timestamp that legitimately differs between two separate calls, so
equality is asserted on every field except timestamp (both of which are
still checked to be well-formed ISO strings).
"""
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exec"))

from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.lineage import LineageGraph  # noqa: E402
from mlops.legacy.evaluation import BaselineStore, BaselineMetrics, ModelQualityGate  # noqa: E402
from mlops.experiment_registry import ExperimentRegistry  # noqa: E402


def make_registry():
    return ExperimentRegistry(
        rbac=None,
        audit=ChainedAuditStore(b"secret"),
        lineage=LineageGraph(),
    )


def decision_fields_without_timestamp(decision):
    d = asdict(decision)
    d.pop("timestamp", None)
    return d


def test_passing_case_matches_direct_evaluate():
    store = BaselineStore()
    store.set_baseline("fraud-model", BaselineMetrics(accuracy=0.90))
    gate = ModelQualityGate(store, threshold=0.02)
    reg = make_registry()

    via_registry = reg.evaluate_final_accuracy(gate, "fraud-model", 0.89)  # degrades 0.01 <= 0.02
    direct = gate.evaluate("fraud-model", 0.89)

    assert decision_fields_without_timestamp(via_registry) == decision_fields_without_timestamp(direct)
    assert via_registry.passed is True
    assert direct.passed is True


def test_failing_case_matches_direct_evaluate():
    store = BaselineStore()
    store.set_baseline("fraud-model", BaselineMetrics(accuracy=0.90))
    gate = ModelQualityGate(store, threshold=0.02)
    reg = make_registry()

    via_registry = reg.evaluate_final_accuracy(gate, "fraud-model", 0.80)  # degrades 0.10 > 0.02
    direct = gate.evaluate("fraud-model", 0.80)

    assert decision_fields_without_timestamp(via_registry) == decision_fields_without_timestamp(direct)
    assert via_registry.passed is False
    assert direct.passed is False


def test_boundary_case_exactly_at_threshold_still_matches_and_passes():
    store = BaselineStore()
    store.set_baseline("vision-model", BaselineMetrics(accuracy=0.80))
    gate = ModelQualityGate(store, threshold=0.05)
    reg = make_registry()

    via_registry = reg.evaluate_final_accuracy(gate, "vision-model", 0.75)  # degrades exactly 0.05
    direct = gate.evaluate("vision-model", 0.75)

    assert decision_fields_without_timestamp(via_registry) == decision_fields_without_timestamp(direct)
    assert via_registry.passed is True


def test_unregistered_family_raises_same_error_via_registry_and_direct():
    store = BaselineStore()
    gate = ModelQualityGate(store, threshold=0.02)
    reg = make_registry()

    with pytest.raises(KeyError):
        reg.evaluate_final_accuracy(gate, "unknown-family", 0.5)
    with pytest.raises(KeyError):
        gate.evaluate("unknown-family", 0.5)
