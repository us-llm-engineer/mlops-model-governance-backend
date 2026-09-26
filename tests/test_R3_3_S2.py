"""R3.3-S2: integration of F3 framework outputs at real module boundaries."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.experiments import LogEntry
from mlops.lineage import LineageGraph
from mlops.model_stages import Stage
from mlops.policy_engine import PolicyBundle, Rule


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(tmp_path))


def _pairs(n=30):
    return [(float(index), float(index % 5)) for index in range(n)]


def _binary_xy(n=12):
    return [float(index) for index in range(n)], [0.0] * (n // 2) + [1.0] * (n // 2)


def test_drift_frame_summary_can_be_replayed_as_typed_run_metadata(tmp_path):
    from mlops.ext.drift_frame import DriftWindowFrame

    rows = len(DriftWindowFrame(_pairs()).to_daily_summary())
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("param", "daily_rows", rows, None)], "drift-run", "f3")
    restored = bridge.get_mirrored_run("drift-run")["params"]["daily_rows"]
    assert type(restored) is int
    assert restored == rows


def test_isotonic_probability_can_be_replayed_as_mlflow_metric(tmp_path):
    from mlops.ext.calibration_sk import IsotonicCrossCheck

    x, y = _binary_xy()
    check = IsotonicCrossCheck()
    check.fit(x, y)
    probability = check.predict([5.5])[0]
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("metric", "calibrated", probability, 1)], "cal-run", "f3")
    assert bridge.get_mirrored_run("cal-run")["metrics"]["calibrated"] == pytest.approx(probability)


def test_platt_probability_can_be_replayed_as_mlflow_metric(tmp_path):
    from mlops.ext.calibration_sk import PlattCrossCheck

    x, y = _binary_xy()
    check = PlattCrossCheck()
    check.fit(x, [int(value) for value in y])
    probability = check.predict_proba_positive([5.5])[0]
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("metric", "platt", probability, 1)], "platt-run", "f3")
    assert bridge.get_mirrored_run("platt-run")["metrics"]["platt"] == pytest.approx(probability)


def test_lineage_blast_radius_count_can_be_replayed_as_typed_metadata(tmp_path):
    from mlops.ext.lineage_graph import blast_radius

    graph = LineageGraph()
    data = graph.add_node("data", {"name": "d"}, "2026-01-01T00:00:00Z")
    train = graph.add_node("training", {"name": "t"}, "2026-01-01T00:01:00Z")
    model = graph.add_node("model", {"name": "m"}, "2026-01-01T00:02:00Z")
    graph.add_edge(data, train, "input")
    graph.add_edge(train, model, "produces")
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry("param", "affected_nodes", len(blast_radius(graph, data)), None)], "lineage-run", "f3"
    )
    assert bridge.get_mirrored_run("lineage-run")["params"]["affected_nodes"] == 2


def test_casbin_decision_can_be_replayed_as_boolean_metadata(tmp_path):
    from mlops.interop.casbin_adapter import build_enforcer_from_bundle, enforcer_decide

    bundle = PolicyBundle(
        [Rule("allow-deploy", 1, "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"})],
        name="f3-policy",
        version=1,
    )
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    assert untranslatable == []
    allowed = enforcer_decide(enforcer, "admin", "deploy", "prod")
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("param", "policy_allowed", allowed, None)], "policy-run", "f3")
    assert bridge.get_mirrored_run("policy-run")["params"]["policy_allowed"] is True


def test_worker_tick_result_can_be_replayed_as_metric(tmp_path):
    from mlops.svc.workers import incident_escalation_tick

    class Manager:
        """Duck-types the real IncidentManager contract incident_escalation_tick
        uses: 3 open incidents, all of which rise a tier this tick."""

        def list_open_incident_ids(self):
            return ["a", "b", "c"]

        def get(self, incident_id):
            return {"current_tier": 1}

        def escalate_if_overdue(self, incident_id):
            return 2

    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry("metric", "escalated", incident_escalation_tick(Manager()), 0)], "worker-run", "f3"
    )
    assert bridge.get_mirrored_run("worker-run")["metrics"]["escalated"] == 3.0


def test_stage_alias_and_replayed_run_coexist_in_one_bridge_instance(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("param", "source", "integration", None)], "both-run", "f3")
    bridge.mirror_stage_transition("both-model", "1", Stage.PRODUCTION)
    assert bridge.get_mirrored_run("both-run")["params"]["source"] == "integration"
    from mlflow import MlflowClient

    assert str(MlflowClient(tracking_uri=_uri(tmp_path)).get_model_version_by_alias("both-model", "production").version) == "1"


def test_f3_boundary_output_remains_json_safe_after_replay(tmp_path):
    from mlops.ext.drift_frame import DriftWindowFrame

    records = DriftWindowFrame(_pairs()).export_records()
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([LogEntry("param", "record_count", len(records), None)], "json-run", "f3")
    payload = {"records": records, "mirrored": bridge.get_mirrored_run("json-run")}
    assert json.loads(json.dumps(payload))["mirrored"]["params"]["record_count"] == len(records)
