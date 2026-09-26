"""X2.1-S1 (C49): log structure (param/metric entries, append order) + pivot query."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.state_machine import PipelineRunManager
from mlops.legacy.artifacts import ArtifactStore
from mlops.audit import ChainedAuditStore
from mlops.experiments import ExperimentTracker

SECRET = b"x2-1-s1-secret"


def make_tracker():
    return ExperimentTracker(PipelineRunManager(), ArtifactStore(), ChainedAuditStore(SECRET))


def test_start_run_logs_each_param_as_a_param_entry():
    tracker = make_tracker()
    tracker.start_run("run-a", {"lr": 0.01, "batch_size": 32, "optimizer": "adam"})
    entries = tracker.entries("run-a")
    param_entries = [e for e in entries if e.kind == "param"]
    assert len(param_entries) == 3
    by_name = {e.name: e.value for e in param_entries}
    assert by_name == {"lr": 0.01, "batch_size": 32, "optimizer": "adam"}


def test_log_metric_appends_in_call_order_not_sorted():
    tracker = make_tracker()
    tracker.start_run("run-b", {})
    tracker.log_metric("run-b", "recall", 0.5, step=1)
    tracker.log_metric("run-b", "acc", 0.9, step=1)
    tracker.log_metric("run-b", "acc", 0.95, step=2)
    metric_entries = [e for e in tracker.entries("run-b") if e.kind == "metric"]
    # exact call order preserved, independent of alphabetical name order
    assert [(e.name, e.value, e.step) for e in metric_entries] == [
        ("recall", 0.5, 1),
        ("acc", 0.9, 1),
        ("acc", 0.95, 2),
    ]


def test_entries_is_a_copy_and_reflects_params_then_metrics_append_order():
    tracker = make_tracker()
    tracker.start_run("run-c", {"lr": 0.1})
    tracker.log_metric("run-c", "loss", 1.0)
    entries = tracker.entries("run-c")
    assert [e.kind for e in entries] == ["param", "metric"]
    # mutating the returned list must not affect internal state
    entries.append("garbage")
    entries2 = tracker.entries("run-c")
    assert len(entries2) == 2
    assert entries2[-1].kind == "metric"


def test_pivot_across_multiple_runs_omits_missing_metric_not_null():
    tracker = make_tracker()
    tracker.start_run("run-x", {})
    tracker.start_run("run-y", {})
    tracker.log_metric("run-x", "acc", 0.8)
    tracker.log_metric("run-x", "recall", 0.7)
    tracker.log_metric("run-y", "acc", 0.6)
    # run-y never logs "recall"
    table = tracker.pivot(["acc", "recall"])
    assert table["run-x"] == {"acc": 0.8, "recall": 0.7}
    assert table["run-y"] == {"acc": 0.6}
    assert "recall" not in table["run-y"]


def test_pivot_uses_latest_value_when_metric_logged_multiple_times():
    tracker = make_tracker()
    tracker.start_run("run-z", {})
    tracker.log_metric("run-z", "acc", 0.1, step=1)
    tracker.log_metric("run-z", "acc", 0.2, step=2)
    tracker.log_metric("run-z", "acc", 0.3, step=3)
    table = tracker.pivot(["acc"])
    assert table["run-z"]["acc"] == 0.3


def test_pivot_run_with_no_requested_metrics_is_absent_entirely():
    tracker = make_tracker()
    tracker.start_run("run-p", {})
    tracker.start_run("run-q", {})
    tracker.log_metric("run-p", "f1", 0.5)
    table = tracker.pivot(["f1"])
    assert "run-p" in table
    assert "run-q" not in table
