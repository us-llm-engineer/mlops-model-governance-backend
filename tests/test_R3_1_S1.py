"""R3.1-S1: replay an experiment log into an isolated MLflow SQLite store."""
import os
import sys

import pytest
from mlflow import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.experiments import LogEntry


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    # Deferred so this harness stays collectable without the bridge module;
    # execution, rather than collection, owns the API.
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(tmp_path))


def _client(tmp_path):
    return MlflowClient(tracking_uri=_uri(tmp_path))


def test_replay_creates_named_experiment_and_trace_tag(tmp_path):
    run = _bridge(tmp_path).mirror_experiment_run([], "trace-1", "evals")
    experiment = _client(tmp_path).get_experiment_by_name("evals")
    assert experiment is not None
    assert _client(tmp_path).get_run(run).data.tags["mlops_run_id"] == "trace-1"


def test_replay_preserves_metric_history_steps_at_raw_boundary(tmp_path):
    bridge = _bridge(tmp_path)
    run = bridge.mirror_experiment_run(
        [
            LogEntry(kind="metric", name="loss", value=0.8, step=0),
            LogEntry(kind="metric", name="loss", value=0.2, step=4),
        ],
        "trace-steps",
        "evals",
    )
    history = _client(tmp_path).get_metric_history(run, "loss")
    assert [(item.step, item.value) for item in history] == [(0, 0.8), (4, 0.2)]


def test_replay_skips_non_replayable_log_kinds_without_polluting_run(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [
            LogEntry(kind="param", name="seed", value=7, step=None),
            LogEntry(kind="checkpoint", name="artifact", value="sha", step=2),
            LogEntry(kind="backfill", name="loss", value=0.3, step=2),
            LogEntry(kind="metric", name="accuracy", value=0.9, step=2),
        ],
        "trace-kinds",
        "evals",
    )
    mirrored = bridge.get_mirrored_run("trace-kinds")
    assert mirrored["params"] == {"seed": 7}
    assert mirrored["metrics"] == {"accuracy": 0.9}


def test_replay_empty_log_creates_a_retrievable_tagged_run(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([], "trace-empty", "empty-log")
    mirrored = bridge.get_mirrored_run("trace-empty")
    assert mirrored["params"] == {}
    assert mirrored["metrics"] == {}
    assert mirrored["tags"]["mlops_run_id"] == "trace-empty"


def test_replay_through_fresh_bridge_reuses_tagged_run(tmp_path):
    first = _bridge(tmp_path).mirror_experiment_run(
        [LogEntry(kind="param", name="seed", value=3, step=None)], "trace-reuse", "evals"
    )
    second = _bridge(tmp_path).mirror_experiment_run(
        [LogEntry(kind="param", name="seed", value=3, step=None)], "trace-reuse", "evals"
    )
    assert second == first
    experiment = _client(tmp_path).get_experiment_by_name("evals")
    assert len(_client(tmp_path).search_runs([experiment.experiment_id], "tags.mlops_run_id = 'trace-reuse'")) == 1


def test_replay_existing_run_accepts_a_new_metric_entry(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry(kind="param", name="seed", value=3, step=None)], "trace-extend", "evals"
    )
    bridge.mirror_experiment_run(
        [LogEntry(kind="metric", name="accuracy", value=0.91, step=5)], "trace-extend", "evals"
    )
    assert bridge.get_mirrored_run("trace-extend")["metrics"]["accuracy"] == 0.91


def test_replay_keeps_experiments_separate_by_name(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run([], "trace-a", "experiment-a")
    bridge.mirror_experiment_run([], "trace-b", "experiment-b")
    client = _client(tmp_path)
    experiment_a = client.get_experiment_by_name("experiment-a")
    experiment_b = client.get_experiment_by_name("experiment-b")
    assert experiment_a.experiment_id != experiment_b.experiment_id
    assert len(client.search_runs([experiment_a.experiment_id])) == 1
    assert len(client.search_runs([experiment_b.experiment_id])) == 1


def test_replay_integer_metric_is_recorded_as_a_numeric_metric(tmp_path):
    bridge = _bridge(tmp_path)
    run = bridge.mirror_experiment_run(
        [LogEntry(kind="metric", name="retries", value=2, step=1)], "trace-int-metric", "evals"
    )
    history = _client(tmp_path).get_metric_history(run, "retries")
    assert len(history) == 1
    assert history[0].value == 2.0
