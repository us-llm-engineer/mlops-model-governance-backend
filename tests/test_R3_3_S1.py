"""R3.3-S1: per-instance store isolation and non-global MLflow boundaries."""
import os
import sys

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops import kernel
from mlops.experiments import LogEntry
from mlops.model_stages import Stage


def _uri(path):
    return f"sqlite:///{path}/mlflow.db"


def _bridge(path):
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(path))


def test_constructor_never_calls_global_tracking_configuration(tmp_path, monkeypatch):
    def forbidden_global_configuration(*_args, **_kwargs):
        raise AssertionError("bridge must use its own MlflowClient, not global tracking state")

    monkeypatch.setattr(mlflow, "set_tracking_uri", forbidden_global_configuration)
    assert _bridge(tmp_path) is not None


def test_different_store_paths_isolate_replayed_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _bridge(a).mirror_experiment_run([], "same-run", "isolated")
    with pytest.raises(kernel.NotFound):
        _bridge(b).get_mirrored_run("same-run")


def test_different_store_paths_isolate_registered_models(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _bridge(a).mirror_stage_transition("same-model", "1", Stage.STAGING)
    with pytest.raises(MlflowException):
        MlflowClient(tracking_uri=_uri(b)).get_registered_model("same-model")


def test_same_store_path_shares_replay_state_between_bridge_instances(tmp_path):
    _bridge(tmp_path).mirror_experiment_run(
        [LogEntry(kind="param", name="seed", value=9, step=None)], "shared-run", "shared"
    )
    assert _bridge(tmp_path).get_mirrored_run("shared-run")["params"] == {"seed": 9}


def test_same_store_path_shares_alias_state_between_bridge_instances(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("shared-model", "1", Stage.STAGING)
    _bridge(tmp_path).mirror_stage_transition("shared-model", "1", Stage.PRODUCTION)
    assert str(MlflowClient(tracking_uri=_uri(tmp_path)).get_model_version_by_alias("shared-model", "production").version) == "1"


def test_first_run_creation_race_reconciles_to_one_active_tagged_run(tmp_path, monkeypatch):
    """An interleaved first create must converge instead of preserving twins.

    MLflow does not enforce uniqueness for ``mlops_run_id`` tags.  The wrapper
    models the exact gap between the bridge's initial tag lookup and its
    ``create_run`` call: a competing client writes the tagged run first, then
    the original provider call continues.  No threads are needed, so this
    remains a deterministic regression for post-create reconciliation.
    """
    bridge = _bridge(tmp_path)
    verifier = MlflowClient(tracking_uri=_uri(tmp_path))
    original_create_run = bridge._client.create_run

    def create_after_competitor(experiment_id, tags=None, **kwargs):
        verifier.create_run(experiment_id, tags=tags)
        return original_create_run(experiment_id, tags=tags, **kwargs)

    monkeypatch.setattr(bridge._client, "create_run", create_after_competitor)
    mirrored_id = bridge.mirror_experiment_run([], "first-race", "race-exp")

    experiment = verifier.get_experiment_by_name("race-exp")
    active_matches = verifier.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlops_run_id = 'first-race'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert len(active_matches) == 1
    assert mirrored_id == active_matches[0].info.run_id
    assert bridge.get_mirrored_run("first-race")["tags"]["mlops_run_id"] == "first-race"


def test_invalid_uri_is_rejected_before_global_tracking_is_touched(tmp_path, monkeypatch):
    def forbidden_global_configuration(*_args, **_kwargs):
        raise AssertionError("validation must not fall back to global MLflow configuration")

    monkeypatch.setattr(mlflow, "set_tracking_uri", forbidden_global_configuration)
    from mlops.interop.mlflow_bridge import MlflowBridge

    with pytest.raises(kernel.ValidationFailed):
        MlflowBridge(f"file://{tmp_path}/mlruns")


def test_a_second_store_has_no_experiment_created_by_first_store(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _bridge(a).mirror_experiment_run([], "run-a", "only-a")
    assert MlflowClient(tracking_uri=_uri(b)).get_experiment_by_name("only-a") is None
