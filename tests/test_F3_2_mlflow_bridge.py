"""F3.2 slim suite: exec/mlops/interop/mlflow_bridge.py, contract in
the API contract section 2 (mlflow_bridge).

Frozen expectations (contract-derived):
  MlflowBridge(tracking_uri) validates tracking_uri starts with "sqlite:///"
  (exactly, three slashes) else mlops.kernel.ValidationFailed -- this
  structurally enforces the file-store maintenance-mode trap; a bare "file:"
  URI or a plain filesystem path must both be rejected the same way. Each
  MlflowBridge builds its OWN mlflow.MlflowClient(tracking_uri=...); it never
  touches global mlflow state, so two bridges over different tmp sqlite paths
  never see each other's runs/models (mirrors the F1 telemetry instance-
  isolation lesson).

  .mirror_experiment_run(tracker_entries: list[LogEntry], run_id: str,
  experiment_name: str) -> str: creates the MLflow experiment if missing,
  opens (or reuses) an MLflow run tagged mlops_run_id=run_id, replays each
  "param"-kind entry via log_param and each "metric"-kind entry via
  log_metric(step=...). Entries of any other kind (e.g. "checkpoint",
  "backfill") are not replayed -- silently skipped, never an error. Calling
  mirror_experiment_run twice for the SAME run_id reuses the same MLflow run
  (found via a tags.mlops_run_id search) rather than creating a duplicate;
  calling it again with the SAME param values is a no-op success (mlflow
  itself allows re-logging an unchanged param value); calling it again with a
  DIFFERENT value for an already-logged param name is a genuine data
  conflict -- mlflow's own log_param raises MlflowException
  (INVALID_PARAMETER_VALUE) in that case, and the bridge must never let that
  raw exception escape: it is translated to mlops.kernel.Conflict. A metric
  entry whose value cannot be interpreted as a float is a caller input error
  -- mlflow's own log_metric also raises MlflowException
  (INVALID_PARAMETER_VALUE) for a non-numeric value, and the bridge
  translates THAT raw exception to mlops.kernel.ValidationFailed (never lets
  it escape as MlflowException either) -- these are the suite's two distinct
  "a raw mlflow error path is always translated, never leaked" cases.

  .get_mirrored_run(run_id) -> {"params": {...original python types
  restored, never strings...}, "metrics": {...}, "tags": {...}} found via a
  tags.mlops_run_id search; mlops.kernel.NotFound for an unknown run_id.
  Trap B round-trip: mlflow.log_param stores everything as a string, so a
  float param logged as 0.01 must come back from get_mirrored_run as the
  python float 0.01, an int param as a python int, a bool param as a python
  bool, and a str param as a python str -- never all flattened to string.

  .mirror_stage_transition(model_name, version, stage: Stage) -> None:
  registers the model/version in MLflow if missing (source/run_id may be
  empty strings -- this is a mirror, not the system of record); version must
  be a numeric string (ASCII digits only, non-empty) else
  mlops.kernel.ValidationFailed. Alias mapping: Stage.STAGING -> alias
  "staging", Stage.PRODUCTION -> alias "production", Stage.REGISTERED and
  Stage.ARCHIVED clear any alias (after either, get_model_version_by_alias
  for both "staging" and "production" must raise/return nothing for that
  model). Idempotent: transitioning the same (model, version) to the same
  stage twice never raises. Setting the same alias for a different version
  of the same model moves the alias (mlflow's own alias semantics: an alias
  name maps to exactly one version).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops import kernel
from mlops.experiments import LogEntry
from mlops.model_stages import Stage
from mlops.interop.mlflow_bridge import MlflowBridge


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    return MlflowBridge(_uri(tmp_path))


def _raw_client(tmp_path):
    """A raw mlflow client on the SAME store, for out-of-band verification."""
    return MlflowClient(tracking_uri=_uri(tmp_path))


# ---------------------------------------------------------------------------
# Constructor validation (Trap A)
# ---------------------------------------------------------------------------


def test_constructor_accepts_sqlite_triple_slash_uri(tmp_path):
    bridge = MlflowBridge(_uri(tmp_path))
    assert bridge is not None


def test_constructor_rejects_file_uri(tmp_path):
    with pytest.raises(kernel.ValidationFailed):
        MlflowBridge(f"file://{tmp_path}/mlruns")


def test_constructor_rejects_bare_filesystem_path(tmp_path):
    with pytest.raises(kernel.ValidationFailed):
        MlflowBridge(str(tmp_path / "mlflow.db"))


def test_constructor_rejects_sqlite_two_slash_uri(tmp_path):
    # "sqlite://" (relative, two slashes) is NOT "sqlite:///" (absolute, three).
    # A relative host/path form -- deliberately not tmp_path-derived, since a
    # leading "/" from an absolute path would silently reconstruct a third slash.
    with pytest.raises(kernel.ValidationFailed):
        MlflowBridge("sqlite://relative-host/mlflow.db")


def test_constructor_rejects_non_string_uri(tmp_path):
    with pytest.raises(kernel.ValidationFailed):
        MlflowBridge(None)


# ---------------------------------------------------------------------------
# mirror_experiment_run: basic replay + get_mirrored_run
# ---------------------------------------------------------------------------


def test_mirror_creates_experiment_and_run(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [LogEntry(kind="param", name="lr", value=0.01, step=None)]
    mlflow_run_id = bridge.mirror_experiment_run(entries, "run-1", "exp-alpha")
    assert isinstance(mlflow_run_id, str) and mlflow_run_id

    client = _raw_client(tmp_path)
    exp = client.get_experiment_by_name("exp-alpha")
    assert exp is not None
    run = client.get_run(mlflow_run_id)
    assert run.data.tags.get("mlops_run_id") == "run-1"


def test_mirror_replays_params_and_metrics(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [
        LogEntry(kind="param", name="lr", value=0.01, step=None),
        LogEntry(kind="metric", name="acc", value=0.75, step=1),
    ]
    bridge.mirror_experiment_run(entries, "run-2", "exp-alpha")
    mirrored = bridge.get_mirrored_run("run-2")
    assert mirrored["params"]["lr"] == 0.01
    assert mirrored["metrics"]["acc"] == 0.75


def test_get_mirrored_run_restores_original_param_types(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [
        LogEntry(kind="param", name="lr", value=0.01, step=None),
        LogEntry(kind="param", name="epochs", value=10, step=None),
        LogEntry(kind="param", name="use_amp", value=True, step=None),
        LogEntry(kind="param", name="dataset", value="baseline-v3", step=None),
    ]
    bridge.mirror_experiment_run(entries, "run-types", "exp-types")
    params = bridge.get_mirrored_run("run-types")["params"]

    assert params["lr"] == 0.01
    assert isinstance(params["lr"], float)
    assert not isinstance(params["lr"], str)

    assert params["epochs"] == 10
    assert isinstance(params["epochs"], int)
    assert not isinstance(params["epochs"], bool)
    assert not isinstance(params["epochs"], str)

    assert params["use_amp"] is True

    assert params["dataset"] == "baseline-v3"
    assert isinstance(params["dataset"], str)


def test_get_mirrored_run_tags_contain_mlops_run_id(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [LogEntry(kind="param", name="lr", value=0.01, step=None)]
    bridge.mirror_experiment_run(entries, "run-tags", "exp-alpha")
    mirrored = bridge.get_mirrored_run("run-tags")
    assert mirrored["tags"]["mlops_run_id"] == "run-tags"


def test_mirror_skips_non_param_metric_kinds(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [
        LogEntry(kind="param", name="lr", value=0.01, step=None),
        LogEntry(kind="checkpoint", name="checkpoint", value="artifact-123", step=1),
        LogEntry(kind="metric", name="acc", value=0.9, step=1),
    ]
    # Must not raise even though "checkpoint" is not param/metric.
    bridge.mirror_experiment_run(entries, "run-mixed", "exp-alpha")
    mirrored = bridge.get_mirrored_run("run-mixed")
    assert mirrored["params"]["lr"] == 0.01
    assert mirrored["metrics"]["acc"] == 0.9
    assert "checkpoint" not in mirrored["params"]
    assert "checkpoint" not in mirrored["metrics"]


# ---------------------------------------------------------------------------
# mirror_experiment_run: idempotency / reuse
# ---------------------------------------------------------------------------


def test_mirror_twice_same_run_id_reuses_run(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [LogEntry(kind="param", name="lr", value=0.01, step=None)]
    first = bridge.mirror_experiment_run(entries, "run-dup", "exp-alpha")
    second = bridge.mirror_experiment_run(entries, "run-dup", "exp-alpha")
    assert first == second

    client = _raw_client(tmp_path)
    exp = client.get_experiment_by_name("exp-alpha")
    runs = client.search_runs(
        [exp.experiment_id], filter_string="tags.mlops_run_id = 'run-dup'"
    )
    assert len(runs) == 1


def test_mirror_twice_same_values_is_noop_success(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [
        LogEntry(kind="param", name="lr", value=0.01, step=None),
        LogEntry(kind="metric", name="acc", value=0.5, step=1),
    ]
    bridge.mirror_experiment_run(entries, "run-same", "exp-alpha")
    # Re-mirroring identical entries must not raise.
    bridge.mirror_experiment_run(entries, "run-same", "exp-alpha")
    mirrored = bridge.get_mirrored_run("run-same")
    assert mirrored["params"]["lr"] == 0.01
    assert mirrored["metrics"]["acc"] == 0.5


def test_mirror_different_run_ids_create_different_runs_reusing_experiment(tmp_path):
    bridge = _bridge(tmp_path)
    entries = [LogEntry(kind="param", name="lr", value=0.01, step=None)]
    first = bridge.mirror_experiment_run(entries, "run-x", "exp-shared")
    second = bridge.mirror_experiment_run(entries, "run-y", "exp-shared")
    assert first != second

    client = _raw_client(tmp_path)
    all_experiments = [e for e in client.search_experiments() if e.name == "exp-shared"]
    assert len(all_experiments) == 1


# ---------------------------------------------------------------------------
# mirror_experiment_run: input validation + translated mlflow errors
# ---------------------------------------------------------------------------


def test_mirror_rejects_empty_run_id(tmp_path):
    bridge = _bridge(tmp_path)
    with pytest.raises(kernel.ValidationFailed):
        bridge.mirror_experiment_run([], "", "exp-alpha")


def test_mirror_rejects_empty_experiment_name(tmp_path):
    bridge = _bridge(tmp_path)
    with pytest.raises(kernel.ValidationFailed):
        bridge.mirror_experiment_run([], "run-1", "")


def test_mirror_conflicting_param_value_raises_conflict_not_mlflow_exception(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry(kind="param", name="lr", value=0.01, step=None)],
        "run-conflict",
        "exp-alpha",
    )
    # Second mirror with a DIFFERENT value for the same param name -- mlflow's
    # own log_param raises MlflowException(INVALID_PARAMETER_VALUE) here; the
    # bridge must translate it, never let MlflowException itself propagate.
    with pytest.raises(kernel.Conflict):
        bridge.mirror_experiment_run(
            [LogEntry(kind="param", name="lr", value=0.02, step=None)],
            "run-conflict",
            "exp-alpha",
        )


def test_mirror_non_numeric_metric_value_raises_validation_failed_not_mlflow_exception(
    tmp_path,
):
    bridge = _bridge(tmp_path)
    entries = [LogEntry(kind="metric", name="acc", value="not-a-number", step=1)]
    # mlflow's own log_metric raises MlflowException(INVALID_PARAMETER_VALUE)
    # for a non-numeric value; the bridge must translate it.
    with pytest.raises(kernel.ValidationFailed):
        bridge.mirror_experiment_run(entries, "run-bad-metric", "exp-alpha")


# ---------------------------------------------------------------------------
# get_mirrored_run: NotFound
# ---------------------------------------------------------------------------


def test_get_mirrored_run_unknown_run_id_raises_not_found(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry(kind="param", name="lr", value=0.01, step=None)],
        "run-real",
        "exp-alpha",
    )
    with pytest.raises(kernel.NotFound):
        bridge.get_mirrored_run("run-real-but-not-quite")


# ---------------------------------------------------------------------------
# mirror_stage_transition: alias mapping
# ---------------------------------------------------------------------------


def test_transition_staging_sets_staging_alias_and_is_idempotent(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("model-a", "1", Stage.STAGING)
    # A repeat transition to the same stage must not raise and must leave
    # the alias pointing at the same version.
    bridge.mirror_stage_transition("model-a", "1", Stage.STAGING)
    client = _raw_client(tmp_path)
    mv = client.get_model_version_by_alias("model-a", "staging")
    assert str(mv.version) == "1"


def test_transition_production_sets_production_alias(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("model-b", "1", Stage.PRODUCTION)
    client = _raw_client(tmp_path)
    mv = client.get_model_version_by_alias("model-b", "production")
    assert str(mv.version) == "1"


def test_transition_registered_clears_any_alias(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("model-c", "1", Stage.STAGING)
    bridge.mirror_stage_transition("model-c", "1", Stage.REGISTERED)
    client = _raw_client(tmp_path)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("model-c", "staging")
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("model-c", "production")


def test_transition_archived_clears_any_alias(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("model-d", "1", Stage.PRODUCTION)
    bridge.mirror_stage_transition("model-d", "1", Stage.ARCHIVED)
    client = _raw_client(tmp_path)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("model-d", "staging")
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("model-d", "production")


def test_transition_creates_registered_model_and_version_if_missing(tmp_path):
    bridge = _bridge(tmp_path)
    # No prior register call anywhere -- the bridge must create everything.
    bridge.mirror_stage_transition("brand-new-model", "1", Stage.STAGING)
    client = _raw_client(tmp_path)
    model = client.get_registered_model("brand-new-model")
    assert model is not None
    mv = client.get_model_version_by_alias("brand-new-model", "staging")
    assert str(mv.version) == "1"


def test_transition_moves_alias_between_versions(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("model-f", "1", Stage.STAGING)
    bridge.mirror_stage_transition("model-f", "2", Stage.STAGING)
    client = _raw_client(tmp_path)
    mv = client.get_model_version_by_alias("model-f", "staging")
    assert str(mv.version) == "2"


@pytest.mark.parametrize("bad_version", ["", "abc", "-1", "1.5"])
def test_transition_rejects_non_numeric_version(tmp_path, bad_version):
    bridge = _bridge(tmp_path)
    with pytest.raises(kernel.ValidationFailed):
        bridge.mirror_stage_transition("model-g", bad_version, Stage.STAGING)


def test_transition_first_version_request_out_of_sequence_is_a_conflict(tmp_path):
    """Real gap found by two independent implementations converging on opposite bad fixes:
    MLflow auto-assigns version numbers (1, 2, 3, ...) and cannot be told to create version
    "5" directly on a brand-new model. One fix silently accepted whatever MLflow assigned
    (misaligning the mirror from the real domain version number); another created 4 throwaway
    junk versions to force the counter up (wasteful, racy under concurrent bridges). Neither is
    acceptable: the correct behavior is to FAIL LOUD when the requested version cannot be
    honored, matching this project's fail-closed style everywhere else."""
    bridge = _bridge(tmp_path)
    with pytest.raises(kernel.Conflict):
        bridge.mirror_stage_transition("brand-new-model-2", "5", Stage.STAGING)
    # And the mirror must not have silently left a wrongly-numbered version behind.
    client = _raw_client(tmp_path)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("brand-new-model-2", "staging")


def test_transition_leading_zero_out_of_sequence_is_conflict_without_model_or_alias(tmp_path):
    bridge = _bridge(tmp_path)
    # "007" is syntactically numeric and canonicalizes to 7.  MLflow cannot
    # create version 7 first, so the mirror must fail closed rather than map it
    # to version 1 or burn six dummy versions to advance MLflow's counter.
    with pytest.raises(kernel.Conflict):
        bridge.mirror_stage_transition("model-h", "007", Stage.STAGING)
    client = _raw_client(tmp_path)
    with pytest.raises(MlflowException):
        client.get_registered_model("model-h")
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("model-h", "staging")


# ---------------------------------------------------------------------------
# Instance isolation (mirrors the F1 telemetry-instance-isolation lesson)
# ---------------------------------------------------------------------------


def test_two_bridges_different_paths_do_not_share_runs(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    bridge_a = MlflowBridge(f"sqlite:///{dir_a}/mlflow.db")
    bridge_b = MlflowBridge(f"sqlite:///{dir_b}/mlflow.db")

    bridge_a.mirror_experiment_run(
        [LogEntry(kind="param", name="lr", value=0.01, step=None)],
        "shared-run-id",
        "exp-alpha",
    )
    with pytest.raises(kernel.NotFound):
        bridge_b.get_mirrored_run("shared-run-id")


def test_two_bridges_different_paths_do_not_share_stage_transitions(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    bridge_a = MlflowBridge(f"sqlite:///{dir_a}/mlflow.db")
    bridge_b = MlflowBridge(f"sqlite:///{dir_b}/mlflow.db")

    bridge_a.mirror_stage_transition("shared-model", "1", Stage.PRODUCTION)

    client_b = MlflowClient(tracking_uri=f"sqlite:///{dir_b}/mlflow.db")
    with pytest.raises(MlflowException):
        client_b.get_registered_model("shared-model")

    # bridge_b independently creating the "same" model/version is unaffected
    # by bridge_a's alias.
    bridge_b.mirror_stage_transition("shared-model", "1", Stage.STAGING)
    mv = client_b.get_model_version_by_alias("shared-model", "staging")
    assert str(mv.version) == "1"
    with pytest.raises(MlflowException):
        client_b.get_model_version_by_alias("shared-model", "production")
