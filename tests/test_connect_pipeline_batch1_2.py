"""connect-pipeline sweep, Batches 1-2: proves the real entrypoint wiring gap is closed.

Before this: exec/mlops/svc/__main__.py called create_app(settings) with no extensions=,
workers=, or repo=, so an entire cluster of already-tested, already-wired-to-each-other code
(5 extension factories, 3 routers, telemetry, domain metrics, both periodic workers, the
dataset_version validation hook, drift-monitor persistence, SBOM self-check) was never
actually reachable when the service ran. These tests build the app the same way serve()/
__main__.py now do (via build_default_wiring) and prove each closed orphan now has a real,
non-test, non-trivial caller -- not just a new unit test of the orphan in isolation.

Run: cd .mlops-control-plane && PYTHONPATH=exec python3 -m pytest tests/test_connect_pipeline_batch1_2.py -q
"""
import os
import sys
import tempfile

import anyio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient

from mlops.svc.app import build_default_wiring, create_app
from mlops.svc.settings import Settings

ADMIN = "tk-admin-00000000001"
TOKENS = {ADMIN: {"name": "root", "role": "admin"}}


@pytest.fixture
def wired_client(tmp_path):
    """A real app built exactly the way serve()/__main__.py build it, with a real,
    freshly-migrated sqlite db backing OpsRepo/SqlRepo (not :memory:, since neither
    can be constructed against a non-persistent URL)."""
    db_path = str(tmp_path / "wiring.db")
    settings = Settings(
        audit_secret="s" * 20, tokens=TOKENS, db_path=db_path,
        incident_escalation_interval_s=0.05, drift_evaluation_interval_s=0.05,
        _env_file=None,
    )
    wiring = build_default_wiring(settings)
    app = create_app(settings, **wiring)
    with TestClient(app) as client:
        yield client, app, wiring


def _auth():
    return {"Authorization": f"Bearer {ADMIN}"}


# ---------------------------------------------------------------- extensions/routers reachable

def test_incidents_router_is_mounted_and_reachable(wired_client):
    """Before Batch 1: __main__.py never passed extensions=[incidents_extension(...)],
    so /v1/incidents 404'd against a server started the documented way. sdk.py's
    open_incident/get_incident POST/GET this exact path."""
    client, _, _ = wired_client
    r = client.post("/v1/incidents", json={"title": "t", "signal": {"accuracy_drop_pct": 15}}, headers=_auth())
    assert r.status_code == 201
    iid = r.json()["incident_id"]
    r2 = client.get(f"/v1/incidents/{iid}", headers=_auth())
    assert r2.status_code == 200


def test_audit_verify_router_is_mounted_and_reachable(wired_client):
    client, _, _ = wired_client
    r = client.post(
        "/v1/audit/verify",
        json={"export": [], "head": {"length": 0, "signature": "x", "anchor": "y"}},
        headers=_auth(),
    )
    assert r.status_code == 200


def test_policy_history_router_is_mounted_and_reachable(wired_client):
    client, _, _ = wired_client
    r = client.get("/v1/policy/versions", headers=_auth())
    assert r.status_code == 200


# ---------------------------------------------------------------- dataset_version real validation

def test_dataset_version_is_validated_against_a_real_registry(wired_client):
    """Before Batch 2: ModelRegistry was always built with datasets=None, so ANY
    dataset_version (valid-looking or not) raised "datasets registry not set" -- the
    field was accepted by the schema but never actually validated against anything."""
    client, _, _ = wired_client
    r = client.post(
        "/v1/datasets",
        json={"name": "ds1", "content_hash": "a" * 64, "rows": 10, "dataset_schema": {"type": "object"}},
        headers=_auth(),
    )
    assert r.status_code == 201
    version_id = r.json()["version_id"]

    ok = client.post(
        "/v1/models",
        json={"model_id": "m1", "version_id": "v1", "artifact_hash": "b" * 64, "dataset_version": version_id},
        headers=_auth(),
    )
    assert ok.status_code == 201
    assert ok.json()["dataset_version"] == version_id

    bad = client.post(
        "/v1/models",
        json={"model_id": "m2", "version_id": "v1", "artifact_hash": "c" * 64, "dataset_version": "does-not-exist"},
        headers=_auth(),
    )
    assert bad.status_code == 404


def test_model_registration_mirrors_into_the_real_sql_repo(wired_client):
    """state.repo (SqlRepo) was previously always None in production; _mirror_record's
    upsert_model call was 100% dead code, never exercised against a real OpsRepo/SqlRepo
    (which don't share a method name -- this caught a real repo-type mismatch bug)."""
    client, app, wiring = wired_client
    r = client.post(
        "/v1/models",
        json={"model_id": "m1", "version_id": "v1", "artifact_hash": "d" * 64},
        headers=_auth(),
    )
    assert r.status_code == 201
    mirrored = wiring["repo"].get_model("m1")
    assert mirrored and mirrored[0]["version_id"] == "v1"


# ---------------------------------------------------------------- drift monitor persistence + worker

def test_named_drift_check_persists_a_monitor_the_worker_can_later_evaluate(wired_client):
    client, app, wiring = wired_client
    r = client.post(
        "/v1/drift/check",
        json={"reference": list(range(30)), "window": list(range(30)), "name": "m1-drift"},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert "m1-drift" in app.state.mlops.drift_monitors


def test_real_periodic_workers_tick_against_real_objects_without_error(tmp_path):
    """Runs the exact PeriodicWorker instances build_default_wiring() constructs (real
    IncidentManager, real DriftMonitor dict, real OpsRepo) for a few intervals and
    proves neither records an error -- this is what the incident_escalation_tick
    contract-mismatch bug (calling escalate_if_overdue() with no args against the real
    class) and the state.repo/OpsRepo type mismatch would have surfaced immediately.

    Deliberately does NOT go through create_app/TestClient here: create_app's own
    lifespan would start these same worker objects running in TestClient's background
    thread, and running the same PeriodicWorker instances from a second event loop
    at the same time races on their shared _stop_event/errors state. Driving the
    workers directly (no app, no lifespan) is the correct isolated way to test them.
    """
    from mlops.svc.settings import Settings
    from mlops.svc.app import build_default_wiring

    db_path = str(tmp_path / "workers.db")
    settings = Settings(
        audit_secret="s" * 20, tokens=TOKENS, db_path=db_path,
        incident_escalation_interval_s=0.05, drift_evaluation_interval_s=0.05,
        _env_file=None,
    )
    wiring = build_default_wiring(settings)

    async def run_workers_briefly():
        async with anyio.create_task_group() as tg:
            stop_events = []
            for worker in wiring["workers"]:
                se = anyio.Event()
                stop_events.append(se)
                tg.start_soon(lambda w=worker, e=se: w.run(stop_event=e))
            await anyio.sleep(0.25)
            for se in stop_events:
                se.set()

    anyio.run(run_workers_briefly)

    for worker in wiring["workers"]:
        assert list(worker.errors) == [], f"{worker.name} recorded errors: {worker.errors}"


# ---------------------------------------------------------------- SBOM self-check

def test_sbom_build_self_check_catches_a_build_that_drifted_from_its_input(tmp_path, monkeypatch):
    """cli_ops.py's sbom build now schema-validates and round-trips its own output
    before ever writing it. Force build_cyclonedx_sbom to silently drop a component
    (simulating a hypothetical future regression) and prove the CLI refuses to hand
    out the corrupted SBOM instead of writing it silently."""
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli
    import mlops.svc.cli_ops as cli_ops_mod

    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pkg-a==1.0\npkg-b==2.0\n")

    real_build = cli_ops_mod.build_cyclonedx_sbom

    def dropping_build(components):
        sbom = real_build(components)
        sbom["components"] = sbom["components"][:1]  # drop one component
        return sbom

    monkeypatch.setattr(cli_ops_mod, "build_cyclonedx_sbom", dropping_build)

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(app, ["sbom", "build", str(req_file)])

    assert result.exit_code == 3
    assert "self-check failed" in result.output


def test_sbom_build_still_succeeds_normally_when_output_is_correct(tmp_path):
    """Byte-identical happy path: the self-check must not reject a genuinely
    correct build (regression guard for the test above)."""
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli

    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pkg-a==1.0\n")

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(app, ["sbom", "build", str(req_file)])

    assert result.exit_code == 0
    assert "pkg-a" in result.output


# ---------------------------------------------------------------- manifest_io CLI wiring

def test_lint_manifest_cli_command_lints_a_real_local_yaml_file(tmp_path):
    """manifest_io.load_and_lint had zero production callers (flagged in
    the review notes, never resolved). `mlops lint-manifest <file>` now calls it
    directly against a real file on disk."""
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli

    manifest = tmp_path / "pod.yaml"
    manifest.write_text(
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: demo\nspec:\n"
        "  containers:\n  - name: demo\n    image: nginx:latest\n"
    )

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(app, ["lint-manifest", str(manifest)])

    assert result.exit_code == 1  # the :latest tag is a block-severity finding
    assert "latest-tag" in result.output


def test_lint_manifest_cli_command_rejects_invalid_yaml(tmp_path):
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli

    manifest = tmp_path / "bad.yaml"
    manifest.write_text("{not: valid: yaml: [")

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(app, ["lint-manifest", str(manifest)])

    assert result.exit_code == 2


# ---------------------------------------------------------------- calibration-crosscheck CLI

def test_calibration_crosscheck_runs_a_real_cross_check(tmp_path):
    """calibration_sk.compare_to_calibrator (and IsotonicCrossCheck/PlattCrossCheck) had
    zero callers outside tests. `mlops policy calibration-crosscheck` now runs it for
    real against mlops.policy_calibration.evaluate()'s real output."""
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli
    import json as _json

    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_text(_json.dumps({
        "name": "cal", "version": 1,
        "rules": [{"id": "r1", "version": 1, "kind": "min_metric", "severity": "block",
                   "params": {"metric": "score", "min": 0.5}}],
    }))
    feedback_file = tmp_path / "feedback.json"
    feedback_file.write_text(_json.dumps([
        {"context": {"score": 0.1 * i}, "should_block": i < 5} for i in range(10)
    ]))

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(
        app, ["policy", "calibration-crosscheck", str(bundle_file), str(feedback_file), "--metric", "score"]
    )

    assert result.exit_code == 0
    assert "existing_result" in result.output
    assert "sklearn_crosscheck" in result.output


def test_calibration_crosscheck_orients_the_score_so_the_isotonic_fit_is_not_flat(tmp_path):
    """A min_metric rule blocks LOW scores, so the label (blocked) falls as the score
    rises; sklearn's isotonic regression is increasing and, fed the raw score, collapses
    to the constant base rate. The command negates the score for --blocks-when below,
    and it names the isotonic-vs-Platt difference as what it is."""
    from typer.testing import CliRunner
    from mlops.svc.cli import create_cli
    import json as _json

    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_text(_json.dumps({
        "name": "cal", "version": 1,
        "rules": [{"id": "r1", "version": 1, "kind": "min_metric", "severity": "block",
                   "params": {"metric": "score", "min": 0.5}}],
    }))
    feedback_file = tmp_path / "feedback.json"
    feedback_file.write_text(_json.dumps([
        {"context": {"score": i / 20}, "should_block": i < 10} for i in range(20)
    ]))

    runner = CliRunner()
    app = create_cli(lambda: None)
    result = runner.invoke(
        app, ["policy", "calibration-crosscheck", str(bundle_file), str(feedback_file), "--metric", "score"]
    )
    assert result.exit_code == 0
    out = _json.loads(result.output[result.output.index("{"):])
    assert len(set(out["sklearn_crosscheck"]["isotonic"])) > 1
    assert "isotonic_vs_platt_mean_abs_diff" in out["sklearn_crosscheck"]

    bad = runner.invoke(
        app,
        ["policy", "calibration-crosscheck", str(bundle_file), str(feedback_file),
         "--metric", "score", "--blocks-when", "sideways"],
    )
    assert bad.exit_code == 2


# ---------------------------------------------------------------- mlflow opt-in mirror

def test_mlflow_mirror_is_off_by_default():
    """Default settings (mlflow_tracking_uri=None) must never construct a bridge or
    attempt to import mlflow at all -- byte-identical to pre-mirror behavior."""
    from mlops.svc.settings import Settings
    from mlops.svc.app import create_app

    tokens = {ADMIN: {"name": "root", "role": "admin"}}
    settings = Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None)
    app = create_app(settings)
    assert app.state.mlops.mlflow_bridge is None


def test_mlflow_mirror_records_a_real_run_when_enabled(tmp_path):
    """Opt-in (mlflow_tracking_uri set): a real registration is mirrored into a real
    MLflow SQLite store and can be read back through the real MlflowBridge."""
    from mlops.svc.settings import Settings
    from mlops.svc.app import create_app

    mlflow_db = tmp_path / "mlflow.db"
    tokens = {ADMIN: {"name": "root", "role": "admin"}}
    settings = Settings(
        audit_secret="s" * 20, tokens=tokens,
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}", _env_file=None,
    )
    app = create_app(settings)
    client = TestClient(app)
    r = client.post(
        "/v1/models",
        json={"model_id": "m1", "version_id": "1", "artifact_hash": "b" * 64},
        headers=_auth(),
    )
    assert r.status_code == 201

    mirrored = app.state.mlops.mlflow_bridge.get_mirrored_run("m1:1")
    assert mirrored["params"]["artifact_hash"] == "b" * 64


# ---------------------------------------------------------------- lineage endpoints

def test_lineage_ancestors_and_blast_radius_reflect_a_real_transition(wired_client):
    client, _, _ = wired_client
    client.post(
        "/v1/models", json={"model_id": "m1", "version_id": "v1", "artifact_hash": "b" * 64}, headers=_auth()
    )
    client.post("/v1/models/m1/versions/v1/transition", json={"to_stage": "staging"}, headers=_auth())

    anc = client.get("/v1/lineage/m1/v1/ancestors", headers=_auth())
    assert anc.status_code == 200
    assert len(anc.json()["ancestors"]) == 1  # the registration node precedes the transition node

    br = client.get("/v1/lineage/m1/v1/blast-radius", headers=_auth())
    assert br.status_code == 200


def test_lineage_endpoint_404s_for_an_unknown_model(wired_client):
    client, _, _ = wired_client
    r = client.get("/v1/lineage/nope/v1/ancestors", headers=_auth())
    assert r.status_code == 404


# ---------------------------------------------------------------- real-DB audit resource bug

def test_policy_publish_activate_decide_write_real_audit_entries_with_a_real_db(tmp_path):
    """policy_store.publish/activate/rollback and PolicyDecisionPoint.decide all
    hardcoded resource="" in their audit.append calls. ChainedAuditStore (used by
    virtually every existing test's default :memory: db_path) never validates
    resource at all, so this was invisible until a real, file-backed
    DurableAuditStore (what any real deployment uses) rejected it outright with
    ValueError("resource cannot be empty") -- found by running the demo driver."""
    from mlops.svc.settings import Settings
    from mlops.svc.app import create_app

    db_path = str(tmp_path / "audit.db")
    settings = Settings(audit_secret="s" * 20, tokens=TOKENS, db_path=db_path, _env_file=None)
    app = create_app(settings)
    client = TestClient(app)

    published = client.post(
        "/v1/policy/publish",
        json={"name": "p", "version": 1, "rules": []},
        headers=_auth(),
    )
    assert published.status_code == 201, published.json()

    activated = client.post(
        f"/v1/policy/{published.json()['version']}/activate",
        json={"human_approved_by": "root"},
        headers=_auth(),
    )
    assert activated.status_code == 200, activated.json()

    decided = client.post(
        "/v1/policy/decide",
        json={"action": "deploy", "context": {"resource": "prod"}},
        headers=_auth(),
    )
    assert decided.status_code == 200, decided.json()


# ---------------------------------------------------------------- settings env whitelist

def test_load_settings_honors_env_vars_for_the_new_and_deployment_flags():
    """load_settings turns pydantic-settings' own env reading off and parses a fixed
    whitelist of MLOPS_* names by hand, so a field missing from that whitelist is
    silently ignored through the real entrypoint. MLOPS_METRICS_PUBLIC (added for
    deployment flexibility) was exactly that: it worked when Settings was built
    directly and did nothing under `python -m mlops.svc`."""
    from mlops.kernel import ValidationFailed
    from mlops.svc.settings import load_settings

    base = {"MLOPS_AUDIT_SECRET": "s" * 20, "MLOPS_TOKENS": '{"tk-admin-00001": {"name": "r", "role": "admin"}}'}

    assert load_settings(None, env=dict(base)).metrics_public is False
    assert load_settings(None, env={**base, "MLOPS_METRICS_PUBLIC": "true"}).metrics_public is True
    assert load_settings(None, env={**base, "MLOPS_METRICS_PUBLIC": "0"}).metrics_public is False
    s = load_settings(None, env={**base, "MLOPS_INCIDENT_ESCALATION_INTERVAL_S": "12.5"})
    assert s.incident_escalation_interval_s == 12.5
    with pytest.raises(ValidationFailed):
        load_settings(None, env={**base, "MLOPS_METRICS_PUBLIC": "maybe"})


# ---------------------------------------------------------------- demo driver

def test_demo_driver_runs_end_to_end_without_error():
    """The connect-pipeline sweep's own observable proof: exec/mlops/demo.py must
    run start to finish against one real, fully-wired app instance."""
    import subprocess
    import sys as _sys

    exec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
    result = subprocess.run(
        [_sys.executable, "-m", "mlops.demo"],
        cwd=exec_dir, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Done" in result.stdout


# ---------------------------------------------------------------- drift_frame integration

def test_drift_evaluation_tick_builds_a_real_pandas_report_when_given_frame_reports():
    """ext.drift_frame.DriftWindowFrame had zero callers outside tests. The drift
    worker now builds a real multi-window report alongside its baseline upsert."""
    from mlops.svc.workers import drift_evaluation_tick
    from mlops.drift import DriftMonitor

    monitor = DriftMonitor(list(range(100)))
    for v in range(40):
        monitor.observe(float(v))

    class FakeRepo:
        def upsert_drift_baseline(self, name, reference):
            pass

    reports = {}
    drift_evaluation_tick({"m1": monitor}, FakeRepo(), reports)
    assert "m1" in reports
    assert "daily_summary" in reports["m1"]
    assert "rolling_psi_tail" in reports["m1"]


def test_drift_evaluation_tick_upserts_a_real_list_not_a_deque(tmp_path):
    """Real DriftMonitor keeps _observations as a collections.deque; a real OpsRepo
    strictly requires a list for upsert_drift_baseline (only the frozen suite's fake
    monitor already stored a list, hiding this). Found by running the demo driver
    against a real, file-backed OpsRepo -- the exact scenario __main__.py now uses."""
    from mlops.svc.settings import Settings
    from mlops.svc.app import build_default_wiring
    from mlops.svc.workers import drift_evaluation_tick
    from mlops.drift import DriftMonitor

    db_path = str(tmp_path / "drift.db")
    settings = Settings(audit_secret="s" * 20, tokens=TOKENS, db_path=db_path, _env_file=None)
    wiring = build_default_wiring(settings)

    monitor = DriftMonitor([0.0] * 40)
    for v in range(40):
        monitor.observe(float(v) * 10)  # far outside reference -> non-ok level

    # Must not raise ValidationFailed("reference must be a list")
    drift_evaluation_tick({"m1": monitor}, wiring["ops_repo"])
    assert wiring["ops_repo"].get_drift_baseline("m1") is not None


def test_drift_evaluation_tick_still_works_with_frame_reports_omitted():
    """Byte-identical default: omitting frame_reports must not change the baseline
    upsert behavior the F3.6 frozen contract already pins."""
    from mlops.svc.workers import drift_evaluation_tick
    from mlops.drift import DriftMonitor

    monitor = DriftMonitor(list(range(100)))
    for v in range(40):
        monitor.observe(float(v))

    calls = []

    class FakeRepo:
        def upsert_drift_baseline(self, name, reference):
            calls.append(name)

    drift_evaluation_tick({"m1": monitor}, FakeRepo())
    assert calls == ["m1"]
