"""Terminal-only, end-to-end demo of the real MLOps control-plane backend.

Run: cd .mlops-control-plane && PYTHONPATH=exec python3 -m mlops.demo

Builds the exact same app serve()/`python -m mlops.svc` build (via
svc.app.build_default_wiring), then drives it through a real registration ->
policy -> lint -> drift -> incident -> audit -> SBOM -> lineage walk using only
real call paths -- no mocks. This is the connect-pipeline sweep's own proof
that the pipeline is actually connected end to end, not just individually
correct module by module: every step below prints the real object/response it
got back, not a description of what it would print.

Each numbered section is independent enough to read on its own; run the whole
file top to bottom for the full walkthrough.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from mlops.svc.app import build_default_wiring, create_app
from mlops.svc.settings import Settings


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _show(label: str, value) -> None:
    if isinstance(value, (dict, list)):
        print(f"-- {label} --\n{json.dumps(value, indent=2, sort_keys=True, default=str)}")
    else:
        print(f"-- {label} --\n{value}")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="mlops-demo-")
    db_path = str(Path(workdir) / "mlops.db")
    mlflow_uri = f"sqlite:///{Path(workdir) / 'mlflow.db'}"

    tokens = {
        "tk-admin-0000000000001": {"name": "root", "role": "admin"},
        "tk-viewer-000000000001": {"name": "viewer1", "role": "viewer"},
    }
    settings = Settings(
        audit_secret="demo-secret-key-0123456789",
        tokens=tokens,
        db_path=db_path,
        mlflow_tracking_uri=mlflow_uri,
        incident_escalation_interval_s=1.0,
        drift_evaluation_interval_s=1.0,
        _env_file=None,
    )

    _hr("0. Building the real, fully-wired app (same wiring serve()/__main__.py use)")
    wiring = build_default_wiring(settings)
    app = create_app(settings, **wiring)
    client = TestClient(app)
    ADMIN = {"Authorization": "Bearer tk-admin-0000000000001"}
    print(f"work dir: {workdir}")
    print("extensions mounted:", len(wiring["extensions"]))
    print("workers configured:", [w.name for w in wiring["workers"]])

    # ------------------------------------------------------------------ 1. Dataset + model lifecycle
    _hr("1. Register a dataset, then a model version validated against it")
    ds = client.post(
        "/v1/datasets",
        json={"name": "training-set-v1", "content_hash": "a" * 64, "rows": 50_000,
              "dataset_schema": {"type": "object", "properties": {"label": {"type": "integer"}}}},
        headers=ADMIN,
    ).json()
    _show("dataset registered", ds)

    model = client.post(
        "/v1/models",
        json={"model_id": "fraud-detector", "version_id": "1", "artifact_hash": "b" * 64,
              "dataset_version": ds["version_id"]},
        headers=ADMIN,
    ).json()
    _show("model registered (dataset_version validated for real)", model)

    bad_ds = client.post(
        "/v1/models",
        json={"model_id": "fraud-detector", "version_id": "2", "artifact_hash": "c" * 64,
              "dataset_version": "not-a-real-dataset"},
        headers=ADMIN,
    )
    _show("registration with an unknown dataset_version (real validation, not accepted)",
          {"status": bad_ds.status_code, "body": bad_ds.json()})

    staged = client.post(
        "/v1/models/fraud-detector/versions/1/transition",
        json={"to_stage": "staging"}, headers=ADMIN,
    ).json()
    _show("transitioned to staging", staged)

    # ------------------------------------------------------------------ 2. Policy: publish, decide (scoped rule)
    _hr("2. Publish a scoped policy and evaluate real decisions")
    policy_rules = [
        {"id": "admin-only-deploy", "version": 1, "kind": "role_in", "severity": "block",
         "params": {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"}},
    ]
    policy = client.post(
        "/v1/policy/publish",
        json={"name": "deploy-gate", "version": 1, "rules": policy_rules},
        headers=ADMIN,
    ).json()
    _show("policy published", policy)
    client.post(f"/v1/policy/{policy['version']}/activate", json={"human_approved_by": "root"}, headers=ADMIN)

    decision = client.post(
        "/v1/policy/decide",
        json={"action": "deploy", "context": {"role": "admin", "resource": "prod"}},
        headers=ADMIN,
    ).json()
    _show("admin deploying to prod", decision)

    # The rule is scoped (action="deploy", resource="prod"): it applies to that request
    # only. A viewer is denied there; the same viewer on another resource is not touched.
    denied = client.post(
        "/v1/policy/decide",
        json={"action": "deploy", "context": {"role": "viewer", "resource": "prod"}},
        headers=ADMIN,
    ).json()
    _show("viewer deploying to prod (rule applies, fails)", {"allow": denied.get("allow"), "reasons": denied.get("reasons")})
    elsewhere = client.post(
        "/v1/policy/decide",
        json={"action": "deploy", "context": {"role": "viewer", "resource": "dev"}},
        headers=ADMIN,
    ).json()
    _show("viewer deploying to dev (rule out of scope)", {"allow": elsewhere.get("allow"), "reasons": elsewhere.get("reasons")})

    # ------------------------------------------------------------------ 3. Manifest lint (local, no server round trip)
    _hr("3. Lint a raw Kubernetes manifest (manifest_io.load_and_lint, local call)")
    from mlops.svc.manifest_io import load_and_lint

    manifest_yaml = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: fraud-detector\nspec:\n"
        "  containers:\n  - name: server\n    image: fraud-detector:latest\n"
    )
    docs, findings = load_and_lint(manifest_yaml)
    _show("lint findings", [f if isinstance(f, dict) else
                             {"rule_id": f.rule_id, "severity": f.severity, "message": f.message}
                             for f in findings])

    # ------------------------------------------------------------------ 4. Drift check + worker tick + pandas report
    _hr("4. Named drift check, then run the real periodic worker tick once")
    import random
    random.seed(0)
    reference = [random.gauss(0, 1) for _ in range(200)]
    window = [random.gauss(0.8, 1) for _ in range(60)]  # shifted distribution -> real drift
    drift = client.post(
        "/v1/drift/check",
        json={"reference": reference, "window": window, "name": "fraud-detector-scores"},
        headers=ADMIN,
    ).json()
    _show("one-shot drift check", drift)

    from mlops.svc.workers import drift_evaluation_tick

    drift_evaluation_tick(app.state.mlops.drift_monitors, wiring["ops_repo"], wiring["drift_frame_reports"])
    report = wiring["drift_frame_reports"].get("fraud-detector-scores")
    if report:
        _show("real pandas multi-window report (ext.drift_frame.DriftWindowFrame)",
              {"rolling_psi_tail": report["rolling_psi_tail"], "daily_summary_rows": len(report["daily_summary"])})
    else:
        print("(fewer than 30 observations recorded on the persisted monitor -- no frame report yet)")

    # ------------------------------------------------------------------ 5. Incident lifecycle + escalation tick
    _hr("5. Open an incident, complete a step, run the real escalation tick")
    incident = client.post(
        "/v1/incidents",
        json={"title": "elevated fraud false-positive rate", "signal": {"accuracy_drop_pct": 12}},
        headers=ADMIN,
    ).json()
    _show("incident opened", incident)
    step = client.post(
        f"/v1/incidents/{incident['incident_id']}/steps",
        json={"step": "detection", "note": "false-positive rate confirmed on dashboard"}, headers=ADMIN,
    )
    _show("checklist step 'detection' (steps must go in order)", {"status": step.status_code, "body": step.json()})
    out_of_order = client.post(
        f"/v1/incidents/{incident['incident_id']}/steps",
        json={"step": "root_cause", "note": "skipping ahead"}, headers=ADMIN,
    )
    _show("skipping ahead to 'root_cause' is refused", {"status": out_of_order.status_code, "body": out_of_order.json()})

    from mlops.svc.workers import incident_escalation_tick

    escalated = incident_escalation_tick(app.state.mlops.incidents_mgr)
    print(f"-- real incident_escalation_tick() result --\n{escalated} "
          f"incident(s) escalated this tick (0 is correct: nothing is overdue yet)")

    # ------------------------------------------------------------------ 6. Audit chain tail/verify
    _hr("6. Tail and verify the real hash-chained audit log")
    export = wiring["audit"].export()
    head = wiring["audit"].head()
    verify = client.post("/v1/audit/verify", json={"export": export, "head": head}, headers=ADMIN).json()
    _show(f"audit chain ({len(export)} entries)", {"head": head, "verify_result": verify})

    # ------------------------------------------------------------------ 7. SBOM build + self-check
    _hr("7. Build a CycloneDX SBOM and self-validate it before it would be written out")
    from mlops.ext.cyclonedx_sbom import build_cyclonedx_sbom, validate_cyclonedx_sbom, sbom_matches_requirements

    components = [("fastapi", "0.115.0"), ("pydantic", "2.9.0")]
    sbom = build_cyclonedx_sbom(components)
    validate_cyclonedx_sbom(sbom)  # raises on schema mismatch
    matches = sbom_matches_requirements(sbom, components)
    _show("SBOM self-check", {"schema_valid": True, "matches_requirements": matches,
                               "component_count": len(sbom["components"])})

    # ------------------------------------------------------------------ 8. Lineage
    _hr("8. Query the real in-memory lineage graph built by every register/transition above")
    ancestors = client.get("/v1/lineage/fraud-detector/1/ancestors", headers=ADMIN).json()
    blast = client.get("/v1/lineage/fraud-detector/1/blast-radius", headers=ADMIN).json()
    # Both endpoints query the CURRENT node for (model_id, version_id) -- the
    # staging-transition node, since that's what _mirror_record last recorded.
    _show("ancestors of the current (staging) node -- reaches the earlier registration node", ancestors)
    _show("blast radius of the current (staging) node -- empty: nothing newer descends from it yet", blast)

    # ------------------------------------------------------------------ 9. MLflow mirror
    _hr("9. Read back the MLflow mirror of the registration above")
    bridge = app.state.mlops.mlflow_bridge
    if bridge is not None:
        _show("mirrored MLflow run", bridge.get_mirrored_run("fraud-detector:1"))
    else:
        print("(mlflow_tracking_uri not configured)")

    _hr("Done")
    print(f"Everything above ran against one real, in-process app instance -- work dir: {workdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
