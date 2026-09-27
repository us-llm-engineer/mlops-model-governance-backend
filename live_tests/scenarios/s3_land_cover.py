"""S3: land-cover monitoring. Covertype (tabular cartographic, split by Elevation -- its own
declared drift_order) and EuroSAT RGB (images). Exercises the periodic drift-evaluation worker for
real (short interval) and checks, per RESEARCH-NOTES.md mlops-live-b-q5, whether a single-feature
drift check on Elevation is the only public way to watch this system, since DriftMonitor is
univariate and drift_frame_reports/get_drift_baseline have no HTTP route (recorded as a finding,
not routed around).

Usage: python -m live_tests.scenarios.s3_land_cover --data <data dir> --seed N --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from live_tests.harness import History, LiveServer  # noqa: E402
from live_tests import train  # noqa: E402
from live_tests.scenarios import common as c  # noqa: E402

TOKENS = {
    "tk-admin-live-s3-00001": {"name": "root", "role": "admin"},
    "tk-oper-live-s3-000001": {"name": "olga", "role": "operator"},
    "tk-view-live-s3-000001": {"name": "vic", "role": "viewer"},
}
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(data_dir: Path, seed: int, out_dir: Path) -> History:
    h = History("S3_land_cover", seed, out_dir)
    cover_dir, euro_dir = data_dir / "datasets" / "covertype", data_dir / "datasets" / "eurosat"

    cover_df = pd.read_parquet(cover_dir / "covertype.parquet")
    cover_model, cover_clf, cover_train, cover_test = train.train_covertype_classifier(cover_df)
    h.record("train.covertype", "info", {"metric": cover_model.metric_name, "train": cover_model.train_metric,
                                          "test_future": cover_model.test_metric, "extra": cover_model.extra})

    euro_df = pd.read_parquet(euro_dir / "eurosat_rgb.parquet")
    euro_model, euro_clf, euro_train, euro_test = train.train_eurosat_classifier(euro_df, seed=seed)
    h.record("train.eurosat", "info", {"metric": euro_model.metric_name, "train": euro_model.train_metric,
                                        "test_future": euro_model.test_metric})

    mlflow_db = out_dir / "server" / "mlflow.db"
    srv = LiveServer(out_dir / "server", tokens=TOKENS,
                     extra_env={"MLOPS_MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}",
                                "MLOPS_DRIFT_EVALUATION_INTERVAL_S": "1.0"})
    _, outcome = h.try_step("server.start", lambda: srv.start(timeout_s=60))
    if outcome != "ok":
        h.record("run.abort", "crash", {"reason": "server failed to start"})
        return h
    api = c.Api(srv.base_url, TOKENS)

    try:
        h.try_step("rbac.viewer_cannot_register_model", lambda: c.expect_status(
            api.as_("viewer").post("/v1/models", json={"model_id": "x", "version_id": "1",
                                                        "artifact_hash": "a" * 64}), 403))

        cover_ds, _ = h.try_step("dataset.register_covertype", lambda: c.register_dataset(api, "covertype", cover_train))
        euro_ds, _ = h.try_step("dataset.register_eurosat", lambda: c.register_dataset(api, "eurosat", euro_train))

        cover_base, _ = h.try_step("model.register_stage_covertype", lambda: c.register_and_stage(
            api, "covertype-classifier", cover_model.artifact_hash, cover_ds, {"macro_f1": cover_model.test_metric}))
        euro_base, _ = h.try_step("model.register_stage_eurosat", lambda: c.register_and_stage(
            api, "eurosat-classifier", euro_model.artifact_hash, euro_ds, {"macro_f1": euro_model.test_metric}))

        def idem_replay():
            key = "s3-idem-0001"
            body = {"to_stage": "staging"}
            r1 = api.as_("operator").post(f"{cover_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            r2 = api.as_("operator").post(f"{cover_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            return {"first": r1.status_code, "replay": r2.status_code, "replayed_header": r2.headers.get("idempotent-replayed")}
        h.try_step("idempotency.replayed_transition", idem_replay)

        rules = [
            {"id": "cover-f1", "version": 1, "kind": "min_metric", "severity": "block",
             "params": {"metric": "macro_f1", "min": 0.6, "action": "stage.production", "resource": "covertype-classifier"}},
            {"id": "euro-f1", "version": 1, "kind": "min_metric", "severity": "block",
             "params": {"metric": "macro_f1", "min": 0.6, "action": "stage.production", "resource": "eurosat-classifier"}},
        ]
        h.try_step("policy.publish_activate", lambda: c.publish_and_activate(api, rules, 1))
        h.try_step("policy.promote_covertype_real_metric",
                   lambda: c.promote(api, cover_base, {"macro_f1": cover_model.test_metric}))
        h.try_step("policy.promote_eurosat_real_metric",
                   lambda: c.promote(api, euro_base, {"macro_f1": euro_model.test_metric}))
        # central-claim attack: the original run's version claimed a LOWER fake score (0.0) on a
        # model already denied by its real, lower score (0.28) -- both honest and forged attempts
        # were denied for the same reason, testing nothing about metric forgery (Finding 4). The
        # real attack needs a forged score ABOVE the gate (0.6) on the same already-validated,
        # already-denied model.
        h.try_step("attack.self_attested_metric_lets_a_bad_model_through",
                   lambda: c.promote(api, cover_base, {"macro_f1": 0.99}))

        # ------------------------------------------------------- drift: single-feature (Elevation), real worker tick
        elev_ref = cover_train["Elevation"].tolist()[:5000]
        elev_window = cover_test["Elevation"].tolist()[:300]
        report, _ = h.try_step("drift.elevation_named_monitor",
                               lambda: c.drift_check(api, elev_ref, elev_window, name="covertype-elevation"))

        h.record("drift.waiting_for_periodic_worker_tick", "info", {"interval_s": 1.0})
        time.sleep(3.0)
        baseline, _ = h.try_step("drift.periodic_worker_persisted_baseline_readonly_db_check", lambda: (
            c.read_drift_baseline_row_readonly(Path(srv.db_path), "covertype-elevation")
            if report and report.get("level") != "ok" else {"skipped": "level was ok, no baseline expected"}))
        # Finding 3 (no HTTP route for the persisted baseline) is fixed since the original run --
        # GET /v1/drift/baselines/{name} now exists. Confirm a real client can retrieve it.
        h.try_step("drift.baseline_retrievable_over_http", lambda: (
            c.read_drift_baseline_http(api, "covertype-elevation")
            if report and report.get("level") != "ok" else {"skipped": "level was ok, no baseline expected"}))

        h.record("finding.drift_frame_reports_has_no_http_route", "info", {
            "grep": "state.drift_frame_reports is set in svc/app.py and populated by workers.drift_evaluation_tick, "
                    "but no route in svc/app.py or svc/routers/*.py ever reads it",
            "consequence": "the richer multi-window DriftWindowFrame report the worker builds is invisible to any real client -- "
                           "still true; only get_drift_baseline got a route, this one is a separate, smaller open question"})

        if report and report.get("level") != "ok":
            incident, _ = h.try_step("incident.open_from_elevation_drift", lambda: api.as_("operator").post(
                "/v1/incidents", json={"title": "Elevation drift", "signal": {"drift_level": report["level"],
                                                                              "feature_psi": report["psi_adjusted"]}}
            ).json())
            if incident and "incident_id" in incident:
                from mlops.incidents import CHECKLIST

                def checklist():
                    # Same body-shape bug S1's and S2's re-runs found and fixed (own copy of the
                    # pattern here, not shared via common.py): /steps needs "note", /resolve needs
                    # "postmortem" even when None. Fixed here before it ever ran wrong.
                    out = []
                    for step in CHECKLIST:
                        out.append((step, api.as_("operator").post(
                            f"/v1/incidents/{incident['incident_id']}/steps",
                            json={"step": step, "note": f"real check: {step}"}).status_code))
                    out.append(("resolve", api.as_("operator").post(
                        f"/v1/incidents/{incident['incident_id']}/resolve", json={"postmortem": None}).status_code))
                    return out
                h.try_step("incident.checklist_and_resolve", checklist)
        else:
            h.record("incident.open_from_elevation_drift", "info", {"skipped": "no drift detected"})

        h.try_step("audit.export_verify_and_tamper_a_copy", lambda: c.audit_roundtrip(api))
        h.try_step("lineage.blast_radius_and_ancestors", lambda: c.lineage(api, "covertype-classifier"))
        h.try_step("sbom.build_and_self_validate", lambda: c.sbom(REPO_ROOT))
        h.try_step("lint.model_server_manifest", lambda: c.lint("covertype-classifier"))
        h.try_step("telemetry.metrics_move", lambda: c.metrics_moved(api))
        h.try_step("sdk.policy_versions", lambda: c.sdk_roundtrip(srv.base_url, "tk-view-live-s3-000001"))
        h.try_step("mlflow.mirror_has_runs", lambda: c.mlflow_run_count(mlflow_db))

        h.record("durability.pre_kill_snapshot", "info", {
            "active_policy": c.safe_json(api.as_("viewer").get("/v1/policy/active")),
            "cover_stage": c.safe_json(api.as_("viewer").get("/v1/models/covertype-classifier"))})
        api.close()
        srv.kill9()
        srv2 = LiveServer(out_dir / "server", tokens=TOKENS, port=srv.port,
                          extra_env={"MLOPS_MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}"})
        _, outcome = h.try_step("durability.restart_after_kill9", lambda: srv2.start(timeout_s=60))
        if outcome == "ok":
            api2 = c.Api(srv2.base_url, TOKENS)
            h.try_step("durability.policy_survived", lambda: c.safe_json(api2.as_("viewer").get("/v1/policy/active")))
            h.try_step("durability.model_state_after_restart",
                       lambda: c.safe_json(api2.as_("viewer").get("/v1/models/covertype-classifier")))
            api2.close()
            srv2.stop()
    finally:
        try:
            api.close()
        except Exception:
            pass
        try:
            srv.stop()
        except Exception:
            pass
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    h = run(Path(a.data), a.seed, out)
    (out / "summary.json").write_text(json.dumps(h.summary(), indent=2))
    print(json.dumps(h.summary(), indent=2))


if __name__ == "__main__":
    main()
