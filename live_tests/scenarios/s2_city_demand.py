"""S2: city demand forecasting. NYC TLC Yellow Taxi 2019 (tabular events, stratified-sampled from
an 86.7M-row, >1GB raw source) and Jena Climate 2009-2016 (time series). Two real regressors, gated
on real MAE via max_metric (RESEARCH-NOTES.md: no min_metric-on-a-negated-number invention), and two
SEPARATE named drift monitors (payment-type mix, temperature) per mlops-live-b-q5's univariate-only
finding -- never one bundled check.

Usage: python -m live_tests.scenarios.s2_city_demand --data <data dir> --seed N --out <dir>
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
    "tk-admin-live-s2-00001": {"name": "root", "role": "admin"},
    "tk-oper-live-s2-000001": {"name": "olga", "role": "operator"},
    "tk-view-live-s2-000001": {"name": "vic", "role": "viewer"},
}
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Explicit, stated design thresholds (mlops-live-a-q9: no literature threshold exists for either
# regression problem; these are MY choices, sized to what the EDA showed each target's scale to be).
TAXI_MAE_MAX = 3.0     # dollars of tip error
JENA_MAE_MAX = 3.0     # degrees C


def run(data_dir: Path, seed: int, out_dir: Path) -> History:
    h = History("S2_city_demand", seed, out_dir)
    taxi_dir, jena_dir = data_dir / "datasets" / "taxi", data_dir / "datasets" / "jena"

    taxi_df = pd.read_parquet(taxi_dir / "yellow_2019_sample.parquet")
    taxi_model, taxi_reg, taxi_train, taxi_test = train.train_taxi_regressor(
        taxi_df, train_months=list(range(1, 11)), test_months=[11, 12])
    h.record("train.taxi", "info", {"metric": taxi_model.metric_name, "train": taxi_model.train_metric,
                                     "test_future": taxi_model.test_metric, "extra": taxi_model.extra,
                                     "note": "MAE reported as-is (mlops-live-a-q9); max_metric gates it, not min_metric"})

    jena_df = pd.read_parquet(jena_dir / "jena_climate.parquet")
    jena_model, jena_reg, jena_train, jena_test = train.train_jena_regressor(jena_df)
    h.record("train.jena", "info", {"metric": jena_model.metric_name, "train": jena_model.train_metric,
                                     "test_future": jena_model.test_metric})

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
        h.try_step("rbac.viewer_cannot_register_dataset", lambda: c.expect_status(
            api.as_("viewer").post("/v1/datasets", json={"name": "x", "content_hash": "a" * 64, "rows": 1,
                                                          "dataset_schema": {}}), 403))

        taxi_ds, _ = h.try_step("dataset.register_taxi", lambda: c.register_dataset(api, "taxi", taxi_train))
        jena_ds, _ = h.try_step("dataset.register_jena", lambda: c.register_dataset(api, "jena", jena_train))
        h.try_step("dataset.unknown_version_rejected", lambda: c.expect_status(
            api.as_("operator").post("/v1/models", json={"model_id": "m-bad", "version_id": "1",
                                                          "artifact_hash": "b" * 64, "dataset_version": "no-such-version"}),
            (400, 404, 422)))

        taxi_base, _ = h.try_step("model.register_stage_taxi", lambda: c.register_and_stage(
            api, "taxi-tip-model", taxi_model.artifact_hash, taxi_ds, {"mae": taxi_model.test_metric}))
        jena_base, _ = h.try_step("model.register_stage_jena", lambda: c.register_and_stage(
            api, "jena-temp-model", jena_model.artifact_hash, jena_ds, {"mae": jena_model.test_metric}))

        def idem_replay():
            key = "s2-idem-0001"
            body = {"to_stage": "staging"}
            r1 = api.as_("operator").post(f"{taxi_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            r2 = api.as_("operator").post(f"{taxi_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            return {"first": r1.status_code, "replay": r2.status_code, "replayed_header": r2.headers.get("idempotent-replayed")}
        h.try_step("idempotency.replayed_transition", idem_replay)

        rules = [
            {"id": "taxi-mae", "version": 1, "kind": "max_metric", "severity": "block",
             "params": {"metric": "mae", "max": TAXI_MAE_MAX, "action": "stage.production", "resource": "taxi-tip-model"}},
            {"id": "jena-mae", "version": 1, "kind": "max_metric", "severity": "block",
             "params": {"metric": "mae", "max": JENA_MAE_MAX, "action": "stage.production", "resource": "jena-temp-model"}},
        ]
        h.try_step("policy.publish_activate", lambda: c.publish_and_activate(api, rules, 1))
        h.try_step("policy.promote_taxi_real_metric", lambda: c.promote(api, taxi_base, {"mae": taxi_model.test_metric}))
        h.try_step("policy.promote_jena_real_metric", lambda: c.promote(api, jena_base, {"mae": jena_model.test_metric}))
        # central-claim attack: targets jena_base, the model the gate just correctly denied above
        # (real MAE ~3.16 > 3.00) and which is still sitting in staging. The original run pointed
        # this at taxi_base, which was already production by this point, so the attack only hit the
        # state machine's production->production guard (409) and never tested the metric-forgery
        # path at all -- see experiment.md Finding 3.
        h.try_step("attack.self_attested_metric_lets_a_bad_model_through",
                   lambda: c.promote(api, jena_base, {"mae": 0.0}))

        # ------------------------------------------------------- two SEPARATE named drift monitors
        jan_payment = taxi_df[taxi_df["month"] == 1]["payment_type"].astype(float).tolist()[:5000]
        jul_payment = taxi_df[taxi_df["month"] == 7]["payment_type"].astype(float).tolist()[:300]
        taxi_report, _ = h.try_step("drift.taxi_payment_type_jan_vs_jul",
                                    lambda: c.drift_check(api, jan_payment, jul_payment, name="taxi-payment-type"))

        jan_temp = jena_df[jena_df["Date Time"].dt.month == 1]["T (degC)"].tolist()[:5000]
        jul_temp = jena_df[jena_df["Date Time"].dt.month == 7]["T (degC)"].tolist()[:300]
        jena_report, _ = h.try_step("drift.jena_temperature_jan_vs_jul",
                                    lambda: c.drift_check(api, jan_temp, jul_temp, name="jena-temperature"))

        h.record("drift.waiting_for_periodic_worker_tick", "info", {"interval_s": 1.0})
        time.sleep(3.0)
        for report, name in [(taxi_report, "taxi-payment-type"), (jena_report, "jena-temperature")]:
            h.try_step(f"drift.periodic_worker_persisted_baseline_readonly_db_check.{name}", lambda r=report, n=name: (
                c.read_drift_baseline_row_readonly(Path(srv.db_path), n) if r and r.get("level") != "ok"
                else {"skipped": "level was ok, no baseline expected"}))
        h.record("finding.same_as_s3_drift_frame_reports_has_no_http_route", "info", {
            "reference": "see live_tests/experiments/s3_land_cover/experiment.md -- identical gap, not re-argued here"})

        opened = None
        for report, feature in [(taxi_report, "payment_type"), (jena_report, "temperature")]:
            if report and report.get("level") != "ok" and opened is None:
                opened, _ = h.try_step(f"incident.open_from_{feature}_drift", lambda r=report, f=feature: api.as_("operator").post(
                    "/v1/incidents", json={"title": f"{f} drift (Jan vs Jul)",
                                           "signal": {"drift_level": r["level"], "feature_psi": r["psi_adjusted"]}}).json())
        if opened is None:
            h.record("incident.open_from_seasonal_drift", "info", {"skipped": "neither monitor reported non-ok"})
        elif "incident_id" in opened:
            from mlops.incidents import CHECKLIST

            def checklist():
                # Same driver bug S1's re-run found and fixed: /v1/incidents/{id}/steps requires
                # both "step" and "note"; /resolve requires the "postmortem" key even when its
                # value is None. This copy of the pattern (S2 has its own inline checklist(), not
                # shared via common.py) still had the original bug -- fixed here the same way.
                out = []
                for step in CHECKLIST:
                    out.append((step, api.as_("operator").post(f"/v1/incidents/{opened['incident_id']}/steps",
                                                                json={"step": step, "note": f"real check: {step}"}).status_code))
                out.append(("resolve", api.as_("operator").post(f"/v1/incidents/{opened['incident_id']}/resolve",
                                                                 json={"postmortem": None}).status_code))
                return out
            h.try_step("incident.checklist_and_resolve", checklist)

        # ------------------------------------------------------- rollback (not exercised by S1/S3): promote a
        # second taxi version, then roll back to the first -- drives model_stages.ModelRegistry.rollback for real.
        def rollback_flow():
            r = api.as_("operator").post("/v1/models", json={"model_id": "taxi-tip-model", "version_id": "2",
                                                              "artifact_hash": "c" * 64, "dataset_version": taxi_ds})
            r.raise_for_status()
            base2 = "/v1/models/taxi-tip-model/versions/2"
            api.as_("operator").post(f"{base2}/validate", json={"evidence": {"mae": taxi_model.test_metric}}).raise_for_status()
            api.as_("operator").post(f"{base2}/transition", json={"to_stage": "staging"}).raise_for_status()
            promo = api.as_("admin").post(f"{base2}/transition", json={"to_stage": "production",
                                                                       "context": {"mae": taxi_model.test_metric}})
            rb = api.as_("admin").post("/v1/models/taxi-tip-model/rollback", json={"context": {"mae": taxi_model.test_metric}})
            return {"promote_v2": promo.status_code, "rollback": rb.status_code, "rollback_body": c.safe_json(rb)}
        h.try_step("model.rollback_after_second_version", rollback_flow)

        h.try_step("audit.export_verify_and_tamper_a_copy", lambda: c.audit_roundtrip(api))
        h.try_step("lineage.blast_radius_and_ancestors", lambda: c.lineage(api, "taxi-tip-model"))
        h.try_step("sbom.build_and_self_validate", lambda: c.sbom(REPO_ROOT))
        h.try_step("lint.model_server_manifest", lambda: c.lint("taxi-tip-model"))
        h.try_step("telemetry.metrics_move", lambda: c.metrics_moved(api))
        h.try_step("sdk.policy_versions", lambda: c.sdk_roundtrip(srv.base_url, "tk-view-live-s2-000001"))
        h.try_step("mlflow.mirror_has_runs", lambda: c.mlflow_run_count(mlflow_db))

        # No real classifier in this scenario (both models are regressors) -- calibration-crosscheck
        # takes a binary y/score pair, which doesn't exist here. Recorded, not faked.
        h.record("calibration.crosscheck_cli_real_scores", "info",
                {"skipped": "S2 has no classifier; calibration-crosscheck needs binary y/score pairs"})

        h.record("durability.pre_kill_snapshot", "info", {
            "active_policy": c.safe_json(api.as_("viewer").get("/v1/policy/active")),
            "taxi_stage": c.safe_json(api.as_("viewer").get("/v1/models/taxi-tip-model"))})
        api.close()
        srv.kill9()
        srv2 = LiveServer(out_dir / "server", tokens=TOKENS, port=srv.port,
                          extra_env={"MLOPS_MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}"})
        _, outcome = h.try_step("durability.restart_after_kill9", lambda: srv2.start(timeout_s=60))
        if outcome == "ok":
            api2 = c.Api(srv2.base_url, TOKENS)
            h.try_step("durability.policy_survived", lambda: c.safe_json(api2.as_("viewer").get("/v1/policy/active")))
            h.try_step("durability.model_state_after_restart",
                       lambda: c.safe_json(api2.as_("viewer").get("/v1/models/taxi-tip-model")))
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
