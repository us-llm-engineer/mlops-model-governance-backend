"""S1: payments risk desk. Two real datasets: ULB credit-card fraud (tabular, imbalanced) and
AG News (text). Drives the real server through every component in README's "system components"
table, entirely through HTTP (SDK methods where they exist), and records a full history for the
findings write-up. Per RESEARCH-NOTES.md: time-forward split (mlops-live-a-q2), AUPRC not accuracy
for the imbalanced target (mlops-live-a-q6), drift windows >= 250 samples (mlops-live-b-q3), and a
tight-loop drift-check burst to probe alert fatigue (mlops-live-b-q8).

Usage: python -m live_tests.scenarios.s1_payments_risk --data <data dir> --seed N --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "exec"))  # so in-process `from mlops...` imports resolve
from live_tests.harness import History, LiveServer  # noqa: E402
from live_tests import train  # noqa: E402

TOKENS = {
    "tk-admin-live-s1-00001": {"name": "root", "role": "admin"},
    "tk-oper-live-s1-000001": {"name": "olga", "role": "operator"},
    "tk-view-live-s1-000001": {"name": "vic", "role": "viewer"},
}


def content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


class Api:
    """Thin wrapper: one client per role, records status codes for the caller to classify."""

    def __init__(self, base_url: str):
        self.clients = {role: httpx.Client(base_url=base_url, timeout=30.0,
                                           headers={"Authorization": f"Bearer {t}"})
                        for t, m in TOKENS.items() for role in [m["role"]]}

    def as_(self, role: str) -> httpx.Client:
        return self.clients[role]

    def close(self):
        for c in self.clients.values():
            c.close()


def run(data_dir: Path, seed: int, out_dir: Path) -> History:
    h = History("S1_payments_risk", seed, out_dir)
    fraud_dir, agnews_dir = data_dir / "datasets" / "fraud", data_dir / "datasets" / "agnews"

    # ---------------------------------------------------------------- 0. real training (time-forward)
    fraud_df = pd.read_parquet(fraud_dir / "creditcard.parquet")
    fraud_model, fraud_clf, fraud_train, fraud_test = train.train_fraud_classifier(fraud_df)
    h.record("train.fraud", "info", {"metric": fraud_model.metric_name, "train": fraud_model.train_metric,
                                      "test_future": fraud_model.test_metric, "extra": fraud_model.extra})

    news_train = pd.read_parquet(agnews_dir / "train.parquet")
    news_test = pd.read_parquet(agnews_dir / "test.parquet")
    news_model, news_clf, news_win_a, news_win_b = train.train_agnews_classifier(news_train, news_test)
    h.record("train.agnews", "info", {"metric": news_model.metric_name, "window_a": news_model.train_metric,
                                       "window_b": news_model.test_metric})

    # ---------------------------------------------------------------- server
    srv = LiveServer(out_dir / "server", tokens=TOKENS,
                     extra_env={"MLOPS_MLFLOW_TRACKING_URI": f"sqlite:///{out_dir / 'server' / 'mlflow.db'}"})
    _, outcome = h.try_step("server.start", lambda: srv.start(timeout_s=60))
    if outcome != "ok":
        h.record("run.abort", "crash", {"reason": "server failed to start"})
        return h
    api = Api(srv.base_url)

    try:
        # ------------------------------------------------------- 1. auth + RBAC
        h.try_step("rbac.viewer_cannot_register_dataset", lambda: _expect_status(
            api.as_("viewer").post("/v1/datasets", json={"name": "x", "content_hash": "a" * 64, "rows": 1,
                                                          "dataset_schema": {}}), 403))
        h.try_step("rbac.operator_cannot_activate_policy", lambda: _expect_status(
            api.as_("operator").post("/v1/policy/1/activate", json={}), (403, 404)))

        # ------------------------------------------------------- 3. dataset registry (both kinds)
        def register_dataset(role, name, df):
            r = api.as_(role).post("/v1/datasets", json={
                "name": name, "content_hash": content_hash(df), "rows": int(len(df)),
                "dataset_schema": {"type": "object", "properties": {c: {} for c in df.columns[:5]}}})
            r.raise_for_status()
            return r.json()["version_id"]

        fraud_ds_id, _ = h.try_step("dataset.register_fraud", lambda: register_dataset("operator", "fraud", fraud_train))
        news_ds_id, _ = h.try_step("dataset.register_agnews", lambda: register_dataset("operator", "agnews", news_train))
        h.try_step("dataset.unknown_version_rejected", lambda: _expect_status(
            api.as_("operator").post("/v1/models", json={"model_id": "m-bad", "version_id": "1",
                                                          "artifact_hash": "b" * 64, "dataset_version": "no-such-version"}),
            (400, 404, 422)))

        # ------------------------------------------------------- 4. model registry lifecycle (per model)
        def register_and_stage(model_id, artifact_hash, dataset_version, evidence):
            r = api.as_("operator").post("/v1/models", json={
                "model_id": model_id, "version_id": "1", "artifact_hash": artifact_hash,
                "dataset_version": dataset_version})
            r.raise_for_status()
            base = f"/v1/models/{model_id}/versions/1"
            api.as_("operator").post(f"{base}/validate", json={"evidence": evidence}).raise_for_status()
            api.as_("operator").post(f"{base}/transition", json={"to_stage": "staging"}).raise_for_status()
            return base

        fraud_base, _ = h.try_step("model.register_stage_fraud", lambda: register_and_stage(
            "fraud-detector", fraud_model.artifact_hash, fraud_ds_id, {"auprc": fraud_model.test_metric}))
        news_base, _ = h.try_step("model.register_stage_agnews", lambda: register_and_stage(
            "news-classifier", news_model.artifact_hash, news_ds_id, {"macro_f1": news_model.test_metric}))

        # ------------------------------------------------------- 2. idempotency
        def idem_replay():
            key = "s1-idem-key-0001"
            body = {"to_stage": "staging"}
            r1 = api.as_("operator").post(f"{fraud_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            r2 = api.as_("operator").post(f"{fraud_base.replace('/versions/1','')}/versions/1/transition",
                                          json=body, headers={"Idempotency-Key": key})
            return {"first": r1.status_code, "replay": r2.status_code, "replayed_header": r2.headers.get("idempotent-replayed")}
        h.try_step("idempotency.replayed_transition", idem_replay)

        # ------------------------------------------------------- 5. policy (real AUPRC/F1 vs a rule)
        def publish_and_activate(rules, version):
            r = api.as_("admin").post("/v1/policy/publish", json={"name": "s1-gate", "version": version, "rules": rules})
            r.raise_for_status()
            api.as_("admin").post(f"/v1/policy/{r.json()['version']}/activate",
                                  json={"human_approved_by": "root"}).raise_for_status()
            return r.json()["version"]

        # Design decision (a-q9): no literature threshold exists; using AUPRC >= 0.5 and macro-F1 >= 0.8
        # as an explicit, stated choice, scoped to stage.production so staging is never blocked.
        rules = [
            {"id": "fraud-auprc", "version": 1, "kind": "min_metric", "severity": "block",
             "params": {"metric": "auprc", "min": 0.5, "action": "stage.production", "resource": "fraud-detector"}},
            {"id": "news-f1", "version": 1, "kind": "min_metric", "severity": "block",
             "params": {"metric": "macro_f1", "min": 0.8, "action": "stage.production", "resource": "news-classifier"}},
        ]
        h.try_step("policy.publish_activate", lambda: publish_and_activate(rules, 1))

        def promote(model_id, base, metric_key, value):
            r = api.as_("admin").post(f"{base}/transition", json={"to_stage": "production",
                                                                   "context": {metric_key: value}})
            return {"status": r.status_code, "body": _safe_json(r)}
        h.try_step("policy.promote_fraud_real_metric",
                   lambda: promote("fraud-detector", fraud_base, "auprc", fraud_model.test_metric))
        h.try_step("policy.promote_news_real_metric",
                   lambda: promote("news-classifier", news_base, "macro_f1", news_model.test_metric))
        # central-claim attack: self-attested metric -- a caller can just lie
        h.try_step("policy.attack_self_attested_metric_lets_a_bad_model_through",
                   lambda: promote("fraud-detector", fraud_base, "auprc", 0.999))

        # ------------------------------------------------------- 7/8. drift + incidents (real windows)
        v_train = fraud_train["V14"].tolist()
        v_ref = v_train[:5000]
        window_ok = v_train[5000:5300]                                    # >= 250, per b-q3
        # Finding 3 remedy (re-run): the original window_small was a later slice of the same
        # non-stationary series, which confounded the cry-wolf attack with genuine drift.
        # Resample with replacement FROM the reference itself so this baseline is honestly
        # driftless by construction.
        window_small = np.random.default_rng(seed).choice(v_ref, size=30, replace=True).tolist()
        window_shifted = (fraud_test["V14"] + 8.0).tolist()[:300]           # a real, injected shift

        def drift_check(client_role, ref, window, name=None):
            body = {"reference": ref, "window": window}
            if name:
                body["name"] = name
            r = api.as_(client_role).post("/v1/drift/check", json=body)
            r.raise_for_status()
            return r.json()

        base_report, _ = h.try_step("drift.small_window_v14", lambda: drift_check("operator", v_ref, window_small))
        ok_report, _ = h.try_step("drift.recommended_window_v14_no_shift", lambda: drift_check("operator", v_ref, window_ok))
        shift_report, _ = h.try_step("drift.recommended_window_v14_injected_shift",
                                     lambda: drift_check("operator", v_ref, window_shifted, name="fraud-v14"))

        def open_incident_from_drift(report):
            level = report["level"]
            signal = {"drift_level": level, "feature_psi": report["psi_adjusted"]}
            r = api.as_("operator").post("/v1/incidents", json={"title": "V14 drift on fraud model", "signal": signal})
            r.raise_for_status()
            return r.json()
        if shift_report and shift_report.get("level") != "ok":
            incident, _ = h.try_step("incident.open_from_real_drift", lambda: open_incident_from_drift(shift_report))
        else:
            incident = None
            h.record("incident.open_from_real_drift", "info", {"skipped": "injected shift did not trigger a drift level"})

        # Phase-2 attack (b-q8): burst the SAME small window many times -- does the system open an
        # unbounded number of incidents from noise alone (a real "cry wolf" reproduction)?
        def burst():
            opened = 0
            for _ in range(10):
                rep = drift_check("operator", v_ref, window_small)
                if rep["level"] != "ok":
                    r = api.as_("operator").post("/v1/incidents", json={"title": "burst",
                                                                        "signal": {"drift_level": rep["level"],
                                                                                   "feature_psi": rep["psi_adjusted"]}})
                    if r.status_code == 201:
                        opened += 1
            return {"burst_checks": 10, "incidents_opened_from_noise": opened}
        h.try_step("attack.drift_alert_burst_on_small_window", burst)

        if incident:
            from mlops.incidents import CHECKLIST
            def work_checklist():
                # Driver bug found on re-run (fixed here): /v1/incidents/{id}/steps requires both
                # "step" and "note"; /resolve requires the "postmortem" key even when its value is
                # None. The original script omitted "note" and "postmortem", so every call 422'd
                # and the checklist/resolve machinery was never actually exercised.
                results = []
                for step in CHECKLIST:
                    r = api.as_("operator").post(f"/v1/incidents/{incident['incident_id']}/steps",
                                                  json={"step": step, "note": f"real check: {step}"})
                    results.append((step, r.status_code))
                r = api.as_("operator").post(f"/v1/incidents/{incident['incident_id']}/resolve",
                                              json={"postmortem": None})
                results.append(("resolve", r.status_code))
                return results
            h.try_step("incident.checklist_and_resolve", work_checklist)

        # ------------------------------------------------------- 6. audit chain export + verify + tamper
        def audit_roundtrip():
            # GET /v1/audit/export returns {"entries": [...], "head": {...}}; limit defaults to
            # the last 100 entries server-side, so ask for the full run explicitly.
            export_resp = api.as_("admin").get("/v1/audit/export", params={"limit": 1000}).json()
            entries, head = export_resp["entries"], export_resp["head"]
            good = api.as_("admin").post("/v1/audit/verify", json={"export": entries, "head": head}).json()
            tampered = [dict(e) for e in entries]
            if tampered:
                tampered[0]["resource"] = tampered[0].get("resource", "") + "-tampered"
            bad_result = api.as_("admin").post("/v1/audit/verify", json={"export": tampered, "head": head}).json()
            return {"entries": len(entries), "verify_real": good, "verify_tampered": bad_result}
        h.try_step("audit.export_verify_and_tamper_a_copy", audit_roundtrip)

        # ------------------------------------------------------- 9. lineage
        def lineage():
            anc = api.as_("viewer").get(f"/v1/lineage/fraud-detector/1/ancestors").json()
            br = api.as_("viewer").get(f"/v1/lineage/fraud-detector/1/blast-radius").json()
            return {"ancestors": anc, "blast_radius": br}
        h.try_step("lineage.blast_radius_and_ancestors", lineage)

        # ------------------------------------------------------- 11. SBOM
        def sbom():
            from mlops.ext.cyclonedx_sbom import build_cyclonedx_sbom, sbom_matches_requirements, validate_cyclonedx_sbom
            reqs = (Path(__file__).resolve().parent.parent.parent / "requirements.txt").read_text().splitlines()
            comps = [tuple(l.split("==")) for l in reqs if "==" in l]
            sb = build_cyclonedx_sbom(comps)
            validate_cyclonedx_sbom(sb)
            matches = sbom_matches_requirements(sb, comps)
            return {"components": len(comps), "self_validated": True, "matches_requirements": matches}
        h.try_step("sbom.build_and_self_validate", sbom)

        # ------------------------------------------------------- 12. manifest lint
        def lint():
            from mlops.svc.manifest_io import load_and_lint
            manifest = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: fraud-detector\nspec:\n"
                        "  replicas: 1\n  template:\n    spec:\n      containers:\n"
                        "      - name: server\n        image: fraud-detector:latest\n")
            docs, findings = load_and_lint(manifest)
            return {"findings": [f if isinstance(f, dict) else f.__dict__ for f in findings]}
        h.try_step("lint.model_server_manifest", lint)

        # ------------------------------------------------------- 15. calibration cross-check (real scores)
        def calibration_cli():
            import subprocess
            proba = fraud_clf.predict_proba(fraud_test.drop(columns=["Class"]))[:, 1]
            data = [{"y": int(y), "score": float(s)} for y, s in zip(fraud_test["Class"].tolist()[:2000], proba[:2000])]
            f = out_dir / "calibration_input.json"
            f.write_text(json.dumps(data))
            res = subprocess.run([sys.executable, "-m", "mlops.svc.cli", "policy", "calibration-crosscheck", str(f),
                                  "--blocks-when", "below"], cwd=str(Path(__file__).resolve().parent.parent.parent / "exec"),
                                 env=srv.env, capture_output=True, text=True, timeout=120)
            return {"returncode": res.returncode, "stdout_tail": res.stdout[-800:], "stderr_tail": res.stderr[-400:]}
        h.try_step("calibration.crosscheck_cli_real_scores", calibration_cli)

        # ------------------------------------------------------- 13. metrics/telemetry
        def metrics_moved():
            before = api.as_("viewer").get("/metrics").text
            api.as_("operator").post("/v1/policy/decide", json={"action": "stage.production", "context": {}})
            after = api.as_("viewer").get("/metrics").text
            return {"changed": before != after, "sample_line": next((l for l in after.splitlines()
                                                                     if l.startswith("mlops_") and not l.startswith("#")), None)}
        h.try_step("telemetry.metrics_move", metrics_moved)

        # ------------------------------------------------------- 14. CLI + SDK
        def sdk_roundtrip():
            from mlops.svc.sdk import MlopsClient
            c = MlopsClient(base_url=srv.base_url, token="tk-view-live-s1-000001")
            versions = c.policy_versions()
            return {"policy_versions": versions}
        h.try_step("sdk.policy_versions", sdk_roundtrip)

        # ------------------------------------------------------- 10. MLflow mirror
        def mlflow_check():
            import sqlite3
            conn = sqlite3.connect(str(out_dir / "server" / "mlflow.db"))
            runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            conn.close()
            return {"mlflow_runs": runs}
        h.try_step("mlflow.mirror_has_runs", mlflow_check)

        # ------------------------------------------------------- 16. durability (real kill -9 + restart)
        h.record("durability.pre_kill_snapshot", "info", {
            "active_policy": _safe_json(api.as_("viewer").get("/v1/policy/active")),
            "fraud_stage": _safe_json(api.as_("viewer").get("/v1/models/fraud-detector"))})
        api.close()
        srv.kill9()
        srv2 = LiveServer(out_dir / "server", tokens=TOKENS, port=srv.port,
                          extra_env={"MLOPS_MLFLOW_TRACKING_URI": f"sqlite:///{out_dir / 'server' / 'mlflow.db'}"})
        _, outcome = h.try_step("durability.restart_after_kill9", lambda: srv2.start(timeout_s=60))
        if outcome == "ok":
            api2 = Api(srv2.base_url)
            h.try_step("durability.policy_survived", lambda: _safe_json(api2.as_("viewer").get("/v1/policy/active")))
            h.try_step("durability.model_state_after_restart", lambda: _safe_json(api2.as_("viewer").get("/v1/models/fraud-detector")))
            h.try_step("durability.audit_head_after_restart", lambda: _safe_json(api2.as_("viewer").get("/v1/audit/head")))
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


def _expect_status(resp: httpx.Response, expected) -> dict:
    expected = expected if isinstance(expected, tuple) else (expected,)
    ok = resp.status_code in expected
    return {"status": resp.status_code, "expected": expected, "matched": ok, "body": _safe_json(resp)}


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    h = run(Path(a.data), a.seed, out)
    summary = h.summary()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
