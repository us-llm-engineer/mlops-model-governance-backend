"""Shared building blocks for the S1/S2/S3 scenario drivers. Every function drives the real,
public HTTP surface only; a read-only DB/audit peek is explicitly labeled as observation, never
used to advance state.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "exec"))  # so in-process `from mlops...` imports resolve


def content_hash(df: pd.DataFrame) -> str:
    # Driver bug found on S3's original run (fixed here): hash_pandas_object can't hash a column
    # of unhashable Python objects (e.g. EuroSAT's `image` column, one {"bytes": ..., "path": ...}
    # dict per row). Fall back to hashing a stable repr() of any column that isn't hashable as-is,
    # rather than letting the whole call crash and silently leaving dataset lineage unregistered.
    def _hashable(col: pd.Series) -> pd.Series:
        try:
            pd.util.hash_pandas_object(col)
            return col
        except TypeError:
            return col.map(repr)
    safe_df = pd.DataFrame({name: _hashable(df[name]) for name in df.columns})
    return hashlib.sha256(pd.util.hash_pandas_object(safe_df, index=False).values.tobytes()).hexdigest()


class Api:
    """One httpx client per role."""

    def __init__(self, base_url: str, tokens: dict):
        self.clients = {m["role"]: httpx.Client(base_url=base_url, timeout=30.0,
                                                 headers={"Authorization": f"Bearer {t}"})
                        for t, m in tokens.items()}

    def as_(self, role: str) -> httpx.Client:
        return self.clients[role]

    def close(self):
        for c in self.clients.values():
            c.close()


def expect_status(resp: httpx.Response, expected) -> dict:
    expected = expected if isinstance(expected, tuple) else (expected,)
    return {"status": resp.status_code, "expected": expected, "matched": resp.status_code in expected,
            "body": safe_json(resp)}


def safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text[:300]


def register_dataset(api: Api, name: str, df: pd.DataFrame) -> str:
    r = api.as_("operator").post("/v1/datasets", json={
        "name": name, "content_hash": content_hash(df), "rows": int(len(df)),
        "dataset_schema": {"type": "object", "properties": {c: {} for c in list(df.columns)[:5]}}})
    r.raise_for_status()
    return r.json()["version_id"]


def register_and_stage(api: Api, model_id: str, artifact_hash: str, dataset_version: Optional[str], evidence: dict) -> str:
    r = api.as_("operator").post("/v1/models", json={
        "model_id": model_id, "version_id": "1", "artifact_hash": artifact_hash, "dataset_version": dataset_version})
    r.raise_for_status()
    base = f"/v1/models/{model_id}/versions/1"
    api.as_("operator").post(f"{base}/validate", json={"evidence": evidence}).raise_for_status()
    api.as_("operator").post(f"{base}/transition", json={"to_stage": "staging"}).raise_for_status()
    return base


def publish_and_activate(api: Api, rules: list, version: int) -> int:
    r = api.as_("admin").post("/v1/policy/publish", json={"name": "gate", "version": version, "rules": rules})
    r.raise_for_status()
    v = r.json()["version"]
    api.as_("admin").post(f"/v1/policy/{v}/activate", json={"human_approved_by": "root"}).raise_for_status()
    return v


def promote(api: Api, base: str, context: dict) -> dict:
    r = api.as_("admin").post(f"{base}/transition", json={"to_stage": "production", "context": context})
    return {"status": r.status_code, "body": safe_json(r)}


def drift_check(api: Api, ref: list, window: list, name: Optional[str] = None) -> dict:
    body = {"reference": ref, "window": window}
    if name:
        body["name"] = name
    r = api.as_("operator").post("/v1/drift/check", json=body)
    r.raise_for_status()
    return r.json()


def audit_roundtrip(api: Api) -> dict:
    # GET /v1/audit/export returns {"entries": [...], "head": {...}}; limit defaults to the last
    # 100 entries server-side, so ask for the full run explicitly.
    export_resp = api.as_("admin").get("/v1/audit/export", params={"limit": 1000}).json()
    entries, head = export_resp["entries"], export_resp["head"]
    good = api.as_("admin").post("/v1/audit/verify", json={"export": entries, "head": head}).json()
    tampered = [dict(e) for e in entries]
    if tampered:
        tampered[0]["resource"] = tampered[0].get("resource", "") + "-tampered"
    bad_result = api.as_("admin").post("/v1/audit/verify", json={"export": tampered, "head": head}).json()
    return {"entries": len(entries), "verify_real": good, "verify_tampered": bad_result}


def lineage(api: Api, model_id: str) -> dict:
    anc = api.as_("viewer").get(f"/v1/lineage/{model_id}/1/ancestors").json()
    br = api.as_("viewer").get(f"/v1/lineage/{model_id}/1/blast-radius").json()
    return {"ancestors": anc, "blast_radius": br}


def sbom(repo_root: Path) -> dict:
    from mlops.ext.cyclonedx_sbom import build_cyclonedx_sbom, validate_cyclonedx_sbom
    reqs = (repo_root / "requirements.txt").read_text().splitlines()
    comps = [tuple(l.split("==")) for l in reqs if "==" in l]
    sb = build_cyclonedx_sbom(comps)
    validate_cyclonedx_sbom(sb)
    return {"components": len(comps), "self_validated": True}


def lint(model_id: str) -> dict:
    from mlops.svc.manifest_io import load_and_lint
    manifest = (f"apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {model_id}\nspec:\n"
                "  replicas: 1\n  template:\n    spec:\n      containers:\n"
                f"      - name: server\n        image: {model_id}:latest\n")
    docs, findings = load_and_lint(manifest)
    return {"findings": [f if isinstance(f, dict) else f.__dict__ for f in findings]}


def metrics_moved(api: Api) -> dict:
    before = api.as_("viewer").get("/metrics").text
    api.as_("operator").post("/v1/policy/decide", json={"action": "stage.production", "context": {}})
    after = api.as_("viewer").get("/metrics").text
    return {"changed": before != after,
            "sample_line": next((l for l in after.splitlines() if l.startswith("mlops_") and not l.startswith("#")), None)}


def sdk_roundtrip(base_url: str, token: str) -> dict:
    from mlops.svc.sdk import MlopsClient
    c = MlopsClient(base_url=base_url, token=token)
    return {"policy_versions": c.policy_versions()}


def mlflow_run_count(mlflow_db_path: Path) -> dict:
    conn = sqlite3.connect(str(mlflow_db_path))
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    return {"mlflow_runs": n}


def read_drift_baseline_http(api: Api, name: str) -> dict:
    """Real client call to GET /v1/drift/baselines/{name} -- the route that didn't exist when S3
    first ran (Finding 3), added since. A caller can now retrieve a baseline the periodic worker
    persisted, instead of it being reachable only by querying the DB directly."""
    r = api.as_("viewer").get(f"/v1/drift/baselines/{name}")
    return {"status": r.status_code, "body": safe_json(r)}


def read_drift_baseline_row_readonly(db_path: Path, name: str) -> Optional[dict]:
    """Read-only peek at the ops DB's drift_baseline table (observation only, never used to
    advance a flow). Confirms the periodic drift-evaluation worker actually persisted a baseline."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT name, created_ts, length(reference_json) FROM drift_baseline WHERE name=?",
                           (name,)).fetchone()
    except sqlite3.OperationalError as e:
        return {"error": str(e)}
    finally:
        conn.close()
    return {"name": row[0], "created_ts": row[1], "reference_json_len": row[2]} if row else None
