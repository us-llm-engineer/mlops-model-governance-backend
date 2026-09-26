"""S3.3 slim suite: the HTTP service layer (mlops.svc.app.create_app), contract in
the API contract (svc/app.py, svc/deps.py, svc/errors.py, svc/idempotency.py).

Frozen expectations (contract-derived; shape guesses are listed in the report):
  create_app(settings, *, audit=None, registry=None, policy_store=None, clock=None, repo=None)
  error envelope {"error": {"code": str, "message": str}}; 401 carries WWW-Authenticate: Bearer.
  Model record JSON = ModelVersionRecord fields (model_id, version_id, stage, validated, previous_production...).
  Registry deny-audits 404/409 itself (see test_S3_0_release_fixes): those may add <= 1 "deny" entry;
  every other rejection (401/403/400/413/415/422) adds exactly ZERO audit entries.
"""
import json
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.svc.app import create_app
from mlops.svc.settings import Settings

SECRET = b"s" * 20
TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}
RANK = {"viewer": 0, "operator": 1, "admin": 2}
H1, H2 = "11" * 32, "22" * 32
JSON_CT = {"content-type": "application/json"}


def make_settings(**kw):
    tokens = {TOK[r]: {"name": NAME[r], "role": r} for r in TOK}
    return Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **kw)


def make_client(settings=None, **create_kw):
    audit = create_kw.pop("audit", None) or ChainedAuditStore(SECRET)
    app = create_app(settings or make_settings(), audit=audit, **create_kw)
    return TestClient(app, raise_server_exceptions=False), audit


def H(role, **extra):
    return {"Authorization": "Bearer " + TOK[role], **extra}


def n(audit):
    return audit.head()["length"]


def reg_body(m="m", v="v1", h=H1, **extra):
    return {"model_id": m, "version_id": v, "artifact_hash": h, **extra}


def mpath(m, v, act):
    return f"/v1/models/{m}/versions/{v}/{act}"


def rule(rid="r1", minimum=0.9):
    return {"id": rid, "version": 1, "kind": "min_metric", "params": {"metric": "acc", "min": minimum},
            "severity": "block"}


def bundle(minimum=0.9, name="p"):
    return {"name": name, "version": 1, "rules": [rule(minimum=minimum)]}


_rng = random.Random(1)
REF = [_rng.gauss(0, 1) for _ in range(100)]


def register(c, m="m", v="v1", h=H1, role="admin"):
    r = c.post("/v1/models", json=reg_body(m, v, h), headers=H(role))
    assert r.status_code == 201, r.text
    return r


def validate(c, m="m", v="v1", role="admin"):
    r = c.post(mpath(m, v, "validate"), json={"evidence": {"metric": "acc", "value": 0.95}}, headers=H(role))
    assert r.status_code == 200, r.text
    return r


def to_stage(c, stage, m="m", v="v1", role="admin"):
    r = c.post(mpath(m, v, "transition"), json={"to_stage": stage}, headers=H(role))
    assert r.status_code == 200, r.text
    return r


def promote(c, m="m", v="v1", h=H1):
    register(c, m, v, h)
    validate(c, m, v)
    to_stage(c, "staging", m, v)
    return to_stage(c, "production", m, v)


def hist(body):
    return body if isinstance(body, list) else body["history"]


def err(r):
    body = r.json()
    assert set(body) == {"error"} and isinstance(body["error"]["code"], str) and isinstance(body["error"]["message"], str)
    return body["error"]


ROUTES = [
    ("get", "/healthz"), ("get", "/metrics"), ("post", "/v1/models"),
    ("post", "/v1/models/{model_id}/versions/{version_id}/validate"),
    ("post", "/v1/models/{model_id}/versions/{version_id}/transition"),
    ("post", "/v1/models/{model_id}/rollback"), ("get", "/v1/models/{model_id}"),
    ("post", "/v1/policy/publish"), ("post", "/v1/policy/{version}/activate"),
    ("post", "/v1/policy/decide"), ("get", "/v1/policy/active"), ("post", "/v1/lint"),
    ("post", "/v1/drift/check"), ("get", "/v1/audit/head"), ("get", "/v1/audit/export"),
]


# ------------------------------------------------------------------ (1) health / openapi / docs

def test_healthz_needs_no_auth():
    c, _ = make_client()
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "env" in r.json()


def test_openapi_lists_every_contract_route():
    c, _ = make_client()
    r = c.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for method, path in ROUTES:
        assert path in paths and method in paths[path], (method, path)


@pytest.mark.parametrize("env,status", [("prod", 404), ("dev", 200)])
def test_docs_only_in_dev(env, status):
    c, _ = make_client(make_settings(env_name=env))
    assert c.get("/docs").status_code == status


# ------------------------------------------------------------------ (2) auth + RBAC

RBAC = [
    ("GET", "/metrics", None, "viewer"),
    ("POST", "/v1/models", reg_body("rb", "v9", H2), "operator"),
    ("POST", mpath("m", "v1", "validate"), {"evidence": {"k": 1}}, "operator"),
    ("POST", mpath("m", "v1", "transition"), {"to_stage": "staging"}, "operator"),
    ("POST", mpath("m2", "v2", "transition"), {"to_stage": "production"}, "admin"),
    ("POST", "/v1/models/m2/rollback", {}, "admin"),
    ("GET", "/v1/models/m", None, "viewer"),
    ("POST", "/v1/policy/publish", bundle(), "admin"),
    ("POST", "/v1/policy/1/activate", {}, "admin"),
    ("POST", "/v1/policy/decide", {"action": "a", "context": {"acc": 0.95}}, "operator"),
    ("GET", "/v1/policy/active", None, "viewer"),
    ("POST", "/v1/lint", {"documents": []}, "viewer"),
    ("POST", "/v1/drift/check", {"reference": REF, "window": REF}, "viewer"),
    ("GET", "/v1/audit/head", None, "viewer"),
    ("GET", "/v1/audit/export", None, "admin"),
]


@pytest.mark.parametrize("role", ["viewer", "operator", "admin"])
@pytest.mark.parametrize("method,path,body,min_role", RBAC)
def test_rbac_matrix(role, method, path, body, min_role):
    c, audit = make_client()
    register(c, "m", "v1")                                   # registered only
    promote_ready = ("m2", "v2")
    register(c, *promote_ready, H2)
    validate(c, *promote_ready)
    to_stage(c, "staging", *promote_ready)                   # staged + validated, for the production edge
    assert c.post("/v1/policy/publish", json=bundle(), headers=H("admin")).status_code in (200, 201)
    assert c.post("/v1/policy/1/activate", json={}, headers=H("admin")).status_code == 200
    before = n(audit)
    r = c.request(method, path, json=body, headers=H(role))
    if RANK[role] < RANK[min_role]:
        assert r.status_code == 403, (role, method, path, r.text)
        assert err(r)["code"]
        assert n(audit) == before
    else:
        assert r.status_code not in (401, 403) and r.status_code < 500, (role, method, path, r.text)


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer "}, {"Authorization": "Bearer wrong-token-0000"},
    {"Authorization": "Basic " + TOK["admin"]}, {"Authorization": TOK["admin"]},
    {"Authorization": "Bearer " + TOK["admin"].upper()},
])
def test_missing_or_bad_token_is_401(headers):
    c, audit = make_client()
    for method, path, body in [("GET", "/v1/models/m", None), ("POST", "/v1/models", reg_body()),
                               ("GET", "/v1/audit/head", None)]:
        r = c.request(method, path, json=body, headers=headers)
        assert r.status_code == 401, (method, path)
        assert r.headers["www-authenticate"].lower().startswith("bearer")
        assert err(r)["code"]
    assert n(audit) == 0


# ------------------------------------------------------------------ (3) model lifecycle

def test_lifecycle_happy_path_and_rollback():
    c, _ = make_client()
    r = register(c, "m", "v1", H1)
    rec = r.json()
    assert rec["model_id"] == "m" and rec["version_id"] == "v1" and rec["stage"] == "registered"
    assert rec["validated"] is False
    v = validate(c, "m", "v1").json()
    assert v["validated"] is True
    assert to_stage(c, "staging").json()["stage"] == "staging"
    assert to_stage(c, "production").json()["stage"] == "production"
    stages = {h["version_id"]: h["stage"] for h in hist(c.get("/v1/models/m", headers=H("viewer")).json())}
    assert stages["v1"] == "production"
    # second version promoted: first is archived, previous_production recorded
    register(c, "m", "v2", H2)
    validate(c, "m", "v2")
    to_stage(c, "staging", "m", "v2")
    p2 = to_stage(c, "production", "m", "v2").json()
    assert p2["stage"] == "production" and p2["previous_production"] == "v1"
    latest = {}
    for h in hist(c.get("/v1/models/m", headers=H("viewer")).json()):
        latest[h["version_id"]] = h["stage"]                 # later entries are newer
    assert latest == {"v1": "archived", "v2": "production"}
    rb = c.post("/v1/models/m/rollback", json={}, headers=H("admin"))
    assert rb.status_code == 200 and rb.json()["version_id"] == "v1" and rb.json()["stage"] == "production"


def test_lifecycle_error_statuses():
    c, _ = make_client()
    register(c, "m", "v1")
    assert c.post(mpath("m", "v1", "transition"), json={"to_stage": "production"}, headers=H("admin")).status_code == 409  # not staged
    assert c.post(mpath("m", "v1", "transition"), json={"to_stage": "archived"}, headers=H("admin")).status_code == 409   # illegal edge
    assert c.post("/v1/models", json=reg_body("m", "v1"), headers=H("admin")).status_code == 409                          # duplicate
    for path, body in [(mpath("ghost", "v1", "transition"), {"to_stage": "staging"}),
                       (mpath("m", "vX", "validate"), {"evidence": {"a": 1}}),
                       ("/v1/models/ghost/rollback", {})]:
        r = c.post(path, json=body, headers=H("admin"))
        assert r.status_code == 404, path
        assert err(r)["code"] == "not_found"
    assert err(c.post("/v1/models", json=reg_body("m", "v1"), headers=H("admin")))["code"] == "conflict"


@pytest.mark.parametrize("path,body", [
    ("/v1/models", reg_body(h="abc")),
    ("/v1/models", reg_body(h="G" * 64)),
    ("/v1/models", reg_body(h=H1[:-1])),
    ("/v1/models", reg_body(extra_field=1)),
    ("/v1/models", {"model_id": "m", "version_id": "v1"}),
    ("/v1/models", reg_body(dataset_version=5)),
    (mpath("m", "v1", "transition"), {"to_stage": "staging "}),
    (mpath("m", "v1", "transition"), {"to_stage": 5}),
    (mpath("m", "v1", "transition"), {"to_stage": "registered"}),
    (mpath("m", "v1", "transition"), {"to_stage": "staging", "extra": 1}),
    (mpath("m", "v1", "validate"), {"evidence": "not-a-dict"}),
])
def test_invalid_bodies_are_422_with_zero_audit(path, body):
    c, audit = make_client()
    register(c, "m", "v1")
    before = n(audit)
    r = c.post(path, json=body, headers=H("admin"))
    assert r.status_code == 422, r.text
    assert err(r)["code"] == "validation"
    assert n(audit) == before


def test_audit_grows_by_one_per_success_and_zero_per_rejection():
    c, audit = make_client()
    steps = [
        ("POST", "/v1/models", reg_body(), "admin"),
        ("POST", mpath("m", "v1", "validate"), {"evidence": {"k": 1}}, "operator"),
        ("POST", mpath("m", "v1", "transition"), {"to_stage": "staging"}, "operator"),
        ("POST", mpath("m", "v1", "transition"), {"to_stage": "production"}, "admin"),
        ("POST", "/v1/policy/publish", bundle(), "admin"),
        ("POST", "/v1/policy/1/activate", {}, "admin"),
    ]
    for method, path, body, role in steps:
        before = n(audit)
        r = c.request(method, path, json=body, headers=H(role))
        assert r.status_code in (200, 201), (path, r.text)
        assert n(audit) == before + 1, path
    register(c, "m", "v2", H2)
    validate(c, "m", "v2")
    to_stage(c, "staging", "m", "v2")
    to_stage(c, "production", "m", "v2")
    before = n(audit)
    assert c.post("/v1/models/m/rollback", json={}, headers=H("admin")).status_code == 200
    assert n(audit) == before + 1
    # rejections that never reach domain code: exactly zero
    rejected = [
        ("POST", "/v1/models", reg_body("x", "v1"), {}, 401),
        ("POST", "/v1/models", reg_body("x", "v1"), H("viewer"), 403),
        ("POST", mpath("m", "v2", "transition"), {"to_stage": "production"}, H("operator"), 403),
        ("POST", "/v1/models", reg_body("x", "v1", "zz"), H("admin"), 422),
        ("POST", "/v1/models", reg_body("x", "v1"), H("admin", **{"Idempotency-Key": "short"}), 400),
    ]
    for method, path, body, headers, status in rejected:
        before = n(audit)
        assert c.request(method, path, json=body, headers=headers).status_code == status
        assert n(audit) == before
    # domain rejections: registry may write one deny entry, never more, never an allow
    for path, body in [(mpath("ghost", "v1", "validate"), {"evidence": {"a": 1}}),
                       (mpath("m", "v1", "transition"), {"to_stage": "staging"})]:
        before = n(audit)
        assert c.post(path, json=body, headers=H("admin")).status_code in (404, 409)
        new = audit.export()[before:]
        assert len(new) <= 1 and all(e["decision"] == "deny" for e in new)


def test_audit_chain_verifies_and_actor_is_calling_principal():
    c, audit = make_client()
    register(c, "m", "v1", role="admin")                                   # root
    validate(c, "m", "v1", role="operator")                                # olga
    to_stage(c, "staging", role="operator")                                # olga
    to_stage(c, "production", role="admin")                                # root
    c.post(mpath("m", "v1", "transition"), json={"to_stage": "archived"}, headers=H("viewer"))   # 403 vic
    c.post(mpath("m", "v1", "transition"), json={"to_stage": "staging"}, headers=H("operator"))  # 409 olga
    c.post("/v1/policy/publish", json=bundle(), headers=H("admin"))
    entries = audit.export()
    assert [e["actor"] for e in entries[:4]] == ["root", "olga", "olga", "root"]
    assert entries[-1]["actor"] == "root"
    assert {e["actor"] for e in entries} <= {"root", "olga"}
    assert ChainVerifier(SECRET).verify_export(audit.export(), head=audit.head()) is True


# ------------------------------------------------------------------ (4) policy

def test_policy_publish_activate_decide_and_loosening_gate():
    c, audit = make_client()
    r = c.post("/v1/policy/decide", json={"action": "a", "context": {"acc": 0.95}}, headers=H("operator"))
    assert r.status_code == 403 and err(r)["code"] == "policy_denied"     # no active bundle: fail-closed
    assert c.post("/v1/policy/publish", json=bundle(0.9), headers=H("admin")).status_code in (200, 201)
    assert c.post("/v1/policy/1/activate", json={}, headers=H("admin")).status_code == 200   # first activation
    ok = c.post("/v1/policy/decide", json={"action": "a", "context": {"acc": 0.95}}, headers=H("operator"))
    no = c.post("/v1/policy/decide", json={"action": "a", "context": {"acc": 0.5}}, headers=H("operator"))
    miss = c.post("/v1/policy/decide", json={"action": "a", "context": {}}, headers=H("operator"))
    assert ok.status_code == no.status_code == miss.status_code == 200
    assert ok.json()["allow"] is True and no.json()["allow"] is False and miss.json()["allow"] is False
    assert set(ok.json()) >= {"allow", "reasons", "version", "policy_hash"} and ok.json()["version"] == 1
    assert isinstance(no.json()["reasons"], list) and no.json()["reasons"]
    # looser v2
    assert c.post("/v1/policy/publish", json=bundle(0.5), headers=H("admin")).status_code in (200, 201)
    before = n(audit)
    r = c.post("/v1/policy/2/activate", json={}, headers=H("admin"))
    assert r.status_code == 403 and err(r)["code"] == "policy_denied"
    assert c.get("/v1/policy/active", headers=H("viewer")).json()["version"] == 1
    assert n(audit) == before
    assert c.post("/v1/policy/2/activate", json={"human_approved_by": "alice"}, headers=H("admin")).status_code == 200
    assert c.get("/v1/policy/active", headers=H("viewer")).json()["version"] == 2
    assert c.post("/v1/policy/9/activate", json={}, headers=H("admin")).status_code == 404


@pytest.mark.parametrize("raw", [b'{"action":"a","context":{"acc":NaN}}', b'{"action":"a","context":{"acc":true}}',
                                 b'{"action":"a","context":{"acc":Infinity}}'])
def test_policy_decide_nan_or_bool_metric_never_500(raw):
    c, _ = make_client()
    c.post("/v1/policy/publish", json=bundle(), headers=H("admin"))
    c.post("/v1/policy/1/activate", json={}, headers=H("admin"))
    r = c.post("/v1/policy/decide", content=raw, headers=H("operator", **JSON_CT))
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        assert r.json()["allow"] is False
    else:
        assert err(r)["code"] == "validation"


# ------------------------------------------------------------------ (5) lint

def test_lint_reports_findings_score_and_context():
    c, audit = make_client()
    doc = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web", "namespace": "prod"},
           "spec": {"replicas": 2, "template": {"spec": {"containers": [
               {"name": "c", "image": "app:latest", "securityContext": {"privileged": True}}]}}}}
    r = c.post("/v1/lint", json={"documents": [doc]}, headers=H("viewer"))
    assert r.status_code == 200
    body = r.json()
    ids = {f["rule_id"] for f in body["findings"]}
    assert {"privileged", "latest-tag"} <= ids
    assert isinstance(body["score"], float) and 0.0 < body["score"] <= 1.0
    assert set(body["context"]) >= {"lint_blockers", "lint_warnings", "lint_score"}
    assert body["context"]["lint_blockers"] >= 1
    assert n(audit) == 0                                                    # pure function


@pytest.mark.parametrize("body,statuses", [
    ({"documents": [{}] * 5001}, (400, 422)),
    ({"documents": "nope"}, (422,)),
    ({"documents": {"a": 1}}, (422,)),
    ({"documents": [1, 2]}, (400, 422)),
    ({"documents": [], "unknown": 1}, (422,)),
])
def test_lint_rejects_bad_inputs(body, statuses):
    c, _ = make_client()
    r = c.post("/v1/lint", json=body, headers=H("viewer"))
    assert r.status_code in statuses, r.text
    assert err(r)["code"]


# ------------------------------------------------------------------ (6) drift

def test_drift_identical_is_ok_and_shifted_is_alert():
    c, _ = make_client()
    ok = c.post("/v1/drift/check", json={"reference": REF, "window": list(REF)}, headers=H("viewer"))
    assert ok.status_code == 200
    body = ok.json()
    assert body["level"] == "ok" and body["n"] == 100
    assert set(body) >= {"level", "psi", "psi_adjusted", "ks", "ks_pvalue", "n"}
    shifted = c.post("/v1/drift/check", json={"reference": REF, "window": [x + 3 for x in REF]}, headers=H("viewer"))
    assert shifted.status_code == 200 and shifted.json()["level"] == "alert"


def test_drift_rejects_short_window_and_nan():
    c, audit = make_client()
    r = c.post("/v1/drift/check", json={"reference": REF, "window": REF[:10]}, headers=H("viewer"))
    assert r.status_code in (400, 409) and err(r)["code"]          # ASSUMPTION: Conflict(409) per DriftMonitor.check
    raw = json.dumps({"reference": REF, "window": REF}).replace(str(REF[0]), "NaN", 1).encode()
    assert b"NaN" in raw
    r = c.post("/v1/drift/check", content=raw, headers=H("viewer", **JSON_CT))
    assert r.status_code == 422 and err(r)["code"] == "validation"
    assert n(audit) == 0


# ------------------------------------------------------------------ (7) audit endpoints

def test_audit_head_matches_store():
    c, audit = make_client()
    register(c)
    r = c.get("/v1/audit/head", headers=H("viewer"))
    assert r.status_code == 200
    assert r.json() == audit.head() and set(r.json()) == {"length", "signature", "anchor"}


def test_audit_export_limit_and_role():
    c, audit = make_client()
    register(c)
    validate(c)
    to_stage(c, "staging")
    total = n(audit)
    assert total == 3
    assert c.get("/v1/audit/export", headers=H("viewer")).status_code == 403
    assert c.get("/v1/audit/export", headers=H("operator")).status_code == 403
    r = c.get("/v1/audit/export?limit=2", headers=H("admin"))
    assert r.status_code == 200
    body = r.json()
    assert len(body["entries"]) == 2 and body["head"] == audit.head()      # ASSUMPTION: keys "entries"/"head"
    assert body["entries"] == audit.export()[-2:]                          # newest entries
    assert len(c.get("/v1/audit/export", headers=H("admin")).json()["entries"]) == 3
    assert c.get("/v1/audit/export?limit=1000", headers=H("admin")).status_code == 200
    for bad in ("0", "1001", "-1", "abc"):
        assert c.get(f"/v1/audit/export?limit={bad}", headers=H("admin")).status_code == 422, bad


# ------------------------------------------------------------------ (8) limits

@pytest.mark.parametrize("chunked", [False, True])
def test_oversized_body_is_413(chunked):
    c, audit = make_client(make_settings(max_body_bytes=2048))
    payload = json.dumps(reg_body(dataset_version="x" * 3000)).encode()
    assert len(payload) > 2048
    content = (payload[i:i + 500] for i in range(0, len(payload), 500)) if chunked else payload
    r = c.post("/v1/models", content=content, headers=H("admin", **JSON_CT))
    assert r.status_code == 413
    assert n(audit) == 0
    small = c.post("/v1/models", json=reg_body(), headers=H("admin"))         # limit is per request, not a lockout
    assert small.status_code == 201


def test_non_json_content_type_is_415():
    c, audit = make_client()
    body = json.dumps(reg_body())
    for ct in ("text/plain", "application/x-www-form-urlencoded"):
        r = c.post("/v1/models", content=body, headers=H("admin", **{"content-type": ct}))
        assert r.status_code == 415, ct
    assert n(audit) == 0


@pytest.mark.parametrize("raw", [b"{not json", b"", b'{"model_id": ', b"[1,2", b"\xff\xfe"])
def test_malformed_json_uses_standard_envelope(raw):
    c, audit = make_client()
    r = c.post("/v1/models", content=raw, headers=H("admin", **JSON_CT))
    assert r.status_code in (400, 422), r.text
    assert err(r)["code"]
    assert n(audit) == 0


# ------------------------------------------------------------------ (9) idempotency

def test_idempotent_replay_returns_same_response_and_audits_once():
    c, audit = make_client()
    hdr = H("admin", **{"Idempotency-Key": "reg-key-0001"})
    r1 = c.post("/v1/models", json=reg_body(), headers=hdr)
    assert r1.status_code == 201
    after_first = n(audit)
    r2 = c.post("/v1/models", json=reg_body(), headers=hdr)
    assert r2.status_code == 201 and r2.json() == r1.json()
    assert n(audit) == after_first == 1
    assert len(hist(c.get("/v1/models/m", headers=H("viewer")).json())) == 1


def test_idempotency_key_conflict_and_principal_scope():
    c, audit = make_client()
    key = {"Idempotency-Key": "shared-key-01"}
    assert c.post("/v1/models", json=reg_body("m", "v1"), headers=H("admin", **key)).status_code == 201
    before = n(audit)
    r = c.post("/v1/models", json=reg_body("m", "v2", H2), headers=H("admin", **key))
    assert r.status_code == 422 and err(r)["code"] == "idempotency_conflict"
    assert n(audit) == before
    r = c.post("/v1/models", json=reg_body("m", "v2", H2), headers=H("operator", **key))   # other principal
    assert r.status_code == 201
    assert n(audit) == before + 1


@pytest.mark.parametrize("key", ["short", "a" * 129, "has space in it", "bad/char/key!", "x" * 7])
def test_invalid_idempotency_key_is_400(key):
    c, audit = make_client()
    r = c.post("/v1/models", json=reg_body(), headers=H("admin", **{"Idempotency-Key": key}))
    assert r.status_code == 400 and err(r)["code"]
    assert n(audit) == 0
    assert c.get("/v1/models/m", headers=H("viewer")).status_code == 404 or hist(
        c.get("/v1/models/m", headers=H("viewer")).json()) == []


def test_failed_request_is_not_cached_as_success():
    c, _ = make_client()
    hdr = H("admin", **{"Idempotency-Key": "retry-key-01"})
    body = {"to_stage": "staging"}
    r1 = c.post(mpath("m", "v1", "transition"), json=body, headers=hdr)
    assert r1.status_code == 404
    register(c)
    validate(c)
    r2 = c.post(mpath("m", "v1", "transition"), json=body, headers=hdr)      # same key+request now succeeds
    assert r2.status_code == 200 and r2.json()["stage"] == "staging"


# ------------------------------------------------------------------ (10) metrics

def _series(text, name):
    return [ln for ln in text.splitlines() if ln.startswith(name)]


def test_metrics_use_route_templates_audit_gauge_and_count_401():
    c, audit = make_client()
    register(c, "zz-model-1")
    validate(c, "zz-model-1")
    c.get("/v1/models/zz-model-1", headers=H("viewer"))
    c.get("/v1/models/zz-model-2", headers=H("viewer"))
    c.get("/v1/audit/head")                                                  # no token -> 401
    r = c.get("/metrics", headers=H("viewer"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "mlops_http_requests_total" in text and "mlops_http_request_seconds" in text
    assert 'path="/v1/models/{model_id}"' in text
    assert "zz-model-1" not in text and "zz-model-2" not in text and 'path="/v1/models/zz' not in text
    gauge = _series(text, "mlops_audit_length")
    assert [float(ln.split()[-1]) for ln in gauge if not ln.startswith("#")] == [float(n(audit))]
    assert any('status="401"' in ln and 'path="/v1/audit/head"' in ln
               for ln in _series(text, "mlops_http_requests_total"))
    assert c.get("/metrics", headers=H("viewer", **{"x": "y"})).status_code == 200
    assert TestClient(c.app).get("/metrics").status_code == 401


# ------------------------------------------------------------------ (11) no leakage / 500 envelope

class BoomRegistry:
    def __getattr__(self, name):
        def boom(*a, **k):
            raise RuntimeError("boom-internal")
        return boom


def test_unhandled_error_is_generic_500_and_writes_nothing():
    c, audit = make_client(registry=BoomRegistry())
    for method, path, body in [("POST", "/v1/models", reg_body()), ("GET", "/v1/models/m", None),
                               ("POST", mpath("m", "v1", "validate"), {"evidence": {"a": 1}})]:
        r = c.request(method, path, json=body, headers=H("admin"))
        assert r.status_code == 500, (method, path)
        assert r.json() == {"error": {"code": "internal", "message": "internal error"}}
        assert "boom-internal" not in r.text and "RuntimeError" not in r.text
    assert n(audit) == 0
    assert c.get("/healthz").status_code == 200                              # service survives


def test_no_secret_or_path_leakage_anywhere():
    c, _ = make_client()
    responses = []
    calls = [("GET", "/healthz", None, {}), ("GET", "/openapi.json", None, {}), ("GET", "/docs", None, {}),
             ("GET", "/v1/models/m", None, {"Authorization": "Bearer leaky-bad-token-77"}),
             ("POST", "/v1/models", reg_body(h="zz"), H("admin")), ("POST", "/v1/models", reg_body(), H("admin")),
             ("POST", "/v1/models", reg_body(), H("viewer")), ("GET", "/metrics", None, H("viewer")),
             ("GET", "/v1/audit/head", None, H("viewer")), ("GET", "/v1/audit/export", None, H("admin")),
             ("GET", "/v1/policy/active", None, H("viewer")),
             ("POST", "/v1/policy/decide", {"action": "a", "context": {}}, H("operator")),
             ("GET", "/no/such/route", None, H("admin"))]
    for method, path, body, headers in calls:
        responses.append((path, c.request(method, path, json=body, headers=headers)))
    responses.append(("bad-json", c.post("/v1/models", content=b"{oops", headers=H("admin", **JSON_CT))))
    forbidden = list(TOK.values()) + ["s" * 20, "leaky-bad-token-77", "Traceback", "site-packages", "/home/", ".py\""]
    for path, r in responses:
        blob = r.text + json.dumps(dict(r.headers))
        for bad in forbidden:
            assert bad not in blob, (path, bad)


# ------------------------------------------------------------------ (12) repo integration

def test_repo_backs_reads_and_mirrors_registry(tmp_path):
    sqldb = pytest.importorskip("mlops.sqldb")
    url = f"sqlite:///{tmp_path / 'api.db'}"
    sqldb.upgrade(url)
    repo = sqldb.SqlRepo(url)
    try:
        c, _ = make_client(repo=repo)
        register(c, "m", "v1")
        assert [(r["version_id"], r["stage"]) for r in repo.get_model("m")] == [("v1", "registered")]
        validate(c)
        to_stage(c, "staging")
        rows = repo.get_model("m")
        assert [(r["version_id"], r["stage"], r["validated"]) for r in rows] == [("v1", "staging", True)]
        to_stage(c, "production")
        assert {r["version_id"]: r["stage"] for r in repo.get_model("m")} == {"v1": "production"}
        repo.set_stage("m", "v1", "archived")                               # change ONLY the repo
        got = hist(c.get("/v1/models/m", headers=H("viewer")).json())
        assert [(h["version_id"], h["stage"]) for h in got] == [("v1", "archived")]   # read comes from repo
    finally:
        repo.close()
