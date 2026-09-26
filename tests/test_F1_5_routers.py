"""F1.5 slim suite: incidents + policy_history FastAPI routers.
Frozen BEFORE mlops.svc.routers.{incidents,policy_history} exist.
Contract: the API contract sections 5, 6, 7.

Fixture pattern follows tests/test_S3_3_api.py (make_client / H / err / JSON_CT),
extended to mount the two new routers via create_app(..., extensions=[...]).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient

from mlops.audit import ChainedAuditStore
from mlops.kernel import ManualClock
from mlops.incidents import IncidentManager
from mlops.policy_store import PolicyStore
from mlops.svc.app import create_app
from mlops.svc.extensions import incidents_extension, policy_history_extension
from mlops.svc.settings import Settings

SECRET = b"f1-5-secret-000000"
TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}
JSON_CT = {"content-type": "application/json"}

P1_SIGNAL = {"accuracy_drop_pct": 15.0}
P3_SIGNAL = {"pipeline_delay": True}
STEPS = ["detection", "impact_assessment", "recent_changes_review", "mitigation_options", "root_cause"]
PM_KEYS = ["timeline", "root_cause", "user_impact", "preventive_measures", "monitoring_gap"]


def _pm():
    return {k: f"text for {k}" for k in PM_KEYS}


def make_settings(**kw):
    tokens = {TOK[r]: {"name": NAME[r], "role": r} for r in TOK}
    return Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **kw)


def make_client(settings=None, manager=None, store=None, clock=None):
    clock = clock or ManualClock(start=1000.0)
    audit = ChainedAuditStore(SECRET)
    store = store if store is not None else PolicyStore(audit=audit, clock=clock)
    manager = manager if manager is not None else IncidentManager(audit, clock)
    app = create_app(
        settings or make_settings(),
        audit=audit,
        policy_store=store,
        clock=clock,
        extensions=[incidents_extension(manager), policy_history_extension()],
    )
    return TestClient(app, raise_server_exceptions=False), audit, manager, store, clock


def H(role, **extra):
    return {"Authorization": "Bearer " + TOK[role], **extra}


def err(r):
    body = r.json()
    assert set(body) == {"error"} and isinstance(body["error"]["code"], str) and isinstance(body["error"]["message"], str)
    return body["error"]


def _open(c, role="operator", title="bad model", signal=None):
    return c.post("/v1/incidents", json={"title": title, "signal": signal or P1_SIGNAL}, headers=H(role))


def _do_steps(c, iid, n=5, role="operator"):
    for step in STEPS[:n]:
        r = c.post(f"/v1/incidents/{iid}/steps", json={"step": step, "note": f"note {step}"}, headers=H(role))
        assert r.status_code == 200, r.text


# ------------------------------------------------------------------ incidents: happy paths

def test_open_incident_returns_201_with_id_severity_deadline():
    c, audit, _, _, _ = make_client()
    r = _open(c)
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body) == {"incident_id", "severity", "deadline"}
    assert body["severity"] == "P1"
    assert body["deadline"] == 1000.0 + 60 * 60


def test_get_incident_returns_incident_model_shape():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    r = c.get(f"/v1/incidents/{iid}", headers=H("viewer"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "incident_id", "title", "severity", "status",
        "opened_ts", "deadline", "current_tier", "steps_completed",
    }
    assert body["incident_id"] == iid
    assert body["severity"] == "P1"
    assert body["status"] == "open"
    assert body["steps_completed"] == []


def test_get_unknown_incident_is_404():
    c, _, _, _, _ = make_client()
    r = c.get("/v1/incidents/ghost", headers=H("viewer"))
    assert r.status_code == 404
    assert err(r)["code"] == "not_found"


def test_steps_happy_path_in_order():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    for step in STEPS:
        r = c.post(f"/v1/incidents/{iid}/steps", json={"step": step, "note": "n"}, headers=H("operator"))
        assert r.status_code == 200 and r.json() == {"status": "ok"}
    got = c.get(f"/v1/incidents/{iid}", headers=H("viewer")).json()
    assert got["steps_completed"] == STEPS


def test_steps_out_of_order_is_409():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    r = c.post(f"/v1/incidents/{iid}/steps", json={"step": STEPS[1], "note": "n"}, headers=H("operator"))
    assert r.status_code == 409
    assert err(r)["code"] == "conflict"


def test_steps_unknown_step_name_is_400_domain_validation():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    r = c.post(f"/v1/incidents/{iid}/steps", json={"step": "not-a-real-step", "note": "n"}, headers=H("operator"))
    assert r.status_code == 400
    assert err(r)["code"] == "validation_failed"


def test_steps_empty_note_is_400_domain_validation():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    r = c.post(f"/v1/incidents/{iid}/steps", json={"step": STEPS[0], "note": ""}, headers=H("operator"))
    assert r.status_code == 400
    assert err(r)["code"] == "validation_failed"


def test_resolve_happy_path_p1_requires_postmortem():
    c, _, _, _, _ = make_client()
    iid = _open(c, signal=P1_SIGNAL).json()["incident_id"]
    _do_steps(c, iid)
    r = c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": _pm()}, headers=H("operator"))
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    got = c.get(f"/v1/incidents/{iid}", headers=H("viewer")).json()
    assert got["status"] == "resolved"


def test_resolve_p1_without_postmortem_is_400_domain_validation():
    c, _, _, _, _ = make_client()
    iid = _open(c, signal=P1_SIGNAL).json()["incident_id"]
    _do_steps(c, iid)
    r = c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": None}, headers=H("operator"))
    assert r.status_code == 400
    assert err(r)["code"] == "validation_failed"


def test_resolve_p3_without_postmortem_is_ok():
    c, _, _, _, _ = make_client()
    iid = _open(c, signal=P3_SIGNAL).json()["incident_id"]
    _do_steps(c, iid)
    r = c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": None}, headers=H("operator"))
    assert r.status_code == 200


def test_resolve_before_steps_complete_is_409():
    c, _, _, _, _ = make_client()
    iid = _open(c, signal=P3_SIGNAL).json()["incident_id"]
    r = c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": None}, headers=H("operator"))
    assert r.status_code == 409
    assert err(r)["code"] == "conflict"


def test_resolve_twice_is_409():
    c, _, _, _, _ = make_client()
    iid = _open(c, signal=P3_SIGNAL).json()["incident_id"]
    _do_steps(c, iid)
    assert c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": None}, headers=H("operator")).status_code == 200
    r = c.post(f"/v1/incidents/{iid}/resolve", json={"postmortem": None}, headers=H("operator"))
    assert r.status_code == 409
    assert err(r)["code"] == "conflict"


# ------------------------------------------------------------------ incidents: auth order + limits

def test_open_incident_no_token_is_401():
    c, _, _, _, _ = make_client()
    r = c.post("/v1/incidents", json={"title": "t", "signal": P1_SIGNAL})
    assert r.status_code == 401
    assert r.headers["www-authenticate"].lower().startswith("bearer")


@pytest.mark.parametrize("content_type,body_bytes", [
    ("text/plain", b'{"title": "t", "signal": {"pipeline_delay": true}}'),
    ("application/json", b"{not json"),
])
def test_viewer_post_is_403_before_body_parsing(content_type, body_bytes):
    """A viewer's POST must fail with 403 even when the body would otherwise be 415/422."""
    c, audit, _, _, _ = make_client()
    before = audit.head()["length"]
    r = c.post("/v1/incidents", content=body_bytes, headers=H("viewer", **{"content-type": content_type}))
    assert r.status_code == 403, r.text
    assert err(r)["code"] == "policy_denied"
    assert audit.head()["length"] == before


def test_wrong_content_type_is_415():
    c, _, _, _, _ = make_client()
    body = json.dumps({"title": "t", "signal": P1_SIGNAL})
    r = c.post("/v1/incidents", content=body, headers=H("operator", **{"content-type": "text/plain"}))
    assert r.status_code == 415


def test_oversized_body_is_413():
    c, _, _, _, _ = make_client(settings=make_settings(max_body_bytes=64))
    payload = json.dumps({"title": "t" * 500, "signal": P1_SIGNAL}).encode()
    assert len(payload) > 64
    r = c.post("/v1/incidents", content=payload, headers=H("operator", **JSON_CT))
    assert r.status_code == 413


@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2, 3]"])
def test_bad_json_shape_is_422(raw):
    c, _, _, _, _ = make_client()
    r = c.post("/v1/incidents", content=raw, headers=H("operator", **JSON_CT))
    assert r.status_code == 422
    assert err(r)["code"] == "validation"


def test_responses_never_leak_actor_secret_or_token():
    c, _, _, _, _ = make_client()
    iid = _open(c).json()["incident_id"]
    get_r = c.get(f"/v1/incidents/{iid}", headers=H("viewer"))
    responses_text = json.dumps(_open(c).json()) + json.dumps(get_r.json())
    for tok in TOK.values():
        assert tok not in responses_text


# ------------------------------------------------------------------ policy_history

def _publish_versions(store, n=2):
    from mlops.policy_engine import PolicyBundle, Rule
    versions = []
    for i in range(1, n + 1):
        rule = Rule(id=f"r{i}", version=1, kind="min_metric", params={"metric": "acc", "min": 0.5}, severity="block")
        bundle = PolicyBundle(rules=[rule], name=f"p{i}", version=i)
        versions.append(store.publish(actor="admin", bundle=bundle, note=f"note{i}"))
    return versions


def test_policy_versions_list():
    c, _, _, store, _ = make_client()
    _publish_versions(store, n=2)
    r = c.get("/v1/policy/versions", headers=H("viewer"))
    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["versions"]
    assert [v["version"] for v in body["versions"]] == [1, 2]


def test_policy_version_detail_includes_hash():
    c, _, _, store, _ = make_client()
    _publish_versions(store, n=1)
    r = c.get("/v1/policy/versions/1", headers=H("viewer"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "p1" and body["version"] == 1
    assert body["hash"] == store.get(1).hash
    assert len(body["rules"]) == 1


@pytest.mark.parametrize("version", [0, "abc"])
def test_policy_version_bad_id_is_422(version):
    c, _, _, store, _ = make_client()
    _publish_versions(store, n=1)
    r = c.get(f"/v1/policy/versions/{version}", headers=H("viewer"))
    assert r.status_code == 422


def test_policy_version_unknown_is_404():
    c, _, _, store, _ = make_client()
    _publish_versions(store, n=1)
    r = c.get("/v1/policy/versions/99", headers=H("viewer"))
    assert r.status_code == 404
    assert err(r)["code"] == "not_found"


def test_policy_decisions_requires_operator_or_admin():
    c, _, _, store, _ = make_client()
    r = c.get("/v1/policy/decisions", headers=H("viewer"))
    assert r.status_code == 403


def test_policy_decisions_default_limit_and_explicit_limit():
    c, _, _, store, clock = make_client()
    versions = _publish_versions(store, n=1)
    store.activate(actor="admin", version=versions[0])
    for i in range(5):
        store.decide(action=f"a{i}", context={"acc": 0.9}, actor="system")
    r = c.get("/v1/policy/decisions", headers=H("operator"))
    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["decisions"]
    assert [d["action"] for d in body["decisions"]] == ["a0", "a1", "a2", "a3", "a4"]
    r2 = c.get("/v1/policy/decisions?limit=2", headers=H("operator"))
    assert [d["action"] for d in r2.json()["decisions"]] == ["a3", "a4"]


@pytest.mark.parametrize("limit", [0, 501])
def test_policy_decisions_limit_out_of_bounds_is_422(limit):
    c, _, _, store, _ = make_client()
    r = c.get(f"/v1/policy/decisions?limit={limit}", headers=H("operator"))
    assert r.status_code == 422


@pytest.mark.parametrize("limit", [1, 500])
def test_policy_decisions_limit_boundary_ok(limit):
    c, _, _, store, _ = make_client()
    versions = _publish_versions(store, n=1)
    store.activate(actor="admin", version=versions[0])
    store.decide(action="a", context={}, actor="system")
    r = c.get(f"/v1/policy/decisions?limit={limit}", headers=H("operator"))
    assert r.status_code == 200


def test_policy_decisions_no_token_is_401():
    c, _, _, _, _ = make_client()
    r = c.get("/v1/policy/decisions")
    assert r.status_code == 401
