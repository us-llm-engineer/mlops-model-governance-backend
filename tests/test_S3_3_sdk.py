"""S3 wave 2, svc/sdk.py (owner P): frozen contract tests for MlopsClient / MlopsApiError.

Run: cd .mlops-control-plane && PYTHONPATH=exec python3 -m pytest tests/test_S3_3_sdk.py -q
(A) unit tests use httpx.MockTransport (no network, no sleeping: backoff_s=0.001).
(B) integration tests drive the real app from svc/app.py through an injected TestClient.
"""
import json
import re
import math

import httpx
import pytest

from mlops.svc.sdk import MlopsApiError, MlopsClient

TOKEN = "tok-SECRET-value-XYZ123"
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
HASH = "a" * 64


class Script:
    """Scripted MockTransport handler: replays `steps` (Response or Exception); last step repeats."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        i = min(len(self.requests) - 1, len(self.steps) - 1)
        step = self.steps[i]
        if isinstance(step, Exception):
            raise step
        return step

    def client(self, **kw):
        http = httpx.Client(transport=httpx.MockTransport(self), base_url="http://svc.test")
        kw.setdefault("backoff_s", 0.001)
        return MlopsClient(client=http, token=TOKEN, **kw)


def ok(body=None, status=200):
    return httpx.Response(status, json=body if body is not None else {"ok": True})


def err(status, code="some_code", message="boom"):
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


# ---------------------------------------------------------------- (A1) constructor

def test_constructor_requires_exactly_one_of_base_url_or_client():
    http = httpx.Client(transport=httpx.MockTransport(Script(ok())))
    with pytest.raises(ValueError):
        MlopsClient(token=TOKEN)
    with pytest.raises(ValueError):
        MlopsClient(base_url="http://x", client=http, token=TOKEN)
    MlopsClient(client=http, token=TOKEN)
    MlopsClient(base_url="http://svc.test", token=TOKEN)


@pytest.mark.parametrize("bad", ["", None, 123, b"tok", ["t"]])
def test_constructor_rejects_bad_token(bad):
    http = httpx.Client(transport=httpx.MockTransport(Script(ok())))
    with pytest.raises(ValueError):
        MlopsClient(client=http, token=bad)


def test_token_never_in_repr_or_error_messages():
    s = Script(err(403, "policy_denied", "nope"))
    c = s.client()
    assert TOKEN not in repr(c) and TOKEN not in str(c)
    with pytest.raises(MlopsApiError) as ei:
        c.register_model("m", "v1", HASH)
    assert TOKEN not in str(ei.value) and TOKEN not in repr(ei.value) and TOKEN not in ei.value.message
    # transport failure and bad response paths also stay clean
    c2 = Script(httpx.ConnectError("connect failed")).client(retries=1)
    with pytest.raises(MlopsApiError) as ei2:
        c2.health()
    assert TOKEN not in str(ei2.value) and TOKEN not in repr(ei2.value)
    c3 = Script(httpx.Response(200, text="<html>")).client()
    with pytest.raises(MlopsApiError) as ei3:
        c3.health()
    assert TOKEN not in str(ei3.value) and TOKEN not in repr(ei3.value)
    with pytest.raises(ValueError) as ei4:
        Script(ok()).client().decide("a", {"x": float("nan")})
    assert TOKEN not in str(ei4.value)


# ---------------------------------------------------------------- (A2) headers

def test_bearer_on_every_call_and_idempotency_key_only_on_post():
    s = Script(ok())
    c = s.client()
    c.health()
    c.get_model("m")
    c.active_policy()
    c.audit_head()
    c.register_model("m", "v1", HASH)
    c.decide("act", {"a": 1})
    posts = [r for r in s.requests if r.method == "POST"]
    gets = [r for r in s.requests if r.method == "GET"]
    assert len(posts) == 2 and len(gets) == 4
    for r in s.requests:
        assert r.headers["Authorization"] == f"Bearer {TOKEN}"
    for r in gets:
        assert "Idempotency-Key" not in r.headers
    keys = [r.headers.get("Idempotency-Key", "") for r in posts]
    assert all(KEY_RE.match(k) for k in keys)
    assert keys[0] != keys[1]


def test_metrics_text_returns_raw_text():
    s = Script(httpx.Response(200, text="# HELP x\nmlops_audit_length 3.0\n", headers={"content-type": "text/plain"}))
    text = s.client().metrics_text()
    assert isinstance(text, str) and "mlops_audit_length 3.0" in text
    assert s.requests[0].method == "GET" and s.requests[0].url.path == "/metrics"
    assert "Idempotency-Key" not in s.requests[0].headers


# ---------------------------------------------------------------- (A3) retries

def test_retries_503_then_success_reuse_same_idempotency_key():
    s = Script(err(503, "unavailable"), err(503, "unavailable"), ok({"model_id": "m"}, 201))
    out = s.client().register_model("m", "v1", HASH)
    assert out == {"model_id": "m"}
    assert len(s.requests) == 3
    keys = {r.headers["Idempotency-Key"] for r in s.requests}
    assert len(keys) == 1 and KEY_RE.match(next(iter(keys)))
    assert len({r.content for r in s.requests}) == 1  # identical body each attempt


@pytest.mark.parametrize("status", [502, 503, 504])
def test_transient_statuses_are_retried(status):
    s = Script(err(status), ok())
    assert s.client().health() == {"ok": True}
    assert len(s.requests) == 2


@pytest.mark.parametrize("status,code", [
    (500, "internal"), (400, "validation"), (401, "unauthorized"), (403, "policy_denied"),
    (404, "notfound"), (409, "conflict"), (422, "validation"),
])
def test_other_statuses_not_retried_and_envelope_mapped(status, code):
    s = Script(err(status, code, "msg-" + code), ok())
    with pytest.raises(MlopsApiError) as ei:
        s.client().register_model("m", "v1", HASH)
    assert len(s.requests) == 1
    assert ei.value.status == status and ei.value.code == code
    assert ei.value.message == "msg-" + code


def test_retry_budget_and_final_error_type():
    s = Script(err(503, "unavailable", "down"))
    with pytest.raises(MlopsApiError) as ei:
        s.client(retries=1).decide("a", {})
    assert len(s.requests) == 2  # retries=1 -> at most 2 attempts
    assert ei.value.status == 503 and ei.value.code == "unavailable"
    s0 = Script(err(503))
    with pytest.raises(MlopsApiError):
        s0.client(retries=0).health()
    assert len(s0.requests) == 1
    s3 = Script(err(503))
    with pytest.raises(MlopsApiError):
        s3.client().health()  # default retries=3 -> 4 attempts
    assert len(s3.requests) == 4


@pytest.mark.parametrize("exc", [httpx.ConnectError("no route"), httpx.ReadTimeout("slow")])
def test_transport_errors_retried_then_surface_as_transport_error(exc):
    s = Script(exc)
    with pytest.raises(MlopsApiError) as ei:
        s.client(retries=2).health()
    assert len(s.requests) == 3
    assert ei.value.status == 0 and ei.value.code == "transport"
    s2 = Script(exc, ok({"status": "ok"}))
    assert s2.client().health() == {"status": "ok"}
    assert len(s2.requests) == 2


def test_non_json_bodies_are_bad_response():
    s = Script(httpx.Response(502, text="<html>bad gateway</html>", headers={"content-type": "text/html"}))
    with pytest.raises(MlopsApiError) as ei:
        s.client(retries=0).health()
    assert ei.value.code == "bad_response" and ei.value.status == 502
    s2 = Script(httpx.Response(404, text="<html>nope</html>"))
    with pytest.raises(MlopsApiError) as ei2:
        s2.client().health()
    assert ei2.value.code == "bad_response" and len(s2.requests) == 1
    s3 = Script(httpx.Response(200, text="not json at all"))
    with pytest.raises(MlopsApiError) as ei3:
        s3.client().health()
    assert ei3.value.code == "bad_response"


# ---------------------------------------------------------------- (A4) URL and body building

def test_register_and_transition_request_shapes():
    s = Script(ok({}, 201))
    c = s.client()
    c.register_model("m", "v1", HASH)
    c.register_model("m", "v2", HASH, dataset_version="ds1")
    r1, r2 = s.requests
    assert r1.method == "POST" and r1.url.path == "/v1/models"
    assert json.loads(r1.content) == {"model_id": "m", "version_id": "v1", "artifact_hash": HASH}
    assert json.loads(r2.content) == {"model_id": "m", "version_id": "v2", "artifact_hash": HASH, "dataset_version": "ds1"}
    s2 = Script(ok())
    c2 = s2.client()
    c2.transition("m", "v1", "staging")
    c2.transition("m", "v1", "production", context={"acc": 0.9})
    a, b = s2.requests
    assert a.url.path == "/v1/models/m/versions/v1/transition"
    assert json.loads(a.content) == {"to_stage": "staging"}
    assert json.loads(b.content) == {"to_stage": "production", "context": {"acc": 0.9}}


def test_path_segments_are_percent_encoded():
    s = Script(ok())
    c = s.client()
    c.transition("a/b c", "v?1#x", "staging")
    c.get_model("x/../y z")
    assert s.requests[0].url.raw_path == b"/v1/models/a%2Fb%20c/versions/v%3F1%23x/transition"
    assert s.requests[1].url.raw_path == b"/v1/models/x%2F..%2Fy%20z"


def test_other_routes_use_contract_paths_and_verbs():
    s = Script(ok())
    c = s.client()
    docs = [{"kind": "Pod"}]
    c.lint(docs)
    c.lint(docs, image_allowlist=["reg.io"])
    c.drift_check([1.0, 2.0], [1.5, 2.5])
    c.decide("deploy", {"acc": 0.9})
    c.publish_policy(rules=[], name="p", version=1)
    c.activate_policy(3)
    c.activate_policy(4, human_approved_by="alice")
    c.audit_head()
    c.active_policy()
    reqs = s.requests
    expect = [
        ("POST", "/v1/lint"), ("POST", "/v1/lint"), ("POST", "/v1/drift/check"), ("POST", "/v1/policy/decide"),
        ("POST", "/v1/policy/publish"), ("POST", "/v1/policy/3/activate"), ("POST", "/v1/policy/4/activate"),
        ("GET", "/v1/audit/head"), ("GET", "/v1/policy/active"),
    ]
    assert [(r.method, r.url.path) for r in reqs] == expect
    assert json.loads(reqs[0].content) == {"documents": docs}  # image_allowlist omitted when None
    assert json.loads(reqs[1].content) == {"documents": docs, "image_allowlist": ["reg.io"]}
    assert json.loads(reqs[2].content) == {"reference": [1.0, 2.0], "window": [1.5, 2.5]}
    assert json.loads(reqs[3].content) == {"action": "deploy", "context": {"acc": 0.9}}
    pub = json.loads(reqs[4].content)
    assert pub["name"] == "p" and pub["version"] == 1 and pub["rules"] == []
    assert json.loads(reqs[5].content) == {}  # key omitted when human_approved_by is None
    assert json.loads(reqs[6].content) == {"human_approved_by": "alice"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_rejected_before_any_request(bad):
    s = Script(ok())
    c = s.client()
    with pytest.raises(ValueError):
        c.decide("a", {"x": bad})
    with pytest.raises(ValueError):
        c.drift_check([1.0, bad], [1.0])
    with pytest.raises(ValueError):
        c.validate_model("m", "v1", {"nested": [{"v": bad}]})
    with pytest.raises(ValueError):
        c.transition("m", "v1", "staging", context={"x": bad})
    assert s.requests == []


def test_non_json_able_objects_rejected_before_any_request():
    s = Script(ok())
    c = s.client()
    for evidence in ({"x": object()}, {"x": {1, 2}}, {"x": b"bytes"}):
        with pytest.raises((TypeError, ValueError)):
            c.validate_model("m", "v1", evidence)
    assert s.requests == []


# ---------------------------------------------------------------- (B) integration against the real app

@pytest.fixture
def app_parts():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from mlops.svc.app import create_app
    from mlops.svc.settings import Settings

    tokens = {
        "viewer-token-0000000001": {"name": "vera", "role": "viewer"},
        "operator-token-000000002": {"name": "olga", "role": "operator"},
        "admin-token-00000000003": {"name": "adam", "role": "admin"},
    }
    settings = Settings(audit_secret="s" * 32, tokens=tokens)
    tc = TestClient(create_app(settings))

    def make(token):
        return MlopsClient(client=tc, token=token, backoff_s=0.001)

    return tc, make, tokens


def test_end_to_end_register_validate_stage_production_as_admin(app_parts):
    _, make, _ = app_parts
    admin = make("admin-token-00000000003")
    assert admin.health()["status"] == "ok"
    admin.register_model("fraud", "v1", HASH)
    admin.validate_model("fraud", "v1", {"auc": 0.93})
    admin.transition("fraud", "v1", "staging")
    admin.transition("fraud", "v1", "production")
    hist = admin.get_model("fraud")
    assert isinstance(hist, list) and len(hist) >= 1


def test_role_and_auth_failures_map_to_status(app_parts):
    _, make, _ = app_parts
    with pytest.raises(MlopsApiError) as ei:
        make("viewer-token-0000000001").register_model("m", "v1", HASH)
    assert ei.value.status == 403
    with pytest.raises(MlopsApiError) as ei2:
        make("totally-wrong-token-xxxx").audit_head()
    assert ei2.value.status == 401
    assert "totally-wrong-token-xxxx" not in str(ei2.value)


def test_policy_publish_activate_decide_and_lint(app_parts):
    _, make, _ = app_parts
    admin = make("admin-token-00000000003")
    op = make("operator-token-000000002")
    rules = [{"id": "acc", "version": 1, "kind": "min_metric", "params": {"metric": "acc", "min": 0.9}, "severity": "block"}]
    admin.publish_policy(rules=rules, name="gate", version=1)
    admin.activate_policy(1)  # first activation needs no approval
    assert admin.active_policy() is not None
    allow = op.decide("promote", {"acc": 0.95})
    deny = op.decide("promote", {"acc": 0.5})
    assert allow["allow"] is True and deny["allow"] is False
    deployment = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "a", "namespace": "prod"},
                  "spec": {"replicas": 1, "template": {"spec": {"containers": [{"name": "c", "image": "nginx:latest"}]}}}}
    out = make("viewer-token-0000000001").lint([deployment])
    assert isinstance(out["findings"], list) and len(out["findings"]) > 0 and "score" in out


def test_audit_head_grows_on_write_not_on_rejected_write(app_parts):
    _, make, _ = app_parts
    admin = make("admin-token-00000000003")
    before = admin.audit_head()["length"]
    admin.register_model("m", "v1", HASH)
    after = admin.audit_head()["length"]
    assert after > before
    with pytest.raises(MlopsApiError):
        make("viewer-token-0000000001").register_model("m", "v2", HASH)
    with pytest.raises(MlopsApiError) as ei:
        admin.register_model("m", "v3", "not-a-hash")
    assert ei.value.status in (400, 422)
    assert admin.audit_head()["length"] == after


def test_metrics_text_from_real_app(app_parts):
    _, make, _ = app_parts
    viewer = make("viewer-token-0000000001")
    viewer.audit_head()
    assert "mlops_http_requests_total" in viewer.metrics_text()


def test_retry_after_lost_response_replays_and_writes_single_audit_entry(app_parts):
    tc, make, tokens = app_parts
    admin = make("admin-token-00000000003")
    base = admin.audit_head()["length"]
    seen = []

    def handler(request):
        seen.append(request)
        resp = tc.request(request.method, request.url.path, headers={
            "Authorization": request.headers["Authorization"],
            "Idempotency-Key": request.headers["Idempotency-Key"],
            "Content-Type": request.headers["Content-Type"],
        }, content=request.content)
        if len(seen) == 1:  # the app processed the request, but the response is lost
            return httpx.Response(503, json={"error": {"code": "unavailable", "message": "lost"}})
        return httpx.Response(resp.status_code, content=resp.content, headers={"content-type": "application/json"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    flaky = MlopsClient(client=http, token="operator-token-000000002", backoff_s=0.001)
    rec = flaky.register_model("idem", "v1", HASH)
    assert isinstance(rec, dict)
    assert len(seen) == 2
    assert seen[0].headers["Idempotency-Key"] == seen[1].headers["Idempotency-Key"]
    assert admin.audit_head()["length"] == base + 1
