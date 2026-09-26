"""S3.5 slim suite: constraints for the HIGH/MED findings of the review notes.

Written BEFORE the fixes: every test below targets a defect reproduced by probe.
Files under test: exec/mlops/svc/app.py, svc/deps.py, policy_store.py, sqldb.py, svc/__main__.py.
"""
import inspect
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXEC = os.path.join(ROOT, "exec")
sys.path.insert(0, EXEC)

import httpx
from fastapi.testclient import TestClient

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.kernel import ValidationFailed
from mlops.model_stages import ModelRegistry
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore
from mlops.svc.app import create_app
from mlops.svc.settings import Settings

SECRET = b"s" * 20
TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}
H1 = "11" * 32
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


def reg_body(m="m", v="v1", h=H1, **extra):
    return {"model_id": m, "version_id": v, "artifact_hash": h, **extra}


def mpath(m, v, act):
    return f"/v1/models/{m}/versions/{v}/{act}"


def raw_post(c, path, text, role="admin", **hdr):
    return c.post(path, content=text, headers={**H(role), **JSON_CT, **hdr})


def assert_envelope(r, status=None):
    if status is not None:
        assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and {"code", "message"} <= set(body["error"])
    assert "Traceback" not in r.text


# ------------------------------------------------------------ stub registries
class StubRegistry(ModelRegistry):
    """Real registry whose register() can block on an event / raise on chosen calls."""

    def __init__(self, audit, block=False, fail_first=False):
        super().__init__(audit=audit)
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.fail_first = fail_first

    def register(self, *a, **kw):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("boom")
        self.entered.set()
        if self.block:
            assert self.release.wait(15), "stub never released"
        return super().register(*a, **kw)


KEY = "key-0001-abcdef"


# =========================================================== (1) in_flight
def test_in_flight_duplicate_is_409_and_not_reexecuted():
    audit = ChainedAuditStore(SECRET)
    stub = StubRegistry(audit, block=True)
    app = create_app(make_settings(), audit=audit, registry=stub)
    hdr = H("admin", **{"Idempotency-Key": KEY})
    results = {}

    def run_a():
        results["a"] = TestClient(app, raise_server_exceptions=False).post("/v1/models", json=reg_body(), headers=hdr)

    t = threading.Thread(target=run_a)
    t.start()
    try:
        assert stub.entered.wait(10)
        rb = TestClient(app, raise_server_exceptions=False).post("/v1/models", json=reg_body(), headers=hdr)
        assert rb.status_code == 409, rb.text
        assert rb.json()["error"]["code"] == "idempotency_in_flight"
        assert rb.headers.get("retry-after") == "1"
        assert stub.calls == 1
    finally:
        stub.release.set()
        t.join(20)
    assert results["a"].status_code == 201
    rc = TestClient(app, raise_server_exceptions=False).post("/v1/models", json=reg_body(), headers=hdr)
    assert rc.status_code == 201 and rc.headers.get("idempotent-replayed") == "true"
    assert rc.json() == results["a"].json()
    assert stub.calls == 1


# =========================================================== (2) aborted keys
def test_unexpected_exception_aborts_key_so_retry_runs():
    audit = ChainedAuditStore(SECRET)
    stub = StubRegistry(audit, fail_first=True)
    c, _ = make_client(audit=audit, registry=stub)
    hdr = H("admin", **{"Idempotency-Key": KEY})
    r1 = c.post("/v1/models", json=reg_body(), headers=hdr)
    assert_envelope(r1, 500)
    r2 = c.post("/v1/models", json=reg_body(), headers=hdr)
    assert r2.status_code == 201, r2.text
    assert "idempotent-replayed" not in r2.headers
    assert stub.calls == 2


def test_domain_error_is_not_cached_under_key():
    audit = ChainedAuditStore(SECRET)
    stub = StubRegistry(audit)
    c, _ = make_client(audit=audit, registry=stub)
    assert c.post("/v1/models", json=reg_body(), headers=H("admin")).status_code == 201   # calls == 1
    hdr = H("admin", **{"Idempotency-Key": KEY})
    r1 = c.post("/v1/models", json=reg_body(), headers=hdr)
    r2 = c.post("/v1/models", json=reg_body(), headers=hdr)
    assert r1.status_code == 409 and r2.status_code == 409
    assert "idempotent-replayed" not in r2.headers
    assert stub.calls == 3


# =========================================================== (3) bearer tokens
@pytest.mark.parametrize("token", [
    "tök€n-ünïcode".encode("utf-8"),
    "é".encode("latin-1"),
    b"x" * 10000,
    b"tk-viewer\x00-0001",
    b"tk-viewer\x7f-0001",
])
def test_odd_bearer_tokens_are_401_envelope(token):
    c, _ = make_client()
    r = c.get("/v1/audit/head", headers={"Authorization": b"Bearer " + token})
    assert_envelope(r, 401)
    assert r.headers.get("www-authenticate") == "Bearer"


def test_bearer_token_with_inner_space_is_401():
    c, _ = make_client()
    r = c.get("/v1/audit/head", headers={"Authorization": "Bearer tk-viewer 0001"})
    assert_envelope(r, 401)


# =========================================================== (4) policy publish
def _rule(**kw):
    r = {"id": "r", "version": 1, "kind": "min_metric", "params": {"metric": "acc", "min": 0.9}, "severity": "block"}
    r.update(kw)
    return r


def _pub(c, rules, name="p"):
    return raw_post(c, "/v1/policy/publish", json.dumps({"name": name, "version": 1, "rules": rules}))


BAD_RULES = {
    "missing_params": [_rule(params={})],
    "min_string": [_rule(params={"metric": "acc", "min": "0.9"})],
    "unknown_kind": [_rule(kind="bogus")],
    "rule_not_object": ["x"],
    "duplicate_ids": [_rule(), _rule()],
    "bad_severity": [_rule(severity="loud")],
    "params_not_dict": [_rule(params="x")],
    "metric_not_str": [_rule(params={"metric": 5, "min": 0.9})],
}


@pytest.mark.parametrize("name", sorted(BAD_RULES))
def test_publish_rejects_malformed_rules(name):
    c, audit = make_client()
    r = _pub(c, BAD_RULES[name])
    assert r.status_code in (400, 422), r.text
    assert_envelope(r)


def test_publish_rejects_infinite_min():
    c, _ = make_client()
    text = ('{"name":"p","version":1,"rules":[{"id":"r","version":1,"kind":"min_metric",'
            '"params":{"metric":"acc","min":1e999},"severity":"block"}]}')
    r = raw_post(c, "/v1/policy/publish", text)
    assert r.status_code in (400, 422), r.text


def test_publish_wellformed_bundle_then_activate():
    c, _ = make_client()
    r = _pub(c, [_rule()])
    assert r.status_code == 201, r.text
    a = raw_post(c, f"/v1/policy/{r.json()['version']}/activate", "{}")
    assert a.status_code == 200, a.text


def test_policystore_publish_validates_and_consumes_no_version():
    store = PolicyStore(audit=ChainedAuditStore(SECRET))
    with pytest.raises(ValidationFailed):
        store.publish("root", PolicyBundle([Rule("r", 1, "min_metric", {}, "block")], "n", 1))
    good = PolicyBundle([Rule("r", 1, "min_metric", {"metric": "acc", "min": 0.9}, "block")], "n", 1)
    assert store.publish("root", good) == 1


def test_policystore_activate_never_raises_type_or_attribute_error():
    store = PolicyStore(audit=ChainedAuditStore(SECRET))
    good = PolicyBundle([Rule("r", 1, "min_metric", {"metric": "acc", "min": 0.9}, "block")], "n", 1)
    v1 = store.publish("root", good)
    store.activate("root", v1)
    try:
        v2 = store.publish("root", PolicyBundle([Rule("r", 1, "min_metric", {}, "block")], "n", 2))
    except ValidationFailed:
        return
    try:
        store.activate("root", v2)
    except (TypeError, AttributeError) as e:  # pragma: no cover
        pytest.fail(f"activate raised {type(e).__name__}")
    except Exception:
        pass


# =========================================================== (5) drift 500s
FIN = [0.01 * i for i in range(100)]


def _drift(c, ref, win):
    return raw_post(c, "/v1/drift/check", json.dumps({"reference": ref, "window": win}), role="viewer")


@pytest.mark.parametrize("ref,win", [
    (["a" * 40] * 40, FIN),
    (FIN, ["a" * 40] * 40),
    ([True] * 40, FIN),
    (FIN, [True] * 40),
    (FIN, [[1.0]] * 40),
    ([[1.0]] * 40, FIN),
    (FIN, [None] * 40),
    ([None] * 40, FIN),
])
def test_drift_bad_values_are_422(ref, win):
    c, _ = make_client()
    r = _drift(c, ref, win)
    assert r.status_code in (400, 422), r.text
    assert_envelope(r)


def test_drift_inf_literal_is_4xx():
    c, _ = make_client()
    body = '{"reference":[%s],"window":[%s]}' % (",".join(["1e999"] * 40), ",".join(["1.0"] * 40))
    r = raw_post(c, "/v1/drift/check", body, role="viewer")
    assert 400 <= r.status_code < 500, r.text
    body = '{"reference":[%s],"window":[%s]}' % (",".join(["1.0"] * 40), ",".join(["1e999"] * 40))
    r = raw_post(c, "/v1/drift/check", body, role="viewer")
    assert 400 <= r.status_code < 500, r.text


def test_drift_too_few_window_values_409_and_good_body_200():
    c, _ = make_client()
    r = _drift(c, FIN, [0.1] * 5)
    assert_envelope(r, 409)
    assert _drift(c, FIN, FIN).status_code == 200


# =========================================================== (6) operator vs production
def _to_production(c, m="m", v="v1"):
    assert c.post("/v1/models", json=reg_body(m, v), headers=H("admin")).status_code == 201
    assert c.post(mpath(m, v, "validate"), json={"evidence": {"k": 1}}, headers=H("admin")).status_code == 200
    assert c.post(mpath(m, v, "transition"), json={"to_stage": "staging"}, headers=H("admin")).status_code == 200
    assert c.post(mpath(m, v, "transition"), json={"to_stage": "production"}, headers=H("admin")).status_code == 200


def test_operator_cannot_archive_production_or_rollback_admin_can():
    c, _ = make_client()
    _to_production(c)
    r = c.post(mpath("m", "v1", "transition"), json={"to_stage": "archived"}, headers=H("operator"))
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_denied"
    r = c.post("/v1/models/m/rollback", json={}, headers=H("operator"))
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_denied"
    r = c.post(mpath("m", "v1", "transition"), json={"to_stage": "archived"}, headers=H("admin"))
    assert r.status_code == 200 and r.json()["stage"] == "archived"


def test_operator_can_archive_from_staging():
    c, _ = make_client()
    assert c.post("/v1/models", json=reg_body(), headers=H("admin")).status_code == 201
    assert c.post(mpath("m", "v1", "validate"), json={"evidence": {"k": 1}}, headers=H("admin")).status_code == 200
    assert c.post(mpath("m", "v1", "transition"), json={"to_stage": "staging"}, headers=H("operator")).status_code == 200
    r = c.post(mpath("m", "v1", "transition"), json={"to_stage": "archived"}, headers=H("operator"))
    assert r.status_code == 200 and r.json()["stage"] == "archived"


# =========================================================== (7) event loop
def _endpoint(app, path, method):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", ()):
            return r.endpoint
    raise AssertionError(f"route {method} {path} not found")


@pytest.mark.parametrize("path", ["/v1/lint", "/v1/drift/check"])
def test_cpu_bound_handlers_are_plain_def(path):
    app = create_app(make_settings(), audit=ChainedAuditStore(SECRET))
    assert not inspect.iscoroutinefunction(_endpoint(app, path, "POST"))


def test_slow_lint_does_not_block_healthz(monkeypatch):
    import mlops.svc.app as appmod
    import uvicorn

    monkeypatch.setattr(appmod, "lint_documents", lambda *a, **k: (time.sleep(1.0), [])[1])
    app = create_app(make_settings(), audit=ChainedAuditStore(SECRET))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        slow = threading.Thread(target=lambda: httpx.post(
            base + "/v1/lint", json={"documents": []}, headers=H("viewer"), timeout=10))
        slow.start()
        time.sleep(0.2)
        t0 = time.time()
        assert httpx.get(base + "/healthz", timeout=5).status_code == 200
        assert time.time() - t0 < 0.5
        slow.join(10)
    finally:
        server.should_exit = True
        th.join(10)


# =========================================================== (7b) 1e999
def test_inf_literal_in_register_body_is_422():
    c, _ = make_client()
    text = '{"model_id":"m","version_id":"v","artifact_hash":"%s","dataset_version":1e999}' % H1
    assert raw_post(c, "/v1/models", text).status_code == 422


def test_negative_inf_context_never_allows():
    c, _ = make_client()
    p = _pub(c, [{"id": "r", "version": 1, "kind": "max_metric",
                  "params": {"metric": "loss", "max": 0.5}, "severity": "block"}])
    assert p.status_code == 201, p.text
    assert raw_post(c, f"/v1/policy/{p.json()['version']}/activate", "{}").status_code == 200
    r = raw_post(c, "/v1/policy/decide", '{"action":"a","context":{"loss":-1e999}}', role="operator")
    assert r.status_code != 500
    assert r.status_code == 422 or (r.status_code == 200 and r.json()["allow"] is False), r.text


# =========================================================== (8) docs in prod
def test_openapi_requires_auth_in_prod_and_docs_hidden():
    c, _ = make_client(make_settings(env_name="prod"))
    r = c.get("/openapi.json")
    assert r.status_code == 401
    assert c.get("/openapi.json", headers=H("viewer")).status_code == 200
    assert c.get("/docs").status_code == 404
    assert c.get("/redoc").status_code == 404


def test_openapi_public_in_dev():
    c, _ = make_client(make_settings(env_name="dev"))
    assert c.get("/openapi.json").status_code == 200


# =========================================================== (9) db_path durability
def test_db_path_gives_durable_audit_across_restart(tmp_path):
    settings = make_settings(db_path=str(tmp_path / "a.db"))
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        assert c.post("/v1/models", json=reg_body(), headers=H("admin")).status_code == 201
        before = c.get("/v1/audit/head", headers=H("viewer")).json()["length"]
        assert before >= 1
    with TestClient(create_app(settings), raise_server_exceptions=False) as c2:
        head = c2.get("/v1/audit/head", headers=H("viewer")).json()
        assert head["length"] == before
        ex = c2.get("/v1/audit/export", headers=H("admin")).json()
        assert ChainVerifier(SECRET).verify_export(ex["entries"], head=ex["head"]) is True


# =========================================================== (10) sqldb created_ts
def test_upsert_model_preserves_created_ts(tmp_path):
    from mlops import sqldb
    url = f"sqlite:///{tmp_path}/t.db"
    sqldb.upgrade(url)
    repo = sqldb.SqlRepo(url)
    try:
        rec = {"model_id": "m", "version_id": "v1", "artifact_hash": "a" * 64, "dataset_version": None,
               "stage": "registered", "validated": False, "previous_production": None,
               "created_ts": 1.0, "updated_ts": 1.0}
        repo.upsert_model(rec)
        repo.upsert_model({**rec, "stage": "staging", "created_ts": 5.0, "updated_ts": 5.0})
        row = repo.get_model("m")[0]
        assert row["created_ts"] == 1.0 and row["updated_ts"] == 5.0 and row["stage"] == "staging"
    finally:
        repo.close()


# =========================================================== (11) entry point
def _run_main(*args):
    env = {**os.environ, "PYTHONPATH": EXEC}
    return subprocess.run([sys.executable, "-m", "mlops.svc", *args], capture_output=True, text=True,
                          env=env, timeout=60)


def test_main_help_mentions_flags():
    p = _run_main("--help")
    assert p.returncode == 0
    for flag in ("--host", "--port", "--config"):
        assert flag in p.stdout


def test_main_missing_config_exits_2_without_traceback(tmp_path):
    p = _run_main("--config", str(tmp_path / "nope.json"))
    assert p.returncode == 2
    assert "Traceback" not in p.stderr + p.stdout
    assert len((p.stderr + p.stdout).strip().splitlines()) == 1


def test_main_check_builds_app_and_exits_0(tmp_path):
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"audit_secret": "s" * 20,
                               "tokens": {TOK["admin"]: {"name": "root", "role": "admin"}}}))
    p = _run_main("--config", str(cfg), "--port", "0", "--check")
    assert p.returncode == 0, p.stderr


def test_serve_exists_and_main_module_imports():
    import importlib
    import mlops.svc.app as appmod
    assert callable(getattr(appmod, "serve", None))
    importlib.import_module("mlops.svc.__main__")
