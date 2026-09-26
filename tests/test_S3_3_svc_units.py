"""S3.3 slim suite: svc units (errors.py, deps.py, idempotency.py). Frozen BEFORE the code exists.
Throwaway FastAPI apps are built here; the real app (svc/app.py) is not used.
Assumption: get_principal/require_role find tokens via get_state(); tests override
get_state with SimpleNamespace(settings=SimpleNamespace(tokens={token: SimpleNamespace(name, role)})).
"""
import os
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, StrictInt

from mlops.kernel import (
    Conflict,
    IntegrityError,
    ManualClock,
    NotFound,
    PolicyDenied,
    ValidationFailed,
)
from mlops.svc import deps
from mlops.svc.deps import Principal, get_principal, get_state, require_role
from mlops.svc.errors import install_error_handlers
from mlops.svc.idempotency import IdempotencyStore

SECRET = "hunter2-secret"
TOK_ADMIN, TOK_OP, TOK_VIEW = "tok-admin-AAAA1111", "tok-oper-BBBB2222", "tok-view-CCCC3333"


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    replica_count: StrictInt
    label: str


def _fake_state():
    spec = lambda n, r: SimpleNamespace(name=n, role=r)  # noqa: E731
    return SimpleNamespace(
        settings=SimpleNamespace(
            tokens={
                TOK_ADMIN: spec("alice", "admin"),
                TOK_OP: spec("bob", "operator"),
                TOK_VIEW: spec("carol", "viewer"),
            }
        )
    )


def _make_app():
    app = FastAPI()
    install_error_handlers(app)
    app.dependency_overrides[get_state] = _fake_state

    def raiser(exc):
        def route():
            raise exc

        return route

    app.add_api_route("/validation", raiser(ValidationFailed("bad input")))
    app.add_api_route("/denied", raiser(PolicyDenied("nope")))
    app.add_api_route("/missing", raiser(NotFound("gone")))
    app.add_api_route("/conflict", raiser(Conflict("clash")))
    app.add_api_route("/integrity", raiser(IntegrityError("chain broken at 7 sig=deadbeef")))
    app.add_api_route("/boom", raiser(RuntimeError("secret traceback text")))
    app.add_api_route("/http404", raiser(HTTPException(status_code=404, detail="nothing here")))

    @app.post("/body")
    def body(b: Body):
        return {"n": b.replica_count}

    @app.get("/who")
    def who(p: Principal = Depends(get_principal)):
        return {"name": p.name, "role": p.role}

    @app.get("/viewer")
    def viewer(p: Principal = Depends(require_role("viewer", "operator", "admin"))):
        return {"name": p.name}

    @app.get("/admin")
    def admin(p: Principal = Depends(require_role("admin"))):
        return {"name": p.name}

    return app


@pytest.fixture()
def client():
    return TestClient(_make_app(), raise_server_exceptions=False)


def _bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def _err(resp):
    j = resp.json()
    assert set(j) == {"error"}
    assert set(j["error"]) == {"code", "message"}
    return j["error"]


# ------------------------------ (1) error handlers ---------------------------
@pytest.mark.parametrize(
    "path,status,code",
    [
        ("/validation", 400, "validation_failed"),
        ("/denied", 403, "policy_denied"),
        ("/missing", 404, "not_found"),
        ("/conflict", 409, "conflict"),
    ],
)
def test_kernel_errors_map_to_status_and_envelope(client, path, status, code):
    r = client.get(path)
    assert r.status_code == status
    e = _err(r)
    assert e["code"] == code
    assert isinstance(e["message"], str) and e["message"]


def test_integrity_error_hides_details(client):
    r = client.get("/integrity")
    assert r.status_code == 500
    e = _err(r)
    assert e["message"] == "integrity failure"
    assert "deadbeef" not in r.text and "chain broken" not in r.text


def test_unhandled_exception_is_opaque_500(client):
    r = client.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal", "message": "internal error"}}
    assert "secret traceback text" not in r.text
    assert "Traceback" not in r.text


def test_http_exception_uses_envelope(client):
    r = client.get("/http404")
    assert r.status_code == 404
    _err(r)
    r2 = client.get("/no-such-route")
    assert r2.status_code == 404
    _err(r2)


def test_request_validation_names_field_never_echoes_value(client):
    r = client.post("/body", json={"replica_count": SECRET, "label": "x"})
    assert r.status_code == 422
    e = _err(r)
    assert e["code"] == "validation"
    assert "replica_count" in e["message"]
    assert SECRET not in r.text


def test_extra_field_rejected_without_echo(client):
    r = client.post("/body", json={"replica_count": 1, "label": "x", "sneaky": SECRET})
    assert r.status_code == 422
    assert _err(r)["code"] == "validation"
    assert SECRET not in r.text


def test_bool_is_not_a_strict_int(client):
    r = client.post("/body", json={"replica_count": True, "label": "x"})
    assert r.status_code == 422
    assert _err(r)["code"] == "validation"
    ok = client.post("/body", json={"replica_count": 3, "label": "x"})
    assert ok.status_code == 200 and ok.json() == {"n": 3}


def test_nan_literal_rejected(client):
    r = client.post(
        "/body",
        content=b'{"replica_count": NaN, "label": "x"}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    assert _err(r)["code"] == "validation"


# ------------------------------ (2) auth -------------------------------------
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer unknown-token-ZZZZ9999"},
    ],
)
def test_bad_credentials_401_with_challenge(client, headers):
    r = client.get("/who", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    _err(r)
    assert "unknown-token-ZZZZ9999" not in r.text


def test_valid_tokens_resolve_principal(client):
    r = client.get("/who", headers=_bearer(TOK_VIEW))
    assert r.status_code == 200 and r.json() == {"name": "carol", "role": "viewer"}
    assert client.get("/viewer", headers=_bearer(TOK_VIEW)).status_code == 200


def test_role_matrix(client):
    assert client.get("/admin", headers=_bearer(TOK_VIEW)).status_code == 403
    assert client.get("/admin", headers=_bearer(TOK_OP)).status_code == 403
    r = client.get("/admin", headers=_bearer(TOK_ADMIN))
    assert r.status_code == 200 and r.json() == {"name": "alice"}
    denied = client.get("/admin", headers=_bearer(TOK_OP))
    assert _err(denied)["code"]
    assert TOK_OP not in denied.text


def test_principal_is_frozen():
    p = Principal(name="a", role="admin")
    with pytest.raises(Exception):
        p.role = "viewer"


def test_token_check_uses_compare_digest(client, monkeypatch):
    import hmac as real_hmac

    calls = []
    orig = real_hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return orig(a, b)

    if hasattr(deps, "compare_digest"):
        monkeypatch.setattr(deps, "compare_digest", spy)
    else:
        monkeypatch.setattr(deps.hmac, "compare_digest", spy)
    r = client.get("/who", headers=_bearer(TOK_OP))
    assert r.status_code == 200
    assert calls, "compare_digest must be used for token checks"
    flat = [x if isinstance(x, (str, bytes)) else "" for pair in calls for x in pair]
    assert any(TOK_OP in (x if isinstance(x, str) else x.decode()) for x in flat)
    n_ok = len(calls)
    calls.clear()
    assert client.get("/who", headers=_bearer("unknown-token-ZZZZ9999")).status_code == 401
    assert len(calls) >= 3, "unknown token must be compared against all configured tokens"
    assert n_ok >= 1


# ------------------------------ (3) idempotency ------------------------------
def _store(ttl=60, max_entries=10_000):
    clock = ManualClock(1000.0)
    return IdempotencyStore(clock, ttl, max_entries=max_entries), clock


def _req(key="key-00000001", principal="alice", method="POST", path="/v1/x", h="h1"):
    return (key, principal, method, path, h)


def test_new_finish_replay_and_conflict():
    st, _ = _store()
    assert st.begin(*_req()) == ("new", None)
    st.finish(*_req(), status=201, body={"id": 1})
    kind, cached = st.begin(*_req())
    assert kind == "replay" and tuple(cached) == (201, {"id": 1})
    assert st.begin(*_req(h="h2"))[0] == "conflict"
    assert st.begin(*_req(path="/v1/y"))[0] == "conflict"
    assert st.begin(*_req(method="PUT"))[0] == "conflict"


def test_keys_scoped_per_principal():
    st, _ = _store()
    assert st.begin(*_req(principal="alice"))[0] == "new"
    st.finish(*_req(principal="alice"), status=200, body={})
    assert st.begin(*_req(principal="bob"))[0] == "new"


def test_in_flight_and_abort():
    st, _ = _store()
    assert st.begin(*_req())[0] == "new"
    assert st.begin(*_req())[0] == "in_flight"
    st.abort(*_req())
    assert st.begin(*_req())[0] == "new"


def test_expiry_after_ttl():
    st, clock = _store(ttl=60)
    st.begin(*_req())
    st.finish(*_req(), status=200, body={"a": 1})
    clock.advance(30)
    assert st.begin(*_req())[0] == "replay"
    clock.advance(31 + 60)
    assert st.begin(*_req())[0] == "new"


def test_bounded_evicts_oldest():
    st, clock = _store(max_entries=3)
    keys = [f"key-0000000{i}" for i in range(1, 6)]
    for k in keys:
        assert st.begin(*_req(key=k))[0] == "new"
        st.finish(*_req(key=k), status=200, body={"k": k})
        clock.advance(1)
    assert st.begin(*_req(key=keys[4]))[0] == "replay"
    assert st.begin(*_req(key=keys[3]))[0] == "replay"
    assert st.begin(*_req(key=keys[0]))[0] == "new"


@pytest.mark.parametrize("bad", ["short", "has space in it", "a" * 129, "bad/chars-here", 12345678, None, b"key-00000001"])
def test_invalid_key_syntax_rejected(bad):
    st, _ = _store()
    with pytest.raises(ValidationFailed):
        st.begin(*_req(key=bad))
    ok = st.begin(*_req(key="a" * 128))
    assert ok[0] == "new"


def test_concurrent_begin_single_winner():
    st, _ = _store()
    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        r = st.begin(*_req())[0]
        with lock:
            results.append(r)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert len(results) == 8
    assert results.count("new") == 1
    assert all(r in ("new", "in_flight") for r in results)


def test_replayed_body_is_not_an_alias():
    st, _ = _store()
    st.begin(*_req())
    orig = {"items": [1, 2], "n": 1}
    st.finish(*_req(), status=200, body=orig)
    orig["items"].append(99)  # caller mutates its own object after finish
    _, cached = st.begin(*_req())
    assert cached[1] == {"items": [1, 2], "n": 1}
    cached[1]["items"].append(7)
    cached[1]["n"] = 42
    _, again = st.begin(*_req())
    assert again[1] == {"items": [1, 2], "n": 1}
