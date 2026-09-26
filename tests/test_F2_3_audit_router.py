"""F2.3 slim suite: audit router (GET /v1/audit/tail SSE, POST /v1/audit/verify).
Contract: the API contract section 3 (svc/routers/audit.make_audit_router).
Frozen BEFORE mlops.svc.routers.audit / mlops.svc.audit_stream exist.

Fixture pattern follows tests/test_F1_5_routers.py: the router is mounted onto the real
create_app() via an `extensions=[...]` entry (defined locally here, so this suite does not
depend on the section-6 wiring in svc/extensions.py). That keeps the real auth (401),
role (403) and error-envelope (422) machinery in play.

Decisions documented here (the contract leaves them implicit):
  * make_audit_router(audit, chain_verifier_cls) calls `chain_verifier_cls()` with NO
    arguments, but the real ChainVerifier needs the HMAC secret, so tests pass
    `functools.partial(ChainVerifier, SECRET)` (any zero-arg callable works).
  * Starlette's TestClient only returns once the ASGI app finishes the response, and the
    real tail_events(audit) defaults (limit=1000, poll 0.5s) would never finish. The
    streaming tests therefore patch `tail_events` (in BOTH mlops.svc.audit_stream and
    mlops.svc.routers.audit) with a wrapper that forces poll_interval_s=0.01, limit=2,
    and one raw-ASGI test drives the UNPATCHED default generator and disconnects after
    the first `data:` chunk (bounded by asyncio.wait_for, never blocks indefinitely).
  * sse-starlette renders a dict `data` with str(), so a `data:` line is NOT JSON; the
    tests only assert that an entry's field values appear in it.
  * Rejections (401/403/422) add exactly zero audit entries; auth (401/403) is checked
    before body validation (422).
"""
import asyncio
import functools
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.svc.app import create_app
from mlops.svc.settings import Settings
from mlops.svc import audit_stream as audit_stream_mod
from mlops.svc.routers import audit as audit_router_mod
from mlops.svc.routers.audit import make_audit_router

SECRET = b"f2-3-router-secret-0000"
TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}
SHORT_CODE = re.compile(r"^[a-z0-9_]{1,40}$")

_REAL_TAIL = audit_stream_mod.tail_events


def make_settings(**kw):
    tokens = {TOK[r]: {"name": NAME[r], "role": r} for r in TOK}
    return Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **kw)


def _audit_extension(chain_verifier_cls):
    def ext(app, state):
        app.include_router(make_audit_router(state.audit, chain_verifier_cls))
    return ext


def make_client(chain_verifier_cls=None, entries=3):
    audit = ChainedAuditStore(SECRET)
    for i in range(entries):
        audit.append("alice", f"action{i}", f"res{i}", "allow", meta={"i": i})
    cls = chain_verifier_cls or functools.partial(ChainVerifier, SECRET)
    app = create_app(make_settings(), audit=audit, extensions=[_audit_extension(cls)])
    return TestClient(app, raise_server_exceptions=False), audit, app


def H(role):
    return {"Authorization": "Bearer " + TOK[role]}


def n(audit):
    return audit.head()["length"]


def err(r):
    body = r.json()
    assert set(body) == {"error"} and isinstance(body["error"]["code"], str)
    return body["error"]


def good_body(audit):
    return {"export": audit.export(), "head": audit.head()}


def flip_hex(s):
    return s[:-1] + ("0" if s[-1] != "0" else "1")


@pytest.fixture
def short_stream(monkeypatch):
    async def short(audit, **kw):
        # limit=2 ends a healthy stream; the 3s wall-clock deadline guarantees the response
        # ends even if the implementation under test stalls (a stall then fails an assertion
        # instead of hanging TestClient, which only returns once the response completes).
        agen = _REAL_TAIL(audit, poll_interval_s=0.01, limit=2)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3.0
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return
                try:
                    item = await asyncio.wait_for(agen.__anext__(), remaining)
                except (StopAsyncIteration, asyncio.TimeoutError):
                    return
                yield item
        finally:
            await agen.aclose()
    monkeypatch.setattr(audit_stream_mod, "tail_events", short)
    monkeypatch.setattr(audit_router_mod, "tail_events", short, raising=False)


def read_stream(c, role="operator", max_lines=40):
    """Bounded read of an SSE response (the patched generator ends by itself)."""
    lines = []
    with c.stream("GET", "/v1/audit/tail", headers=H(role)) as r:
        status, ctype = r.status_code, r.headers.get("content-type", "")
        for line in r.iter_lines():
            lines.append(line)
            if len(lines) >= max_lines:
                break
    return status, ctype, lines


# ------------------------------------------------------------------ GET /v1/audit/tail

def test_tail_viewer_is_403_and_adds_no_audit(short_stream):
    c, audit, _ = make_client()
    before = n(audit)
    r = c.get("/v1/audit/tail", headers=H("viewer"))
    assert r.status_code == 403
    assert err(r)["code"] == "policy_denied"
    assert n(audit) == before


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-0000"}])
def test_tail_missing_or_bad_token_is_401(short_stream, headers):
    c, _, _ = make_client()
    r = c.get("/v1/audit/tail", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].lower().startswith("bearer")


@pytest.mark.parametrize("role", ["operator", "admin"])
def test_tail_streams_event_stream_with_data_lines(short_stream, role):
    c, _, _ = make_client()
    status, ctype, lines = read_stream(c, role)
    assert status == 200
    assert ctype.startswith("text/event-stream")
    assert any(line.startswith("data:") for line in lines)
    assert any(line.startswith("event:") and "audit" in line for line in lines)


def test_tail_data_lines_carry_real_entries_in_order(short_stream):
    c, audit, _ = make_client()
    _, _, lines = read_stream(c)
    data_lines = [line for line in lines if line.startswith("data:")]
    entries = audit.export()
    assert len(data_lines) == 2                                  # the patched limit
    for line, entry in zip(data_lines, entries):
        assert entry["signature"] in line
        assert entry["action"] in line


def test_tail_data_lines_are_valid_json_equal_to_the_entries(short_stream):
    """Wire-format guard (added by the round reviewer, 2026-09-26): sse-starlette renders a dict
    `data` with str(), i.e. a Python repr with single quotes and `True`, which json.loads rejects
    and which the SDK's audit_tail (it json-parses each data line) cannot read. The router must
    hand sse-starlette a JSON STRING, so every data: line is parseable JSON equal to the entry."""
    import json as _json
    c, audit, _ = make_client()
    _, _, lines = read_stream(c)
    data_lines = [line for line in lines if line.startswith("data:")]
    entries = audit.export()
    assert len(data_lines) == 2
    for line, entry in zip(data_lines, entries):
        parsed = _json.loads(line[len("data:"):].strip())
        assert parsed == _json.loads(_json.dumps(entry))


def test_tail_stream_never_contains_the_secret(short_stream):
    c, _, _ = make_client()
    _, _, lines = read_stream(c, "admin")
    text = "\n".join(lines)
    assert "data:" in text
    assert SECRET.decode() not in text
    assert SECRET.hex() not in text


def test_tail_default_generator_streams_then_disconnects_cleanly():
    """UNPATCHED tail_events (limit=1000, poll 0.5s): first data chunk arrives at once for a
    non-empty store; a client disconnect ends the response without raising."""
    _, audit, app = make_client()

    async def scenario():
        disconnect = asyncio.Event()
        msgs = []

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(m):
            msgs.append(m)
            if m["type"] == "http.response.body" and b"data:" in m.get("body", b""):
                disconnect.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "path": "/v1/audit/tail", "raw_path": b"/v1/audit/tail", "query_string": b"",
            "headers": [(b"authorization", ("Bearer " + TOK["operator"]).encode()), (b"host", b"t")],
            "server": ("t", 80), "client": ("c", 1), "scheme": "http", "root_path": "",
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=8.0)
        return msgs

    msgs = asyncio.run(scenario())
    start = [m for m in msgs if m["type"] == "http.response.start"][0]
    assert start["status"] == 200
    ctype = dict(start["headers"])[b"content-type"].decode()
    assert ctype.startswith("text/event-stream")
    body = b"".join(m.get("body", b"") for m in msgs if m["type"] == "http.response.body")
    assert b"data:" in body
    assert SECRET not in body


# ------------------------------------------------------------------ POST /v1/audit/verify: auth

def test_verify_viewer_is_403_even_with_valid_body():
    c, audit, _ = make_client()
    r = c.post("/v1/audit/verify", json=good_body(audit), headers=H("viewer"))
    assert r.status_code == 403
    assert err(r)["code"] == "policy_denied"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-0000"}])
def test_verify_missing_or_bad_token_is_401(headers):
    c, audit, _ = make_client()
    r = c.post("/v1/audit/verify", json=good_body(audit), headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].lower().startswith("bearer")


def test_verify_auth_is_checked_before_body_validation():
    c, audit, _ = make_client()
    before = n(audit)
    assert c.post("/v1/audit/verify", json={"export": "nope"}, headers={}).status_code == 401
    assert c.post("/v1/audit/verify", json={"export": "nope"}, headers=H("viewer")).status_code == 403
    assert n(audit) == before


# ------------------------------------------------------------------ POST /v1/audit/verify: results

@pytest.mark.parametrize("role", ["operator", "admin"])
def test_verify_genuine_export_and_head_is_valid_true(role):
    c, audit, _ = make_client()
    r = c.post("/v1/audit/verify", json=good_body(audit), headers=H(role))
    assert r.status_code == 200, r.text
    assert r.json() == {"valid": True}


def test_verify_flipped_signature_char_is_200_valid_false_with_short_code():
    c, audit, _ = make_client()
    body = good_body(audit)
    body["export"][1]["signature"] = flip_hex(body["export"][1]["signature"])
    r = c.post("/v1/audit/verify", json=body, headers=H("operator"))
    assert r.status_code == 200, r.text                          # an EXPECTED outcome, not 4xx/5xx
    out = r.json()
    assert out["valid"] is False
    assert set(out) == {"valid", "reason"}
    assert isinstance(out["reason"], str) and SHORT_CODE.match(out["reason"]), out["reason"]
    for leak in ("Traceback", "Error", "Exception", "KeyError", "line "):
        assert leak not in out["reason"]


def test_verify_tampered_content_truncation_and_forged_head_are_all_valid_false():
    c, audit, _ = make_client()
    tampered = good_body(audit)
    tampered["export"][0]["decision"] = "deny"
    truncated = good_body(audit)
    truncated["export"] = truncated["export"][:-1]               # length no longer matches head
    forged = good_body(audit)
    forged["head"]["anchor"] = flip_hex(forged["head"]["anchor"])
    for body in (tampered, truncated, forged):
        r = c.post("/v1/audit/verify", json=body, headers=H("admin"))
        assert r.status_code == 200, r.text
        assert r.json()["valid"] is False
        assert SHORT_CODE.match(r.json()["reason"])


def test_verify_uses_the_supplied_chain_verifier_cls_with_export_and_head():
    calls = []

    class Stub:
        verdict = True

        def verify_export(self, export_records, head=None):
            calls.append((export_records, head))
            return Stub.verdict

    c, audit, _ = make_client(chain_verifier_cls=Stub)
    body = good_body(audit)
    r = c.post("/v1/audit/verify", json=body, headers=H("operator"))
    assert r.status_code == 200 and r.json() == {"valid": True}
    assert calls == [(body["export"], body["head"])]
    Stub.verdict = False
    r = c.post("/v1/audit/verify", json=body, headers=H("operator"))
    assert r.status_code == 200 and r.json()["valid"] is False
    assert len(calls) == 2


# ------------------------------------------------------------------ POST /v1/audit/verify: 422 body shapes

@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("export"),                                   # missing "export"
    lambda b: b.update(export="not-a-list"),                     # export not a list
    lambda b: b.update(export={"0": 1}),                         # export a dict
    lambda b: b.update(head=["not", "a", "dict"]),               # head not a dict
    lambda b: b.update(head="nope"),                             # head a string
])
def test_verify_malformed_body_is_422_with_zero_audit(mutate):
    c, audit, _ = make_client()
    body = good_body(audit)
    mutate(body)
    before = n(audit)
    r = c.post("/v1/audit/verify", json=body, headers=H("operator"))
    assert r.status_code == 422, r.text
    assert err(r)["code"] == "validation"
    assert n(audit) == before


# ------------------------------------------------------------------ secret hygiene

def test_verify_responses_never_contain_the_secret():
    c, audit, _ = make_client()
    ok = c.post("/v1/audit/verify", json=good_body(audit), headers=H("admin"))
    bad_body = good_body(audit)
    bad_body["export"][0]["signature"] = flip_hex(bad_body["export"][0]["signature"])
    bad = c.post("/v1/audit/verify", json=bad_body, headers=H("admin"))
    malformed = c.post("/v1/audit/verify", json={"export": 1, "head": {}}, headers=H("admin"))
    forbidden = c.post("/v1/audit/verify", json=good_body(audit), headers=H("viewer"))
    for r in (ok, bad, malformed, forbidden):
        assert SECRET.decode() not in r.text
        assert SECRET.hex() not in r.text
