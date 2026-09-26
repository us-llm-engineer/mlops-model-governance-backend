"""Regression tests added by the independent F2 review (the review notes).

NEW file: the seven frozen F2 test files were not touched. Each test pins a production defect that
was reproduced with a standalone probe first and then fixed:

 M1  sqldb_ops: concurrent first-insert of the SAME key failed with ValidationFailed (lost the
     SELECT-then-INSERT race) instead of upserting.
 M2  sqldb_ops: the ValidationFailed text embedded str(IntegrityError), i.e. the bound parameters
     (policy bundle / incident signal); a busy database surfaced a raw sqlalchemy OperationalError.
 M3  sqldb_ops: unhashable severity/status -> raw TypeError; huge ints -> raw OverflowError /
     StatementError; lone surrogates -> UnicodeEncodeError; title/name/incident_id accepted
     control characters; getters with a wrong-typed key -> raw ProgrammingError; OpsRepo skipped the
     URL check SqlRepo performs.
 M4  manifest_io: `&a [*a]` (self-referential anchor) and very deeply nested flow sequences under
     max_bytes -> raw RecursionError; non-str / lone-surrogate input -> raw AttributeError /
     UnicodeEncodeError.
 M5  POST /v1/audit/verify bypassed the app's body pipeline (no 413 size cap, no 415 content-type
     check) and answered valid:true for a truncated export when `head` was omitted or had no anchor.
 M6  GET /v1/audit/tail had no concurrency bound, no send timeout, and tail_events deep-copied the
     whole chain (under the store lock) on every poll of every connection.
 H1  MlopsClient.audit_tail died with a raw httpx.ReadTimeout after 5 s of silence (the server's
     keep-alive is every 15 s), and surfaced raw httpx/RecursionError/non-dict payloads.
 M7  `audit verify` exited 0 for a truthy non-bool "valid"; `sbom build` echoed a credentialed
     requirements URL (in the component name) to stderr; serial_number with a trailing newline
     raised a raw ValueError.
"""
import asyncio
import functools
import json
import os
import sys
import threading

import httpx
import pytest
import sqlalchemy as sa
from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient

from mlops import sqldb
from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.ext.cyclonedx_sbom import build_cyclonedx_sbom
from mlops.kernel import Conflict, ValidationFailed
from mlops.sqldb_ops import OpsRepo
from mlops.svc import audit_stream
from mlops.svc.app import create_app
from mlops.svc.cli import create_cli
from mlops.svc.manifest_io import load_manifests
from mlops.svc.routers import audit as audit_router_mod
from mlops.svc.routers.audit import make_audit_router
from mlops.svc.sdk import MlopsApiError, MlopsClient
from mlops.svc.settings import Settings

SECRET = b"f2-review-secret-000"
TOK = {"operator": "tk-operator-01"}


# --------------------------------------------------------------------------- sqldb_ops fixtures

@pytest.fixture
def repo(tmp_path):
    url = f"sqlite:///{tmp_path}/ops.db"
    sqldb.upgrade(url)
    r = OpsRepo(url)
    yield r
    r.close()


def _policy(version=1, **kw):
    rec = dict(version=version, name="p", bundle_json='{"k": 1}', note="", published_ts=1.0, active=False)
    rec.update(kw)
    return rec


def _incident(**kw):
    rec = dict(incident_id="i1", title="t", severity="P1", status="open", opened_ts=1.0, signal_json="{}")
    rec.update(kw)
    return rec


# --------------------------------------------------------------------------- M1: concurrent upserts

def test_concurrent_first_insert_of_same_policy_version_never_fails_and_keeps_one_active(repo):
    errors = []
    for version in range(1, 16):
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            try:
                repo.upsert_policy_version(_policy(version, active=True))
            except BaseException as e:  # noqa: BLE001 - any failure is the defect
                errors.append(f"{type(e).__name__}")

        threads = [threading.Thread(target=worker) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert [r["version"] for r in repo.list_policy_versions() if r["active"]] == [version]
    assert errors == []


def test_concurrent_upserts_of_incident_and_drift_baseline_never_fail(repo):
    errors = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        try:
            repo.upsert_incident(_incident())
            repo.upsert_drift_baseline("b", [float(i) for i in range(30)])
        except BaseException as e:  # noqa: BLE001
            errors.append(type(e).__name__)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert repo.count_open_incidents_by_severity() == {"P0": 0, "P1": 1, "P2": 0, "P3": 0}


# --------------------------------------------------------------------------- M2: error hygiene

def test_integrity_error_message_never_contains_bound_parameters(repo):
    def work(session):
        raise sa.exc.IntegrityError("INSERT ...", {"bundle_json": "hunter2-SECRET"}, Exception("UNIQUE failed"))

    with pytest.raises(ValidationFailed) as ei:
        repo._run_write(work)
    assert "hunter2" not in str(ei.value) and "INSERT" not in str(ei.value)


def test_busy_database_is_a_typed_conflict_not_a_raw_operational_error(repo):
    def work(session):
        raise sa.exc.OperationalError("INSERT ...", {"note": "hunter2"}, Exception("database is locked"))

    with pytest.raises(Conflict) as ei:
        repo._run_write(work)
    assert "hunter2" not in str(ei.value)


# --------------------------------------------------------------------------- M3: validation

@pytest.mark.parametrize("call", [
    lambda r: r.upsert_incident(_incident(severity=["P1"])),
    lambda r: r.upsert_incident(_incident(status={})),
    lambda r: r.upsert_incident(_incident(severity=None)),
    lambda r: r.upsert_incident(_incident(title="a\x00b")),
    lambda r: r.upsert_incident(_incident(title="a\nb")),
    lambda r: r.upsert_incident(_incident(incident_id="a\x1b[31m")),
    lambda r: r.upsert_incident(_incident(opened_ts=10**400)),
    lambda r: r.upsert_incident(_incident(signal_json="a\ud800")),
    lambda r: r.upsert_policy_version(_policy(2**70)),
    lambda r: r.upsert_policy_version(_policy(published_ts=10**400)),
    lambda r: r.upsert_policy_version(_policy(name="a\x00b")),
    lambda r: r.upsert_policy_version(_policy(name="a\ud800b")),
    lambda r: r.upsert_policy_version(_policy(note="a\ud800")),
    lambda r: r.upsert_drift_baseline("b", [10**400] * 30),
    lambda r: r.upsert_drift_baseline("b\ud800", [1.0] * 30),
    lambda r: r.get_policy_version(2**70),
    lambda r: r.get_policy_version(True),
    lambda r: r.get_policy_version("1"),
    lambda r: r.get_incident([1]),
    lambda r: r.get_incident(None),
    lambda r: r.get_incident("a\ud800"),
    lambda r: r.get_drift_baseline(None),
])
def test_bad_input_is_validation_failed_never_a_raw_exception(repo, call):
    with pytest.raises(ValidationFailed):
        call(repo)


@pytest.mark.parametrize("url", ["postgresql://u:pw@h/db", "mysql://x", "", None, "sqlite:///a\x00b.db"])
def test_ops_repo_applies_the_same_url_check_as_sql_repo(url):
    with pytest.raises(ValidationFailed):
        OpsRepo(url)


def test_valid_records_still_round_trip_after_hardening(repo):
    repo.upsert_policy_version(_policy(1, active=True, note="multi\nline ok", bundle_json='{"a":\n1}'))
    repo.upsert_incident(_incident(signal_json='{"x":\n1}'))
    assert repo.get_policy_version(1)["active"] is True
    assert repo.get_incident("i1")["signal_json"] == '{"x":\n1}'


# --------------------------------------------------------------------------- M4: manifest_io

@pytest.mark.parametrize("raw", [
    "a: &x [*x]",
    "&x {a: *x}",
    "&x\na: *x",
    "a: " + "[" * 100000 + "]" * 100000,
    "a: " + "[" * 20000 + "]" * 20000,
])
def test_recursive_or_pathologically_nested_manifest_is_validation_failed(raw):
    with pytest.raises(ValidationFailed) as ei:
        load_manifests(raw)
    assert "[[[[" not in str(ei.value)


@pytest.mark.parametrize("raw", [None, b"a: 1", 5, ["a: 1"], "a: \ud800"])
def test_non_text_manifest_input_is_validation_failed(raw):
    with pytest.raises(ValidationFailed):
        load_manifests(raw)


# --------------------------------------------------------------------------- M5/M6: audit router

def _settings():
    return Settings(audit_secret="s" * 20, tokens={TOK["operator"]: {"name": "olga", "role": "operator"}},
                    _env_file=None)


def _app(audit, **router_kw):
    def ext(app, state):
        app.include_router(make_audit_router(state.audit, functools.partial(ChainVerifier, SECRET), **router_kw))
    return create_app(_settings(), audit=audit, extensions=[ext])


def _store(n=3):
    audit = ChainedAuditStore(SECRET)
    for i in range(n):
        audit.append("alice", f"a{i}", "r", "allow", meta={"i": i})
    return audit


H = {"Authorization": "Bearer " + TOK["operator"]}


def test_verify_enforces_the_body_size_cap_with_413():
    audit = _store()
    c = TestClient(_app(audit), raise_server_exceptions=False)
    entry = {"actor": "a", "action": "x", "resource": "r", "decision": "allow", "ts": "t", "meta": {},
             "signature": "0" * 64}
    body = {"export": [entry] * 20000, "head": audit.head()}   # ~3-4 MB, cap is 1 MiB
    r = c.post("/v1/audit/verify", json=body, headers=H)
    assert r.status_code == 413, r.text
    assert r.json()["error"]["code"] == "payload_too_large"


def test_verify_requires_json_content_type_with_415():
    audit = _store()
    c = TestClient(_app(audit), raise_server_exceptions=False)
    body = json.dumps({"export": audit.export(), "head": audit.head()})
    assert c.post("/v1/audit/verify", content=body, headers=H).status_code == 415
    assert c.post("/v1/audit/verify", content=body, headers={**H, "content-type": "text/plain"}).status_code == 415


def test_verify_rejects_non_finite_and_over_deep_bodies_with_422():
    c = TestClient(_app(_store()), raise_server_exceptions=False)
    hdr = {**H, "content-type": "application/json"}
    assert c.post("/v1/audit/verify", content='{"export": [], "head": {"length": NaN}}', headers=hdr).status_code == 422
    deep = '{"export": [{"meta": ' + "[" * 500 + "]" * 500 + "}], \"head\": {}}"
    assert c.post("/v1/audit/verify", content=deep, headers=hdr).status_code == 422


def test_verify_never_says_valid_for_a_truncated_export_without_a_complete_head():
    audit = _store(3)
    c = TestClient(_app(audit), raise_server_exceptions=False)
    full = audit.export()
    truncated = full[:2]
    forged_head_no_anchor = {"length": 2, "signature": truncated[-1]["signature"]}
    for body in (
        {"export": truncated},                                        # head omitted
        {"export": truncated, "head": None},
        {"export": truncated, "head": forged_head_no_anchor},         # anchorless, self-consistent head
        {"export": truncated, "head": {}},
    ):
        r = c.post("/v1/audit/verify", json=body, headers=H)
        assert r.status_code == 422, (body.get("head"), r.text)
        assert "valid" not in r.json()
    # the genuine export + signed head is still accepted
    ok = c.post("/v1/audit/verify", json={"export": full, "head": audit.head()}, headers=H)
    assert ok.status_code == 200 and ok.json() == {"valid": True}
    # and a truncated export with the REAL head is an honest valid:false
    bad = c.post("/v1/audit/verify", json={"export": truncated, "head": audit.head()}, headers=H)
    assert bad.status_code == 200 and bad.json()["valid"] is False


def test_tail_events_does_not_export_when_nothing_changed(monkeypatch):
    audit = _store(2)
    calls = {"n": 0}
    real = audit.export

    def counting_export(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    audit.export = counting_export

    async def scenario():
        agen = audit_stream.tail_events(audit, poll_interval_s=0.01, limit=3)
        first = [await agen.__anext__(), await agen.__anext__()]
        n_after_first_batch = calls["n"]
        nxt = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.25)                     # ~25 idle polls
        idle_calls = calls["n"] - n_after_first_batch
        audit.append("bob", "late", "r", "allow")
        item = await asyncio.wait_for(nxt, 2.0)
        return first, idle_calls, item

    first, idle_calls, item = asyncio.run(scenario())
    assert idle_calls == 0
    assert item["data"]["action"] == "late"


def test_max_tails_validation():
    for bad in (0, -1, 1001, True, 1.5, "8", None):
        with pytest.raises(ValidationFailed):
            make_audit_router(_store(), lambda: None, max_tails=bad)


def _scope():
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "path": "/v1/audit/tail", "raw_path": b"/v1/audit/tail", "query_string": b"",
        "headers": [(b"authorization", ("Bearer " + TOK["operator"]).encode()), (b"host", b"t")],
        "server": ("t", 80), "client": ("c", 1), "scheme": "http", "root_path": "",
    }


def test_tail_concurrency_is_bounded_and_slots_are_released_on_disconnect(monkeypatch):
    async def endless(audit, **kw):
        yield {"event": "audit", "data": {"seq": 1}}
        await asyncio.sleep(3600)

    monkeypatch.setattr(audit_router_mod, "tail_events", endless)
    monkeypatch.setattr(audit_stream, "tail_events", endless)
    app = _app(_store(), max_tails=2)

    async def open_tail():
        """Returns (status, disconnect_event, task) once the response has started."""
        disconnect = asyncio.Event()
        started = asyncio.get_running_loop().create_future()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(m):
            if m["type"] == "http.response.start" and not started.done():
                started.set_result(m["status"])

        task = asyncio.ensure_future(app(_scope(), receive, send))
        status = await asyncio.wait_for(started, 5.0)
        return status, disconnect, task

    async def scenario():
        a = await open_tail()
        b = await open_tail()
        c = await open_tail()                         # over the cap of 2
        statuses_full = [a[0], b[0], c[0]]
        await asyncio.wait_for(c[2], 5.0)             # a 429 ends by itself
        a[1].set()                                    # client A disconnects
        await asyncio.wait_for(a[2], 5.0)             # ... and its task ends (no leak)
        d = await open_tail()                         # slot is free again
        statuses_after = d[0]
        for x in (b, d):
            x[1].set()
            await asyncio.wait_for(x[2], 5.0)
        return statuses_full, statuses_after

    full, after = asyncio.run(scenario())
    assert full == [200, 200, 429]
    assert after == 200


def test_tail_sets_a_send_timeout_so_a_stalled_client_cannot_pin_a_stream():
    from mlops.svc.routers.audit import _SSE_SEND_TIMEOUT_S
    assert 0 < _SSE_SEND_TIMEOUT_S <= 120
    captured = {}
    real_init = audit_router_mod.EventSourceResponse.__init__

    def spy(self, *a, **k):
        captured.update(k)
        real_init(self, *a, **k)

    audit_router_mod.EventSourceResponse.__init__ = spy
    try:
        async def scenario():
            app = _app(_store())
            disconnect = asyncio.Event()

            async def receive():
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(m):
                if m["type"] == "http.response.body" and b"data:" in m.get("body", b""):
                    disconnect.set()

            await asyncio.wait_for(app(_scope(), receive, send), 8.0)

        asyncio.run(scenario())
    finally:
        audit_router_mod.EventSourceResponse.__init__ = real_init
    assert captured.get("send_timeout") == _SSE_SEND_TIMEOUT_S


# --------------------------------------------------------------------------- H1: SDK audit_tail

def _client(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    return MlopsClient(client=http, token="tok")


def test_audit_tail_allows_silence_longer_than_httpx_default_read_timeout():
    seen = {}

    def handler(request):
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, content=b'data: {"a": 1}\n\n')

    assert list(_client(handler).audit_tail()) == [{"a": 1}]
    # Server keep-alive comment cadence is 15 s; httpx's default 5 s read timeout killed idle tails.
    assert seen["read"] is not None and seen["read"] >= 30


class _Boom(httpx.SyncByteStream):
    def __init__(self, exc):
        self.exc = exc

    def __iter__(self):
        yield b'data: {"a": 1}\n\n'
        raise self.exc


@pytest.mark.parametrize("exc", [
    httpx.RemoteProtocolError("peer closed connection"),
    httpx.ReadTimeout("timed out"),
    httpx.ReadError("reset"),
])
def test_audit_tail_transport_failures_surface_as_one_typed_error(exc):
    client = _client(lambda req: httpx.Response(200, stream=_Boom(exc)))
    got = []
    with pytest.raises(MlopsApiError) as ei:
        for ev in client.audit_tail():
            got.append(ev)
    assert got == [{"a": 1}]
    assert ei.value.code == "transport" and "peer closed" not in str(ei.value)


@pytest.mark.parametrize("payload", [
    b"data: 5\n\n",
    b"data: [1]\n\n",
    b'data: "str"\n\n',
    b"data: null\n\n",
    b'data: {"a": NaN}\n\n',
    b"data: " + b"[" * 100000 + b"]" * 100000 + b"\n\n",
])
def test_audit_tail_rejects_non_object_or_hostile_payloads_with_typed_error(payload):
    client = _client(lambda req: httpx.Response(200, content=payload))
    with pytest.raises(MlopsApiError) as ei:
        list(client.audit_tail())
    assert ei.value.code == "bad_response"


# --------------------------------------------------------------------------- M7: CLI / SBOM

runner = CliRunner()


def test_audit_verify_exits_nonzero_unless_valid_is_literally_true(tmp_path):
    ef, hf = tmp_path / "e.json", tmp_path / "h.json"
    ef.write_text("[]")
    hf.write_text("{}")

    class Client:
        def __init__(self, resp):
            self.resp = resp

        def audit_verify(self, export, head):
            return self.resp

    for resp in ({"valid": "yes"}, {"valid": 1}, {"valid": "false"}, {"valid": [1]}, {}):
        res = runner.invoke(create_cli(lambda r=resp: Client(r)), ["audit", "verify", str(ef), str(hf)])
        assert res.exit_code == 1, resp
        assert json.loads(res.stdout.strip().splitlines()[-1])["valid"] is False
    res = runner.invoke(create_cli(lambda: Client({"valid": True})), ["audit", "verify", str(ef), str(hf)])
    assert res.exit_code == 0


def test_sbom_build_never_echoes_a_credentialed_requirements_line(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("https://user:S3CR3T-pw@host.example/pkg==1.0\n")
    res = runner.invoke(create_cli(lambda: None), ["sbom", "build", str(req)])
    assert res.exit_code == 2
    assert "S3CR3T" not in res.output and "user:" not in res.output


@pytest.mark.parametrize("components", [
    [("https://u:S3CR3T@h/p", "1")],
    [("ok", "1@S3CR3T")],
    [{"nam": "S3CR3T"}],
    [(5, "S3CR3T")],
])
def test_build_cyclonedx_sbom_errors_never_echo_input_values(components):
    with pytest.raises(ValidationFailed) as ei:
        build_cyclonedx_sbom(components)
    assert "S3CR3T" not in str(ei.value)


@pytest.mark.parametrize("serial", [
    "urn:uuid:12345678-1234-1234-1234-123456789abc\n",
    "urn:uuid:12345678-1234-1234-1234-123456789abc ",
    "\nurn:uuid:12345678-1234-1234-1234-123456789abc",
])
def test_serial_number_with_trailing_or_leading_whitespace_is_validation_failed(serial):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([("a", "1")], serial_number=serial)
