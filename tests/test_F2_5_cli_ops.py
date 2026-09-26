"""F2.5 wave 2, svc/cli_ops.py + svc/sdk.py + svc/cli.py additions: frozen contract tests.

Run: cd .mlops-control-plane && PYTHONPATH=exec python3 -m pytest tests/test_F2_5_cli_ops.py -q

SCOPE DECISION (documented per the contract's F2 section 6 CORRECTION note): F1's incidents
router has no "list" route (only POST /v1/incidents, GET /v1/incidents/{id}, POST .../steps,
POST .../resolve). `list_open_incidents` / `incidents list-open` therefore has no backing HTTP
endpoint and is DROPPED from CLI scope. This file contains NO test for `incidents list-open` and
no test requires an SDK `list_open_incidents` method. Only `incidents open` and `incidents get`
are tested.

Two test groups:
(A) CLI tests -- `typer.testing.CliRunner` driving `mlops.svc.cli.create_cli(factory)`, the real
    integration point where `register_ops_commands` mounts the new commands (per contract section
    6). `factory` returns a small hand-written fake/stub client (never a real network client) so
    these tests pin CLI-level behavior (argument parsing, exit codes, output shape, error
    handling) without depending on a live server.
(B) SDK unit tests -- `httpx.MockTransport`, following the `Script`/`ok`/`err` pattern established
    in tests/test_S3_3_sdk.py, pinning the new `MlopsClient` methods' request shape (method+path
    +body) and return values, plus the contract-mandated NON-retry behavior of `audit_tail`
    (contrasted against `audit_export`, which DOES retry like every other method).

`sbom build`'s tests exercise the real parsing/error/output-routing logic in cli_ops.py end to
end, but only assert loosely on the eventual SBOM's content (component name/version strings
present as substrings in the serialized JSON) -- the exact CycloneDX schema shape is section 4's
contract (`mlops.ext.cyclonedx_sbom.build_cyclonedx_sbom`), tested elsewhere; this file only pins
this module's (cli_ops.py's) own requirements-line parsing, output routing, and error handling.
"""
import json

import httpx
import pytest
from typer.testing import CliRunner

from mlops.svc.cli import create_cli
from mlops.svc.sdk import MlopsApiError, MlopsClient

runner = CliRunner()

TOKEN = "tok-SECRET-value-XYZ123"


# =================================================================================================
# (A) CLI tests
# =================================================================================================

# ---------------------------------------------------------------- incidents open

class _OpenHappyClient:
    def open_incident(self, title, signal):
        assert title == "db is down"
        assert signal == {"metric": "error_rate", "value": 0.9}
        return {"incident_id": "inc-happy-0001", "severity": "P1", "deadline": 999.0}


def test_incidents_open_happy_path_prints_incident_id_exit_0(tmp_path):
    signal_file = tmp_path / "signal.json"
    signal_file.write_text(json.dumps({"metric": "error_rate", "value": 0.9}))
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["incidents", "open", "db is down", str(signal_file)])
    assert result.exit_code == 0
    assert result.output.strip() == "inc-happy-0001"  # plain text id, not JSON
    assert "Traceback" not in result.output


class _OpenDomainFailureClient:
    def open_incident(self, title, signal):
        raise MlopsApiError(400, "validation_failed", "signal must contain a numeric value")


def test_incidents_open_domain_failure_nonzero_exit_clean_message_no_traceback(tmp_path):
    signal_file = tmp_path / "signal.json"
    signal_file.write_text(json.dumps({"metric": "error_rate", "value": "not-a-number"}))
    app = create_cli(lambda: _OpenDomainFailureClient())
    result = runner.invoke(app, ["incidents", "open", "db is down", str(signal_file)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    error_lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(error_lines) <= 2  # a clean one-line message, not a dump
    assert "validation_failed" in result.output or "signal must contain a numeric value" in result.output


def test_incidents_open_bad_signal_json_exits_2_no_traceback(tmp_path):
    signal_file = tmp_path / "signal.json"
    signal_file.write_text("{not valid json")
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["incidents", "open", "title", str(signal_file)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


# ---------------------------------------------------------------- incidents get

_GET_RECORD = {
    "incident_id": "inc-get-0002",
    "title": "latency spike",
    "severity": "P0",
    "status": "open",
    "opened_ts": 1000.5,
    "deadline": 2000.0,
    "current_tier": 1,
    "steps_completed": [],
}


class _GetClient:
    def get_incident(self, incident_id):
        assert incident_id == "inc-get-0002"
        return dict(_GET_RECORD)


def test_incidents_get_json_default_contains_full_record():
    app = create_cli(lambda: _GetClient())
    result = runner.invoke(app, ["incidents", "get", "inc-get-0002"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed == _GET_RECORD


def test_incidents_get_table_flag_contains_id_and_severity_substrings():
    app = create_cli(lambda: _GetClient())
    result = runner.invoke(app, ["incidents", "get", "inc-get-0002", "--table"])
    assert result.exit_code == 0
    assert "inc-get-0002" in result.output
    assert "P0" in result.output
    # a rendered table, not the JSON dump: output must not parse as JSON nor carry JSON keys
    with pytest.raises(ValueError):
        json.loads(result.output)
    assert '"severity"' not in result.output and '"incident_id"' not in result.output
    assert "Traceback" not in result.output


class _GetNotFoundClient:
    def get_incident(self, incident_id):
        raise MlopsApiError(404, "not_found", "no such incident")


def test_incidents_get_not_found_exit_1_clean_message_no_traceback():
    app = create_cli(lambda: _GetNotFoundClient())
    result = runner.invoke(app, ["incidents", "get", "inc-missing"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "not_found" in result.output or "no such incident" in result.output


# ---------------------------------------------------------------- audit verify

class _AuditVerifyClient:
    def __init__(self, result):
        self._result = result

    def audit_verify(self, export, head):
        assert isinstance(export, list) and isinstance(head, dict)
        return dict(self._result)


def test_audit_verify_valid_pair_prints_valid_true_exit_0(tmp_path):
    export_file = tmp_path / "export.json"
    head_file = tmp_path / "head.json"
    export_file.write_text(json.dumps([{"actor": "a", "signature": "x"}]))
    head_file.write_text(json.dumps({"length": 1, "signature": "x"}))
    app = create_cli(lambda: _AuditVerifyClient({"valid": True}))
    result = runner.invoke(app, ["audit", "verify", str(export_file), str(head_file)])
    assert result.exit_code == 0
    assert "true" in result.output.lower()
    assert "Traceback" not in result.output


def test_audit_verify_tampered_pair_prints_valid_false_exit_1(tmp_path):
    export_file = tmp_path / "export.json"
    head_file = tmp_path / "head.json"
    export_file.write_text(json.dumps([{"actor": "a", "signature": "TAMPERED"}]))
    head_file.write_text(json.dumps({"length": 1, "signature": "x"}))
    app = create_cli(lambda: _AuditVerifyClient({"valid": False, "reason": "chain_broken"}))
    result = runner.invoke(app, ["audit", "verify", str(export_file), str(head_file)])
    # CLI-level UX decision (per contract): an honest "invalid" answer is a non-zero CLI exit,
    # even though the /v1/audit/verify API route itself returns HTTP 200 for it.
    assert result.exit_code == 1
    assert "false" in result.output.lower()
    assert "Traceback" not in result.output


# ---------------------------------------------------------------- audit tail

class _AuditTailClient:
    def __init__(self, events):
        self._events = events

    def audit_tail(self):
        for e in self._events:
            yield e


def test_audit_tail_prints_each_event_as_one_json_line():
    events = [
        {"event": "audit", "data": {"seq": 1, "actor": "a"}},
        {"event": "audit", "data": {"seq": 2, "actor": "a"}},
        {"event": "audit", "data": {"seq": 3, "actor": "b"}},
    ]
    app = create_cli(lambda: _AuditTailClient(events))
    result = runner.invoke(app, ["audit", "tail"])
    assert result.exit_code == 0
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 3
    for line, expected in zip(lines, events):
        assert json.loads(line) == expected


class _AuditTailKeyboardInterruptClient:
    def audit_tail(self):
        yield {"event": "audit", "data": {"seq": 1}}
        raise KeyboardInterrupt()


def test_audit_tail_keyboard_interrupt_exits_0_no_traceback():
    app = create_cli(lambda: _AuditTailKeyboardInterruptClient())
    result = runner.invoke(app, ["audit", "tail"])
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    assert "KeyboardInterrupt" not in result.output


# ---------------------------------------------------------------- sbom build

def test_sbom_build_parses_requirements_skips_comments_and_blanks_stdout(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "# a leading comment\n"
        "\n"
        "requests==2.31.0\n"
        "   \n"
        "# another comment\n"
        "flask==3.0.0\n"
    )
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["sbom", "build", str(req)])
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    json.loads(result.output)  # must be valid JSON
    assert "requests" in result.output and "2.31.0" in result.output
    assert "flask" in result.output and "3.0.0" in result.output


def test_sbom_build_writes_pretty_json_to_out_file(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26.4\n")
    out = tmp_path / "sbom.json"
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["sbom", "build", str(req), str(out)])
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    written = json.loads(out.read_text())
    assert "numpy" in json.dumps(written) and "1.26.4" in json.dumps(written)


def test_sbom_build_bad_line_exit_2_one_line_no_traceback(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("this-line-has-no-equals-signs\n")
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["sbom", "build", str(req)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    error_lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(error_lines) <= 2


def test_sbom_build_empty_file_exit_2_no_traceback(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("# only a comment\n\n")
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["sbom", "build", str(req)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_sbom_build_duplicate_names_exit_2_no_traceback(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("requests==2.31.0\nrequests==2.31.1\n")
    app = create_cli(lambda: _OpenHappyClient())
    result = runner.invoke(app, ["sbom", "build", str(req)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


# ---- messy real-world requirement lines (added by the round owner, 2026-09-26) ----------------
# Found by probing both coder clones: every frozen test passed while the SBOM recorded the VERSION
# `1.26.4  # pinned` (inline comment kept) and `306 ; sys_platform == "win32"` (marker kept). A supply-
# chain artifact must never carry silently wrong data, so: strip inline comments, strip extras from
# the name (they do not change the distribution), and REJECT environment markers (exit 2) because
# recording the pin unconditionally would over-report what is installed.

def _components(result):
    return {c["name"]: c["version"] for c in json.loads(result.output)["components"]}


def test_sbom_build_strips_inline_comment_from_the_version(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26.4  # pinned for the ABI\nscipy==1.11.0 #x\n")
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 0 and "Traceback" not in result.output
    assert _components(result) == {"numpy": "1.26.4", "scipy": "1.11.0"}


def test_sbom_build_strips_extras_from_the_component_name(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("requests[socks,security]==2.31.0\n")
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 0
    assert _components(result) == {"requests": "2.31.0"}


def test_sbom_build_rejects_environment_markers_exit_2(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text('pywin32==306 ; sys_platform == "win32"\n')
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len([l for l in result.output.splitlines() if l.strip()]) <= 2


@pytest.mark.parametrize("line", [
    "numpy==1.26.4 # c", "numpy==1.26.4\t# c", "numpy[extra]==1.26.4", "  numpy==1.26.4  ",
    "numpy == 1.26.4",
])
def test_sbom_build_never_records_junk_in_a_name_or_version(tmp_path, line):
    """Whatever the accepted spelling, the recorded name/version contain no whitespace, '#', ';' or '['."""
    req = tmp_path / "requirements.txt"
    req.write_text(line + "\n")
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    if result.exit_code == 0:
        for name, version in _components(result).items():
            assert name == "numpy" and version == "1.26.4"
            assert not any(ch in name + version for ch in " \t#;[]")
    else:
        assert result.exit_code == 2 and "Traceback" not in result.output


def test_sbom_build_never_echoes_a_rejected_lines_content(tmp_path):
    """A requirements file may hold credentialed URLs; a rejected line is reported by NUMBER only."""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26.4\ngit+https://bob:s3cr3t-tok3n@example.com/x/y.git#egg=y\n")
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 2 and "Traceback" not in result.output
    assert "s3cr3t-tok3n" not in result.output and "bob" not in result.output
    assert "2" in result.output          # the line number is still reported


def test_sbom_build_marker_error_does_not_echo_the_line_either(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text('pywin32==306 ; sys_platform == "win32-secretmarker"\n')
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 2 and "secretmarker" not in result.output


def test_sbom_build_utf8_bom_does_not_end_up_inside_the_first_name(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_bytes("\ufeffnumpy==1.26.4\n".encode("utf-8"))
    result = runner.invoke(create_cli(lambda: _OpenHappyClient()), ["sbom", "build", str(req)])
    assert result.exit_code == 0
    assert _components(result) == {"numpy": "1.26.4"}


# =================================================================================================
# (B) SDK unit tests (httpx.MockTransport, mirrors tests/test_S3_3_sdk.py's Script pattern)
# =================================================================================================

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


def test_sdk_open_incident_request_shape_and_response():
    s = Script(ok({"incident_id": "inc-1", "severity": "P2", "deadline": 5.0}, 201))
    out = s.client().open_incident("db down", {"metric": "err", "value": 1.0})
    assert out == {"incident_id": "inc-1", "severity": "P2", "deadline": 5.0}
    r = s.requests[0]
    assert r.method == "POST" and r.url.path == "/v1/incidents"
    assert json.loads(r.content) == {"title": "db down", "signal": {"metric": "err", "value": 1.0}}


def test_sdk_get_incident_request_shape_and_response_percent_encoded():
    s = Script(ok({"incident_id": "inc/1 x", "status": "open"}))
    out = s.client().get_incident("inc/1 x")
    assert out == {"incident_id": "inc/1 x", "status": "open"}
    r = s.requests[0]
    assert r.method == "GET"
    assert r.url.raw_path == b"/v1/incidents/inc%2F1%20x"


def test_sdk_complete_incident_step_request_shape_and_response():
    s = Script(ok({"status": "ok"}))
    out = s.client().complete_incident_step("inc-1", "contain", "rolled back")
    assert out == {"status": "ok"}
    r = s.requests[0]
    assert r.method == "POST" and r.url.path == "/v1/incidents/inc-1/steps"
    assert json.loads(r.content) == {"step": "contain", "note": "rolled back"}


def test_sdk_resolve_incident_request_shape_and_response():
    s = Script(ok({"status": "ok"}))
    out = s.client().resolve_incident("inc-1", postmortem={"summary": "fixed"})
    assert out == {"status": "ok"}
    r = s.requests[0]
    assert r.method == "POST" and r.url.path == "/v1/incidents/inc-1/resolve"
    assert json.loads(r.content) == {"postmortem": {"summary": "fixed"}}


def test_sdk_policy_versions_request_shape_and_response():
    s = Script(ok({"versions": [{"version": 1}]}))
    out = s.client().policy_versions()
    assert out == {"versions": [{"version": 1}]}
    r = s.requests[0]
    assert r.method == "GET" and r.url.path == "/v1/policy/versions"


def test_sdk_policy_version_request_shape_and_response():
    s = Script(ok({"version": 3, "name": "p"}))
    out = s.client().policy_version(3)
    assert out == {"version": 3, "name": "p"}
    r = s.requests[0]
    assert r.method == "GET" and r.url.path == "/v1/policy/versions/3"


def test_sdk_policy_decisions_default_and_custom_limit_request_shape():
    s = Script(ok({"decisions": []}))
    c = s.client()
    c.policy_decisions()
    c.policy_decisions(limit=5)
    r1, r2 = s.requests
    assert r1.url.path == "/v1/policy/decisions" and dict(r1.url.params) == {"limit": "50"}
    assert r2.url.path == "/v1/policy/decisions" and dict(r2.url.params) == {"limit": "5"}


def test_sdk_audit_export_request_shape_with_and_without_limit():
    s = Script(ok({"entries": [], "head": {}}))
    c = s.client()
    c.audit_export()
    c.audit_export(limit=10)
    r1, r2 = s.requests
    assert r1.url.path == "/v1/audit/export" and dict(r1.url.params) == {}
    assert r2.url.path == "/v1/audit/export" and dict(r2.url.params) == {"limit": "10"}


def test_sdk_audit_verify_request_shape_and_response():
    s = Script(ok({"valid": False, "reason": "chain_broken"}))
    export = [{"actor": "a", "signature": "x"}]
    head = {"length": 1, "signature": "x"}
    out = s.client().audit_verify(export, head)
    assert out == {"valid": False, "reason": "chain_broken"}
    r = s.requests[0]
    assert r.method == "POST" and r.url.path == "/v1/audit/verify"
    assert json.loads(r.content) == {"export": export, "head": head}


def test_sdk_audit_tail_yields_parsed_sse_dicts_from_data_lines():
    events = [
        {"event": "audit", "data": {"seq": 1}},
        {"event": "audit", "data": {"seq": 2}},
        {"event": "audit", "data": {"seq": 3}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    s = Script(httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    out = list(s.client().audit_tail())
    assert out == events


def test_sdk_audit_tail_not_wrapped_in_shared_retry_logic_unlike_audit_export():
    # audit_tail: a single transient failure must NOT be retried (unsafe on a half-open stream).
    s_tail = Script(httpx.ConnectError("connection dropped mid-stream"), ok())
    gen = s_tail.client().audit_tail()
    with pytest.raises(Exception):
        next(gen)
    assert len(s_tail.requests) == 1  # no retry attempted

    # Contrast: audit_export (an ordinary method) DOES go through the shared retry wrapper, so a
    # single transient 503 is retried transparently, just like every other method in test_S3_3_sdk.
    s_export = Script(err(503, "unavailable"), ok({"entries": [], "head": {}}))
    out = s_export.client().audit_export()
    assert out == {"entries": [], "head": {}}
    assert len(s_export.requests) == 2  # retried once, then succeeded
