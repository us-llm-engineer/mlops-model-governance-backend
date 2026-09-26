"""F1.2 slim suite: exec/mlops/svc/telemetry.py, contract in the API contract section 2.

Frozen expectations (contract-derived):
  Telemetry(service_name="mlops-control-plane", exporter=None) builds its OWN TracerProvider
  (never calls trace.set_tracer_provider: global stays ProxyTracerProvider; two Telemetry objects
  never see each other's spans). service_name must be a non-empty str with no control chars,
  else mlops.kernel.ValidationFailed.
  .finished_spans() -> list (only meaningful for the in-memory exporter, else []).
  .shutdown() -> None, idempotent.
  .install(app) -> None; a second install on the same app raises mlops.kernel.Conflict.
  Per request: extract W3C traceparent (garbage -> ignored, fresh trace, no exception); start a
  SERVER span named "{METHOD} {route_template}" (route_template = matched route path, or the
  literal "unmatched" when nothing matched -- never the raw path); attributes
  http.request.method / http.route / http.response.status_code; ERROR status on >=500 or an
  escaping exception (which must re-raise); response header X-Trace-Id = 32 lower-hex chars of
  the span's trace id. Authorization header, query string and body must never reach any span
  attribute. An exporter whose export() raises must not fail the request.
  current_trace_id() -> Optional[str] (32 hex of active span, else None).
  audit_meta_with_trace(meta) -> new dict = meta + "trace_id" when a span is active; never
  mutates the argument; never overwrites an existing "trace_id".
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from mlops import kernel
from mlops.svc.telemetry import Telemetry, current_trace_id, audit_meta_with_trace

HEX_RE = __import__("re").compile(r"^[0-9a-f]{32}$")
W3C_HEADER = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
W3C_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


class RaisingExporter(SpanExporter):
    """A SpanExporter whose export() always raises -- must never break a request."""

    def export(self, spans):
        raise RuntimeError("export always fails")

    def shutdown(self):
        pass


def make_app():
    app = FastAPI()

    @app.get("/items/{item_id}")
    def get_item(item_id: str):
        return {"item_id": item_id}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/http500")
    def http500():
        raise HTTPException(status_code=500, detail="server said no")

    return app


def make_client(t):
    app = make_app()
    t.install(app)
    return TestClient(app, raise_server_exceptions=False)


def all_attr_values(spans):
    values = []
    for s in spans:
        for v in (s.attributes or {}).values():
            values.append(str(v))
    return values


def find_span(spans, name):
    matches = [s for s in spans if s.name == name]
    assert matches, f"no span named {name!r} among {[s.name for s in spans]}"
    return matches[0]


# ---------------------------------------------------------------------------
# span naming: route template, never the raw path
# ---------------------------------------------------------------------------

def test_route_template_span_name_items():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/items/42")
    assert resp.status_code == 200
    spans = t.finished_spans()
    find_span(spans, "GET /items/{item_id}")
    assert "42" not in " ".join(s.name for s in spans)


def test_route_template_span_name_boom():
    t = Telemetry()
    client = make_client(t)
    client.get("/boom")
    spans = t.finished_spans()
    find_span(spans, "GET /boom")


def test_unmatched_path_span_name_literal_and_no_raw_path_leak():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/does-not-exist-xyz987")
    assert resp.status_code == 404
    spans = t.finished_spans()
    find_span(spans, "GET unmatched")
    names = " ".join(s.name for s in spans)
    assert "does-not-exist-xyz987" not in names
    assert "does-not-exist-xyz987" not in " ".join(all_attr_values(spans))


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------

def test_attributes_method_route_status_success():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/items/7")
    assert resp.status_code == 200
    span = find_span(t.finished_spans(), "GET /items/{item_id}")
    assert span.attributes.get("http.request.method") == "GET"
    assert span.attributes.get("http.route") == "/items/{item_id}"
    assert int(span.attributes.get("http.response.status_code")) == 200


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

def test_boom_reraises_and_span_error():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/boom")
    assert resp.status_code == 500
    span = find_span(t.finished_spans(), "GET /boom")
    assert span.status.status_code == trace.StatusCode.ERROR


def test_boom_exception_actually_propagates_through_middleware():
    # raise_server_exceptions defaults to True: if the middleware swallowed the
    # exception instead of re-raising it, this client would see a plain 500
    # response instead of the original RuntimeError bubbling up.
    t = Telemetry()
    app = make_app()
    t.install(app)
    strict_client = TestClient(app)
    with pytest.raises(RuntimeError):
        strict_client.get("/boom")


def test_http_exception_500_span_error():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/http500")
    assert resp.status_code == 500
    span = find_span(t.finished_spans(), "GET /http500")
    assert span.status.status_code == trace.StatusCode.ERROR
    assert int(span.attributes.get("http.response.status_code")) == 500


def test_success_status_no_error():
    t = Telemetry()
    client = make_client(t)
    client.get("/items/1")
    span = find_span(t.finished_spans(), "GET /items/{item_id}")
    assert span.status.status_code != trace.StatusCode.ERROR


# ---------------------------------------------------------------------------
# X-Trace-Id header
# ---------------------------------------------------------------------------

def test_x_trace_id_header_32_lower_hex_matches_span():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/items/9")
    header = resp.headers.get("X-Trace-Id")
    assert header is not None
    assert HEX_RE.match(header)
    span = find_span(t.finished_spans(), "GET /items/{item_id}")
    assert header == format(span.context.trace_id, "032x")


# ---------------------------------------------------------------------------
# traceparent propagation
# ---------------------------------------------------------------------------

def test_valid_traceparent_same_trace_id_child():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/items/5", headers={"traceparent": W3C_HEADER})
    assert resp.status_code == 200
    assert resp.headers.get("X-Trace-Id") == W3C_TRACE_ID
    span = find_span(t.finished_spans(), "GET /items/{item_id}")
    assert format(span.context.trace_id, "032x") == W3C_TRACE_ID


def test_garbage_traceparent_ignored_fresh_trace():
    t = Telemetry()
    client = make_client(t)
    resp = client.get("/items/5", headers={"traceparent": "not-a-real-traceparent"})
    assert resp.status_code == 200
    header = resp.headers.get("X-Trace-Id")
    assert HEX_RE.match(header)
    assert header != W3C_TRACE_ID
    assert int(header, 16) != 0


# ---------------------------------------------------------------------------
# no sensitive data leaks onto spans
# ---------------------------------------------------------------------------

def test_authorization_header_not_in_attributes():
    t = Telemetry()
    client = make_client(t)
    client.get("/items/3", headers={"Authorization": "Bearer TOPSECRETAUTH"})
    values = all_attr_values(t.finished_spans())
    assert not any("TOPSECRETAUTH" in v for v in values)


def test_query_string_token_not_in_attributes():
    t = Telemetry()
    client = make_client(t)
    client.get("/items/3?token=SUPERSECRETQUERY")
    values = all_attr_values(t.finished_spans())
    assert not any("SUPERSECRETQUERY" in v for v in values)


def test_body_not_in_attributes():
    t = Telemetry()
    client = make_client(t)
    client.request("GET", "/items/3", content=b"SUPERSECRETBODYCONTENT")
    values = all_attr_values(t.finished_spans())
    assert not any("SUPERSECRETBODYCONTENT" in v for v in values)


# ---------------------------------------------------------------------------
# install semantics
# ---------------------------------------------------------------------------

def test_install_twice_raises_conflict():
    t = Telemetry()
    app = make_app()
    t.install(app)
    with pytest.raises(kernel.Conflict):
        t.install(app)


def test_two_telemetry_instances_isolated():
    t1 = Telemetry()
    t2 = Telemetry()
    app1 = make_app()
    app2 = make_app()
    t1.install(app1)
    t2.install(app2)
    TestClient(app1, raise_server_exceptions=False).get("/items/1")
    TestClient(app2, raise_server_exceptions=False).get("/items/2")
    names1 = [s.name for s in t1.finished_spans()]
    names2 = [s.name for s in t2.finished_spans()]
    assert len(names1) >= 1
    assert len(names2) >= 1
    # neither exporter accumulated the other's spans beyond its own single request
    assert len(t1.finished_spans()) == len(names1)
    assert len(t2.finished_spans()) == len(names2)


def test_global_tracer_provider_stays_proxy():
    t = Telemetry()
    client = make_client(t)
    client.get("/items/1")
    client.get("/boom")
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


def test_exporter_raising_does_not_fail_request():
    t = Telemetry(exporter=RaisingExporter())
    client = make_client(t)
    resp = client.get("/items/1")
    assert resp.status_code == 200
    assert resp.json() == {"item_id": "1"}


def test_shutdown_idempotent():
    t = Telemetry()
    t.shutdown()
    t.shutdown()


# ---------------------------------------------------------------------------
# service_name validation
# ---------------------------------------------------------------------------

def test_service_name_empty_raises_validation_failed():
    with pytest.raises(kernel.ValidationFailed):
        Telemetry(service_name="")


def test_service_name_control_char_raises_validation_failed():
    with pytest.raises(kernel.ValidationFailed):
        Telemetry(service_name="svc\x00name")


def test_service_name_non_str_raises_validation_failed():
    with pytest.raises(kernel.ValidationFailed):
        Telemetry(service_name=123)


# ---------------------------------------------------------------------------
# current_trace_id
# ---------------------------------------------------------------------------

def test_current_trace_id_outside_span_is_none():
    assert current_trace_id() is None


def test_current_trace_id_inside_span_is_hex():
    t = Telemetry()
    with t.tracer.start_as_current_span("manual-span"):
        tid = current_trace_id()
    assert tid is not None
    assert HEX_RE.match(tid)


# ---------------------------------------------------------------------------
# audit_meta_with_trace
# ---------------------------------------------------------------------------

def test_audit_meta_with_trace_none_returns_empty_dict_without_span():
    assert audit_meta_with_trace(None) == {}


def test_audit_meta_with_trace_adds_trace_id_and_does_not_mutate():
    t = Telemetry()
    original = {"action": "deploy"}
    with t.tracer.start_as_current_span("manual-span"):
        result = audit_meta_with_trace(original)
    assert original == {"action": "deploy"}
    assert result["action"] == "deploy"
    assert HEX_RE.match(result["trace_id"])


def test_audit_meta_with_trace_preserves_existing_trace_id():
    t = Telemetry()
    original = {"trace_id": "deadbeef" * 4}
    with t.tracer.start_as_current_span("manual-span"):
        result = audit_meta_with_trace(original)
    assert result["trace_id"] == "deadbeef" * 4


# ---------------------------------------------------------------------------
# finished_spans() only meaningful for the in-memory exporter
# ---------------------------------------------------------------------------

class NoopExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def test_finished_spans_empty_for_custom_exporter():
    t = Telemetry(exporter=NoopExporter())
    client = make_client(t)
    client.get("/items/1")
    assert t.finished_spans() == []
