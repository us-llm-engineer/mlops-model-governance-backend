"""F1.6 slim suite: create_app(extensions=...) wiring, mlops.svc.extensions composition,
and the mann_whitney_u -> stats_scipy delegation. Frozen BEFORE these exist.
Contract: the API contract section 7.
"""
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from mlops.audit import ChainedAuditStore
from mlops.incidents import IncidentManager
from mlops.kernel import ManualClock, ValidationFailed
from mlops.policy_store import PolicyStore
from mlops.svc.app import create_app
from mlops.svc.settings import Settings

EXEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")

TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}


def make_settings(**kw):
    tokens = {TOK[r]: {"name": NAME[r], "role": r} for r in TOK}
    return Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **kw)


def H(role, **extra):
    return {"Authorization": "Bearer " + TOK[role], **extra}


# ------------------------------------------------------------------ (1) create_app(extensions=...)

def test_create_app_without_extensions_behaves_as_before():
    app = create_app(make_settings())
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/healthz").status_code == 200
    r = c.post("/v1/models", json={"model_id": "m", "version_id": "v1", "artifact_hash": "1" * 64},
               headers=H("admin"))
    assert r.status_code == 201


@pytest.mark.parametrize("bad", [
    5, "not-a-list", {},
    [1], [lambda a, b: None, "not-callable"],
])
def test_extensions_wrong_type_raises_validation_failed(bad):
    with pytest.raises(ValidationFailed):
        create_app(make_settings(), extensions=bad)


def test_extensions_none_is_accepted():
    app = create_app(make_settings(), extensions=None)
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/healthz").status_code == 200


def test_extensions_tuple_of_callables_is_accepted():
    calls = []

    def ext(app, state):
        calls.append((app, state))

    create_app(make_settings(), extensions=(ext,))
    assert len(calls) == 1


def test_extensions_called_in_order_after_routes_exist():
    order = []

    def first(app, state):
        order.append("first")
        assert any(r.path == "/healthz" for r in app.routes)
        assert state is app.state.mlops

    def second(app, state):
        order.append("second")

    create_app(make_settings(), extensions=[first, second])
    assert order == ["first", "second"]


# ------------------------------------------------------------------ (2) composed extensions

def _full_app(clock=None):
    from mlops.svc.extensions import (
        domain_metrics_extension,
        incidents_extension,
        policy_history_extension,
        telemetry_extension,
    )
    from mlops.svc.extensions import _IncidentCounts
    from mlops.svc.telemetry import Telemetry

    clock = clock or ManualClock(start=1000.0)
    audit = ChainedAuditStore(b"f1-6-secret-00000000")
    store = PolicyStore(audit=audit, clock=clock)
    manager = IncidentManager(audit, clock)
    telemetry = Telemetry(service_name="f1-6-test")
    app = create_app(
        make_settings(),
        audit=audit,
        policy_store=store,
        clock=clock,
        extensions=[
            telemetry_extension(telemetry),
            incidents_extension(manager),
            policy_history_extension(),
            domain_metrics_extension(audit=audit, policy_store=store, incidents=_IncidentCounts(manager)),
        ],
    )
    return app, audit, store, manager, telemetry


def test_composed_extensions_trace_header_incidents_and_metrics():
    app, audit, store, manager, telemetry = _full_app()
    c = TestClient(app, raise_server_exceptions=False)

    r = c.get("/healthz")
    assert r.status_code == 200
    assert len(r.headers.get("X-Trace-Id", "")) == 32

    opened = c.post("/v1/incidents", json={"title": "t", "signal": {"pipeline_delay": True}},
                     headers=H("operator"))
    assert opened.status_code == 201, opened.text

    metrics_text = c.get("/metrics", headers=H("viewer")).text
    assert "mlops_audit_chain_length" in metrics_text


def test_domain_metrics_extension_requires_metrics_registry():
    from mlops.svc.extensions import domain_metrics_extension

    ext = domain_metrics_extension(audit=ChainedAuditStore(b"x" * 20))
    fake_state = types.SimpleNamespace(metrics_registry=None)
    with pytest.raises(ValidationFailed):
        ext(app=None, state=fake_state)


def test_domain_metrics_extension_registers_on_real_registry():
    from mlops.svc.extensions import domain_metrics_extension

    ext = domain_metrics_extension(audit=ChainedAuditStore(b"x" * 20))
    fake_state = types.SimpleNamespace(metrics_registry=CollectorRegistry())
    ext(app=None, state=fake_state)  # must not raise


# ------------------------------------------------------------------ (3) _IncidentCounts

def test_incident_counts_all_severities_nonnegative_and_open_only():
    from mlops.svc.extensions import _IncidentCounts

    clock = ManualClock(start=1000.0)
    audit = ChainedAuditStore(b"y" * 20)
    manager = IncidentManager(audit, clock)
    counts = _IncidentCounts(manager)

    empty = counts.open_counts()
    assert set(empty) == {"P0", "P1", "P2", "P3"}
    assert all(isinstance(v, int) and v >= 0 for v in empty.values())
    assert empty == {"P0": 0, "P1": 0, "P2": 0, "P3": 0}

    iid1 = manager.open("alice", "t1", {"accuracy_drop_pct": 15.0})   # P1
    iid2 = manager.open("alice", "t2", {"feature_psi": 0.31})         # P2
    manager.open("alice", "t3", {"pipeline_delay": True})             # P3
    manager.open("alice", "t0", {"serving_failure": True})            # P0: the most severe
    after_open = counts.open_counts()
    assert after_open == {"P0": 1, "P1": 1, "P2": 1, "P3": 1}

    for step in ("detection", "impact_assessment", "recent_changes_review",
                 "mitigation_options", "root_cause"):
        manager.complete_step("alice", iid1, step, "n")
    manager.resolve("alice", iid1, {
        "timeline": "x", "root_cause": "x", "user_impact": "x",
        "preventive_measures": "x", "monitoring_gap": "x",
    })
    after_resolve = counts.open_counts()
    assert after_resolve == {"P0": 1, "P1": 0, "P2": 1, "P3": 1}   # resolved incident no longer counted


# ------------------------------------------------------------------ (4) mann_whitney_u delegation

def test_mann_whitney_u_matches_reference_on_no_tie_data():
    from mlops.legacy import metrics

    a, b = [1.0, 2.0], [3.0, 4.0]
    pub = metrics.mann_whitney_u(a, b)
    ref = metrics._mann_whitney_u_reference(a, b)
    assert pub == pytest.approx(ref, abs=1e-9)


def test_mann_whitney_u_matches_scipy_on_tied_data():
    from mlops.legacy import metrics
    from scipy import stats as scipy_stats

    a = [1.0, 1.0, 2.0, 5.0]
    b = [1.0, 2.0, 2.0, 6.0]
    pub_u, pub_p = metrics.mann_whitney_u(a, b)
    expected = scipy_stats.mannwhitneyu(a, b, alternative="less", method="asymptotic", use_continuity=False)
    assert pub_u == pytest.approx(float(expected.statistic), abs=1e-9)
    assert pub_p == pytest.approx(float(expected.pvalue), abs=1e-9)


def test_mann_whitney_u_all_tied_input_gives_p_one():
    from mlops.legacy import metrics

    u, p = metrics.mann_whitney_u([5.0, 5.0], [5.0, 5.0])
    assert p == 1.0


@pytest.mark.parametrize("a,b", [([1.0], [1.0, 2.0]), ([1.0, 2.0], [1.0])])
def test_mann_whitney_u_n_lt_2_raises_value_error(a, b):
    from mlops.legacy import metrics

    with pytest.raises(ValueError, match="requires at least 2 observations per group"):
        metrics.mann_whitney_u(a, b)


def test_mann_whitney_u_reference_exists_and_is_callable():
    from mlops.legacy import metrics

    assert callable(metrics._mann_whitney_u_reference)


# ------------------------------------------------------------------ (5) import hygiene

def test_import_mlops_does_not_pull_in_heavy_deps():
    code = (
        "import sys\n"
        "import mlops\n"
        "heavy = {'fastapi', 'opentelemetry', 'prometheus_client'}\n"
        "leaked = heavy & set(m.split('.')[0] for m in sys.modules)\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXEC_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
