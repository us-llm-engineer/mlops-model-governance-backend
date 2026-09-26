"""F3.6 slim suite: exec/mlops/svc/workers.py (anyio background workers) and the
tiny create_app(workers=...) wiring in exec/mlops/svc/app.py. Frozen BEFORE these
exist. Contract: the API contract sections 6 and 7.

No pytest-anyio/asyncio plugin is configured in this repo (checked: no conftest.py,
no pyproject.toml/pytest.ini anyio_mode setting, and test_S3_3_svc_units.py-style
suites never use `@pytest.mark.anyio`). Every async scenario below is therefore
driven with a plain synchronous `def test_...():` function that calls
`anyio.run(scenario)` on a small `async def scenario():` closure, per the harness
instructions for this file.

Every test uses short intervals (0.001-0.05s) and explicit anyio.Event stop
signals, and every scenario that could hang on a buggy implementation is wrapped
in `anyio.fail_after(...)` so a broken stop_event/loop fails fast instead of
hanging the suite.

Design decision this suite freezes (api-contract.md section 6 leaves the exact
arguments to `baseline_repo.upsert_drift_baseline(...)` as "..."): the "current
window" persisted as the new baseline is read from the monitor's private
`_observations` attribute -- the same attribute name mlops.drift.DriftMonitor
uses internally for its sliding window -- converted to a `list`. DriftMonitor
exposes no public accessor for the raw window, so this is the same-package
internal `drift_evaluation_tick` is expected to reach into. Tests use small fake
DriftMonitor/baseline_repo doubles (never real DriftMonitor/OpsRepo) that expose
exactly this shape, per the harness instructions.
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

import anyio
from fastapi.testclient import TestClient

from mlops.kernel import ValidationFailed
from mlops.svc.app import create_app
from mlops.svc.settings import Settings
from mlops.svc.workers import (
    PeriodicWorker,
    WorkerRegistry,
    drift_evaluation_tick,
    incident_escalation_tick,
)

TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
NAME = {"viewer": "vic", "operator": "olga", "admin": "root"}


def make_settings(**kw):
    tokens = {TOK[r]: {"name": NAME[r], "role": r} for r in TOK}
    return Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **kw)


# ------------------------------------------------------------------ fakes / doubles

class FakeDriftReport:
    def __init__(self, level):
        self.level = level


class FakeDriftMonitor:
    """Duck-types the parts of mlops.drift.DriftMonitor drift_evaluation_tick uses."""

    def __init__(self, level, observations):
        self._level = level
        self._observations = list(observations)
        self.check_calls = 0

    def check(self):
        self.check_calls += 1
        return FakeDriftReport(self._level)


class FakeBaselineRepo:
    def __init__(self):
        self.calls = []

    def upsert_drift_baseline(self, name, reference):
        self.calls.append((name, list(reference)))


class FakeIncidentManager:
    """Duck-types the real IncidentManager contract that incident_escalation_tick
    now uses: list_open_incident_ids(), get(id)['current_tier'], escalate_if_overdue(id).
    tiers_before/tiers_after map incident_id -> tier, simulating the real class's
    before/after-tier behavior without needing a live IncidentManager+audit+clock.
    """

    def __init__(self, tiers_before, tiers_after):
        self._before = dict(tiers_before)
        self._after = dict(tiers_after)
        self.calls = []

    def list_open_incident_ids(self):
        return list(self._before.keys())

    def get(self, incident_id):
        return {"current_tier": self._before[incident_id]}

    def escalate_if_overdue(self, incident_id):
        self.calls.append(incident_id)
        return self._after[incident_id]


# ============================================================ (1) PeriodicWorker.__init__ validation

@pytest.mark.parametrize("bad_interval", [0, 0.0, -1, -0.5, 3600.0001, 3601, 999999,
                                           float("inf"), float("-inf"), float("nan"),
                                           "0.5", None, [], {}, object()])
def test_periodic_worker_rejects_invalid_interval(bad_interval):
    with pytest.raises(ValidationFailed):
        PeriodicWorker("w", bad_interval, lambda: None)


@pytest.mark.parametrize("bad_bool", [True, False])
def test_periodic_worker_rejects_bool_interval_even_if_in_range(bad_bool):
    # True == 1 and False == 0 would otherwise look plausible; bool must always
    # be rejected regardless of numeric value, per the contract's "not bool".
    with pytest.raises(ValidationFailed):
        PeriodicWorker("w", bad_bool, lambda: None)


@pytest.mark.parametrize("good_interval", [0.001, 0.01, 1, 1.0, 3600, 3600.0])
def test_periodic_worker_accepts_valid_interval(good_interval):
    worker = PeriodicWorker("w", good_interval, lambda: None)
    assert worker.interval_s == float(good_interval)


def test_periodic_worker_stores_name_and_fn():
    def fn():
        pass

    worker = PeriodicWorker("my-worker", 0.01, fn)
    assert worker.name == "my-worker"
    assert worker.fn is fn


# ============================================================ (2) PeriodicWorker.run()

def test_run_calls_fn_repeatedly_and_returns_after_stop_event():
    calls = []

    def fn():
        calls.append(1)

    async def scenario():
        worker = PeriodicWorker("w", 0.005, fn)
        stop_event = anyio.Event()

        async def stopper():
            while len(calls) < 5:
                await anyio.sleep(0.005)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    anyio.run(scenario)
    assert len(calls) >= 5


def test_run_does_not_call_fn_at_all_if_stop_event_already_set():
    calls = []

    def fn():
        calls.append(1)

    async def scenario():
        worker = PeriodicWorker("w", 0.01, fn)
        stop_event = anyio.Event()
        stop_event.set()
        with anyio.fail_after(5):
            await worker.run(stop_event=stop_event)

    anyio.run(scenario)
    assert calls == []


def test_run_on_error_receives_the_exact_exception_object_and_loop_continues():
    calls = []
    seen = []
    boom = RuntimeError("boom")

    def fn():
        calls.append(1)
        raise boom

    def on_error(exc):
        seen.append(exc)

    async def scenario():
        worker = PeriodicWorker("w", 0.005, fn, on_error=on_error)
        stop_event = anyio.Event()

        async def stopper():
            while len(calls) < 4:
                await anyio.sleep(0.005)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    anyio.run(scenario)
    assert len(calls) >= 4
    assert len(seen) >= 4
    assert all(e is boom for e in seen)


def test_run_on_error_means_worker_errors_stays_empty():
    def fn():
        raise ValueError("x")

    async def scenario(worker):
        stop_event = anyio.Event()

        async def stopper():
            while worker.fn.calls < 3:
                await anyio.sleep(0.005)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    class CountingRaiser:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            raise ValueError("x")

    fn = CountingRaiser()
    worker = PeriodicWorker("w", 0.005, fn, on_error=lambda exc: None)
    anyio.run(scenario, worker)
    assert fn.calls >= 3
    assert list(worker.errors) == []


def test_run_without_on_error_records_exceptions_in_worker_errors():
    calls = []

    def fn():
        calls.append(1)
        raise KeyError("no-handler")

    async def scenario(worker):
        stop_event = anyio.Event()

        async def stopper():
            while len(calls) < 3:
                await anyio.sleep(0.005)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    worker = PeriodicWorker("w", 0.005, fn)
    anyio.run(scenario, worker)
    assert len(calls) >= 3
    assert len(worker.errors) >= 3
    assert all(isinstance(e, KeyError) for e in worker.errors)


def test_worker_errors_is_bounded_deque_maxlen_100():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("always fails")

    async def scenario(worker):
        stop_event = anyio.Event()

        async def stopper():
            while len(calls) < 150:
                await anyio.sleep(0.001)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    worker = PeriodicWorker("w", 0.001, fn)
    anyio.run(scenario, worker)
    assert len(calls) >= 150
    assert len(worker.errors) == 100
    assert worker.errors.maxlen == 100


def test_run_a_single_failing_tick_does_not_stop_subsequent_successful_ticks():
    state = {"n": 0}
    ok_calls = []

    def fn():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("first tick fails")
        ok_calls.append(state["n"])

    async def scenario(worker):
        stop_event = anyio.Event()

        async def stopper():
            while len(ok_calls) < 2:
                await anyio.sleep(0.005)
            stop_event.set()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(stopper)
                await worker.run(stop_event=stop_event)

    worker = PeriodicWorker("w", 0.005, fn)
    anyio.run(scenario, worker)
    assert len(worker.errors) == 1
    assert len(ok_calls) >= 2


# ============================================================ (3) drift_evaluation_tick

def test_drift_tick_calls_check_on_every_given_monitor():
    monitors = {
        "a": FakeDriftMonitor("ok", [1.0] * 30),
        "b": FakeDriftMonitor("warn", [2.0] * 30),
    }
    repo = FakeBaselineRepo()
    drift_evaluation_tick(monitors, repo)
    assert monitors["a"].check_calls == 1
    assert monitors["b"].check_calls == 1


def test_drift_tick_upserts_baseline_only_for_non_ok_levels():
    monitors = {
        "ok_mon": FakeDriftMonitor("ok", [1.0] * 30),
        "warn_mon": FakeDriftMonitor("warn", [2.0] * 30),
        "alert_mon": FakeDriftMonitor("alert", [3.0] * 30),
    }
    repo = FakeBaselineRepo()
    drift_evaluation_tick(monitors, repo)
    called_names = {c[0] for c in repo.calls}
    assert called_names == {"warn_mon", "alert_mon"}
    assert len(repo.calls) == 2


def test_drift_tick_all_ok_never_upserts():
    monitors = {
        "a": FakeDriftMonitor("ok", [1.0] * 30),
        "b": FakeDriftMonitor("ok", [2.0] * 30),
    }
    repo = FakeBaselineRepo()
    drift_evaluation_tick(monitors, repo)
    assert repo.calls == []


def test_drift_tick_passes_the_current_window_as_the_new_reference():
    obs = [float(i) for i in range(30)]
    monitors = {"m": FakeDriftMonitor("alert", obs)}
    repo = FakeBaselineRepo()
    drift_evaluation_tick(monitors, repo)
    assert repo.calls == [("m", obs)]


def test_drift_tick_empty_monitor_dict_is_a_noop():
    repo = FakeBaselineRepo()
    drift_evaluation_tick({}, repo)
    assert repo.calls == []


# ============================================================ (4) incident_escalation_tick

def test_incident_tick_counts_only_incidents_whose_tier_actually_rose():
    mgr = FakeIncidentManager({"a": 1, "b": 2}, {"a": 2, "b": 2})
    result = incident_escalation_tick(mgr)
    assert mgr.calls == ["a", "b"]
    assert result == 1


def test_incident_tick_returns_zero_when_nothing_escalates():
    mgr = FakeIncidentManager({"a": 1, "b": 3}, {"a": 1, "b": 3})
    result = incident_escalation_tick(mgr)
    assert mgr.calls == ["a", "b"]
    assert result == 0


def test_incident_tick_calls_escalate_exactly_once_per_open_incident():
    mgr = FakeIncidentManager({"x": 1}, {"x": 3})
    incident_escalation_tick(mgr)
    assert mgr.calls == ["x"]


# ============================================================ (5) WorkerRegistry

def test_registry_start_all_spawns_worker_and_stop_all_stops_it():
    calls = []

    def fn():
        calls.append(1)

    async def scenario():
        worker = PeriodicWorker("w1", 0.005, fn)
        registry = WorkerRegistry()
        registry.add(worker)
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                registry.start_all(tg)
                while len(calls) < 3:
                    await anyio.sleep(0.005)
                registry.stop_all()

    anyio.run(scenario)
    assert len(calls) >= 3


def test_registry_start_all_spawns_every_added_worker():
    calls_a = []
    calls_b = []

    async def scenario():
        worker_a = PeriodicWorker("a", 0.005, lambda: calls_a.append(1))
        worker_b = PeriodicWorker("b", 0.005, lambda: calls_b.append(1))
        registry = WorkerRegistry()
        registry.add(worker_a)
        registry.add(worker_b)
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                registry.start_all(tg)
                while len(calls_a) < 2 or len(calls_b) < 2:
                    await anyio.sleep(0.005)
                registry.stop_all()

    anyio.run(scenario)
    assert len(calls_a) >= 2
    assert len(calls_b) >= 2


def test_registry_stop_all_before_any_ticks_still_lets_run_return():
    calls = []

    async def scenario():
        worker = PeriodicWorker("w", 3600, lambda: calls.append(1))
        registry = WorkerRegistry()
        registry.add(worker)
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                registry.start_all(tg)
                await anyio.sleep(0.01)
                registry.stop_all()

    anyio.run(scenario)
    # long interval_s means it never sleeps out naturally; only stop_all() lets
    # the task group exit within the fail_after bound.


def test_registry_add_with_no_start_all_never_runs_anything():
    calls = []
    worker = PeriodicWorker("w", 0.005, lambda: calls.append(1))
    registry = WorkerRegistry()
    registry.add(worker)
    time.sleep(0.05)
    assert calls == []


# ============================================================ (6) app.py wiring (section 7)

def test_create_app_without_workers_kwarg_is_zero_regression():
    with TestClient(create_app(make_settings())) as c:
        assert c.get("/healthz").status_code == 200


def test_create_app_explicit_workers_none_is_zero_regression():
    with TestClient(create_app(make_settings(), workers=None)) as c:
        assert c.get("/healthz").status_code == 200


def test_create_app_with_workers_runs_worker_during_request_and_stops_on_exit():
    # Runs the TestClient `with` block (whose __exit__ drives the FastAPI lifespan
    # shutdown) on a background thread with a bounded join, instead of directly in
    # this test function -- a mutant that fails to cancel/stop the worker on
    # shutdown would otherwise hang the whole suite forever, since a plain sync
    # test body has no anyio event loop of its own to hang `anyio.fail_after` off.
    calls = []

    def fn():
        calls.append(1)

    worker = PeriodicWorker("bg", 0.01, fn)
    app = create_app(make_settings(), workers=[worker])

    result = {}

    def body():
        with TestClient(app) as c:
            r = c.get("/healthz")
            result["status"] = r.status_code
            deadline = time.time() + 3.0
            while len(calls) < 2 and time.time() < deadline:
                time.sleep(0.01)
        result["exited"] = True

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=8.0)
    assert not t.is_alive(), "TestClient context (app lifespan shutdown) did not complete in time"
    assert result.get("status") == 200
    assert result.get("exited") is True
    assert len(calls) >= 1, "background worker should have ticked at least once"

    calls_at_exit = len(calls)
    time.sleep(0.2)
    assert len(calls) == calls_at_exit, "worker must stop ticking once the TestClient context exits"
