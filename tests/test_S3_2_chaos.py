"""S3.2 slim suite: mlops.chaos_scenarios (SteadyState, FaultSpec, Scenario, SimulatedCluster,
run_scenario, chaos_context). Frozen BEFORE the module exists; hand-computed oracles.
Pods in fixtures are p1..p4 (app/web); the service model is
  ready_pods = #ready, error_rate = 1 - ready/total, latency_ms = 20.0 + sum(delay latencies).
"""
import copy
import dataclasses
import math
import os
import signal
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.chaos_scenarios import (
    FAULT_TYPES,
    FaultSpec,
    Scenario,
    SimulatedCluster,
    SteadyState,
    chaos_context,
    run_scenario,
)
from mlops.kernel import ValidationFailed

BAD = (ValidationFailed, ValueError)
BAD_T = (ValidationFailed, ValueError, TypeError)
WEB = {"namespaces": ["app"], "labelSelectors": {"tier": "web"}}


def _within(seconds, fn):
    """Run fn under an alarm so a hang FAILS instead of blocking the suite."""
    def _boom(signum, frame):
        raise AssertionError("timed out after %ss (hang)" % seconds)
    old = signal.signal(signal.SIGALRM, _boom)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _pods(ready=(True, True, True, True)):
    return {"p%d" % (i + 1): {"namespace": "app", "labels": {"tier": "web"}, "ready": r}
            for i, r in enumerate(ready)}


def _cluster(ready=(True, True, True, True)):
    return SimulatedCluster(_pods(ready))


def _mixed():
    pods = _pods()
    pods["p5"] = {"namespace": "app", "labels": {"tier": "db"}, "ready": True}
    pods["p6"] = {"namespace": "other", "labels": {"tier": "web"}, "ready": True}
    return SimulatedCluster(pods)


def _kill(mode="all", value=None, selector=WEB):
    return FaultSpec("PodChaos", "pod-kill", copy.deepcopy(selector), mode, value)


def _ss():
    return [SteadyState("enough_pods", "ready_pods", ">=", 3),
            SteadyState("low_errors", "error_rate", "<=", 0.3)]


# ------------------------------------------------------------------ SteadyState

@pytest.mark.parametrize("op,below,at,above", [
    # (holds(4), holds(5), holds(6)) for threshold 5
    ("<=", True, True, False),
    ("<", True, False, False),
    (">=", False, True, True),
    (">", False, False, True),
])
def test_steady_state_holds_all_ops(op, below, at, above):
    s = SteadyState("x", "m", op, 5)
    assert (s.holds(4), s.holds(5), s.holds(6)) == (below, at, above)
    assert (s.holds(4.999), s.holds(5.0)) == (below, at)


@pytest.mark.parametrize("args", [
    ("x", "m", "==", 1), ("x", "m", "=<", 1), ("x", "m", None, 1), ("x", "m", "", 1),
    ("x", "m", "<=", float("nan")), ("x", "m", "<=", float("inf")), ("x", "m", "<=", -math.inf),
    ("x", "m", "<=", True), ("x", "m", "<=", "5"), ("x", "m", "<=", None),
    ("", "m", "<=", 1), (None, "m", "<=", 1),
])
def test_steady_state_invalid(args):
    with pytest.raises(BAD_T):
        SteadyState(*args)


# ------------------------------------------------------------------ FAULT_TYPES

def test_fault_types_exact_table():
    expected = {
        "PodChaos": {"pod-kill", "container-kill"},
        "NetworkChaos": {"delay", "loss", "duplicate", "corrupt", "partition"},
        "DNSChaos": {"error", "random"},
        "HTTPChaos": {"delay", "abort"},
        "StressChaos": {"cpu", "memory"},
        "IOChaos": {"latency", "fault"},
        "TimeChaos": {"skew"},
    }
    assert {k: set(v) for k, v in FAULT_TYPES.items()} == expected
    assert "pod-failure" not in FAULT_TYPES["PodChaos"]


# -------------------------------------------------------------------- FaultSpec

@pytest.mark.parametrize("args,kw", [
    (("PodChaos", "pod-kill", WEB), {}),
    (("NetworkChaos", "delay", WEB), {"params": {"latency_ms": 100}}),
    (("DNSChaos", "error", WEB), {}),
    (("HTTPChaos", "abort", WEB), {}),
    (("StressChaos", "cpu", WEB), {"params": {"workers": 2}}),
    (("IOChaos", "latency", WEB), {}),
    (("TimeChaos", "skew", WEB), {}),
    (("PodChaos", "container-kill", {"namespaces": ["app"]}), {"mode": "one"}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed", "value": 2}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed-percent", "value": 100}),
    (("PodChaos", "pod-kill", WEB), {"mode": "random-max-percent", "value": 1}),
    (("NetworkChaos", "loss", WEB), {"params": {"loss_pct": 100}}),
])
def test_fault_spec_valid(args, kw):
    f = FaultSpec(*args, **kw)
    assert f.type == args[0] and f.action == args[1]


@pytest.mark.parametrize("args,kw", [
    (("Nope", "pod-kill", WEB), {}),
    (("PodChaos", "pod-failure", WEB), {}),          # excluded action
    (("PodChaos", "delay", WEB), {}),                # action of another type
    (("PodChaos", "pod-kill", None), {}),
    (("PodChaos", "pod-kill", {}), {}),              # no blast-radius bound
    (("PodChaos", "pod-kill", {"labelSelectors": {"tier": "web"}}), {}),  # namespaces missing
    (("PodChaos", "pod-kill", {"namespaces": []}), {}),
    (("PodChaos", "pod-kill", {"namespaces": "app"}), {}),
    (("PodChaos", "pod-kill", {"namespaces": [""]}), {}),
    (("PodChaos", "pod-kill", {"namespaces": ["app", 3]}), {}),
    (("PodChaos", "pod-kill", {"namespaces": ["app"], "labelSelectors": ["tier"]}), {}),
    (("PodChaos", "pod-kill", {"namespaces": ["app"], "labelSelectors": {"tier": 1}}), {}),
    (("PodChaos", "pod-kill", {"namespaces": ["app"], "labelSelectors": {1: "web"}}), {}),
    (("PodChaos", "pod-kill", WEB), {"mode": "some"}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed"}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed", "value": 0}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed", "value": True}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed", "value": 1.5}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed-percent", "value": 0}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed-percent", "value": 101}),
    (("PodChaos", "pod-kill", WEB), {"mode": "fixed-percent", "value": True}),
    (("PodChaos", "pod-kill", WEB), {"mode": "random-max-percent", "value": 101}),
    (("NetworkChaos", "delay", WEB), {}),
    (("NetworkChaos", "delay", WEB), {"params": {"latency_ms": 0}}),
    (("NetworkChaos", "delay", WEB), {"params": {"latency_ms": -5}}),
    (("NetworkChaos", "delay", WEB), {"params": {"latency_ms": float("nan")}}),
    (("NetworkChaos", "loss", WEB), {"params": {"loss_pct": 0}}),
    (("NetworkChaos", "loss", WEB), {"params": {"loss_pct": 101}}),
    (("NetworkChaos", "loss", WEB), {}),
    (("StressChaos", "cpu", WEB), {"params": {"workers": 0}}),
    (("StressChaos", "cpu", WEB), {}),
])
def test_fault_spec_invalid(args, kw):
    with pytest.raises(BAD):
        FaultSpec(*args, **kw)


def test_fault_spec_frozen():
    f = _kill()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.mode = "one"


# --------------------------------------------------------------------- Scenario

def _sc(steps=None, steady=None, name="s"):
    return Scenario([[_kill("one")]] if steps is None else steps, _ss() if steady is None else steady, name)


@pytest.mark.parametrize("make", [
    lambda: _sc(steps=[]),
    lambda: _sc(steps=[[]]),
    lambda: _sc(steps=[[_kill()], []]),
    lambda: _sc(steps="x"),
    lambda: _sc(steps=[_kill()]),                    # inner must be a list
    lambda: _sc(steps=[["fault"]]),
    lambda: _sc(steps=[[_kill()] for _ in range(21)]),
    lambda: _sc(steps=[[_kill() for _ in range(11)]]),
    lambda: _sc(steady=[]),
    lambda: _sc(steady=["ready_pods>=3"]),
    lambda: _sc(steady=None if False else "x"),
])
def test_scenario_invalid(make):
    with pytest.raises(BAD_T):
        make()


def test_scenario_bounds_accepted_and_tuples_stored():
    # exactly 20 steps and exactly 10 faults per step are legal
    Scenario([[_kill()] for _ in range(20)], _ss(), "max-steps")
    Scenario([[_kill() for _ in range(10)]], _ss(), "max-faults")
    steps, steady = [[_kill("one")], [_kill()]], _ss()
    sc = Scenario(steps, steady, "t")
    steps.append([_kill()])
    steps[0].append(_kill())
    steady.clear()
    assert isinstance(sc.steps, tuple) and all(isinstance(s, tuple) for s in sc.steps)
    assert isinstance(sc.steady_states, tuple)
    assert (len(sc.steps), len(sc.steps[0]), len(sc.steady_states)) == (2, 1, 2)


# ------------------------------------------------------------- SimulatedCluster

def test_cluster_select_namespace_and_labels():
    c = _mixed()
    assert c.select(WEB) == ["p1", "p2", "p3", "p4"]
    assert c.select({"namespaces": ["app"], "labelSelectors": {"tier": "db"}}) == ["p5"]
    assert c.select({"namespaces": ["app", "other"]}) == ["p1", "p2", "p3", "p4", "p5", "p6"]
    assert c.select({"namespaces": ["nowhere"]}) == []


@pytest.mark.parametrize("mode,value,expected", [
    ("one", None, {"p1"}),                      # first sorted
    ("fixed", 2, {"p1", "p2"}),                 # first 2 sorted
    ("fixed-percent", 50, {"p1", "p2"}),        # ceil(0.5*4) = 2
    ("fixed-percent", 30, {"p1", "p2"}),        # ceil(1.2) = 2
    ("fixed-percent", 100, {"p1", "p2", "p3", "p4"}),
    ("all", None, {"p1", "p2", "p3", "p4"}),
    ("random-max-percent", 100, {"p1", "p2", "p3", "p4"}),   # ceil(4.0) = 4
])
def test_blast_radius_modes(mode, value, expected):
    c = _mixed()
    f = _kill(mode, value)
    assert set(c.blast_radius([f])) == expected
    assert set(c.blast_radius([f])) == expected            # deterministic on repeat


def test_blast_radius_union_and_empty():
    c = _mixed()
    db = FaultSpec("PodChaos", "pod-kill", {"namespaces": ["app"], "labelSelectors": {"tier": "db"}})
    assert set(c.blast_radius([_kill("one"), db])) == {"p1", "p5"}     # union of {p1} and {p5}
    nothing = FaultSpec("PodChaos", "pod-kill", {"namespaces": ["zzz"]})
    assert set(c.blast_radius([nothing])) == set()                     # no error
    assert set(c.blast_radius([])) == set()


def test_cluster_metrics_and_pod_validation():
    m = _cluster().metrics()
    assert (m["ready_pods"], m["error_rate"], m["latency_ms"]) == (4, 0.0, 20.0)
    m = _cluster((True, False, False, True)).metrics()      # 2 ready of 4
    assert (m["ready_pods"], m["error_rate"]) == (2, 0.5)
    for bad in [
        {"p1": {"labels": {}, "ready": True}},                       # namespace missing
        {"p1": {"namespace": "app", "ready": True}},                 # labels missing
        {"p1": {"namespace": "app", "labels": {}, "ready": "yes"}},  # non-bool ready
        {"p1": {"namespace": "app", "labels": {}, "ready": 1}},
        {"p1": {"namespace": "", "labels": {}, "ready": True}},
        {"p1": {"namespace": "app", "labels": {"a": 1}, "ready": True}},
        {"p1": "pod"}, [], None,
    ]:
        with pytest.raises(BAD_T):
            SimulatedCluster(bad)


# ----------------------------------------------------------------- run_scenario

def test_run_single_kill_passes():
    # kill one of 4: ready 3 (>=3 ok), error 1-3/4 = 0.25 (<=0.3 ok)
    r = run_scenario(_sc(), _cluster())
    assert r.passed is True and list(r.violated) == []
    assert [s["stage"] for s in r.stages] == ["pre", "inject", "post"]
    inj = r.stages[1]
    assert (inj["metrics"]["ready_pods"], inj["metrics"]["error_rate"]) == (3, 0.25)
    assert inj["violated"] == []


def test_run_two_steps_second_violates_and_post_recovers():
    # step 2 kills fixed 2: ready <= 2 (<3 violated), error >= 0.5 (>0.3 violated)
    sc = _sc(steps=[[_kill("one")], [_kill("fixed", 2)]])
    orig = _cluster()
    r = run_scenario(sc, orig)
    assert r.passed is False
    joined = " | ".join(r.violated)
    assert "enough_pods" in joined and "low_errors" in joined
    assert [s["stage"] for s in r.stages] == ["pre", "inject", "inject", "post"]
    assert r.stages[1]["violated"] == []
    assert sorted(r.stages[2]["violated"]) == ["enough_pods", "low_errors"] or (
        "enough_pods" in " ".join(r.stages[2]["violated"]) and "low_errors" in " ".join(r.stages[2]["violated"]))
    assert r.stages[2]["metrics"]["ready_pods"] <= 2 and r.stages[2]["metrics"]["error_rate"] >= 0.5
    # post evaluates the recovered ORIGINAL cluster: 4 ready, error 0.0, passes
    post = r.stages[3]
    assert post["metrics"]["ready_pods"] == 4 and post["metrics"]["error_rate"] == 0.0
    assert post["violated"] == []


def test_run_delay_violates_latency_steady_state():
    delay = FaultSpec("NetworkChaos", "delay", WEB, params={"latency_ms": 300})
    sc = Scenario([[delay]], [SteadyState("fast", "latency_ms", "<=", 250)], "lat")
    r = run_scenario(sc, _cluster())
    assert r.passed is False
    assert r.stages[1]["metrics"]["latency_ms"] == 320.0            # 20 + 300
    assert r.stages[1]["violated"] == ["fast"] or "fast" in " ".join(r.stages[1]["violated"])
    assert r.stages[-1]["metrics"]["latency_ms"] == 20.0            # recovered
    ok = Scenario([[delay]], [SteadyState("fast", "latency_ms", "<=", 320)], "lat-eq")
    assert run_scenario(ok, _cluster()).passed is True              # 320 <= 320 (non-strict)


def test_run_pre_validation_failure_injects_nothing():
    # 2 of 4 ready: ready_pods = 2 violates >= 3 before any fault
    c = _cluster((True, True, False, False))
    r = run_scenario(_sc(), c)
    assert r.passed is False
    assert all(s["stage"] != "inject" for s in r.stages)
    assert any("pre-validation" in v for v in r.violated)
    assert any("enough_pods" in s for st in r.stages for s in st["violated"])


def test_run_never_mutates_input_cluster():
    pods = _pods()
    before = copy.deepcopy(pods)
    c = SimulatedCluster(pods)
    m_before = c.metrics()
    run_scenario(_sc(steps=[[_kill("all")], [_kill("fixed", 2)]]), c)
    assert pods == before
    assert c.metrics() == m_before and c.select(WEB) == ["p1", "p2", "p3", "p4"]


def test_run_post_passes_iff_pre_passes():
    # passing pre state => post entries never violated even when injection stages fail
    r = run_scenario(_sc(steps=[[_kill("all")]]), _cluster())
    assert r.passed is False
    assert r.stages[1]["metrics"]["ready_pods"] == 0 and r.stages[1]["metrics"]["error_rate"] == 1.0
    assert r.stages[-1]["stage"] == "post" and r.stages[-1]["violated"] == []


# ---------------------------------------------------------------- chaos_context

def test_chaos_context_counts_violated_entries():
    good = run_scenario(_sc(), _cluster())
    assert chaos_context(good) == {"chaos_passed": True, "chaos_violations": 0}
    bad = run_scenario(_sc(steps=[[_kill("fixed", 2)]]), _cluster())
    ctx = chaos_context(bad)
    assert set(ctx) == {"chaos_passed", "chaos_violations"}
    assert ctx["chaos_passed"] is False and isinstance(ctx["chaos_violations"], int)
    assert ctx["chaos_violations"] == len(bad.violated) >= 2         # both steady states violated
    pre = run_scenario(_sc(), _cluster((True, True, False, False)))
    assert chaos_context(pre)["chaos_violations"] == len(pre.violated) >= 1
    for junk in (None, {}, "r"):
        with pytest.raises(BAD_T + (AttributeError,)):
            chaos_context(junk)


# ---------------------------------------------------------------------- hostile

@pytest.mark.parametrize("scenario,cluster", [
    (None, None), (None, "c"), ({"steps": []}, "c"), ("scenario", "c"),
])
def test_run_scenario_rejects_wrong_types(scenario, cluster):
    c = _cluster() if cluster == "c" else cluster
    with pytest.raises(BAD_T):
        run_scenario(scenario, c)
    with pytest.raises(BAD_T):
        run_scenario(_sc(), {"p1": {"namespace": "app", "labels": {}, "ready": True}})
    with pytest.raises(BAD_T):
        run_scenario(_sc(), None)


def test_run_scenario_large_cluster_is_bounded():
    pods = {"p%05d" % i: {"namespace": "app", "labels": {"tier": "web"}, "ready": True}
            for i in range(5000)}
    c = SimulatedCluster(pods)
    steps = [[_kill("fixed-percent", 10)] for _ in range(20)]
    sc = Scenario(steps, [SteadyState("some", "ready_pods", ">=", 0)], "big")
    t0 = time.process_time()  # CPU time: immune to machine load
    r = _within(20, lambda: run_scenario(sc, c))
    assert time.process_time() - t0 < 3.0
    assert r.passed is True and len(r.stages) == 22               # pre + 20 inject + post
    assert r.stages[1]["metrics"]["ready_pods"] == 4500           # ceil(0.10*5000) = 500 killed
