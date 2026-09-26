"""S3.0 constraints for review findings H1, H8, M1, M2 (floor_guard / slo_guard)."""
import os
import signal
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops import slo_guard
from mlops.floor_guard import (
    AssuredFirstDispatcher, FloorConfig, Request, TenantFloor, simulate_guard,
)
from mlops.rbac import RBACEngine

NAN = float("nan")
INF = float("inf")
REJECT = (ValueError, TypeError)


class _Hang(Exception):
    pass


def _within(seconds, fn):
    def _h(signum, frame):
        raise _Hang("did not return within %ss" % seconds)
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _rejects(fn):
    with pytest.raises(REJECT):
        _within(12, fn)


def cfg(**kw):
    base = dict(credits=2, window_s=10.0, batch_slots=2, s_max=1.0)
    base.update(kw)
    return FloorConfig(**base)


# ------------------------------------------------------------ (1) H1 honesty
def test_reject_counts_rejected_assured_as_misses():
    r = simulate_guard(3, 5, 2, 0, cfg(credits=1, excess="reject"), True)
    assert r["assured_offered"] == 10
    assert r["assured_rejected"] == 5
    assert r["assured_rejected"] + r["assured_admitted"] == r["assured_offered"]
    assert r["assured_miss_rate"] >= 0.5


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("excess", ["reject", "demote"])
def test_guarded_rate_never_below_rejected_share(seed, excess):
    r = simulate_guard(seed, 5, 2, 50, cfg(credits=1, excess=excess), True)
    assert r["assured_miss_rate"] >= r["assured_rejected"] / r["assured_offered"]


def test_demote_not_counted_as_miss_when_they_meet_deadline():
    r = simulate_guard(3, 5, 2, 0, cfg(credits=1, excess="demote"), True)
    assert r["assured_demoted"] == 5
    assert r["assured_rejected"] == 0
    assert r["assured_offered"] == 10
    # no load: demoted (as opportunistic) finish within their own deadline
    assert r["assured_miss_rate"] == 0.0


def test_flood_still_guarded_better_and_control_zero():
    for seed in range(5):
        g = simulate_guard(seed, 5, 2, 200, cfg(), True)["assured_miss_rate"]
        u = simulate_guard(seed, 5, 2, 200, cfg(), False)["assured_miss_rate"]
        assert g < u and g <= 0.6, (seed, g, u)
        assert simulate_guard(seed, 5, 2, 0, cfg(), True)["assured_miss_rate"] == 0
        assert simulate_guard(seed, 5, 2, 0, cfg(), False)["assured_miss_rate"] == 0


# ------------------------------------------------------- (2) H8 slo_guard table
def _stab(**kw):
    a = dict(B=3, s_max=1.0, C=2, W=10)
    a.update(kw)
    return slo_guard.stability(a["B"], a["s_max"], a["C"], a["W"])


@pytest.mark.parametrize("ov", [
    {"B": NAN}, {"B": True}, {"B": "5"}, {"B": None}, {"B": -1}, {"B": INF},
    {"s_max": NAN}, {"s_max": True}, {"s_max": "1"}, {"s_max": INF}, {"s_max": -1.0},
    {"C": 0}, {"C": -1}, {"C": True}, {"C": "2"}, {"C": NAN},
    {"W": 0}, {"W": -1}, {"W": NAN}, {"W": True}, {"W": "10"}, {"W": INF},
])
def test_stability_validation(ov):
    _rejects(lambda: _stab(**ov))


@pytest.mark.parametrize("args", [
    (-1.0, 1, 1), (NAN, 1, 1), (INF, 1, 1), (True, 1, 1), ("1", 1, 1),
    (1.0, -5, 1), (1.0, NAN, 1), (1.0, INF, 1), (1.0, True, 1),
    (1.0, 1, 0), (1.0, 1, -1), (1.0, 1, NAN), (1.0, 1, True),
])
def test_sojourn_bound_validation(args):
    _rejects(lambda: slo_guard.sojourn_bound(*args))


@pytest.mark.parametrize("credits,window", [
    ("5", 10), (None, 10), (True, 10), (-1, 10), (NAN, 10), (0, 10),
    (5, "10"), (5, None), (5, True), (5, -1), (5, NAN), (5, 0),
])
def test_tenant_reservation_ctor_validation(credits, window):
    _rejects(lambda: slo_guard.TenantReservation(RBACEngine(), "ns", credits, window))


@pytest.mark.parametrize("ts,n", [
    (NAN, 1), (5, -3), (5, True), (INF, 1), (5, 1.5), (True, 1), ("5", 1), (5, None),
])
def test_tenant_reservation_admit_validation(ts, n):
    res = slo_guard.TenantReservation(RBACEngine(), "ns", 3, 10)
    _rejects(lambda: res.admit(ts, n))


@pytest.mark.parametrize("obs,tgt", [
    (5, 0.9), (-0.1, 0.9), (1.5, 0.9), (0.5, 1.0), (0.5, 1.5), (0.5, -0.1),
    (NAN, 0.9), (0.5, NAN), (True, 0.9), (0.5, True), (INF, 0.9), (0.5, INF),
])
def test_burn_rate_validation(obs, tgt):
    _rejects(lambda: slo_guard.burn_rate(obs, tgt))


@pytest.mark.parametrize("thr", [NAN, INF, -INF, -1.0, True, "2", None])
def test_budget_alert_tracker_threshold_validation(thr):
    _rejects(lambda: slo_guard.BudgetAlertTracker(object(), thr, 2))


# ------------------------------------------------------------ (3) M2 poisoning
def test_implausible_timestamp_rejected_and_state_unchanged():
    tf = TenantFloor(cfg())
    assert tf.admit(1.0, 1).admitted == 1
    _rejects(lambda: tf.admit(1e300, 1))
    r = tf.admit(2.0, 1)
    assert r.admitted == 1 and r.window_index == 0  # not poisoned; credit 2 of 2 used
    assert tf.admit(3.0, 1).admitted == 0


def test_implausible_timestamp_on_fresh_floor_then_normal_works():
    tf = TenantFloor(cfg())
    _rejects(lambda: tf.admit(1e300, 1))
    assert tf.admit(5.0, 1).admitted == 1


def test_admit_1e308_is_value_error_not_overflow():
    with pytest.raises(REJECT):
        TenantFloor(cfg(window_s=1e-3)).admit(1e308, 1)
    with pytest.raises(REJECT):
        TenantFloor(cfg()).admit(1e308, 1)


@pytest.mark.parametrize("ov", [{"window_s": 1e-300}, {"s_max": 10**400},
                                {"window_s": 10**400}, {"t0": 10**400}])
def test_floor_config_extreme_values_value_error(ov):
    with pytest.raises(ValueError):
        cfg(**ov)


# ------------------------------------------------------------- (4) M1 scale
def test_dispatcher_20000_requests_fast():
    reqs = [Request(id="r%d" % i, cls="assured" if i % 2 else "opportunistic",
                    arrival_ts=i * 0.01, service_s=0.001, deadline_ts=1e9)
            for i in range(20000)]
    t = time.process_time()  # CPU time: immune to machine load
    out = _within(32, lambda: AssuredFirstDispatcher(cfg(batch_slots=4)).run(reqs))
    assert time.process_time() - t < 5
    assert len(out) == 20000


@pytest.mark.parametrize("kw", [
    {"n_windows": 10001}, {"offered_assured": 10001}, {"offered_opportunistic": 10001},
])
def test_simulate_guard_bounded_inputs_rejected_promptly(kw):
    a = dict(seed=1, n_windows=2, offered_assured=1, offered_opportunistic=1)
    a.update(kw)
    t = time.process_time()  # CPU time: immune to machine load
    with pytest.raises(ValueError):
        _within(12, lambda: simulate_guard(a["seed"], a["n_windows"], a["offered_assured"],
                                          a["offered_opportunistic"], cfg(), True))
    assert time.process_time() - t < 1


def test_simulate_guard_huge_n_windows_fast():
    t = time.process_time()  # CPU time: immune to machine load
    with pytest.raises(ValueError):
        _within(12, lambda: simulate_guard(1, 10**9, 1, 1, cfg(), True))
    assert time.process_time() - t < 1
