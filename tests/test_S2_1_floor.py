"""S2.1 slim suite: runtime floor guard (mlops.floor_guard) and slo_guard fixes.
Oracle-based; hand-computed expected values are in comments. Fail-closed.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops import floor_guard, slo_guard
from mlops.floor_guard import (
    AssuredFirstDispatcher,
    FloorConfig,
    Request,
    StabilityError,
    TenantFloor,
    check_preconditions,
    require_guarantee,
    simulate_guard,
    sojourn_bound as floor_sojourn_bound,
)
from mlops.kernel import MlopsError
from mlops.rbac import RBACEngine

NAN = float("nan")
INF = float("inf")


def cfg(**kw):
    base = dict(credits=3, window_s=10.0, batch_slots=2, s_max=0.5)
    base.update(kw)
    return FloorConfig(**base)


# ---------------------------------------------------------------- (1) config
@pytest.mark.parametrize(
    "override",
    [
        {"credits": 0}, {"credits": -1}, {"credits": True}, {"credits": 2.5},
        {"credits": "3"}, {"credits": None},
        {"window_s": 0}, {"window_s": -1}, {"window_s": NAN}, {"window_s": INF},
        {"window_s": True},
        {"batch_slots": 0}, {"batch_slots": True},
        {"s_max": 0}, {"s_max": NAN},
        {"excess": "bogus"},
    ],
)
def test_floor_config_rejects_invalid(override):
    with pytest.raises(ValueError):
        cfg(**override)


def test_floor_config_valid_and_frozen():
    c = cfg()
    assert (c.credits, c.window_s, c.batch_slots, c.s_max) == (3, 10.0, 2, 0.5)
    assert c.t0 == 0.0 and c.excess == "reject" and c.drop_lower_bound == 0.0
    with pytest.raises(AttributeError):  # FrozenInstanceError
        c.credits = 9
    assert issubclass(StabilityError, MlopsError)


# ---------------------------------------------------- (2) window alignment/t0
def test_window_alignment_with_t0():
    f = TenantFloor(cfg(credits=2, window_s=10.0, t0=100.0))
    r = f.admit(100.0, 2)          # idx floor(0/10)=0, 2 credits -> admit 2
    assert (r.window_index, r.admitted, r.rejected) == (0, 2, 0)
    r = f.admit(109.9, 1)          # idx floor(9.9/10)=0, still window 0, 0 left
    assert (r.window_index, r.admitted, r.rejected) == (0, 0, 1)
    r = f.admit(110.0, 2)          # idx floor(10/10)=1 -> refill, admit 2
    assert (r.window_index, r.admitted, r.rejected) == (1, 2, 0)


# ---------------------------------------------------------- (3) time backwards
def test_time_backwards_raises_and_does_not_refill():
    f = TenantFloor(cfg(credits=3, window_s=10.0))
    assert f.admit(25.0, 3).admitted == 3      # window 2, credits now 0
    with pytest.raises(ValueError):
        f.admit(5.0, 1)                        # window 0 < 2
    r = f.admit(26.0, 1)                       # window 2 again: still 0 credits
    assert (r.window_index, r.admitted, r.rejected) == (2, 0, 1)


# ------------------------------------------------------------ (4) partial admit
def test_partial_admit_reject_and_demote():
    r = TenantFloor(cfg(credits=3)).admit(0.0, 5)  # min(3,5)=3, excess 2
    assert (r.admitted, r.rejected, r.demoted) == (3, 2, 0)
    r = TenantFloor(cfg(credits=3, excess="demote")).admit(0.0, 5)
    assert (r.admitted, r.rejected, r.demoted) == (3, 0, 2)
    r = TenantFloor(cfg()).admit(0.0, 0)
    assert (r.admitted, r.rejected, r.demoted) == (0, 0, 0)


@pytest.mark.parametrize("bad", [-1, True, 2.0, "3", None])
def test_admit_rejects_bad_n(bad):
    with pytest.raises(ValueError):
        TenantFloor(cfg()).admit(0.0, bad)


def test_teeth_not_all_or_nothing_and_no_reset_every_call():
    f = TenantFloor(cfg(credits=3, window_s=10.0))
    # all-or-nothing would give 0 for the first call (2<=3 ok) but 0 for the
    # second (2 > 1 left); partial admit must give 1 with 1 rejected.
    a = f.admit(100.0, 2)   # 3 -> 1 left
    b = f.admit(100.5, 2)   # same window: min(2,1)=1 admitted, 1 rejected
    assert (a.admitted, b.admitted, b.rejected) == (2, 1, 1)
    # a reset-every-call implementation would admit 3 here; must be 0 (0 left)
    c = f.admit(101.0, 3)
    assert (c.admitted, c.rejected) == (0, 3)


# ------------------------------------------------------- (5) preconditions
def test_check_preconditions_hand_computed():
    ok = cfg(credits=4, window_s=1.0, batch_slots=2, s_max=0.5)
    p = check_preconditions(ok, 4)   # rho = 4*0.5/(2*1) = 1.0 (boundary inclusive)
    assert p["rho_a"] == pytest.approx(1.0)
    assert p["stable"] is True and p["backlog_ok"] is True
    assert p["guarantee_void"] is False
    p = check_preconditions(ok, 5)   # backlog 5 > B=4
    assert p["stable"] is True and p["backlog_ok"] is False
    assert p["guarantee_void"] is True

    over = cfg(credits=5, window_s=1.0, batch_slots=2, s_max=0.5)
    p = check_preconditions(over, 0)  # rho = 5*0.5/2 = 1.25
    assert p["rho_a"] == pytest.approx(1.25)
    assert p["stable"] is False and p["guarantee_void"] is True


def test_require_guarantee_and_sojourn_bound():
    ok = cfg(credits=4, window_s=1.0, batch_slots=2, s_max=0.5)
    over = cfg(credits=5, window_s=1.0, batch_slots=2, s_max=0.5)
    require_guarantee(ok, 4)  # no raise
    with pytest.raises(StabilityError):
        require_guarantee(ok, 5)
    with pytest.raises(StabilityError):
        require_guarantee(over, 0)
    # 0.5 * (1 + ceil(4/2)) = 0.5 * 3 = 1.5
    assert floor_sojourn_bound(ok) == pytest.approx(1.5)
    assert floor_sojourn_bound(ok, 4) == pytest.approx(1.5)
    assert floor_sojourn_bound(ok, 5) is None
    assert floor_sojourn_bound(over) is None


# ------------------------------------------------------------- (6) dispatcher
def _req(i, cls, arr, svc, dl=1000.0):
    return Request(id=i, cls=cls, arrival_ts=arr, service_s=svc, deadline_ts=dl)


def test_assured_first_non_preemptive_hand_computed():
    d = AssuredFirstDispatcher(cfg(batch_slots=1))
    out = {o.id: o for o in d.run([
        _req("o0", "opportunistic", 0.0, 2.0),   # runs 0..2 (non-preemptive)
        _req("o1", "opportunistic", 0.5, 1.0),
        _req("o2", "opportunistic", 0.7, 1.0),
        _req("a1", "assured", 1.0, 1.0),
        _req("a2", "assured", 1.5, 1.0),
    ])}
    # slot frees at 2: assured first (FIFO a1,a2), then opportunistic FIFO o1,o2
    expected = {
        "o0": (0.0, 2.0), "a1": (2.0, 3.0), "a2": (3.0, 4.0),
        "o1": (4.0, 5.0), "o2": (5.0, 6.0),
    }
    for rid, (s, f) in expected.items():
        assert out[rid].start_ts == pytest.approx(s), rid
        assert out[rid].finish_ts == pytest.approx(f), rid
        assert out[rid].status == "completed" and out[rid].missed is False


# ------------------------------------------------------------ (7) doom drop
def test_doom_sound_drop_exact():
    d = AssuredFirstDispatcher(cfg(batch_slots=1, drop_lower_bound=2.0))
    out = {o.id: o for o in d.run([
        _req("r0", "assured", 0.0, 5.0, 100.0),  # start 0: 0+2<=100 keep, finish 5
        _req("r1", "assured", 1.0, 1.0, 7.0),    # start 5: 5+2 == 7 -> NOT dropped
        _req("r2", "assured", 1.5, 1.0, 7.5),    # start 6: 6+2=8 > 7.5 -> dropped
        _req("r3", "assured", 2.0, 10.0, 12.0),  # 8<=12 keep; finishes late -> missed
    ])}
    dropped = sorted(i for i, o in out.items() if o.status == "dropped")
    assert dropped == ["r2"]
    assert out["r2"].missed is True
    assert out["r1"].status == "completed" and out["r1"].finish_ts == pytest.approx(6.0)
    assert out["r1"].missed is False
    # r3 not dropped even though it will miss: only doom-sound drops allowed
    assert out["r3"].status == "completed" and out["r3"].missed is True


# --------------------------------------------------------------- (8) simulate
SIM_CFG = dict(credits=2, window_s=10.0, batch_slots=2, s_max=1.0)


def _sim(seed, opp, guarded):
    return simulate_guard(seed, 5, 2, opp, cfg(**SIM_CFG), guarded)


def test_simulate_guard_deterministic_and_labelled():
    a, b = _sim(7, 200, True), _sim(7, 200, True)
    assert a == b
    assert a["simulated"] is True
    assert _sim(7, 200, False)["simulated"] is True
    assert "assured_miss_rate" in a and "opportunistic_miss_rate" in a


def test_simulate_guard_flood_guarded_beats_unguarded():
    for seed in range(5):
        g = _sim(seed, 200, True)["assured_miss_rate"]
        u = _sim(seed, 200, False)["assured_miss_rate"]
        assert g < u, (seed, g, u)


def test_simulate_guard_no_flood_control_both_zero():
    for seed in range(5):
        assert _sim(seed, 0, True)["assured_miss_rate"] == 0
        assert _sim(seed, 0, False)["assured_miss_rate"] == 0


# ----------------------------------------------------------- (9) slo_guard
def test_tenant_reservation_backwards_time_no_refill():
    rbac = RBACEngine()
    res = slo_guard.TenantReservation(rbac, "ns", 3, 10)
    assert res.admit(25, 3) is True          # window 2, budget used up
    with pytest.raises(ValueError):
        res.admit(5, 1)                      # window 0 < 2
    assert rbac.current_jobs["ns"] == 3      # not refilled to 0
    assert res.admit(26, 1) is False         # still window 2, 3+1 > 3


@pytest.mark.parametrize(
    "args",
    [(0.5, 4, 0), (0.5, 4, -1), (0.5, 4, NAN), (0.5, 4, INF), (0.5, 4, True),
     (NAN, 4, 2), (INF, 4, 2), (0.5, True, 2)],
)
def test_slo_sojourn_bound_rejects_bad_inputs(args):
    with pytest.raises(ValueError):
        slo_guard.sojourn_bound(*args)


def test_slo_stability_strict_and_burn_rate():
    assert slo_guard.sojourn_bound(0.5, 4, 2) == pytest.approx(1.5)
    s = slo_guard.stability(4, 0.5, 2, 1, strict=True)   # rho = 1.0 -> stable
    assert s["rho_a"] == pytest.approx(1.0) and s["stable"] is True
    with pytest.raises(floor_guard.StabilityError):
        slo_guard.stability(5, 0.5, 2, 1, strict=True)   # rho = 1.25
    assert slo_guard.stability(5, 0.5, 2, 1)["stable"] is False  # non-strict returns
    # (1-0.99)/(1-0.9) = 0.1
    assert slo_guard.burn_rate(0.99, 0.9) == pytest.approx(0.1)
    for a in [(NAN, 0.9), (True, 0.9), (0.99, NAN), (0.99, True)]:
        with pytest.raises(ValueError):
            slo_guard.burn_rate(*a)


# ================= S2.1 amendment: fail-closed validation + no-hang probes
import copy
import signal

REJECT = (ValueError, TypeError)


class _Hang(Exception):
    pass


def _within(seconds, fn):
    """Run fn(); an infinite loop FAILS (raises _Hang after 3s) instead of hanging."""
    def _h(signum, frame):
        raise _Hang("did not return within %ss (infinite loop?)" % seconds)
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _rejects(fn):
    """fn must raise ValueError/TypeError within 3s (a hang or OverflowError fails)."""
    with pytest.raises(REJECT):
        _within(3, fn)


# (1) TenantFloor.admit timestamp validation
@pytest.mark.parametrize("bad", [NAN, INF, -INF, True, None, "5"])
def test_admit_rejects_bad_timestamp(bad):
    _rejects(lambda: TenantFloor(cfg()).admit(bad, 1))


def test_admit_valid_float_timestamp_works():
    r = TenantFloor(cfg()).admit(5.0, 1)
    assert r.admitted == 1


# (2) dispatcher request validation
def _run(*reqs, **cfgkw):
    return AssuredFirstDispatcher(cfg(batch_slots=1, **cfgkw)).run(list(reqs))


def _mk(**kw):
    base = dict(id="x", cls="assured", arrival_ts=0.0, service_s=1.0, deadline_ts=1000.0)
    base.update(kw)
    return Request(**base)


@pytest.mark.parametrize(
    "override",
    [
        {"arrival_ts": NAN}, {"arrival_ts": INF}, {"arrival_ts": True},
        {"arrival_ts": None},
        {"service_s": NAN}, {"service_s": INF}, {"service_s": 0},
        {"service_s": -1}, {"service_s": True},
        {"deadline_ts": NAN}, {"deadline_ts": True},
        {"cls": "vip"}, {"cls": None},
        {"id": ""}, {"id": 5},
    ],
    ids=lambda o: str(o),
)
def test_dispatcher_rejects_invalid_request(override):
    _rejects(lambda: _run(_mk(**override)))


def test_dispatcher_rejects_duplicate_ids():
    _rejects(lambda: _run(_mk(id="d", arrival_ts=0.0), _mk(id="d", arrival_ts=1.0)))


def test_dispatcher_does_not_mutate_inputs():
    reqs = [_mk(id="a", arrival_ts=1.0), _mk(id="b", cls="opportunistic", arrival_ts=0.0)]
    before = copy.deepcopy(reqs)
    _within(3, lambda: AssuredFirstDispatcher(cfg(batch_slots=1)).run(reqs))
    assert reqs == before
    assert [id(r) for r in reqs] == [id(r) for r in reqs]  # list order untouched
    assert [r.id for r in reqs] == ["a", "b"]


def test_dispatcher_unsorted_input_processed_by_arrival():
    # batch_slots=1. b arrives 0.0 (opportunistic, svc 2) runs 0..2 alone;
    # a1 arrives 1.0, a2 arrives 0.5 (both assured, svc 1) wait; at t=2 FIFO by
    # arrival: a2 (0.5) 2..3, then a1 (1.0) 3..4. Input given unsorted.
    out = {o.id: o for o in _within(3, lambda: AssuredFirstDispatcher(cfg(batch_slots=1)).run([
        _mk(id="a1", arrival_ts=1.0, service_s=1.0),
        _mk(id="a2", arrival_ts=0.5, service_s=1.0),
        _mk(id="b", cls="opportunistic", arrival_ts=0.0, service_s=2.0),
    ]))}
    assert (out["b"].start_ts, out["b"].finish_ts) == (0.0, 2.0)
    assert (out["a2"].start_ts, out["a2"].finish_ts) == (2.0, 3.0)
    assert (out["a1"].start_ts, out["a1"].finish_ts) == (3.0, 4.0)


# (3) simulate_guard argument table
_SG = dict(seed=1, n_windows=2, offered_assured=1, offered_opportunistic=1,
           cfg=FloorConfig(credits=2, window_s=10.0, batch_slots=1, s_max=1.0),
           guarded=True)


def _sg(**kw):
    a = dict(_SG)
    a.update(kw)
    return simulate_guard(a["seed"], a["n_windows"], a["offered_assured"],
                          a["offered_opportunistic"], a["cfg"], a["guarded"])


def test_simulate_guard_valid_args_baseline():
    assert _within(3, _sg)["simulated"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"seed": None}, {"seed": 1.5}, {"seed": True}, {"seed": "1"},
        {"n_windows": 0}, {"n_windows": -1}, {"n_windows": True}, {"n_windows": 2.5},
        {"offered_assured": -1}, {"offered_assured": True}, {"offered_assured": 1.5},
        {"offered_opportunistic": -1}, {"offered_opportunistic": True},
        {"guarded": 1}, {"guarded": None}, {"guarded": "yes"},
        {"cfg": None},
    ],
    ids=lambda o: str(o),
)
def test_simulate_guard_rejects_bad_args(override):
    _rejects(lambda: _sg(**override))


# (4) backlog validation
_BAD_BACKLOG = [-1, True, NAN, 1.5, None, "2"]


@pytest.mark.parametrize("bad", _BAD_BACKLOG, ids=repr)
def test_backlog_validation_everywhere(bad):
    c = cfg(credits=4, window_s=1.0, batch_slots=2, s_max=0.5)
    _rejects(lambda: check_preconditions(c, bad))
    _rejects(lambda: require_guarantee(c, bad))
    _rejects(lambda: floor_sojourn_bound(c, bad))


def test_backlog_zero_and_B_work():
    c = cfg(credits=4, window_s=1.0, batch_slots=2, s_max=0.5)
    assert check_preconditions(c, 0)["backlog_ok"] is True
    assert check_preconditions(c, 4)["backlog_ok"] is True


# (5) FloorConfig upper sanity
@pytest.mark.parametrize("override", [{"credits": 10**6 + 1}, {"batch_slots": 10**6 + 1}],
                         ids=lambda o: str(o))
def test_floor_config_rejects_runaway(override):
    with pytest.raises(ValueError):
        cfg(**override)
