"""S3.2 slim suite: mlops.drift (PSI, KS, KL/JS, entropy, kappa, DriftMonitor).
Frozen BEFORE the module exists. Oracles are hand-computed (see comments).
Levels: PSI_WARN 0.1, PSI_ALERT 0.25, KS_ALERT 0.1, all comparisons strict.
"""
import dataclasses
import math
import os
import random
import signal
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.drift import (
    DriftMonitor,
    DriftReport,
    classify_drift,
    cohens_kappa,
    drift_context,
    js_divergence,
    kl_divergence,
    ks_statistic,
    predictive_entropy,
    psi,
    psi_from_samples,
)
from mlops.kernel import Conflict
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule

LN2 = math.log(2)
NAN, INF = float("nan"), float("inf")


def _within(seconds, fn):
    """Run fn under an alarm so a hang FAILS instead of stalling the suite."""
    def _boom(signum, frame):
        raise AssertionError("call did not return within %ss" % seconds)

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _gauss(seed, n, shift=0.0):
    r = random.Random(seed)
    return [r.gauss(shift, 1.0) for _ in range(n)]


def _monitor(**kw):
    return DriftMonitor(_gauss(1, 100), **kw)


# ------------------------------------------------------------------ psi

def test_psi_hand_values_and_symmetry():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    # (0.9-0.5)ln(0.9/0.5) + (0.1-0.5)ln(0.1/0.5) = 0.4 ln1.8 + 0.4 ln5 = 0.878890
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.4 * math.log(1.8) + 0.4 * math.log(5), abs=1e-5)
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.878890, abs=1e-5)
    for a, b in [([0.2, 0.3, 0.5], [0.4, 0.4, 0.2]), ([0.7, 0.3], [0.1, 0.9])]:
        assert psi(a, b) == pytest.approx(psi(b, a), abs=1e-9)


def test_psi_zero_bin_floored_with_eps():
    v = psi([0.5, 0.5], [1.0, 0.0])
    assert math.isfinite(v) and v > 0
    v2 = psi([1.0, 0.0], [0.5, 0.5])
    assert math.isfinite(v2) and v2 > 0


@pytest.mark.parametrize("e,a", [
    ([0.5, 0.4], [0.5, 0.5]),          # sum 0.9
    ([0.5, 0.5], [0.6, 0.6]),          # sum 1.2
    ([1.5, -0.5], [0.5, 0.5]),         # negative
    ([0.5, NAN], [0.5, 0.5]),
    ([0.5, 0.5], [INF, 0.0]),
    ([True, 0.0], [0.5, 0.5]),         # bool is not a number
    (["0.5", "0.5"], [0.5, 0.5]),
    ([0.5, 0.5, 0.0], [0.5, 0.5]),     # unequal lengths
    ([1.0], [1.0]),                    # length < 2
    ([0.5, 0.5], [0.5, 0.5000011]),    # off by > 1e-6
])
def test_psi_rejects_bad_proportions(e, a):
    with pytest.raises(ValueError):
        psi(e, a)


# ------------------------------------------------------------- psi_from_samples

def test_psi_from_samples_identical_and_shifted():
    s = _gauss(3, 100)
    assert psi_from_samples(s, list(s)) == pytest.approx(0.0, abs=1e-6)
    # shift of +3 sd: nearly all mass lands in the top bin -> far above PSI_ALERT
    assert psi_from_samples(s, _gauss(4, 100, shift=3.0)) > 0.25


@pytest.mark.parametrize("exp,act,bins", [
    (_gauss(1, 29), _gauss(2, 100), 10),               # too few expected
    (_gauss(1, 100), _gauss(2, 29), 10),               # too few actual
    (_gauss(1, 99) + [NAN], _gauss(2, 100), 10),
    (_gauss(1, 100), _gauss(2, 99) + [INF], 10),
    (_gauss(1, 99) + [True], _gauss(2, 100), 10),
    (_gauss(1, 99) + ["1"], _gauss(2, 100), 10),
    (_gauss(1, 100), _gauss(2, 100), 1),               # bins < 2
    (_gauss(1, 100), _gauss(2, 100), True),            # bool bins
])
def test_psi_from_samples_rejects(exp, act, bins):
    with pytest.raises(ValueError):
        psi_from_samples(exp, act, bins)


def test_psi_from_samples_constant_reference_terminates():
    def call():
        try:
            return psi_from_samples([5.0] * 50, _gauss(2, 50))
        except ValueError:
            return None

    out = _within(3, call)
    assert out is None or (math.isfinite(out) and out >= 0)


# ------------------------------------------------------------------ ks

@pytest.mark.parametrize("a,b,want", [
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([1, 2, 3], [4, 5, 6], 1.0),
    # ECDF gaps at 1,2,3,4,5,6: 0, 0, .25, .5, .25, 0 -> 0.5
    ([1, 2, 3, 4], [1, 2, 5, 6], 0.5),
    # ties: a=[1,2,2,3] b=[2,2,3,3]; F_a: .25,.75,1  F_b: 0,.5,1 -> max .25
    ([1, 2, 2, 3], [2, 2, 3, 3], 0.25),
    ([7, 7, 7], [7, 7], 0.0),          # all tied: no spurious gap
])
def test_ks_oracles(a, b, want):
    assert ks_statistic(a, b) == pytest.approx(want, abs=1e-12)
    assert ks_statistic(b, a) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("a,b", [
    ([1], [1, 2]), ([1, 2], [1]), ([], [1, 2]),
    ([1, NAN], [1, 2]), ([1, 2], [1, INF]),
    ([1, True], [1, 2]), ([1, 2], ["1", 2]),
])
def test_ks_rejects(a, b):
    with pytest.raises(ValueError):
        ks_statistic(a, b)


# ------------------------------------------------------- kl / js / entropy

def test_kl_oracles_and_asymmetry():
    # 0.5 ln(0.5/0.25) + 0.5 ln(0.5/0.75) = 0.5 ln2 + 0.5 ln(2/3) = 0.143841
    assert kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.143841, abs=1e-5)
    assert kl_divergence([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence([0.25, 0.75], [0.5, 0.5]) != pytest.approx(kl_divergence([0.5, 0.5], [0.25, 0.75]), abs=1e-4)
    v = kl_divergence([0.5, 0.5], [1.0, 0.0])   # zero q where p > 0: floored, finite
    assert math.isfinite(v) and v > 0


def test_js_oracles_symmetry_range():
    p, q = [0.5, 0.5], [0.25, 0.75]
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-12)
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(LN2, abs=1e-6)  # 0.693147
    r = random.Random(7)
    for _ in range(50):
        x, y = [r.random() for _ in range(4)], [r.random() for _ in range(4)]
        x, y = [v / sum(x) for v in x], [v / sum(y) for v in y]
        assert -1e-12 <= js_divergence(x, y) <= LN2 + 1e-9


@pytest.mark.parametrize("probs,want", [
    ([0.5, 0.5], LN2),
    ([1.0, 0.0], 0.0),
    ([0.25] * 4, math.log(4)),
])
def test_predictive_entropy_oracles(probs, want):
    assert predictive_entropy(probs) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("bad", [[0.5, 0.4], [1.2, -0.2], [NAN, 1.0], [INF, 0.0], [True, 0.0], ["1"], []])
def test_distribution_functions_reject_invalid(bad):
    with pytest.raises(ValueError):
        predictive_entropy(bad)
    with pytest.raises(ValueError):
        kl_divergence(bad, [0.5, 0.5])
    with pytest.raises(ValueError):
        js_divergence([0.5, 0.5], bad)


# ---------------------------------------------------------------- kappa

@pytest.mark.parametrize("a,b,want", [
    # po = 3/4; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75-0.5)/(1-0.5) = 0.5
    ([1, 1, 0, 0], [1, 0, 0, 0], 0.5),
    ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
    ([1, 1, 1], [1, 1, 1], 1.0),       # pe = 1: defined as 1.0, no ZeroDivisionError
])
def test_kappa_oracles(a, b, want):
    assert cohens_kappa(a, b) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("a,b", [([], []), ([1, 0], [1]), ([1], [1, 0])])
def test_kappa_rejects(a, b):
    with pytest.raises(ValueError):
        cohens_kappa(a, b)


# ------------------------------------------------------------ classify

@pytest.mark.parametrize("p,k,want", [
    (0.05, 0.05, "ok"),
    (0.10, 0.05, "ok"),      # exactly PSI_WARN is not warn
    (0.11, 0.05, "warn"),
    (0.25, 0.05, "warn"),    # exactly PSI_ALERT is not alert
    (0.26, 0.05, "alert"),
    (0.05, 0.10, "ok"),      # exactly KS_ALERT is not alert
    (0.05, 0.11, "alert"),
    (0.15, 0.11, "alert"),   # alert dominates warn
])
def test_classify_drift_table(p, k, want):
    assert classify_drift(p, k) == want


@pytest.mark.parametrize("p,k", [(-0.1, 0.0), (0.0, -0.1), (NAN, 0.0), (0.0, NAN), (INF, 0.0), (True, 0.0), (0.0, False), ("0.1", 0.0)])
def test_classify_drift_rejects(p, k):
    with pytest.raises(ValueError):
        classify_drift(p, k)


# -------------------------------------------------------------- monitor

def test_monitor_same_distribution_ok_and_shifted_alert():
    m = _monitor()
    for v in _gauss(5, 200):        # same N(0,1); checked offline: PSI ~0.08, KS ~0.08
        m.observe(v)
    rep = m.check()
    assert rep.level == "ok" and rep.n == 200

    m2 = _monitor()
    for v in _gauss(5, 200, shift=3.0):
        m2.observe(v)
    rep2 = m2.check()
    assert rep2.level == "alert" and rep2.psi > 0.25 and rep2.ks > 0.1


def test_monitor_needs_30_observations_and_window_bounded():
    m = _monitor()
    for v in _gauss(5, 29):
        m.observe(v)
    with pytest.raises(Conflict):
        m.check()
    m.observe(0.0)
    assert m.check().n == 30

    w = _monitor(window_size=200)
    for v in _gauss(6, 1000):
        w.observe(v)
    rep = w.check()
    assert rep.n == 200 and rep.window_full is True
    part = _monitor(window_size=200)
    for v in _gauss(6, 50):
        part.observe(v)
    assert part.check().window_full is False


@pytest.mark.parametrize("bad", [NAN, INF, -INF, True, "1", None])
def test_monitor_observe_rejects(bad):
    with pytest.raises(ValueError):
        _monitor().observe(bad)


@pytest.mark.parametrize("ref,kw", [
    (_gauss(1, 29), {}),                    # reference < 30
    (_gauss(1, 100), {"window_size": 29}),
    (_gauss(1, 100), {"window_size": 100001}),
    (_gauss(1, 100), {"bins": True}),
    (_gauss(1, 100), {"bins": 1}),
    (_gauss(1, 99) + [NAN], {}),
])
def test_monitor_constructor_validation(ref, kw):
    with pytest.raises(ValueError):
        DriftMonitor(ref, **kw)


def test_report_is_frozen_dataclass():
    m = _monitor()
    for v in _gauss(5, 40):
        m.observe(v)
    rep = m.check()
    assert isinstance(rep, DriftReport) and dataclasses.is_dataclass(rep)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.level = "alert"


# -------------------------------------------------------- drift_context

def test_drift_context_codes_and_policy_rule():
    pdp = PolicyDecisionPoint(PolicyBundle(
        rules=[Rule("d", 1, "max_metric", {"metric": "drift_level_code", "max": 1})],
        name="drift", version=1))
    seen = {}
    for level, code, psi_v, ks_v in [("ok", 0, 0.02, 0.03), ("warn", 1, 0.15, 0.05), ("alert", 2, 0.4, 0.3)]:
        rep = DriftReport(level=level, psi=psi_v, ks=ks_v, n=200, window_full=True)
        ctx = drift_context(rep)
        assert ctx["drift_level_code"] == code
        assert ctx["drift_psi"] == pytest.approx(psi_v)
        assert ctx["drift_ks"] == pytest.approx(ks_v)
        seen[level] = pdp.decide("promote", ctx).allow
    assert seen == {"ok": True, "warn": True, "alert": False}
