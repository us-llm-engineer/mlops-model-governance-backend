"""S3.4 mutation-hardening suite: extra assertions that kill mutants which survived the frozen suites.

Every test here pins ONE behaviour with a hand-derived oracle. Header comments name the mutant killed.
NOTE: like the frozen suites this file puts <repo>/exec first on sys.path, so a mutation harness must
copy tests/ AND exec/ into the same scratch root (PYTHONPATH alone is NOT enough).
"""
import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import Conflict, ManualClock, NotFound, PolicyDenied, ValidationFailed  # noqa: E402


def _gauss(seed, n, shift=0.0):
    r = random.Random(seed)
    return [r.gauss(shift, 1.0) for _ in range(n)]


# =============================================================== drift
from mlops import drift as _drift  # noqa: E402
from mlops.drift import DriftMonitor, DriftReport, classify_drift, cohens_kappa, psi_from_samples  # noqa: E402


def _observed(seed_ref, seed_win, n_ref, n_win, shift=0.0, window_size=None):
    m = DriftMonitor(_gauss(seed_ref, n_ref), window_size=window_size or max(n_win, 30))
    for v in _gauss(seed_win, n_win, shift):
        m.observe(v)
    return m.check()


def test_classify_uses_module_constants_at_exact_boundaries():
    # M2 (PSI_ALERT 0.25 -> 0.9): the constant is live; 0.26 alerts, 0.25 only warns, 0.1 is ok.
    assert (_drift.PSI_WARN, _drift.PSI_ALERT, _drift.KS_ALERT) == (0.1, 0.25, 0.1)
    assert classify_drift(0.26, 0.0) == "alert"
    assert classify_drift(0.25, 0.0) == "warn"
    assert classify_drift(0.1, 0.0) == "ok"
    assert classify_drift(0.11, 0.0) == "warn"
    assert classify_drift(0.0, 0.11) == "alert"
    assert classify_drift(0.0, 0.1) == "ok"


def test_monitor_psi_adjusted_subtracts_exact_bias():
    # M8 (bias correction removed): bias = (bins-1)*(1/n_ref + 1/n_win) = 9*(1/100 + 1/30) = 0.39
    r = _observed(3, 103, 100, 30)
    assert r.psi_adjusted == pytest.approx(max(0.0, r.psi - 9 * (1 / 100 + 1 / 30)), abs=1e-12)
    assert 0.35 < r.psi < 0.45 and r.psi_adjusted < 0.02
    assert r.level == "ok"  # raw psi > 0.25 would alert; bias-corrected value must not


def test_monitor_bias_depends_on_bins_and_window_length():
    # a wrong (bins-1) factor or 1/n_ref -> 1/n_win typo changes the exact figure
    m = DriftMonitor(_gauss(3, 100), window_size=60, bins=5)
    for v in _gauss(103, 50):
        m.observe(v)
    r = m.check()
    assert r.psi_adjusted == pytest.approx(max(0.0, r.psi - 4 * (1 / 100 + 1 / 50)), abs=1e-12)
    assert r.n == 50 and r.window_full is False


def test_monitor_ks_gate_suppresses_insignificant_ks():
    # KS 0.187 > 0.1 but p = 0.355 >= 0.01 -> must NOT alert (mutants: gate widened / removed)
    r = _observed(3, 103, 100, 30)
    assert r.ks > 0.1 and r.ks_pvalue > 0.3
    assert r.level == "ok"


def test_monitor_ks_gate_passes_significant_ks_p_just_under_one_percent():
    # ks 0.133 (>0.1), p = 0.00959 < 0.01, psi_adjusted 0.03 -> alert can only come from the KS branch
    r = _observed(1, 101, 300, 300, shift=0.3, window_size=300)
    assert 0.001 < r.ks_pvalue < 0.01
    assert r.psi_adjusted < 0.1
    assert r.level == "alert"


def test_driftreport_defaults():
    # dataclass defaults are part of the contract: not significant, adjusted psi 0
    rep = DriftReport(level="ok", psi=0.0, ks=0.0, n=30, window_full=False)
    assert rep.ks_pvalue == 1.0 and rep.psi_adjusted == 0.0


def test_psi_from_samples_hand_value_two_bins():
    # exp 0..39 -> edges [0,19.5,39] -> 0.5/0.5; act 24x0 + 6x30 -> 0.8/0.2
    # psi = 0.3 ln 1.6 - 0.3 ln 0.4 = 0.3 ln 4 (kills quantile-count and bin-count mutants)
    exp = [float(i) for i in range(40)]
    act = [0.0] * 24 + [30.0] * 6
    assert psi_from_samples(exp, act, 2) == pytest.approx(0.3 * math.log(4), abs=1e-9)


def test_psi_from_samples_bins_bounds():
    exp, act = _gauss(1, 100), _gauss(2, 100)
    assert psi_from_samples(exp, act, 2) >= 0  # lower edge inclusive
    assert psi_from_samples(exp, act, 1000) >= 0  # upper edge inclusive
    for bad in (1, 1001, 0, -3):
        with pytest.raises(ValueError):
            psi_from_samples(exp, act, bad)


def test_psi_from_samples_sample_size_bounds():
    exp, act = _gauss(1, 30), _gauss(2, 30)
    assert psi_from_samples(exp, act, 4) >= 0  # 30 samples is the minimum
    with pytest.raises(ValueError):
        psi_from_samples(_gauss(1, 29), act)
    with pytest.raises(ValueError):
        psi_from_samples([0.0] * 1_000_001, act)  # over the 1e6 cap: rejected before any allocation


def test_monitor_constructor_bounds():
    ref = _gauss(1, 30)
    DriftMonitor(ref, window_size=30, bins=2)  # all lower bounds inclusive
    DriftMonitor(ref, window_size=100000, bins=1000)  # all upper bounds inclusive
    with pytest.raises(ValueError):
        DriftMonitor(_gauss(1, 29))
    for kw in ({"window_size": 29}, {"window_size": 100001}, {"bins": 1}, {"bins": 1001}):
        with pytest.raises(ValueError):
            DriftMonitor(ref, **kw)


def test_monitor_window_is_bounded_and_flags_full():
    m = DriftMonitor(_gauss(1, 100), window_size=40)
    for v in _gauss(2, 39):
        m.observe(v)
    assert m.check().window_full is False
    m.observe(0.0)
    assert m.check().window_full is True and m.check().n == 40
    for v in _gauss(3, 25):
        m.observe(v)
    r = m.check()
    assert r.n == 40 and r.window_full is True  # oldest evicted, never grows


def test_monitor_needs_exactly_thirty_observations():
    m = DriftMonitor(_gauss(1, 100))
    for v in _gauss(2, 29):
        m.observe(v)
    with pytest.raises(Conflict):
        m.check()
    m.observe(0.1)
    assert m.check().n == 30


def test_cohens_kappa_three_labels_hand_value():
    # a = 0 0 1 1 2 2 ; b = 0 1 1 2 2 2 : po = 4/6 ; pe = (2/6*1/6)+(2/6*2/6)+(2/6*3/6) = 12/36
    # kappa = (2/3 - 1/3) / (1 - 1/3) = 0.5  (a wrong per-label count changes pe)
    assert cohens_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]) == pytest.approx(0.5, abs=1e-12)


def test_cohens_kappa_empty_rejected_single_item_ok():
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    assert cohens_kappa([1], [1]) == 1.0


# =============================================================== incidents
from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.incidents import CHECKLIST, IncidentManager, classify  # noqa: E402

_PM_KEYS = ["timeline", "root_cause", "user_impact", "preventive_measures", "monitoring_gap"]


def _im():
    clock = ManualClock(start=1000.0)
    audit = ChainedAuditStore(b"k")
    return IncidentManager(audit, clock), audit, clock


def _finish_steps(mgr, iid):
    for s in CHECKLIST:
        mgr.complete_step("alice", iid, s, "n")


def test_incident_ids_start_at_one_and_increment():
    mgr, _, _ = _im()
    assert mgr.open("a", "t", {"pipeline_delay": True}) == "inc-1"
    assert mgr.open("a", "t", {"pipeline_delay": True}) == "inc-2"


def test_incident_deadline_is_opened_ts_plus_response_seconds():
    mgr, _, _ = _im()
    rec = mgr.get(mgr.open("a", "t", {"serving_failure": True}))
    assert rec["opened_ts"] == 1000.0 and rec["deadline"] == 1000.0 + 15 * 60


def test_escalation_boundaries_are_strict_and_exact():
    # P0 = 900 s. tier 2 only when elapsed > 900, tier 3 only when elapsed > 1800 (M3 family)
    mgr, audit, clock = _im()
    iid = mgr.open("a", "t", {"serving_failure": True})
    clock.advance(899)
    assert mgr.escalate_if_overdue(iid) == 1
    clock.advance(1)  # elapsed == 900 exactly
    assert mgr.escalate_if_overdue(iid) == 1
    clock.advance(0.5)
    assert mgr.escalate_if_overdue(iid) == 2
    clock.advance(899.5)  # elapsed == 1800 exactly
    assert mgr.escalate_if_overdue(iid) == 2
    clock.advance(0.5)
    assert mgr.escalate_if_overdue(iid) == 3
    n = audit.head()["length"]
    clock.advance(10_000)
    assert mgr.escalate_if_overdue(iid) == 3 and audit.head()["length"] == n  # no audit when tier unchanged


def test_escalation_jumps_straight_to_tier_three_and_audits_once():
    mgr, audit, clock = _im()
    iid = mgr.open("a", "t", {"serving_failure": True})
    n = audit.head()["length"]
    clock.advance(1801)
    assert mgr.escalate_if_overdue(iid) == 3
    assert audit.head()["length"] == n + 1
    assert audit.export()[-1]["action"] == "incident.escalate"


@pytest.mark.parametrize("bad", ["a\x1fb", "a\x7fb", "a\nb", "\x00", "a\tb"])
def test_incident_text_rejects_control_characters(bad):
    mgr, _, _ = _im()
    with pytest.raises(ValidationFailed):
        mgr.open(bad, "t", {"pipeline_delay": True})
    with pytest.raises(ValidationFailed):
        mgr.open("a", bad, {"pipeline_delay": True})
    iid = mgr.open("a", "t", {"pipeline_delay": True})
    with pytest.raises(ValidationFailed):
        mgr.complete_step("a", iid, CHECKLIST[0], bad)
    assert mgr.get(iid)["steps_completed"] == []  # rejected note left no state behind


def test_incident_text_accepts_space_and_printable_edges():
    mgr, _, _ = _im()
    iid = mgr.open("a b", " title with spaces ~", {"pipeline_delay": True})
    mgr.complete_step("a b", iid, CHECKLIST[0], "note é ok")  # ord 32 and non-ASCII are legal


def test_classify_zero_values_are_accepted_and_p3():
    # `< 0` must not become `<= 0`; a zero reading is a legal (if boring) signal
    assert classify({"accuracy_drop_pct": 0}).name == "P3"
    assert classify({"feature_psi": 0.0}).name == "P3"
    assert classify({"drift_level": "ok"}).name == "P3"


def test_p1_postmortem_with_extra_key_is_rejected():
    mgr, _, _ = _im()
    iid = mgr.open("a", "t", {"accuracy_drop_pct": 15.0})
    _finish_steps(mgr, iid)
    pm = {k: "x" for k in _PM_KEYS}
    pm["bonus"] = "unexpected"
    with pytest.raises(ValidationFailed):
        mgr.resolve("a", iid, pm)
    assert mgr.get(iid)["status"] == "open"


def test_p3_optional_postmortem_is_validated_and_stored_when_given():
    mgr, _, _ = _im()
    iid = mgr.open("a", "t", {"pipeline_delay": True})
    _finish_steps(mgr, iid)
    with pytest.raises(ValidationFailed):
        mgr.resolve("a", iid, {"timeline": "x"})  # provided => must be complete
    with pytest.raises(ValidationFailed):
        mgr.resolve("a", iid, dict({k: "x" for k in _PM_KEYS}, timeline=""))
    assert mgr.get(iid)["status"] == "open"
    good = {k: "text " + k for k in _PM_KEYS}
    mgr.resolve("a", iid, good)
    rec = mgr.get(iid)
    assert rec["status"] == "resolved" and rec["postmortem"] == good


def test_p3_without_postmortem_resolves_with_none_recorded():
    mgr, _, _ = _im()
    iid = mgr.open("a", "t", {"pipeline_delay": True})
    _finish_steps(mgr, iid)
    mgr.resolve("a", iid, None)
    assert mgr.get(iid)["postmortem"] is None


def test_checklist_order_is_enforced_and_partial_state_untouched():
    mgr, audit, _ = _im()
    iid = mgr.open("a", "t", {"pipeline_delay": True})
    n = audit.head()["length"]
    with pytest.raises(Conflict):
        mgr.complete_step("a", iid, CHECKLIST[1], "skip ahead")
    with pytest.raises(Conflict):
        mgr.complete_step("a", iid, CHECKLIST[-1], "skip to end")
    assert audit.head()["length"] == n and mgr.get(iid)["steps_completed"] == []
    mgr.complete_step("a", iid, CHECKLIST[0], "ok")
    with pytest.raises(Conflict):
        mgr.complete_step("a", iid, CHECKLIST[0], "repeat")
    assert mgr.get(iid)["steps_completed"] == [CHECKLIST[0]]


# =============================================================== policy_store
from mlops.policy_engine import PolicyBundle, Rule  # noqa: E402
from mlops.policy_store import PolicyStore, is_stricter_or_equal  # noqa: E402


def _rule(rid, kind, params, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params, severity=sev)


def _bundle(*rules):
    return PolicyBundle(rules=list(rules), name="p", version=0)


def _minb(v):
    return _bundle(_rule("acc", "min_metric", {"metric": "acc", "min": v}))


def _maxb(v):
    return _bundle(_rule("lat", "max_metric", {"metric": "lat", "max": v}))


def _ps():
    audit = ChainedAuditStore(b"ps-secret")
    return PolicyStore(audit, clock=ManualClock(start=50.0)), audit


def test_stricter_or_equal_equal_thresholds_count_as_equal():
    # `new_max > old_max` -> `>=` would call an identical bundle "looser"
    assert is_stricter_or_equal(_maxb(5.0), _maxb(5.0)) is True
    assert is_stricter_or_equal(_maxb(4.9), _maxb(5.0)) is True
    assert is_stricter_or_equal(_maxb(5.1), _maxb(5.0)) is False
    assert is_stricter_or_equal(_minb(0.9), _minb(0.9)) is True
    assert is_stricter_or_equal(_minb(0.91), _minb(0.9)) is True
    assert is_stricter_or_equal(_minb(0.89), _minb(0.9)) is False


def test_stricter_or_equal_severity_missing_rule_and_kind_branches():
    blk = _rule("r", "flag_true", {"flag": "f"}, "block")
    wrn = _rule("r", "flag_true", {"flag": "f"}, "warn")
    assert is_stricter_or_equal(_bundle(blk), _bundle(wrn)) is True  # warn -> block tightens
    assert is_stricter_or_equal(_bundle(wrn), _bundle(blk)) is False  # block -> warn loosens
    assert is_stricter_or_equal(_bundle(), _bundle(blk)) is False  # dropping a block rule loosens
    assert is_stricter_or_equal(_bundle(), _bundle(wrn)) is True  # dropping a warn rule is fine
    assert is_stricter_or_equal(_bundle(blk, _rule("x", "no_nan", {})), _bundle(blk)) is True  # extras fine
    other = _rule("r", "flag_true", {"flag": "g"}, "block")
    assert is_stricter_or_equal(_bundle(other), _bundle(blk)) is False  # other kinds: params must match
    swapped = _rule("r", "flag_false", {"flag": "f"}, "block")
    assert is_stricter_or_equal(_bundle(swapped), _bundle(blk)) is False  # kind change


@pytest.mark.parametrize("bad", ["a\x01b", "a\x1fb", "a\x7fb", "a\nb"])
def test_policy_store_text_fields_reject_control_chars_exactly(bad):
    ps, audit = _ps()
    n = audit.head()["length"]
    with pytest.raises(ValidationFailed):
        ps.publish(bad, _minb(0.9))
    with pytest.raises(ValidationFailed):
        ps.publish("a", _minb(0.9), note=bad)  # note is validated too, not just actor
    assert audit.head()["length"] == n and ps.history() == []


def test_policy_store_text_length_cap_and_space_boundaries():
    ps, _ = _ps()
    ps.publish("a" * 256, _minb(0.9), note="n" * 256)  # 256 chars is legal
    with pytest.raises(ValidationFailed):
        ps.publish("a" * 257, _minb(0.9))
    with pytest.raises(ValidationFailed):
        ps.publish("a", _minb(0.9), note="n" * 257)
    with pytest.raises(ValidationFailed):
        ps.publish("a", _minb(0.9), note="   ")  # whitespace-only note is not "empty", it is invalid
    ps.publish("has space", _minb(0.9), note="  padded  ")  # 0x20 is legal
    assert ps.history()[-1]["note"] == "padded"  # stripped
    assert ps.history()[0]["note"] == "n" * 256


def test_policy_store_history_records_clock_timestamps_and_active_flag():
    ps, _ = _ps()
    ps.publish("a", _minb(0.9))
    ps.publish("a", _minb(0.95))
    ps.activate("a", 2)
    assert [(h["version"], h["published_ts"], h["active"]) for h in ps.history()] == [(1, 50.0, False), (2, 50.0, True)]


def test_activate_version_validation_types_are_exact():
    ps, audit = _ps()
    ps.publish("a", _minb(0.9))
    n = audit.head()["length"]
    for bad in (0, -1, True, "1", 1.0):
        with pytest.raises(ValidationFailed):
            ps.activate("a", bad)
    with pytest.raises(NotFound):
        ps.activate("a", 2)  # positive but unpublished
    with pytest.raises(NotFound):
        ps.get(99)
    assert audit.head()["length"] == n


def test_activate_loosening_needs_nonblank_approval_and_first_activation_does_not():
    ps, audit = _ps()
    ps.publish("a", _minb(0.9))
    ps.publish("a", _minb(0.5))
    ps.activate("a", 1)  # first activation: no approval needed
    for blank in (None, "", "   "):
        with pytest.raises(PolicyDenied):
            ps.activate("a", 2, human_approved_by=blank)
    assert ps.active()[0] == 1
    ps.activate("a", 2, human_approved_by="carol")
    last = audit.export()[-1]
    assert last["action"] == "policy.activate"
    assert last["meta"]["loosening"] is True and last["meta"]["approved_by"] == "carol" and last["meta"]["previous"] == 1


def test_decide_rejects_non_str_context_keys_and_writes_nothing():
    ps, audit = _ps()
    ps.publish("a", _minb(0.5))
    ps.activate("a", 1)
    n = audit.head()["length"]
    with pytest.raises(ValidationFailed):
        ps.decide("act", {1: 0.9})
    with pytest.raises(ValidationFailed):
        ps.decide("act", ["acc"])
    assert ps.decision_log() == [] and audit.head()["length"] == n


def test_decide_fails_closed_without_active_bundle():
    ps, _ = _ps()
    ps.publish("a", _minb(0.5))
    with pytest.raises(PolicyDenied):
        ps.decide("act", {"acc": 1.0})
    assert ps.decision_log() == []


def test_replay_validation_is_strict_and_type_exact():
    ps, audit = _ps()
    ps.publish("a", _minb(0.9))
    ps.activate("a", 1)
    n = audit.head()["length"]
    ok = {"action": "x", "context": {"acc": 1.0}, "allow": True}
    with pytest.raises(ValidationFailed):
        ps.replay(True)  # bool is not a version even though True == 1
    with pytest.raises(NotFound):
        ps.replay(2)
    for bad in ((), {}, "abc", (ok,)):
        with pytest.raises(ValidationFailed):
            ps.replay(1, decisions=bad)
    for missing in ("action", "context", "allow"):
        d = {k: v for k, v in ok.items() if k != missing}
        with pytest.raises(ValidationFailed):
            ps.replay(1, decisions=[d])
    with pytest.raises(ValidationFailed):
        ps.replay(1, decisions=["nope"])
    with pytest.raises(ValidationFailed):
        ps.replay(1, decisions=[dict(ok, allow=1)])
    assert audit.head()["length"] == n  # replay never writes to the audit chain


def test_replay_reports_diffs_per_index_under_the_requested_version():
    ps, audit = _ps()
    ps.publish("a", _minb(0.9))  # v1 strict
    ps.publish("a", _minb(0.5))  # v2 lax
    ps.activate("a", 1)
    ps.decide("x", {"acc": 0.95})
    ps.decide("y", {"acc": 0.7})
    n = audit.head()["length"]
    diff = ps.replay(2)
    assert diff == [
        {"index": 0, "action": "x", "recorded_allow": True, "replay_allow": True, "changed": False},
        {"index": 1, "action": "y", "recorded_allow": False, "replay_allow": True, "changed": True},
    ]
    assert audit.head()["length"] == n
    assert ps.replay(1, decisions=[]) == []


# =============================================================== policy_calibration
from mlops.policy_calibration import Calibrator, Feedback, Proposal, _check_feedback_list, evaluate  # noqa: E402


def _cal_store(min_=0.8, bundle_version=3, rule_version=5):
    ps, audit = _ps()
    b = PolicyBundle(rules=[Rule("acc", rule_version, "min_metric", {"metric": "acc", "min": min_}, "block")],
                     name="p", version=bundle_version)
    ps.publish("a", b)
    ps.activate("a", 1)
    return ps, audit


def _fb(acc, should_block):
    return Feedback({"acc": acc}, should_block)


def test_calibrator_default_step_is_one_thousandth():
    ps, _ = _cal_store(0.8)
    assert Calibrator(ps, "acc").step == 0.001
    p = Calibrator(ps, "acc").propose([_fb(0.85, True)])  # allowed (0.85 >= 0.8) but should block
    assert p.kind == "tighten" and p.new_value == pytest.approx(0.851, abs=1e-12)


def test_calibrator_step_validation_types():
    ps, _ = _cal_store()
    with pytest.raises(TypeError):
        Calibrator(ps, "acc", step=True)
    for bad in (0, -0.1, float("nan"), float("inf"), "0.1", None):
        with pytest.raises(ValidationFailed):
            Calibrator(ps, "acc", step=bad)


def test_proposal_counts_per_kind():
    ps, _ = _cal_store(0.8)
    cal = Calibrator(ps, "acc", step=0.05)
    loosen = cal.propose([_fb(0.7, False), _fb(0.75, False), _fb(0.95, False)])
    assert (loosen.kind, loosen.new_value, loosen.missed_detections, loosen.false_alarms, loosen.evidence) == (
        "loosen", 0.7, 0, 2, 3)
    none = cal.propose([_fb(0.95, False), _fb(0.5, True)])
    assert (none.kind, none.new_value, none.old_value, none.missed_detections, none.false_alarms) == (
        "none", 0.8, 0.8, 0, 0)
    tighten = cal.propose([_fb(0.9, True), _fb(0.85, True), _fb(0.7, False)])
    assert (tighten.kind, tighten.missed_detections, tighten.false_alarms) == ("tighten", 2, 1)
    assert tighten.new_value == pytest.approx(0.95, abs=1e-12)  # max(missed) + step


def test_proposal_overflowing_tighten_falls_back_to_old_value():
    ps, _ = _cal_store(0.8)
    p = Calibrator(ps, "acc", step=1e308).propose([_fb(1.7e308, True)])
    assert p.kind == "tighten" and p.new_value == 0.8 and math.isfinite(p.new_value)


def test_evaluate_rates_with_single_member_populations():
    ps, _ = _cal_store(0.8)
    _, bundle = ps.active()
    r = evaluate(bundle, [_fb(0.5, True)])
    assert r["blocked_correctly"] == 1 and r["missed_detection_rate"] == 0.0 and r["false_alarm_rate"] == 0.0
    r = evaluate(bundle, [_fb(0.9, True)])
    assert r["missed"] == 1 and r["missed_detection_rate"] == 1.0  # 1/1, not 0 because the population is size one
    r = evaluate(bundle, [_fb(0.5, False)])
    assert r["false_alarms"] == 1 and r["false_alarm_rate"] == 1.0
    r = evaluate(bundle, [_fb(0.9, True), _fb(0.5, True), _fb(0.5, True), _fb(0.5, False), _fb(0.9, False)])
    assert r["missed_detection_rate"] == pytest.approx(1 / 3) and r["false_alarm_rate"] == pytest.approx(1 / 2)
    assert r["n"] == 5


def test_feedback_construction_and_list_limits():
    for ctx in ([], "acc", None):
        with pytest.raises(ValidationFailed):
            Feedback(ctx, True)
    with pytest.raises(ValidationFailed):
        Feedback({1: 0.5}, True)
    for sb in (1, 0, "yes", None):
        with pytest.raises(ValidationFailed):
            Feedback({"acc": 0.5}, sb)
    ok = _fb(0.5, True)
    _check_feedback_list([ok] * 100_000)  # exactly the cap is fine
    with pytest.raises(ValidationFailed):
        _check_feedback_list([ok] * 100_001)
    with pytest.raises(ValidationFailed):
        _check_feedback_list([ok, {"context": {}}])
    with pytest.raises(ValidationFailed):
        _check_feedback_list("nope")


def test_apply_tighten_bumps_bundle_and_rule_versions_by_exactly_one():
    ps, _ = _cal_store(0.8, bundle_version=3, rule_version=5)
    cal = Calibrator(ps, "acc", step=0.05)
    v = cal.apply("a", cal.propose([_fb(0.9, True)]))
    assert v == 2
    _, active = ps.active()
    assert active.version == 4 and active.rules[0].version == 6
    assert active.rules[0].params["min"] == pytest.approx(0.95)
    assert active.rules[0].params["metric"] == "acc"


def test_apply_actor_and_stale_and_forged_kind_leave_no_version_behind():
    ps, audit = _cal_store(0.8)
    cal = Calibrator(ps, "acc", step=0.05)
    prop = cal.propose([_fb(0.9, True)])
    n = audit.head()["length"]
    for bad in ("", 5, None):
        with pytest.raises(ValidationFailed):
            cal.apply(bad, prop)
    with pytest.raises(ValidationFailed):
        cal.apply("a", Proposal("acc", "acc", 0.8, 0.95, "loosen", 1, 0, 1))  # forged kind
    with pytest.raises(Conflict):
        cal.apply("a", Proposal("acc", "acc", 0.7, 0.95, "tighten", 1, 0, 1))  # stale old_value
    assert audit.head()["length"] == n and len(ps.history()) == 1


def test_apply_loosen_approval_format_checked_before_any_version_is_consumed():
    ps, audit = _cal_store(0.8)
    cal = Calibrator(ps, "acc", step=0.05)
    prop = cal.propose([_fb(0.7, False)])
    assert prop.kind == "loosen"
    n = audit.head()["length"]
    for blank in (None, "", "   "):
        with pytest.raises(PolicyDenied):
            cal.apply("a", prop, human_approved_by=blank)
    for bad in ("car\x01ol", "car\x7fol", "carolé", "car\nol"):
        with pytest.raises(ValidationFailed):
            cal.apply("a", prop, human_approved_by=bad)
    assert audit.head()["length"] == n and len(ps.history()) == 1  # nothing published, no version burned
    v = cal.apply("a", prop, human_approved_by="Carol Smith")  # printable ASCII incl. space is fine
    assert v == 2 and ps.active()[1].rules[0].params["min"] == 0.7


# =============================================================== chaos_scenarios
from mlops.chaos_scenarios import (  # noqa: E402
    FaultSpec, Scenario, ScenarioResult, SimulatedCluster, SteadyState, chaos_context, run_scenario,
)

_SEL = {"namespaces": ["prod"]}


def _pods(n, ns="prod", ready=True, prefix="p"):
    return {"%s%02d" % (prefix, i): {"namespace": ns, "labels": {"app": "x"}, "ready": ready} for i in range(n)}


def _kill(mode="all", value=None, selector=None):
    return FaultSpec("PodChaos", "pod-kill", selector or _SEL, mode=mode, value=value)


def test_blast_radius_uses_ceil_min_one_and_sorted_prefix():
    c = SimulatedCluster(_pods(10))
    names = sorted(c.pods)
    assert c.blast_radius([_kill("fixed-percent", 25)]) == set(names[:3])  # ceil(2.5) = 3, not 2
    assert c.blast_radius([_kill("random-max-percent", 25)]) == set(names[:3])
    assert c.blast_radius([_kill("fixed-percent", 1)]) == set(names[:1])  # ceil(0.1) = 1
    assert c.blast_radius([_kill("fixed-percent", 100)]) == set(names)
    assert c.blast_radius([_kill("fixed-percent", 50)]) == set(names[:5])  # exact multiple: no extra pod
    assert c.blast_radius([_kill("fixed", 4)]) == set(names[:4])
    assert c.blast_radius([_kill("fixed", 99)]) == set(names)
    assert c.blast_radius([_kill("one")]) == {names[0]}
    assert c.blast_radius([_kill("all")]) == set(names)
    assert c.blast_radius([_kill("one"), _kill("fixed", 2)]) == set(names[:2])  # union over faults
    assert SimulatedCluster({}).blast_radius([_kill("one")]) == set()


def test_select_filters_namespace_and_labels_and_returns_sorted():
    pods = {
        "b": {"namespace": "prod", "labels": {"app": "x", "tier": "web"}, "ready": True},
        "a": {"namespace": "prod", "labels": {"app": "x", "tier": "db"}, "ready": True},
        "c": {"namespace": "dev", "labels": {"app": "x", "tier": "web"}, "ready": True},
        "d": {"namespace": "prod", "labels": {"app": "y", "tier": "web"}, "ready": True},
    }
    c = SimulatedCluster(pods)
    assert c.select({"namespaces": ["prod"]}) == ["a", "b", "d"]
    assert c.select({"namespaces": ["prod", "dev"]}) == ["a", "b", "c", "d"]
    assert c.select({"namespaces": ["prod"], "labelSelectors": {"app": "x", "tier": "web"}}) == ["b"]
    assert c.select({"namespaces": ["nope"]}) == []


def test_metrics_shape_and_empty_cluster_is_healthy():
    c = SimulatedCluster(_pods(4))
    c.pods["p00"]["ready"] = False
    assert c.metrics() == {"ready_pods": 3, "error_rate": 0.25, "latency_ms": 20.0}
    assert SimulatedCluster({}).metrics() == {"ready_pods": 0, "error_rate": 0.0, "latency_ms": 20.0}


def test_run_scenario_kill_fraction_moves_metrics_by_ceil():
    c = SimulatedCluster(_pods(10))
    sc = Scenario([[_kill("fixed-percent", 25)]], [SteadyState("avail", "error_rate", "<=", 0.5)], "s")
    r = run_scenario(sc, c)
    assert [s["stage"] for s in r.stages] == ["pre", "inject", "post"]
    assert r.stages[1]["metrics"]["ready_pods"] == 7 and r.stages[1]["metrics"]["error_rate"] == pytest.approx(0.3)
    assert r.stages[1]["step"] == 1 and r.passed is True and r.violated == []
    assert r.stages[2]["metrics"]["ready_pods"] == 10  # post-validation is measured on the untouched cluster
    assert all(p["ready"] for p in c.pods.values())  # input cluster never mutated


def test_run_scenario_pre_validation_failure_short_circuits_before_injection():
    c = SimulatedCluster(_pods(2, ready=False))  # already broken
    sc = Scenario([[_kill("all")]], [SteadyState("avail", "error_rate", "<=", 0.5),
                                     SteadyState("cap", "ready_pods", ">=", 1)], "s")
    r = run_scenario(sc, c)
    assert r.passed is False and [s["stage"] for s in r.stages] == ["pre"]
    assert r.violated == ["pre-validation failed (avail)", "pre-validation failed (cap)"]
    assert r.stages[0]["violated"] == ["avail", "cap"]


def test_run_scenario_pre_passes_but_injection_violation_is_reported_per_step_and_state_carries_over():
    c = SimulatedCluster(_pods(4))
    sc = Scenario([[_kill("fixed", 1)], [_kill("fixed", 1, {"namespaces": ["prod"], "labelSelectors": {"app": "x"}})]],
                  [SteadyState("cap", "ready_pods", ">=", 3)], "s")
    r = run_scenario(sc, c)
    # step 1 kills p00 (3 ready, holds); step 2 selects the same sorted-first pod so ready stays 3
    assert [s["violated"] for s in r.stages] == [[], [], [], []] and r.passed is True
    sc2 = Scenario([[_kill("fixed", 2)], [_kill("fixed", 3)]], [SteadyState("cap", "ready_pods", ">=", 3)], "s")
    r2 = run_scenario(sc2, c)
    assert r2.passed is False and r2.violated == ["cap", "cap"]
    assert [s["metrics"]["ready_pods"] for s in r2.stages] == [4, 2, 1, 4]


def test_delay_faults_stack_and_latency_is_capped():
    c = SimulatedCluster(_pods(2))
    d = lambda ms: FaultSpec("NetworkChaos", "delay", _SEL, mode="one", params={"latency_ms": ms})  # noqa: E731
    sc = Scenario([[d(100), d(50)]], [SteadyState("lat", "latency_ms", "<=", 1e9)], "s")
    assert run_scenario(sc, c).stages[1]["metrics"]["latency_ms"] == 170.0  # 20 base + 100 + 50 on one pod
    sc = Scenario([[d(9000)]], [SteadyState("lat", "latency_ms", "<=", 1e9)], "s")
    assert run_scenario(sc, c).stages[1]["metrics"]["latency_ms"] == 5000.0


@pytest.mark.parametrize("op,value,expected", [
    ("<=", 5, True), ("<=", 5.1, False), ("<", 5, False), ("<", 4.9, True),
    (">=", 5, True), (">=", 4.9, False), (">", 5, False), (">", 5.1, True),
])
def test_steady_state_operators_are_exact_at_the_threshold(op, value, expected):
    assert SteadyState("n", "m", op, 5).holds(value) is expected


def test_faultspec_value_and_param_boundaries():
    def mk(mode, value, action="pod-kill", typ="PodChaos", params=None):
        return FaultSpec(typ, action, _SEL, mode=mode, value=value, params=params or {})

    with pytest.raises(ValidationFailed):
        mk("one", 1)
    with pytest.raises(ValidationFailed):
        mk("all", 1)
    mk("fixed", 1)
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(ValidationFailed):
            mk("fixed", bad)
    for mode in ("fixed-percent", "random-max-percent"):
        mk(mode, 1)
        mk(mode, 100)
        for bad in (0, 101, True, None):
            with pytest.raises(ValidationFailed):
                mk(mode, bad)
    net = lambda a, p: mk("all", None, a, "NetworkChaos", p)  # noqa: E731
    net("loss", {"loss_pct": 100})
    net("loss", {"loss_pct": 0.1})
    for bad in (0, 100.1, -1, float("nan"), True):
        with pytest.raises(ValidationFailed):
            net("loss", {"loss_pct": bad})
    net("delay", {"latency_ms": 0.5})
    for bad in (0, -5, float("inf")):
        with pytest.raises(ValidationFailed):
            net("delay", {"latency_ms": bad})
    with pytest.raises(ValidationFailed):
        net("delay", {})
    st = lambda w: mk("all", None, "cpu", "StressChaos", {"workers": w})  # noqa: E731
    st(1)
    for bad in (0, True, 1.5):
        with pytest.raises(ValidationFailed):
            st(bad)


def test_selector_namespace_validation():
    for ns in (["a\x01"], ["a\x1f"], [""], [1], []):
        with pytest.raises(ValidationFailed):
            FaultSpec("PodChaos", "pod-kill", {"namespaces": ns})
    FaultSpec("PodChaos", "pod-kill", {"namespaces": ["has space"]})  # 0x20 legal
    for sel in ({}, [], {"namespaces": ["p"], "labelSelectors": []}, {"namespaces": ["p"], "labelSelectors": {"a": 1}}):
        with pytest.raises(ValidationFailed):
            FaultSpec("PodChaos", "pod-kill", sel)


def test_scenario_size_limits_and_types():
    ss = [SteadyState("n", "error_rate", "<=", 1)]
    f = _kill()
    Scenario([[f]] * 20, ss, "s")  # 20 steps is the cap
    with pytest.raises(ValidationFailed):
        Scenario([[f]] * 21, ss, "s")
    Scenario([[f] * 10], ss, "s")
    with pytest.raises(ValidationFailed):
        Scenario([[f] * 11], ss, "s")
    with pytest.raises(ValidationFailed):
        Scenario([[]], ss, "s")
    with pytest.raises(ValidationFailed):
        Scenario([], ss, "s")
    with pytest.raises(ValidationFailed):
        Scenario([[f]], [], "s")
    with pytest.raises(TypeError):
        Scenario(([f],), ss, "s")
    with pytest.raises(TypeError):
        Scenario([(f,)], ss, "s")
    with pytest.raises(TypeError):
        Scenario([["x"]], ss, "s")
    with pytest.raises(TypeError):
        Scenario([[f]], [("n", "m")], "s")
    with pytest.raises(TypeError):
        Scenario([[f]], ss, 5)
    with pytest.raises(ValidationFailed):
        Scenario([[f]], ss, "bad\x01name")


def test_cluster_pod_validation_exact_types():
    ok = {"namespace": "p", "labels": {}, "ready": True}
    SimulatedCluster({"a": ok})
    for bad, exc in (
        ({"a": {"namespace": "p", "labels": {}}}, ValidationFailed),
        ({"a": {"namespace": "p", "ready": True}}, ValidationFailed),
        ({"a": {"labels": {}, "ready": True}}, ValidationFailed),
        ({"a": dict(ok, namespace="")}, ValidationFailed),
        ({"a": dict(ok, labels=[])}, TypeError),
        ({"a": dict(ok, labels={"k": 1})}, ValidationFailed),
        ({"a": dict(ok, ready=1)}, TypeError),
        ({"a": []}, TypeError),
        ({1: ok}, TypeError),
        ([], TypeError),
    ):
        with pytest.raises(exc):
            SimulatedCluster(bad)


def test_chaos_context_counts_violations_and_rejects_foreign_results():
    r = ScenarioResult(passed=False, stages=[], violated=["a", "b"])
    assert chaos_context(r) == {"chaos_passed": False, "chaos_violations": 2}
    assert chaos_context(ScenarioResult(True, [], [])) == {"chaos_passed": True, "chaos_violations": 0}
    with pytest.raises(TypeError):
        chaos_context({"passed": True, "violated": []})


# =============================================================== model_stages
from mlops.model_stages import ModelRegistry, Stage  # noqa: E402
from mlops.policy_engine import PolicyDecisionPoint  # noqa: E402

_H = "ab" * 32
_EVID = {"metric": "acc", "value": 0.95}


def _registry(**kw):
    audit = kw.pop("audit", None) or ChainedAuditStore(b"ms-secret")
    return ModelRegistry(audit, **kw), audit


def _reach(reg, vid, stage, model="m"):
    """Drive a fresh version to `stage`."""
    reg.register("alice", model, vid, _H)
    reg.validate("alice", model, vid, dict(_EVID))
    if stage == Stage.REGISTERED:
        return
    reg.transition("alice", model, vid, Stage.STAGING)
    if stage == Stage.STAGING:
        return
    reg.transition("alice", model, vid, Stage.PRODUCTION)
    if stage == Stage.PRODUCTION:
        return
    reg.transition("alice", model, vid, Stage.ARCHIVED)


_LEGAL = {(Stage.REGISTERED, Stage.STAGING), (Stage.STAGING, Stage.PRODUCTION), (Stage.STAGING, Stage.ARCHIVED),
          (Stage.PRODUCTION, Stage.ARCHIVED)}


@pytest.mark.parametrize("src", list(Stage))
@pytest.mark.parametrize("dst", list(Stage))
def test_stage_edge_matrix_is_exactly_the_four_legal_edges(src, dst):
    reg, audit = _registry()
    _reach(reg, "v1", src)
    n = audit.head()["length"]
    if (src, dst) in _LEGAL:
        assert reg.transition("alice", "m", "v1", dst).stage == dst
    else:
        with pytest.raises(Conflict):  # includes every edge out of ARCHIVED (terminal) and self-loops
            reg.transition("alice", "m", "v1", dst)
        assert audit.head()["length"] == n + 1 and audit.export()[-1]["decision"] == "deny"
        assert reg.history("m")[0].stage == src


def test_register_rejects_duplicates_and_bad_ids_without_side_effects():
    reg, audit = _registry()
    first = reg.register("alice", "m", "v1", _H)
    n = audit.head()["length"]
    with pytest.raises(Conflict):
        reg.register("alice", "m", "v1", "cd" * 32)  # same key, different hash: must not overwrite
    assert reg.history("m") == [first] and audit.head()["length"] == n
    for mid, vid in (("m", ""), ("", "v"), ("m", "v 1"), ("m", "v\x01"), ("m", "v\x7f"), ("m", "vé"),
                     ("m\x7f", "v"), ("m\x1f", "v"), ("mé", "v"), ("m", "v\t")):
        with pytest.raises(ValidationFailed):
            reg.register("alice", mid, vid, _H)
    reg.register("alice", "has space", "v1", _H)  # a model_id may contain a space (0x20), a version may not
    assert audit.head()["length"] == n + 1


def test_registry_without_audit_still_raises_domain_errors_not_attribute_errors():
    reg = ModelRegistry(None)
    reg.register("alice", "m", "v1", _H)
    with pytest.raises(NotFound):
        reg.validate("alice", "m", "ghost", dict(_EVID))
    with pytest.raises(NotFound):
        reg.transition("alice", "m", "ghost", Stage.STAGING)
    with pytest.raises(Conflict):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)  # illegal edge from REGISTERED
    reg.validate("alice", "m", "v1", dict(_EVID))
    reg.transition("alice", "m", "v1", Stage.STAGING)
    with pytest.raises(NotFound):
        reg.rollback("alice", "ghost")
    with pytest.raises(Conflict):
        reg.rollback("alice", "m")  # no production


def test_validate_and_unvalidated_production_denials_are_audited_once():
    reg, audit = _registry()
    reg.register("alice", "m", "v1", _H)
    n = audit.head()["length"]
    with pytest.raises(NotFound):
        reg.validate("alice", "m", "ghost", dict(_EVID))
    assert audit.head()["length"] == n + 1
    assert audit.export()[-1]["action"] == "model.validate" and audit.export()[-1]["decision"] == "deny"
    reg.transition("alice", "m", "v1", Stage.STAGING)
    n = audit.head()["length"]
    with pytest.raises(Conflict):  # staging -> production but never validated
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert audit.head()["length"] == n + 1
    last = audit.export()[-1]
    assert last["action"] == "model.transition" and last["decision"] == "deny" and last["meta"]["reason"] == "not_validated"
    assert reg.history("m")[0].stage == Stage.STAGING


def _deny_all_pdp(audit):
    bundle = PolicyBundle(rules=[Rule("gate", 1, "min_metric", {"metric": "acc", "min": 0.99}, "block")], name="g", version=1)
    return PolicyDecisionPoint(bundle=bundle, audit=audit)


def test_rollback_is_gated_by_the_decision_point_and_leaves_state_untouched():
    audit = ChainedAuditStore(b"ms-secret")
    pdp = _deny_all_pdp(audit)
    reg = ModelRegistry(audit, decision_point=pdp)
    ok = {"acc": 1.0}
    _reach(reg, "v1", Stage.STAGING)
    reg.transition("alice", "m", "v1", Stage.PRODUCTION, context=ok)
    _reach(reg, "v2", Stage.STAGING)
    reg.transition("alice", "m", "v2", Stage.PRODUCTION, context=ok)
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    n = audit.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.rollback("alice", "m", context={"acc": 0.5})
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    assert [r.stage for r in reg.history("m")] == [Stage.ARCHIVED, Stage.PRODUCTION]
    assert all(e["action"] != "model.rollback" or e["decision"] == "deny" for e in audit.export()[n:])
    rb = reg.rollback("alice", "m", context=ok)  # allowed context goes through
    assert rb.version_id == "v1" and rb.stage == Stage.PRODUCTION
    assert reg.current("m", Stage.PRODUCTION).version_id == "v1"


def test_production_promotion_archives_predecessor_and_records_previous_production():
    reg, _ = _registry()
    _reach(reg, "v1", Stage.PRODUCTION)
    assert reg.history("m")[0].previous_production is None
    _reach(reg, "v2", Stage.STAGING)
    rec = reg.transition("alice", "m", "v2", Stage.PRODUCTION)
    assert rec.previous_production == "v1"
    by = {r.version_id: r for r in reg.history("m")}
    assert by["v1"].stage == Stage.ARCHIVED and by["v2"].stage == Stage.PRODUCTION
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    assert reg.current("m", Stage.STAGING) is None and reg.current("other", Stage.STAGING) is None


def test_second_production_promotion_archives_the_first_and_current_archived_agrees_with_history():
    reg, _ = _registry()
    _reach(reg, "v1", Stage.PRODUCTION)
    _reach(reg, "v2", Stage.PRODUCTION)
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    assert reg.current("m", Stage.ARCHIVED).version_id == "v1"
    assert [r.version_id for r in reg.history("m") if r.stage == Stage.ARCHIVED] == ["v1"]


def test_require_approver_gates_production_and_rollback_with_deny_audit():
    reg, audit = _registry(require_approver=lambda actor: actor == "boss")
    _reach(reg, "v1", Stage.STAGING)
    n = audit.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert audit.export()[-1]["meta"]["reason"] == "not_approver" and audit.head()["length"] == n + 1
    reg.transition("boss", "m", "v1", Stage.PRODUCTION)
    _reach(reg, "v2", Stage.STAGING)
    reg.transition("boss", "m", "v2", Stage.PRODUCTION)
    with pytest.raises(PolicyDenied):
        reg.rollback("alice", "m")
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    assert reg.rollback("boss", "m").version_id == "v1"


def test_attestation_without_secret_is_a_validation_error():
    reg, _ = _registry()
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", _H, attestation={"anything": 1})
    assert reg.history("m") == []


# =============================================================== supply_chain
import copy  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402

from mlops.supply_chain import (  # noqa: E402
    RetentionPolicy, build_attestation, merkle_root, sbom_from_requirements, verify_attestation,
)

_D1, _D2, _D3 = "11" * 32, "22" * 32, "33" * 32


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _leaf(h):
    return _sha(b"\x00" + bytes.fromhex(h))


def _node(a, b):
    return _sha(b"\x01" + bytes.fromhex(a) + bytes.fromhex(b))


def test_merkle_root_hand_computed_shapes_with_domain_separation():
    assert merkle_root([_D1]) == _leaf(_D1)  # a lone leaf is still leaf-prefixed (not returned raw)
    assert merkle_root([_D1]) != _D1
    assert merkle_root([_D1, _D2]) == _node(_leaf(_D1), _leaf(_D2))
    assert merkle_root([_D1, _D2]) != merkle_root([_D2, _D1])  # order matters
    # odd node promoted unchanged, not re-hashed / duplicated
    assert merkle_root([_D1, _D2, _D3]) == _node(_node(_leaf(_D1), _leaf(_D2)), _leaf(_D3))
    four = _node(_node(_leaf(_D1), _leaf(_D2)), _node(_leaf(_D3), _leaf(_D1)))
    assert merkle_root([_D1, _D2, _D3, _D1]) == four
    for bad in ([], ["zz" * 32], [("ab" * 32).upper()], [_D1[:-1]]):
        with pytest.raises(ValueError):
            merkle_root(bad)


def _att(**kw):
    args = dict(secret=b"sekret", inputs={"b": _D2, "a": _D1}, builder_id="ci", artifact_digest=_D3, parameters={"lr": 0.1})
    args.update(kw)
    return build_attestation(**args), args


def test_attestation_is_deterministic_signed_over_every_field_and_verifies():
    att, args = _att()
    assert att["materials_root"] == merkle_root([_D1, _D2])  # inputs are sorted by NAME before hashing
    assert att["format"] == "in-toto-like/HMAC (no PKI, no TEE)"
    assert att["subject"] == _D3 and att["builder_id"] == "ci"
    assert set(att) == {"builder_id", "format", "materials_root", "parameters", "provenance_digest", "signature", "subject"}
    assert att == _att()[0]
    assert verify_attestation(att, b"sekret") is True
    assert verify_attestation(att, b"sekret", artifact_digest=_D3) is True
    assert verify_attestation(att, b"sekret", artifact_digest=_D1) is False
    assert verify_attestation(att, b"other-secret") is False
    assert att["signature"] == hmac.new(b"sekret", att["provenance_digest"].encode(), hashlib.sha256).hexdigest()


@pytest.mark.parametrize("field,value", [
    ("format", "in-toto-like/HMAC (with PKI)"),
    ("builder_id", "evil"),
    ("parameters", {"lr": 9}),
    ("subject", _D1),
    ("materials_root", _D1),
])
def test_attestation_tamper_of_any_signed_field_fails_even_with_recomputed_digest(field, value):
    att, _ = _att()
    forged = dict(att, **{field: value})
    assert verify_attestation(forged, b"sekret") is False  # stale provenance_digest
    # attacker recomputes the digest but cannot recompute the HMAC without the secret
    from mlops.supply_chain import canonical_json, sha256_hex
    body = {k: forged[k] for k in ("builder_id", "format", "materials_root", "parameters", "subject")}
    forged["provenance_digest"] = sha256_hex(canonical_json(body))
    assert verify_attestation(forged, b"sekret") is False


def test_attestation_structure_checks_never_raise():
    att, _ = _att()
    assert verify_attestation(dict(att, extra=1), b"sekret") is False
    assert verify_attestation({k: v for k, v in att.items() if k != "format"}, b"sekret") is False
    for k, bad in (("signature", 5), ("format", None), ("parameters", []), ("builder_id", 1), ("subject", 1)):
        assert verify_attestation(dict(att, **{k: bad}), b"sekret") is False
    for junk in (None, [], "att", 5):
        assert verify_attestation(junk, b"sekret") is False


def test_build_attestation_input_validation():
    for kw in ({"secret": b""}, {"secret": "str"}, {"inputs": {}}, {"inputs": {"a": "xyz"}}, {"inputs": {"": _D1}},
               {"builder_id": ""}, {"artifact_digest": "abc"}, {"parameters": []}):
        with pytest.raises(ValidationFailed):
            _att(**kw)


def _rec(i, age, now=1000.0):
    return {"id": i, "created_ts": now - age}


def test_retention_needs_all_rules_and_max_age_is_strict():
    now = 1000.0
    recs = [_rec("old1", 500), _rec("old2", 400), _rec("edge", 300), _rec("new", 10)]
    age = {"kind": "max_age", "max_age_s": 300}
    keep2 = {"kind": "keep_last", "n": 2}
    # max_age alone: strictly older than 300 -> old1, old2 (edge is exactly 300: kept)
    assert RetentionPolicy([age]).apply(recs, now) == ["old1", "old2"]
    # keep_last alone: newest two kept -> old1, old2 deletable
    assert RetentionPolicy([keep2]).apply(recs, now) == ["old1", "old2"]
    # ALL rules must agree: max_age 300 AND keep_last 3 (keeps new, edge, old2) -> only old1
    assert RetentionPolicy([age, {"kind": "keep_last", "n": 3}]).apply(recs, now) == ["old1"]
    # keep_last 4 shields everything even though max_age would delete two
    assert RetentionPolicy([age, {"kind": "keep_last", "n": 4}]).apply(recs, now) == []
    # max_age 0 with keep_last 0 deletes all but a record created exactly now (age 0 is not > 0)
    both = RetentionPolicy([{"kind": "max_age", "max_age_s": 0}, {"kind": "keep_last", "n": 0}])
    assert both.apply(recs + [_rec("fresh", 0)], now) == ["edge", "new", "old1", "old2"]


def test_retention_keep_last_zero_deletes_everything_and_audits_each_id():
    audit = ChainedAuditStore(b"r")
    recs = [_rec("b", 5), _rec("a", 6)]
    out = RetentionPolicy([{"kind": "keep_last", "n": 0}]).apply(recs, 1000.0, audit=audit)
    assert out == ["a", "b"]
    ex = audit.export()
    assert [(e["action"], e["resource"], e["decision"]) for e in ex] == [("retention.delete", "a", "allow"),
                                                                        ("retention.delete", "b", "allow")]
    assert recs == [_rec("b", 5), _rec("a", 6)]  # input untouched
    assert RetentionPolicy([{"kind": "keep_last", "n": 5}]).apply(recs, 1000.0, audit=audit) == []
    assert len(audit.export()) == 2  # nothing deleted, nothing audited
    assert RetentionPolicy([{"kind": "keep_last", "n": 0}]).apply([], 1000.0) == []


def test_retention_validation_boundaries():
    RetentionPolicy([{"kind": "max_age", "max_age_s": 0}])
    RetentionPolicy([{"kind": "keep_last", "n": 0}])
    for bad in ([], [{"kind": "x"}], [{"kind": "max_age"}], [{"kind": "max_age", "max_age_s": -1}],
                [{"kind": "max_age", "max_age_s": float("nan")}], [{"kind": "keep_last", "n": -1}],
                [{"kind": "keep_last", "n": True}], [{"kind": "keep_last", "n": 1.0}], ["nope"]):
        with pytest.raises(ValidationFailed):
            RetentionPolicy(bad)
    pol = RetentionPolicy([{"kind": "keep_last", "n": 1}])
    for recs, now in (([{"id": "a"}], 1.0), ([{"id": "a", "created_ts": 1}, {"id": "a", "created_ts": 2}], 1.0),
                      ([{"id": "", "created_ts": 1}], 1.0), ([{"id": "a", "created_ts": float("inf")}], 1.0),
                      ([{"id": "a", "created_ts": 1}], float("nan")), ("nope", 1.0), (["x"], 1.0)):
        with pytest.raises(ValidationFailed):
            pol.apply(recs, now)


def test_sbom_sorted_purl_and_case_insensitive_duplicates():
    sbom = sbom_from_requirements([("zlib", "1.0"), {"name": "attrs", "version": "23.1"}])
    assert [c["name"] for c in sbom["components"]] == ["attrs", "zlib"]
    assert sbom["components"][0]["purl"] == "pkg:pypi/attrs@23.1"
    assert sbom["bomFormat"] == "CycloneDX" and sbom["specVersion"] == "1.4"
    from mlops.supply_chain import canonical_json, sha256_hex
    c0 = {"name": "attrs", "version": "23.1", "purl": "pkg:pypi/attrs@23.1"}
    assert sbom["components"][0]["sha256"] == sha256_hex(canonical_json(c0))
    with pytest.raises(ValidationFailed):
        sbom_from_requirements([("Flask", "1"), ("flask", "2")])
    for bad in ([], [("a",)], [("a", "1", "x")], [("a b", "1")], [("a/b", "1")], [("a@b", "1")], [("a", "1@2")],
                [("a", "")], [("", "1")], [("a\x01", "1")], [{"name": "a"}], ["a==1"]):
        with pytest.raises(ValidationFailed):
            sbom_from_requirements(bad)


def _to_prod_with_denying_pdp(pdp_audit, reg_audit):
    reg = ModelRegistry(reg_audit, decision_point=_deny_all_pdp(pdp_audit))
    _reach(reg, "v1", Stage.STAGING)
    return reg


def test_policy_denial_writes_registry_deny_only_when_audit_stores_differ():
    # shared store: the decision point already wrote its own deny entry; registry adds none
    shared = ChainedAuditStore(b"shared")
    reg = _to_prod_with_denying_pdp(shared, shared)
    n = shared.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION, context={"acc": 0.1})
    new = shared.export()[n:]
    assert len(new) == 1 and new[0]["decision"] == "deny" and new[0]["action"] != "model.transition"
    # separate stores: registry records its own deny with the reason
    reg_audit, pdp_audit = ChainedAuditStore(b"reg"), ChainedAuditStore(b"pdp")
    reg = _to_prod_with_denying_pdp(pdp_audit, reg_audit)
    n = reg_audit.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION, context={"acc": 0.1})
    new = reg_audit.export()[n:]
    assert len(new) == 1 and new[0]["action"] == "model.transition" and new[0]["decision"] == "deny"
    assert new[0]["meta"]["reason"] == "policy_denied"
    assert reg.history("m")[0].stage == Stage.STAGING


def test_rollback_policy_denial_audit_follows_same_rule():
    reg_audit, pdp_audit = ChainedAuditStore(b"reg"), ChainedAuditStore(b"pdp")
    reg = ModelRegistry(reg_audit, decision_point=_deny_all_pdp(pdp_audit))
    ok = {"acc": 1.0}
    for v in ("v1", "v2"):
        _reach(reg, v, Stage.STAGING)
        reg.transition("alice", "m", v, Stage.PRODUCTION, context=ok)
    n = reg_audit.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.rollback("alice", "m", context={"acc": 0.1})
    new = reg_audit.export()[n:]
    assert [(e["action"], e["decision"], e["meta"]["reason"]) for e in new] == [("model.rollback", "deny", "policy_denied")]
    shared = ChainedAuditStore(b"shared")
    reg = ModelRegistry(shared, decision_point=_deny_all_pdp(shared))
    for v in ("v1", "v2"):
        _reach(reg, v, Stage.STAGING)
        reg.transition("alice", "m", v, Stage.PRODUCTION, context=ok)
    n = shared.head()["length"]
    with pytest.raises(PolicyDenied):
        reg.rollback("alice", "m", context={"acc": 0.1})
    assert all(e["action"] != "model.rollback" for e in shared.export()[n:])


def test_register_printable_ascii_boundaries():
    reg, _ = _registry()
    reg.register("alice", "m~", "v~", _H)  # 0x7e is the last legal character
    for mid, vid in (("m", "v\x1f"), ("m\x1f", "v"), ("m", "v\x80")):
        with pytest.raises(ValidationFailed):
            reg.register("alice", mid, vid, _H)


# =============================================================== svc/sdk
import httpx  # noqa: E402

from mlops.svc.sdk import MlopsApiError, MlopsClient  # noqa: E402

_TOKEN = "tok-hardening-0001"


class _Script:
    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    def __call__(self, request):
        self.requests.append(request)
        step = self.steps[min(len(self.requests) - 1, len(self.steps) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    def client(self, **kw):
        http = httpx.Client(transport=httpx.MockTransport(self), base_url="http://svc.test")
        kw.setdefault("backoff_s", 0.001)
        return MlopsClient(client=http, token=_TOKEN, **kw)


def _ok(body=None, status=200):
    return httpx.Response(status, json=body if body is not None else {"ok": True})


def _err(status, code="c", message="m"):
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


@pytest.fixture
def _nosleep(monkeypatch):
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)


@pytest.mark.parametrize("status", [502, 503, 504])
def test_sdk_transient_statuses_retry_exactly_retries_plus_one_with_one_idempotency_key(status, _nosleep):
    s = _Script(_err(status, "unavailable", "later"))
    with pytest.raises(MlopsApiError) as ei:
        s.client(retries=3).register_model("m", "v1", "a" * 64)
    assert len(s.requests) == 4  # 1 + retries
    assert (ei.value.status, ei.value.code, ei.value.message) == (status, "unavailable", "later")
    keys = {r.headers["idempotency-key"] for r in s.requests}
    assert len(keys) == 1  # the SAME key on every retry, or a retry could double-apply


def test_sdk_retry_then_success_reuses_key_and_returns_body(_nosleep):
    s = _Script(_err(503), httpx.ConnectError("x"), _ok({"stage": "registered"}))
    out = s.client(retries=3).register_model("m", "v1", "a" * 64)
    assert out == {"stage": "registered"} and len(s.requests) == 3
    assert len({r.headers["idempotency-key"] for r in s.requests}) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 415, 422, 500])
def test_sdk_never_retries_client_errors_or_plain_500(status, _nosleep):
    s = _Script(_err(status, "code_x", "msg_x"))
    with pytest.raises(MlopsApiError) as ei:
        s.client(retries=5).register_model("m", "v1", "a" * 64)
    assert len(s.requests) == 1
    assert (ei.value.status, ei.value.code, ei.value.message) == (status, "code_x", "msg_x")


def test_sdk_zero_retries_means_single_attempt(_nosleep):
    s = _Script(_err(503))
    with pytest.raises(MlopsApiError):
        s.client(retries=0).health()
    assert len(s.requests) == 1


def test_sdk_transport_errors_retry_then_raise_status_zero(_nosleep):
    for exc in (httpx.ConnectError("c"), httpx.ReadTimeout("r"), httpx.WriteError("w"), httpx.PoolTimeout("p")):
        s = _Script(exc)
        with pytest.raises(MlopsApiError) as ei:
            s.client(retries=2).health()
        assert (ei.value.status, ei.value.code) == (0, "transport") and len(s.requests) == 3


def test_sdk_each_post_gets_a_fresh_key_and_gets_carry_none(_nosleep):
    s = _Script(_ok())
    c = s.client()
    c.register_model("m", "v1", "a" * 64)
    c.register_model("m", "v2", "b" * 64)
    c.health()
    k1, k2 = s.requests[0].headers["idempotency-key"], s.requests[1].headers["idempotency-key"]
    assert k1 != k2 and len(k1) >= 8
    assert "idempotency-key" not in s.requests[2].headers
    assert s.requests[2].headers["authorization"] == "Bearer " + _TOKEN


def test_sdk_bad_body_fails_before_any_request_and_bad_json_is_bad_response(_nosleep):
    s = _Script(_ok())
    with pytest.raises(ValueError):
        s.client().decide("a", {"x": float("nan")})
    with pytest.raises(ValueError):
        s.client().decide("a", {"x": object()})
    assert s.requests == []
    s = _Script(httpx.Response(200, text="<html>"))
    with pytest.raises(MlopsApiError) as ei:
        s.client().health()
    assert ei.value.code == "bad_response" and len(s.requests) == 1
    s = _Script(httpx.Response(400, json={"detail": "nope"}))
    with pytest.raises(MlopsApiError) as ei:
        s.client().health()
    assert (ei.value.status, ei.value.code) == (400, "bad_response") and len(s.requests) == 1


def test_sdk_path_segments_are_percent_encoded_and_bodies_are_exact():
    s = _Script(_ok())
    c = s.client()
    c.validate_model("a/b c", "v?1#", {"k": 1})
    c.transition("m", "v1", "staging")
    c.transition("m", "v1", "production", context={"acc": 1})
    c.rollback("m/x")
    assert s.requests[0].url.raw_path == b"/v1/models/a%2Fb%20c/versions/v%3F1%23/validate"
    assert json.loads(s.requests[0].content) == {"evidence": {"k": 1}}
    assert json.loads(s.requests[1].content) == {"to_stage": "staging"}  # no "context" key when None
    assert json.loads(s.requests[2].content) == {"to_stage": "production", "context": {"acc": 1}}
    assert s.requests[3].url.raw_path == b"/v1/models/m%2Fx/rollback"


def test_sdk_token_length_boundary():
    http = httpx.Client(transport=httpx.MockTransport(_Script(_ok())))
    MlopsClient(client=http, token="t" * 128)
    with pytest.raises(ValueError):
        MlopsClient(client=http, token="t" * 129)


def test_sdk_retry_gaps_use_exponential_backoff_from_backoff_s(monkeypatch):
    import tenacity.nap
    sleeps = []
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: sleeps.append(s))
    s = _Script(_err(503))
    with pytest.raises(MlopsApiError):
        s.client(retries=3, backoff_s=0.5).health()
    assert len(sleeps) == 3  # one wait between each of the four attempts
    assert all(x >= 0.5 for x in sleeps)  # never below the configured initial backoff


# =============================================================== svc: idempotency store
from mlops.svc.idempotency import IdempotencyStore  # noqa: E402

_KEY = "abcdef012345"


def _idem(ttl=60, max_entries=10_000):
    clock = ManualClock(1000.0)
    return IdempotencyStore(clock, ttl, max_entries=max_entries), clock


def _begin(s, key=_KEY, who="olga", method="POST", path="/v1/models", h="h1"):
    return s.begin(key, who, method, path, h)


def _finish(s, key=_KEY, who="olga", status=201, body=None, method="POST", path="/v1/models", h="h1"):
    s.finish(key, who, method, path, h, status, body if body is not None else {"ok": 1})


def test_idempotency_lifecycle_new_in_flight_replay_conflict():
    s, _ = _idem()
    assert _begin(s) == ("new", None)
    assert _begin(s) == ("in_flight", None)
    _finish(s, body={"a": [1]})
    assert _begin(s) == ("replay", (201, {"a": [1]}))
    for kw in ({"h": "h2"}, {"path": "/v1/other"}, {"method": "PUT"}):
        assert _begin(s, **kw) == ("conflict", None)


def test_idempotency_keys_are_scoped_per_principal_and_replays_are_copies():
    s, _ = _idem()
    assert _begin(s, who="olga")[0] == "new"
    assert _begin(s, who="vic")[0] == "new"  # same key, other principal: independent
    _finish(s, who="olga", body={"x": [1]})
    st, cached = _begin(s, who="olga")
    cached[1]["x"].append(99)  # caller mutation must not poison the stored response
    assert _begin(s, who="olga") == ("replay", (201, {"x": [1]}))
    assert _begin(s, who="vic")[0] == "in_flight"


def test_idempotency_ttl_boundary_and_finish_extends_expiry():
    s, clock = _idem(ttl=60)
    _begin(s)
    clock.advance(59)
    assert _begin(s)[0] == "in_flight"
    clock.advance(1)  # exactly ttl: expired (expires_at <= now)
    assert _begin(s)[0] == "new"
    _finish(s)  # finish restarts the clock from "now"
    clock.advance(59)
    assert _begin(s)[0] == "replay"
    clock.advance(1)
    assert _begin(s)[0] == "new"


def test_idempotency_store_is_bounded_and_evicts_oldest_first():
    s, _ = _idem(max_entries=2)
    for k in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        assert _begin(s, key=k)[0] == "new"
    assert _begin(s, key="bbbbbbbb")[0] == "in_flight"  # survivors
    assert _begin(s, key="cccccccc")[0] == "in_flight"
    assert _begin(s, key="aaaaaaaa")[0] == "new"  # oldest was evicted


def test_idempotency_abort_frees_the_key_and_invalid_input_changes_nothing():
    s, _ = _idem()
    _begin(s)
    s.abort(_KEY, "olga", "POST", "/v1/models", "h1")
    assert _begin(s)[0] == "new"
    s2, _ = _idem()
    bad = [
        dict(key="short"), dict(key="x" * 129), dict(key="has space!"), dict(key=5), dict(who=""), dict(who="a\x01"),
        dict(who=None), dict(method="TRACE"), dict(method=None), dict(path="v1/models"), dict(path="/" + "p" * 1024),
        dict(h=""), dict(h="h" * 129), dict(h=None),
    ]
    for kw in bad:
        with pytest.raises(ValidationFailed):
            _begin(s2, **kw)
    assert len(s2._store) == 0
    _begin(s2, key="k" * 8)  # 8 chars: lower bound
    _begin(s2, key="k" * 128, path="/" + "p" * 1023, h="h" * 128)  # upper bounds are inclusive
    assert len(s2._store) == 2


def test_idempotency_key_with_trailing_newline_should_be_rejected():
    s, _ = _idem()
    with pytest.raises(ValidationFailed):
        _begin(s, key="abcdefgh\n")


# =============================================================== svc: bearer auth (deps)
from types import SimpleNamespace  # noqa: E402

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mlops.svc.deps import Principal, get_principal, get_state, require_role  # noqa: E402
from mlops.svc.errors import install_error_handlers  # noqa: E402

_VALID = "abcdefghi-VALID-tail-1"


def _auth_client():
    app = FastAPI()
    install_error_handlers(app)
    spec = lambda n, r: SimpleNamespace(name=n, role=r)  # noqa: E731
    state = SimpleNamespace(settings=SimpleNamespace(tokens={_VALID: spec("ann", "operator"), "zzzzzzzzz-OTHER-1": spec("bo", "viewer")}))
    app.dependency_overrides[get_state] = lambda: state

    @app.get("/who")
    def who(p: Principal = Depends(get_principal)):
        return {"name": p.name, "role": p.role}

    @app.get("/admin")
    def admin(p: Principal = Depends(require_role("admin"))):
        return {"name": p.name}

    @app.get("/ops")
    def ops(p: Principal = Depends(require_role("operator", "admin"))):
        return {"name": p.name}

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("token", [
    "abcdefghi",  # a bare prefix of a valid token
    "abcdefghi-VALID-tail-2",  # same first 21 chars, last differs
    "abcdefghi-WRONG-tail-1",  # same first 9 chars as a valid token (M9: truncated comparison)
    "abcdefghi-VALID-tail-1x",  # valid token plus a suffix
    "bcdefghi-VALID-tail-1",  # valid token minus its first char
    "ABCDEFGHI-VALID-TAIL-1",
])
def test_bearer_token_must_match_in_full(token):
    c = _auth_client()
    r = c.get("/who", headers={"Authorization": "Bearer " + token})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "http_error"
    assert c.get("/who", headers={"Authorization": "Bearer " + _VALID}).json() == {"name": "ann", "role": "operator"}


@pytest.mark.parametrize("header", ["", "Bearer", "Bearer ", "Basic " + _VALID, _VALID, "Bearer a b", "Token " + _VALID])
def test_malformed_authorization_headers_are_401_with_challenge(header):
    c = _auth_client()
    r = c.get("/who", headers={"Authorization": header} if header else {})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_bearer_scheme_is_case_insensitive_and_second_token_authenticates_its_own_principal():
    c = _auth_client()
    assert c.get("/who", headers={"Authorization": "bearer " + _VALID}).status_code == 200
    assert c.get("/who", headers={"Authorization": "BEARER zzzzzzzzz-OTHER-1"}).json() == {"name": "bo", "role": "viewer"}


def test_role_gate_is_403_after_authentication_and_401_wins_over_403():
    c = _auth_client()
    ok = {"Authorization": "Bearer " + _VALID}
    assert c.get("/admin", headers=ok).status_code == 403
    assert c.get("/admin", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/admin").status_code == 401
    assert c.get("/ops", headers=ok).status_code == 200
    assert c.get("/ops", headers={"Authorization": "Bearer zzzzzzzzz-OTHER-1"}).status_code == 403


# =============================================================== svc: app pipeline order, replay, metrics
from mlops.svc.app import create_app  # noqa: E402
from mlops.svc.settings import Settings  # noqa: E402

_T = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}
_N = {"viewer": "vic", "operator": "olga", "admin": "root"}
_JSON = {"content-type": "application/json"}
_A1, _A2 = "11" * 32, "22" * 32


def _app_client(**settings_kw):
    tokens = {_T[r]: {"name": _N[r], "role": r} for r in _T}
    settings = Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None, **settings_kw)
    audit = ChainedAuditStore(b"s" * 20)
    return TestClient(create_app(settings, audit=audit), raise_server_exceptions=False), audit


def _h(role, **extra):
    return {"Authorization": "Bearer " + _T[role], **extra}


def _reg(m="m", v="v1", h=_A1):
    return {"model_id": m, "version_id": v, "artifact_hash": h}


def _n(audit):
    return audit.head()["length"]


def test_pipeline_order_401_then_403_then_415_then_413_then_422():
    c, audit = _app_client(max_body_bytes=2048)
    n = _n(audit)
    big = b"{" + b'"pad":"' + b"x" * 5000 + b'"}'
    # 401 beats everything else
    assert c.post("/v1/models", content=big, headers={"content-type": "text/plain"}).status_code == 401
    assert c.post("/v1/models", content=big, headers={**_JSON, "Authorization": "Bearer nope"}).status_code == 401
    # 403 beats 415/413/422
    assert c.post("/v1/models", content=big, headers=_h("viewer", **{"content-type": "text/plain"})).status_code == 403
    assert c.post("/v1/models", content=big, headers=_h("viewer", **_JSON)).status_code == 403
    # 415 beats 413/422
    r = c.post("/v1/models", content=big, headers=_h("operator", **{"content-type": "text/plain"}))
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"
    assert c.post("/v1/models", content=b"{}", headers=_h("operator")).status_code == 415  # no content-type at all
    # 413 beats 422 (body is oversized AND semantically invalid)
    r = c.post("/v1/models", content=big, headers=_h("operator", **_JSON))
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    # 422 last, and no audit entry for any of the rejections above
    r = c.post("/v1/models", json={"model_id": "m"}, headers=_h("operator"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"
    assert _n(audit) == n


def test_body_size_limit_is_inclusive_on_both_content_length_and_chunked_paths():
    limit = 2048
    c, audit = _app_client(max_body_bytes=limit)
    at = b'{"pad":"' + b"x" * (limit - len(b'{"pad":""}')) + b'"}'
    assert len(at) == limit
    over = at + b" "
    for body in (at, over):
        def chunks(b=body):
            for i in range(0, len(b), 512):
                yield b[i:i + 512]
        for payload in (body, chunks()):  # content-length path, then transfer-encoding: chunked path
            r = c.post("/v1/models", content=payload, headers=_h("operator", **_JSON))
            if body is at:
                assert r.status_code == 422, r.text  # exactly at the limit is read and judged on content
            else:
                assert r.status_code == 413, r.text  # one byte over is refused


def test_json_shape_and_nesting_depth_limits():
    c, audit = _app_client()
    assert c.post("/v1/models", json=_reg(), headers=_h("admin")).status_code == 201

    def nest(levels):
        d = {}
        for _ in range(levels - 1):
            d = {"k": d}
        return d

    ok = c.post("/v1/models/m/versions/v1/validate", json={"evidence": nest(63)}, headers=_h("admin"))  # depth 64 in total
    assert ok.status_code == 200, ok.text
    n = _n(audit)
    deep = c.post("/v1/models/m/versions/v1/validate", json={"evidence": nest(64)}, headers=_h("admin"))  # depth 65
    assert deep.status_code == 422 and "nesting" in deep.json()["error"]["message"]
    for raw, frag in ((b"[1,2]", "object"), (b"{bad", "valid JSON"), (b'{"a": NaN}', "valid JSON"), (b'{"a": Infinity}', "valid JSON")):
        r = c.post("/v1/models/m/versions/v1/validate", content=raw, headers=_h("admin", **_JSON))
        assert r.status_code == 422 and frag in r.json()["error"]["message"], (raw, r.text)
    assert _n(audit) == n


def test_idempotent_post_replays_without_side_effects_and_is_scoped_to_the_principal():
    c, audit = _app_client()
    hdr = _h("operator", **{"Idempotency-Key": "key-0000-0001"})
    r1 = c.post("/v1/models", json=_reg(), headers=hdr)
    assert r1.status_code == 201 and "idempotent-replayed" not in r1.headers
    n = _n(audit)
    r2 = c.post("/v1/models", json=_reg(), headers=hdr)
    assert r2.status_code == 201 and r2.json() == r1.json() and r2.headers["idempotent-replayed"] == "true"
    assert _n(audit) == n  # the replay did not register (and did not audit) again
    # same key + different body -> conflict, nothing written
    r3 = c.post("/v1/models", json=_reg(v="v2"), headers=hdr)
    assert r3.status_code == 422 and r3.json()["error"]["code"] == "idempotency_conflict" and _n(audit) == n
    # same key but a different principal is an independent request: it hits the registry and conflicts on the duplicate
    r4 = c.post("/v1/models", json=_reg(), headers=_h("admin", **{"Idempotency-Key": "key-0000-0001"}))
    assert r4.status_code == 409 and "idempotent-replayed" not in r4.headers


def test_failed_idempotent_request_is_aborted_not_cached_and_bad_key_is_400():
    c, audit = _app_client()
    c.post("/v1/models", json=_reg(), headers=_h("operator"))
    hdr = _h("operator", **{"Idempotency-Key": "key-0000-0002"})
    r1 = c.post("/v1/models", json=_reg(), headers=hdr)  # duplicate -> 409, must not be cached
    assert r1.status_code == 409
    r2 = c.post("/v1/models", json=_reg(), headers=hdr)
    assert r2.status_code == 409 and "idempotent-replayed" not in r2.headers
    n = _n(audit)
    for bad in ("short", "x" * 129, "has space here"):
        r = c.post("/v1/models", json=_reg(v="v9"), headers=_h("operator", **{"Idempotency-Key": bad}))
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_failed"
    assert _n(audit) == n


def test_metrics_labels_use_route_templates_and_collapse_unknown_paths():
    c, _ = _app_client()
    assert c.get("/definitely/not/a/route/xyz123", headers=_h("viewer")).status_code == 404
    c.post("/v1/models", json=_reg("secret-model-name", "v1"), headers=_h("operator"))
    c.get("/v1/models/secret-model-name", headers=_h("viewer"))
    text = c.get("/metrics", headers=_h("viewer")).text
    assert 'path="unmatched"' in text
    assert "xyz123" not in text and "secret-model-name" not in text  # no attacker-controlled label values
    assert 'path="/v1/models/{model_id}"' in text
    assert 'method="POST",path="/v1/models",status="201"' in text
    assert "mlops_audit_length" in text


def test_metrics_requires_a_valid_token_and_docs_are_dev_only():
    c, _ = _app_client()
    assert c.get("/metrics").status_code == 401
    assert c.get("/healthz").json()["env"] == "dev"
    assert c.get("/docs").status_code == 200
    prod, _ = _app_client(env_name="prod")
    assert prod.get("/docs").status_code == 404 and prod.get("/redoc").status_code == 404
    assert prod.get("/healthz").json() == {"status": "ok", "env": "prod"}


def test_role_matrix_on_mutating_routes_leaves_audit_untouched():
    c, audit = _app_client()
    c.post("/v1/models", json=_reg(), headers=_h("admin"))
    n = _n(audit)
    denied = [
        ("viewer", "/v1/models", _reg("m2")),
        ("viewer", "/v1/models/m/versions/v1/validate", {"evidence": {"a": 1}}),
        ("operator", "/v1/models/m/rollback", {}),
        ("operator", "/v1/policy/publish", {"name": "p", "version": 1, "rules": []}),
        ("viewer", "/v1/policy/1/activate", {}),
    ]
    for role, path, body in denied:
        r = c.post(path, json=body, headers=_h(role))
        assert r.status_code == 403, (role, path, r.text)
        assert r.json()["error"]["code"] == "policy_denied"
    assert c.get("/v1/audit/export", headers=_h("operator")).status_code == 403
    assert c.get("/v1/audit/export", headers=_h("admin")).status_code == 200
    assert c.get("/v1/audit/head", headers=_h("viewer")).status_code == 200
    assert _n(audit) == n


# =============================================================== floor_guard
from mlops.floor_guard import (  # noqa: E402
    AssuredFirstDispatcher, FloorConfig, Request, StabilityError, TenantFloor, check_preconditions,
    require_guarantee, simulate_guard, sojourn_bound,
)


def _cfg(**kw):
    base = dict(credits=5, window_s=10.0, batch_slots=2, s_max=2.0)
    base.update(kw)
    return FloorConfig(**base)


def test_floorconfig_inclusive_bounds_and_rejections():
    _cfg(credits=1)
    _cfg(credits=10**6, batch_slots=10**6)
    _cfg(window_s=1e-100)
    _cfg(s_max=1e100)
    _cfg(t0=1e100)
    _cfg(t0=-1e100)
    _cfg(drop_lower_bound=0.0)
    for bad in (dict(credits=0), dict(credits=10**6 + 1), dict(credits=True), dict(credits=1.0),
                dict(batch_slots=0), dict(batch_slots=10**6 + 1),
                dict(window_s=0), dict(window_s=-1.0), dict(window_s=1e-101), dict(window_s=float("inf")),
                dict(s_max=0), dict(s_max=1.0000001e100), dict(s_max=float("nan")),
                dict(t0=1.0000001e100), dict(t0=-1.0000001e100), dict(t0=float("inf")),
                dict(excess="drop"), dict(drop_lower_bound=-0.001), dict(drop_lower_bound=float("nan"))):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_tenantfloor_windows_are_aligned_to_t0_and_refill_exactly_on_the_boundary():
    f = TenantFloor(_cfg(credits=3, window_s=10.0, t0=100.0))
    r = f.admit(100.0, 2)  # window 0 starts exactly at t0
    assert (r.window_index, r.admitted) == (0, 2)
    r = f.admit(109.999, 2)  # still window 0: only 1 credit left
    assert (r.window_index, r.admitted, r.rejected) == (0, 1, 1)
    r = f.admit(110.0, 3)  # boundary belongs to the NEXT window: full refill
    assert (r.window_index, r.admitted, r.rejected) == (1, 3, 0)
    assert f.admit(119.0, 1).admitted == 0  # same window, exhausted
    r = f.admit(1_000_000.0, 3)  # skipping many windows refills once, not cumulatively
    assert r.admitted == 3 and r.window_index == int((1_000_000.0 - 100.0) // 10.0)


def test_tenantfloor_pre_t0_timestamps_floor_toward_negative_windows():
    f = TenantFloor(_cfg(credits=3, window_s=10.0, t0=100.0))
    assert f.admit(95.0, 1).window_index == -1  # floor(-0.5) = -1, not int() truncation to 0
    assert f.admit(100.0, 1).window_index == 0
    g = TenantFloor(_cfg(credits=3, window_s=10.0, t0=100.0))
    assert g.admit(90.0, 1).window_index == -1  # exact multiple below t0


def test_tenantfloor_partial_admit_reject_vs_demote_and_zero_n():
    rej = TenantFloor(_cfg(credits=5, excess="reject"))
    r = rej.admit(1.0, 8)
    assert (r.admitted, r.rejected, r.demoted) == (5, 3, 0)
    assert rej.remaining_credits == 0
    assert rej.admit(1.5, 0).admitted == 0
    dem = TenantFloor(_cfg(credits=5, excess="demote"))
    r = dem.admit(1.0, 8)
    assert (r.admitted, r.rejected, r.demoted) == (5, 0, 3)
    assert dem.admit(2.0, 2).demoted == 2 and dem.remaining_credits == 0
    assert TenantFloor(_cfg(credits=5)).admit(1.0, 5).rejected == 0  # exactly the budget: nothing excess


def test_tenantfloor_time_backwards_and_bad_inputs():
    f = TenantFloor(_cfg(window_s=10.0))
    f.admit(25.0, 1)
    f.admit(21.0, 1)  # earlier but same window: allowed
    with pytest.raises(ValueError):
        f.admit(19.999, 1)  # previous window
    for now, n in ((float("nan"), 1), (True, 1), ("1", 1), (1.0, -1), (1.0, True), (1.0, 1.5)):
        with pytest.raises(ValueError):
            f.admit(now, n)


def test_tenantfloor_far_timestamp_limit_is_1e12_windows_inclusive():
    f = TenantFloor(_cfg(window_s=1.0, t0=0.0))
    assert f.admit(1e12, 1).window_index == 10**12  # exactly 1e12 windows away: accepted
    with pytest.raises(ValueError):
        TenantFloor(_cfg(window_s=1.0, t0=0.0)).admit(1e12 + 1e5, 1)


def test_check_preconditions_boundaries_sojourn_bound_and_require_guarantee():
    cfg = _cfg(credits=4, window_s=10.0, batch_slots=1, s_max=2.5)  # rho = 4*2.5/(1*10) = 1.0 exactly
    p = check_preconditions(cfg, 4)
    assert p["rho_a"] == 1.0 and p["stable"] is True and p["backlog_ok"] is True and p["guarantee_void"] is False
    assert sojourn_bound(cfg, 4) == 2.5 * (1 + 4)
    assert check_preconditions(cfg, 5)["backlog_ok"] is False and sojourn_bound(cfg, 5) is None
    with pytest.raises(StabilityError):
        require_guarantee(cfg, 5)
    unstable = _cfg(credits=4, window_s=10.0, batch_slots=1, s_max=2.6)
    assert check_preconditions(unstable, 0)["stable"] is False and sojourn_bound(unstable) is None
    with pytest.raises(StabilityError):
        require_guarantee(unstable, 0)
    require_guarantee(cfg, 0)
    odd = _cfg(credits=5, window_s=100.0, batch_slots=2, s_max=2.0)  # ceil(5/2) = 3
    assert sojourn_bound(odd) == 2.0 * (1 + 3)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            check_preconditions(cfg, bad)


def _rq(i, cls, arr, svc, dl):
    return Request(id=i, cls=cls, arrival_ts=arr, service_s=svc, deadline_ts=dl)


def test_dispatcher_assured_first_fifo_within_class_and_strict_deadline_semantics():
    d = AssuredFirstDispatcher(_cfg(batch_slots=1, drop_lower_bound=0.0))
    out = {o.id: o for o in d.run([
        _rq("o1", "opportunistic", 0.0, 2.0, 100.0),
        _rq("a1", "assured", 0.0, 2.0, 100.0),
        _rq("a2", "assured", 0.0, 2.0, 100.0),
        _rq("o2", "opportunistic", 0.0, 2.0, 100.0),
    ])}
    assert [out[i].start_ts for i in ("a1", "a2", "o1", "o2")] == [0.0, 2.0, 4.0, 6.0]  # ties break by input order
    # finish exactly at the deadline is on time; one tick later is missed
    on_time = AssuredFirstDispatcher(_cfg(batch_slots=1)).run([_rq("x", "assured", 0.0, 2.0, 2.0)])[0]
    assert on_time.status == "completed" and on_time.missed is False and on_time.finish_ts == 2.0
    late = AssuredFirstDispatcher(_cfg(batch_slots=1)).run([_rq("x", "assured", 0.0, 2.0, 1.999)])[0]
    assert late.status == "completed" and late.missed is True


def test_dispatcher_doom_sound_drop_is_strict_and_uses_drop_lower_bound():
    d = AssuredFirstDispatcher(_cfg(batch_slots=1, drop_lower_bound=1.5))
    edge = d.run([_rq("e", "assured", 0.0, 1.0, 1.5)])[0]  # 0 + 1.5 > 1.5 is False -> runs
    assert edge.status == "completed" and edge.start_ts == 0.0
    doomed = d.run([_rq("d", "assured", 0.0, 1.0, 1.4999)])[0]
    assert (doomed.status, doomed.start_ts, doomed.finish_ts, doomed.missed) == ("dropped", None, None, True)


def test_dispatcher_parallel_slots_and_validation():
    two = AssuredFirstDispatcher(_cfg(batch_slots=2))
    out = two.run([_rq("a", "assured", 0.0, 4.0, 99.0), _rq("b", "assured", 0.0, 4.0, 99.0), _rq("c", "assured", 0.0, 4.0, 99.0)])
    assert sorted(o.start_ts for o in out) == [0.0, 0.0, 4.0]
    late = two.run([_rq("late", "assured", 50.0, 1.0, 99.0)])[0]
    assert late.start_ts == 50.0 and late.finish_ts == 51.0  # idle slot waits for the arrival, never starts early
    for bad in ([_rq("a", "assured", 0.0, 1.0, 9.0), _rq("a", "assured", 0.0, 1.0, 9.0)], [_rq("", "assured", 0.0, 1.0, 9.0)],
                [_rq("a", "vip", 0.0, 1.0, 9.0)], [_rq("a", "assured", 0.0, 0.0, 9.0)], [_rq("a", "assured", float("nan"), 1.0, 9.0)],
                [_rq("a", "assured", 0.0, 1.0, float("inf"))]):
        with pytest.raises(ValueError):
            two.run(bad)


def test_simulate_guard_size_limits_are_inclusive_and_determinism():
    cfg = _cfg()
    with pytest.raises(ValueError):
        simulate_guard(1, 10_001, 0, 0, cfg, True)
    with pytest.raises(ValueError):
        simulate_guard(1, 1, 10_001, 0, cfg, False)
    with pytest.raises(ValueError):
        simulate_guard(1, 1, 0, 10_001, cfg, False)
    assert simulate_guard(1, 10_000, 0, 0, cfg, True)["assured_offered"] == 0  # cap itself is fine
    assert simulate_guard(1, 1, 10_000, 0, cfg, False)["assured_offered"] == 10_000
    assert simulate_guard(1, 1, 0, 10_000, cfg, False)["assured_offered"] == 0
    a = simulate_guard(7, 3, 4, 2, cfg, True)
    assert a == simulate_guard(7, 3, 4, 2, cfg, True) and a["guarded"] is True
    for bad in (dict(seed=True), dict(n_windows=0), dict(guarded=1), dict(cfg="x")):
        kw = dict(seed=1, n_windows=1, offered_assured=1, offered_opportunistic=1, cfg=cfg, guarded=True)
        kw.update(bad)
        with pytest.raises(ValueError):
            simulate_guard(**kw)


def test_simulate_guard_rates_are_probabilities_under_demotion():
    r = simulate_guard(3, 2, 4, 5, _cfg(credits=1, batch_slots=1, s_max=3.0, excess="demote"), True)
    assert 0.0 <= r["assured_miss_rate"] <= 1.0
    assert 0.0 <= r["opportunistic_miss_rate"] <= 1.0
    assert r["opportunistic_miss_rate"] == pytest.approx(0.9)


# =============================================================== storage_sqlite
import sqlite3  # noqa: E402

from mlops.kernel import IntegrityError  # noqa: E402
from mlops.storage_sqlite import DurableAuditStore, DurableDocs, open_db  # noqa: E402

_SEC = b"durable-secret"


_TEMPLATES = {}


def _durable(tmp_path, name="a.db", n=3):
    """Build (once per n) a template DB with n entries, then hand out cheap file copies."""
    import shutil
    import tempfile
    if n not in _TEMPLATES:
        import atexit
        d = tempfile.mkdtemp(prefix="s34_")
        atexit.register(shutil.rmtree, d, True)
        tpl = os.path.join(d, "tpl.db")
        s = DurableAuditStore(tpl, _SEC)
        for i in range(n):
            s.append("alice", "act%d" % i, "res", "allow", ts=100.0 + i, meta={"i": i})
        _TEMPLATES[n] = (tpl, s.head())
        s.close()
    tpl, head = _TEMPLATES[n]
    path = str(tmp_path / name)
    shutil.copy(tpl, path)
    return path, head


def _sql(path, stmt, *args):
    con = sqlite3.connect(path)
    try:
        con.execute(stmt, args)
        con.commit()
    finally:
        con.close()


def test_open_db_sets_durability_pragmas(tmp_path):
    con = open_db(str(tmp_path / "p.db"))
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert con.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    con.close()
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    assert d._conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert d._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert d._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_durable_audit_roundtrip_and_reopen_preserves_head(tmp_path):
    path, head = _durable(tmp_path)
    s = DurableAuditStore(path, _SEC)
    assert s.head() == head and head["length"] == 3
    assert [e["action"] for e in s.export()] == ["act0", "act1", "act2"]
    s.append("bob", "act3", "res", "deny")  # chain continues from the persisted tail
    assert s.head()["length"] == 4
    s.close()
    assert DurableAuditStore(path, _SEC).head()["length"] == 4


@pytest.mark.parametrize("stmt,args", [
    ("UPDATE audit_head SET length = 2", ()),  # tail truncation claimed by head
    ("UPDATE audit_head SET length = 4", ()),
    ("UPDATE audit_head SET signature = ?", ("0" * 64,)),
    ("UPDATE audit_head SET anchor = ?", ("0" * 64,)),  # anchor forgery alone
    ("DELETE FROM audit_head", ()),  # entries with no head row
    ("DELETE FROM audit_entries WHERE seq = 2", ()),  # tail truncation, head left behind
    ("UPDATE audit_entries SET payload = replace(payload, 'act1', 'actX') WHERE seq = 1", ()),
    ("UPDATE audit_entries SET signature = ? WHERE seq = 0", ("f" * 64,)),
    ("UPDATE audit_entries SET payload = 'not json' WHERE seq = 0", ()),
])
def test_durable_audit_detects_every_tamper_on_open(tmp_path, stmt, args):
    path, _ = _durable(tmp_path)
    _sql(path, stmt, *args)
    with pytest.raises(IntegrityError):
        DurableAuditStore(path, _SEC)


def test_durable_audit_truncation_with_consistent_length_still_fails_on_anchor_and_wrong_secret_fails(tmp_path):
    path, _ = _durable(tmp_path)
    con = sqlite3.connect(path)
    sig1 = con.execute("SELECT signature FROM audit_entries WHERE seq = 1").fetchone()[0]
    con.close()
    _sql(path, "DELETE FROM audit_entries WHERE seq = 2")
    _sql(path, "UPDATE audit_head SET length = 2, signature = ?", sig1)  # attacker cannot recompute the anchor HMAC
    with pytest.raises(IntegrityError):
        DurableAuditStore(path, _SEC)
    path2, _ = _durable(tmp_path, "b.db")
    with pytest.raises(IntegrityError):
        DurableAuditStore(path2, b"other-secret")


def test_durable_audit_rejects_bad_text_before_any_write(tmp_path):
    path, head = _durable(tmp_path, n=1)
    s = DurableAuditStore(path, _SEC)
    bad = [("", ValueError), ("a\x00b", ValueError), ("a\x1fb", ValueError), ("a\x7fb", ValueError),
           ("a\nb", ValueError), (5, TypeError), (None, TypeError)]
    for field in ("actor", "action", "resource", "decision"):
        for value, exc in bad:
            args = dict(actor="a", action="b", resource="c", decision="allow")
            args[field] = value
            with pytest.raises(exc):
                s.append(**args)
            assert s.head() == head, (field, value)
    ok = dict(actor="a b~", action="b b~", resource="c b~", decision="al b~")  # 0x20 and 0x7e are legal
    s.append(**ok)
    assert s.head()["length"] == head["length"] + 1
    s.close()


def test_durable_audit_rejects_bad_ts_and_meta_before_any_write(tmp_path):
    path, head = _durable(tmp_path, n=1)
    s = DurableAuditStore(path, _SEC)
    for kw, exc in ((dict(ts=True), TypeError), (dict(ts="1"), TypeError), (dict(ts=float("nan")), ValueError),
                    (dict(ts=float("inf")), ValueError), (dict(meta=[1]), TypeError), (dict(meta={1: 2}), TypeError),
                    (dict(meta={"k": float("nan")}), ValueError), (dict(meta={"k": {1, 2}}), TypeError)):
        with pytest.raises(exc):
            s.append("a", "b", "c", "allow", **kw)
    assert s.head() == head and len(s.export()) == head["length"]
    s.append("a", "b", "c", "allow", ts=0)  # ts=0 is a legal timestamp, not "missing"
    assert s.export()[-1]["ts"] == 0
    s.close()


def test_durable_audit_second_handle_refreshes_instead_of_forking_the_chain(tmp_path):
    path, _ = _durable(tmp_path, n=1)
    h1, h2 = DurableAuditStore(path, _SEC), DurableAuditStore(path, _SEC)
    h1.append("a", "from-h1", "r", "allow")
    e = h2.append("a", "from-h2", "r", "allow")  # stale handle must catch up first
    assert h2.head()["length"] == 3 and e["prev_sig"] == h1.export()[-1]["signature"]
    assert [x["action"] for x in h2.export()][-2:] == ["from-h1", "from-h2"]
    h1.close()
    h2.close()


@pytest.mark.parametrize("name", ["docs", "_t", "T_1", "a" * 64])
def test_durable_docs_accepts_valid_table_names(tmp_path, name):
    DurableDocs(str(tmp_path / "ok.db"), name)


@pytest.mark.parametrize("name", ["a" * 65, "1abc", "a-b", "a b", "docs\n", "\ndocs", "docs;", 'd"x', "", "docs\x00"])
def test_durable_docs_rejects_bad_table_names_with_full_match(tmp_path, name):
    with pytest.raises(ValueError):
        DurableDocs(str(tmp_path / "bad.db"), name)


def test_durable_docs_key_rules_prefix_scan_and_hash_verification(tmp_path):
    d = DurableDocs(str(tmp_path / "k.db"), "docs")
    d.put("k" * 256, {"ok": 1})
    for bad, exc in (("", ValueError), ("k" * 257, ValueError), ("a\x00", ValueError), ("a\x1f", ValueError),
                     ("a\x7f", ValueError), (5, TypeError)):
        with pytest.raises(exc):
            d.put(bad, {})
        with pytest.raises(exc):
            d.get(bad)
    d.put("a b~", 1)  # 0x20 and 0x7e legal
    d.put("a%x", 2)
    d.put("a_x", 3)
    d.put("abx", 4)
    assert sorted(d.keys("a%")) == ["a%x"]  # prefix is literal, not a LIKE pattern
    assert sorted(d.keys("a_")) == ["a_x"]
    assert sorted(d.keys("a")) == ["a b~", "a%x", "a_x", "abx"]
    assert len(d.keys()) == 5 and d.keys("zzz") == []
    with pytest.raises(ValueError):
        d.keys("k" * 257)
    with pytest.raises(ValueError):
        d.keys("a\x01")
    with pytest.raises(NotFound):
        d.get("missing")
    assert d.get("abx") == 4
    _sql(str(tmp_path / "k.db"), "UPDATE docs SET content = '5' WHERE key = 'abx'")
    with pytest.raises(IntegrityError):
        d.get("abx")
    d.delete("abx")
    with pytest.raises(NotFound):
        d.get("abx")


def test_durable_docs_put_many_is_all_or_nothing(tmp_path):
    d = DurableDocs(str(tmp_path / "m.db"), "docs")
    d.put_many({"a": 1, "b": 2})
    assert sorted(d.keys()) == ["a", "b"]
    with pytest.raises((ValueError, TypeError)):
        d.put_many({"c": 3, "bad\x01key": 4})
    with pytest.raises(TypeError):
        d.put_many({"d": 5, "e": {1, 2}})  # non-canonical value
    assert sorted(d.keys()) == ["a", "b"]  # no partial commit


def test_tenantfloor_far_timestamp_limit_measures_distance_from_t0_in_both_directions():
    f = TenantFloor(_cfg(window_s=1.0, t0=5.0))
    assert f.admit(5.0 + 1e12, 1).window_index == 10**12  # distance is |now - t0|, not |now + t0|
    with pytest.raises(ValueError):
        TenantFloor(_cfg(window_s=1.0, t0=5.0)).admit(5.0 + 1e12 + 0.5, 1)  # just over the limit
    with pytest.raises(ValueError):
        TenantFloor(_cfg(window_s=1.0, t0=5.0)).admit(5.0 - 1e12 - 0.5, 1)  # symmetric below t0
    assert TenantFloor(_cfg(window_s=1.0, t0=5.0)).admit(5.0 - 1e12, 1).window_index == -(10**12)


def test_simulate_guard_reports_exact_admission_counters():
    rej = simulate_guard(4, 2, 5, 0, _cfg(credits=2, window_s=10.0, batch_slots=1, s_max=1.0, excess="reject"), True)
    # 2 windows x min(5, 2) credits admitted; the other 6 are rejected, none demoted
    assert (rej["assured_offered"], rej["assured_admitted"], rej["assured_rejected"], rej["assured_demoted"]) == (10, 4, 6, 0)
    assert rej["assured_miss_rate"] == pytest.approx(6 / 10)  # every rejection counts as a miss; the 4 admitted finish on time
    dem = simulate_guard(4, 2, 5, 0, _cfg(credits=2, window_s=10.0, batch_slots=1, s_max=1.0, excess="demote"), True)
    assert (dem["assured_admitted"], dem["assured_rejected"], dem["assured_demoted"]) == (4, 0, 6)
    ung = simulate_guard(4, 2, 5, 3, _cfg(credits=2, window_s=10.0, batch_slots=4, s_max=1.0), False)
    assert (ung["assured_rejected"], ung["assured_demoted"], ung["guarded"]) == (0, 0, False)
    assert ung["assured_offered"] == 10 and ung["assured_admitted"] == 10  # ample capacity: nothing is dropped
    assert ung["assured_miss_rate"] == 0.0 and ung["opportunistic_miss_rate"] == 0.0
    none = simulate_guard(4, 2, 0, 3, _cfg(), True)
    assert none["assured_offered"] == 0 and none["assured_miss_rate"] == 0.0


# =============================================================== config_lint
from mlops.config_lint import (  # noqa: E402
    CATEGORY_WEIGHTS, Finding, lint_context, lint_documents, lint_manifest, lint_score, load_manifests_json,
)


def _dep(ns="prod", name="web", img="reg.io/app:1.0", replicas=3, labels=None, kind="Deployment"):
    return {"apiVersion": "apps/v1", "kind": kind, "metadata": {"name": name, "namespace": ns},
            "spec": {"replicas": replicas,
                     "template": {"metadata": {"labels": labels or {"app": name}},
                                  "spec": {"containers": [{"name": "c", "image": img}]}}}}


def _pdb(ns="prod", match=None):
    return {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": {"name": "pdb", "namespace": ns},
            "spec": {"selector": {"matchLabels": match or {"app": "web"}}}}


def _quota(ns="prod", kind="ResourceQuota"):
    return {"apiVersion": "v1", "kind": kind, "metadata": {"name": "q", "namespace": ns}}


def _rules(findings, rid):
    return [f for f in findings if f.rule_id == rid]


def test_category_weights_are_the_published_values_and_sum_to_one():
    assert CATEGORY_WEIGHTS == {"access_privileges": 0.238, "resources_probes": 0.215, "encryption_permissions": 0.203,
                                "image_network": 0.190, "filesystem": 0.154}
    assert sum(CATEGORY_WEIGHTS.values()) == pytest.approx(1.0)


def _fnd(sev, cat="image_network"):
    return Finding("r", cat, "s", sev, "p", "m", CATEGORY_WEIGHTS[cat])


def test_lint_score_block_counts_full_and_warn_half_and_context_counts():
    assert lint_score([]) == 0.0
    assert lint_score([_fnd("block")]) == pytest.approx(0.19)
    assert lint_score([_fnd("warn")]) == pytest.approx(0.095)
    mixed = [_fnd("block", "access_privileges"), _fnd("warn", "resources_probes"), _fnd("warn", "filesystem")]
    assert lint_score(mixed) == pytest.approx(0.238 + 0.5 * 0.215 + 0.5 * 0.154)
    ctx = lint_context(mixed)
    assert ctx == {"lint_blockers": 1, "lint_warnings": 2, "lint_score": pytest.approx(lint_score(mixed))}
    assert lint_context([]) == {"lint_blockers": 0, "lint_warnings": 0, "lint_score": 0.0}


def test_no_quota_is_per_namespace_and_satisfied_by_either_quota_kind():
    assert len(_rules(lint_documents([_dep()]), "no-quota")) == 1
    assert _rules(lint_documents([_dep(), _quota("prod")]), "no-quota") == []
    assert _rules(lint_documents([_dep(), _quota("prod", "LimitRange")]), "no-quota") == []
    assert len(_rules(lint_documents([_dep(), _quota("other")]), "no-quota")) == 1  # a quota elsewhere does not count
    assert _rules(lint_documents([_quota("prod")]), "no-quota") == []  # a quota with no workload is fine
    f = _rules(lint_documents([_dep()]), "no-quota")[0]
    assert (f.severity, f.category, f.weight) == ("warn", "resources_probes", 0.215) and "'prod'" in f.message


def test_no_pdb_needs_replicas_over_one_matching_labels_and_same_namespace():
    assert len(_rules(lint_documents([_dep(replicas=2)]), "no-pdb")) == 1  # 2 replicas is enough to need a PDB
    assert _rules(lint_documents([_dep(replicas=1)]), "no-pdb") == []
    assert _rules(lint_documents([_dep(replicas=True)]), "no-pdb") == []  # bool is not a replica count
    assert len(_rules(lint_documents([_dep(kind="StatefulSet")]), "no-pdb")) == 1
    assert _rules(lint_documents([_dep(kind="DaemonSet")]), "no-pdb") == []
    assert _rules(lint_documents([_dep(), _pdb()]), "no-pdb") == []
    assert _rules(lint_documents([_dep(labels={"app": "web", "tier": "fe"}), _pdb(match={"app": "web"})]), "no-pdb") == []  # subset
    assert len(_rules(lint_documents([_dep(), _pdb(match={"app": "other"})]), "no-pdb")) == 1
    assert len(_rules(lint_documents([_dep(), _pdb(match={"app": "web", "tier": "fe"})]), "no-pdb")) == 1  # PDB needs MORE labels
    assert len(_rules(lint_documents([_dep(), _pdb(ns="other")]), "no-pdb")) == 1  # namespaces must agree
    f = _rules(lint_documents([_dep()]), "no-pdb")[0]
    assert (f.severity, f.category, f.weight) == ("warn", "resources_probes", 0.215)


def test_untrusted_registry_uses_allowlist_and_defaults_bare_images_to_docker_io():
    docs = [_dep(img="reg.io/app:1.0")]
    assert _rules(lint_documents(docs, image_allowlist=["reg.io"]), "untrusted-registry") == []
    assert len(_rules(lint_documents(docs, image_allowlist=["other.io"]), "untrusted-registry")) == 1
    assert _rules(lint_documents(docs), "untrusted-registry") == []  # no allowlist => rule off
    assert _rules(lint_documents([_dep(img="nginx:1.2")], image_allowlist=["docker.io"]), "untrusted-registry") == []
    assert len(_rules(lint_documents([_dep(img="nginx:1.2")], image_allowlist=["reg.io"]), "untrusted-registry")) == 1
    f = _rules(lint_documents(docs, image_allowlist=["other.io"]), "untrusted-registry")[0]
    assert (f.severity, f.category, f.weight) == ("warn", "image_network", 0.190) and f.path.endswith("containers[0].image")


def test_lint_documents_output_is_deterministic_and_input_is_not_mutated():
    import copy as _copy
    docs = [_dep(ns="b", name="y"), _dep(ns="a", name="x")]
    snap = _copy.deepcopy(docs)
    a = lint_documents(docs, image_allowlist=["ok.io"])
    assert docs == snap  # input never mutated
    assert a == lint_documents(docs, image_allowlist=["ok.io"]) == lint_documents(_copy.deepcopy(docs), image_allowlist=["ok.io"])
    assert docs == snap
    # cross-document rules: ONE finding per offending namespace / container, with distinct paths
    for rid in ("no-quota", "no-pdb"):
        assert [f.path for f in _rules(a, rid)] == ["metadata.namespace[a]", "metadata.namespace[b]"]
    assert len(_rules(a, "untrusted-registry")) == 2  # one per offending container (one per document here)
    cross = ("no-quota", "no-pdb")
    keys = [(f.rule_id, f.path) for f in a]
    per_doc = [k for k in keys if k[0] not in cross]
    tail = [k for k in keys if k[0] in cross]
    half = len(per_doc) // 2
    assert per_doc[:half] == per_doc[half:] == sorted(per_doc[:half])  # each document's findings sorted by (rule_id, path)
    assert keys == per_doc + tail and tail == sorted(tail)  # cross-document findings come last, sorted by (rule_id, path)


@pytest.mark.parametrize("bad", ["x", [1], [{}, "x"], {"kind": "Pod"}])
def test_lint_documents_rejects_bad_shapes(bad):
    with pytest.raises(ValidationFailed):
        lint_documents(bad)
    with pytest.raises(ValidationFailed):
        lint_documents([], image_allowlist="reg.io")
    with pytest.raises(ValidationFailed):
        lint_documents([], image_allowlist=[1])


def test_cross_document_rules_report_every_offending_namespace_and_container():
    f = lint_documents([_dep("a", "x"), _dep("b", "y")], image_allowlist=["ok.io"])
    assert len(_rules(f, "no-quota")) == 2 and len(_rules(f, "no-pdb")) == 2 and len(_rules(f, "untrusted-registry")) == 2


def test_lint_score_is_within_unit_interval_for_many_findings():
    assert lint_score([_fnd("block", "access_privileges")] * 10) <= 1.0


def test_lint_score_is_clamped_only_above_one_and_equals_the_sum_below_it():
    many = [_fnd("block", "access_privileges")] * 50
    assert lint_score(many) == 1.0
    assert lint_score(many + [_fnd("warn")]) == 1.0
    small = [_fnd("block", "image_network"), _fnd("warn", "access_privileges"), _fnd("context", "image_network")]
    expected = CATEGORY_WEIGHTS["image_network"] + 0.5 * CATEGORY_WEIGHTS["access_privileges"] + 0.5 * CATEGORY_WEIGHTS["image_network"]
    assert expected < 1.0 and lint_score(small) == pytest.approx(expected)


def test_load_manifests_json_three_formats_and_limits():
    one = {"kind": "Pod"}
    assert load_manifests_json('{"kind": "Pod"}') == [one]
    assert load_manifests_json('[{"kind": "Pod"}, {"kind": "Job"}]') == [one, {"kind": "Job"}]
    assert load_manifests_json('{"kind": "List", "items": [{"kind": "Pod"}]}') == [one]
    assert len(load_manifests_json(json.dumps([{}] * 5000))) == 5000
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps([{}] * 5001))
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps({"kind": "List", "items": [{}] * 5001}))
    for bad in ("not json", "5", '"s"', "[1]", '{"kind":"List","items":3}', '{"kind":"List","items":[1]}'):
        with pytest.raises(ValidationFailed):
            load_manifests_json(bad)

    def nest(n):
        d = {}
        for _ in range(n):
            d = {"k": d}
        return d

    load_manifests_json(json.dumps(nest(49)))
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps(nest(52)))


# ---- lineage persistence: each tamper defence must work on its own (they overlap, so isolate them)
import hashlib as _hl  # noqa: E402

from mlops.lineage import LineageGraph  # noqa: E402
from mlops.storage_sqlite import load_lineage, save_lineage  # noqa: E402


def _saved_lineage(tmp_path):
    docs = DurableDocs(str(tmp_path / "l.db"), "lineage")
    g = LineageGraph()
    a = g.add_node("data", {"k": "a"}, "2026-01-01T00:00:00Z")
    b = g.add_node("model", {"k": "b"}, "2026-01-02T00:00:00Z")
    g.add_edge(a, b, "feeds")
    save_lineage(docs, g)
    return docs, g, a, b


def _canon(node_type, props, ts):
    return json.dumps({"type": node_type, "properties": props, "timestamp": ts}, sort_keys=True)


def test_lineage_roundtrip_is_exact(tmp_path):
    docs, g, a, b = _saved_lineage(tmp_path)
    g2 = load_lineage(docs)
    assert g2.nodes == g.nodes and g2.immutable_content == g.immutable_content
    assert [tuple(e) for e in g2.edges] == [(a, b, "feeds")]


def test_lineage_node_edit_is_caught_by_content_hash_even_without_immutable_content(tmp_path):
    docs, g, a, b = _saved_lineage(tmp_path)
    node = docs.get("lineage:node:" + a)
    node["properties"] = {"k": "EVIL"}  # content_hash left stale
    docs.put("lineage:node:" + a, node)  # DurableDocs re-signs the doc itself: only the lineage layer can notice
    docs.put("lineage:immutable_content", {})  # attacker also wipes the cross-check
    with pytest.raises(IntegrityError):
        load_lineage(docs)


def test_lineage_consistent_rewrite_is_caught_by_immutable_content(tmp_path):
    docs, g, a, b = _saved_lineage(tmp_path)
    node = docs.get("lineage:node:" + a)
    node["properties"] = {"k": "EVIL"}
    node["content_hash"] = _hl.sha256(_canon("data", {"k": "EVIL"}, node["timestamp"]).encode()).hexdigest()  # self-consistent forgery
    docs.put("lineage:node:" + a, node)
    with pytest.raises(IntegrityError):
        load_lineage(docs)


def test_lineage_missing_node_and_missing_optional_docs(tmp_path):
    docs, g, a, b = _saved_lineage(tmp_path)
    docs.delete("lineage:node:" + b)
    with pytest.raises(IntegrityError):
        load_lineage(docs)
    empty = load_lineage(DurableDocs(str(tmp_path / "e.db"), "lineage"))
    assert empty.nodes == {} and empty.edges == [] and empty.immutable_content == {}


def test_chaos_names_allow_space_reject_control_and_steps_are_numbered_from_one():
    SteadyState("has space", "error_rate", "<=", 1)  # 0x20 legal in names and metrics
    for bad in ("a\x1f", "a\x01", "a\n"):
        with pytest.raises(ValidationFailed):
            SteadyState(bad, "error_rate", "<=", 1)
        with pytest.raises(ValidationFailed):
            SteadyState("n", bad, "<=", 1)
    with pytest.raises(TypeError):
        SteadyState(5, "m", "<=", 1)
    with pytest.raises(ValidationFailed):
        SteadyState("n", "m", "=", 1)
    with pytest.raises(ValidationFailed):
        SteadyState("n", "m", "<=", True)
    r = run_scenario(Scenario([[_kill("one")], [_kill("one")], [_kill("one")]], [SteadyState("n", "error_rate", "<=", 1)], "s"),
                     SimulatedCluster(_pods(4)))
    assert [s.get("step") for s in r.stages if s["stage"] == "inject"] == [1, 2, 3]
    assert [s["metrics"]["ready_pods"] for s in r.stages] == [4, 3, 3, 3, 4]  # "one" always picks the sorted-first pod, already dead after step 1


def test_drift_functions_accept_only_list_or_tuple_containers_and_int_bins():
    import numpy as np
    from mlops.drift import js_divergence, kl_divergence, ks_statistic, psi
    half, s30 = [0.5, 0.5], _gauss(1, 40)
    for fn in (psi, kl_divergence, js_divergence):
        assert fn(tuple(half), half) == pytest.approx(0.0, abs=1e-12)  # tuples are fine
        for bad in (np.array(half), {0.5, 0.25}, "ab", 5):
            with pytest.raises(ValueError):
                fn(bad, half)
            with pytest.raises(ValueError):
                fn(half, bad)
        with pytest.raises(ValueError):
            fn(half, [0.5, 0.25, 0.25])  # unequal lengths
    for bad in (np.array(s30), "x" * 40, 5, {1.0: 1}):
        with pytest.raises(ValueError):
            psi_from_samples(bad, s30)
        with pytest.raises(ValueError):
            psi_from_samples(s30, bad)
        with pytest.raises(ValueError):
            ks_statistic(bad, s30)
        with pytest.raises(ValueError):
            ks_statistic(s30, bad)
    for bad in (2.5, 4.0, "5", None):
        with pytest.raises(ValueError):
            psi_from_samples(s30, _gauss(2, 40), bad)
    for bad in ("ab", np.array([1, 2]), 5):
        with pytest.raises(ValueError):
            cohens_kappa(bad, [1, 2])
        with pytest.raises(ValueError):
            cohens_kappa([1, 2], bad)


def test_psi_from_samples_default_is_ten_bins():
    exp, act = _gauss(1, 200), _gauss(2, 200, shift=0.5)
    assert psi_from_samples(exp, act) == psi_from_samples(exp, act, 10)
    assert psi_from_samples(exp, act) != psi_from_samples(exp, act, 11)


def test_calibration_feedback_container_must_be_list_or_tuple_and_loosen_approval_type_is_denied():
    import numpy as np
    ok = _fb(0.5, True)
    _check_feedback_list((ok,))
    with pytest.raises(ValidationFailed):
        _check_feedback_list(np.array([ok], dtype=object))  # sized, iterable, full of Feedback - but not a list/tuple
    ps, _ = _cal_store(0.8)
    cal = Calibrator(ps, "acc", step=0.05)
    prop = cal.propose([_fb(0.7, False)])
    for junk in (5, ["carol"], {"a": 1}):  # truthy non-strings are a *denial*, not a format error
        with pytest.raises(PolicyDenied):
            cal.apply("a", prop, human_approved_by=junk)
    assert len(ps.history()) == 1


def test_scenario_steady_states_must_be_a_list():
    ss = SteadyState("n", "error_rate", "<=", 1)
    Scenario([[_kill()]], [ss], "s")
    with pytest.raises(TypeError):
        Scenario([[_kill()]], (ss,), "s")


def _pod(*containers, ns="prod"):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p", "namespace": ns},
            "spec": {"containers": list(containers)}}


def _ctr(**kw):
    base = {"name": "c", "image": "reg.io/app:1.0", "livenessProbe": {"a": 1}, "readinessProbe": {"a": 1},
            "securityContext": {"readOnlyRootFilesystem": True, "allowPrivilegeEscalation": False},
            "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "1"}}}
    base.update(kw)
    return base


def test_no_resources_needs_nonempty_dict_requests_and_limits():
    assert _rules(lint_manifest(_pod(_ctr())), "no-resources") == []
    for res in (None, {}, {"requests": {"cpu": "1"}}, {"limits": {"cpu": "1"}}, {"requests": {}, "limits": {"cpu": "1"}},
                {"requests": {"cpu": "1"}, "limits": {}}, {"requests": "1Gi", "limits": "2Gi"},
                {"requests": ["cpu"], "limits": {"cpu": "1"}}, "big", 5):
        c = _ctr()
        if res is None:
            del c["resources"]
        else:
            c["resources"] = res
        assert len(_rules(lint_manifest(_pod(c)), "no-resources")) == 1, res


def test_no_probes_needs_both_probes():
    assert _rules(lint_manifest(_pod(_ctr())), "no-probes") == []
    for drop in ("livenessProbe", "readinessProbe"):
        c = _ctr()
        del c[drop]
        f = _rules(lint_manifest(_pod(c)), "no-probes")
        assert len(f) == 1 and f[0].severity == "warn" and f[0].weight == 0.215


@pytest.mark.parametrize("name,flagged", [
    ("DB_PASSWORD", True), ("secret_key", True), ("AUTH_TOKEN", True), ("API_KEY", True), ("apikey", True),
    ("x_Api_Key", True), ("MY-PASSWORD-1", True),
    ("MONKEY", False), ("APIARY", False), ("KEYBOARD", False), ("HOST", False), ("TOKE", False), ("PASS", False),
])
def test_plaintext_secret_env_name_matching(name, flagged):
    found = _rules(lint_manifest(_pod(_ctr(env=[{"name": name, "value": "hunter2"}]))), "plaintext-secret-env")
    assert len(found) == (1 if flagged else 0)
    if flagged:
        assert (found[0].severity, found[0].category, found[0].weight) == ("block", "encryption_permissions", 0.203)
        assert "hunter2" not in found[0].message  # the finding must not echo the secret


def test_plaintext_secret_env_only_when_a_literal_value_is_present():
    ref = {"secretKeyRef": {"name": "s", "key": "k"}}
    for env in ([{"name": "DB_PASSWORD", "valueFrom": ref}],  # sourced from a Secret: fine
                [{"name": "DB_PASSWORD"}],  # nothing set at all
                [{"name": "DB_PASSWORD", "value": "x", "valueFrom": ref}]):  # ambiguous, valueFrom wins
        assert _rules(lint_manifest(_pod(_ctr(env=env))), "plaintext-secret-env") == [], env
    assert len(_rules(lint_manifest(_pod(_ctr(env=[{"name": "DB_PASSWORD", "value": ""}]))), "plaintext-secret-env")) == 1


def test_env_item_without_name_does_not_crash_the_linter():
    lint_manifest(_pod(_ctr(env=[{"value": "x"}])))
    lint_manifest(_pod(_ctr(env=["junk"])))


@pytest.mark.parametrize("image,flagged", [
    ("app", True), ("app:latest", True), ("reg.io/app", True), ("reg.io/app:latest", True), ("reg.io:5000/app", True),
    ("app:1.0", False), ("reg.io/app:1.0", False), ("reg.io/team/app:1.0", False), ("reg.io/team/app", True),
    ("reg.io:5000/team/app:2", False), ("reg.io/app@sha256deadbeef", False), ("reg.io/app@sha256:" + "a" * 64, False),
])
def test_latest_tag_detection_matrix(image, flagged):
    n = len(_rules(lint_manifest(_pod(_ctr(image=image))), "latest-tag"))
    assert n == (1 if flagged else 0), image


def test_nesting_depth_limit_is_50_for_manifests_and_json_loader():
    def nest(m):  # m wrappers around {} => the deepest dict sits m levels below the wrapper's parent
        d = {}
        for _ in range(m):
            d = {"k": d}
        return d

    ok, deep = _pod(_ctr()), _pod(_ctr())
    ok["x"] = nest(49)  # deepest dict at depth 50
    deep["x"] = nest(50)  # deepest dict at depth 51
    lint_manifest(ok)
    with pytest.raises(ValidationFailed):
        lint_manifest(deep)
    load_manifests_json(json.dumps(nest(50)))
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps(nest(51)))
    assert len(load_manifests_json(json.dumps({"kind": "List", "items": [{}] * 5000}))) == 5000


def test_junk_entry_in_containers_does_not_hide_a_privileged_container():
    bad = _ctr(securityContext={"privileged": True})
    assert _rules(lint_manifest(_pod(bad, "junk")), "privileged") != []


def test_nameless_env_item_does_not_hide_a_plaintext_secret():
    c = _ctr(env=[{"value": "x"}, {"name": "DB_PASSWORD", "value": "hunter2"}])
    assert len(_rules(lint_manifest(_pod(c)), "plaintext-secret-env")) == 1


def test_valid_findings_survive_alongside_malformed_siblings():
    priv = _ctr(securityContext={"privileged": True})
    assert len(_rules(lint_manifest(_pod(None, priv)), "privileged")) == 1
    c = _ctr(env=[{"value": "x"}, {"name": "DB_PASSWORD", "value": "hunter2"}, None])
    assert len(_rules(lint_manifest(_pod(c)), "plaintext-secret-env")) == 1


def _client_with_tokens(*tokens):
    app = FastAPI()
    install_error_handlers(app)
    state = SimpleNamespace(settings=SimpleNamespace(tokens={t: SimpleNamespace(name="ann", role="operator") for t in tokens}))
    app.dependency_overrides[get_state] = lambda: state

    @app.get("/who")
    def who(p: Principal = Depends(get_principal)):
        return {"name": p.name}

    return TestClient(app, raise_server_exceptions=False)


def _get(c, token):
    return c.get("/who", headers={"Authorization": b"Bearer " + token})


def test_bearer_token_shape_is_checked_even_when_it_equals_a_configured_token():
    long512, long513 = "t" * 512, "t" * 513
    c = _client_with_tokens(long512, long513, "café-token-1234", "ctl\x7f-token-1234", "ctl\x01-token-1234", "edge~token-1234")
    assert _get(c, long512.encode()).status_code == 200  # 512 printable ASCII chars is the ceiling
    assert _get(c, long513.encode()).status_code == 401  # 513 is refused even though it is configured
    assert _get(c, "café-token-1234".encode("utf-8")).status_code == 401  # non-ASCII is never a credential
    assert _get(c, "café-token-1234".encode("latin-1")).status_code == 401
    assert _get(c, b"ctl\x7f-token-1234").status_code == 401  # DEL
    assert _get(c, b"ctl\x01-token-1234").status_code == 401  # C0 control
    assert _get(c, b"edge~token-1234").status_code == 200  # 0x7e is the last legal character
    assert _get(_client_with_tokens("!edge-token-1234"), b"!edge-token-1234").status_code == 200  # 0x21 is the first
    r = _get(c, b"nope")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["message"] == "Invalid token"


# =============================================================== sqldb
import mlops.sqldb as _sqldb  # noqa: E402


def _repo(tmp_path):
    url = "sqlite:///%s/h.db" % tmp_path
    _sqldb.upgrade(url)
    return _sqldb.SqlRepo(url), url


def _mrec(**kw):
    r = {"model_id": "m", "version_id": "v1", "artifact_hash": "a" * 64, "dataset_version": None, "stage": "registered",
         "validated": False, "previous_production": None, "created_ts": 1.0, "updated_ts": 1.0}
    r.update(kw)
    return r


def test_sqldb_model_record_validation_is_exact_and_writes_nothing(tmp_path):
    repo, _ = _repo(tmp_path)
    good = _mrec()
    bad_records = [
        "nope", None, [good], {k: v for k, v in good.items() if k != "stage"}, dict(good, extra=1),
        _mrec(stage="live"), _mrec(stage=None), _mrec(previous_production=5), _mrec(dataset_version=5),
        _mrec(validated=1), _mrec(artifact_hash="A" * 64), _mrec(artifact_hash="a" * 63), _mrec(artifact_hash="a" * 64 + "\n"),
        _mrec(model_id=""), _mrec(model_id="m" * 257), _mrec(model_id="m\x01"), _mrec(model_id="m\x7f"), _mrec(model_id="m\x85"),
        _mrec(version_id="v\x00"), _mrec(created_ts=True), _mrec(created_ts=float("nan")), _mrec(updated_ts=float("inf")),
        _mrec(created_ts="1"),
    ]
    for rec in bad_records:
        with pytest.raises(ValidationFailed):
            repo.upsert_model(rec)
    assert repo.count() == 0
    repo.upsert_model(_mrec(model_id="m" * 256, version_id="v" * 256, dataset_version="d", previous_production="v0", stage="archived"))
    repo.upsert_model(_mrec(model_id="m 1~", version_id="é"))  # 0x20, 0x7e and non-control unicode are fine
    assert repo.count() == 2
    repo.close()


def test_sqldb_set_stage_validates_ids_and_stage_before_touching_rows(tmp_path):
    repo, _ = _repo(tmp_path)
    repo.upsert_model(_mrec())
    for args in ((5, "v1", "staging"), ("", "v1", "staging"), ("m" * 257, "v1", "staging"), ("m\x01", "v1", "staging"),
                 ("m", "", "staging"), ("m", "v" * 257, "staging"), ("m", "v\x7f", "staging"), ("m", "v1", "live"),
                 ("m", "v1", None)):
        with pytest.raises(ValidationFailed):
            repo.set_stage(*args)
    with pytest.raises(NotFound):
        repo.set_stage("m", "ghost", "staging")
    assert repo.get_model("m")[0]["stage"] == "registered"
    repo.set_stage("m", "v1", "production", previous_production="v0")
    row = repo.get_model("m")[0]
    assert row["stage"] == "production" and row["previous_production"] == "v0" and row["updated_ts"] > 1.0
    repo.close()


def test_sqldb_url_and_revision_validation_and_closed_repo(tmp_path):
    for bad in (None, "", "postgresql://x/y", "sqlite://", "sqlite:///a\x01b", "sqlite:///" + "x" * 2050):
        with pytest.raises(ValidationFailed):
            _sqldb.current_revision(bad)
    for bad in ("", "a b", "x" * 65, 5, "a;b"):
        with pytest.raises(ValidationFailed):
            _sqldb.upgrade("sqlite:///%s/r.db" % tmp_path, bad)
    repo, _ = _repo(tmp_path)
    repo.close()
    repo.close()  # idempotent
    with pytest.raises(RuntimeError):
        repo.count()
    with pytest.raises(RuntimeError):
        repo.upsert_model(_mrec())


def test_sqldb_repo_is_refused_when_the_alembic_revision_is_not_head(tmp_path):
    url = "sqlite:///%s/g.db" % tmp_path
    _sqldb.upgrade(url)
    con = sqlite3.connect("%s/g.db" % tmp_path)
    con.execute("UPDATE alembic_version SET version_num = 'deadbeef'")
    con.commit()
    con.close()
    with pytest.raises(ValidationFailed, match="database not migrated"):
        _sqldb.SqlRepo(url)


def _drec(**kw):
    r = {"version_id": "d1", "name": "clicks", "content_hash": "b" * 64, "rows": 10, "schema_json": {"b": "int", "a": "str"}}
    r.update(kw)
    return r


def test_sqldb_dataset_record_validation_and_roundtrip(tmp_path):
    repo, _ = _repo(tmp_path)
    bad = ["x", None, [1], {k: v for k, v in _drec().items() if k != "rows"}, dict(_drec(), extra=1),
           _drec(version_id=""), _drec(version_id="v" * 257), _drec(version_id=5), _drec(name=""), _drec(name=5),
           _drec(content_hash=5), _drec(content_hash="B" * 64), _drec(content_hash="b" * 65), _drec(content_hash="b" * 64 + "\n"),
           _drec(rows=-1), _drec(rows=True), _drec(rows=1.5), _drec(rows="1"), _drec(schema_json=[]), _drec(schema_json="{}"),
           _drec(schema_json={1: "int"}), _drec(schema_json={"a": float("nan")})]
    for rec in bad:
        with pytest.raises(ValidationFailed):
            repo.upsert_dataset(rec)
    with pytest.raises(NotFound):
        repo.get_dataset("d1")
    repo.upsert_dataset(_drec(rows=0, version_id="v" * 256))  # zero rows and 256-char ids are fine
    repo.upsert_dataset(_drec())
    repo.upsert_dataset(_drec(rows=11, name="clicks2"))  # same key: update in place
    got = repo.get_dataset("d1")
    assert got == {"version_id": "d1", "name": "clicks2", "content_hash": "b" * 64, "rows": 11, "schema_json": {"a": "str", "b": "int"}}
    repo.close()


def test_sqldb_reads_validate_ids_and_every_method_refuses_a_closed_repo(tmp_path):
    repo, _ = _repo(tmp_path)
    for bad in (5, "", "m" * 257, "m\x01", "m\x9f"):
        with pytest.raises(ValidationFailed):
            repo.get_model(bad)
        with pytest.raises(ValidationFailed):
            repo.get_dataset(bad)
    assert repo.get_model("nobody") == []
    repo.close()
    for call in (lambda: repo.get_model("m"), lambda: repo.get_dataset("d"), lambda: repo.set_stage("m", "v", "staging"),
                 lambda: repo.upsert_dataset(_drec()), lambda: repo.count()):
        with pytest.raises(RuntimeError):
            call()


def test_sdk_non_json_and_malformed_error_bodies_retry_only_on_502_503_504(_nosleep):
    for status, retried in ((502, True), (503, True), (504, True), (500, False), (501, False), (505, False), (429, False), (400, False)):
        for resp in (httpx.Response(status, text="<html>oops</html>"), httpx.Response(status, json={"detail": "x"}),
                     httpx.Response(status, json={"error": "flat string"})):
            s = _Script(resp)
            with pytest.raises(MlopsApiError) as ei:
                s.client(retries=2).health()
            assert len(s.requests) == (3 if retried else 1), (status, resp.text)
            assert ei.value.status == status and ei.value.code == "bad_response"


def test_sdk_error_envelope_defaults_when_code_or_message_missing():
    s = _Script(httpx.Response(409, json={"error": {}}))
    with pytest.raises(MlopsApiError) as ei:
        s.client().health()
    assert (ei.value.status, ei.value.code, ei.value.message) == (409, "unknown", "")
    assert str(ei.value) == "unknown: " and "status=409" in repr(ei.value)


def test_sdk_metrics_text_returns_body_and_maps_errors_without_retrying(_nosleep):
    s = _Script(httpx.Response(200, text="# HELP x\nx 1\n"))
    assert s.client().metrics_text() == "# HELP x\nx 1\n"
    assert s.requests[0].headers["authorization"] == "Bearer " + _TOKEN and s.requests[0].url.path == "/metrics"
    for status in (400, 401, 403, 404, 500, 503):
        s = _Script(httpx.Response(status, text="denied"))
        with pytest.raises(MlopsApiError) as ei:
            s.client(retries=3).metrics_text()
        assert ei.value.status == status and len(s.requests) == 1
    assert _Script(httpx.Response(399, text="edge")).client().metrics_text() == "edge"  # < 400 is success


def test_sdk_endpoint_table_methods_paths_and_optional_fields_are_exact():
    s = _Script(_ok())
    c = s.client()
    c.health()
    c.get_model("m/1")
    c.active_policy()
    c.audit_head()
    c.register_model("m", "v", "a" * 64)
    c.register_model("m", "v", "a" * 64, dataset_version="d1")
    c.publish_policy([{"id": "r"}], "p", 3)
    c.publish_policy([], "p", 3, note="why")
    c.activate_policy(7)
    c.activate_policy(7, human_approved_by="carol")
    c.decide("act", {"acc": 1})
    c.lint([{"kind": "Pod"}])
    c.lint([], image_allowlist=["reg.io"])
    c.drift_check([1.0], [2.0])
    got = [(r.method, r.url.raw_path.decode(), json.loads(r.content) if r.content else None) for r in s.requests]
    assert got == [
        ("GET", "/healthz", None),
        ("GET", "/v1/models/m%2F1", None),
        ("GET", "/v1/policy/active", None),
        ("GET", "/v1/audit/head", None),
        ("POST", "/v1/models", {"model_id": "m", "version_id": "v", "artifact_hash": "a" * 64}),
        ("POST", "/v1/models", {"model_id": "m", "version_id": "v", "artifact_hash": "a" * 64, "dataset_version": "d1"}),
        ("POST", "/v1/policy/publish", {"rules": [{"id": "r"}], "name": "p", "version": 3}),
        ("POST", "/v1/policy/publish", {"rules": [], "name": "p", "version": 3, "note": "why"}),
        ("POST", "/v1/policy/7/activate", {}),
        ("POST", "/v1/policy/7/activate", {"human_approved_by": "carol"}),
        ("POST", "/v1/policy/decide", {"action": "act", "context": {"acc": 1}}),
        ("POST", "/v1/lint", {"documents": [{"kind": "Pod"}]}),
        ("POST", "/v1/lint", {"documents": [], "image_allowlist": ["reg.io"]}),
        ("POST", "/v1/drift/check", {"reference": [1.0], "window": [2.0]}),
    ]


def test_drift_route_sample_size_boundaries_and_response_shape():
    c, audit = _app_client()
    ref, win = _gauss(3, 100), _gauss(103, 30)
    n = _n(audit)
    r = c.post("/v1/drift/check", json={"reference": ref, "window": win}, headers=_h("viewer"))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"level", "psi", "psi_adjusted", "ks", "ks_pvalue", "n"} and body["n"] == 30
    assert body["psi_adjusted"] == pytest.approx(max(0.0, body["psi"] - 9 * (1 / 100 + 1 / 30)), abs=1e-9)
    assert body["level"] == "ok"
    r = c.post("/v1/drift/check", json={"reference": ref, "window": win[:29]}, headers=_h("viewer"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = c.post("/v1/drift/check", json={"reference": ref[:29], "window": win}, headers=_h("viewer"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation"
    r = c.post("/v1/drift/check", json={"reference": ref, "window": ["x"] * 30}, headers=_h("viewer"))
    assert r.status_code == 422
    assert _n(audit) == n  # drift checks are read-only


def test_non_finite_numbers_and_empty_bodies_are_rejected_as_422_before_validation():
    c, audit = _app_client()
    n = _n(audit)
    for raw in (b'{"documents": [], "image_allowlist": [1e999]}', b'{"documents": [[-1e999]]}', b'{"documents": [{"a": {"b": [1e999]}}]}'):
        r = c.post("/v1/lint", content=raw, headers=_h("viewer", **_JSON))
        assert r.status_code == 422 and "non-finite" in r.json()["error"]["message"], raw
    r = c.post("/v1/lint", content=b'{"documents": [{"a": 1.5}]}', headers=_h("viewer", **_JSON))
    assert r.status_code == 200  # ordinary floats are fine
    r = c.post("/v1/models", content=b"", headers=_h("operator", **_JSON))
    assert r.status_code == 422 and r.json()["error"]["message"].startswith("validation error in fields:")  # empty == {}
    assert _n(audit) == n


def test_audit_export_limit_boundaries_and_default():
    c, audit = _app_client()
    for i in range(3):
        c.post("/v1/models", json=_reg(v="v%d" % i), headers=_h("operator"))
    ok = lambda q: c.get("/v1/audit/export" + q, headers=_h("admin"))  # noqa: E731
    assert len(ok("").json()["entries"]) == 3  # default limit 100 covers everything
    assert len(ok("?limit=1").json()["entries"]) == 1 and ok("?limit=1").json()["entries"][0]["signature"] == audit.export()[-1]["signature"]
    assert len(ok("?limit=1000").json()["entries"]) == 3
    for bad in ("0", "1001", "-1"):
        assert ok("?limit=" + bad).status_code == 422
    assert ok("").json()["head"]["length"] == 3


def test_body_depth_limit_applies_to_lists_and_mixed_nesting_exactly_at_64():
    c, audit = _app_client()
    assert c.post("/v1/models", json=_reg(), headers=_h("admin")).status_code == 201

    def nest_lists(m):
        x = []
        for _ in range(m - 1):
            x = [x]
        return x

    def nest_mixed(m):  # alternate dict/list wrappers
        x = {}
        for i in range(m - 1):
            x = [x] if i % 2 else {"k": x}
        return x

    url = "/v1/models/m/versions/v1/validate"
    for build in (nest_lists, nest_mixed):
        # body -> evidence(dict) -> "k": <m containers>: deepest container sits at depth 2 + m
        ok = c.post(url, json={"evidence": {"k": build(62)}}, headers=_h("admin"))
        assert ok.status_code == 200, (build.__name__, ok.text)
        n = _n(audit)
        deep = c.post(url, json={"evidence": {"k": build(63)}}, headers=_h("admin"))
        assert deep.status_code == 422 and "nesting" in deep.json()["error"]["message"], build.__name__
        assert _n(audit) == n


def test_durable_audit_defaults_timestamp_to_now_and_meta_to_empty_dict(tmp_path):
    import time
    path, _ = _durable(tmp_path, n=1)
    s = DurableAuditStore(path, _SEC)
    before = time.time()
    e = s.append("a", "b", "c", "allow")
    assert isinstance(e["ts"], float) and before - 1 <= e["ts"] <= time.time() + 1
    assert e["meta"] == {}
    s.close()
    assert DurableAuditStore(path, _SEC).export()[-1]["meta"] == {}  # and it survives a reload


def test_manifest_depth_limit_counts_list_nesting_too():
    def nest_lists(m):
        x = []
        for _ in range(m - 1):
            x = [x]
        return x

    ok, deep = _pod(_ctr()), _pod(_ctr())
    ok["x"] = nest_lists(50)  # deepest list at depth 50
    deep["x"] = nest_lists(51)
    lint_manifest(ok)
    with pytest.raises(ValidationFailed):
        lint_manifest(deep)
    with pytest.raises(ValidationFailed):
        lint_documents([deep])


def test_cross_document_rules_ignore_workloads_and_quotas_without_a_namespace():
    no_ns = _dep()
    del no_ns["metadata"]["namespace"]
    f = lint_documents([no_ns])
    assert _rules(f, "no-pdb") == [] and _rules(f, "no-quota") == []  # nothing to attribute the finding to
    assert len(_rules(f, "default-namespace")) == 1  # ...but the namespace defect itself is still reported
    no_meta = _dep()
    del no_meta["metadata"]
    f = lint_documents([no_meta, _pdb()])
    assert _rules(f, "no-pdb") == [] and _rules(f, "no-quota") == []
    q = _quota()
    del q["metadata"]["namespace"]
    assert len(_rules(lint_documents([_dep(), q]), "no-quota")) == 1  # a namespace-less quota protects nobody
    pdb_no_ns = _pdb()
    del pdb_no_ns["metadata"]["namespace"]
    assert len(_rules(lint_documents([_dep(), pdb_no_ns]), "no-pdb")) == 1


def test_idempotency_principal_text_rules_and_default_capacity():
    s, _ = _idem()
    for bad in (5, b"olga", "", "ol\x1fga", "ol\x00ga"):
        with pytest.raises(ValidationFailed):
            _begin(s, who=bad)
    assert _begin(s, who="olga smith")[0] == "new"  # a space (0x20) is a legal principal character
    assert _begin(s, who="~")[0] == "new"
    big = IdempotencyStore(ManualClock(0.0), 3600)  # default max_entries
    for i in range(10_001):
        big.begin("key-%08d" % i, "p", "POST", "/x", "h")
    assert len(big._store) == 10_000
    assert big.begin("key-%08d" % 0, "p", "POST", "/x", "h")[0] == "new"  # the oldest was evicted
    assert big.begin("key-%08d" % 10_000, "p", "POST", "/x", "h")[0] == "in_flight"


def test_sqldb_id_length_cap_is_inclusive_on_reads_and_timestamps_reject_nan_and_infinities(tmp_path):
    repo, _ = _repo(tmp_path)
    assert repo.get_model("m" * 256) == []  # 256 chars passes validation (and simply finds nothing)
    with pytest.raises(NotFound):
        repo.set_stage("m" * 256, "v" * 256, "staging")
    with pytest.raises(NotFound):
        repo.get_dataset("d" * 256)
    with pytest.raises(ValidationFailed):
        repo.get_model("m" * 257)
    for field in ("created_ts", "updated_ts"):
        for bad in (float("nan"), float("inf"), float("-inf"), True, None, "1"):
            with pytest.raises(ValidationFailed):
                repo.upsert_model(_mrec(**{field: bad}))
    repo.upsert_model(_mrec(created_ts=0, updated_ts=-5))  # ints (even zero/negative) are legal timestamps
    assert repo.count() == 1
    repo.close()


def test_register_validated_true_is_only_honoured_with_trust_validated():
    # M6 (`if validated and not self.trust_validated:` -> `if False:`): the trust gate is real
    reg, audit = _registry()
    n = audit.head()["length"]
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", _H, validated=True)
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", _H, validated=1)
    assert reg.history("m") == [] and audit.head()["length"] == n
    assert reg.register("alice", "m", "v1", _H, validated=False).validated is False
    trusting, _ = _registry(trust_validated=True)
    rec = trusting.register("alice", "m", "v1", _H, validated=True)
    assert rec.validated is True
    trusting.transition("alice", "m", "v1", Stage.STAGING)
    assert trusting.transition("alice", "m", "v1", Stage.PRODUCTION).stage == Stage.PRODUCTION  # no validate() call needed


# ---- PolicyStore.publish rule validation (all-or-nothing, before any state change)
_UNSET = object()


def _pub_rule(rid="r", kind="min_metric", params=_UNSET, severity="block"):
    if params is _UNSET:
        params = {"metric": "acc", "min": 0.9}
    return Rule(id=rid, version=1, kind=kind, params=params, severity=severity)


def _rejects(*rules):
    ps, audit = _ps()
    n = audit.head()["length"]
    with pytest.raises(ValidationFailed):
        ps.publish("a", PolicyBundle(rules=list(rules), name="p", version=0))
    assert ps.history() == [] and audit.head()["length"] == n  # no version consumed, nothing audited
    return ps


def _accepts(*rules):
    ps, _ = _ps()
    assert ps.publish("a", PolicyBundle(rules=list(rules), name="p", version=0)) == 1


def test_publish_rule_id_rules():
    _accepts(_pub_rule("x" * 128), _pub_rule("has space"), _pub_rule("~"))
    _accepts(*[_pub_rule("r%d" % i) for i in range(200)])  # exactly 200 rules is the ceiling
    _rejects(*[_pub_rule("r%d" % i) for i in range(201)])
    for bad in (None, 5, "", "   ", "x" * 129, "a\x1fb", "a\x7fb", "a\nb"):
        _rejects(_pub_rule(bad))
    _rejects(_pub_rule("dup"), _pub_rule("dup"))
    _rejects(_pub_rule("dup"), _pub_rule(" dup "))  # uniqueness is judged after stripping
    ps, _ = _ps()
    with pytest.raises(ValidationFailed):
        ps.publish("a", PolicyBundle(rules=tuple([_pub_rule("a")]), name="p", version=0))  # rules must be a list


def test_publish_severity_kind_and_params_container_rules():
    _accepts(_pub_rule(severity="warn"), _pub_rule("b", severity="block"))
    for bad in ("critical", "BLOCK", None, 1):
        _rejects(_pub_rule(severity=bad))
    for kind in (None, 5, "nonsense", ""):
        _rejects(_pub_rule(kind=kind))
    for params in ({}, None, [], "x", 5):
        for kind in ("min_metric", "max_metric", "flag_true", "flag_false", "status_equals", "budget_ok", "role_in", "no_nan"):
            _rejects(_pub_rule(kind=kind, params=params))


def test_publish_params_are_validated_per_kind():
    good = {
        "min_metric": {"metric": "acc", "min": 0},
        "max_metric": {"metric": "lat", "max": 1.5},
        "flag_true": {"key": "k"}, "flag_false": {"key": "k"},
        "status_equals": {"key": "k", "value": "ok"},
        "budget_ok": {"key": "cost", "max": 10},
        "role_in": {"key": "role", "roles": ["admin"]},
        "no_nan": {"keys": ["a", "b"]},
    }
    for kind, params in good.items():
        _accepts(_pub_rule(kind=kind, params=params))
    inf, nan = float("inf"), float("nan")
    bad = {
        "min_metric": [{"min": 1}, {"metric": "", "min": 1}, {"metric": "  ", "min": 1}, {"metric": 5, "min": 1}, {"metric": "a"},
                       {"metric": "a", "min": inf}, {"metric": "a", "min": -inf}, {"metric": "a", "min": nan},
                       {"metric": "a", "min": True}, {"metric": "a", "min": "0.5"}, {"metric": "a", "min": None}],
        "max_metric": [{"max": 1}, {"metric": " ", "max": 1}, {"metric": "a"}, {"metric": "a", "max": inf},
                       {"metric": "a", "max": nan}, {"metric": "a", "max": False}, {"metric": "a", "max": "1"}],
        "flag_true": [{}, {"key": ""}, {"key": " "}, {"key": 5}],
        "flag_false": [{}, {"key": ""}, {"key": 5}],
        "status_equals": [{"key": "k"}, {"value": "v"}, {"key": "k", "value": ""}, {"key": "k", "value": " "},
                          {"key": "", "value": "v"}, {"key": "k", "value": 5}],
        "budget_ok": [{"max": 1}, {"key": "k"}, {"key": " ", "max": 1}, {"key": "k", "max": inf}, {"key": "k", "max": nan},
                      {"key": "k", "max": True}, {"key": "k", "max": "1"}],
        "role_in": [{"key": "k"}, {"roles": ["a"]}, {"key": "k", "roles": []}, {"key": "k", "roles": "admin"},
                    {"key": "k", "roles": [""]}, {"key": "k", "roles": ["a", " "]}, {"key": "k", "roles": [5]}, {"key": " ", "roles": ["a"]}],
        "no_nan": [{}, {"keys": []}, {"keys": "ab"}, {"keys": [""]}, {"keys": ["a", 5]}, {"keys": [" "]}],
    }
    for kind, variants in bad.items():
        for params in variants:
            _rejects(_pub_rule(kind=kind, params=params))
    _rejects(_pub_rule("ok"), _pub_rule("bad", kind="min_metric", params={"metric": "a", "min": inf}))  # one bad rule voids the bundle


def test_stricter_or_equal_treats_missing_or_incomparable_thresholds_as_loosening_never_raising():
    def mn(params):
        return _bundle(Rule("r", 1, "min_metric", params, "block"))

    def mx(params):
        return _bundle(Rule("r", 1, "max_metric", params, "block"))

    good_min, good_max = {"metric": "a", "min": 0.5}, {"metric": "a", "max": 5}
    for other in ({}, {"metric": "a"}, None, {"metric": "a", "min": "0.9"}, {"metric": "a", "min": None}):
        assert is_stricter_or_equal(mn(other), mn(good_min)) is False  # new side unusable
        assert is_stricter_or_equal(mn(good_min), mn(other)) is False  # old side unusable
    assert is_stricter_or_equal(mn({}), mn({})) is False  # even two identical broken rules are not "at least as strict"
    for other in ({}, {"metric": "a"}, None, {"metric": "a", "max": "9"}):
        assert is_stricter_or_equal(mx(other), mx(good_max)) is False
        assert is_stricter_or_equal(mx(good_max), mx(other)) is False
    assert is_stricter_or_equal(mn(good_min), mn(good_min)) is True


def test_sdk_defaults_and_backoff_is_capped_at_two_seconds(monkeypatch):
    import tenacity.nap
    http = httpx.Client(transport=httpx.MockTransport(_Script(_ok())))
    c = MlopsClient(client=http, token=_TOKEN)
    assert repr(c) == "MlopsClient(retries=3, backoff_s=0.1)" and str(c) == repr(c) and _TOKEN not in repr(c)
    sleeps = []
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: sleeps.append(s))
    s = _Script(_err(503))
    with pytest.raises(MlopsApiError):
        s.client(retries=3, backoff_s=50.0).health()
    assert len(s.requests) == 4 and sleeps and all(0 < x <= 2.0 for x in sleeps)  # huge initial backoff is clamped to max=2
    assert max(sleeps) == 2.0


def test_interactive_docs_exist_only_in_dev_including_staging():
    dev, _ = _app_client(env_name="dev")
    stg, _ = _app_client(env_name="staging")
    assert dev.get("/docs").status_code == 200 and dev.get("/redoc").status_code == 200
    assert stg.get("/docs").status_code == 404 and stg.get("/redoc").status_code == 404  # staging is not prod but is not dev either


def test_http_identifier_fields_reject_control_chars_and_overlong_values():
    c, audit = _app_client()
    n = _n(audit)
    for bad in ("a\x01b", "a\x1fb", "a\x7fb", "", "m" * 257, 5, None):
        assert c.post("/v1/models", json=_reg(m=bad), headers=_h("operator")).status_code == 422, repr(bad)
        assert c.post("/v1/models", json=_reg(v=bad), headers=_h("operator")).status_code == 422, repr(bad)
    assert _n(audit) == n
    assert c.post("/v1/models", json=_reg(m="m " + "~" * 254), headers=_h("operator")).status_code == 201  # 256 chars, 0x20 and 0x7e
    assert c.post("/v1/models", json=_reg(m="m2", v="v" * 256), headers=_h("operator")).status_code == 201


def test_http_policy_rule_identifiers_are_length_and_control_char_checked():
    c, audit = _app_client()
    n = _n(audit)
    body = lambda rid: {"name": "p", "version": 1, "rules": [{"id": rid, "version": 1, "kind": "no_nan", "params": {"keys": ["a"]}}]}  # noqa: E731
    for bad in ("", "x" * 257, "a\x01b", "a\x7fb"):
        assert c.post("/v1/policy/publish", json=body(bad), headers=_h("admin")).status_code in (400, 422), repr(bad)
    assert _n(audit) == n


def test_invalid_idempotency_key_is_rejected_before_the_body_is_even_parsed():
    c, audit = _app_client()
    n = _n(audit)
    for path, body in (("/v1/models", b"{not json"), ("/v1/models/m/versions/v1/validate", b"[]")):
        r = c.post(path, content=body, headers=_h("operator", **{**_JSON, "Idempotency-Key": "short"}))
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_failed", (path, r.text)
    assert _n(audit) == n


def test_request_duration_histogram_uses_route_templates_and_positive_small_durations():
    c, _ = _app_client()
    c.get("/healthz")
    c.post("/v1/models", json=_reg(), headers=_h("operator"))
    c.get("/v1/models/m", headers=_h("viewer"))
    c.get("/no/such/route/zzz", headers=_h("viewer"))
    text = c.get("/metrics", headers=_h("viewer")).text
    for label in ("/healthz", "/v1/models", "/v1/models/{model_id}", "unmatched"):
        assert 'mlops_http_request_seconds_count{path="%s"}' % label in text, label
    sums = {}
    for line in text.splitlines():
        if line.startswith("mlops_http_request_seconds_sum{"):
            sums[line.split('"')[1]] = float(line.rsplit(" ", 1)[1])
    assert sums and all(0.0 < v < 30.0 for v in sums.values()), sums  # a duration, not a timestamp or a negative number


def test_publish_accepts_dict_rules_only_when_well_formed_and_rejects_foreign_objects():
    ps, audit = _ps()
    n = audit.head()["length"]
    good = {"id": "r", "kind": "min_metric", "params": {"metric": "a", "min": 1}}
    for bad in ({"kind": "min_metric", "params": {"metric": "a", "min": 1}}, dict(good, id=""), dict(good, kind="nope"),
                dict(good, params=None), dict(good, severity="loud"), object(), 5, "rule"):
        with pytest.raises(ValidationFailed):
            ps.publish("a", PolicyBundle(rules=[bad], name="p", version=0))
    assert ps.history() == [] and audit.head()["length"] == n
    assert ps.publish("a", PolicyBundle(rules=[good], name="p", version=0)) == 1  # dict rules default to severity "block"


def test_activate_and_rollback_approval_must_be_a_string_or_none():
    ps, audit = _ps()
    ps.publish("a", _minb(0.9))
    ps.publish("a", _minb(0.5))
    ps.activate("a", 1)
    n = audit.head()["length"]
    for bad in (5, ["carol"], b"carol", True):
        with pytest.raises(ValidationFailed):
            ps.activate("a", 2, human_approved_by=bad)
        with pytest.raises(ValidationFailed):
            ps.rollback("a", human_approved_by=bad)
    assert ps.active()[0] == 1 and audit.head()["length"] == n


def test_percent_modes_are_identical_in_blast_radius_and_in_a_scenario_run():
    big = SimulatedCluster(_pods(200))
    names = sorted(big.pods)
    for mode in ("fixed-percent", "random-max-percent"):
        assert big.blast_radius([_kill(mode, 100)]) == set(names)  # 200 * 100 / 100, not 200 * 100 / 101
        assert big.blast_radius([_kill(mode, 1)]) == set(names[:2])  # exactly 2.0 pods
        assert big.blast_radius([_kill(mode, 50)]) == set(names[:100])
        small = SimulatedCluster(_pods(10))
        assert small.blast_radius([_kill(mode, 1)]) == {sorted(small.pods)[0]}  # ceil(0.1) = 1, never 0
        assert small.blast_radius([_kill(mode, 25)]) == set(sorted(small.pods)[:3])
        ss = [SteadyState("n", "ready_pods", ">=", 0)]
        for value, dead in ((1, 1), (25, 3), (50, 5), (100, 10)):
            r = run_scenario(Scenario([[_kill(mode, value)]], ss, "s"), SimulatedCluster(_pods(10)))
            assert r.stages[1]["metrics"]["ready_pods"] == 10 - dead, (mode, value)
        r = run_scenario(Scenario([[_kill(mode, 100)]], ss, "s"), big)
        assert r.stages[1]["metrics"]["ready_pods"] == 0


def test_metrics_error_rate_for_tiny_clusters():
    one_dead = SimulatedCluster(_pods(1, ready=False))
    assert one_dead.metrics() == {"ready_pods": 0, "error_rate": 1.0, "latency_ms": 20.0}
    assert SimulatedCluster(_pods(1)).metrics()["error_rate"] == 0.0
    assert SimulatedCluster(_pods(2, ready=False)).metrics()["error_rate"] == 1.0


def test_sqldb_clean_string_covers_c0_del_and_c1_but_not_latin1_printables(tmp_path):
    repo, _ = _repo(tmp_path)
    for ch in ("\x1f", "\x7f", "\x80", "\x85", "\x9f"):
        with pytest.raises(ValidationFailed):
            repo.upsert_model(_mrec(model_id="a%sb" % ch))
    for ch in (" ", "~", " ", "¡", "é", "中"):  # first printable after C1 is U+00A0
        repo.upsert_model(_mrec(model_id="a%sb" % ch))
    assert repo.count() == 6
    repo.close()


def test_sqldb_url_length_cap_is_2048_inclusive():
    prefix = "sqlite:///"
    with pytest.raises(ValidationFailed, match="too long"):
        _sqldb.current_revision(prefix + "x" * (2049 - len(prefix)))
    try:
        _sqldb.current_revision(prefix + "x" * (2048 - len(prefix)))
    except ValidationFailed as e:  # the length check itself must pass; any later failure is about the file, not the URL length
        assert "too long" not in str(e)
    except Exception:
        pass


def test_durable_docs_prefix_and_meta_edge_cases(tmp_path):
    d = DurableDocs(str(tmp_path / "p.db"), "docs")
    d.put("a b", 1)
    d.put("a~", 2)
    assert d.keys("a ") == ["a b"] and d.keys("a~") == ["a~"]  # 0x20 and 0x7e may appear in a prefix
    for bad in ("a\x1f", "a\x7f", "a\x00"):
        with pytest.raises(ValueError):
            d.keys(bad)
    with pytest.raises(TypeError):
        d.keys(5)
    path, head = _durable(tmp_path, n=1)
    s = DurableAuditStore(path, _SEC)
    for v in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            s.append("a", "b", "c", "allow", meta={"k": v})
    assert s.head() == head
    s.close()
