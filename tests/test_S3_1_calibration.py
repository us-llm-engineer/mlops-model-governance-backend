"""S3.1 slim suite: policy calibration (Feedback, Proposal, Calibrator, evaluate).
Frozen BEFORE mlops.policy_calibration / mlops.policy_store exist. Hand-computed oracles.
"""
import dataclasses
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.kernel import Conflict, NotFound, PolicyDenied, ValidationFailed
from mlops.policy_calibration import Calibrator, Feedback, Proposal, evaluate
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore, is_stricter_or_equal

BAD = (ValidationFailed, ValueError, TypeError)
ACC = lambda: Rule("acc_floor", 1, "min_metric", {"metric": "acc", "min": 0.90})


def _setup(rules=None, rule_id="acc_floor", step=0.001):
    audit = ChainedAuditStore(b"s3-1-secret")
    store = PolicyStore(audit)
    v = store.publish("alice", PolicyBundle(rules=rules or [ACC()], name="cal", version=1))
    store.activate("alice", v)
    return store, Calibrator(store, rule_id, step=step), audit


def _min(bundle, rid="acc_floor"):
    return [r for r in bundle.rules if r.id == rid][0].params["min"]


def _active_min(store):
    return _min(store.active()[1])


def _fb(acc, block):
    return Feedback({"acc": acc}, block)


def _actions(audit):
    return [e["action"] for e in audit.export()]


# ------------------------------------------------------------------ evaluate

def _bundle():
    return PolicyBundle(rules=[ACC()], name="ev", version=1)


def test_evaluate_hand_computed_and_no_audit_write():
    # blocked iff acc < .90 -> items 2 (.85) and 4 (.88) blocked.
    # missed = should_block & allowed = item 3 (.92); false alarm = item 4;
    # blocked_correctly = item 2. should_block count 2, not-should_block count 2.
    _, _, audit = _setup()
    before = len(audit.export())
    r = evaluate(_bundle(), [_fb(.95, False), _fb(.85, True), _fb(.92, True), _fb(.88, False)])
    assert r["n"] == 4 and r["missed"] == 1 and r["false_alarms"] == 1 and r["blocked_correctly"] == 1
    assert r["missed_detection_rate"] == pytest.approx(0.5)
    assert r["false_alarm_rate"] == pytest.approx(0.5)
    assert len(audit.export()) == before


def test_evaluate_empty_and_single_class_populations_give_zero():
    r = evaluate(_bundle(), [])
    assert r["n"] == 0 and r["missed_detection_rate"] == 0.0 and r["false_alarm_rate"] == 0.0
    # only healthy items: should_block population empty -> missed rate 0.0; one false alarm of 1 -> 1.0
    r = evaluate(_bundle(), [_fb(.80, False)])
    assert r["missed_detection_rate"] == 0.0 and r["false_alarm_rate"] == pytest.approx(1.0)


# ------------------------------------------------------------ construction

def test_calibrator_rejects_unknown_and_non_min_metric_rule():
    store, _, _ = _setup()
    with pytest.raises(NotFound):
        Calibrator(store, "nope")
    store2, _, _ = _setup(rules=[ACC(), Rule("lat", 1, "max_metric", {"metric": "lat", "max": 5.0})])
    with pytest.raises(ValidationFailed):
        Calibrator(store2, "lat")


@pytest.mark.parametrize("step", [0, -1, -0.001, float("nan"), float("inf"), True, "0.1", None])
def test_calibrator_rejects_bad_step(step):
    store, _, _ = _setup()
    with pytest.raises(BAD):
        Calibrator(store, "acc_floor", step=step)


# ------------------------------------------------------------------ propose

def test_propose_tighten_oracle():
    # active min .90; missed (allowed & should_block): .92, .94; .80 is blocked correctly; .96 healthy.
    # new = max(missed) + step = .94 + .001 = .941
    _, cal, _ = _setup()
    p = cal.propose([_fb(.92, True), _fb(.94, True), _fb(.80, True), _fb(.96, False)])
    assert isinstance(p, Proposal)
    assert p.kind == "tighten" and p.rule_id == "acc_floor" and p.metric == "acc"
    assert p.old_value == pytest.approx(0.90) and p.new_value == pytest.approx(0.941)
    assert p.missed_detections == 2 and p.false_alarms == 0


def test_propose_loosen_and_none():
    # no missed; false alarms at .88 and .89 (healthy but blocked) -> loosen to min = .88
    _, cal, _ = _setup()
    p = cal.propose([_fb(.88, False), _fb(.89, False), _fb(.80, True), _fb(.97, False)])
    assert p.kind == "loosen" and p.new_value == pytest.approx(0.88)
    assert p.old_value == pytest.approx(0.90) and p.missed_detections == 0 and p.false_alarms == 2
    # no evidence of either kind
    for fb in ([], [_fb(.95, False), _fb(.80, True)]):
        p = cal.propose(fb)
        assert p.kind == "none" and p.new_value == p.old_value == pytest.approx(0.90)


@pytest.mark.parametrize("bad_ctx", [
    {}, {"other": 1.0}, {"acc": float("nan")}, {"acc": float("inf")}, {"acc": True}, {"acc": "0.95"}, {"acc": None},
])
def test_propose_rejects_bad_feedback_contexts(bad_ctx):
    _, cal, _ = _setup()
    with pytest.raises(ValidationFailed):
        cal.propose([_fb(.95, False), Feedback(bad_ctx, True)])


# -------------------------------------------------------------------- apply

def test_apply_tighten_auto_activates_and_is_monotone():
    store, cal, audit = _setup()
    old_bundle = store.active()[1]
    p = cal.propose([_fb(.92, True), _fb(.94, True)])
    n_pub = _actions(audit).count("policy.publish")
    v = cal.apply("bot", p)  # no human approval needed
    assert isinstance(v, int) and v == 2
    ver, new_bundle = store.active()
    assert ver == v and _min(new_bundle) == pytest.approx(0.941)
    assert _min(store.get(1)) == pytest.approx(0.90)  # old version retrievable, unchanged
    assert _actions(audit).count("policy.publish") == n_pub + 1
    assert _actions(audit)[-1] == "policy.activate"
    assert is_stricter_or_equal(new_bundle, old_bundle) is True
    assert is_stricter_or_equal(old_bundle, new_bundle) is False


def test_apply_loosen_needs_human_approval():
    store, cal, audit = _setup()
    p = cal.propose([_fb(.88, False), _fb(.80, True)])
    assert p.kind == "loosen"
    n = len(audit.export())
    with pytest.raises(PolicyDenied):
        cal.apply("bot", p)
    with pytest.raises(PolicyDenied):
        cal.apply("bot", p, human_approved_by="")
    assert store.active()[0] == 1 and _active_min(store) == pytest.approx(0.90)
    assert len(audit.export()) == n  # nothing published on denial
    v = cal.apply("bot", p, human_approved_by="alice")
    assert store.active()[0] == v == 2 and _active_min(store) == pytest.approx(0.88)
    act = [e for e in audit.export() if e["action"] == "policy.activate"][-1]
    assert act["meta"]["loosening"] is True and act["meta"]["approved_by"] == "alice"


def test_apply_stale_proposal_conflicts():
    store, cal, _ = _setup()
    stale = cal.propose([_fb(.92, True)])            # old_value .90
    fresh = cal.propose([_fb(.95, True)])
    cal.apply("bot", fresh)                           # active min now .951
    with pytest.raises(Conflict):
        cal.apply("bot", stale)
    assert _active_min(store) == pytest.approx(0.951)


def test_apply_none_returns_none_and_changes_nothing():
    store, cal, audit = _setup()
    n = len(audit.export())
    assert cal.apply("bot", cal.propose([])) is None
    assert store.active()[0] == 1 and len(audit.export()) == n


# ------------------------------------------------------------ monotone teeth

def test_repeated_cycles_never_lower_threshold():
    store, cal, _ = _setup()
    seen = [_active_min(store)]
    for i in range(5):
        # growing missed set: accs .91, .915, ... (should_block=True) plus a healthy .99 item
        fb = [_fb(0.91 + 0.005 * k, True) for k in range(i + 1)] + [_fb(.99, False)]
        cal.apply("bot", cal.propose(fb))
        seen.append(_active_min(store))
    assert all(b >= a for a, b in zip(seen, seen[1:])), seen
    assert seen[-1] > seen[0]


def test_calibrator_does_not_trust_edited_proposal_kind():
    store, cal, audit = _setup()
    p = cal.propose([_fb(.88, False), _fb(.80, True)])       # genuine loosen, new .88
    forged = dataclasses.replace(p, kind="tighten")           # new_value .88 < old .90
    n = len(audit.export())
    with pytest.raises(ValidationFailed):
        cal.apply("bot", forged)
    with pytest.raises(ValidationFailed):                     # even with approval offered
        cal.apply("bot", forged, human_approved_by="alice")
    assert store.active()[0] == 1 and _active_min(store) == pytest.approx(0.90)
    assert len(audit.export()) == n


# ---------------------------------------------------------- G9 analog (OWN synthetic corpora)

def _corpus(seed):
    # OWN synthetic corpus (not the paper's): 100 healthy N(.95,.02) + 100 defective N(.91,.03), clipped to [0,1].
    rng = random.Random(seed)
    clip = lambda x: min(1.0, max(0.0, x))
    fb = [_fb(clip(rng.gauss(.95, .02)), False) for _ in range(100)]
    fb += [_fb(clip(rng.gauss(.91, .03)), True) for _ in range(100)]
    rng.shuffle(fb)
    return fb


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_g9_analog_tightening_reduces_missed_detections(seed):
    store, cal, _ = _setup()
    fb = _corpus(seed)
    before = evaluate(store.active()[1], fb)
    assert before["missed_detection_rate"] > 0.0
    cycles, converged = 0, False
    while cycles < 20:
        p = cal.propose(fb)
        if p.kind != "tighten":
            converged = True
            break
        cal.apply("bot", p)
        cycles += 1
    assert converged, "no convergence within 20 cycles"
    after = evaluate(store.active()[1], fb)
    assert after["missed_detection_rate"] <= before["missed_detection_rate"]
    # Tightening trades false alarms for detections: false alarms may rise, never fall.
    assert after["false_alarm_rate"] >= before["false_alarm_rate"]
    assert _active_min(store) >= 0.90
