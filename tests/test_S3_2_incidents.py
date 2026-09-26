"""S3.2 slim suite: incident classification, checklist, escalation, resolution.
Frozen BEFORE mlops.incidents exists. Time is driven by ManualClock(start=1000.0).
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.incidents import CHECKLIST, IncidentManager, Severity, classify
from mlops.kernel import Conflict, ManualClock, NotFound, ValidationFailed

INF = float("inf")
NAN = float("nan")
STEPS = ["detection", "impact_assessment", "recent_changes_review", "mitigation_options", "root_cause"]
PM_KEYS = ["timeline", "root_cause", "user_impact", "preventive_measures", "monitoring_gap"]
P1_SIGNAL = {"accuracy_drop_pct": 15.0}
P3_SIGNAL = {"pipeline_delay": True}


def _pm():
    return {k: f"text for {k}" for k in PM_KEYS}


def _mk():
    clock = ManualClock(start=1000.0)
    audit = ChainedAuditStore(b"k")
    return IncidentManager(audit, clock), audit, clock


def _actions(audit):
    return [e["action"] for e in audit.export()]


def _open(mgr, signal=None, title="t"):
    return mgr.open("alice", title, P1_SIGNAL if signal is None else signal)


def _do_steps(mgr, iid, n=5):
    for s in STEPS[:n]:
        mgr.complete_step("alice", iid, s, f"note {s}")


def _sev(rec):
    v = rec["severity"]
    return getattr(v, "name", v)


def test_severity_response_minutes():
    assert [Severity[n].response_minutes for n in ("P0", "P1", "P2", "P3")] == [15, 60, 240, 1440]
    assert CHECKLIST == tuple(STEPS)


@pytest.mark.parametrize("signal,expected", [
    ({"serving_failure": True}, "P0"),
    ({"null_predictions": True}, "P0"),
    ({"accuracy_drop_pct": 10.01}, "P1"),
    ({"accuracy_drop_pct": 10.0}, "P3"),
    ({"accuracy_drop_pct": 0.5}, "P3"),
    ({"feature_psi": 0.31}, "P2"),
    ({"feature_psi": 0.3}, "P3"),
    ({"drift_level": "alert"}, "P2"),
    ({"drift_level": "warn"}, "P3"),
    ({"pipeline_delay": True}, "P3"),
    ({"serving_failure": True, "drift_level": "alert"}, "P0"),
    ({"accuracy_drop_pct": 11, "feature_psi": 0.9, "pipeline_delay": True}, "P1"),
    ({"feature_psi": 0.5, "drift_level": "warn"}, "P2"),
])
def test_classify_table(signal, expected):
    assert classify(signal).name == expected


@pytest.mark.parametrize("bad", [
    {}, {"unknown": True}, {"serving_failure": True, "bogus": 1},
    {"serving_failure": 1}, {"null_predictions": "yes"}, {"pipeline_delay": None},
    {"accuracy_drop_pct": -1.0}, {"accuracy_drop_pct": NAN}, {"accuracy_drop_pct": INF},
    {"accuracy_drop_pct": True}, {"accuracy_drop_pct": "5"},
    {"feature_psi": -0.1}, {"feature_psi": NAN}, {"feature_psi": INF}, {"feature_psi": False},
    {"drift_level": "critical"}, {"drift_level": 1}, {"drift_level": None},
    None, [], "serving_failure", 5,
])
def test_classify_rejects_invalid(bad):
    with pytest.raises(ValidationFailed):
        classify(bad)


def test_open_record_and_audit():
    mgr, audit, _ = _mk()
    iid = _open(mgr, title="bad model")
    rec = mgr.get(iid)
    assert _sev(rec) == "P1"
    assert rec["opened_ts"] == 1000.0
    assert rec["deadline"] == 4600.0
    assert _actions(audit) == ["incident.open"]
    iid2 = mgr.open("bob", "x", {"serving_failure": True})
    assert iid2 != iid
    assert mgr.get(iid2)["deadline"] == 1000.0 + 15 * 60


@pytest.mark.parametrize("actor,title,signal", [
    ("", "t", P1_SIGNAL), (None, "t", P1_SIGNAL), (5, "t", P1_SIGNAL),
    ("alice", "", P1_SIGNAL), ("alice", None, P1_SIGNAL), ("alice", 7, P1_SIGNAL),
    ("alice", "t", {}), ("alice", "t", {"nope": 1}),
])
def test_open_invalid_writes_nothing(actor, title, signal):
    mgr, audit, _ = _mk()
    with pytest.raises(ValidationFailed):
        mgr.open(actor, title, signal)
    assert audit.export() == []


def test_steps_in_order_and_audited():
    mgr, audit, _ = _mk()
    iid = _open(mgr)
    _do_steps(mgr, iid)
    assert len(audit.export()) == 6  # open + 5 steps; step action name is not frozen


@pytest.mark.parametrize("first", ["root_cause", "mitigation_options", "impact_assessment"])
def test_out_of_order_conflict_changes_nothing(first):
    mgr, audit, _ = _mk()
    iid = _open(mgr)
    before, n = mgr.get(iid), len(audit.export())
    with pytest.raises(Conflict):
        mgr.complete_step("alice", iid, first, "n")
    assert mgr.get(iid) == before and len(audit.export()) == n
    _do_steps(mgr, iid, 2)
    n, before = len(audit.export()), mgr.get(iid)
    with pytest.raises(Conflict):  # skipping recent_changes_review
        mgr.complete_step("alice", iid, "mitigation_options", "n")
    with pytest.raises(Conflict):  # repeat
        mgr.complete_step("alice", iid, "detection", "n")
    assert mgr.get(iid) == before and len(audit.export()) == n


def test_complete_step_validation_and_notfound():
    mgr, audit, _ = _mk()
    iid = _open(mgr)
    n = len(audit.export())
    for step, note, actor in [("bogus", "n", "alice"), ("", "n", "alice"), (None, "n", "alice"),
                              ("detection", "", "alice"), ("detection", None, "alice"),
                              ("detection", "n", ""), ("detection", "n", None)]:
        with pytest.raises(ValidationFailed):
            mgr.complete_step(actor, iid, step, note)
    with pytest.raises(NotFound):
        mgr.complete_step("alice", "inc-missing", "detection", "n")
    assert len(audit.export()) == n
    mgr.complete_step("alice", iid, "detection", "n")  # still works in order


def test_escalation_ladder_p1():
    mgr, audit, clock = _mk()
    iid = _open(mgr)
    assert mgr.escalate_if_overdue(iid) == 1
    clock.advance(3599.0)  # 4599
    assert mgr.escalate_if_overdue(iid) == 1
    clock.advance(1.0)  # 4600: equality is not overdue
    assert mgr.escalate_if_overdue(iid) == 1
    assert _actions(audit).count("incident.escalate") == 0
    clock.advance(1.0)  # 4601
    assert mgr.escalate_if_overdue(iid) == 2
    assert mgr.escalate_if_overdue(iid) == 2
    assert _actions(audit).count("incident.escalate") == 1
    clock.advance(8201.0 - 4601.0 - 1.0)  # 8200: not yet 2x deadline
    assert mgr.escalate_if_overdue(iid) == 2
    clock.advance(1.0)  # 8201
    assert mgr.escalate_if_overdue(iid) == 3
    assert mgr.escalate_if_overdue(iid) == 3
    assert _actions(audit).count("incident.escalate") == 2
    clock.advance(10 ** 6)
    assert mgr.escalate_if_overdue(iid) == 3
    assert _actions(audit).count("incident.escalate") == 2


def test_escalate_unknown_and_resolved_final_tier():
    mgr, audit, clock = _mk()
    with pytest.raises(NotFound):
        mgr.escalate_if_overdue("inc-missing")
    iid = _open(mgr, P3_SIGNAL)
    clock.advance(1440 * 60 + 1)
    assert mgr.escalate_if_overdue(iid) == 2
    _do_steps(mgr, iid)
    mgr.resolve("alice", iid, None)
    n = len(audit.export())
    clock.advance(10 ** 7)
    assert mgr.escalate_if_overdue(iid) == 2
    assert len(audit.export()) == n


def test_resolve_requires_all_steps():
    mgr, audit, _ = _mk()
    iid = _open(mgr, P3_SIGNAL)
    for k in range(5):
        n = len(audit.export())
        with pytest.raises(Conflict):
            mgr.resolve("alice", iid, None)
        assert mgr.get(iid)["status"] != "resolved" and len(audit.export()) == n
        mgr.complete_step("alice", iid, STEPS[k], "n")
    mgr.resolve("alice", iid, None)
    assert mgr.get(iid)["status"] == "resolved"


def _bad_postmortems():
    out = [None, [], "text", {}]
    for k in PM_KEYS:
        pm = _pm()
        del pm[k]
        out.append(pm)
        pm = _pm()
        pm[k] = ""
        out.append(pm)
        pm = _pm()
        pm[k] = 5
        out.append(pm)
    return out


@pytest.mark.parametrize("signal", [{"serving_failure": True}, P1_SIGNAL])
def test_p0_p1_need_complete_postmortem(signal):
    mgr, audit, _ = _mk()
    iid = _open(mgr, signal)
    _do_steps(mgr, iid)
    n = len(audit.export())
    for bad in _bad_postmortems():
        with pytest.raises(ValidationFailed):
            mgr.resolve("alice", iid, bad)
    assert mgr.get(iid)["status"] != "resolved" and len(audit.export()) == n
    mgr.resolve("alice", iid, _pm())
    assert mgr.get(iid)["status"] == "resolved"


@pytest.mark.parametrize("signal", [{"feature_psi": 0.5}, P3_SIGNAL])
def test_p2_p3_resolve_without_postmortem(signal):
    mgr, _, _ = _mk()
    iid = _open(mgr, signal)
    _do_steps(mgr, iid)
    mgr.resolve("alice", iid, None)
    rec = mgr.get(iid)
    assert rec["status"] == "resolved"


def test_postmortem_deep_copied_and_double_resolve():
    mgr, audit, _ = _mk()
    iid = _open(mgr)
    _do_steps(mgr, iid)
    pm = _pm()
    mgr.resolve("alice", iid, pm)
    pm["timeline"] = "MUTATED"
    assert mgr.get(iid)["postmortem"] == _pm()
    n = len(audit.export())
    with pytest.raises(Conflict):
        mgr.resolve("alice", iid, _pm())
    with pytest.raises(Conflict):
        mgr.complete_step("alice", iid, "detection", "n")
    assert len(audit.export()) == n
    with pytest.raises(NotFound):
        mgr.resolve("alice", "inc-missing", None)


def test_resolve_invalid_actor():
    mgr, audit, _ = _mk()
    iid = _open(mgr, P3_SIGNAL)
    _do_steps(mgr, iid)
    n = len(audit.export())
    for actor in ("", None, 3):
        with pytest.raises(ValidationFailed):
            mgr.resolve(actor, iid, None)
    assert len(audit.export()) == n


def test_get_returns_copy_and_notfound():
    mgr, _, _ = _mk()
    iid = _open(mgr)
    rec = mgr.get(iid)
    snapshot = copy.deepcopy(rec)
    rec["deadline"] = -1
    rec["status"] = "hacked"
    rec["severity"] = "P3"
    rec.clear()
    assert mgr.get(iid) == snapshot
    with pytest.raises(NotFound):
        mgr.get("inc-missing")


def test_full_lifecycle_audit_chain_and_counts():
    mgr, audit, clock = _mk()
    iid = _open(mgr)
    assert len(audit.export()) == 1
    for k, s in enumerate(STEPS):
        mgr.complete_step("alice", iid, s, "n")
        assert len(audit.export()) == 2 + k
    clock.advance(3601.0)
    mgr.escalate_if_overdue(iid)
    mgr.escalate_if_overdue(iid)
    assert len(audit.export()) == 7
    mgr.resolve("alice", iid, _pm())
    export = audit.export()
    assert len(export) == 8
    acts = [e["action"] for e in export]
    assert acts[0] == "incident.open" and acts[6] == "incident.escalate"
    assert acts.count("incident.escalate") == 1
    assert ChainVerifier(b"k").verify_export(export, head=audit.head())
