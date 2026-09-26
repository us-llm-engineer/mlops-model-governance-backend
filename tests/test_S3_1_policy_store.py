"""S3.1 slim suite: mlops.policy_store (PolicyStore, is_stricter_or_equal).
Oracle-based, hand-computed expectations; frozen BEFORE the module exists.
Contract: the API contract, section policy_store.py (G).
"""
import os
import signal
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.kernel import Conflict, MlopsError, NotFound, PolicyDenied, ValidationFailed
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore, is_stricter_or_equal

SECRET = b"s3-1-secret"
BAD = (ValidationFailed, ValueError, TypeError)
BADVER = (ValidationFailed, NotFound, ValueError, TypeError)


def _within(seconds, fn):
    """Run fn(); a hang raises TimeoutError instead of blocking the suite (POSIX)."""
    def _h(signum, frame):
        raise TimeoutError("timed out after %ss" % seconds)
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def R(rid, kind, params, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params, severity=sev)


def MINR(v, rid="acc", metric="acc", sev="block"):
    return R(rid, "min_metric", {"metric": metric, "min": v}, sev)


def MAXR(v, rid="lat", metric="lat", sev="block"):
    return R(rid, "max_metric", {"metric": metric, "max": v})


def B(*rules, name="p"):
    return PolicyBundle(rules=list(rules), name=name, version=0)


def store():
    audit = ChainedAuditStore(SECRET)
    return PolicyStore(audit), audit


def acts(audit, action):
    return [e for e in audit.export() if e["action"] == action]


def two_versions(strict_first=False):
    """v1/v2 where 'acc' min differs: loose 0.8, strict 0.9."""
    st, audit = store()
    lo, hi = B(MINR(0.8)), B(MINR(0.9))
    if strict_first:
        st.publish("alice", hi), st.publish("alice", lo)
    else:
        st.publish("alice", lo), st.publish("alice", hi)
    return st, audit


# ---------------------------------------------------------------- (1) ordering
STRICT_TABLE = [
    # (new, old, expected, why)
    (B(MINR(0.95)), B(MINR(0.9)), True, "min 0.95>=0.9"),
    (B(MINR(0.9)), B(MINR(0.9)), True, "min equal"),
    (B(MINR(0.8)), B(MINR(0.9)), False, "min 0.8<0.9 loosens"),
    (B(MAXR(5)), B(MAXR(10)), True, "max 5<=10"),
    (B(MAXR(20)), B(MAXR(10)), False, "max 20>10 loosens"),
    (B(MINR(0.9, sev="warn")), B(MINR(0.9, sev="block")), False, "block->warn"),
    (B(MINR(0.9, sev="block")), B(MINR(0.9, sev="warn")), True, "warn->block"),
    (B(), B(MINR(0.9)), False, "block rule removed"),
    (B(), B(MINR(0.9, sev="warn")), True, "warn rule removed is fine"),
    (B(MINR(0.9), MAXR(10)), B(MINR(0.9)), True, "rule added"),
    (B(R("acc", "max_metric", {"metric": "acc", "max": 1})), B(MINR(0.9)), False, "kind changed"),
    (B(R("f", "flag_true", {"key": "ok"})), B(R("f", "flag_true", {"key": "ok"})), True, "flag equal"),
    (B(R("f", "flag_true", {"key": "other"})), B(R("f", "flag_true", {"key": "ok"})), False, "flag differs"),
]


@pytest.mark.parametrize("new,old,expected,why", STRICT_TABLE, ids=[t[3] for t in STRICT_TABLE])
def test_is_stricter_or_equal_table(new, old, expected, why):
    assert is_stricter_or_equal(new, old) is expected


# ---------------------------------------------------------------- (2) publish
def test_publish_versions_and_audit():
    st, audit = store()
    vs = [st.publish("alice", B(MINR(0.5 + i / 10)), note="n%d" % i) for i in range(3)]
    assert vs == [1, 2, 3] and all(type(v) is int for v in vs)
    entries = acts(audit, "policy.publish")
    assert len(entries) == 3  # exactly one per call
    assert all(e["actor"] == "alice" for e in entries)
    assert [h["version"] for h in st.history()] == [1, 2, 3]
    assert [h["note"] for h in st.history()] == ["n0", "n1", "n2"]


def test_publish_immutable_and_get_returns_copy():
    st, _ = store()
    b = B(MINR(0.9))
    v = st.publish("alice", b)
    b.rules[0].params["min"] = 0.1  # caller mutates its own object afterwards
    b.rules.append(MAXR(1))
    stored = st.get(v)
    assert len(stored.rules) == 1 and stored.rules[0].params["min"] == 0.9
    stored.rules[0].params["min"] = 0.0  # mutate the returned copy
    stored.rules.clear()
    again = st.get(v)
    assert len(again.rules) == 1 and again.rules[0].params["min"] == 0.9


@pytest.mark.parametrize("actor,bundle", [
    ("", B(MINR(0.9))), (None, B(MINR(0.9))), (5, B(MINR(0.9))),
    ("alice", None), ("alice", "bundle"), ("alice", {"rules": []}),
])
def test_publish_invalid_consumes_no_version(actor, bundle):
    st, audit = store()
    with pytest.raises(BAD):
        st.publish(actor, bundle)
    assert audit.export() == []
    assert st.publish("alice", B(MINR(0.9))) == 1  # no version was burned


# --------------------------------------------------------------- (3) activate
def test_activate_first_needs_no_approval_and_stricter_ok_and_meta():
    st, audit = two_versions()
    with pytest.raises(NotFound):
        st.active()  # nothing active yet
    st.activate("alice", 1)  # first activation, no approver
    assert st.active()[0] == 1
    st.activate("alice", 2)  # 0.9 >= 0.8 stricter, no approval needed
    ver, bundle = st.active()
    assert ver == 2 and bundle.to_dict() == st.get(2).to_dict()
    m = acts(audit, "policy.activate")[-1]["meta"]
    assert m["version"] == 2 and m["previous"] == 1 and m["loosening"] is False
    assert [h["active"] for h in st.history()] == [False, True]


@pytest.mark.parametrize("approver", [None, ""])
def test_loosening_without_approval_denied_state_unchanged(approver):
    # TEETH: an activate() that skipped the loosening check would let v1 (0.8) replace v2 (0.9).
    st, audit = two_versions()
    st.activate("alice", 2)
    n = len(acts(audit, "policy.activate"))
    with pytest.raises(PolicyDenied):
        st.activate("alice", 1, human_approved_by=approver)
    assert st.active()[0] == 2
    assert len(acts(audit, "policy.activate")) == n  # no success entry written


def test_loosening_with_approval_succeeds_and_records():
    st, audit = two_versions()
    st.activate("alice", 2)
    st.activate("alice", 1, human_approved_by="carol")
    assert st.active()[0] == 1
    m = acts(audit, "policy.activate")[-1]["meta"]
    assert m["loosening"] is True and m["approved_by"] == "carol"
    assert m["version"] == 1 and m["previous"] == 2


def test_activate_unknown_version_notfound():
    st, _ = two_versions()
    st.activate("alice", 1)
    with pytest.raises(NotFound):
        st.activate("alice", 99)
    assert st.active()[0] == 1


@pytest.mark.parametrize("bad", [True, 1.5, "2", 0, -1])
def test_activate_bad_version_types(bad):
    st, _ = two_versions()
    st.activate("alice", 1)
    with pytest.raises(BADVER):
        st.activate("alice", bad)
    assert st.active()[0] == 1  # bool True must not be treated as version 1 / anything else


# ------------------------------------------------------- (4) atomic activation
def test_concurrent_activation_never_tears():
    st, _ = store()
    st.publish("alice", B(MINR(0.9), name="A"))  # v1
    st.publish("alice", B(MINR(0.9), name="B"))  # v2: identical rules, so either direction is "equal"
    st.activate("alice", 1)
    names = {1: "A", 2: "B"}
    errors, stop = [], threading.Event()

    def writer(seed):
        try:
            for i in range(60):
                st.activate("alice", 1 + ((i + seed) % 2))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                ver, bundle = st.active()
                d = bundle.to_dict()
                if d != st.get(ver).to_dict() or d["name"] != names[ver]:
                    errors.append(("torn", ver, d["name"]))
                    return
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def run():
        ws = [threading.Thread(target=writer, args=(s,)) for s in range(8)]
        r = threading.Thread(target=reader)
        r.start()
        [t.start() for t in ws]
        [t.join() for t in ws]
        stop.set()
        r.join()
    _within(20, run)
    assert errors == []


# --------------------------------------------------------------- (5) rollback
def test_rollback_returns_previous_and_audits():
    st, audit = two_versions()
    st.activate("alice", 1)
    st.activate("alice", 2)
    with pytest.raises(PolicyDenied):  # v1 (0.8) is looser than v2 (0.9)
        st.rollback("alice")
    assert st.active()[0] == 2 and acts(audit, "policy.rollback") == []
    assert st.rollback("alice", human_approved_by="carol") == 1
    assert st.active()[0] == 1
    assert len(acts(audit, "policy.rollback")) == 1


def test_rollback_stricter_needs_no_approval():
    st, audit = two_versions(strict_first=True)  # v1 = 0.9 strict, v2 = 0.8 loose
    st.activate("alice", 1)
    st.activate("alice", 2, human_approved_by="carol")  # loosening, approved
    assert st.rollback("alice") == 1  # back to strict: no approver required
    assert st.active()[0] == 1
    assert len(acts(audit, "policy.rollback")) == 1


def test_rollback_without_previous_conflict():
    st, _ = two_versions()
    with pytest.raises(Conflict):
        st.rollback("alice")  # nothing ever active
    st.activate("alice", 1)
    with pytest.raises(Conflict):
        st.rollback("alice")  # only one activation so far: no previous
    assert st.active()[0] == 1


# ------------------------------------------------------------------ (6) decide
def test_decide_fail_closed_without_active_bundle():
    st, audit = two_versions()
    with pytest.raises(PolicyDenied):
        st.decide("promote", {"acc": 0.99})
    assert st.decision_log() == []


def test_decide_logs_deep_copied_context_and_appends_chain():
    st, audit = two_versions()
    st.activate("alice", 2)  # acc >= 0.9
    n = len(audit.export())
    ctx = {"acc": 0.95, "tags": ["a"]}
    d = st.decide("promote", ctx)
    assert d.allow is True and len(audit.export()) == n + 1
    ctx["acc"] = 0.0
    ctx["tags"].append("mutated")
    d2 = st.decide("promote", {"acc": 0.85})  # 0.85 < 0.9 -> deny
    assert d2.allow is False and len(audit.export()) == n + 2
    log = st.decision_log()
    assert len(log) == 2
    e = log[0]
    assert e["action"] == "promote" and e["allow"] is True and e["version"] == 2
    assert e["context"] == {"acc": 0.95, "tags": ["a"]}  # deep copy taken at call time
    assert e["policy_hash"] == d.policy_hash == st.get(2).hash
    assert log[1]["allow"] is False


def test_decision_log_bounded_keeps_newest():
    st, _ = store()
    st.publish("alice", B(R("ok", "flag_true", {"key": "ok"})))
    st.activate("alice", 1)

    def run():
        for i in range(10050):
            st.decide("a", {"ok": True, "i": i})
    _within(30, run)
    log = st.decision_log()
    assert len(log) == 10000
    assert log[-1]["context"]["i"] == 10049  # newest kept
    assert log[0]["context"]["i"] == 50  # 10050 - 10000 oldest dropped


# ----------------------------------------------------------------- (7) replay
DECS = [
    {"action": "promote", "context": {"acc": 0.85}, "allow": True},   # v1 (0.8) allow; v2 (0.9) deny
    {"action": "promote", "context": {"acc": 0.95}, "allow": True},   # allow under both
    {"action": "promote", "context": {"acc": 0.5}, "allow": False},   # deny under both
]


def test_replay_diff_oracle_ordered_and_deterministic():
    st, _ = two_versions()
    out = st.replay(2, DECS)
    assert [o["index"] for o in out] == [0, 1, 2]
    assert [o["action"] for o in out] == ["promote"] * 3
    assert [o["recorded_allow"] for o in out] == [True, True, False]
    assert [o["replay_allow"] for o in out] == [False, True, False]
    assert [o["changed"] for o in out] == [True, False, False]
    assert st.replay(2, DECS) == out
    # under v1 (0.8) all recorded outcomes reproduce: 0.85 ok, 0.95 ok, 0.5 deny
    assert [o["changed"] for o in st.replay(1, DECS)] == [False, False, False]


def test_replay_from_log_writes_nothing_to_chain():
    # TEETH: a replay that appended to the chain (e.g. via a PolicyDecisionPoint with audit) fails here.
    st, audit = two_versions()
    st.activate("alice", 1)
    st.decide("promote", {"acc": 0.85})  # allowed under v1
    before = audit.export()
    out = st.replay(2)  # decisions=None -> uses the log
    assert len(out) == 1
    assert (out[0]["recorded_allow"], out[0]["replay_allow"], out[0]["changed"]) == (True, False, True)
    st.replay(2, DECS)
    assert audit.export() == before and len(st.decision_log()) == 1


def test_replay_errors():
    st, audit = two_versions()
    n = len(audit.export())
    with pytest.raises(NotFound):
        st.replay(99, DECS)
    for bad in (
        "nope", [None], [{"action": "p", "context": {"acc": 1}}],  # missing allow
        [{"action": "p", "context": [], "allow": True}],  # context not dict
        [{"action": "p", "context": {}, "allow": "yes"}],  # allow not bool
        [{"action": "", "context": {}, "allow": True}],  # empty action
    ):
        with pytest.raises(BAD):
            st.replay(1, bad)
    assert len(audit.export()) == n


# ------------------------------------------------------------------- (8) chain
def test_whole_chain_verifies_after_mixed_sequence():
    st, audit = two_versions()
    st.activate("alice", 1)
    st.activate("alice", 2)
    st.decide("promote", {"acc": 0.95})
    st.decide("promote", {"acc": 0.1})
    st.replay(1, DECS)
    with pytest.raises(PolicyDenied):
        st.activate("alice", 1)
    st.rollback("alice", human_approved_by="carol")
    st.publish("bob", B(MINR(0.99)))
    export = audit.export()
    # 2 publish + 2 activate + 2 decide + 1 rollback + 1 publish = 8 (denied activate writes none)
    assert len(export) >= 8
    assert ChainVerifier(SECRET).verify_export(export, head=audit.head())
