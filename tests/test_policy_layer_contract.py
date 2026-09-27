"""Contract suite for the raw policy layer (policy_engine + policy_store).

Every expectation is hand-derived from docs/POLICY-CONTRACT.md, not copied from the
implementation. Boundary and matrix style: each threshold is probed at, just below and
just above; each invalid-number class is probed individually. Guarded by hand-mutation
(see inboxes/.../workflows/policy-verification-recipe.md).
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.kernel import Conflict, NotFound, PolicyDenied, ValidationFailed
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule
from mlops.policy_store import PolicyStore, is_stricter_or_equal

NAN, INF = float("nan"), float("inf")


def R(rid, kind, params, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params, severity=sev)


def B(*rules, name="p", version=0):
    return PolicyBundle(rules=list(rules), name=name, version=version)


def allow(rule, ctx, action="a"):
    return PolicyDecisionPoint(bundle=B(rule)).decide(action, ctx).allow


# ---------------------------------------------------------------- numeric boundaries
@pytest.mark.parametrize("kind,pkey,ctxkey", [
    ("min_metric", "min", "metric"),
    ("max_metric", "max", "metric"),
    ("budget_ok", "max", "key"),
])
def test_threshold_boundary_at_below_above(kind, pkey, ctxkey):
    rule = R("r", kind, {ctxkey: "x", pkey: 0.5})
    at, below, above = allow(rule, {"x": 0.5}), allow(rule, {"x": 0.4999}), allow(rule, {"x": 0.5001})
    if kind == "min_metric":
        assert (at, below, above) == (True, False, True)
    else:
        assert (at, below, above) == (True, True, False)


@pytest.mark.parametrize("kind,pkey,ctxkey", [
    ("min_metric", "min", "metric"), ("max_metric", "max", "metric"), ("budget_ok", "max", "key"),
])
def test_int_and_float_compare_numerically(kind, pkey, ctxkey):
    rule = R("r", kind, {ctxkey: "x", pkey: 1})
    assert allow(rule, {"x": 1}) is True
    assert allow(rule, {"x": 1.0}) is True


def test_zero_and_negative_thresholds():
    assert allow(R("r", "min_metric", {"metric": "x", "min": 0}), {"x": 0}) is True
    assert allow(R("r", "min_metric", {"metric": "x", "min": 0}), {"x": -1}) is False
    assert allow(R("r", "max_metric", {"metric": "x", "max": -2}), {"x": -2}) is True
    assert allow(R("r", "max_metric", {"metric": "x", "max": -2}), {"x": -1}) is False


INVALID_NUMBERS = [True, False, "1", None, NAN, INF, -INF, [1], {"v": 1}]


@pytest.mark.parametrize("kind,params", [
    ("min_metric", {"metric": "x", "min": -1e9}),
    ("max_metric", {"metric": "x", "max": 1e9}),
    ("budget_ok", {"key": "x", "max": 1e9}),
])
@pytest.mark.parametrize("bad", INVALID_NUMBERS, ids=repr)
def test_metric_kinds_reject_non_finite_or_non_number(kind, params, bad):
    # thresholds are so loose any real number would pass; only validity can fail it
    assert allow(R("r", kind, params), {"x": bad}) is False


@pytest.mark.parametrize("kind,params", [
    ("min_metric", {"metric": "x", "min": 0}),
    ("max_metric", {"metric": "x", "max": 0}),
    ("budget_ok", {"key": "x", "max": 0}),
    ("flag_true", {"key": "x"}),
    ("flag_false", {"key": "x"}),
    ("status_equals", {"key": "x", "value": "v"}),
    ("role_in", {"key": "x", "roles": ["v"]}),
    ("no_nan", {"keys": ["x"]}),
])
def test_missing_key_fails_closed(kind, params):
    assert allow(R("r", kind, params), {"other": 0}) is False


# ---------------------------------------------------------------- flags, status, role, no_nan
def test_flag_true_is_identity_not_truthiness():
    r = R("r", "flag_true", {"key": "f"})
    assert allow(r, {"f": True}) is True
    for bad in (1, "true", None, [1], 0):
        assert allow(r, {"f": bad}) is False


def test_flag_false_is_identity_not_falsiness():
    r = R("r", "flag_false", {"key": "f"})
    assert allow(r, {"f": False}) is True
    for bad in (0, "", None, [], "false", True):
        assert allow(r, {"f": bad}) is False


def test_status_equals_exact_and_case_sensitive():
    r = R("r", "status_equals", {"key": "s", "value": "ok"})
    assert allow(r, {"s": "ok"}) is True
    for bad in ("OK", "ok ", "okay", "", None, 1):
        assert allow(r, {"s": bad}) is False


def test_role_in_membership():
    r = R("r", "role_in", {"key": "role", "roles": ["admin", "operator"]})
    assert allow(r, {"role": "admin"}) is True
    assert allow(r, {"role": "operator"}) is True
    for bad in ("viewer", "Admin", "", None):
        assert allow(r, {"role": bad}) is False


def test_no_nan_all_keys_must_be_present_and_finite():
    r = R("r", "no_nan", {"keys": ["a", "b"]})
    assert allow(r, {"a": 1, "b": 0.0}) is True
    assert allow(r, {"a": 1}) is False                       # b missing
    for bad in (NAN, INF, -INF, None, True, "1"):
        assert allow(r, {"a": 1, "b": bad}) is False
        assert allow(r, {"a": bad, "b": 1}) is False


# ---------------------------------------------------------------- effect / severity
def test_all_block_rules_must_pass_and_reason_names_the_failure():
    b = B(R("acc", "min_metric", {"metric": "acc", "min": 0.9}),
          R("lat", "max_metric", {"metric": "lat", "max": 100}))
    d = PolicyDecisionPoint(bundle=b).decide("promote", {"acc": 0.95, "lat": 150})
    assert d.allow is False
    assert d.reasons == ["Rule lat (max_metric) failed"]
    ok = PolicyDecisionPoint(bundle=b).decide("promote", {"acc": 0.95, "lat": 100})
    assert ok.allow is True and ok.reasons == ["All policy rules passed"]


def test_warn_failure_never_blocks_but_is_reported():
    b = B(R("w", "min_metric", {"metric": "acc", "min": 0.9}, "warn"),
          R("f", "flag_true", {"key": "go"}))
    d = PolicyDecisionPoint(bundle=b).decide("x", {"acc": 0.1, "go": True})
    assert d.allow is True
    assert {r["id"]: r["passed"] for r in d.rule_results} == {"w": False, "f": True}
    assert d.reasons == ["All policy rules passed"]


def test_only_warn_rules_all_failing_still_allows():
    b = B(R("w1", "flag_true", {"key": "a"}, "warn"), R("w2", "flag_true", {"key": "b"}, "warn"))
    assert PolicyDecisionPoint(bundle=b).decide("x", {}).allow is True


def test_block_failure_with_passing_warn_denies():
    b = B(R("w", "flag_true", {"key": "a"}, "warn"), R("blk", "flag_true", {"key": "b"}))
    assert PolicyDecisionPoint(bundle=b).decide("x", {"a": True, "b": False}).allow is False


def test_rule_results_carry_id_kind_severity_passed_in_bundle_order():
    b = B(R("z", "flag_true", {"key": "a"}), R("a", "flag_false", {"key": "b"}, "warn"))
    d = PolicyDecisionPoint(bundle=b).decide("x", {"a": True, "b": True})
    assert d.rule_results == [
        {"id": "z", "kind": "flag_true", "severity": "block", "passed": True, "applicable": True},
        {"id": "a", "kind": "flag_false", "severity": "warn", "passed": False, "applicable": True},
    ]


def test_empty_and_absent_bundle_fail_closed_unless_allow_empty():
    assert PolicyDecisionPoint(bundle=B()).decide("x", {}).allow is False
    assert PolicyDecisionPoint(bundle=None).decide("x", {}).allow is False
    assert PolicyDecisionPoint(bundle=B(), allow_empty=True).decide("x", {}).allow is True
    assert PolicyDecisionPoint(bundle=None, allow_empty=True).decide("x", {}).allow is True


def test_unknown_kind_and_broken_params_fail_closed():
    assert allow(R("r", "made_up", {"key": "x"}), {"x": True}) is False
    assert allow(R("r", "flag_true", None), {"x": True}) is False
    assert allow(R("r", "min_metric", {"metric": "x"}), {"x": 1.0}) is False  # min missing


def test_action_is_a_label_only_in_the_unscoped_layer():
    b = B(R("f", "flag_true", {"key": "go"}))
    dp = PolicyDecisionPoint(bundle=b)
    assert dp.decide("promote", {"go": True}).allow is dp.decide("anything.else", {"go": True}).allow
    assert dp.decide("promote", {"go": True}).action == "promote"


def test_enforce_returns_decision_or_raises_policy_denied():
    dp = PolicyDecisionPoint(bundle=B(R("f", "flag_true", {"key": "go"})))
    assert dp.enforce("x", {"go": True}).allow is True
    with pytest.raises(PolicyDenied) as e:
        dp.enforce("x", {"go": False})
    assert "Rule f (flag_true) failed" in str(e.value)


# ---------------------------------------------------------------- hash
def test_bundle_hash_is_order_independent_and_param_sensitive():
    a, b = R("a", "flag_true", {"key": "k"}), R("b", "min_metric", {"metric": "m", "min": 1})
    assert B(a, b).hash == B(b, a).hash
    assert B(a, b).hash != B(a, R("b", "min_metric", {"metric": "m", "min": 2})).hash
    assert B(a, b).hash != B(a, b, name="other").hash
    assert B(a, b).hash != B(a, b, version=1).hash
    assert B(a, b).hash != B(a, R("b", "min_metric", {"metric": "m", "min": 1}, "warn")).hash


def test_decision_carries_the_bundle_hash():
    b = B(R("a", "flag_true", {"key": "k"}))
    assert PolicyDecisionPoint(bundle=b).decide("x", {"k": True}).policy_hash == b.hash


# ---------------------------------------------------------------- audit trail
def test_exactly_one_audit_entry_per_decision_with_resource_fallback():
    audit = ChainedAuditStore(b"contract-secret")
    dp = PolicyDecisionPoint(bundle=B(R("f", "flag_true", {"key": "go"}),
                                       R("w", "flag_true", {"key": "w"}, "warn")), audit=audit)
    dp.decide("promote", {"go": True, "w": True, "resource": "model-1"}, actor="alice")
    dp.decide("promote", {"go": False, "w": False})
    e1, e2 = audit.export()
    assert (e1["actor"], e1["action"], e1["resource"], e1["decision"]) == ("alice", "policy.promote", "model-1", "allow")
    assert (e2["resource"], e2["decision"]) == ("promote", "deny")        # falls back to the action
    assert e2["meta"]["failed"] == ["f"]                                  # warn failures are not listed
    assert e1["meta"]["failed"] == []


# ---------------------------------------------------------------- loosening guard matrix
MIN = lambda v: R("m", "min_metric", {"metric": "acc", "min": v})
MAX = lambda v: R("m", "max_metric", {"metric": "lat", "max": v})


@pytest.mark.parametrize("new,old,expected", [
    (B(MIN(0.9)), B(MIN(0.8)), True),                       # higher min is stricter
    (B(MIN(0.8)), B(MIN(0.8)), True),                       # equal
    (B(MIN(0.7)), B(MIN(0.8)), False),                      # lower min loosens
    (B(MAX(50)), B(MAX(100)), True),                        # lower max is stricter
    (B(MAX(100)), B(MAX(100)), True),
    (B(MAX(150)), B(MAX(100)), False),                      # higher max loosens
    (B(R("m", "min_metric", {"metric": "acc", "min": 0.8}, "warn")),
     B(MIN(0.8)), False),                                   # block -> warn loosens
    (B(MIN(0.8)), B(R("m", "min_metric", {"metric": "acc", "min": 0.8}, "warn")), True),  # warn -> block
    (B(), B(MIN(0.8)), False),                              # removing a block rule loosens
    (B(), B(R("m", "min_metric", {"metric": "acc", "min": 0.8}, "warn")), True),         # removing warn ok
    (B(MIN(0.8), R("x", "flag_true", {"key": "k"})), B(MIN(0.8)), True),                  # extra rule ok
    (B(R("m", "max_metric", {"metric": "acc", "max": 1})), B(MIN(0.8)), False),           # kind changed
    (B(R("r", "role_in", {"key": "role", "roles": ["admin", "ops"]})),
     B(R("r", "role_in", {"key": "role", "roles": ["admin"]})), False),                   # other kinds: params equal
    (B(R("r", "role_in", {"key": "role", "roles": ["admin"]})),
     B(R("r", "role_in", {"key": "role", "roles": ["admin"]})), True),
    (B(R("m", "min_metric", {"metric": "acc"})), B(MIN(0.8)), False),                     # incomparable => loosening
])
def test_is_stricter_or_equal_matrix(new, old, expected):
    assert is_stricter_or_equal(new, old) is expected


# ---------------------------------------------------------------- store
SECRET = b"contract-store-secret"


def store():
    audit = ChainedAuditStore(SECRET)
    return PolicyStore(audit), audit


def test_first_activation_needs_no_approval_and_loosening_needs_a_real_approval():
    st, _ = store()
    st.publish("alice", B(MIN(0.9)))
    st.publish("alice", B(MIN(0.7)))
    st.activate("alice", 1)
    for blank in (None, "", "   "):
        with pytest.raises(PolicyDenied):
            st.activate("alice", 2, human_approved_by=blank)
    assert st.active()[0] == 1
    st.activate("alice", 2, human_approved_by="bob")
    assert st.active()[0] == 2


def test_tightening_activation_needs_no_approval():
    st, _ = store()
    st.publish("alice", B(MIN(0.7)))
    st.publish("alice", B(MIN(0.9)))
    st.activate("alice", 1)
    st.activate("alice", 2)
    assert st.active()[0] == 2


def test_rollback_reactivates_previous_and_loosening_rollback_needs_approval():
    st, audit = store()
    st.publish("alice", B(MIN(0.7)))
    st.publish("alice", B(MIN(0.9)))
    st.activate("alice", 1)
    st.activate("alice", 2)
    with pytest.raises(PolicyDenied):                         # v2 -> v1 is looser
        st.rollback("alice")
    assert st.rollback("alice", human_approved_by="bob") == 1
    assert st.active()[0] == 1
    assert [e["action"] for e in audit.export()][-2:] == ["policy.activate", "policy.rollback"]


def test_rollback_with_no_previous_is_a_conflict():
    st, _ = store()
    st.publish("alice", B(MIN(0.7)))
    st.activate("alice", 1)
    with pytest.raises(Conflict):
        st.rollback("alice")


def test_versions_are_monotonic_and_get_returns_the_published_bundle():
    st, _ = store()
    assert [st.publish("a", B(MIN(0.5 + i / 10))) for i in range(3)] == [1, 2, 3]
    assert st.get(2).rules[0].params["min"] == 0.6
    with pytest.raises(NotFound):
        st.get(9)


def test_decide_without_active_policy_is_denied_fail_closed():
    st, _ = store()
    st.publish("alice", B(MIN(0.5)))
    with pytest.raises(PolicyDenied):
        st.decide("x", {"acc": 1.0})


def test_decide_validates_inputs_and_writes_nothing_on_failure():
    st, audit = store()
    st.publish("alice", B(MIN(0.5)))
    st.activate("alice", 1)
    n = len(audit.export())
    for action, ctx in [("", {}), ("  ", {}), ("x", []), ("x", {1: 2})]:
        with pytest.raises((ValidationFailed, TypeError)):
            st.decide(action, ctx)
    assert len(audit.export()) == n and st.decision_log() == []


def test_decision_log_copies_context_and_is_bounded_at_10000():
    st, _ = store()
    st.publish("alice", B(MIN(0.5)))
    st.activate("alice", 1)
    ctx = {"acc": 0.9}
    st.decide("x", ctx)
    ctx["acc"] = 0.0
    assert st.decision_log()[0]["context"] == {"acc": 0.9}     # not aliased
    for i in range(1, 10_003):
        st.decide("x", {"acc": i})
    log = st.decision_log()
    assert len(log) == 10_000
    assert log[0]["context"]["acc"] == 3 and log[-1]["context"]["acc"] == 10_002  # oldest dropped, order kept
    assert log[0]["version"] == 1 and log[0]["allow"] is True


def test_replay_reports_changes_under_another_version_and_writes_no_audit():
    st, audit = store()
    st.publish("alice", B(MIN(0.7)))
    st.publish("alice", B(MIN(0.9)))
    st.activate("alice", 1)
    st.decide("x", {"acc": 0.8})                       # allowed under v1
    st.decide("x", {"acc": 0.95})                      # allowed under both
    n = len(audit.export())
    diff = st.replay(2)
    assert [d["changed"] for d in diff] == [True, False]
    assert (diff[0]["recorded_allow"], diff[0]["replay_allow"]) == (True, False)
    assert len(audit.export()) == n
    with pytest.raises(NotFound):
        st.replay(9)
    with pytest.raises(ValidationFailed):
        st.replay(True)


def test_publish_rejects_invalid_rules_without_state_change():
    st, audit = store()
    bad_bundles = [
        B(R("a", "flag_true", {"key": "k"}), R("a", "flag_true", {"key": "k"})),   # duplicate id
        B(R("a", "nope", {"key": "k"})),                                            # unknown kind
        B(R("a", "min_metric", {"metric": "m", "min": NAN})),                       # non-finite
        B(R("a", "min_metric", {"metric": "m", "min": True})),                      # bool is not a number
        B(R("a", "flag_true", {"key": "k"}, "fatal")),                              # bad severity
        B(R("", "flag_true", {"key": "k"})),                                        # empty id
        B(R("a", "role_in", {"key": "r", "roles": []})),                            # empty roles
    ]
    for b in bad_bundles:
        with pytest.raises(ValidationFailed):
            st.publish("alice", b)
    assert audit.export() == [] and st.history() == []
