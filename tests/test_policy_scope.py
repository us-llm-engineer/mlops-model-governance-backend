"""Rule scope (casbin keyMatch on action and resource) for the raw policy layer.

Written before the code. Contract: docs/POLICY-CONTRACT.md, section Scope.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.kernel import ValidationFailed
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule, key_match
from mlops.policy_store import PolicyStore, is_stricter_or_equal


def R(rid, kind, params, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params, severity=sev)


def B(*rules):
    return PolicyBundle(rules=list(rules), name="p", version=0)


def decide(bundle, action, ctx):
    return PolicyDecisionPoint(bundle=bundle).decide(action, ctx)


def res(d, rid):
    return next(r for r in d.rule_results if r["id"] == rid)


# ------------------------------------------------------------------ key_match
@pytest.mark.parametrize("pattern,value,expected", [
    ("*", "anything", True),
    ("*", "", True),
    ("stage.production", "stage.production", True),
    ("stage.production", "stage.staging", False),
    ("stage.production", "stage.production.x", False),
    ("stage.*", "stage.production", True),
    ("stage.*", "stage.", True),
    ("stage.*", "stage", False),
    ("stage.*", "other.production", False),
    ("model-*", "model-42", True),
    ("model-*", "Model-42", False),
    ("a", "", False),
    ("stage.*", "x.stage.production", False),      # prefix means starts-with, not contains
    ("model-*", "big-model-1", False),
])
def test_key_match_table(pattern, value, expected):
    assert key_match(pattern, value) is expected


# ------------------------------------------------------------------ applicability
def test_action_scoped_rule_applies_only_to_matching_action():
    b = B(R("f", "flag_true", {"key": "go", "action": "stage.production"}))
    hit = decide(b, "stage.production", {"go": False})
    miss = decide(b, "stage.staging", {"go": False})
    assert hit.allow is False and res(hit, "f") == {
        "id": "f", "kind": "flag_true", "severity": "block", "passed": False, "applicable": True}
    assert miss.allow is True and res(miss, "f") == {
        "id": "f", "kind": "flag_true", "severity": "block", "passed": True, "applicable": False}


def test_inapplicable_rule_never_blocks_even_when_its_condition_fails():
    b = B(R("f", "flag_true", {"key": "go", "action": "x"}), R("g", "flag_true", {"key": "ok"}))
    assert decide(b, "y", {"go": False, "ok": True}).allow is True
    assert decide(b, "y", {"go": False, "ok": False}).allow is False      # unscoped g still blocks


def test_resource_scope_and_wildcards():
    b = B(R("f", "flag_true", {"key": "go", "resource": "model-*"}))
    assert decide(b, "x", {"go": False, "resource": "model-1"}).allow is False
    assert decide(b, "x", {"go": True, "resource": "model-1"}).allow is True
    assert decide(b, "x", {"go": False, "resource": "dataset-1"}).allow is True   # not applicable


def test_action_and_resource_scope_must_both_match():
    b = B(R("f", "flag_true", {"key": "go", "action": "a.*", "resource": "r1"}))
    assert decide(b, "a.b", {"go": False, "resource": "r1"}).allow is False
    assert decide(b, "z.b", {"go": False, "resource": "r1"}).allow is True
    assert decide(b, "a.b", {"go": False, "resource": "r2"}).allow is True


def test_resource_scope_without_a_string_resource_fails_closed():
    b = B(R("f", "flag_true", {"key": "go", "resource": "r1"}))
    for ctx in ({"go": True}, {"go": True, "resource": 7}, {"go": True, "resource": None}):
        d = decide(b, "x", ctx)
        assert d.allow is False and res(d, "f")["applicable"] is True


def test_default_scope_is_everything_and_needs_no_resource():
    b = B(R("f", "flag_true", {"key": "go"}), R("g", "flag_true", {"key": "go", "action": "*", "resource": "*"}))
    d = decide(b, "whatever", {"go": True})
    assert d.allow is True and all(r["applicable"] for r in d.rule_results)


def test_warn_rule_scope_and_all_inapplicable_bundle_allows():
    b = B(R("w", "flag_true", {"key": "go", "action": "x"}, "warn"),
          R("f", "flag_true", {"key": "go", "action": "x"}))
    d = decide(b, "y", {"go": False})
    assert d.allow is True and [r["applicable"] for r in d.rule_results] == [False, False]


def test_scoped_role_in_is_the_casbin_style_policy_row():
    b = B(R("deploy", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"}))
    assert decide(b, "deploy", {"role": "admin", "resource": "prod"}).allow is True
    assert decide(b, "deploy", {"role": "viewer", "resource": "prod"}).allow is False
    assert decide(b, "deploy", {"role": "viewer", "resource": "dev"}).allow is True   # other resource: no row


def test_malformed_scope_on_an_unvalidated_rule_fails_closed_at_decision_time():
    # bundles built directly (not via PolicyStore.publish) skip validation; the engine must not fail open
    for params in ({"key": "go", "action": 5}, {"key": "go", "resource": ["r"]},
                   {"key": "go", "action": None}):
        d = decide(B(R("f", "flag_true", params)), "x", {"go": True, "resource": "r"})
        assert d.allow is False and res(d, "f")["applicable"] is True and res(d, "f")["passed"] is False


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("bad", ["", "  ", "a*b", "*a", "a**", "a\x00", 5, None, ["x"]])
@pytest.mark.parametrize("dim", ["action", "resource"])
def test_publish_rejects_a_malformed_scope(dim, bad):
    st = PolicyStore(ChainedAuditStore(b"s"))
    with pytest.raises(ValidationFailed):
        st.publish("alice", B(R("f", "flag_true", {"key": "go", dim: bad})))
    assert st.history() == []


@pytest.mark.parametrize("good", ["*", "a", "stage.production", "stage.*", "model-*"])
def test_publish_accepts_a_wellformed_scope(good):
    st = PolicyStore(ChainedAuditStore(b"s"))
    assert st.publish("alice", B(R("f", "flag_true", {"key": "go", "action": good, "resource": good}))) == 1


def test_unscoped_bundle_hash_is_unchanged_by_this_feature():
    # no scope keys added implicitly: the hash covers exactly the params the caller wrote
    b = B(R("f", "flag_true", {"key": "go"}))
    assert "action" not in b.rules[0].params and "resource" not in b.rules[0].params
    assert b.hash != B(R("f", "flag_true", {"key": "go", "action": "x"})).hash


# ------------------------------------------------------------------ loosening guard with scope
def scoped_min(v, **scope):
    return R("m", "min_metric", {"metric": "acc", "min": v, **scope})


@pytest.mark.parametrize("new_scope,old_scope,expected", [
    ({}, {"action": "stage.production"}, True),                          # widened to everything
    ({"action": "stage.production"}, {"action": "stage.production"}, True),
    ({"action": "stage.*"}, {"action": "stage.production"}, True),       # prefix covers exact
    ({"action": "stage.*"}, {"action": "stage.prod*"}, True),            # prefix covers longer prefix
    ({"action": "stage.production"}, {}, False),                         # narrowed from everything
    ({"action": "stage.production"}, {"action": "stage.*"}, False),      # narrowed prefix -> exact
    ({"action": "stage.prod*"}, {"action": "stage.*"}, False),           # longer prefix is narrower
    ({"action": "other"}, {"action": "stage.production"}, False),        # unrelated
    ({"resource": "m1"}, {"resource": "m*"}, False),
    ({"resource": "m*"}, {"resource": "m1"}, True),
    ({"action": "a"}, {"action": "a", "resource": "r"}, True),           # dropped a dimension = wider
    ({"action": "a", "resource": "r"}, {"action": "a"}, False),          # added a dimension = narrower
])
def test_is_stricter_or_equal_is_scope_aware(new_scope, old_scope, expected):
    assert is_stricter_or_equal(B(scoped_min(0.8, **new_scope)), B(scoped_min(0.8, **old_scope))) is expected


def test_scope_and_threshold_are_both_checked():
    assert is_stricter_or_equal(B(scoped_min(0.9)), B(scoped_min(0.8, action="x"))) is True
    assert is_stricter_or_equal(B(scoped_min(0.7)), B(scoped_min(0.8, action="x"))) is False   # wider but lower


def test_other_kinds_compare_params_ignoring_scope_then_check_scope():
    old = B(R("r", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy"}))
    wider = B(R("r", "role_in", {"key": "role", "roles": ["admin"]}))
    narrower = B(R("r", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"}))
    other_roles = B(R("r", "role_in", {"key": "role", "roles": ["admin", "ops"], "action": "deploy"}))
    assert is_stricter_or_equal(wider, old) is True
    assert is_stricter_or_equal(narrower, old) is False
    assert is_stricter_or_equal(other_roles, old) is False
