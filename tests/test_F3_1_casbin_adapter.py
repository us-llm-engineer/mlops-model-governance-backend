"""
F3.1 frozen test contract: exec/mlops/interop/casbin_adapter.py (casbin 1.43.0).

Pins the behavior described in the API contract section 1:
  - CASBIN_MODEL_TEXT is a real, loadable casbin model (r=sub,act,res / p=sub,act,res,eft /
    deny-overrides-allow / default-deny). No wildcard matching exists in this matcher: a
    stored "*" is a literal string, not a glob -- build_enforcer_from_bundle's use of "*" as
    a default action/resource is a literal placeholder value only.
  - build_enforcer_from_bundle translates ONLY `role_in` rules with severity == "block" into
    casbin policy rows (one row per allowed role); the other 7 rule kinds (min_metric,
    max_metric, flag_true, flag_false, status_equals, budget_ok, no_nan) are evaluative, not
    access-control, and are ALWAYS returned as untranslatable regardless of severity; a
    `role_in` rule with severity == "warn" is also returned as untranslatable.
  - enforcer_decide validates subject/action/resource are non-empty strings with no control
    characters (chr(0)-chr(31), chr(127)) before ever calling casbin, and defensively converts
    ANY casbin-side exception (e.g. the raw RuntimeError casbin raises on a request-size
    mismatch) into mlops.kernel.ValidationFailed -- a raw RuntimeError must never reach a
    caller.
  - DifferentialResult.disagreements is capped at 200 entries, but `total` and `agreements`
    always reflect the true, uncapped sweep -- the cap must never corrupt the summary counts.
  - run_differential is a pure, deterministic function of (pdp, bundle, requests): no randomness,
    two calls with the same inputs produce an equal DifferentialResult. It builds the minimal
    PDP context a role/resource check needs by mapping the request subject onto every distinct
    `role_in`-rule context key found in the bundle (in this contract's rules that key is always
    "role"), and mapping the request resource onto "resource"; it never filters bundle rules by
    the request's action (PolicyDecisionPoint.decide evaluates the *whole* bundle every time,
    per policy_engine.py).
  - generate_requests(bundle, subjects, seed) is a deterministic combinatorial generator over
    (subjects) x (actions/resources appearing in the bundle's translatable role_in rules) x a
    couple of fixed edge values including "unknown-resource"; only iteration ORDER is seeded
    via random.Random(seed), never the value set itself.

Hand-built exact-result bundle used below (`_hand_bundle`):
    rA: role_in  block  key=role  roles=[admin]     action=deploy   resource=prod
    rB: role_in  block  key=role  roles=[operator]  action=rollback resource=staging
    rC: budget_ok block key=cost_ratio max=0.5                      (untranslatable; context
        never carries "cost_ratio", so this rule always fails -> PDP always denies for this
        bundle, for every subject, since rC is ANDed with rA/rB and can never pass)
This lets both the casbin side (per-row OR-matching) and the PDP side (whole-bundle
AND-matching, always deny here) be worked out by hand.
"""

from __future__ import annotations

import dataclasses
import tempfile
import os

import casbin
import pytest

from mlops.kernel import ValidationFailed
from mlops.policy_engine import Rule, PolicyBundle, PolicyDecisionPoint
from mlops.interop.casbin_adapter import (
    CASBIN_MODEL_TEXT,
    build_enforcer_from_bundle,
    enforcer_decide,
    DifferentialResult,
    run_differential,
    generate_requests,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rule(id, kind, params, severity="block", version=1):
    return Rule(id=id, version=version, kind=kind, params=params, severity=severity)


def _bundle(rules, name="b", version=1):
    return PolicyBundle(rules=rules, name=name, version=version)


def _raw_enforcer_from_text(model_text):
    """Build a *real* casbin Enforcer straight from model text, bypassing our adapter."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False)
    try:
        f.write(model_text)
        f.close()
        return casbin.Enforcer(f.name)
    finally:
        os.unlink(f.name)


FULL_MIX_RULES = [
    _rule("min1", "min_metric", {"metric": "m", "min": 0.5}),
    _rule("max1", "max_metric", {"metric": "m", "max": 0.9}),
    _rule("flagT1", "flag_true", {"key": "ft"}),
    _rule("flagF1", "flag_false", {"key": "ff"}),
    _rule("status1", "status_equals", {"key": "st", "value": "ok"}),
    _rule("budget1", "budget_ok", {"key": "b", "max": 0.8}),
    _rule("nonan1", "no_nan", {"keys": ["m"]}),
    _rule(
        "roleblock1",
        "role_in",
        {"key": "role", "roles": ["admin", "ops"], "action": "deploy", "resource": "prod"},
    ),
    _rule(
        "rolewarn1",
        "role_in",
        {"key": "role", "roles": ["auditor"], "action": "view", "resource": "logs"},
        severity="warn",
    ),
]


def _hand_bundle():
    return _bundle(
        [
            _rule(
                "rA", "role_in",
                {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"},
            ),
            _rule(
                "rB", "role_in",
                {"key": "role", "roles": ["operator"], "action": "rollback", "resource": "staging"},
            ),
            _rule("rC", "budget_ok", {"key": "cost_ratio", "max": 0.5}),
        ],
        name="hand",
    )


CONTROL_CHARS = ["\x00", "\x1f", "\x7f"]


# ---------------------------------------------------------------------------
# A. CASBIN_MODEL_TEXT validity
# ---------------------------------------------------------------------------

def test_model_text_is_nonempty_string():
    assert isinstance(CASBIN_MODEL_TEXT, str)
    assert CASBIN_MODEL_TEXT.strip() != ""


def test_model_text_builds_real_enforcer():
    e = _raw_enforcer_from_text(CASBIN_MODEL_TEXT)
    assert isinstance(e, casbin.Enforcer)
    e.add_policy("alice", "read", "doc1", "allow")
    assert e.enforce("alice", "read", "doc1") is True


def test_model_text_deny_overrides_allow():
    e = _raw_enforcer_from_text(CASBIN_MODEL_TEXT)
    e.add_policy("alice", "read", "doc1", "allow")
    e.add_policy("alice", "read", "doc1", "deny")
    assert e.enforce("alice", "read", "doc1") is False


def test_model_text_unmatched_request_defaults_deny():
    e = _raw_enforcer_from_text(CASBIN_MODEL_TEXT)
    assert e.enforce("nobody", "read", "doc1") is False


# ---------------------------------------------------------------------------
# B. build_enforcer_from_bundle
# ---------------------------------------------------------------------------

def test_full_mix_bundle_untranslatable_count_is_exact():
    bundle = _bundle(FULL_MIX_RULES, name="mix")
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    assert len(untranslatable) == 8
    ids = {r.id for r in untranslatable}
    assert ids == {
        "min1", "max1", "flagT1", "flagF1", "status1", "budget1", "nonan1", "rolewarn1",
    }
    for r in untranslatable:
        assert isinstance(r, Rule)


def test_full_mix_bundle_translated_rows_enforce_correctly():
    bundle = _bundle(FULL_MIX_RULES, name="mix")
    enforcer, _untranslatable = build_enforcer_from_bundle(bundle)
    assert isinstance(enforcer, casbin.Enforcer)
    # both roles from roleblock1 get a row
    assert enforcer.enforce("admin", "deploy", "prod") is True
    assert enforcer.enforce("ops", "deploy", "prod") is True
    # warn-severity role_in was never translated
    assert enforcer.enforce("auditor", "view", "logs") is False
    # an unrelated triple is unmatched -> deny
    assert enforcer.enforce("admin", "delete", "prod") is False


def test_role_in_missing_action_defaults_to_literal_star():
    bundle = _bundle(
        [_rule("r1", "role_in", {"key": "role", "roles": ["admin"], "resource": "prod"})]
    )
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    assert untranslatable == []
    assert enforcer.enforce("admin", "*", "prod") is True
    # "*" is a literal string in this matcher, never a wildcard
    assert enforcer.enforce("admin", "deploy", "prod") is False


def test_role_in_missing_resource_defaults_to_literal_star():
    bundle = _bundle(
        [_rule("r1", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy"})]
    )
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    assert untranslatable == []
    assert enforcer.enforce("admin", "deploy", "*") is True
    assert enforcer.enforce("admin", "deploy", "prod") is False


def test_role_in_missing_both_action_and_resource():
    bundle = _bundle([_rule("r1", "role_in", {"key": "role", "roles": ["admin"]})])
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    assert untranslatable == []
    assert enforcer.enforce("admin", "*", "*") is True
    assert enforcer.enforce("admin", "anything", "*") is False
    assert enforcer.enforce("admin", "*", "anything") is False


# ---------------------------------------------------------------------------
# C. enforcer_decide validation
# ---------------------------------------------------------------------------

def _small_enforcer():
    bundle = _bundle(
        [_rule("r1", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"})]
    )
    enforcer, _ = build_enforcer_from_bundle(bundle)
    return enforcer


def test_enforcer_decide_allow_true_case():
    enforcer = _small_enforcer()
    assert enforcer_decide(enforcer, "admin", "deploy", "prod") is True


def test_enforcer_decide_allow_false_case():
    enforcer = _small_enforcer()
    assert enforcer_decide(enforcer, "admin", "delete", "prod") is False


@pytest.mark.parametrize("bad", [123, None, 1.5, [], {}, True])
def test_enforcer_decide_rejects_non_str_subject(bad):
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, bad, "deploy", "prod")


@pytest.mark.parametrize("bad", [123, None, [], {}])
def test_enforcer_decide_rejects_non_str_action(bad):
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "admin", bad, "prod")


@pytest.mark.parametrize("bad", [123, None, [], {}])
def test_enforcer_decide_rejects_non_str_resource(bad):
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "admin", "deploy", bad)


def test_enforcer_decide_rejects_empty_strings():
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "", "deploy", "prod")
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "admin", "", "prod")
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "admin", "deploy", "")


@pytest.mark.parametrize("ctrl", CONTROL_CHARS)
def test_enforcer_decide_rejects_control_chars_in_subject(ctrl):
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, f"admin{ctrl}", "deploy", "prod")


@pytest.mark.parametrize("ctrl", CONTROL_CHARS)
def test_enforcer_decide_rejects_control_chars_in_resource(ctrl):
    enforcer = _small_enforcer()
    with pytest.raises(ValidationFailed):
        enforcer_decide(enforcer, "admin", "deploy", f"prod{ctrl}")


def test_enforcer_decide_accepts_printable_unicode():
    enforcer = _small_enforcer()
    # unicode letters are not control characters and must not be rejected
    assert enforcer_decide(enforcer, "admín", "deploy", "prod") is False


def test_enforcer_decide_never_lets_raw_runtime_error_escape():
    """A casbin enforcer built against a MISMATCHED model (4-field request) raises a raw
    RuntimeError("invalid request size") when called with a 3-value request. enforcer_decide
    must catch that and re-raise as ValidationFailed, never let the RuntimeError through."""
    mismatched_model = """
[request_definition]
r = sub, act, res, extra

[policy_definition]
p = sub, act, res, extra, eft

[policy_effect]
e = some(where (p.eft == allow)) && !some(where (p.eft == deny))

[matchers]
m = r.sub == p.sub && r.act == p.act && r.res == p.res && r.extra == p.extra
"""
    bad_enforcer = _raw_enforcer_from_text(mismatched_model)
    with pytest.raises(ValidationFailed):
        enforcer_decide(bad_enforcer, "admin", "deploy", "prod")


# ---------------------------------------------------------------------------
# D. DifferentialResult shape and the 200-entry cap
# ---------------------------------------------------------------------------

def test_differential_result_is_frozen_dataclass():
    dr = DifferentialResult(total=0, agreements=0, disagreements=[], untranslatable_rule_ids=[])
    assert dataclasses.is_dataclass(dr)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dr.total = 5


def test_differential_result_field_types():
    dr = DifferentialResult(total=1, agreements=1, disagreements=[], untranslatable_rule_ids=["x"])
    assert isinstance(dr.total, int)
    assert isinstance(dr.agreements, int)
    assert isinstance(dr.disagreements, list)
    assert isinstance(dr.untranslatable_rule_ids, list)


def _all_disagree_bundle_and_pdp():
    # role_in rule that casbin will ALWAYS allow for role "x" on action/resource "act"/"res",
    # but the PDP will ALWAYS deny (a second, disjoint role_in rule makes the AND unsatisfiable).
    bundle = _bundle(
        [
            _rule("rX", "role_in", {"key": "role", "roles": ["x"], "action": "act", "resource": "res"}),
            _rule("rY", "role_in", {"key": "role", "roles": ["__never__"], "action": "act", "resource": "res"}),
        ],
        name="always-disagree",
    )
    pdp = PolicyDecisionPoint(bundle=bundle)
    return bundle, pdp


def test_disagreements_capped_at_200_but_total_and_agreements_reflect_full_count():
    bundle, pdp = _all_disagree_bundle_and_pdp()
    # casbin only has a row for the literal subject "x", so repeat that exact triple: every
    # request is a genuine disagreement (casbin allow, PDP deny), 250 times.
    requests = [("x", "act", "res")] * 250
    result = run_differential(pdp, bundle, requests)
    assert result.total == 250
    assert len(result.disagreements) == 200
    assert result.agreements == 0
    # the cap must never be silently absorbed into agreements/total
    assert result.total != result.agreements + len(result.disagreements)


def test_disagreement_entry_shape():
    bundle, pdp = _all_disagree_bundle_and_pdp()
    result = run_differential(pdp, bundle, [("x", "act", "res")])
    assert len(result.disagreements) == 1
    entry = result.disagreements[0]
    assert set(entry.keys()) == {"subject", "action", "resource", "pdp_result", "casbin_result"}
    assert entry["subject"] == "x"
    assert entry["action"] == "act"
    assert entry["resource"] == "res"
    assert entry["pdp_result"] is False
    assert entry["casbin_result"] is True


def test_no_disagreements_when_pdp_and_casbin_agree():
    bundle = _bundle(
        [_rule("r1", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"})]
    )
    pdp = PolicyDecisionPoint(bundle=bundle)
    result = run_differential(pdp, bundle, [("admin", "deploy", "prod"), ("nobody", "deploy", "prod")])
    assert result.total == 2
    assert result.agreements == 2
    assert result.disagreements == []
    assert result.untranslatable_rule_ids == []


# ---------------------------------------------------------------------------
# E. run_differential validation, determinism, and the exact hand-built case
# ---------------------------------------------------------------------------

def test_run_differential_rejects_empty_requests():
    bundle = _hand_bundle()
    pdp = PolicyDecisionPoint(bundle=bundle)
    with pytest.raises(ValidationFailed):
        run_differential(pdp, bundle, [])


@pytest.mark.parametrize(
    "bad_requests",
    [
        [("a", "b")],  # wrong length
        [("a", "b", "c", "d")],  # wrong length
        [(1, "b", "c")],  # non-str element
        [("a", None, "c")],
        ["not-a-tuple"],
    ],
)
def test_run_differential_rejects_malformed_tuples(bad_requests):
    bundle = _hand_bundle()
    pdp = PolicyDecisionPoint(bundle=bundle)
    with pytest.raises(ValidationFailed):
        run_differential(pdp, bundle, bad_requests)


def test_run_differential_is_deterministic():
    bundle = _hand_bundle()
    pdp = PolicyDecisionPoint(bundle=bundle)
    requests = [
        ("admin", "deploy", "prod"),
        ("operator", "rollback", "staging"),
        ("admin", "rollback", "staging"),
        ("someone", "x", "y"),
        ("operator", "deploy", "prod"),
    ]
    r1 = run_differential(pdp, bundle, requests)
    r2 = run_differential(pdp, bundle, requests)
    assert r1 == r2


def test_run_differential_exact_hand_computed_result():
    bundle = _hand_bundle()
    pdp = PolicyDecisionPoint(bundle=bundle)
    requests = [
        ("admin", "deploy", "prod"),        # PDP: deny (rB fails for role=admin); casbin: allow (rA row) -> DISAGREE
        ("operator", "rollback", "staging"),  # PDP: deny (rA fails for role=operator); casbin: allow (rB row) -> DISAGREE
        ("admin", "rollback", "staging"),   # PDP: deny; casbin: no row matches -> AGREE (both deny)
        ("someone", "x", "y"),              # PDP: deny; casbin: no row matches -> AGREE
        ("operator", "deploy", "prod"),     # PDP: deny; casbin: no row matches (subject/action/resource combo unmatched) -> AGREE
    ]
    result = run_differential(pdp, bundle, requests)

    assert result.total == 5
    assert result.agreements == 3
    assert len(result.disagreements) == 2
    assert result.untranslatable_rule_ids == ["rC"]

    got = {(d["subject"], d["action"], d["resource"]) for d in result.disagreements}
    assert got == {("admin", "deploy", "prod"), ("operator", "rollback", "staging")}
    for d in result.disagreements:
        assert d["pdp_result"] is False
        assert d["casbin_result"] is True


# ---------------------------------------------------------------------------
# F. generate_requests
# ---------------------------------------------------------------------------

def _gen_bundle():
    return _bundle(
        [
            _rule("g1", "role_in", {"key": "role", "roles": ["admin"], "action": "deploy", "resource": "prod"}),
            _rule("g2", "role_in", {"key": "role", "roles": ["operator"], "action": "rollback", "resource": "staging"}),
            _rule("g3", "budget_ok", {"key": "cost_ratio", "max": 0.5}),  # untranslatable, ignored by generator
        ]
    )


def test_generate_requests_deterministic_for_fixed_seed():
    bundle = _gen_bundle()
    subjects = ["admin", "operator", "guest"]
    r1 = generate_requests(bundle, subjects, seed=1)
    r2 = generate_requests(bundle, subjects, seed=1)
    assert r1 == r2


def test_generate_requests_different_seeds_same_set_different_order():
    bundle = _gen_bundle()
    subjects = ["admin", "operator", "guest"]
    r1 = generate_requests(bundle, subjects, seed=1)
    r2 = generate_requests(bundle, subjects, seed=2)
    assert set(r1) == set(r2)
    assert r1 != r2


def test_generate_requests_subjects_are_from_given_list():
    bundle = _gen_bundle()
    subjects = ["admin", "operator", "guest"]
    reqs = generate_requests(bundle, subjects, seed=7)
    assert len(reqs) > 0
    for s, _a, _r in reqs:
        assert s in subjects


def test_generate_requests_includes_unknown_resource_edge_value():
    bundle = _gen_bundle()
    subjects = ["admin", "operator"]
    reqs = generate_requests(bundle, subjects, seed=7)
    resources = {r for _s, _a, r in reqs}
    assert "unknown-resource" in resources


def test_generate_requests_are_well_typed_triples():
    bundle = _gen_bundle()
    reqs = generate_requests(bundle, ["admin"], seed=3)
    for item in reqs:
        assert isinstance(item, tuple)
        assert len(item) == 3
        assert all(isinstance(v, str) for v in item)
