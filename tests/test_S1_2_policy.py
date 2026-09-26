"""S1.2 slim suite: policy engine, ML Test Score, and policy wiring into
PromotionExecutor / ExperimentRegistry. Oracle-based; fail-closed everywhere.
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.experiment_registry import ExperimentRegistry
from mlops.kernel import PolicyDenied
from mlops.lineage import LineageGraph
from mlops.ml_test_score import ITEMS, ITEMS_BY_CATEGORY, evaluate_ml_test_score, score_to_context
from mlops.policy_engine import (
    PolicyBundle,
    PolicyDecisionPoint,
    Rule,
    load_policy,
)
from mlops.promotion import (
    EnvironmentManifest,
    PromotionApprover,
    PromotionExecutor,
    PromotionRole,
)
from mlops.promotion import Principal as PromoPrincipal
from mlops.rbac import Principal, Role

SECRET = b"s1-2-secret"


def _bundle(rules, name="b", version=1):
    return PolicyBundle(rules=rules, name=name, version=version)


def _rule(rid="r1", kind="min_metric", params=None, version=1, severity="block"):
    if params is None:
        params = {"metric": "acc", "min": 0.9}
    return Rule(id=rid, version=version, kind=kind, params=params, severity=severity)


def _point(rules, audit=None, allow_empty=False):
    return PolicyDecisionPoint(_bundle(rules), audit=audit, allow_empty=allow_empty)


def _allowed(rule, ctx):
    return _point([rule]).decide("act", ctx).allow


# 1 ---------------------------------------------------------------------------
def test_bundle_hash_deterministic_order_independent_and_sensitive():
    a = _rule("a", "flag_true", {"key": "k"})
    b = _rule("b", "min_metric", {"metric": "acc", "min": 0.9})
    base = _bundle([a, b]).hash
    assert base == _bundle([copy.deepcopy(a), copy.deepcopy(b)]).hash
    assert base == _bundle([b, a]).hash  # rule order irrelevant

    b_ver = _rule("b", "min_metric", {"metric": "acc", "min": 0.9}, version=2)
    b_par = _rule("b", "min_metric", {"metric": "acc", "min": 0.91})
    assert _bundle([a, b_ver]).hash != base
    assert _bundle([a, b_par]).hash != base
    assert _bundle([a, b], version=2).hash != base
    assert len({base, _bundle([a, b_ver]).hash, _bundle([a, b_par]).hash,
                _bundle([a, b], version=2).hash}) == 4


# 2 ---------------------------------------------------------------------------
def test_load_policy_json_and_dict_roundtrip_same_hash():
    import json

    bundle = _bundle([_rule("a", "flag_true", {"key": "k"}),
                      _rule("b", severity="warn")], name="prod", version=3)
    d = bundle.to_dict()
    from_dict = load_policy(d)
    from_text = load_policy(json.dumps(d))
    assert from_dict.hash == from_text.hash == bundle.hash
    assert PolicyBundle.load_policy(d).hash == bundle.hash
    assert [r.severity for r in from_text.rules] == ["block", "warn"]


# 3 ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ctx,expected",
    [
        ({"acc": 0.95}, True),
        ({"acc": 0.9}, True),  # inclusive boundary
        ({"acc": 0.89}, False),
        ({}, False),
        ({"acc": None}, False),
        ({"acc": "0.99"}, False),
        ({"acc": True}, False),
        ({"acc": float("nan")}, False),
        ({"acc": float("inf")}, False),
        ({"acc": float("-inf")}, False),
    ],
)
def test_min_metric_fail_closed_table(ctx, expected):
    assert _allowed(_rule(), ctx) is expected


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind,params,good,bad",
    [
        ("min_metric", {"metric": "m", "min": 1.0}, {"m": 1.0}, {"m": 0.99}),
        ("max_metric", {"metric": "m", "max": 1.0}, {"m": 1.0}, {"m": 1.01}),
        ("flag_true", {"key": "f"}, {"f": True}, {"f": False}),
        ("flag_false", {"key": "f"}, {"f": False}, {"f": True}),
        ("status_equals", {"key": "s", "value": "ok"}, {"s": "ok"}, {"s": "bad"}),
        ("budget_ok", {"key": "b", "max": 0.8}, {"b": 0.8}, {"b": 0.81}),
        ("role_in", {"key": "r", "roles": ["releaser"]}, {"r": "releaser"}, {"r": "viewer"}),
        ("no_nan", {"keys": ["x", "y"]}, {"x": 1.0, "y": 2}, {"x": 1.0, "y": float("nan")}),
    ],
)
def test_each_rule_kind_has_teeth(kind, params, good, bad):
    rule = _rule("r", kind, params)
    assert _allowed(rule, good) is True
    assert _allowed(rule, bad) is False
    assert _allowed(rule, {}) is False  # missing input also denies


# 5 ---------------------------------------------------------------------------
def test_unknown_kind_and_raising_rule_deny():
    unknown = _rule("u", "made_up_kind", {"key": "k"})
    assert _allowed(unknown, {"k": True}) is False
    # min=None makes the comparison raise TypeError -> must deny, not propagate.
    raising = _rule("x", "min_metric", {"metric": "acc", "min": None})
    dec = _point([raising]).decide("act", {"acc": 0.99})
    assert dec.allow is False
    assert dec.rule_results[0]["passed"] is False
    # params=None makes params.get raise AttributeError -> deny.
    broken = _rule("y", "flag_true", None)
    broken.params = None
    assert _point([broken]).decide("act", {"k": True}).allow is False


# 6 ---------------------------------------------------------------------------
def test_empty_bundle_denies_unless_allow_empty():
    assert _point([]).decide("act", {}).allow is False
    assert _point([], allow_empty=True).decide("act", {}).allow is True
    assert PolicyDecisionPoint(None).decide("act", {}).allow is False
    assert PolicyDecisionPoint(None, allow_empty=True).decide("act", {}).allow is True


# 7 ---------------------------------------------------------------------------
def test_warn_rules_never_block_but_are_reported():
    rules = [_rule("hard", "flag_true", {"key": "ok"}),
             _rule("soft", "min_metric", {"metric": "acc", "min": 0.9}, severity="warn")]
    dec = _point(rules).decide("act", {"ok": True, "acc": 0.1})
    assert dec.allow is True
    by_id = {r["id"]: r for r in dec.rule_results}
    assert by_id["soft"]["passed"] is False and by_id["soft"]["severity"] == "warn"
    assert by_id["hard"]["passed"] is True
    # flipping the block rule flips the decision (teeth)
    assert _point(rules).decide("act", {"ok": False, "acc": 0.99}).allow is False


# 8 ---------------------------------------------------------------------------
def test_each_decide_writes_one_verifiable_chain_entry():
    store = ChainedAuditStore(secret=SECRET)
    rules = [_rule("gate", "flag_true", {"key": "ok"}),
             _rule("soft", "flag_true", {"key": "nope"}, severity="warn")]
    pdp = _point(rules, audit=store)

    pdp.decide("promote", {"ok": True})
    assert len(store.export()) == 1
    pdp.decide("promote", {"ok": False}, actor="alice")
    pdp.decide("promote", {})
    records = store.export()
    assert len(records) == 3

    allow_e, deny_e, deny2 = records
    assert allow_e["decision"] == "allow" and allow_e["meta"]["failed"] == []
    assert deny_e["decision"] == "deny" and deny_e["actor"] == "alice"
    assert deny_e["meta"]["failed"] == ["gate"]  # warn rule not counted as failed-block
    assert deny_e["meta"]["policy_hash"] == pdp.bundle.hash
    assert deny2["meta"]["failed"] == ["gate"]
    assert ChainVerifier(SECRET).verify_export(records, head=store.head())
    # tampering with meta is detected
    tampered = copy.deepcopy(records)
    tampered[1]["meta"]["failed"] = []
    assert not ChainVerifier(SECRET).verify_export(tampered, head=store.head())
    # empty-bundle decisions are also audited exactly once
    store2 = ChainedAuditStore(secret=SECRET)
    _point([], audit=store2).decide("act", {})
    assert len(store2.export()) == 1


# 9 ---------------------------------------------------------------------------
def test_enforce_returns_decision_or_raises_policy_denied():
    store = ChainedAuditStore(secret=SECRET)
    pdp = _point([_rule("gate", "flag_true", {"key": "ok"})], audit=store)
    dec = pdp.enforce("promote", {"ok": True})
    assert dec.allow is True and dec.policy_hash == pdp.bundle.hash
    with pytest.raises(PolicyDenied):
        pdp.enforce("promote", {"ok": False})
    assert [e["decision"] for e in store.export()] == ["allow", "deny"]


# 10 --------------------------------------------------------------------------
def test_ml_test_score_rubric_shape_and_min_category_rule():
    cats = {}
    for item in ITEMS:
        cats.setdefault(item["category"], []).append(item["id"])
    assert len(ITEMS) == 28
    assert len({i["id"] for i in ITEMS}) == 28
    assert {c: len(v) for c, v in cats.items()} == {
        "data": 7, "model": 7, "infrastructure": 7, "monitoring": 7}

    empty = evaluate_ml_test_score({})
    assert empty.final == 0 and empty.total == 0

    ev = {i: 1.0 for c in ("data", "infrastructure", "monitoring") for i in cats[c]}
    ev.update({i: 1.0 for i in cats["model"][:3]})
    rep = evaluate_ml_test_score(ev)
    assert rep.by_category["data"] == 7 and rep.by_category["model"] == 3
    assert rep.final == 3 == min(rep.by_category.values())
    assert rep.total == 24

    half = evaluate_ml_test_score({cats["model"][0]: 0.5, cats["model"][1]: 1})
    assert half.by_category["model"] == 1.5


def test_score_to_context_feeds_policy_rules():
    cats = {}
    for item in ITEMS:
        cats.setdefault(item["category"], []).append(item["id"])
    ev = {i: 1 for c in cats for i in cats[c][:5]}  # 5 per category
    ctx = score_to_context(evaluate_ml_test_score(ev))
    assert ctx["ml_test_score"] == 5
    rule = _rule("mts", "min_metric", {"metric": "ml_test_score", "min": 5})
    assert _allowed(rule, ctx) is True
    ev.pop(cats["monitoring"][0])  # weakest category drops to 4
    ctx2 = score_to_context(evaluate_ml_test_score(ev))
    assert ctx2["ml_test_score"] == 4
    assert _allowed(rule, ctx2) is False
    for key in ("ml_test_score_data", "ml_test_score_model",
                "ml_test_score_infrastructure", "ml_test_score_monitoring"):
        assert isinstance(ctx2[key], float) and not math.isnan(ctx2[key])


# 11 --------------------------------------------------------------------------
def _promo_setup(decision_point=None, ctx=None):
    manifest = EnvironmentManifest()
    approver = PromotionApprover()
    manifest.apply("prod", "v1", "seed")
    ex = PromotionExecutor(
        manifest, approver, decision_point=decision_point,
        context_provider=(lambda env, ver: dict(ctx)) if ctx is not None else None)
    return manifest, approver, ex, PromoPrincipal("rel", PromotionRole.RELEASER)


def _gated(ok):
    store = ChainedAuditStore(secret=SECRET)
    pdp = _point([_rule("gate", "flag_true", {"key": "ok"})], audit=store)
    manifest, approver, ex, _ = _promo_setup(pdp, {"ok": ok})
    return store, pdp, manifest, approver, ex


def test_promotion_unauthorized_never_reaches_policy():
    store, pdp, manifest, approver, ex = _gated(False)
    with pytest.raises(PermissionError) as ei:
        ex.promote(PromoPrincipal("v", PromotionRole.VIEWER), "prod", "v2", "v")
    assert not isinstance(ei.value, PolicyDenied)
    assert [e.approval_status for e in ex.audit_log] == ["denied"]
    assert [a["decision"] for a in approver.audit_log] == ["deny"]
    assert store.export() == []
    assert manifest.current_version("prod") == "v1"


def test_promotion_policy_deny_recorded_after_authorization():
    store, pdp, manifest, approver, ex = _gated(False)
    hist_before = manifest.history("prod")
    with pytest.raises(PolicyDenied):
        ex.promote(PromoPrincipal("rel", PromotionRole.RELEASER), "prod", "v2", "rel")
    assert len(ex.audit_log) == 1
    ev = ex.audit_log[0]
    assert ev.approval_status == "denied_by_policy" and pdp.bundle.hash in ev.detail
    assert [a["decision"] for a in approver.audit_log] == ["allow"]
    recs = store.export()
    assert [(r["action"], r["decision"]) for r in recs] == [("policy.promote", "deny")]
    assert manifest.current_version("prod") == "v1"
    assert manifest.history("prod") == hist_before


def test_promotion_policy_allow_promotes():
    store, pdp, manifest, approver, ex = _gated(True)
    ev = ex.promote(PromoPrincipal("rel", PromotionRole.RELEASER), "prod", "v2", "rel")
    assert ev.approval_status == "approved" and ev.converged
    assert manifest.current_version("prod") == "v2"
    assert [r["decision"] for r in store.export()] == ["allow"]


def test_promotion_without_decision_point_is_unchanged():
    store = ChainedAuditStore(secret=SECRET)
    manifest, approver, ex, who = _promo_setup()
    ev = ex.promote(who, "prod", "v2", "rel")
    assert store.export() == []
    assert ev.from_version == "v1" and ev.to_version == "v2" and ev.converged
    assert manifest.current_version("prod") == "v2"
    assert len(approver.audit_log) == 1 and len(ex.audit_log) == 1
    viewer = PromoPrincipal("v", PromotionRole.VIEWER)
    with pytest.raises(PermissionError):
        ex.promote(viewer, "prod", "v3", "v")
    assert manifest.current_version("prod") == "v2"


# 12 --------------------------------------------------------------------------
def _reg_setup(rules=None):
    store = ChainedAuditStore(secret=SECRET)
    lineage = LineageGraph()
    pdp = _point(rules or [_rule("gate", "flag_true", {"key": "ok"})], audit=store)
    reg = ExperimentRegistry(store, lineage, decision_point=pdp)
    return store, lineage, reg


def _deployer(scope="proj"):
    return Principal(name="dep", role=Role.DEPLOYER, scope=scope)


def test_registry_policy_deny_leaves_lineage_untouched_and_runs_after_rbac():
    store, lineage, reg = _reg_setup()
    reg.register_model_version(_deployer(), "proj/r0", "v1", [], context={"ok": True})
    nodes, edges = len(lineage.nodes), len(lineage.edges)
    assert nodes == 1

    # authorized but policy-denied
    n_before = len(store.export())
    with pytest.raises(PolicyDenied):
        reg.register_model_version(_deployer(), "proj/r1", "v1", [], context={"ok": False})
    assert (len(lineage.nodes), len(lineage.edges)) == (nodes, edges)
    new = store.export()[n_before:]
    assert [(e["action"], e["decision"]) for e in new] == [
        ("experiment.register", "allow"), ("policy.register", "deny")]

    # unauthorized: RBAC denies first, policy is never consulted
    n_before = len(store.export())
    with pytest.raises(Exception) as ei:
        reg.register_model_version(_deployer(scope="other"), "proj/r2", "v1", [],
                                   context={"ok": True})
    assert not isinstance(ei.value, PolicyDenied)
    new = store.export()[n_before:]
    assert [(e["action"], e["decision"]) for e in new] == [("experiment.register", "deny")]
    assert (len(lineage.nodes), len(lineage.edges)) == (nodes, edges)
    assert ChainVerifier(SECRET).verify_export(store.export(), head=store.head())


def test_model_version_id_tracks_artifact_hash_and_optional():
    _, _, reg = _reg_setup([_rule("gate", "flag_true", {"key": "ok"})])
    ctx = {"ok": True}
    a1 = reg.register_model_version(_deployer(), "proj/r", "v1", [], artifact_hash="h" * 64, context=ctx)
    a2 = reg.register_model_version(_deployer(), "proj/r", "v1", [], artifact_hash="h" * 64, context=ctx)
    b = reg.register_model_version(_deployer(), "proj/r", "v1", [], artifact_hash="i" * 64, context=ctx)
    none = reg.register_model_version(_deployer(), "proj/r", "v1", [], context=ctx)
    assert a1 == a2
    assert len({a1, b, none}) == 3
    assert all(x.startswith("mv_") for x in (a1, b, none))

    # no decision point, no artifact_hash: legacy call shape still works
    plain = ExperimentRegistry(ChainedAuditStore(secret=SECRET), LineageGraph())
    assert plain.register_model_version(_deployer(), "proj/r", "v1", []) == none


# 13 --------------------------------------------------------------------------
_BRECK = {
    "data": ["feature_expectations_schema", "features_beneficial", "feature_cost_vs_benefit",
             "feature_meta_requirements", "data_pipeline_privacy",
             "new_features_added_quickly", "input_feature_code_tested"],
    "model": ["model_spec_reviewed", "offline_online_metrics_correlate", "hyperparameters_tuned",
              "staleness_impact_known", "simpler_model_not_better", "quality_on_data_slices",
              "inclusion_tested"],
    "infrastructure": ["training_reproducible", "model_spec_unit_tested",
                       "pipeline_integration_tested", "quality_validated_before_serving",
                       "model_debuggable", "models_canaried_before_serving",
                       "serving_rollback_possible"],
    "monitoring": ["dependency_change_notification", "data_invariants_hold",
                   "training_serving_not_skewed", "models_not_too_stale", "numerically_stable",
                   "compute_performance_not_regressed", "prediction_quality_not_regressed"],
}


def test_ml_test_score_items_are_the_breck_28():
    assert len(ITEMS) == 28
    got = {}
    for item in ITEMS:
        got.setdefault(item["category"], []).append(item["id"])
        assert item["source"].startswith("breck-2017")
    assert {c: sorted(v) for c, v in got.items()} == {c: sorted(v) for c, v in _BRECK.items()}
    assert {c: sorted(i["id"] for i in v) for c, v in ITEMS_BY_CATEGORY.items()} == \
        {c: sorted(v) for c, v in _BRECK.items()}
