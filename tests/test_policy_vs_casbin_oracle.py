"""The raw policy layer against casbin, as an independent oracle.

casbin is a test-time reference only (tests/oracle/casbin_oracle.py); the server never
imports it. Two parts: (1) oracle self-tests on hand-computed cases, so a wrong oracle
cannot vouch for the layer; (2) seeded random bundles x contexts where the raw layer's
decision AND each rule's verdict must equal casbin's. Part 3 proves the comparison bites
by breaking the raw layer in-process and requiring disagreements.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mlops import policy_engine
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule
from tests.oracle.casbin_oracle import CasbinOracle

NAN, INF = float("nan"), float("inf")


def R(rid, kind, params, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params, severity=sev)


def B(*rules):
    return PolicyBundle(rules=list(rules), name="p", version=0)


# ------------------------------------------------------------------ 1. oracle self-tests
@pytest.mark.parametrize("rule,ctx,expected", [
    (R("r", "min_metric", {"metric": "x", "min": 0.5}), {"x": 0.5}, True),
    (R("r", "min_metric", {"metric": "x", "min": 0.5}), {"x": 0.49}, False),
    (R("r", "max_metric", {"metric": "x", "max": 0.5}), {"x": 0.5}, True),
    (R("r", "max_metric", {"metric": "x", "max": 0.5}), {"x": 0.51}, False),
    (R("r", "budget_ok", {"key": "x", "max": 2}), {"x": 2}, True),
    (R("r", "budget_ok", {"key": "x", "max": 2}), {"x": 3}, False),
    (R("r", "min_metric", {"metric": "x", "min": 0}), {"x": True}, False),
    (R("r", "min_metric", {"metric": "x", "min": 0}), {"x": NAN}, False),
    (R("r", "min_metric", {"metric": "x", "min": 0}), {}, False),
    (R("r", "flag_true", {"key": "f"}), {"f": True}, True),
    (R("r", "flag_true", {"key": "f"}), {"f": 1}, False),
    (R("r", "flag_false", {"key": "f"}), {"f": False}, True),
    (R("r", "flag_false", {"key": "f"}), {"f": 0}, False),
    (R("r", "status_equals", {"key": "s", "value": "ok"}), {"s": "ok"}, True),
    (R("r", "status_equals", {"key": "s", "value": "ok"}), {"s": "OK"}, False),
    (R("r", "role_in", {"key": "role", "roles": ["a", "b"]}), {"role": "b"}, True),
    (R("r", "role_in", {"key": "role", "roles": ["a", "b"]}), {"role": "c"}, False),
    (R("r", "no_nan", {"keys": ["a", "b"]}), {"a": 1, "b": 2.0}, True),
    (R("r", "no_nan", {"keys": ["a", "b"]}), {"a": 1, "b": INF}, False),
    (R("r", "no_nan", {"keys": ["a", "b"]}), {"a": 1}, False),
    (R("r", "made_up", {}), {}, False),
])
def test_oracle_kind_verdicts(rule, ctx, expected):
    assert CasbinOracle(B(rule)).decide("a", ctx) is expected


def test_oracle_scope_effect_and_empty_bundle():
    o = CasbinOracle(B(R("f", "flag_true", {"key": "go", "action": "stage.*", "resource": "m-*"})))
    assert o.decide("stage.production", {"go": False, "resource": "m-1"}) is False
    assert o.decide("deploy", {"go": False, "resource": "m-1"}) is True          # other action: no row
    assert o.decide("stage.production", {"go": False, "resource": "d-1"}) is True
    assert o.decide("stage.production", {"go": True}) is False                   # resource missing, scoped: closed
    assert o.decide("stage.production", {"go": True, "resource": 7}) is False
    assert CasbinOracle(B()).decide("a", {}) is False
    assert CasbinOracle(B(), allow_empty=True).decide("a", {}) is True
    warn_only = CasbinOracle(B(R("w", "flag_true", {"key": "go"}, "warn")))
    assert warn_only.decide("a", {"go": False}) is True                          # warn never blocks
    assert warn_only.rule_passes("w", "a", {"go": False}) is False               # but is reported


# ------------------------------------------------------------------ 2. randomized agreement
ACTIONS = ["stage.production", "stage.staging", "deploy", "promote"]
RESOURCES = ["model-1", "model-2", "dataset-1", None, 7]
SCOPES = ["*", "stage.*", "stage.production", "deploy", "model-*", "model-1", "dataset-*"]
ROLES = ["admin", "ops", "viewer"]
THRESHOLDS = [0, 0.5, 0.9, 10, 100]


def gen_rule(rng, i):
    kind = rng.choice(["min_metric", "max_metric", "flag_true", "flag_false", "status_equals",
                       "budget_ok", "role_in", "no_nan", "no_nan", "min_metric", "role_in"])
    params = {}
    if kind in ("min_metric", "max_metric"):
        params = {"metric": rng.choice(["acc", "lat"]), "min" if kind == "min_metric" else "max": rng.choice(THRESHOLDS)}
    elif kind == "budget_ok":
        params = {"key": rng.choice(["cost", "acc"]), "max": rng.choice(THRESHOLDS)}
    elif kind in ("flag_true", "flag_false"):
        params = {"key": rng.choice(["ready", "frozen"])}
    elif kind == "status_equals":
        params = {"key": "status", "value": rng.choice(["ok", "bad"])}
    elif kind == "role_in":
        params = {"key": "role", "roles": rng.sample(ROLES, rng.randint(1, 2))}
    elif kind == "no_nan":
        params = {"keys": rng.sample(["acc", "lat", "cost"], rng.randint(1, 2))}
    for dim in ("action", "resource"):
        if rng.random() < 0.45:
            params[dim] = rng.choice(SCOPES)
    return Rule(f"r{i}", 1, kind, params, "block" if rng.random() < 0.8 else "warn")


def gen_value(rng, thresholds=THRESHOLDS):
    t = rng.choice(thresholds)
    return rng.choice([t, t - 0.001, t + 0.001, float(t), -1, 1e6, NAN, INF, -INF, True, False,
                       None, "1", "ok", "bad", 0, [1]])


def gen_context(rng):
    ctx = {}
    for key in ("acc", "lat", "cost"):
        if rng.random() < 0.85:
            ctx[key] = gen_value(rng)
    for key in ("ready", "frozen"):
        if rng.random() < 0.8:
            ctx[key] = rng.choice([True, False, 1, 0, None, "true"])
    if rng.random() < 0.85:
        ctx["status"] = rng.choice(["ok", "bad", "OK", None, 3])
    if rng.random() < 0.85:
        ctx["role"] = rng.choice(ROLES + ["guest", None, 5])
    res = rng.choice(RESOURCES)
    if res is not None:
        ctx["resource"] = res
    return ctx


def cases(seed, n_bundles, n_ctx):
    rng = random.Random(seed)
    for _ in range(n_bundles):
        bundle = B(*[gen_rule(rng, i) for i in range(rng.randint(1, 5))])
        yield bundle, [(rng.choice(ACTIONS), gen_context(rng)) for _ in range(n_ctx)]


def compare(seed, n_bundles=300, n_ctx=12):
    """Return (decisions_compared, allows, disagreements)."""
    total = allows = 0
    bad = []
    for bundle, requests in cases(seed, n_bundles, n_ctx):
        oracle, dp = CasbinOracle(bundle), PolicyDecisionPoint(bundle=bundle)
        for action, ctx in requests:
            decision = dp.decide(action, ctx)
            total += 1
            allows += decision.allow
            if decision.allow != oracle.decide(action, ctx):
                bad.append((bundle.to_dict(), action, ctx, "decision"))
            for rr in decision.rule_results:
                if rr["passed"] != oracle.rule_passes(rr["id"], action, ctx):
                    bad.append((bundle.to_dict(), action, ctx, f"rule {rr['id']}"))
    return total, allows, bad


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_raw_layer_agrees_with_casbin_on_random_bundles(seed):
    total, allows, bad = compare(seed)
    assert total == 3600
    assert not bad, f"{len(bad)} disagreements, first: {bad[0]}"
    assert 0.15 < allows / total < 0.85, "generator is degenerate: almost all one verdict"


# ------------------------------------------------------------------ 3. the comparison bites
def _disagreements(monkeypatch, patch):
    patch(monkeypatch)
    return len(compare(seed=7, n_bundles=120, n_ctx=10)[2])


def test_oracle_catches_a_max_metric_off_by_one(monkeypatch):
    def patch(mp):
        def strict(params, context):
            return (params["metric"] in context and policy_engine.PolicyDecisionPoint._is_valid_number(context[params["metric"]])
                    and context[params["metric"]] < params["max"])
        mp.setattr(PolicyDecisionPoint, "_eval_max_metric", staticmethod(strict))
    assert _disagreements(monkeypatch, patch) > 0


def test_oracle_catches_scope_ignored(monkeypatch):
    assert _disagreements(monkeypatch, lambda mp: mp.setattr(policy_engine, "key_match", lambda p, v: True)) > 0


def test_oracle_catches_warn_treated_as_block(monkeypatch):
    def patch(mp):
        orig = PolicyDecisionPoint._evaluate_rules

        def all_block(self, action, context):
            d = orig(self, action, context)
            d.allow = all(r["passed"] for r in d.rule_results)
            return d
        mp.setattr(PolicyDecisionPoint, "_evaluate_rules", all_block)
    assert _disagreements(monkeypatch, patch) > 0
