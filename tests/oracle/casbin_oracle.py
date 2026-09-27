"""casbin as an independent reference for the raw policy layer (tests only; never imported by the server).

The raw layer (exec/mlops/policy_engine.py) decides at runtime. This module re-expresses
the same contract (docs/POLICY-CONTRACT.md) in casbin's own language so the two can be
compared on generated inputs:

  request      r = act, res, ctx
  policy row   p = rid, kind, pact, pres, key, val, eft      (one deny row per block rule,
                                                               one per key for no_nan)
  effect       e = !some(where (p.eft == deny))               deny-override
  matcher      keyMatch on action/resource scope + one branch per rule kind

A deny row matches when the rule is applicable AND violated. Scope, deny-override, the
comparison operators and the kind dispatch all live in the casbin matcher text; the small
functions registered below are only data access and type predicates.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import casbin
from casbin.model import Model

SEP = "\x1f"

MODEL_TEXT = """
[request_definition]
r = act, res, ctx

[policy_definition]
p = rid, kind, pact, pres, key, val, eft

[policy_effect]
e = !some(where (p.eft == deny))

[matchers]
m = keyMatch(r.act, p.pact) && ((p.pres != "*" && !isstr(r.res)) || ((p.pres == "*" || keyMatch(r.res, p.pres)) && (violated_metric || violated_flag || violated_other)))
""".strip()

_MIN = '(p.kind == "min_metric" && (!has(r.ctx, p.key) || !isnum(get(r.ctx, p.key)) || num(get(r.ctx, p.key)) < num(p.val)))'
_MAX = '((p.kind == "max_metric" || p.kind == "budget_ok") && (!has(r.ctx, p.key) || !isnum(get(r.ctx, p.key)) || num(get(r.ctx, p.key)) > num(p.val)))'
_FLAG = '((p.kind == "flag_true" && (!has(r.ctx, p.key) || !istrue(get(r.ctx, p.key)))) || (p.kind == "flag_false" && (!has(r.ctx, p.key) || !isfalse(get(r.ctx, p.key)))))'
_OTHER = ('((p.kind == "status_equals" && (!has(r.ctx, p.key) || get(r.ctx, p.key) != p.val))'
          ' || (p.kind == "role_in" && (!has(r.ctx, p.key) || !inset(get(r.ctx, p.key), p.val)))'
          ' || (p.kind == "no_nan" && (!has(r.ctx, p.key) || badnum(get(r.ctx, p.key))))'
          ' || p.kind == "unknown")')
MODEL_TEXT = (MODEL_TEXT.replace("violated_metric", f"({_MIN} || {_MAX})")
              .replace("violated_flag", _FLAG).replace("violated_other", _OTHER))

KNOWN_KINDS = {"min_metric", "max_metric", "flag_true", "flag_false", "status_equals",
               "budget_ok", "role_in", "no_nan"}


# --- data access and type predicates (no policy decisions in here) ---------------
def _has(ctx, key): return key in ctx
def _get(ctx, key): return ctx[key]
def _isstr(x): return isinstance(x, str)
def _num(x): return float(x)
def _istrue(v): return v is True
def _isfalse(v): return v is False


def _isnum(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def _badnum(v):
    if v is None or isinstance(v, (bool, str)):
        return True
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def _inset(x, joined):
    return isinstance(x, str) and x in joined.split(SEP)


def _new_enforcer() -> casbin.Enforcer:
    model = Model()
    model.load_model_from_text(MODEL_TEXT)
    e = casbin.Enforcer(model)
    for name, fn in (("has", _has), ("get", _get), ("isstr", _isstr), ("num", _num),
                     ("isnum", _isnum), ("istrue", _istrue), ("isfalse", _isfalse),
                     ("inset", _inset), ("badnum", _badnum)):
        e.add_function(name, fn)
    return e


def _rows(rule) -> List[List[str]]:
    params = rule.params if isinstance(rule.params, dict) else {}
    pact, pres = str(params.get("action", "*")), str(params.get("resource", "*"))
    kind = rule.kind if rule.kind in KNOWN_KINDS else "unknown"
    if kind == "min_metric":
        head = [str(params["metric"]), repr(float(params["min"]))]
    elif kind in ("max_metric", "budget_ok"):
        name = params["metric"] if kind == "max_metric" else params["key"]
        head = [str(name), repr(float(params["max"]))]
    elif kind in ("flag_true", "flag_false"):
        head = [str(params["key"]), ""]
    elif kind == "status_equals":
        head = [str(params["key"]), str(params["value"])]
    elif kind == "role_in":
        head = [str(params["key"]), SEP.join(params["roles"])]
    elif kind == "no_nan":
        return [[rule.id, kind, pact, pres, str(k), "", "deny"] for k in params["keys"]]
    else:
        head = ["", ""]
    return [[rule.id, kind, pact, pres, *head, "deny"]]


class CasbinOracle:
    """Decision and per-rule verdicts for one PolicyBundle, computed by casbin."""

    def __init__(self, bundle, allow_empty: bool = False) -> None:
        self.rules = list(bundle.rules)
        self.allow_empty = allow_empty
        self._blocking = _new_enforcer()
        self._per_rule: Dict[str, casbin.Enforcer] = {}
        for rule in self.rules:
            single = _new_enforcer()
            for row in _rows(rule):
                single.add_policy(*row)
                if rule.severity == "block":
                    self._blocking.add_policy(*row)
            self._per_rule[rule.id] = single

    @staticmethod
    def _request(action: str, ctx: Dict[str, Any]):
        res = ctx.get("resource")
        return action, res if isinstance(res, str) else None, ctx

    def rule_passes(self, rule_id: str, action: str, ctx: Dict[str, Any]) -> bool:
        return bool(self._per_rule[rule_id].enforce(*self._request(action, ctx)))

    def decide(self, action: str, ctx: Dict[str, Any]) -> bool:
        if not self.rules:
            return self.allow_empty
        if not any(r.severity == "block" for r in self.rules):
            return True
        return bool(self._blocking.enforce(*self._request(action, ctx)))
