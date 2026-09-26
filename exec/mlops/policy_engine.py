"""
Policy Engine: declarative versioned fail-closed policy bundles (C77-C78).

A policy bundle is a versioned collection of rules that are evaluated against
context to produce a pass/deny decision. Rules are fail-closed: any unknown
rule kind, missing context key, NaN value, or exception in evaluation causes
the rule to fail (deny). Every decision is logged to the audit chain with the
bundle's hash.

Rule kinds (pure functions of context: dict):
  - min_metric: context[params['metric']] >= params['min']
  - max_metric: context[params['metric']] <= params['max']
  - flag_true: context[params['key']] is True
  - flag_false: context[params['key']] is False
  - status_equals: context[params['key']] == params['value']
  - budget_ok: context[params['key']] <= params['max'] (fractional budget)
  - role_in: context[params['key']] in params['roles']
  - no_nan: no NaN values in context[k] for k in params['keys']

Failure modes (all -> deny):
  - Unknown rule kind
  - Missing context key
  - NaN or non-numeric value (where numeric expected)
  - Rule exception
  - Empty bundle with allow_empty=False (fail-closed default)

PolicyDecisionPoint writes one chain entry per decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .kernel import (
    PolicyDenied,
    canonical_json,
    content_id,
    Clock,
    SystemClock,
)

__all__ = [
    "Rule",
    "PolicyBundle",
    "PolicyDecision",
    "PolicyDecisionPoint",
    "load_policy",
]


@dataclass
class Rule:
    """A policy rule (mutable to support testing).

    Rule is mutable but PolicyBundle never caches its hash - it computes fresh
    on every access. This prevents hash staleness: even if a Rule's params are
    mutated after PolicyBundle creation, the bundle's hash will reflect the
    current state when accessed.
    """

    id: str  # Unique rule identifier
    version: int  # Rule version (for auditing changes)
    kind: str  # Rule kind: min_metric, max_metric, flag_true, flag_false, status_equals, budget_ok, role_in, no_nan
    params: Optional[Dict[str, Any]]  # Rule-specific parameters; None for testing error handling
    severity: str = "block"  # "block" (default) or "warn"


@dataclass(frozen=True)
class PolicyBundle:
    """An immutable versioned collection of policy rules with deterministic hash.

    Frozen to prevent hash staleness: rules and metadata are immutable after
    construction. Hash is computed fresh on every access to guarantee consistency.
    Attempting to set any attribute raises TypeError.
    """

    rules: List[Rule]
    name: str
    version: int

    @property
    def hash(self) -> str:
        """Deterministic hash of the bundle's rules and metadata.

        Computed fresh on every access (not cached) to guarantee the hash
        never goes stale due to mutations. Since PolicyBundle is frozen,
        this is efficient and safe.
        """
        # Sort rules by id for determinism
        sorted_rules = sorted(
            [
                {
                    "id": r.id,
                    "version": r.version,
                    "kind": r.kind,
                    "params": r.params,
                    "severity": r.severity,
                }
                for r in self.rules
            ],
            key=lambda r: r["id"],
        )
        bundle_data = {
            "name": self.name,
            "version": self.version,
            "rules": sorted_rules,
        }
        return content_id("policy", bundle_data)

    def to_dict(self) -> Dict[str, Any]:
        """Export as a JSON-serializable dict."""
        return {
            "name": self.name,
            "version": self.version,
            "rules": [
                {
                    "id": r.id,
                    "version": r.version,
                    "kind": r.kind,
                    "params": r.params,
                    "severity": r.severity,
                }
                for r in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyBundle:
        """Reconstruct a bundle from a dict."""
        rules = [
            Rule(
                id=r["id"],
                version=r["version"],
                kind=r["kind"],
                params=r["params"],
                severity=r.get("severity", "block"),
            )
            for r in data.get("rules", [])
        ]
        return cls(
            rules=rules,
            name=data["name"],
            version=data["version"],
        )

    @classmethod
    def load_policy(cls, text_or_dict: Union[str, Dict[str, Any]]) -> PolicyBundle:
        """Load a bundle from JSON text or a dict."""
        if isinstance(text_or_dict, str):
            data = json.loads(text_or_dict)
        else:
            data = text_or_dict
        return cls.from_dict(data)


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""

    allow: bool  # True if policy allows the action
    reasons: List[str]  # Human-readable reasons for the decision
    rule_results: List[Dict[str, Any]]  # Per-rule evaluation details
    policy_hash: str  # Hash of the policy bundle that produced this decision
    action: str  # Action name (e.g., "promote", "register")


class PolicyDecisionPoint:
    """Evaluates policies and writes decisions to an audit chain.

    Fail-closed design:
    - Empty bundle denies by default unless allow_empty=True
    - Any rule evaluation error (unknown kind, missing key, invalid value, exception) -> deny
    - Warning severity rules never block but appear in rule_results

    Enforcement is opt-in: a decision_point is used only when explicitly set by
    the caller (e.g., PromotionExecutor or ExperimentRegistry). No decision point
    means no policy check.

    Audit sequence note (e.g., for ExperimentRegistry.register_model_version):
    When a decision point is used after an RBAC check, the audit chain records:
    1. RBAC "allow" entry (if principal passes role/scope check)
    2. Policy "deny" entry (if policy denies the action, before any lineage mutation)
    The policy deny entry carries the bundle hash and failed rule ids in metadata,
    allowing downstream auditing to correlate policy denials with their rules.
    """

    def __init__(
        self,
        bundle: Optional[PolicyBundle] = None,
        audit: Optional[Any] = None,
        clock: Optional[Clock] = None,
        allow_empty: bool = False,
    ) -> None:
        """Initialize a decision point.

        Args:
            bundle: PolicyBundle to evaluate (None for pass-through). Empty bundle
                   denies unless allow_empty=True (fail-closed default). The bundle
                   is frozen and immutable; its hash is computed fresh on access.
            audit: Optional audit store (ChainedAuditStore or compatible).
            clock: Optional Clock for timestamps (defaults to SystemClock).
            allow_empty: If True, empty bundle allows; else empty bundle denies
                        (fail-closed default is False).
        """
        self.bundle = bundle
        self.audit = audit
        self.clock = clock or SystemClock()
        self.allow_empty = allow_empty

    def decide(
        self,
        action: str,
        context: Dict[str, Any],
        actor: str = "system",
    ) -> PolicyDecision:
        """Evaluate the policy against context.

        Args:
            action: Action name (e.g., "promote")
            context: Context dict for rule evaluation
            actor: Actor name for audit logging

        Returns:
            PolicyDecision with allow/deny and reasons. Warning severity rules
            never block but are included in rule_results.

        Always writes exactly one entry to the audit chain if audit is set.
        """
        # Empty bundle check (fail-closed by default)
        if not self.bundle or len(self.bundle.rules) == 0:
            decision = PolicyDecision(
                allow=self.allow_empty,
                reasons=["Empty policy bundle (allow_empty=True)"] if self.allow_empty else ["Empty policy bundle (fail-closed)"],
                rule_results=[],
                policy_hash="none" if not self.bundle else self.bundle.hash,
                action=action,
            )
        else:
            # Evaluate each rule
            decision = self._evaluate_rules(action, context)

        # Write audit entry if audit store is available
        if self.audit is not None:
            ts = self.clock.now() if hasattr(self.clock, "now") else None
            audit_decision = "allow" if decision.allow else "deny"
            failed_rule_ids = [
                r["id"] for r in decision.rule_results
                if not r.get("passed", False) and r.get("severity") == "block"
            ]
            meta = {
                "policy_hash": decision.policy_hash,
                "failed": failed_rule_ids,
            }
            # A hardcoded "" here only ever worked against ChainedAuditStore,
            # which does no validation; DurableAuditStore rejects an empty
            # resource outright. context conventionally carries a "resource"
            # key (see interop.casbin_adapter.run_differential); fall back to
            # the action name so this is never empty either way.
            resource = context.get("resource") if isinstance(context, dict) else None
            self.audit.append(
                actor,
                f"policy.{action}",
                resource or action,
                audit_decision,
                ts=ts,
                meta=meta,
            )

        return decision

    def enforce(
        self,
        action: str,
        context: Dict[str, Any],
        actor: str = "system",
    ) -> PolicyDecision:
        """Evaluate policy and raise PolicyDenied if denied.

        Args:
            action: Action name
            context: Context dict
            actor: Actor name for audit

        Returns:
            PolicyDecision if allowed.

        Raises:
            PolicyDenied: if policy denies the action.
        """
        decision = self.decide(action, context, actor)
        if not decision.allow:
            raise PolicyDenied(f"Policy denied {action}: {'; '.join(decision.reasons)}")
        return decision

    def _evaluate_rules(
        self,
        action: str,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate all rules in the bundle."""
        reasons = []
        rule_results = []

        for rule in self.bundle.rules:
            passed = self._evaluate_rule(rule, context)
            rule_results.append(
                {
                    "id": rule.id,
                    "kind": rule.kind,
                    "severity": rule.severity,
                    "passed": passed,
                }
            )
            if not passed:
                if rule.severity == "block":
                    reasons.append(f"Rule {rule.id} ({rule.kind}) failed")
                # Warning rules don't block but are reported

        # Fail if any "block" severity rule failed
        allow = all(
            r["passed"] or r["severity"] == "warn"
            for r in rule_results
        )

        if allow:
            reasons = ["All policy rules passed"]

        return PolicyDecision(
            allow=allow,
            reasons=reasons,
            rule_results=rule_results,
            policy_hash=self.bundle.hash,
            action=action,
        )

    @staticmethod
    def _evaluate_rule(rule: Rule, context: Dict[str, Any]) -> bool:
        """Evaluate a single rule against context.

        Returns False on any error (fail-closed).
        """
        try:
            kind = rule.kind
            params = rule.params

            if kind == "min_metric":
                return PolicyDecisionPoint._eval_min_metric(params, context)
            elif kind == "max_metric":
                return PolicyDecisionPoint._eval_max_metric(params, context)
            elif kind == "flag_true":
                return PolicyDecisionPoint._eval_flag_true(params, context)
            elif kind == "flag_false":
                return PolicyDecisionPoint._eval_flag_false(params, context)
            elif kind == "status_equals":
                return PolicyDecisionPoint._eval_status_equals(params, context)
            elif kind == "budget_ok":
                return PolicyDecisionPoint._eval_budget_ok(params, context)
            elif kind == "role_in":
                return PolicyDecisionPoint._eval_role_in(params, context)
            elif kind == "no_nan":
                return PolicyDecisionPoint._eval_no_nan(params, context)
            else:
                # Unknown kind -> fail
                return False
        except Exception:
            # Any exception -> fail
            return False

    @staticmethod
    def _is_valid_number(val: Any) -> bool:
        """Check if val is a real finite number (not bool, str, None, NaN, inf)."""
        if isinstance(val, bool) or val is None or isinstance(val, str):
            return False
        if isinstance(val, (int, float)):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False
            return True
        return False

    @staticmethod
    def _eval_min_metric(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate min_metric rule: context[metric] >= min. Strict: requires finite number."""
        metric = params.get("metric")
        min_val = params.get("min")
        if metric not in context:
            return False
        val = context[metric]
        if not PolicyDecisionPoint._is_valid_number(val):
            return False
        return val >= min_val

    @staticmethod
    def _eval_max_metric(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate max_metric rule: context[metric] <= max. Strict: requires finite number."""
        metric = params.get("metric")
        max_val = params.get("max")
        if metric not in context:
            return False
        val = context[metric]
        if not PolicyDecisionPoint._is_valid_number(val):
            return False
        return val <= max_val

    @staticmethod
    def _eval_flag_true(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate flag_true rule: context[key] is True"""
        key = params.get("key")
        if key not in context:
            return False
        return context[key] is True

    @staticmethod
    def _eval_flag_false(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate flag_false rule: context[key] is False"""
        key = params.get("key")
        if key not in context:
            return False
        return context[key] is False

    @staticmethod
    def _eval_status_equals(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate status_equals rule: context[key] == value"""
        key = params.get("key")
        value = params.get("value")
        if key not in context:
            return False
        return context[key] == value

    @staticmethod
    def _eval_budget_ok(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate budget_ok rule: context[key] <= max (fractional budget).
        Strict: requires finite number."""
        key = params.get("key")
        max_val = params.get("max")
        if key not in context:
            return False
        val = context[key]
        if not PolicyDecisionPoint._is_valid_number(val):
            return False
        return val <= max_val

    @staticmethod
    def _eval_role_in(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate role_in rule: context[key] in roles"""
        key = params.get("key")
        roles = params.get("roles", [])
        if key not in context:
            return False
        return context[key] in roles

    @staticmethod
    def _eval_no_nan(params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate no_nan rule: no NaN/inf in context[k] for k in keys."""
        keys = params.get("keys", [])
        for key in keys:
            if key not in context:
                return False
            val = context[key]
            if val is None or isinstance(val, bool) or isinstance(val, str):
                return False
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False
        return True


def load_policy(text_or_dict: Union[str, Dict[str, Any]]) -> PolicyBundle:
    """Load a policy bundle from JSON text or a dict.

    Args:
        text_or_dict: JSON string or dict representation of a bundle.

    Returns:
        PolicyBundle with rules reconstructed from the data.
    """
    if isinstance(text_or_dict, str):
        data = json.loads(text_or_dict)
    else:
        data = text_or_dict
    return PolicyBundle.from_dict(data)
