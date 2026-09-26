"""
Casbin interop adapter: translates policy bundles to casbin for cross-check evaluation.

Implements differential testing between PolicyDecisionPoint and casbin to verify
that translatable rules behave equivalently in both systems.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import casbin

from mlops.kernel import ValidationFailed
from mlops.policy_engine import Rule, PolicyBundle, PolicyDecisionPoint


# =============================================================================
# CASBIN_MODEL_TEXT: The casbin model definition
# =============================================================================

CASBIN_MODEL_TEXT = """
[request_definition]
r = sub, act, res

[policy_definition]
p = sub, act, res, eft

[policy_effect]
e = some(where (p.eft == allow)) && !some(where (p.eft == deny))

[matchers]
m = r.sub == p.sub && r.act == p.act && r.res == p.res
"""


# =============================================================================
# build_enforcer_from_bundle: Create a casbin enforcer from a PolicyBundle
# =============================================================================

def build_enforcer_from_bundle(
    bundle: PolicyBundle,
) -> Tuple[casbin.Enforcer, List[Rule]]:
    """Build a casbin Enforcer from a PolicyBundle.

    Translates ONLY role_in rules with severity == "block" into casbin policy rows.
    All other rule kinds and role_in rules with severity == "warn" are returned
    as untranslatable (never silently dropped).

    Args:
        bundle: PolicyBundle to translate

    Returns:
        (enforcer, untranslatable) where enforcer is a casbin.Enforcer and
        untranslatable is a list of Rule objects that could not be translated
    """
    # Write model text to a temporary file (casbin requires a file path)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False)
    try:
        f.write(CASBIN_MODEL_TEXT)
        f.close()
        enforcer = casbin.Enforcer(f.name)
    finally:
        import os
        os.unlink(f.name)

    untranslatable = []

    # Iterate through rules and translate role_in rules with severity="block"
    for rule in bundle.rules:
        if rule.kind == "role_in" and rule.severity == "block":
            # Extract parameters
            params = rule.params or {}
            roles = params.get("roles", [])
            action = params.get("action", "*")
            resource = params.get("resource", "*")

            # Add one policy row per allowed role
            for role in roles:
                enforcer.add_policy(role, action, resource, "allow")
        else:
            # All other rules (including warn-severity role_in) are untranslatable
            untranslatable.append(rule)

    return enforcer, untranslatable


# =============================================================================
# enforcer_decide: Wrapper around casbin.enforce with validation
# =============================================================================

def enforcer_decide(enforcer: casbin.Enforcer, subject: str, action: str, resource: str) -> bool:
    """Evaluate a request against a casbin enforcer with validation.

    Validates that subject, action, and resource are non-empty strings without
    control characters. Converts any casbin exception to ValidationFailed.

    Args:
        enforcer: casbin.Enforcer instance
        subject: Subject (role/user)
        action: Action name
        resource: Resource name

    Returns:
        bool: True if allowed, False if denied

    Raises:
        ValidationFailed: if inputs are invalid or casbin raises an exception
    """
    # Validate each argument
    for name, value in [("subject", subject), ("action", action), ("resource", resource)]:
        if not isinstance(value, str):
            raise ValidationFailed(f"{name} must be a string, got {type(value).__name__}")
        if not value:
            raise ValidationFailed(f"{name} must be non-empty")
        # Check for control characters (chr(0)-chr(31), chr(127))
        for char in value:
            if ord(char) < 32 or ord(char) == 127:
                raise ValidationFailed(f"{name} contains control character")

    try:
        return enforcer.enforce(subject, action, resource)
    except Exception as e:
        # Convert any casbin exception to ValidationFailed
        raise ValidationFailed(f"Casbin enforcement error: {str(e)}")


# =============================================================================
# DifferentialResult: Result of differential testing
# =============================================================================

@dataclass(frozen=True)
class DifferentialResult:
    """Result of differential testing between PDP and casbin.

    Attributes:
        total: Total number of requests evaluated
        agreements: Number of requests where PDP and casbin agreed
        disagreements: List of dicts with keys {subject, action, resource,
                      pdp_result, casbin_result}, capped at 200 entries
        untranslatable_rule_ids: List of rule IDs that couldn't be translated
                                 to casbin
    """
    total: int
    agreements: int
    disagreements: List[Dict[str, Any]]
    untranslatable_rule_ids: List[str]


# =============================================================================
# run_differential: Differential testing of PDP vs casbin
# =============================================================================

def run_differential(
    pdp: PolicyDecisionPoint,
    bundle: PolicyBundle,
    requests: List[Tuple[str, str, str]],
) -> DifferentialResult:
    """Run differential testing between PolicyDecisionPoint and casbin.

    Evaluates each request against both systems and records agreements/disagreements.
    The PDP gets a minimal context with subject mapped to the role context key and
    resource mapped to "resource".

    Args:
        pdp: PolicyDecisionPoint with a bundle
        bundle: PolicyBundle (same as pdp.bundle)
        requests: List of (subject, action, resource) tuples

    Returns:
        DifferentialResult with agreement/disagreement analysis

    Raises:
        ValidationFailed: if requests is empty or malformed
    """
    if not requests:
        raise ValidationFailed("requests must be non-empty")

    # Validate request format
    for req in requests:
        if not isinstance(req, tuple) or len(req) != 3:
            raise ValidationFailed(f"each request must be a (str, str, str) tuple")
        subject, action, resource = req
        if not isinstance(subject, str) or not isinstance(action, str) or not isinstance(resource, str):
            raise ValidationFailed(f"request elements must all be strings")

    # Build enforcer from bundle
    enforcer, untranslatable = build_enforcer_from_bundle(bundle)
    untranslatable_rule_ids = [r.id for r in untranslatable]

    # Extract the context key from role_in rules (should be "role" based on tests)
    # Map subject to this key, and resource to "resource"
    context_key = "role"  # Default, but extract from rules if present
    for rule in bundle.rules:
        if rule.kind == "role_in":
            params = rule.params or {}
            if "key" in params:
                context_key = params["key"]
                break

    # Evaluate each request
    total = len(requests)
    agreements = 0
    disagreements = []

    for subject, action, resource in requests:
        # PDP evaluation: build minimal context
        context = {context_key: subject, "resource": resource}
        pdp_decision = pdp.decide("access", context)
        pdp_result = pdp_decision.allow

        # Casbin evaluation
        try:
            casbin_result = enforcer_decide(enforcer, subject, action, resource)
        except ValidationFailed:
            # If casbin validation fails, treat as deny
            casbin_result = False

        # Record agreement or disagreement
        if pdp_result == casbin_result:
            agreements += 1
        else:
            disagreement = {
                "subject": subject,
                "action": action,
                "resource": resource,
                "pdp_result": pdp_result,
                "casbin_result": casbin_result,
            }
            # Cap disagreements at 200
            if len(disagreements) < 200:
                disagreements.append(disagreement)

    return DifferentialResult(
        total=total,
        agreements=agreements,
        disagreements=disagreements,
        untranslatable_rule_ids=untranslatable_rule_ids,
    )


# =============================================================================
# generate_requests: Generate test requests from a bundle
# =============================================================================

def generate_requests(
    bundle: PolicyBundle,
    subjects: List[str],
    seed: int,
) -> List[Tuple[str, str, str]]:
    """Generate deterministic test requests from a policy bundle.

    Creates requests as a cartesian product of:
    - subjects (provided list)
    - actions/resources from translatable role_in rules
    - fixed edge values (e.g., "unknown-resource")

    Only uses random.Random(seed) for shuffling iteration order, never for
    generating request values.

    Args:
        bundle: PolicyBundle to extract actions/resources from
        subjects: List of subject names
        seed: Random seed for deterministic iteration order

    Returns:
        List of (subject, action, resource) tuples in deterministic order
    """
    # Collect all actions and resources from translatable role_in rules
    actions = set()
    resources = set()

    for rule in bundle.rules:
        if rule.kind == "role_in" and rule.severity == "block":
            params = rule.params or {}
            if "action" in params:
                actions.add(params["action"])
            if "resource" in params:
                resources.add(params["resource"])

    # Add edge values
    actions.add("unknown-action")
    resources.add("unknown-resource")

    # Generate cartesian product
    requests = []
    for subject in subjects:
        for action in actions:
            for resource in resources:
                requests.append((subject, action, resource))

    # Shuffle using seeded random for deterministic but shuffled order
    # Only the iteration order is shuffled, not the values themselves
    rng = random.Random(seed)
    rng.shuffle(requests)

    return requests
