"""S3.1 Policy Calibration: feedback-driven threshold optimization.

Calibrator uses observed feedback (decisions that should/shouldn't have been blocked)
to propose tightening, loosening, or no-op adjustments to policy thresholds.
Tight validation ensures numeric values are real and finite; proposal kind is derived
from numeric comparison, not trusted from the caller.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Dict, List, Optional

from .kernel import Conflict, NotFound, ValidationFailed, PolicyDenied
from .policy_engine import PolicyBundle, PolicyDecisionPoint, Rule
from .policy_store import PolicyStore

__all__ = [
    "Feedback",
    "Proposal",
    "evaluate",
    "Calibrator",
]


@dataclasses.dataclass(frozen=True)
class Feedback:
    """Immutable observation of a decision: context and ground truth blocking requirement.

    Attributes:
        context: Dict with metric values (e.g., {"acc": 0.95})
        should_block: Ground truth: True if item should be blocked, False otherwise
    """
    context: Dict[str, Any]
    should_block: bool

    def __post_init__(self) -> None:
        """Validate context and should_block; copy context to prevent caller mutations."""
        # Validate context is a dict with str keys
        if not isinstance(self.context, dict):
            raise ValidationFailed("Feedback.context must be a dict")
        for key in self.context.keys():
            if not isinstance(key, str):
                raise ValidationFailed("Feedback.context keys must be strings")

        # Validate should_block is exactly a bool (not int, str, None, etc.)
        if not isinstance(self.should_block, bool):
            raise ValidationFailed("Feedback.should_block must be a bool")

        # Copy context to prevent caller mutations (use object.__setattr__ since frozen)
        object.__setattr__(self, "context", dict(self.context))


@dataclasses.dataclass(frozen=True)
class Proposal:
    """Immutable proposal for a policy adjustment.

    Attributes:
        rule_id: ID of the rule to adjust
        metric: Metric name (e.g., "acc")
        old_value: Current threshold value
        new_value: Proposed new threshold value
        kind: "tighten", "loosen", or "none"
        missed_detections: Count of items where should_block=True but allowed
        false_alarms: Count of items where should_block=False but blocked
        evidence: Count of feedback items analyzed
    """
    rule_id: str
    metric: str
    old_value: float
    new_value: float
    kind: str
    missed_detections: int
    false_alarms: int
    evidence: int


def _is_valid_number(val: Any) -> bool:
    """Check if val is a real finite number (not bool, str, None, NaN, inf)."""
    if isinstance(val, bool) or val is None or isinstance(val, str):
        return False
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return True
    return False


def _check_feedback_list(feedback: Any) -> None:
    """Validate that feedback is a list/tuple of Feedback instances.

    Args:
        feedback: Value to validate

    Raises:
        ValidationFailed: if not a list/tuple, has > 100000 items, or contains non-Feedback items
    """
    if not isinstance(feedback, (list, tuple)):
        raise ValidationFailed("feedback must be a list or tuple of Feedback instances")
    if len(feedback) > 100000:
        raise ValidationFailed("feedback list exceeds maximum size of 100000")
    for i, item in enumerate(feedback):
        if not isinstance(item, Feedback):
            raise ValidationFailed(f"feedback[{i}] must be a Feedback instance")


def _approval(value: str) -> None:
    """Validate a human approval signature.

    Must be a non-empty str with no control characters (ord < 0x20 or >= 0x7f).

    Args:
        value: String value to validate (must be a non-empty str)

    Raises:
        ValidationFailed: if it's a string with control characters or wrong type
        PolicyDenied: if it's an empty or whitespace-only string
    """
    if not isinstance(value, str):
        raise ValidationFailed("human_approved_by must be a str")
    # Check if it's empty or whitespace-only
    if not value.strip():
        raise PolicyDenied("human_approved_by cannot be empty or whitespace-only")
    # Check for control characters (ord < 0x20 or ord >= 0x7f)
    for char in value:
        if ord(char) < 0x20 or ord(char) >= 0x7f:
            raise ValidationFailed(
                "human_approved_by contains control characters"
            )


def evaluate(
    bundle: PolicyBundle,
    feedback: List[Feedback],
) -> Dict[str, Any]:
    """Evaluate a policy bundle against observed feedback.

    Builds a PolicyDecisionPoint without audit and compares each decision against
    the ground truth (should_block). Computes detection and false alarm rates.

    Args:
        bundle: PolicyBundle to evaluate
        feedback: List of Feedback observations

    Returns:
        Dict with keys:
        - n: total feedback count
        - missed: count where should_block=True but allowed
        - false_alarms: count where should_block=False but blocked
        - blocked_correctly: count where should_block=True and blocked
        - missed_detection_rate: missed / count(should_block=True), 0.0 if empty
        - false_alarm_rate: false_alarms / count(should_block=False), 0.0 if empty
    """
    # Validate feedback list first
    _check_feedback_list(feedback)

    # Create a decision point without audit (None)
    dp = PolicyDecisionPoint(bundle=bundle, audit=None, allow_empty=False)

    missed = 0
    false_alarms = 0
    blocked_correctly = 0
    should_block_count = 0
    not_should_block_count = 0

    for fb in feedback:
        # Decide on this context (allow==False means blocked)
        decision = dp.decide("calibration", fb.context, actor="system")
        allowed = decision.allow
        blocked = not allowed

        if fb.should_block:
            should_block_count += 1
            if allowed:
                # Missed detection: should block but allowed
                missed += 1
            else:
                # Blocked correctly
                blocked_correctly += 1
        else:
            not_should_block_count += 1
            if blocked:
                # False alarm: shouldn't block but did
                false_alarms += 1

    # Compute rates (0.0 for empty populations)
    missed_detection_rate = (
        missed / should_block_count if should_block_count > 0 else 0.0
    )
    false_alarm_rate = (
        false_alarms / not_should_block_count if not_should_block_count > 0 else 0.0
    )

    return {
        "n": len(feedback),
        "missed": missed,
        "false_alarms": false_alarms,
        "blocked_correctly": blocked_correctly,
        "missed_detection_rate": missed_detection_rate,
        "false_alarm_rate": false_alarm_rate,
    }


class Calibrator:
    """Propose and apply policy adjustments based on feedback.

    Only supports min_metric rules. Analyzes feedback to compute missed detections
    and false alarms, then proposes threshold adjustments.
    """

    def __init__(
        self,
        store: PolicyStore,
        rule_id: str,
        step: float = 0.001,
    ) -> None:
        """Initialize the calibrator.

        Args:
            store: PolicyStore to read active policy from
            rule_id: ID of the rule to calibrate
            step: Increment/decrement for proposals (must be > 0 and finite)

        Raises:
            NotFound: if rule_id does not exist in active policy
            ValidationFailed: if rule is not min_metric, or if step is invalid
            ValueError: if step is invalid
            TypeError: if step is invalid type
        """
        # Validate step strictly: must be a real finite positive number (not bool)
        if isinstance(step, bool):
            raise TypeError("step must be numeric, not bool")
        if not _is_valid_number(step):
            raise ValidationFailed("step must be a real finite positive number")
        if step <= 0:
            raise ValidationFailed("step must be positive")

        # Get active policy and find the rule
        _, active_bundle = store.active()
        rule_dict = {r.id: r for r in active_bundle.rules}

        if rule_id not in rule_dict:
            raise NotFound(f"Rule {rule_id} not found in active policy")

        rule = rule_dict[rule_id]
        if rule.kind != "min_metric":
            raise ValidationFailed(
                f"Calibrator only supports min_metric rules; {rule_id} is {rule.kind}"
            )

        self.store = store
        self.rule_id = rule_id
        self.step = float(step)
        self.metric = rule.params.get("metric")

    def propose(self, feedback: List[Feedback]) -> Proposal:
        """Analyze feedback and propose a policy adjustment.

        Args:
            feedback: List of Feedback observations

        Returns:
            Proposal with kind "tighten", "loosen", or "none"

        Raises:
            ValidationFailed: if feedback contexts are invalid
        """
        # Validate feedback list first (before any state change)
        _check_feedback_list(feedback)

        # Validate all feedback contexts
        for fb in feedback:
            if self.metric not in fb.context:
                raise ValidationFailed(
                    f"Feedback context missing required metric '{self.metric}'"
                )
            val = fb.context[self.metric]
            if not _is_valid_number(val):
                raise ValidationFailed(
                    f"Feedback metric '{self.metric}' must be real finite number"
                )

        # Get current active threshold
        _, active_bundle = self.store.active()
        rule = next(r for r in active_bundle.rules if r.id == self.rule_id)
        old_value = float(rule.params["min"])

        # Build decision point on active bundle without audit
        dp = PolicyDecisionPoint(bundle=active_bundle, audit=None, allow_empty=False)

        missed_contexts = []
        false_alarm_contexts = []

        # Analyze feedback
        for fb in feedback:
            decision = dp.decide("calibration", fb.context, actor="system")
            allowed = decision.allow

            if fb.should_block and allowed:
                # Missed detection
                missed_contexts.append(fb.context[self.metric])
            elif not fb.should_block and not allowed:
                # False alarm
                false_alarm_contexts.append(fb.context[self.metric])

        # Determine proposal kind and new value
        if missed_contexts:
            # Tighten: raise threshold to block more of the missed items
            new_value = max(missed_contexts) + self.step
            # Cap to ensure it's finite (shouldn't happen with valid input, but be safe)
            if math.isinf(new_value):
                new_value = old_value  # Fallback
            kind = "tighten"
            missed_count = len(missed_contexts)
            false_alarm_count = len(false_alarm_contexts)
        elif false_alarm_contexts:
            # Loosen: lower threshold to allow some false alarms
            new_value = min(false_alarm_contexts)
            kind = "loosen"
            missed_count = 0
            false_alarm_count = len(false_alarm_contexts)
        else:
            # No evidence for change
            new_value = old_value
            kind = "none"
            missed_count = 0
            false_alarm_count = 0

        return Proposal(
            rule_id=self.rule_id,
            metric=self.metric,
            old_value=old_value,
            new_value=new_value,
            kind=kind,
            missed_detections=missed_count,
            false_alarms=false_alarm_count,
            evidence=len(feedback),
        )

    def apply(
        self,
        actor: str,
        proposal: Proposal,
        human_approved_by: Optional[str] = None,
    ) -> Optional[int]:
        """Apply a proposal to publish and activate a new policy version.

        Derives the actual kind from numeric comparison, not from proposal.kind.
        - "tighten" (new > old): publishes and activates automatically
        - "loosen" (new < old): requires human_approved_by or raises PolicyDenied
        - "none" (new == old): returns None

        Args:
            actor: Actor applying the proposal (non-empty string)
            proposal: Proposal to apply
            human_approved_by: Optional approval signature (required for loosening)

        Returns:
            New version number (int) or None if kind is "none"

        Raises:
            Conflict: if proposal.old_value != current active threshold (stale)
            ValidationFailed: if the proposal's kind is forged (doesn't match new vs old)
            PolicyDenied: if loosening without human approval
        """
        # Validate inputs BEFORE any state change
        if not isinstance(actor, str) or not actor:
            raise ValidationFailed("actor must be non-empty str")

        # Get current active policy
        _, active_bundle = self.store.active()
        rule = next(r for r in active_bundle.rules if r.id == proposal.rule_id)
        current_min = float(rule.params["min"])

        # Check staleness: proposal must be for the current active threshold
        if not (
            isinstance(proposal.old_value, (int, float))
            and abs(proposal.old_value - current_min) < 1e-10
        ):
            raise Conflict(
                f"Proposal is stale: expected old_value ~{current_min}, "
                f"got {proposal.old_value}"
            )

        # Derive actual kind from numeric comparison, don't trust proposal.kind
        if proposal.new_value > proposal.old_value:
            actual_kind = "tighten"
        elif proposal.new_value < proposal.old_value:
            actual_kind = "loosen"
        else:
            actual_kind = "none"

        # Detect forged kind: if proposal.kind doesn't match derived kind, reject
        if proposal.kind != actual_kind:
            raise ValidationFailed(
                f"Proposal kind mismatch: claimed {proposal.kind}, "
                f"but new_value {proposal.new_value} vs old_value {proposal.old_value} "
                f"indicates {actual_kind}"
            )

        # If none, return None without publishing
        if actual_kind == "none":
            return None

        # Check approval requirement for loosening BEFORE publishing (before consuming version)
        if actual_kind == "loosen":
            if not human_approved_by or not isinstance(human_approved_by, str):
                raise PolicyDenied(
                    "Loosening requires human_approved_by (non-empty str)"
                )
            # Validate the approval string format
            _approval(human_approved_by)

        # Create new bundle with updated rule
        new_rules = []
        for r in active_bundle.rules:
            if r.id == proposal.rule_id:
                # Update the min value for this rule
                new_params = copy.deepcopy(r.params)
                new_params["min"] = proposal.new_value
                new_rule = Rule(
                    id=r.id,
                    version=r.version + 1,
                    kind=r.kind,
                    params=new_params,
                    severity=r.severity,
                )
                new_rules.append(new_rule)
            else:
                new_rules.append(r)

        new_bundle = PolicyBundle(
            rules=new_rules,
            name=active_bundle.name,
            version=active_bundle.version + 1,
        )

        # Publish the new bundle
        version = self.store.publish(
            actor,
            new_bundle,
            note=f"{actual_kind} {proposal.rule_id} to {proposal.new_value}",
        )

        # Activate it (with approval if loosening)
        self.store.activate(
            actor,
            version,
            human_approved_by=human_approved_by if actual_kind == "loosen" else None,
        )

        return version
