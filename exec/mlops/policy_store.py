"""S3.1 Policy Store: versioned policy bundle versioning and activation (G).

PolicyStore maintains versioned policy bundles with RBAC-controlled activation,
decision logging, and replay. It enforces that looser policies require explicit
human approval. All decisions are audited and immutable copies are returned.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .kernel import (
    Clock,
    Conflict,
    IntegrityError,
    NotFound,
    PolicyDenied,
    SystemClock,
    ValidationFailed,
)
from .policy_engine import PolicyBundle, PolicyDecision, PolicyDecisionPoint, scope_covers, scope_of

__all__ = [
    "is_stricter_or_equal",
    "PolicyStore",
    "PolicyStoreGate",
]


def _clean_str(value: Any, name: str, max_len: int = 256) -> str:
    """Validate and clean a string: non-empty after strip, no control chars, max length.

    Args:
        value: Value to validate
        name: Name of parameter (for error messages)
        max_len: Maximum length after stripping (default 256)

    Returns:
        Cleaned string (after strip)

    Raises:
        TypeError: if value is not a string
        ValidationFailed: if empty after strip, contains control chars, or exceeds max_len
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")

    cleaned = value.strip()
    if not cleaned:
        raise ValidationFailed(f"{name} must not be empty or whitespace-only")

    # Check for control characters (ord < 0x20 or == 0x7f)
    for char in cleaned:
        code = ord(char)
        if code < 0x20 or code == 0x7f:
            raise ValidationFailed(f"{name} contains control character")

    if len(cleaned) > max_len:
        raise ValidationFailed(f"{name} exceeds {max_len} characters")

    return cleaned


def _validate_scope(rule_id: str, params: Dict[str, Any]) -> None:
    """action/resource scope: non-empty str, no control chars, '*' only as the last char."""
    for dim in ("action", "resource"):
        if dim not in params:
            continue
        pat = params[dim]
        if not isinstance(pat, str) or not pat.strip():
            raise ValidationFailed(f"rule {rule_id}: {dim} scope must be a non-empty str")
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in pat):
            raise ValidationFailed(f"rule {rule_id}: {dim} scope contains control character")
        if "*" in pat[:-1]:
            raise ValidationFailed(f"rule {rule_id}: {dim} scope may only use '*' as the last character")


def _without_scope(params: Any) -> Any:
    if not isinstance(params, dict):
        return params
    return {k: v for k, v in params.items() if k not in ("action", "resource")}


def _is_finite_number(value: Any) -> bool:
    """Check if value is a finite real number (not bool, int or float is ok if finite).

    Returns True only if value is an int or float (not str, bool, etc) and finite (not NaN/inf).
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        import math
        f = float(value)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


def _validate_rules(rules: List[Any]) -> None:
    """Validate all rules in a bundle BEFORE any state change.

    Checks:
    - rules must be a list
    - at most 200 rules
    - each rule must be a dict
    - rule id: non-empty str, <= 128 chars, no control chars, unique
    - rule kind: one of known kinds
    - rule params: must be a dict
    - rule severity: "block" or "warn"
    - rule params content per kind

    Raises:
        ValidationFailed: if any rule is invalid (names rule id if available)
    """
    if not isinstance(rules, list):
        raise ValidationFailed("rules must be list")

    if len(rules) > 200:
        raise ValidationFailed("at most 200 rules allowed")

    seen_ids = set()

    for rule in rules:
        # Extract fields from Rule object or dict
        if isinstance(rule, dict):
            rule_id = rule.get("id")
            kind = rule.get("kind")
            params = rule.get("params")
            severity = rule.get("severity", "block")
        else:
            # Assume it's a Rule object with id, kind, params, severity attributes
            try:
                rule_id = rule.id
                kind = rule.kind
                params = rule.params
                severity = getattr(rule, "severity", "block")
            except AttributeError:
                raise ValidationFailed("each rule must be dict or Rule object")

        # Validate rule id
        if rule_id is None:
            raise ValidationFailed("rule missing 'id'")
        if not isinstance(rule_id, str):
            raise ValidationFailed(f"rule {rule_id}: id must be str")
        rule_id_clean = rule_id.strip()
        if not rule_id_clean:
            raise ValidationFailed("rule: id must not be empty")
        # Check max length and control chars
        if len(rule_id_clean) > 128:
            raise ValidationFailed(f"rule {rule_id}: id exceeds 128 characters")
        for char in rule_id_clean:
            code = ord(char)
            if code < 0x20 or code == 0x7f:
                raise ValidationFailed(f"rule {rule_id}: id contains control character")
        # Check uniqueness
        if rule_id_clean in seen_ids:
            raise ValidationFailed(f"rule {rule_id}: id not unique")
        seen_ids.add(rule_id_clean)

        # Validate severity
        if severity not in ("block", "warn"):
            raise ValidationFailed(f"rule {rule_id}: severity must be 'block' or 'warn'")

        # Validate kind and params together
        if kind is None:
            raise ValidationFailed(f"rule {rule_id}: missing 'kind'")
        if not isinstance(kind, str):
            raise ValidationFailed(f"rule {rule_id}: kind must be str")

        if params is None:
            raise ValidationFailed(f"rule {rule_id}: missing 'params'")
        if not isinstance(params, dict):
            raise ValidationFailed(f"rule {rule_id}: params must be dict")

        _validate_scope(rule_id, params)

        # Validate params per kind
        if kind == "min_metric":
            metric = params.get("metric")
            if not isinstance(metric, str) or not metric.strip():
                raise ValidationFailed(f"rule {rule_id}: min_metric requires metric: non-empty str")
            if "min" not in params:
                raise ValidationFailed(f"rule {rule_id}: min_metric requires 'min'")
            if not _is_finite_number(params["min"]):
                raise ValidationFailed(f"rule {rule_id}: min_metric 'min' must be finite number")
        elif kind == "max_metric":
            metric = params.get("metric")
            if not isinstance(metric, str) or not metric.strip():
                raise ValidationFailed(f"rule {rule_id}: max_metric requires metric: non-empty str")
            if "max" not in params:
                raise ValidationFailed(f"rule {rule_id}: max_metric requires 'max'")
            if not _is_finite_number(params["max"]):
                raise ValidationFailed(f"rule {rule_id}: max_metric 'max' must be finite number")
        elif kind == "flag_true":
            key = params.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValidationFailed(f"rule {rule_id}: flag_true requires key: non-empty str")
        elif kind == "flag_false":
            key = params.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValidationFailed(f"rule {rule_id}: flag_false requires key: non-empty str")
        elif kind == "status_equals":
            key = params.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValidationFailed(f"rule {rule_id}: status_equals requires key: non-empty str")
            value = params.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValidationFailed(f"rule {rule_id}: status_equals requires value: non-empty str")
        elif kind == "budget_ok":
            key = params.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValidationFailed(f"rule {rule_id}: budget_ok requires key: non-empty str")
            if "max" not in params:
                raise ValidationFailed(f"rule {rule_id}: budget_ok requires 'max'")
            if not _is_finite_number(params["max"]):
                raise ValidationFailed(f"rule {rule_id}: budget_ok 'max' must be finite number")
        elif kind == "role_in":
            key = params.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValidationFailed(f"rule {rule_id}: role_in requires key: non-empty str")
            roles = params.get("roles")
            if not isinstance(roles, list) or not roles:
                raise ValidationFailed(f"rule {rule_id}: role_in requires roles: non-empty list")
            for role in roles:
                if not isinstance(role, str) or not role.strip():
                    raise ValidationFailed(f"rule {rule_id}: role_in roles must be non-empty strings")
        elif kind == "no_nan":
            keys = params.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValidationFailed(f"rule {rule_id}: no_nan requires keys: non-empty list")
            for k in keys:
                if not isinstance(k, str) or not k.strip():
                    raise ValidationFailed(f"rule {rule_id}: no_nan keys must be non-empty strings")
        else:
            raise ValidationFailed(f"rule {rule_id}: unknown kind '{kind}'")


def _approval(value: Optional[str]) -> bool:
    """Check if value is a valid non-blank approval signature.

    Returns True only if value is a non-empty string after strip().
    Used to distinguish "no approval" (None or blank) from "has approval".

    Args:
        value: Value to check (typically human_approved_by)

    Returns:
        True if value is a non-blank string; False if None or blank

    Raises:
        ValidationFailed: if value is a non-str type other than None
    """
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValidationFailed(f"approval must be str or None, got {type(value).__name__}")
    return bool(value.strip())


def is_stricter_or_equal(new: PolicyBundle, old: PolicyBundle) -> bool:
    """Check if new bundle is at least as strict as old bundle.

    Returns True if new is stricter or equal in all dimensions:
    - For each rule id in old: if missing in new => False (unless old was warn)
    - min_metric: new min >= old min (higher is stricter)
    - max_metric: new max <= old max (lower is stricter)
    - Severity "block"->"warn" => False; "warn"->"block" => True
    - Other rule kinds: params must be equal
    - Different kinds for same id => False
    - Extra rules in new are fine

    Args:
        new: The candidate stricter bundle
        old: The baseline bundle to compare against

    Returns:
        True if new is at least as strict as old; False otherwise
    """
    # Build a map of rules by id for both bundles
    old_by_id = {r.id: r for r in old.rules}
    new_by_id = {r.id: r for r in new.rules}

    # Check each rule in old
    for rule_id, old_rule in old_by_id.items():
        if rule_id not in new_by_id:
            # Rule removed from new
            # Only ok if old rule was warn severity
            if old_rule.severity != "warn":
                return False
            continue

        new_rule = new_by_id[rule_id]

        # Check that kind is the same
        if new_rule.kind != old_rule.kind:
            return False

        # Check severity: block->warn is loosening
        if old_rule.severity == "block" and new_rule.severity == "warn":
            return False
        # warn->block is fine (tightening)

        # Check rule-specific strictness, robust against incomparable params
        try:
            # A narrower scope means the rule constrains fewer requests: loosening
            for new_pat, old_pat in zip(scope_of(new_rule.params), scope_of(old_rule.params)):
                if not scope_covers(new_pat, old_pat):
                    return False
            if old_rule.kind == "min_metric":
                # Higher minimum is stricter
                old_min = old_rule.params.get("min") if isinstance(old_rule.params, dict) else None
                new_min = new_rule.params.get("min") if isinstance(new_rule.params, dict) else None
                # If either is missing or non-comparable, treat as "not stricter" (loosening)
                if old_min is None or new_min is None:
                    return False
                if new_min < old_min:
                    return False
            elif old_rule.kind == "max_metric":
                # Lower maximum is stricter
                old_max = old_rule.params.get("max") if isinstance(old_rule.params, dict) else None
                new_max = new_rule.params.get("max") if isinstance(new_rule.params, dict) else None
                # If either is missing or non-comparable, treat as "not stricter" (loosening)
                if old_max is None or new_max is None:
                    return False
                if new_max > old_max:
                    return False
            else:
                # For other kinds, params (ignoring scope, checked above) must be equal
                if _without_scope(new_rule.params) != _without_scope(old_rule.params):
                    return False
        except (TypeError, AttributeError):
            # If comparison fails, treat as "not stricter" (loosening)
            return False

    return True


@dataclass
class PolicyStore:
    """Versioned policy bundle store with approval-gated loosening and audit.

    Maintains an append-only set of policy bundles, exactly one of which is
    active at a time. Activation is atomic (single-assignment under a lock).
    Looser policies require human approval. All operations are audited.

    Attributes:
        audit: ChainedAuditStore for recording all operations
        clock: Optional Clock for timestamps (defaults to SystemClock)
    """

    audit: Any  # ChainedAuditStore
    clock: Optional[Clock] = None
    # Optional durable store (an OpsRepo): every publish/activate is written through BEFORE the
    # in-memory state changes, and versions plus the active one are reloaded on construction.
    repo: Any = None

    def __post_init__(self):
        """Initialize mutable state after dataclass construction."""
        if self.clock is None:
            self.clock = SystemClock()
        self._bundles: Dict[int, PolicyBundle] = {}  # version -> bundle (copies)
        self._notes: Dict[int, str] = {}  # version -> note
        self._published_ts: Dict[int, float] = {}  # version -> timestamp
        self._active_version: Optional[int] = None
        self._previous_active: Optional[int] = None
        self._version_counter = 0
        self._lock = threading.Lock()
        self._decision_log: deque[Dict[str, Any]] = deque(maxlen=10000)
        if self.repo is not None:
            self._reload()

    def _reload(self) -> None:
        """Rebuild versions, notes, timestamps, the counter and the active version from the repo.

        _previous_active is not stored, so a rollback straight after a restart has nothing to
        roll back to. A row that does not parse or validate raises IntegrityError: silently
        starting empty would hide a corrupted policy store.
        """
        try:
            rows = self.repo.list_policy_versions()
            for row in rows:
                version = row["version"]
                bundle = PolicyBundle.from_dict(json.loads(row["bundle_json"]))
                _validate_rules(bundle.rules)
                self._bundles[version] = bundle
                self._notes[version] = row["note"]
                self._published_ts[version] = row["published_ts"]
                if row["active"]:
                    self._active_version = version
            if rows:
                self._version_counter = max(row["version"] for row in rows)
        except Exception as e:
            raise IntegrityError(f"policy store reload failed: {type(e).__name__}: {e}") from e

    def _persist(self, version: int, bundle: PolicyBundle, note: str, ts: float, active: bool) -> None:
        if self.repo is None:
            return
        self.repo.upsert_policy_version({
            "version": version,
            "name": bundle.name,
            "bundle_json": json.dumps(bundle.to_dict(), sort_keys=True),
            "note": note,
            "published_ts": ts,
            "active": active,
        })

    def publish(
        self,
        actor: str,
        bundle: PolicyBundle,
        note: str = "",
    ) -> int:
        """Publish a new policy bundle version.

        Validates inputs BEFORE any state change or audit write, stores an
        immutable copy, and audits the action. Versions increment monotonically
        starting at 1.

        Args:
            actor: Actor publishing the bundle (non-empty string, no control chars)
            bundle: PolicyBundle to publish (must be PolicyBundle instance)
            note: Optional note (defaults to "")

        Returns:
            New version number (1, 2, 3, ...)

        Raises:
            ValidationFailed: if inputs are invalid
            TypeError: if actor is not a string or bundle is wrong type
        """
        # Strict validation BEFORE any state change
        actor = _clean_str(actor, "actor")  # May raise TypeError or ValidationFailed
        if not isinstance(bundle, PolicyBundle):
            raise ValidationFailed("bundle must be PolicyBundle instance")
        if note:  # Only validate if non-empty
            note = _clean_str(note, "note")

        # Validate all rules BEFORE consuming a version
        _validate_rules(bundle.rules)

        # Increment version and store a deep copy
        with self._lock:
            version = self._version_counter + 1
            # Store a deep copy to prevent caller mutations
            bundle_copy = copy.deepcopy(bundle)
            ts = self.clock.now()
            self._persist(version, bundle_copy, note, ts, active=False)  # raises -> nothing changed
            self._version_counter = version
            self._bundles[version] = bundle_copy
            self._notes[version] = note
            self._published_ts[version] = ts

        # Audit the publish
        self.audit.append(
            actor,
            "policy.publish",
            bundle.name,
            "allow",
            ts=self.clock.now(),
            meta={"version": version, "note": note},
        )

        return version

    def activate(
        self,
        actor: str,
        version: int,
        human_approved_by: Optional[str] = None,
    ) -> None:
        """Activate a policy bundle version.

        If no bundle is currently active, the first activation requires no
        approval. If a bundle is active and the new one is NOT stricter or
        equal, human_approved_by must be a non-empty string.

        Validates all inputs BEFORE any state change or audit write.

        Args:
            actor: Actor performing activation (non-empty string, no control chars)
            version: Version to activate (must exist, must be int)
            human_approved_by: Optional approval signature (non-empty str or None)

        Raises:
            ValidationFailed: if actor, version, or approval types are invalid
            NotFound: if version does not exist
            PolicyDenied: if loosening without approval
        """
        # Strict validation BEFORE any state change
        actor = _clean_str(actor, "actor")  # May raise TypeError or ValidationFailed
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValidationFailed("version must be int (not bool)")
        if version <= 0:
            raise ValidationFailed("version must be positive int")

        # Validate approval: raises ValidationFailed for non-str, returns bool for None/blank/str
        has_approval = _approval(human_approved_by)

        # Check version exists
        with self._lock:
            if version not in self._bundles:
                raise NotFound(f"Version {version} not found")

            # Check if this is loosening
            previous = self._active_version
            is_loosening = False
            if previous is not None:
                old_bundle = self._bundles[previous]
                new_bundle = self._bundles[version]
                is_loosening = not is_stricter_or_equal(new_bundle, old_bundle)

            # Validate approval requirement: blank approver counts as no approval
            if is_loosening and not has_approval:
                raise PolicyDenied(
                    f"Version {version} is looser than {previous}; "
                    "human approval required"
                )

            self._persist(version, self._bundles[version], self._notes.get(version, ""),
                          self._published_ts.get(version, 0.0), active=True)

            # Atomically swap the active version
            self._previous_active = self._active_version
            self._active_version = version

        # Audit the activation
        meta = {
            "version": version,
            "previous": previous,
            "loosening": is_loosening,
        }
        if has_approval:
            meta["approved_by"] = human_approved_by
        self.audit.append(
            actor,
            "policy.activate",
            str(version),
            "allow",
            ts=self.clock.now(),
            meta=meta,
        )

    def active(self) -> Tuple[int, PolicyBundle]:
        """Return the currently active policy version and bundle.

        Returns:
            Tuple of (version: int, bundle: PolicyBundle copy)

        Raises:
            NotFound: if no bundle is currently active
        """
        with self._lock:
            if self._active_version is None:
                raise NotFound("No active policy bundle")
            version = self._active_version
            bundle = self._bundles[version]

        # Return a deep copy
        return version, copy.deepcopy(bundle)

    def get(self, version: int) -> PolicyBundle:
        """Get a policy bundle by version.

        Returns a deep copy to prevent mutations from affecting the store.

        Args:
            version: Version number to retrieve

        Returns:
            PolicyBundle copy

        Raises:
            NotFound: if version does not exist
        """
        with self._lock:
            if version not in self._bundles:
                raise NotFound(f"Version {version} not found")
            bundle = self._bundles[version]

        return copy.deepcopy(bundle)

    def history(self) -> List[Dict[str, Any]]:
        """Return the publication history of all versions.

        Each entry includes version, note, published_ts, and active flag.

        Returns:
            List of history dicts, in version order
        """
        with self._lock:
            result = []
            for version in sorted(self._bundles.keys()):
                result.append({
                    "version": version,
                    "note": self._notes.get(version, ""),
                    "published_ts": self._published_ts.get(version, 0.0),
                    "active": version == self._active_version,
                })
            return result

    def rollback(
        self,
        actor: str,
        human_approved_by: Optional[str] = None,
    ) -> int:
        """Reactivate the previously active version.

        Works like activate() but on the previous version. Follows the same
        loosening rule: if rolling back loosens the policy, approval is required.

        Validates all inputs BEFORE any state change.

        Args:
            actor: Actor performing rollback (non-empty string, no control chars)
            human_approved_by: Optional approval signature

        Returns:
            Version that was reactivated

        Raises:
            ValidationFailed: if inputs are invalid
            Conflict: if there is no previous version to roll back to
            PolicyDenied: if rollback would loosen without approval
        """
        # Strict validation BEFORE any state change
        actor = _clean_str(actor, "actor")  # May raise TypeError or ValidationFailed
        has_approval = _approval(human_approved_by)

        with self._lock:
            if self._previous_active is None:
                raise Conflict("No previous version to roll back to")
            previous_version = self._previous_active

        # activate() will handle approval checking
        self.activate(actor, previous_version, human_approved_by=human_approved_by)

        # Audit the rollback
        self.audit.append(
            actor,
            "policy.rollback",
            str(previous_version),
            "allow",
            ts=self.clock.now(),
            meta={"version": previous_version},
        )

        return previous_version

    def decide(
        self,
        action: str,
        context: Dict[str, Any],
        actor: str = "system",
    ) -> PolicyDecision:
        """Evaluate the active policy against context and log the decision.

        Builds a PolicyDecisionPoint on the active bundle, evaluates it,
        and appends the result to the decision log.

        Validates all inputs BEFORE any state change. On validation failure,
        writes nothing to decision_log or audit chain.

        Args:
            action: Action name (non-empty string, no control chars)
            context: Context dict with str keys for evaluation
            actor: Actor name for audit (defaults to "system")

        Returns:
            PolicyDecision with allow/deny and reasons

        Raises:
            ValidationFailed: if action, context, or actor are invalid
            PolicyDenied: if no bundle is active (fail-closed)
        """
        # Strict validation BEFORE any state change
        action = _clean_str(action, "action")  # May raise TypeError or ValidationFailed
        if not isinstance(context, dict):
            raise ValidationFailed("context must be dict")
        # Validate context keys are strings
        for key in context.keys():
            if not isinstance(key, str):
                raise ValidationFailed(f"context keys must be str, got {type(key).__name__}")

        # Get active bundle (fail-closed if none)
        try:
            version, bundle = self.active()
        except NotFound:
            raise PolicyDenied("No active policy bundle (fail-closed)")

        # Deep-copy context to prevent mutations
        context_copy = copy.deepcopy(context)

        # Build decision point on active bundle with this store's audit
        dp = PolicyDecisionPoint(
            bundle=bundle,
            audit=self.audit,
            clock=self.clock,
            allow_empty=False,
        )
        decision = dp.decide(action, context_copy, actor=actor)

        # Append to decision log (only after successful decision)
        log_entry = {
            "action": action,
            "context": context_copy,
            "allow": decision.allow,
            "version": version,
            "policy_hash": decision.policy_hash,
        }
        self._decision_log.append(log_entry)

        return decision

    def decision_log(self) -> List[Dict[str, Any]]:
        """Return the bounded decision log (up to 10000 entries).

        Returns:
            List of decision log entries in order (oldest to newest)
        """
        return list(self._decision_log)

    def replay(
        self,
        version: int,
        decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Re-evaluate logged (or provided) decisions under a specific version.

        Does NOT write to the audit chain. Validates the version and decisions,
        then returns diffs for each decision.

        Args:
            version: Version to replay decisions under
            decisions: Optional list of decision dicts (uses log if None)
                      Each dict must have {action, context, allow}

        Returns:
            List of diff dicts with {index, action, recorded_allow, replay_allow, changed}

        Raises:
            ValidationFailed: if decisions are invalid
            NotFound: if version does not exist
        """
        # Validate version
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValidationFailed("version must be int (not bool)")
        with self._lock:
            if version not in self._bundles:
                raise NotFound(f"Version {version} not found")

        # Use log if no decisions provided
        if decisions is None:
            decisions = list(self._decision_log)

        # Validate decisions structure
        if not isinstance(decisions, list):
            raise ValidationFailed("decisions must be list or None")

        bundle = copy.deepcopy(self._bundles[version])
        result = []

        for index, dec in enumerate(decisions):
            # Strict validation of each decision dict
            if not isinstance(dec, dict):
                raise ValidationFailed(f"decision[{index}] must be dict")

            if "action" not in dec:
                raise ValidationFailed(f"decision[{index}] missing 'action'")
            if "context" not in dec:
                raise ValidationFailed(f"decision[{index}] missing 'context'")
            if "allow" not in dec:
                raise ValidationFailed(f"decision[{index}] missing 'allow'")

            action = dec["action"]
            context = dec["context"]
            recorded_allow = dec["allow"]

            if not isinstance(action, str) or not action.strip():
                raise ValidationFailed(
                    f"decision[{index}]['action'] must be non-empty str"
                )
            if not isinstance(context, dict):
                raise ValidationFailed(
                    f"decision[{index}]['context'] must be dict"
                )
            if not isinstance(recorded_allow, bool):
                raise ValidationFailed(
                    f"decision[{index}]['allow'] must be bool"
                )

            # Re-evaluate under the version (without audit)
            dp = PolicyDecisionPoint(
                bundle=bundle,
                audit=None,  # Do NOT audit replays
                clock=self.clock,
                allow_empty=False,
            )
            decision = dp.decide(action, copy.deepcopy(context), actor="system")

            result.append({
                "index": index,
                "action": action,
                "recorded_allow": recorded_allow,
                "replay_allow": decision.allow,
                "changed": recorded_allow != decision.allow,
            })

        return result


class PolicyStoreGate:
    """Adapts a PolicyStore to the decision-point interface ModelRegistry calls.

    enforce() evaluates the ACTIVE bundle (and logs/audits the decision through the
    store), raising PolicyDenied when it denies. No active bundle also raises
    PolicyDenied (fail-closed).
    """

    def __init__(self, store: PolicyStore) -> None:
        self.store = store
        self.audit = store.audit

    def enforce(self, action: str, context: Dict[str, Any], actor: str = "system") -> PolicyDecision:
        decision = self.store.decide(action, context, actor=actor)
        if not decision.allow:
            raise PolicyDenied(f"Policy denied {action}: {'; '.join(decision.reasons)}")
        return decision
