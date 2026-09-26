"""
R2.2 -- Environment Promotion (GitOps-style).

Claims implemented here:
  C17 (source suite R2_2_S1): promotion is declared in versioned environment
      manifests; each promotion emits an audit event carrying promotion
      timestamp, from/to version, requester and approval status.
  C18 (source suite R2_2_S2): a single-environment promotion (manifest update
      + convergence confirmation) completes well under 30s, and concurrent
      promotions to different environments do not deadlock.
  C20 (source suite R2_2_S4): promotion approvals are enforced -- only
      authorized roles (RELEASER for prod; RELEASER/APPROVER for non-prod) may
      approve, and violations are logged with actor, action and decision.

Executor unification
--------------------
The three source suites ship three slightly different ``PromotionExecutor``
shapes:

  * R2_2_S1 -- returns a :class:`PromotionAuditEvent`; keeps a plain list
    ``audit_log``; the denied path appends the event then raises.
  * R2_2_S2 -- adds a ``threading.Lock`` guarding the audit append and relies
    on the manifest's own lock for safe concurrent writes; returns the event.
  * R2_2_S4 -- returns a plain status string and stores dict entries in
    ``audit_log``.

This module unifies all three into a single :class:`PromotionExecutor` that
returns a :class:`PromotionAuditEvent` and appends those events to
``audit_log`` under an audit lock.  The S4 expectations are preserved by the
dict/str-comparison bridge on :class:`PromotionAuditEvent`:

  * ``event["actor"]`` / ``event["approval_status"]`` work via
    ``__getitem__`` -> ``getattr`` (S4 reads the executor log dict-style).
  * ``event == "approved"`` works via ``__eq__`` comparing the event's
    ``approval_status`` when the other operand is a string (S4 asserts on the
    returned status), while attribute comparison is used between two events
    (S1/S2 read ``.from_version`` / ``.to_version`` / ``.converged`` /
    ``.duration_ms``).

Stdlib only.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "now_iso",
    "PromotionRole",
    "Role",
    "Principal",
    "PromotionRequest",
    "PromotionAuditEvent",
    "EnvironmentManifest",
    "PromotionApprover",
    "PromotionExecutor",
]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class PromotionRole(Enum):
    """Roles recognised by the promotion workflow."""

    VIEWER = "viewer"
    APPROVER = "approver"
    RELEASER = "releaser"


#: Alias kept for the reference-implementation spelling used by the suites.
Role = PromotionRole


@dataclass
class Principal:
    """An authenticated actor with a promotion role."""

    name: str
    role: PromotionRole


@dataclass
class PromotionRequest:
    """A request to promote an environment to a target version."""

    environment: str
    to_version: str
    requester: str
    principal: Optional[Principal] = None


@dataclass
class PromotionAuditEvent:
    """A single promotion decision recorded for audit.

    Supports both attribute access (``event.from_version``) and dict-style
    access (``event["actor"]``), and compares equal to a string when that
    string matches ``approval_status`` -- see the module docstring for why.
    """

    action: str
    environment: str
    from_version: str
    to_version: str
    requester: str
    approval_status: str
    actor: str
    timestamp: str
    duration_ms: float = 0.0
    converged: bool = False
    detail: str = ""  # Optional detail (e.g., "policy_hash=..." for policy denials)

    def __getitem__(self, key: str):
        """Allow dict-style reads (e.g. ``event["actor"]``)."""
        return getattr(self, key)

    def __eq__(self, other):
        """Bridge S4's status-string comparison while keeping event equality."""
        if isinstance(other, str):
            return self.approval_status == other
        if not isinstance(other, PromotionAuditEvent):
            return NotImplemented
        return (
            self.action,
            self.environment,
            self.from_version,
            self.to_version,
            self.requester,
            self.approval_status,
            self.actor,
            self.timestamp,
            self.duration_ms,
            self.converged,
            self.detail,
        ) == (
            other.action,
            other.environment,
            other.from_version,
            other.to_version,
            other.requester,
            other.approval_status,
            other.actor,
            other.timestamp,
            other.duration_ms,
            other.converged,
            other.detail,
        )


class EnvironmentManifest:
    """Versioned, thread-safe manifest of what is deployed per environment.

    ``apply`` performs its read-modify-write of the history under a lock so
    concurrent promotions to the same environment cannot lose updates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Dict[str, str] = {}
        self._history: Dict[str, List[Tuple[str, str, str]]] = {}

    def current_version(self, env: str) -> Optional[str]:
        """Return the version currently deployed to ``env`` (or None)."""
        return self._current.get(env)

    def history(self, env: str) -> List[Tuple[str, str, str]]:
        """Return a copy of ``(version, actor, timestamp)`` entries for ``env``."""
        return list(self._history.get(env, []))

    def apply(self, env: str, version: str, actor: str) -> None:
        """Set ``env`` to ``version`` and append a history entry, under lock."""
        with self._lock:
            self._current[env] = version
            self._history.setdefault(env, []).append((version, actor, now_iso()))


class PromotionApprover:
    """Role-based promotion approval with an append-only audit log."""

    PROD_ROLES = {PromotionRole.RELEASER}
    NON_PROD_ROLES = {PromotionRole.RELEASER, PromotionRole.APPROVER}

    def __init__(self) -> None:
        self.audit_log: List[dict] = []

    def approve(self, principal: Principal, environment: str) -> bool:
        """Decide whether ``principal`` may approve promotion to ``environment``.

        Only RELEASER may approve prod; non-prod also accepts APPROVER. Every
        decision is logged with actor, role, action and decision.
        """
        allowed = self.PROD_ROLES if environment == "prod" else self.NON_PROD_ROLES
        decision = principal.role in allowed
        self.audit_log.append(
            {
                "actor": principal.name,
                "role": principal.role.value,
                "action": f"approve_promotion_to_{environment}",
                "decision": "allow" if decision else "deny",
                "timestamp": now_iso(),
            }
        )
        return decision


class PromotionExecutor:
    """Execute promotions: approve, apply the manifest, confirm convergence.

    See the module docstring for how this unifies the S1/S2/S4 executor
    variants. With a decision_point set, promotion is blocked on deny and the
    manifest remains unchanged.
    """

    def __init__(
        self,
        manifest: EnvironmentManifest,
        approver: PromotionApprover,
        decision_point=None,
        context_provider=None,
    ) -> None:
        self.manifest = manifest
        self.approver = approver
        self.decision_point = decision_point
        self.context_provider = context_provider or (lambda env, to_ver: {})
        self.audit_log: List[PromotionAuditEvent] = []
        self._audit_lock = threading.Lock()

    def promote(
        self,
        principal: Principal,
        environment: str,
        to_version: str,
        requester: str,
    ) -> PromotionAuditEvent:
        """Promote ``environment`` to ``to_version`` if approved.

        Authorization check runs FIRST: an unauthorized principal is denied with
        PermissionError (status "denied") appended to audit_log, and no policy
        evaluation occurs (preventing unauthorized actors from learning policy
        details).

        If authorized, policy is enforced AFTER the approver check and BEFORE
        touching the manifest: a policy denial raises PolicyDenied with an audit
        event (status "denied_by_policy", detail includes policy_hash if available),
        leaving the manifest unchanged. Enforcement is opt-in: no decision_point
        means no policy check; behavior is identical to before.

        Returns the resulting audit event on success.
        """
        start = time.perf_counter()
        from_version = self.manifest.current_version(environment)

        # Check authorization FIRST (an unauthorized principal must never reach policy)
        approved = self.approver.approve(principal, environment)
        status = "approved" if approved else "denied"

        if not approved:
            event = PromotionAuditEvent(
                action="promote",
                environment=environment,
                from_version=from_version,
                to_version=to_version,
                requester=requester,
                approval_status=status,
                actor=principal.name,
                timestamp=now_iso(),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            with self._audit_lock:
                self.audit_log.append(event)
            raise PermissionError(
                f"{principal.name} ({principal.role.value}) not authorized "
                f"to promote to {environment}"
            )

        # Enforce policy AFTER authorization, BEFORE touching manifest (fail-closed)
        if self.decision_point is not None:
            try:
                ctx = self.context_provider(environment, to_version)
                self.decision_point.enforce("promote", ctx, actor=principal.name)
            except Exception as e:
                # Policy denied: record as "denied_by_policy" in audit log and re-raise
                detail = ""
                if hasattr(self.decision_point, "bundle") and self.decision_point.bundle:
                    detail = f"policy_hash={self.decision_point.bundle.hash}"

                event = PromotionAuditEvent(
                    action="promote",
                    environment=environment,
                    from_version=from_version,
                    to_version=to_version,
                    requester=requester,
                    approval_status="denied_by_policy",
                    actor=principal.name,
                    timestamp=now_iso(),
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                    detail=detail,
                )
                with self._audit_lock:
                    self.audit_log.append(event)
                raise

        self.manifest.apply(environment, to_version, principal.name)
        converged = self.manifest.current_version(environment) == to_version
        event = PromotionAuditEvent(
            action="promote",
            environment=environment,
            from_version=from_version,
            to_version=to_version,
            requester=requester,
            approval_status=status,
            actor=principal.name,
            timestamp=now_iso(),
            duration_ms=(time.perf_counter() - start) * 1000.0,
            converged=converged,
        )
        with self._audit_lock:
            self.audit_log.append(event)
        return event
