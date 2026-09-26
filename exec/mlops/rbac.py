"""
C2 -- RBAC Enforcement with Audit Persistence.

Claim C2: Role-based access control decisions are logged with actor, action,
resource, decision (allow/deny), timestamp, reason, and approval trace. No
silent failures: quota violations are recorded before rejection, and a viewer
may not submit pipelines.

Source test suite: R1.1-S2.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

__all__ = ["Role", "Principal", "AuditEvent", "RBACEngine"]


class Role(Enum):
    """Roles recognised by the RBAC engine."""

    VIEWER = "viewer"
    APPROVER = "approver"
    DEPLOYER = "deployer"


@dataclass
class Principal:
    """An authenticated actor with a role and namespace scope."""

    name: str
    role: Role
    scope: str


@dataclass
class AuditEvent:
    """A single RBAC decision recorded for audit."""

    actor: str
    action: str
    resource: str
    decision: str
    timestamp: datetime
    reason: str
    approval_trace: str


class RBACEngine:
    """RBAC with an append-only audit trail."""

    def __init__(self):
        self.audit_log: List[AuditEvent] = []
        self.quotas: Dict[str, Any] = {
            "default": {"max_concurrent_jobs": 2}
        }
        self.current_jobs: Dict[str, int] = {}

    def record_audit(self, event: AuditEvent) -> None:
        """Append an event to the audit log."""
        self.audit_log.append(event)

    def check_permission(self, principal: Principal, action: str, resource: str) -> bool:
        """Check whether a principal may perform an action on a resource.

        Viewers are denied all actions; approvers and deployers can perform actions.
        Every decision is audited, including denials (Claim C75).
        """
        decision = "deny"
        reason = ""
        approval_trace = ""

        # Determine decision (keep original semantics, just audit every call)
        if principal.role == Role.VIEWER:
            # VIEWER denies all actions; was implicit return False before, now explicit deny+audit
            reason = "Viewer role has no permissions beyond read"
            decision = "deny"
        elif principal.role in (Role.APPROVER, Role.DEPLOYER):
            reason = f"{principal.role.value} role can {action}"
            approval_trace = f"authorized_as_{principal.role.value}"
            decision = "allow"
        else:
            reason = f"Unknown role {principal.role}"
            decision = "deny"

        # Record audit for EVERY decision (C75)
        self.record_audit(AuditEvent(
            actor=principal.name,
            action=action,
            resource=resource,
            decision=decision,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            reason=reason,
            approval_trace=approval_trace,
        ))

        return decision == "allow"

    def check_quota(self, principal: Principal, namespace: str, num_jobs: int) -> bool:
        """Check a namespace quota, auditing the outcome before returning.

        A violation is recorded in the audit log before the request is
        rejected.
        """
        quota = self.quotas.get(namespace, {}).get("max_concurrent_jobs", 1)
        current = self.current_jobs.get(namespace, 0)

        if current + num_jobs > quota:
            self.record_audit(AuditEvent(
                actor=principal.name,
                action="submit_job",
                resource=f"{namespace}/job",
                decision="deny",
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                reason=f"Quota exceeded: {current} + {num_jobs} > {quota}",
                approval_trace="quota_check_failed",
            ))
            return False

        self.current_jobs[namespace] = current + num_jobs
        self.record_audit(AuditEvent(
            actor=principal.name,
            action="submit_job",
            resource=f"{namespace}/job",
            decision="allow",
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            reason=f"Quota OK: {current} + {num_jobs} <= {quota}",
            approval_trace="quota_check_passed",
        ))
        return True

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the audit log as JSON-serializable dicts with ISO timestamps."""
        return [
            {
                "actor": event.actor,
                "action": event.action,
                "resource": event.resource,
                "decision": event.decision,
                "timestamp": event.timestamp.isoformat(),
                "reason": event.reason,
                "approval_trace": event.approval_trace,
            }
            for event in self.audit_log
        ]
