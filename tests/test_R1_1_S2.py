"""
R1.1-S2: RBAC/Quota Enforcement (Authorization/Privacy)

Test suite for C2: RBAC decisions are logged with actor, action, resource, decision,
timestamp, and approval trace. No silent failures. Quota violations are recorded before rejection.

Dimension: authorization/privacy cases
Mutation targets:
  - Missing quota enforcement check (mutation: remove quota limit check)
  - Silent quota exceeded (no audit log entry) (mutation: skip audit on quota exceeded)
  - Allowing unauthorized role to perform action (mutation: remove role check)
  - Missing actor/action audit trail (mutation: remove audit logging)
"""

import pytest
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum


class Role(Enum):
    VIEWER = "viewer"
    APPROVER = "approver"
    DEPLOYER = "deployer"


class Action(Enum):
    LIST_PIPELINES = "list_pipelines"
    SUBMIT_PIPELINE = "submit_pipeline"
    APPROVE_PIPELINE = "approve_pipeline"
    DEPLOY_PIPELINE = "deploy_pipeline"


@dataclass
class AuditLog:
    """Audit log entry."""
    actor: str
    action: str
    resource: str
    decision: str  # "allow" or "deny"
    timestamp: float
    approval_trace: str
    reason: Optional[str] = None


class RBACAuthority:
    """RBAC enforcement engine."""

    # Role → allowed actions
    ROLE_PERMISSIONS = {
        Role.VIEWER: {Action.LIST_PIPELINES},
        Role.APPROVER: {Action.LIST_PIPELINES, Action.APPROVE_PIPELINE},
        Role.DEPLOYER: {Action.LIST_PIPELINES, Action.SUBMIT_PIPELINE, Action.DEPLOY_PIPELINE}
    }

    def __init__(self):
        self.audit_logs: List[AuditLog] = []
        self.principal_roles: Dict[str, List[Role]] = {}
        self.quotas: Dict[str, int] = {}  # principal → max concurrent jobs

    def assign_role(self, principal: str, role: Role, namespace: str = "default"):
        """Assign role to principal."""
        if principal not in self.principal_roles:
            self.principal_roles[principal] = []
        self.principal_roles[principal].append(role)

    def set_quota(self, principal: str, max_jobs: int):
        """Set concurrent job quota for principal."""
        self.quotas[principal] = max_jobs

    def get_current_job_count(self, principal: str) -> int:
        """Mock: get current running job count for principal."""
        # In real impl, would query orchestration layer
        return 0

    def check_authorization(self, principal: str, action: Action, resource: str = "") -> bool:
        """Check if principal is authorized for action. Logs decision."""
        if principal not in self.principal_roles:
            self._log_decision(principal, action, resource, "deny", "principal_not_found")
            return False

        roles = self.principal_roles[principal]
        allowed_actions = set()
        for role in roles:
            allowed_actions.update(self.ROLE_PERMISSIONS.get(role, set()))

        if action not in allowed_actions:
            self._log_decision(principal, action, resource, "deny", "insufficient_permissions")
            return False

        self._log_decision(principal, action, resource, "allow", "role_permits_action")
        return True

    def check_quota(self, principal: str, action: Action, resource: str = "") -> bool:
        """Check if principal has not exceeded quota. Logs decision."""
        if action != Action.SUBMIT_PIPELINE:
            return True  # Quota only applies to submissions

        if principal not in self.quotas:
            self._log_decision(principal, action, resource, "allow", "quota_not_set")
            return True

        current = self.get_current_job_count(principal)
        limit = self.quotas[principal]

        if current >= limit:
            self._log_decision(principal, action, resource, "deny", f"quota_exceeded:{current}/{limit}", reason=f"current={current}, limit={limit}")
            return False

        self._log_decision(principal, action, resource, "allow", f"within_quota:{current}/{limit}")
        return True

    def _log_decision(self, actor: str, action: Action, resource: str, decision: str, trace: str, reason: Optional[str] = None):
        """Log auth decision to audit trail."""
        log_entry = AuditLog(
            actor=actor,
            action=action.value,
            resource=resource,
            decision=decision,
            timestamp=time.time(),
            approval_trace=trace,
            reason=reason
        )
        self.audit_logs.append(log_entry)

    def get_audit_logs(self, actor: Optional[str] = None, decision: Optional[str] = None) -> List[AuditLog]:
        """Query audit logs."""
        logs = self.audit_logs
        if actor:
            logs = [l for l in logs if l.actor == actor]
        if decision:
            logs = [l for l in logs if l.decision == decision]
        return logs


class TestRBACAuthorizationSuccess:
    """Success cases: authorized actors can perform actions."""

    def test_deployer_can_submit_pipeline(self):
        """Case 1: Deployer role is authorized to submit pipelines."""
        rbac = RBACAuthority()
        rbac.assign_role("alice@example.com", Role.DEPLOYER)

        authorized = rbac.check_authorization("alice@example.com", Action.SUBMIT_PIPELINE, "fraud-model")

        assert authorized is True
        assert len(rbac.audit_logs) == 1
        assert rbac.audit_logs[0].decision == "allow"

    def test_approver_can_approve_pipeline(self):
        """Case 2: Approver role is authorized to approve pipelines."""
        rbac = RBACAuthority()
        rbac.assign_role("bob@example.com", Role.APPROVER)

        authorized = rbac.check_authorization("bob@example.com", Action.APPROVE_PIPELINE, "pricing-model")

        assert authorized is True
        assert rbac.audit_logs[0].decision == "allow"

    def test_viewer_can_list_pipelines(self):
        """Case 3: Viewer role can list pipelines (read-only)."""
        rbac = RBACAuthority()
        rbac.assign_role("charlie@example.com", Role.VIEWER)

        authorized = rbac.check_authorization("charlie@example.com", Action.LIST_PIPELINES)

        assert authorized is True
        assert rbac.audit_logs[0].decision == "allow"


class TestRBACAuthorizationBoundary:
    """Boundary cases: unauthorized attempts are denied and logged."""

    def test_viewer_cannot_submit_pipeline(self):
        """Case 4: Viewer role is denied pipeline submission."""
        rbac = RBACAuthority()
        rbac.assign_role("viewer_user", Role.VIEWER)

        authorized = rbac.check_authorization("viewer_user", Action.SUBMIT_PIPELINE, "model1")

        assert authorized is False
        assert rbac.audit_logs[0].decision == "deny"
        assert "insufficient_permissions" in rbac.audit_logs[0].approval_trace

    def test_unknown_principal_denied(self):
        """Case 5: Unknown principal (no role assigned) is denied."""
        rbac = RBACAuthority()

        authorized = rbac.check_authorization("unknown@example.com", Action.LIST_PIPELINES)

        assert authorized is False
        assert rbac.audit_logs[0].decision == "deny"

    def test_approver_cannot_deploy(self):
        """Case 6: Approver role is denied deploy action."""
        rbac = RBACAuthority()
        rbac.assign_role("approver_user", Role.APPROVER)

        authorized = rbac.check_authorization("approver_user", Action.DEPLOY_PIPELINE, "model")

        assert authorized is False
        assert rbac.audit_logs[0].decision == "deny"


class TestQuotaEnforcement:
    """Quota enforcement cases: limits are checked and violations logged."""

    def test_quota_exceeded_denied_with_log(self):
        """Case 7: Quota exceeded triggers denial and audit log."""
        rbac = RBACAuthority()
        rbac.assign_role("power_user", Role.DEPLOYER)
        rbac.set_quota("power_user", max_jobs=2)

        # Mock current job count at quota limit
        with patch.object(rbac, 'get_current_job_count', return_value=2):
            allowed = rbac.check_quota("power_user", Action.SUBMIT_PIPELINE, "model")

        assert allowed is False
        # Verify audit log was written
        deny_logs = rbac.get_audit_logs(decision="deny")
        assert len(deny_logs) == 1
        assert "quota_exceeded" in deny_logs[0].approval_trace

    def test_quota_within_limit_allowed(self):
        """Case 8: Submission allowed when under quota."""
        rbac = RBACAuthority()
        rbac.assign_role("user", Role.DEPLOYER)
        rbac.set_quota("user", max_jobs=5)

        with patch.object(rbac, 'get_current_job_count', return_value=2):
            allowed = rbac.check_quota("user", Action.SUBMIT_PIPELINE, "model")

        assert allowed is True
        allow_logs = rbac.get_audit_logs(decision="allow")
        assert len(allow_logs) == 1

    def test_no_quota_set_allows_submission(self):
        """Case 9: User without quota set can submit (no limit)."""
        rbac = RBACAuthority()
        rbac.assign_role("user", Role.DEPLOYER)
        # No quota set

        allowed = rbac.check_quota("user", Action.SUBMIT_PIPELINE, "model")

        assert allowed is True

    def test_audit_log_contains_reason(self):
        """Case 10: Quota denied audit log includes reason (current/limit)."""
        rbac = RBACAuthority()
        rbac.set_quota("user", max_jobs=3)

        with patch.object(rbac, 'get_current_job_count', return_value=3):
            rbac.check_quota("user", Action.SUBMIT_PIPELINE, "model")

        logs = rbac.get_audit_logs(actor="user")
        assert len(logs) > 0
        assert logs[0].reason is not None
        assert "current=3" in logs[0].reason
        assert "limit=3" in logs[0].reason
