"""X2.3-S4: C60 RBAC on registration -- DEPLOYER/APPROVER with matching scope
succeed, VIEWER and wrong-scope principals are denied, and every attempt
(success or failure) produces exactly one new audit entry.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.experiment_registry import ExperimentRegistry
from mlops.features import FeaturePermissionError
from mlops.lineage import LineageGraph
from mlops.rbac import Principal, Role


def _principal(role, scope):
    return Principal(name="p-%s-%s" % (role.value, scope), role=role, scope=scope)


def _build():
    audit = ChainedAuditStore(secret=b"s4-secret")
    lineage = LineageGraph()
    registry = ExperimentRegistry(audit, lineage)
    return audit, lineage, registry


def test_deployer_matching_scope_succeeds_and_audits_allow():
    audit, lineage, registry = _build()
    before = len(audit.export())

    mv_id = registry.register_model_version(
        _principal(Role.DEPLOYER, "proj"), "proj/run-1", "v1", []
    )

    after = audit.export()
    assert isinstance(mv_id, str)
    assert len(after) - before == 1
    assert after[-1]["decision"] == "allow"
    assert after[-1]["action"] == "experiment.register"


def test_approver_matching_scope_succeeds_and_audits_allow():
    audit, lineage, registry = _build()
    before = len(audit.export())

    mv_id = registry.register_model_version(
        _principal(Role.APPROVER, "proj"), "proj/run-2", "v1", []
    )

    after = audit.export()
    assert isinstance(mv_id, str)
    assert len(after) - before == 1
    assert after[-1]["decision"] == "allow"


def test_viewer_denied_no_lineage_node_added_and_audits_deny():
    audit, lineage, registry = _build()
    before_audit = len(audit.export())
    before_nodes = len(lineage.nodes)

    with pytest.raises(FeaturePermissionError):
        registry.register_model_version(
            _principal(Role.VIEWER, "proj"), "proj/run-3", "v1", []
        )

    after_audit = audit.export()
    assert len(after_audit) - before_audit == 1
    assert after_audit[-1]["decision"] == "deny"
    assert len(lineage.nodes) == before_nodes


def test_wrong_scope_matching_role_denied():
    audit, lineage, registry = _build()
    before = len(audit.export())

    with pytest.raises(FeaturePermissionError):
        registry.register_model_version(
            _principal(Role.DEPLOYER, "other-namespace"), "proj/run-4", "v1", []
        )

    after = audit.export()
    assert len(after) - before == 1
    assert after[-1]["decision"] == "deny"


def test_every_attempt_produces_exactly_one_audit_entry_across_a_sequence():
    audit, lineage, registry = _build()
    attempts = [
        (_principal(Role.DEPLOYER, "proj"), "proj/run-5"),
        (_principal(Role.VIEWER, "proj"), "proj/run-6"),
        (_principal(Role.APPROVER, "proj"), "proj/run-7"),
        (_principal(Role.DEPLOYER, "wrong-scope"), "proj/run-8"),
    ]

    for principal, run_id in attempts:
        before = len(audit.export())
        try:
            registry.register_model_version(principal, run_id, "v1", [])
        except FeaturePermissionError:
            pass
        after = len(audit.export())
        assert after - before == 1, "attempt for %s produced %d entries, not 1" % (
            run_id,
            after - before,
        )

    assert len(audit.export()) == len(attempts)
