"""X1.1-S3 (C39): RBAC-gated publish with a verifiable audit chain."""
import copy
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.lineage import LineageGraph
from mlops.rbac import Principal, Role
from mlops.features import (
    FeatureRegistry,
    FeatureError,
    MutableReferenceError,
    ImmutableVersionError,
    FeaturePermissionError,
)

SECRET = b"x1-1-secret"
H1 = "a" * 64
H2 = "b" * 64
NS = "team-a"


def base_def(**over):
    d = {
        "name": "clicks_7d",
        "entity_key": "user_id",
        "source": "dataset:events@sha256:" + H1,
        "column": "clicks",
        "agg": "sum",
        "window_seconds": 604800,
        "dtype": "int",
    }
    d.update(over)
    return d


def deployer(scope=NS, name="dana"):
    return Principal(name=name, role=Role.DEPLOYER, scope=scope)


def make(lineage=False):
    audit = ChainedAuditStore(SECRET)
    lg = LineageGraph() if lineage else None
    return FeatureRegistry(audit, lg) if lineage else FeatureRegistry(audit), audit, lg


@pytest.mark.parametrize("role", [Role.DEPLOYER, Role.APPROVER])
def test_allowed_roles_publish_and_audit_allow(role):
    reg, audit, _ = make()
    p = Principal("alice", role, NS)
    v = reg.publish(p, NS, base_def())
    entries = audit.export()
    assert len(entries) == 1
    e = entries[0]
    assert (e["actor"], e["action"], e["resource"], e["decision"]) == (
        "alice", "feature.publish", NS + "/clicks_7d", "allow")
    assert reg.get(v) == base_def()


def test_viewer_denied_audited_and_nothing_stored():
    reg, audit, _ = make()
    p = Principal("vic", Role.VIEWER, NS)
    with pytest.raises(FeaturePermissionError):
        reg.publish(p, NS, base_def())
    entries = audit.export()
    assert len(entries) == 1
    e = entries[0]
    assert (e["actor"], e["action"], e["resource"], e["decision"]) == (
        "vic", "feature.publish", NS + "/clicks_7d", "deny")
    assert reg.versions(NS, "clicks_7d") == []
    with pytest.raises(Exception):
        reg.latest(NS, "clicks_7d")


@pytest.mark.parametrize("role", [Role.DEPLOYER, Role.APPROVER])
def test_wrong_scope_denied(role):
    reg, audit, _ = make()
    p = Principal("mallory", role, "team-b")
    with pytest.raises(FeaturePermissionError):
        reg.publish(p, NS, base_def())
    e = audit.export()
    assert len(e) == 1 and e[0]["decision"] == "deny" and e[0]["actor"] == "mallory"
    assert reg.versions(NS, "clicks_7d") == []


def test_mixed_sequence_one_entry_per_attempt_and_chain_verifies():
    reg, audit, _ = make()
    reg.publish(deployer(name="d1"), NS, base_def())
    with pytest.raises(FeaturePermissionError):
        reg.publish(Principal("v", Role.VIEWER, NS), NS, base_def(name="x"))
    reg.publish(Principal("ap", Role.APPROVER, NS), NS, base_def(name="y"))
    with pytest.raises(FeaturePermissionError):
        reg.publish(deployer(scope="other"), NS, base_def(name="z"))
    ex = audit.export()
    assert [e["decision"] for e in ex] == ["allow", "deny", "allow", "deny"]
    assert [e["actor"] for e in ex] == ["d1", "v", "ap", "dana"]
    assert [e["resource"] for e in ex] == [NS + "/clicks_7d", NS + "/x", NS + "/y", NS + "/z"]
    ver = ChainVerifier(SECRET)
    assert ver.verify_export(ex)
    tampered = copy.deepcopy(ex)
    tampered[1]["decision"] = "allow"
    assert ver.invalid_indices(tampered)


def test_republish_identical_content_is_still_audited():
    reg, audit, _ = make()
    reg.publish(deployer(), NS, base_def())
    reg.publish(deployer(), NS, base_def())
    assert len(audit.export()) == 2


def test_unauthorized_with_mutable_ref_never_stores_and_is_audited_once():
    reg, audit, _ = make()
    try:
        reg.publish(Principal("v", Role.VIEWER, NS), NS, base_def(source="dataset:e@latest"))
    except FeatureError:
        pass
    assert len(audit.export()) == 1
    assert reg.versions(NS, "clicks_7d") == []
