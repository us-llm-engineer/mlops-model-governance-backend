"""X1.1-S2 (C38): published versions are immutable; attempts are audited."""
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


def test_mutate_raises_and_stored_definition_unchanged():
    reg, _, _ = make()
    v = reg.publish(deployer(), NS, base_def())
    before = copy.deepcopy(reg.get(v))
    with pytest.raises(ImmutableVersionError):
        reg.mutate(v, agg="count")
    assert reg.get(v) == before
    assert isinstance(ImmutableVersionError("x"), FeatureError)


def test_mutate_is_audited_as_deny_each_time():
    reg, audit, _ = make()
    v = reg.publish(deployer(), NS, base_def())
    n0 = len(audit.export())
    for i in range(3):
        with pytest.raises(ImmutableVersionError):
            reg.mutate(v, column="c%d" % i)
    new = audit.export()[n0:]
    assert len(new) == 3
    assert all(e["decision"] == "deny" for e in new)
    assert ChainVerifier(SECRET).verify_export(audit.export())


def test_mutate_with_no_or_unknown_changes_still_raises():
    reg, _, _ = make()
    v = reg.publish(deployer(), NS, base_def())
    with pytest.raises(ImmutableVersionError):
        reg.mutate(v)
    with pytest.raises(ImmutableVersionError):
        reg.mutate(v, nonexistent_field=1)
    assert reg.get(v) == base_def()


def test_get_returns_copy_mutation_does_not_leak():
    reg, _, _ = make()
    v = reg.publish(deployer(), NS, base_def())
    d = reg.get(v)
    d["agg"] = "count"
    d["column"] = "hacked"
    assert reg.get(v) == base_def()
    assert reg.get(v) is not reg.get(v)


def test_caller_dict_mutation_after_publish_does_not_leak():
    reg, _, _ = make()
    d = base_def()
    v = reg.publish(deployer(), NS, d)
    d["agg"] = "count"
    assert reg.get(v) == base_def()


def test_versions_list_copy_and_republish_after_mutation_attempt_same_id():
    reg, _, _ = make()
    v = reg.publish(deployer(), NS, base_def())
    lst = reg.versions(NS, "clicks_7d")
    lst.append("fv_bogus")
    assert "fv_bogus" not in reg.versions(NS, "clicks_7d")
    with pytest.raises(ImmutableVersionError):
        reg.mutate(v, agg="count")
    assert reg.publish(deployer(), NS, base_def()) == v
