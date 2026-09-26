"""X1.1-S1 (C37): content-addressed versioning and mutable-reference rejection."""
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


def test_same_content_same_id_and_key_order_irrelevant():
    reg, _, _ = make()
    a = reg.publish(deployer(), NS, base_def())
    reordered = dict(reversed(list(base_def().items())))
    b = reg.publish(deployer(), NS, reordered)
    assert a == b
    assert a.startswith("fv_") and len(a) == 3 + 16
    int(a[3:], 16)
    assert reg.versions(NS, "clicks_7d") == [a] or reg.versions(NS, "clicks_7d") == [a, a]


@pytest.mark.parametrize("field,new", [
    ("entity_key", "account_id"),
    ("source", "dataset:events@sha256:" + H2),
    ("column", "views"),
    ("agg", "count"),
    ("window_seconds", 3600),
    ("window_seconds", None),
    ("dtype", "float"),
])
def test_any_changed_field_yields_new_id(field, new):
    reg, _, _ = make()
    a = reg.publish(deployer(), NS, base_def())
    b = reg.publish(deployer(), NS, base_def(**{field: new}))
    assert a != b


def test_new_id_for_changed_name_and_ids_distinct_across_many():
    reg, _, _ = make()
    ids = {reg.publish(deployer(), NS, base_def(name="f%d" % i)) for i in range(20)}
    assert len(ids) == 20


@pytest.mark.parametrize("src", [
    "dataset:events@latest",
    "dataset:events@main",
    "dataset:events@sha256:abc123",
    "dataset:events@sha256:" + "g" * 64,
    "dataset:events@sha256:" + "a" * 63,
    "dataset:events@sha256:" + "a" * 65,
    "dataset:events",
])
def test_mutable_or_malformed_reference_rejected_and_not_stored(src):
    reg, audit, _ = make()
    with pytest.raises(MutableReferenceError):
        reg.publish(deployer(), NS, base_def(source=src))
    assert isinstance(MutableReferenceError("x"), FeatureError)
    assert reg.versions(NS, "clicks_7d") == []


def test_versions_in_publish_order_and_latest_tracks_newest():
    reg, _, _ = make()
    v1 = reg.publish(deployer(), NS, base_def())
    v2 = reg.publish(deployer(), NS, base_def(agg="count"))
    v3 = reg.publish(deployer(), NS, base_def(dtype="float"))
    assert reg.versions(NS, "clicks_7d") == [v1, v2, v3]
    assert reg.latest(NS, "clicks_7d") == v3
    other = reg.publish(deployer(), NS, base_def(name="other"))
    assert reg.versions(NS, "clicks_7d") == [v1, v2, v3]
    assert reg.versions(NS, "other") == [other]
    assert reg.get(v1)["agg"] == "sum"
