"""X1.1-S4 (C40): lineage from source datasets through feature versions to models."""
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


def _by_type(lg, t):
    return [n for n in lg.nodes.values() if n.node_type == t]


def _fv_node(lg, vid):
    hits = [n for n in _by_type(lg, "feature_version") if vid in n.properties.values()]
    assert len(hits) == 1
    return hits[0]


def test_publish_adds_dataset_to_feature_version_edge():
    reg, _, lg = make(lineage=True)
    v = reg.publish(deployer(), NS, base_def())
    fv = _fv_node(lg, v)
    src = base_def()["source"]
    assert src in fv.properties.values()
    ds = [n for n in _by_type(lg, "dataset") if n.properties.get("ref") == src]
    assert len(ds) == 1
    assert any(a == ds[0].node_id and b == fv.node_id for a, b, _ in lg.edges)
    assert ds[0].node_id in lg.trace_lineage_backward(fv.node_id)


def test_link_model_and_feature_versions_for_model():
    reg, _, lg = make(lineage=True)
    v1 = reg.publish(deployer(), NS, base_def())
    v2 = reg.publish(deployer(), NS, base_def(name="views_7d", column="views"))
    v3 = reg.publish(deployer(), NS, base_def(name="unrelated", column="z"))
    reg.link_model([v1, v2], "model-1")
    got = reg.feature_versions_for_model("model-1")
    assert set(got) == {v1, v2}
    assert v3 not in got
    assert len(got) == 2


def test_two_models_sharing_a_version():
    reg, _, lg = make(lineage=True)
    v1 = reg.publish(deployer(), NS, base_def())
    v2 = reg.publish(deployer(), NS, base_def(name="b", column="b"))
    v3 = reg.publish(deployer(), NS, base_def(name="c", column="c"))
    reg.link_model([v1, v2], "m-a")
    reg.link_model([v1, v3], "m-b")
    assert set(reg.feature_versions_for_model("m-a")) == {v1, v2}
    assert set(reg.feature_versions_for_model("m-b")) == {v1, v3}


def test_republish_same_content_does_not_duplicate_lineage_result():
    reg, _, lg = make(lineage=True)
    v = reg.publish(deployer(), NS, base_def())
    reg.publish(deployer(), NS, base_def())
    reg.link_model([v], "m")
    assert reg.feature_versions_for_model("m") == [v]


def test_no_lineage_argument_does_not_crash_on_publish():
    reg, audit, _ = make(lineage=False)
    v = reg.publish(deployer(), NS, base_def())
    assert reg.get(v) == base_def()
    assert len(audit.export()) == 1
    try:
        reg.link_model([v], "m")
    except FeatureError:
        pass
    assert reg.latest(NS, "clicks_7d") == v
