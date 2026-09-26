"""X2.3-S1: C57 registry+lineage -- model version id shape, feature-version edges,
unknown feature_version_id raises KeyError, shared feature version across two runs.
"""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.experiment_registry import ExperimentRegistry
from mlops.features import FeatureRegistry
from mlops.lineage import LineageGraph
from mlops.rbac import Principal, Role


def _deployer(name="dep", scope="proj"):
    return Principal(name=name, role=Role.DEPLOYER, scope=scope)


def _publish_feature(registry, namespace="proj", name="f1", value=1):
    definition = {
        "name": name,
        "source": "dataset:orders@sha256:" + ("a" * 64),
        "value": value,
    }
    return registry.publish(_deployer(scope=namespace), namespace, definition)


def _build():
    audit = ChainedAuditStore(secret=b"s1-secret")
    lineage = LineageGraph()
    features = FeatureRegistry(audit, lineage)
    registry = ExperimentRegistry(audit, lineage)
    return audit, lineage, features, registry


def test_model_version_id_has_documented_shape():
    audit, lineage, features, registry = _build()
    mv_id = registry.register_model_version(_deployer(), "proj/run-1", "v1", [])
    assert isinstance(mv_id, str)
    assert mv_id.startswith("mv_")
    expected = "mv_" + hashlib.sha256(("proj/run-1" + "v1").encode()).hexdigest()[:16]
    assert mv_id == expected


def test_edges_added_and_feature_versions_for_model_matches_published_set():
    audit, lineage, features, registry = _build()
    fv1 = _publish_feature(features, name="f1")
    fv2 = _publish_feature(features, name="f2")

    mv_id = registry.register_model_version(
        _deployer(), "proj/run-1", "v1", [fv1, fv2]
    )

    linked = registry.feature_versions_for_model(mv_id)
    assert set(linked) == {fv1, fv2}
    assert len(linked) == 2


def test_unknown_feature_version_id_raises_keyerror():
    audit, lineage, features, registry = _build()
    with pytest.raises(KeyError):
        registry.register_model_version(
            _deployer(), "proj/run-2", "v1", ["fv_doesnotexist0000"]
        )


def test_two_model_versions_from_different_runs_share_one_feature_version():
    audit, lineage, features, registry = _build()
    fv_shared = _publish_feature(features, name="shared")

    mv_a = registry.register_model_version(
        _deployer(), "proj/run-a", "v1", [fv_shared]
    )
    mv_b = registry.register_model_version(
        _deployer(), "proj/run-b", "v1", [fv_shared]
    )

    assert mv_a != mv_b
    assert fv_shared in registry.feature_versions_for_model(mv_a)
    assert fv_shared in registry.feature_versions_for_model(mv_b)


def test_registering_with_no_feature_versions_adds_node_with_empty_lineage():
    audit, lineage, features, registry = _build()
    mv_id = registry.register_model_version(_deployer(), "proj/run-empty", "v1", [])
    assert registry.feature_versions_for_model(mv_id) == []
