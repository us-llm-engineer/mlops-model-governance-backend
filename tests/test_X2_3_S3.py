"""X2.3-S3: C59 audit chain -- a full register/deny lifecycle produces a chain
that verifies end-to-end, and tampering with one exported entry is detected.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.experiment_registry import ExperimentRegistry
from mlops.features import FeaturePermissionError, FeatureRegistry
from mlops.lineage import LineageGraph
from mlops.rbac import Principal, Role


def _deployer(name="dep", scope="proj"):
    return Principal(name=name, role=Role.DEPLOYER, scope=scope)


def _viewer(name="view", scope="proj"):
    return Principal(name=name, role=Role.VIEWER, scope=scope)


def _build():
    audit = ChainedAuditStore(secret=b"s3-secret")
    lineage = LineageGraph()
    features = FeatureRegistry(audit, lineage)
    registry = ExperimentRegistry(audit, lineage)
    return audit, lineage, features, registry


def test_full_lifecycle_chain_verifies_with_zero_invalid_indices():
    audit, lineage, features, registry = _build()

    fv1 = features.publish(
        _deployer(),
        "proj",
        {"name": "f1", "source": "dataset:orders@sha256:" + ("b" * 64)},
    )
    registry.register_model_version(_deployer(), "proj/run-1", "v1", [fv1])

    with pytest.raises(FeaturePermissionError):
        registry.register_model_version(_viewer(), "proj/run-1", "v1", [fv1])

    export = audit.export()
    assert len(export) >= 3  # feature.publish allow, register allow, register deny

    verifier = ChainVerifier(secret=b"s3-secret")
    invalid = verifier.invalid_indices(export)
    assert invalid == []
    assert verifier.verify_export(export) is True


def test_chain_contains_both_allow_and_deny_decisions():
    audit, lineage, features, registry = _build()

    fv1 = features.publish(
        _deployer(),
        "proj",
        {"name": "f2", "source": "dataset:orders@sha256:" + ("c" * 64)},
    )
    registry.register_model_version(_deployer(), "proj/run-2", "v1", [fv1])
    with pytest.raises(FeaturePermissionError):
        registry.register_model_version(_viewer(), "proj/run-2", "v1", [fv1])

    export = audit.export()
    decisions = {e["decision"] for e in export if e["action"] == "experiment.register"}
    assert decisions == {"allow", "deny"}


def test_corrupting_one_entry_field_makes_chain_verifier_report_invalid():
    audit, lineage, features, registry = _build()

    fv1 = features.publish(
        _deployer(),
        "proj",
        {"name": "f3", "source": "dataset:orders@sha256:" + ("d" * 64)},
    )
    registry.register_model_version(_deployer(), "proj/run-3", "v1", [fv1])
    with pytest.raises(FeaturePermissionError):
        registry.register_model_version(_viewer(), "proj/run-3", "v1", [fv1])

    export = audit.export()
    verifier = ChainVerifier(secret=b"s3-secret")
    assert verifier.verify_export(export) is True  # sanity: untampered chain is valid

    tampered = [dict(e) for e in export]
    corrupt_index = len(tampered) - 1
    tampered[corrupt_index]["decision"] = (
        "deny" if tampered[corrupt_index]["decision"] == "allow" else "allow"
    )

    invalid = verifier.invalid_indices(tampered)
    assert invalid != []
    assert corrupt_index in invalid
    assert verifier.verify_export(tampered) is False


def test_corruption_at_an_early_entry_invalidates_every_entry_after_it():
    audit, lineage, features, registry = _build()

    fv1 = features.publish(
        _deployer(),
        "proj",
        {"name": "f4", "source": "dataset:orders@sha256:" + ("e" * 64)},
    )
    registry.register_model_version(_deployer(), "proj/run-4", "v1", [fv1])
    registry.register_model_version(_deployer(), "proj/run-5", "v1", [fv1])

    export = audit.export()
    assert len(export) >= 3

    verifier = ChainVerifier(secret=b"s3-secret")
    tampered = [dict(e) for e in export]
    tampered[0]["resource"] = tampered[0]["resource"] + "-tampered"

    invalid = verifier.invalid_indices(tampered)
    # Corrupting entry 0 cascades: every subsequent entry's expected prev_sig changes too.
    assert invalid == list(range(len(tampered)))
