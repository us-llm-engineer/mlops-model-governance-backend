"""R3.2-S1: strict, fail-closed allocation of MLflow model versions."""
import os
import sys

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops import kernel
from mlops.model_stages import Stage


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(tmp_path))


def _client(tmp_path):
    return MlflowClient(tracking_uri=_uri(tmp_path))


def _assert_model_absent(client, name):
    with pytest.raises(MlflowException):
        client.get_registered_model(name)


def test_first_canonical_version_one_allocates_version_one(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("sequence-one", "1", Stage.STAGING)
    assert str(_client(tmp_path).get_model_version_by_alias("sequence-one", "staging").version) == "1"


def test_first_version_two_conflicts_without_creating_model(tmp_path):
    with pytest.raises(kernel.Conflict):
        _bridge(tmp_path).mirror_stage_transition("sequence-two", "2", Stage.STAGING)
    _assert_model_absent(_client(tmp_path), "sequence-two")


def test_first_leading_zero_version_seven_conflicts_without_creating_model(tmp_path):
    # The oracle is sequence-based: "007" canonicalizes to 7, not MLflow's
    # auto-assigned 1.  A conflict must leave no partial model behind.
    with pytest.raises(kernel.Conflict):
        _bridge(tmp_path).mirror_stage_transition("sequence-seven", "007", Stage.STAGING)
    _assert_model_absent(_client(tmp_path), "sequence-seven")


def test_first_zero_version_conflicts_without_creating_model(tmp_path):
    with pytest.raises(kernel.Conflict):
        _bridge(tmp_path).mirror_stage_transition("sequence-zero", "0", Stage.STAGING)
    _assert_model_absent(_client(tmp_path), "sequence-zero")


def test_immediate_next_version_allocates_after_existing_version(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("sequence-next", "1", Stage.STAGING)
    bridge.mirror_stage_transition("sequence-next", "2", Stage.PRODUCTION)
    assert str(_client(tmp_path).get_model_version_by_alias("sequence-next", "production").version) == "2"


def test_skipped_version_conflict_preserves_existing_alias_and_counter(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("sequence-skip", "1", Stage.STAGING)
    with pytest.raises(kernel.Conflict):
        bridge.mirror_stage_transition("sequence-skip", "3", Stage.STAGING)
    client = _client(tmp_path)
    assert str(client.get_model_version_by_alias("sequence-skip", "staging").version) == "1"
    bridge.mirror_stage_transition("sequence-skip", "2", Stage.STAGING)
    assert str(client.get_model_version_by_alias("sequence-skip", "staging").version) == "2"


def test_existing_version_is_found_using_leading_zero_spelling(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("sequence-normalize", "1", Stage.STAGING)
    bridge.mirror_stage_transition("sequence-normalize", "001", Stage.PRODUCTION)
    assert str(_client(tmp_path).get_model_version_by_alias("sequence-normalize", "production").version) == "1"


def test_whitespace_version_is_validation_failed_not_sequence_conflict(tmp_path):
    with pytest.raises(kernel.ValidationFailed):
        _bridge(tmp_path).mirror_stage_transition("sequence-whitespace", " 1", Stage.STAGING)
