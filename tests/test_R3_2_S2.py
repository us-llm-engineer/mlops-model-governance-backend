"""R3.2-S2: MLflow alias lifecycle for mirrored model stages."""
import os
import sys

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.model_stages import Stage


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(tmp_path))


def _client(tmp_path):
    return MlflowClient(tracking_uri=_uri(tmp_path))


def _missing_alias(client, model, alias):
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(model, alias)


def test_staging_stage_uses_only_staging_alias(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("alias-staging", "1", Stage.STAGING)
    client = _client(tmp_path)
    assert str(client.get_model_version_by_alias("alias-staging", "staging").version) == "1"
    _missing_alias(client, "alias-staging", "production")


def test_production_stage_uses_only_production_alias(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("alias-production", "1", Stage.PRODUCTION)
    client = _client(tmp_path)
    assert str(client.get_model_version_by_alias("alias-production", "production").version) == "1"
    _missing_alias(client, "alias-production", "staging")


def test_moving_staging_alias_does_not_move_production_alias(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("alias-split", "1", Stage.PRODUCTION)
    bridge.mirror_stage_transition("alias-split", "2", Stage.STAGING)
    client = _client(tmp_path)
    assert str(client.get_model_version_by_alias("alias-split", "production").version) == "1"
    assert str(client.get_model_version_by_alias("alias-split", "staging").version) == "2"


def test_registered_clears_both_aliases_when_both_exist(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("alias-registered", "1", Stage.PRODUCTION)
    bridge.mirror_stage_transition("alias-registered", "2", Stage.STAGING)
    bridge.mirror_stage_transition("alias-registered", "2", Stage.REGISTERED)
    client = _client(tmp_path)
    _missing_alias(client, "alias-registered", "staging")
    _missing_alias(client, "alias-registered", "production")


def test_archived_clears_both_aliases_when_both_exist(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("alias-archived", "1", Stage.PRODUCTION)
    bridge.mirror_stage_transition("alias-archived", "2", Stage.STAGING)
    bridge.mirror_stage_transition("alias-archived", "2", Stage.ARCHIVED)
    client = _client(tmp_path)
    _missing_alias(client, "alias-archived", "staging")
    _missing_alias(client, "alias-archived", "production")


def test_same_stage_on_fresh_bridge_is_idempotent(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("alias-repeat", "1", Stage.PRODUCTION)
    _bridge(tmp_path).mirror_stage_transition("alias-repeat", "1", Stage.PRODUCTION)
    assert str(_client(tmp_path).get_model_version_by_alias("alias-repeat", "production").version) == "1"


def test_alias_names_are_scoped_per_registered_model(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.mirror_stage_transition("alias-model-a", "1", Stage.STAGING)
    bridge.mirror_stage_transition("alias-model-b", "1", Stage.PRODUCTION)
    client = _client(tmp_path)
    assert str(client.get_model_version_by_alias("alias-model-a", "staging").version) == "1"
    assert str(client.get_model_version_by_alias("alias-model-b", "production").version) == "1"
    _missing_alias(client, "alias-model-a", "production")
    _missing_alias(client, "alias-model-b", "staging")


def test_registered_stage_on_new_model_creates_no_alias(tmp_path):
    _bridge(tmp_path).mirror_stage_transition("alias-none", "1", Stage.REGISTERED)
    client = _client(tmp_path)
    assert client.get_registered_model("alias-none") is not None
    _missing_alias(client, "alias-none", "staging")
    _missing_alias(client, "alias-none", "production")
