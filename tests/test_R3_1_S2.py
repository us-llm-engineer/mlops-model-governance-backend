"""R3.1-S2: typed run retrieval and MLflow error translation boundaries."""
import os
import sys

import pytest
from mlflow.exceptions import MlflowException

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops import kernel
from mlops.experiments import LogEntry


def _uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def _bridge(tmp_path):
    from mlops.interop.mlflow_bridge import MlflowBridge

    return MlflowBridge(_uri(tmp_path))


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        pytest.param(0.125, float, id="float"),
        pytest.param(12, int, id="int"),
        pytest.param(False, bool, id="bool"),
        pytest.param("001", str, id="numeric-looking-string"),
        pytest.param("false", str, id="boolean-looking-string"),
        pytest.param("baseline-v3", str, id="plain-string"),
    ],
)
def test_get_mirrored_run_restores_declared_param_type(tmp_path, value, expected_type):
    bridge = _bridge(tmp_path)
    bridge.mirror_experiment_run(
        [LogEntry(kind="param", name="value", value=value, step=None)],
        f"typed-{type(value).__name__}-{value}",
        "typed",
    )
    restored = bridge.get_mirrored_run(f"typed-{type(value).__name__}-{value}")["params"]["value"]
    assert type(restored) is expected_type
    assert restored == value


def test_conflicting_param_from_fresh_bridge_raises_kernel_conflict(tmp_path):
    _bridge(tmp_path).mirror_experiment_run(
        [LogEntry(kind="param", name="lr", value=0.01, step=None)], "type-conflict", "typed"
    )
    with pytest.raises(kernel.Conflict) as caught:
        _bridge(tmp_path).mirror_experiment_run(
            [LogEntry(kind="param", name="lr", value=0.02, step=None)], "type-conflict", "typed"
        )
    assert type(caught.value) is kernel.Conflict


def test_provider_read_error_is_translated_and_redacts_provider_detail(tmp_path, monkeypatch):
    """A provider read failure must not disclose provider text or request material.

    ``get_mirrored_run`` starts by listing experiments, so faulting that client
    boundary exercises a real public read path rather than an implementation
    helper.  The sentinel resembles a credential: a public kernel error may
    identify the safe operation category, but must never relay the provider's
    message (which can contain transport details or request values).
    """
    bridge = _bridge(tmp_path)
    sentinel = "mlflow-secret://token=do-not-disclose"

    def provider_read_failure(*_args, **_kwargs):
        raise MlflowException(f"provider read exploded: {sentinel}")

    monkeypatch.setattr(bridge._client, "search_experiments", provider_read_failure)
    with pytest.raises(kernel.ValidationFailed) as caught:
        bridge.get_mirrored_run("read-redaction")

    public_message = str(caught.value)
    assert type(caught.value) is kernel.ValidationFailed
    assert sentinel not in public_message
    assert "provider read exploded" not in public_message
