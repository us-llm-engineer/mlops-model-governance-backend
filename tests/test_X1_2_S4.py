"""X1.2-S4 (C47): data contract enforced atomically on ingest."""
from mlops.gates import DataContractGate
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exec"))

from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.rbac import Principal, Role  # noqa: E402
from mlops.features import FeatureRegistry, ContractViolation, FeatureError  # noqa: E402
from mlops import feature_store as fs_mod  # noqa: E402
from mlops.feature_store import FeatureStore, evaluate_feature  # noqa: E402

NS = "team-a"
SRC = "dataset:events@sha256:" + "ab" * 32


def make_store(defs, contract=None):
    """defs: list of (name, agg, window_seconds). Returns FeatureStore."""
    audit = ChainedAuditStore(b"secret")
    reg = FeatureRegistry(audit)
    p = Principal("dep", Role.DEPLOYER, NS)
    for name, agg, win in defs:
        reg.publish(p, NS, {
            "name": name, "entity_key": "user", "source": SRC, "column": "v",
            "agg": agg, "window_seconds": win, "dtype": "float",
        })
    return FeatureStore(reg, contract)


def rows_of(*triples):
    return [{"entity": e, "event_ts": t, "value": v} for e, t, v in triples]



def gate():
    return DataContractGate(["entity", "event_ts", "value"], {"value": (0.0, 100.0)})


def test_ingest_returns_stored_count_and_rows_visible():
    s = make_store([("f", "last", None)], gate())
    assert s.ingest(NS, "f", rows_of(("u", 1, 5.0), ("u", 2, 6.0))) == {"stored": 2}
    assert s.get_offline(NS, "f", "u", 2) == 6.0
    assert s.ingest(NS, "f", []) == {"stored": 0}


def test_missing_field_names_field_and_raises_contract_violation():
    s = make_store([("f", "last", None)], gate())
    bad = {"entity": "u", "event_ts": 3}  # no value
    with pytest.raises(ContractViolation) as ei:
        s.ingest(NS, "f", [bad])
    assert "value" in str(ei.value)
    assert isinstance(ei.value, FeatureError)


def test_out_of_range_names_field():
    s = make_store([("f", "last", None)], gate())
    with pytest.raises(ContractViolation) as ei:
        s.ingest(NS, "f", rows_of(("u", 3, 500.0)))
    assert "value" in str(ei.value)


def test_bad_row_in_middle_is_atomic_nothing_stored():
    s = make_store([("f", "last", None), ("c", "count", None)], gate())
    batch = rows_of(("u", 1, 5.0), ("u", 2, 6.0)) + [
        {"entity": "u", "event_ts": 3, "value": None}] + rows_of(("u", 4, 7.0))
    with pytest.raises(ContractViolation):
        s.ingest(NS, "f", batch)
    assert s.get_offline(NS, "f", "u", 100) is None
    assert s.get_online(NS, "f", "u", 100) is None


def test_atomic_failure_does_not_disturb_previously_stored_rows():
    s = make_store([("c", "count", None)], gate())
    s.ingest(NS, "c", rows_of(("u", 1, 1.0)))
    with pytest.raises(ContractViolation):
        s.ingest(NS, "c", rows_of(("u", 2, 1.0), ("u", 3, 999.0)))
    assert s.get_offline(NS, "c", "u", 100) == 1
    assert s.ingest(NS, "c", rows_of(("u", 4, 1.0))) == {"stored": 1}
    assert s.get_offline(NS, "c", "u", 100) == 2


def test_unknown_feature_errors_cleanly():
    s = make_store([("f", "last", None)], gate())
    with pytest.raises((FeatureError, KeyError, ValueError)) as ei:
        s.ingest(NS, "nope", rows_of(("u", 1, 1.0)))
    assert not isinstance(ei.value, (AttributeError, TypeError))
    assert "nope" in str(ei.value)
