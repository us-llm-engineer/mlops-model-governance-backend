"""X1.3-S4: package exports, version, end-to-end parity->gate journey, smoke_test coverage."""
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXEC = os.path.join(HERE, "..", "exec")
sys.path.insert(0, EXEC)

import mlops
from mlops import (
    Principal, Role, ChainedAuditStore, DataContractGate,
    FeatureRegistry, FeatureStore, evaluate_feature, ParityReport, check_parity,
    ParityGate, FeatureError, MutableReferenceError, ImmutableVersionError,
    FeaturePermissionError, ContractViolation,
)

SRC = "dataset:raw@sha256:" + "a" * 64


def test_version_string():
    assert mlops.__version__.startswith("1.3.")
    assert re.match(r"^1\.3\.0-x[1-9]\d*$", mlops.__version__), mlops.__version__


def test_new_public_names_importable_from_top_level():
    for name in ["FeatureRegistry", "FeatureStore", "evaluate_feature", "ParityReport",
                 "check_parity", "ParityGate", "FeatureError", "MutableReferenceError",
                 "ImmutableVersionError", "FeaturePermissionError", "ContractViolation"]:
        assert hasattr(mlops, name), name
    for sub in (MutableReferenceError, ImmutableVersionError, FeaturePermissionError, ContractViolation):
        assert issubclass(sub, FeatureError)


def _build():
    audit = ChainedAuditStore(b"secret")
    reg = FeatureRegistry(audit)
    p = Principal("alice", Role.DEPLOYER, "ns")
    vid = reg.publish(p, "ns", {"name": "spend", "entity_key": "user", "source": SRC,
                                "column": "amt", "agg": "last", "window_seconds": None,
                                "dtype": "float"})
    store = FeatureStore(reg, DataContractGate(["entity", "event_ts", "value"]))
    store.ingest("ns", "spend", [
        {"entity": "u1", "event_ts": 10, "value": 1.5},
        {"entity": "u1", "event_ts": 20, "value": 2.5},
        {"entity": "u2", "event_ts": 15, "value": 7.0},
        {"entity": "u3", "event_ts": 5, "value": 4.0},
    ])
    return store, vid


def _pairs(store, corrupt=None):
    out = []
    for e in ["u1", "u2", "u3"]:
        on = store.get_online("ns", "spend", e, 100)
        off = store.get_offline("ns", "spend", e, 100)
        if e == corrupt:
            on = on + 1.0
        out.append((e, on, off))
    return out


def test_journey_parity_pass_then_gate_allows():
    store, vid = _build()
    rep = check_parity(_pairs(store), tolerance=1e-9, min_pairs=3)
    assert rep.status == "pass" and rep.pairs == 3
    assert ParityGate({vid: rep}).allow_promotion([vid]).passed is True


def test_journey_corrupt_online_value_fails_and_blocks():
    store, vid = _build()
    rep = check_parity(_pairs(store, corrupt="u2"), tolerance=1e-9, min_pairs=3)
    assert rep.status == "fail"
    assert rep.mismatches == ["u2"]
    d = ParityGate({vid: rep}).allow_promotion([vid])
    assert d.passed is False
    assert vid in d.reason and "fail" in d.reason


def test_journey_too_few_pairs_is_blocked():
    store, vid = _build()
    rep = check_parity(_pairs(store)[:2], tolerance=1e-9, min_pairs=3)
    assert rep.status == "insufficient_data"
    assert ParityGate({vid: rep}).allow_promotion([vid]).passed is False


def test_smoke_test_has_c37_to_c48_functions():
    text = open(os.path.join(EXEC, "smoke_test.py"), encoding="utf-8").read()
    missing = [n for n in range(37, 49) if not re.search(rf"^\s*def c{n}_", text, re.M)]
    assert missing == [], f"missing smoke functions for claims {missing}"
