"""X1.2-S1 (C41): offline as-of correctness."""
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



def test_never_returns_future_row():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, 1.0), ("u", 20, 2.0), ("u", 30, 3.0)))
    assert s.get_offline(NS, "f", "u", 25) == 2.0
    assert s.get_offline(NS, "f", "u", 19) == 1.0
    assert s.get_offline(NS, "f", "u", 1000) == 3.0


def test_exact_boundary_included():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, 1.0), ("u", 20, 2.0)))
    assert s.get_offline(NS, "f", "u", 20) == 2.0
    assert s.get_offline(NS, "f", "u", 10) == 1.0


def test_none_when_no_row_yet_or_unknown_entity():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, 1.0)))
    assert s.get_offline(NS, "f", "u", 9) is None
    assert s.get_offline(NS, "f", "nobody", 100) is None


def test_entity_isolation():
    s = make_store([("f", "last", None), ("c", "count", None)])
    s.ingest(NS, "f", rows_of(("a", 10, 1.0), ("b", 15, 9.0), ("a", 20, 2.0)))
    s.ingest(NS, "c", rows_of(("a", 10, 1), ("b", 11, 1), ("b", 12, 1)))
    assert s.get_offline(NS, "f", "a", 17) == 1.0
    assert s.get_offline(NS, "f", "b", 17) == 9.0
    assert s.get_offline(NS, "c", "a", 100) == 1
    assert s.get_offline(NS, "c", "b", 100) == 2


def test_sum_count_unwindowed_include_boundary_exclude_future():
    s = make_store([("s", "sum", None), ("c", "count", None)])
    data = rows_of(("u", 10, 1.0), ("u", 20, 2.0), ("u", 30, 4.0))
    s.ingest(NS, "s", data)
    s.ingest(NS, "c", data)
    assert s.get_offline(NS, "s", "u", 20) == 3.0
    assert s.get_offline(NS, "c", "u", 20) == 2
    assert s.get_offline(NS, "s", "u", 29) == 3.0
    assert s.get_offline(NS, "s", "u", 30) == 7.0
    assert s.get_offline(NS, "c", "u", 30) == 3


def test_window_lower_edge_exclusive_upper_inclusive():
    s = make_store([("s", "sum", 10), ("c", "count", 10)])
    data = rows_of(("u", 10, 1.0), ("u", 15, 2.0), ("u", 20, 4.0))
    s.ingest(NS, "s", data)
    s.ingest(NS, "c", data)
    # as_of=20, window 10 -> (10, 20]: ts=10 excluded, 15 and 20 included
    assert s.get_offline(NS, "s", "u", 20) == 6.0
    assert s.get_offline(NS, "c", "u", 20) == 2
    # as_of=19 -> (9, 19]: 10 and 15 included, 20 (future) excluded
    assert s.get_offline(NS, "s", "u", 19) == 3.0
    assert s.get_offline(NS, "c", "u", 19) == 2
    # as_of=25 -> (15, 25]: only 20
    assert s.get_offline(NS, "s", "u", 25) == 4.0
    assert s.get_offline(NS, "c", "u", 25) == 1
