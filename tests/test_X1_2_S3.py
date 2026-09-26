"""X1.2-S3 (C43/C44): out-of-order arrival and ingestion-sequence ties."""
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



def test_late_row_changes_only_asof_at_or_after_its_ts():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, 1.0), ("u", 30, 3.0)))
    probes = [5, 10, 14, 15, 20, 29, 30, 50]
    before = {t: s.get_offline(NS, "f", "u", t) for t in probes}
    s.ingest(NS, "f", rows_of(("u", 15, 1.5)))  # late arrival, old event_ts
    after = {t: s.get_offline(NS, "f", "u", t) for t in probes}
    for t in probes:
        if t < 15:
            assert after[t] == before[t], t
    assert after[15] == 1.5 and after[20] == 1.5 and after[29] == 1.5
    assert after[30] == 3.0 and after[50] == 3.0  # newer event_ts still wins


def test_late_row_changes_windowed_agg_only_from_its_ts():
    s = make_store([("s", "sum", None)])
    s.ingest(NS, "s", rows_of(("u", 10, 1.0), ("u", 30, 4.0)))
    s.ingest(NS, "s", rows_of(("u", 20, 2.0)))
    assert s.get_offline(NS, "s", "u", 19) == 1.0
    assert s.get_offline(NS, "s", "u", 20) == 3.0
    assert s.get_offline(NS, "s", "u", 30) == 7.0


def test_tie_resolves_to_highest_ingestion_sequence():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, "first")))
    s.ingest(NS, "f", rows_of(("u", 10, "second")))
    s.ingest(NS, "f", rows_of(("u", 10, "third")))
    assert s.get_offline(NS, "f", "u", 10) == "third"
    assert s.get_online(NS, "f", "u", 10) == "third"


@pytest.mark.parametrize("splits", [
    [[0, 1, 2, 3]], [[0], [1], [2], [3]], [[0, 1], [2, 3]], [[0], [1, 2, 3]],
])
def test_tie_independent_of_batch_splitting(splits):
    seq = rows_of(("u", 10, 1.0), ("u", 10, 2.0), ("u", 5, 9.0), ("u", 10, 3.0))
    s = make_store([("f", "last", None)])
    for grp in splits:
        s.ingest(NS, "f", [seq[i] for i in grp])
    assert s.get_offline(NS, "f", "u", 10) == 3.0  # last in ingest order
    assert s.get_offline(NS, "f", "u", 9) == 9.0


def test_tie_is_per_entity_and_ignores_value_magnitude():
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("a", 10, 100.0), ("b", 10, 7.0), ("a", 10, 1.0)))
    assert s.get_offline(NS, "f", "a", 10) == 1.0  # later seq, smaller value
    assert s.get_offline(NS, "f", "b", 10) == 7.0


def test_reingest_identical_rows_is_deterministic():
    s = make_store([("f", "last", None), ("c", "count", None)])
    batch = rows_of(("u", 10, 1.0), ("u", 10, 2.0), ("u", 20, 5.0))
    s.ingest(NS, "f", batch)
    first = [s.get_offline(NS, "f", "u", t) for t in (9, 10, 20)]
    s.ingest(NS, "f", batch)  # same rows again, new call
    second = [s.get_offline(NS, "f", "u", t) for t in (9, 10, 20)]
    assert first == second == [None, 2.0, 5.0]
    # two stores fed identically agree exactly
    s2 = make_store([("f", "last", None)])
    s2.ingest(NS, "f", batch)
    s2.ingest(NS, "f", batch)
    assert [s2.get_offline(NS, "f", "u", t) for t in (9, 10, 20)] == second
    assert [s.get_online(NS, "f", "u", t) for t in (9, 10, 20)] == second
