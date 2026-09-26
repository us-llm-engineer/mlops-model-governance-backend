"""X1.2-S2 (C42): online == offline via one shared evaluator."""
import copy
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


AGGS = [("l", "last", None), ("s", "sum", None), ("c", "count", None),
        ("sw", "sum", 7), ("cw", "count", 7), ("lw", "last", 7)]


def seeded_store():
    rng = random.Random(42)
    s = make_store(AGGS)
    ents = ["e0", "e1", "e2", "e3"]
    used = set()
    data = []
    while len(data) < 60:
        e, t = rng.choice(ents), rng.randint(0, 40)
        if (e, t) in used:
            continue
        used.add((e, t))
        data.append((e, t, float(rng.randint(-5, 50))))
    rng.shuffle(data)
    for name, _, _ in AGGS:
        s.ingest(NS, name, rows_of(*data))
    return s, ents, data


def test_online_equals_offline_seeded_random():
    s, ents, _ = seeded_store()
    checked = 0
    for name, _, _ in AGGS:
        for e in ents:
            for now in range(-2, 50, 3):
                on = s.get_online(NS, name, e, now)
                off = s.get_offline(NS, name, e, now)
                assert on == off and type(on) is type(off), (name, e, now)
                checked += 1
    assert checked > 200


def test_online_none_matches_offline_none():
    s, _, _ = seeded_store()
    assert s.get_online(NS, "l", "e0", -100) is None
    assert s.get_offline(NS, "l", "e0", -100) is None
    assert s.get_online(NS, "l", "ghost", 100) is None


def test_evaluate_feature_pure_and_repeatable():
    d = {"name": "x", "entity_key": "user", "source": SRC, "column": "v",
         "agg": "sum", "window_seconds": 10, "dtype": "float"}
    rows = rows_of(("u", 5, 1.0), ("u", 12, 2.0), ("u", 20, 3.0), ("u", 40, 9.0))
    rows_before, d_before = copy.deepcopy(rows), copy.deepcopy(d)
    a = evaluate_feature(d, rows, 20)
    b = evaluate_feature(d, rows, 20)
    assert a == b == 5.0  # window (10, 20]: 12 and 20
    assert rows == rows_before and d == d_before
    assert evaluate_feature(d, [], 20) is None or evaluate_feature(d, [], 20) == 0


def test_evaluate_feature_is_module_level_callable():
    assert callable(fs_mod.evaluate_feature)
    assert fs_mod.evaluate_feature.__module__ == "mlops.feature_store"
    assert "evaluate_feature" not in dir(FeatureStore)


@pytest.mark.parametrize("path", ["online", "offline"])
def test_both_paths_delegate_to_evaluate_feature(monkeypatch, path):
    s = make_store([("f", "last", None)])
    s.ingest(NS, "f", rows_of(("u", 10, 1.0)))
    calls = []

    def spy(definition, rows, as_of_ts):
        calls.append((definition["agg"], as_of_ts))
        return "SENTINEL"

    monkeypatch.setattr(fs_mod, "evaluate_feature", spy)
    got = (s.get_online(NS, "f", "u", 55) if path == "online"
           else s.get_offline(NS, "f", "u", 55))
    assert got == "SENTINEL"
    assert calls == [("last", 55)]
