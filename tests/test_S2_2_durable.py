"""S2.2 durable storage: SQLite WAL audit chain, docs store, lineage persistence.

Table/column names are discovered from sqlite_master at run time except the
contract-named `schema_version`, `audit_head` and its `anchor` column.
"""
import os
import signal
import sqlite3
import subprocess
import sys
import threading

import pytest

EXEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
sys.path.append(EXEC)

from mlops.audit import ChainVerifier
from mlops.kernel import IntegrityError
from mlops.lineage import LineageGraph
from mlops.storage_sqlite import (
    SCHEMA_VERSION,
    DurableAuditStore,
    DurableDocs,
    load_lineage,
    open_db,
    save_lineage,
)

SECRET = b"s2-2-secret"


def _tables(path):
    c = sqlite3.connect(str(path))
    try:
        return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                if not r[0].startswith("sqlite_")]
    finally:
        c.close()


def _audit_table(path):
    cands = [t for t in _tables(path) if t not in ("schema_version", "audit_head")]
    assert len(cands) == 1, cands
    return cands[0]


def _replace_everywhere(path, needle, repl, tables):
    """Rewrite `needle` inside every TEXT cell of the given tables (raw sqlite3)."""
    c = sqlite3.connect(str(path))
    try:
        for t in tables:
            for _, col, typ, *_ in c.execute(f'PRAGMA table_info("{t}")').fetchall():
                if "CHAR" in typ.upper() or "TEXT" in typ.upper() or typ == "":
                    c.execute(f'UPDATE "{t}" SET "{col}" = replace("{col}", ?, ?)', (needle, repl))
        c.commit()
    finally:
        c.close()


def _fill(store, n=5, prefix="actor"):
    for i in range(n):
        store.append(f"{prefix}{i}", "act", f"res{i}", "allow", ts=1000.0 + i, meta={"i": i})


# (1)
def test_open_db_wal_schema_version_and_idempotent_reopen(tmp_path):
    p = tmp_path / "a.db"
    conn = open_db(str(p))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    rows = conn.execute("SELECT * FROM schema_version").fetchall()
    assert any(SCHEMA_VERSION in r for r in rows)
    n = len(rows)
    conn.close()
    conn = open_db(str(p))
    assert len(conn.execute("SELECT * FROM schema_version").fetchall()) == n
    conn.close()


# (2)
def test_audit_reopen_roundtrip_verifies(tmp_path):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    _fill(s)
    exp, head = s.export(), s.head()
    s.close()
    s2 = DurableAuditStore(str(p), SECRET)
    assert s2.export() == exp
    assert s2.head() == head
    assert head["length"] == 5
    assert ChainVerifier(SECRET).verify_export(s2.export(), head=s2.head()) is True
    s2.append("late", "act", "r", "allow", ts=2000.0)  # chain continues after reload
    assert ChainVerifier(SECRET).verify_export(s2.export(), head=s2.head()) is True
    s2.close()


# (3)
def test_reopen_with_wrong_secret_raises(tmp_path):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    _fill(s)
    s.close()
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), b"wrong-secret")


# (4)
def test_on_disk_row_edit_tail_truncation_and_anchor_edit_detected(tmp_path):
    def make(name):
        p = tmp_path / name
        s = DurableAuditStore(str(p), SECRET)
        _fill(s)
        s.close()
        return p

    p = make("edit.db")
    _replace_everywhere(p, "actor2", "mallory", [_audit_table(p)])
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)

    p = make("trunc.db")
    t = _audit_table(p)
    c = sqlite3.connect(str(p))
    c.execute(f'DELETE FROM "{t}" WHERE rowid = (SELECT max(rowid) FROM "{t}")')
    c.commit()
    assert c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] == 4
    c.close()
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)

    p = make("anchor.db")
    c = sqlite3.connect(str(p))
    c.execute("UPDATE audit_head SET anchor = ?", ("0" * 64,))
    c.commit()
    c.close()
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)


# (5)
def test_row_and_head_are_committed_together(tmp_path):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    t = _audit_table(p)
    raw = sqlite3.connect(str(p))
    try:
        for i in range(1, 8):
            s.append(f"a{i}", "act", "r", "allow", ts=float(i))
            rows = raw.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            head_len = raw.execute("SELECT length FROM audit_head").fetchone()[0]
            assert rows == head_len == i == s.head()["length"]
    finally:
        raw.close()
        s.close()


# (6)
def test_sigkill_mid_append_leaves_consistent_db(tmp_path):
    p = tmp_path / "k.db"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {EXEC!r})\n"
        "from mlops.storage_sqlite import DurableAuditStore\n"
        f"s = DurableAuditStore({str(p)!r}, {SECRET!r})\n"
        "i = 0\n"
        "while True:\n"
        "    s.append('w', 'act', 'r%d' % i, 'allow', ts=float(i))\n"
        "    i += 1\n"
        "    print(i, flush=True)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    last = 0
    try:
        for line in proc.stdout:
            last = int(line)
            if last >= 20:
                break
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    assert last >= 20
    s = DurableAuditStore(str(p), SECRET)
    n = len(s.export())
    assert last <= n <= last + 1
    assert ChainVerifier(SECRET).verify_export(s.export(), head=s.head()) is True
    s.close()


# (7)
def test_concurrent_appends_contiguous_and_verifiable(tmp_path):
    p = tmp_path / "c.db"
    s = DurableAuditStore(str(p), SECRET)
    barrier = threading.Barrier(8)
    errors = []

    def work(k):
        try:
            barrier.wait()
            for j in range(25):
                s.append(f"t{k}", "act", f"r{j}", "allow", ts=float(k * 100 + j))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ths = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not errors
    exp, head = s.export(), s.head()
    assert len(exp) == head["length"] == 200
    assert ChainVerifier(SECRET).verify_export(exp, head=head)  # prev_sig linkage => contiguous
    assert len({e["signature"] for e in exp}) == 200
    for k in range(8):  # per-thread order preserved
        ts = [e["ts"] for e in exp if e["actor"] == f"t{k}"]
        assert ts == sorted(ts) and len(ts) == 25
    s.close()
    s2 = DurableAuditStore(str(p), SECRET)
    assert s2.export() == exp
    s2.close()


# (8)
def test_docs_round_trip_missing_key_tamper_and_table_names(tmp_path):
    p = tmp_path / "d.db"
    d = DurableDocs(str(p), "docs")
    d.put("a/1", {"x": 1, "v": "needle-val"})
    d.put("a/2", {"x": 2})
    d.put("b/1", [1, 2, 3])
    assert d.get("a/1") == {"x": 1, "v": "needle-val"}
    assert d.get("b/1") == [1, 2, 3]
    assert sorted(d.keys("a/")) == ["a/1", "a/2"]
    assert sorted(d.keys()) == ["a/1", "a/2", "b/1"]
    d.put("a/2", {"x": 22})  # overwrite
    assert d.get("a/2") == {"x": 22}
    d.delete("a/2")
    assert sorted(d.keys("a/")) == ["a/1"]
    try:  # contract leaves missing-key behaviour open: None or NotFound
        assert d.get("nope") is None
    except Exception as e:
        assert type(e).__name__ == "NotFound"

    _replace_everywhere(p, "needle-val", "forged-val", _tables(p))
    with pytest.raises(IntegrityError):
        d.get("a/1")

    for bad in ["x; DROP TABLE y", "", "a-b"]:
        with pytest.raises(ValueError):
            DurableDocs(str(tmp_path / "e.db"), bad)


# (9)
def _graph():
    g = LineageGraph()
    a = g.add_node("data", {"k": "lin-a"}, "2026-01-01T00:00:00Z")
    b = g.add_node("training", {"k": "lin-b"}, "2026-01-02T00:00:00Z")
    c = g.add_node("model", {"k": "lin-c"}, "2026-01-03T00:00:00Z")
    g.add_edge(a, b, "feeds")
    g.add_edge(b, c)
    return g


def test_lineage_round_trip_and_tamper(tmp_path):
    p = tmp_path / "l.db"
    docs = DurableDocs(str(p), "lineage")
    g = _graph()
    save_lineage(docs, g)
    g2 = load_lineage(docs)
    assert g2.nodes == g.nodes
    assert [tuple(e) for e in g2.edges] == [tuple(e) for e in g.edges]
    assert g2.immutable_content == g.immutable_content
    assert all(n.is_immutable() for n in g2.nodes.values())

    _replace_everywhere(p, "lin-b", "lin-X", _tables(p))
    with pytest.raises(IntegrityError):
        load_lineage(docs)


# (10)
def test_non_canonical_object_rejected_and_nothing_stored(tmp_path):
    p = tmp_path / "d.db"
    d = DurableDocs(str(p), "docs")
    with pytest.raises(TypeError):
        d.put("k", {"s": {1, 2, 3}})
    assert list(d.keys()) == []
    try:
        assert d.get("k") is None
    except Exception as e:
        assert type(e).__name__ == "NotFound"


# ---- input-validation constraints (frozen) ----
_BAD_KEYS = [
    ("", ValueError), (None, TypeError), (5, TypeError), (True, TypeError),
    ("a" * 257, ValueError), ("a\x00b", ValueError), ("a\nb", ValueError),
]


@pytest.mark.parametrize("op", ["put", "get", "delete"])
@pytest.mark.parametrize("bad,exc", _BAD_KEYS, ids=[repr(b)[:12] for b, _ in _BAD_KEYS])
def test_docs_bad_keys_rejected_and_nothing_stored(tmp_path, op, bad, exc):
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    with pytest.raises(exc):
        if op == "put":
            d.put(bad, {"x": 1})
        elif op == "get":
            d.get(bad)
        else:
            d.delete(bad)
    assert list(d.keys()) == []


def test_docs_256_char_key_works(tmp_path):
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    k = "a" * 256
    d.put(k, {"x": 1})
    assert d.get(k) == {"x": 1}
    assert d.keys() == [k]


@pytest.mark.parametrize("bad,exc", [
    (None, TypeError), (5, TypeError), (b"a", TypeError),
    ("a" * 257, ValueError), ("a\x00", ValueError), ("a\nb", ValueError),
])
def test_docs_keys_bad_prefix_rejected(tmp_path, bad, exc):
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    with pytest.raises(exc):
        d.keys(bad)


def test_docs_keys_good_prefixes_accepted(tmp_path):
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    d.put("ab/1", 1)
    d.put("cd/1", 2)
    assert sorted(d.keys("")) == ["ab/1", "cd/1"]
    assert d.keys("ab/") == ["ab/1"]
    assert d.keys("a" * 256) == []


@pytest.mark.parametrize("name", ["t", "_t1", "a" * 64, "Abc_9"])
def test_table_name_valid_accepted(tmp_path, name):
    d = DurableDocs(str(tmp_path / "d.db"), name)
    d.put("k", 1)
    assert d.get("k") == 1
    assert name in _tables(tmp_path / "d.db")


@pytest.mark.parametrize("name,exc", [
    ("", ValueError), ("1a", ValueError), ("a-b", ValueError), ("a b", ValueError),
    ("x; DROP TABLE y", ValueError), ("a" * 65, ValueError), ("a\n", ValueError),
    (5, TypeError), (None, TypeError), (b"t", TypeError),
])
def test_table_name_invalid_rejected_nothing_created(tmp_path, name, exc):
    p = tmp_path / "d.db"
    with pytest.raises((ValueError, TypeError)) as ei:
        DurableDocs(str(p), name)
    assert isinstance(ei.value, exc)
    if p.exists():
        assert _tables(p) == []


@pytest.mark.parametrize("val", [{1, 2}, b"bytes", object(), {"n": {"x": object()}}, [b"x"]])
def test_docs_non_json_value_rejected_stores_nothing(tmp_path, val):
    d = DurableDocs(str(tmp_path / "d.db"), "docs")
    with pytest.raises(TypeError):
        d.put("k", val)
    assert d.keys() == []


# NOTE: no 1 MiB value-size case: current code does not enforce a size limit (reported as gap).


@pytest.mark.parametrize("field", ["actor", "action", "resource", "decision"])
@pytest.mark.parametrize("bad,exc", [("", ValueError), (None, TypeError), (5, TypeError), (b"a", TypeError)])
def test_audit_append_bad_field_rejected_nothing_written(tmp_path, field, bad, exc):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    _fill(s, 2)
    before_head, before_exp = s.head(), s.export()
    args = {"actor": "a", "action": "act", "resource": "r", "decision": "allow"}
    args[field] = bad
    with pytest.raises((ValueError, TypeError)) as ei:
        s.append(**args, ts=1.0)
    assert isinstance(ei.value, exc)
    assert s.head()["length"] == before_head["length"] == 2
    assert s.export() == before_exp
    s.close()
    s2 = DurableAuditStore(str(p), SECRET)
    assert s2.head() == before_head
    s2.close()


@pytest.mark.parametrize("bad", ["str", 5, [1], (1,), True])
def test_audit_append_meta_must_be_dict_or_none(tmp_path, bad):
    s = DurableAuditStore(str(tmp_path / "a.db"), SECRET)
    with pytest.raises((ValueError, TypeError)):
        s.append("a", "act", "r", "allow", ts=1.0, meta=bad)
    assert s.head()["length"] == 0
    s.append("a", "act", "r", "allow", ts=1.0, meta=None)
    s.append("a", "act", "r", "allow", ts=2.0, meta={"k": 1})
    assert s.head()["length"] == 2
    s.close()
