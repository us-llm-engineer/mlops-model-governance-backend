"""S3.0 storage fixes: constraints for review findings H2, H3, H4, M3, M9."""
import os
import sqlite3
import sys
import threading

import pytest

EXEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
sys.path.append(EXEC)

from mlops.audit import ChainVerifier
from mlops.kernel import Conflict, IntegrityError
from mlops.lineage import LineageGraph
from mlops.storage_sqlite import (
    DurableAuditStore,
    DurableDocs,
    load_lineage,
    save_lineage,
)

SECRET = b"s3-0-secret"


def _fill(store, n, prefix="a"):
    for i in range(n):
        store.append(f"{prefix}{i}", "act", f"res{i}", "allow", ts=1000.0 + i, meta={"i": i})


def _raw(path, sql, args=()):
    c = sqlite3.connect(str(path))
    try:
        c.execute(sql, args)
        c.commit()
    finally:
        c.close()


def _make5(path):
    s = DurableAuditStore(str(path), SECRET)
    _fill(s, 5)
    s.close()


def _assert_reopens_valid(path, expect_len=None):
    s = DurableAuditStore(str(path), SECRET)
    try:
        exp, head = s.export(), s.head()
        assert ChainVerifier(SECRET).verify_export(exp, head=head) is True
        if expect_len is not None:
            assert len(exp) == expect_len
    finally:
        s.close()


# ---- (1) H2 fail-open head ----
def test_h2_truncated_tail_and_head_row_deleted(tmp_path):
    p = tmp_path / "a.db"
    _make5(p)
    _raw(p, "DELETE FROM audit_entries WHERE seq >= 4")
    _raw(p, "DELETE FROM audit_head")
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)


def test_h2_head_row_deleted_only(tmp_path):
    p = tmp_path / "a.db"
    _make5(p)
    _raw(p, "DELETE FROM audit_head")
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)


def test_h2_head_length_inflated(tmp_path):
    p = tmp_path / "a.db"
    _make5(p)
    _raw(p, "UPDATE audit_head SET length = 9")
    with pytest.raises(IntegrityError):
        DurableAuditStore(str(p), SECRET)


def test_h2_empty_db_without_head_is_fresh_store(tmp_path):
    p = tmp_path / "a.db"
    DurableAuditStore(str(p), SECRET).close()
    _raw(p, "DELETE FROM audit_head")
    s = DurableAuditStore(str(p), SECRET)
    assert s.export() == []
    s.append("x", "y", "z", "allow", ts=1.0)
    s.close()
    _assert_reopens_valid(p, 1)


# ---- (2) H3 stale second handle ----
def test_h3_stale_handle_does_not_fork(tmp_path):
    p = tmp_path / "a.db"
    A = DurableAuditStore(str(p), SECRET)
    B = DurableAuditStore(str(p), SECRET)
    B.append("b", "act", "r", "allow", ts=1.0)
    outcome = "appended"
    try:
        A.append("a", "act", "r", "allow", ts=2.0)
    except (IntegrityError, Conflict):
        outcome = "raised"
    # in-memory view must match the database exactly
    fresh_c = sqlite3.connect(str(p))
    n_db = fresh_c.execute("SELECT count(*) FROM audit_entries").fetchone()[0]
    fresh_c.close()
    assert len(A.export()) == n_db
    if outcome == "appended":
        assert n_db == 2
    else:
        assert n_db == 1
    A.close()
    B.close()
    _assert_reopens_valid(p, n_db)


def test_h3_two_threads_two_handles(tmp_path):
    p = tmp_path / "a.db"
    A = DurableAuditStore(str(p), SECRET)
    B = DurableAuditStore(str(p), SECRET)
    errs = []

    def work(store, tag):
        for i in range(20):
            try:
                store.append(tag, "act", f"r{i}", "allow", ts=float(i))
            except (IntegrityError, Conflict):
                pass  # clean, allowed error
            except Exception as e:  # noqa: BLE001
                errs.append(e)

    ts = [threading.Thread(target=work, args=(A, "A")), threading.Thread(target=work, args=(B, "B"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    A.close()
    B.close()
    _assert_reopens_valid(p)


def test_h3_busy_timeout(tmp_path):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    assert s._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    s.close()


# ---- (3) H4 lineage tamper ----
def _graph():
    g = LineageGraph()
    a = g.add_node("dataset", {"k": "1"}, "2026-01-01T00:00:00Z")
    b = g.add_node("model", {"k": "2"}, "2026-01-02T00:00:00Z")
    c = g.add_node("deploy", {"k": "3"}, "2026-01-03T00:00:00Z")
    g.add_edge(a, b)
    g.add_edge(b, c)
    return g


def _saved(tmp_path):
    p = tmp_path / "l.db"
    docs = DurableDocs(str(p), "lin")
    g = _graph()
    save_lineage(docs, g)
    return p, docs, g


def _doc_key(docs, needle):
    ks = [k for k in docs.keys() if needle in k]
    assert len(ks) == 1, ks
    return ks[0]


def test_h4_untouched_roundtrip(tmp_path):
    _, docs, g = _saved(tmp_path)
    g2 = load_lineage(docs)
    assert set(g2.nodes) == set(g.nodes)
    assert [tuple(e) for e in g2.edges] == [tuple(e) for e in g.edges]
    assert g2.immutable_content == g.immutable_content


def test_h4_edges_doc_tampered(tmp_path):
    p, docs, _ = _saved(tmp_path)
    k = _doc_key(docs, "edges")
    _raw(p, 'UPDATE "lin" SET content = ? WHERE key = ?', ("[]", k))
    with pytest.raises(IntegrityError):
        load_lineage(docs)


def test_h4_immutable_content_doc_tampered(tmp_path):
    p, docs, _ = _saved(tmp_path)
    k = _doc_key(docs, "immutable")
    _raw(p, 'UPDATE "lin" SET content = ? WHERE key = ?', ("{}", k))
    with pytest.raises(IntegrityError):
        load_lineage(docs)


def test_h4_node_row_deleted(tmp_path):
    p, docs, _ = _saved(tmp_path)
    k = [k for k in docs.keys() if "node" in k][0]
    _raw(p, 'DELETE FROM "lin" WHERE key = ?', (k,))
    with pytest.raises(IntegrityError):
        load_lineage(docs)


# ---- (4) M3 ----
def test_m3_keys_prefix_literal_and_case_sensitive(tmp_path):
    d = DurableDocs(str(tmp_path / "d.db"), "t")
    for k in ("A1", "a1", "a_1", "ab1"):
        d.put(k, {"v": 1})
    assert d.keys("%") == []
    assert d.keys("A") == ["A1"]
    assert d.keys("a_") == ["a_1"]
    assert sorted(d.keys("a")) == ["a1", "a_1", "ab1"]


def test_m3_docs_pragmas(tmp_path):
    d = DurableDocs(str(tmp_path / "d.db"), "t")
    assert d._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert d._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000


def test_m3_save_lineage_atomic(tmp_path, monkeypatch):
    p = tmp_path / "l.db"
    docs = DurableDocs(str(p), "lin")
    good = _graph()
    save_lineage(docs, good)
    before = load_lineage(docs)

    g2 = LineageGraph()
    x = g2.add_node("dataset", {"k": "X"}, "2027-01-01T00:00:00Z")
    y = g2.add_node("model", {"k": "Y"}, "2027-01-02T00:00:00Z")
    g2.add_edge(x, y)

    # Discover the keys the new save writes, using a scratch store.
    scratch = DurableDocs(str(tmp_path / "scratch.db"), "lin")
    save_lineage(scratch, g2)
    node_keys = [k for k in scratch.keys() if k.startswith("lineage:node:")]
    assert len(node_keys) == 2
    victim = node_keys[1]  # a document written mid-transaction

    table = docs._table
    docs._conn.execute(
        f'CREATE TRIGGER inj_ins BEFORE INSERT ON "{table}" '
        f"WHEN NEW.key = '{victim}' BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    docs._conn.execute(
        f'CREATE TRIGGER inj_upd BEFORE UPDATE ON "{table}" '
        f"WHEN NEW.key = '{victim}' BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    docs._conn.commit()
    with pytest.raises(Exception):
        save_lineage(docs, g2)
    docs._conn.execute("DROP TRIGGER inj_ins")
    docs._conn.execute("DROP TRIGGER inj_upd")
    docs._conn.commit()

    after = load_lineage(docs)  # must load cleanly (no mixed state)
    assert set(after.nodes) == set(before.nodes)
    assert [tuple(e) for e in after.edges] == [tuple(e) for e in before.edges]
    assert after.immutable_content == before.immutable_content


# ---- (5) M9 control characters ----
@pytest.mark.parametrize("field", ["actor", "action", "resource", "decision"])
@pytest.mark.parametrize("bad", ["a\nb", "a\x1fb", "a\x7fb", "\x00"])
def test_m9_control_chars_rejected(tmp_path, field, bad):
    p = tmp_path / "a.db"
    s = DurableAuditStore(str(p), SECRET)
    kw = dict(actor="a", action="b", resource="c", decision="allow", ts=1.0)
    kw[field] = bad
    with pytest.raises(ValueError):
        s.append(**kw)
    assert s.export() == []
    s.close()
    c = sqlite3.connect(str(p))
    assert c.execute("SELECT count(*) FROM audit_entries").fetchone()[0] == 0
    c.close()


@pytest.mark.parametrize("meta", [{"x": float("nan")}, {"x": float("inf")}, {"x": {1, 2}}])
def test_m9_bad_meta_rejected(tmp_path, meta):
    s = DurableAuditStore(str(tmp_path / "a.db"), SECRET)
    with pytest.raises((ValueError, TypeError)):
        s.append("a", "b", "c", "allow", ts=1.0, meta=meta)
    assert s.export() == []
    s.close()


def test_m9_docs_key_rejects_del(tmp_path):
    d = DurableDocs(str(tmp_path / "d.db"), "t")
    with pytest.raises(ValueError):
        d.put("a\x7fb", {"v": 1})
    with pytest.raises(ValueError):
        d.get("a\x7fb")
