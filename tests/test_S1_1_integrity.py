"""S1.1 integrity: kernel primitives, audit chain, lineage, RBAC audit, parity fail-closed."""
import copy
import os
import sys
import threading
from datetime import datetime, timezone

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.feature_parity import check_parity
from mlops.kernel import ManualClock, canonical_json, content_id, iso
from mlops.lineage import LineageGraph
from mlops.rbac import Principal, RBACEngine, Role

SECRET = b"s1-secret"


def build_store(n=5, secret=SECRET):
    store = ChainedAuditStore(secret)
    for i in range(n):
        store.append(f"actor{i}", "act", f"res{i}", "allow", ts=1000.0 + i, meta={"i": i})
    return store


# (1) canonical_json
def test_canonical_json_sorted_compact_and_order_stable():
    a = {"b": 1, "a": {"z": [1, 2], "y": None}}
    b = {"a": {"y": None, "z": [1, 2]}, "b": 1}
    assert canonical_json(a) == '{"a":{"y":null,"z":[1,2]},"b":1}'
    assert canonical_json(a) == canonical_json(b)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {"k": [1, float("nan")]}])
def test_canonical_json_rejects_nan_and_inf(bad):
    with pytest.raises(ValueError):
        canonical_json(bad)


# (2) content_id
def test_content_id_stable_and_content_sensitive():
    x = content_id("art", {"a": 1, "b": 2})
    assert x == content_id("art", {"b": 2, "a": 1})
    assert x.startswith("art_") and len(x) == len("art_") + 16
    assert x != content_id("art", {"a": 1, "b": 3})
    assert x != content_id("other", {"a": 1, "b": 2})


# (3) clock
def test_manual_clock_advance_and_iso_is_utc_aware():
    c = ManualClock(100.0)
    assert c.now() == 100.0
    c.advance(0.5)
    c.advance(1.5)
    assert c.now() == 102.0
    assert iso(0) == "1970-01-01T00:00:00+00:00"
    assert iso(86400.0 + 3661) == "1970-01-02T01:01:01+00:00"
    assert datetime.fromisoformat(iso(c.now())).utcoffset().total_seconds() == 0


# (4) truncation needs the anchor
def test_truncation_only_detected_with_head_anchor():
    store = build_store(5)
    records = store.export()
    v = ChainVerifier(SECRET)
    assert v.verify_export(records) is True
    assert v.verify_export(records, head=store.head()) is True
    # A truncated prefix is itself a valid chain: only the anchor exposes it.
    assert v.verify_export(records[:3]) is True
    assert v.verify_export(records[:3], head=store.head()) is False
    assert store.head()["length"] == 5


# (5) tamper detection
@pytest.mark.parametrize("field,value", [("ts", 1.0), ("meta", {"i": 999}), ("decision", "deny")])
def test_tampering_with_signed_field_is_detected(field, value):
    store = build_store(5)
    records = copy.deepcopy(store.export())
    records[1][field] = value
    v = ChainVerifier(SECRET)
    # Entry 1 is corrupted; recomputed prev_sig cascades to every later entry.
    assert v.invalid_indices(records) == [1, 2, 3, 4]
    assert v.verify_export(records) is False


# (6) concurrency
def test_concurrent_appends_keep_length_and_valid_chain():
    store = ChainedAuditStore(SECRET)
    barrier = threading.Barrier(8)

    def work(t):
        barrier.wait()
        for i in range(25):
            store.append(f"t{t}", "act", f"r{i}", "allow", ts=float(t * 100 + i))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    records = store.export()
    assert len(records) == 200
    assert store.head()["length"] == 200
    assert ChainVerifier(SECRET).verify_export(records, head=store.head()) is True
    assert len({r["signature"] for r in records}) == 200


# (7) lineage
def test_lineage_node_immutability_detects_tamper():
    g = LineageGraph()
    nid = g.add_node("data", {"version": "v1"}, "2024-01-01T00:00:00Z")
    node = g.nodes[nid]
    assert node.is_immutable() is True
    assert g.verify_node_immutability(nid) is True
    node.properties["version"] = "v2"
    assert node.is_immutable() is False
    with pytest.raises(ValueError):
        g.verify_node_immutability(nid)
    # Timestamp tamper on a fresh node is also detected.
    nid2 = g.add_node("model", {"h": "abc"}, "2024-01-01T00:00:00Z")
    g.nodes[nid2].timestamp = "2030-01-01T00:00:00Z"
    assert g.nodes[nid2].is_immutable() is False


# (8) RBAC: exactly one audit entry per call
@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("action", ["submit_pipeline", "approve", "read"])
def test_rbac_every_call_appends_exactly_one_audit_entry(role, action):
    eng = RBACEngine()
    p = Principal("alice", role, "ns")
    eng.check_permission(p, action, "pipe/1")
    assert len(eng.audit_log) == 1
    eng.check_permission(p, action, "pipe/1")
    assert len(eng.audit_log) == 2
    last = eng.get_audit_log()[-1]
    assert (last["actor"], last["action"], last["resource"]) == ("alice", action, "pipe/1")


# (9) security regression guard
@pytest.mark.parametrize("action", ["submit_pipeline", "approve", "read", "deploy", "delete"])
def test_viewer_denied_everything_while_approver_deployer_allowed(action):
    eng = RBACEngine()
    assert eng.check_permission(Principal("v", Role.VIEWER, "ns"), action, "r") is False
    assert eng.audit_log[-1].decision == "deny"
    for role in (Role.APPROVER, Role.DEPLOYER):
        assert eng.check_permission(Principal("u", role, "ns"), action, "r") is True
        assert eng.audit_log[-1].decision == "allow"


# (10) parity fails closed
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "abc"])
@pytest.mark.parametrize("side", ["online", "offline"])
def test_parity_non_finite_or_non_numeric_fails_closed(bad, side):
    good = ("ok", 1.0, 1.0)
    row = ("bad", bad, 1.0) if side == "online" else ("bad", 1.0, bad)
    rep = check_parity([good, row], tolerance=1e9, min_pairs=1)
    assert rep.status == "fail"
    assert rep.mismatches == ["bad"]
    # Control: identical finite values pass.
    assert check_parity([good], tolerance=0.0, min_pairs=1).status == "pass"


# (11) teeth: the checks can actually fail
def test_teeth_wrong_secret_and_dropped_middle_entry_rejected():
    store = build_store(5)
    records = store.export()
    assert ChainVerifier(SECRET).verify_export(records) is True
    # Wrong secret rejects every entry.
    assert ChainVerifier(b"wrong").invalid_indices(records) == [0, 1, 2, 3, 4]
    assert ChainVerifier(b"wrong").verify_export(records) is False
    # Dropping entry 2: entry 3 now chains against the wrong prev_sig.
    dropped = records[:2] + records[3:]
    v = ChainVerifier(SECRET)
    assert v.invalid_indices(dropped) == [2, 3]
    assert v.verify_export(dropped) is False
    assert v.verify_export(dropped, head=store.head()) is False
    # Swapping two entries is also rejected.
    swapped = [records[1], records[0]] + records[2:]
    assert v.verify_export(swapped) is False
    # Forged head signature is rejected even on an intact chain.
    forged = {"length": 5, "signature": "f" * 64}
    assert v.verify_export(records, head=forged) is False


# (12) legacy 4-arg append
def test_legacy_four_argument_append_still_works_and_verifies():
    store = ChainedAuditStore(SECRET)
    e1 = store.append("a", "x", "r1", "allow")
    e2 = store.append("b", "y", "r2", "deny")
    assert e1["meta"] == {} and isinstance(e1["ts"], float)
    assert e1["prev_sig"] == ChainedAuditStore.GENESIS
    assert e2["prev_sig"] == e1["signature"]
    records = store.export()
    assert ChainVerifier(SECRET).verify_export(records, head=store.head()) is True
    assert abs(records[0]["ts"] - datetime.now(timezone.utc).timestamp()) < 60


# ---- pins for Coder A's fixes ----
@pytest.mark.parametrize("bad", ["1.5", True, False, None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["online", "offline"])
def test_parity_strict_types_fail_closed(bad, side):
    row = ("bad", bad, 1.0) if side == "online" else ("bad", 1.0, bad)
    rep = check_parity([("ok", 2.0, 2.0), row], tolerance=1e9, min_pairs=1)
    assert rep.status == "fail"
    assert "bad" in rep.mismatches and "ok" not in rep.mismatches


def test_parity_ordinary_ints_floats_and_mix_still_pass():
    pairs = [("a", 1, 1), ("b", 1.5, 1.5), ("c", 2, 2.05), ("d", 3.04, 3)]
    rep = check_parity(pairs, tolerance=0.1, min_pairs=4)
    assert rep.status == "pass" and rep.mismatches == []
    assert rep.max_abs_diff == pytest.approx(0.05)
    assert check_parity([("c", 2, 2.5)], tolerance=0.1, min_pairs=1).status == "fail"


def test_audit_export_and_meta_do_not_alias_store():
    store = ChainedAuditStore(SECRET)
    meta = {"k": {"n": 1}}
    store.append("a", "x", "r", "allow", ts=1.0, meta=meta)
    store.append("b", "y", "r", "allow", ts=2.0)
    # Mutating the caller's meta after append must not reach the store.
    meta["k"]["n"] = 999
    meta["new"] = 1
    assert store.export()[0]["meta"] == {"k": {"n": 1}}
    # Mutating exported records (including nested meta) must not reach the store.
    exported = store.export()
    exported[0]["meta"]["k"]["n"] = -1
    exported[0]["actor"] = "evil"
    exported[1]["signature"] = "0" * 64
    assert store.export()[0]["meta"] == {"k": {"n": 1}}
    assert store.export()[0]["actor"] == "a"
    assert ChainVerifier(SECRET).verify_export(store.export(), head=store.head()) is True
    # Teeth: the mutated copy itself is rejected.
    assert ChainVerifier(SECRET).verify_export(exported) is False


def test_rbac_check_permission_emits_no_utcnow_deprecation():
    import warnings

    eng = RBACEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.check_permission(Principal("a", Role.APPROVER, "ns"), "approve", "r")
        eng.check_quota(Principal("a", Role.APPROVER, "ns"), "default", 1)
        eng.get_audit_log()
    assert not [w for w in caught if "utcnow" in str(w.message)]


# ---- pins for the hardening pass ----
_P = [("a", 1.0, 1.0)]


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), -1.0, True, False, "0.1", None])
def test_parity_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        check_parity(_P, tolerance=tol, min_pairs=1)


@pytest.mark.parametrize("pairs", [_P, []])
@pytest.mark.parametrize("mp", [0, -1, True])
def test_parity_rejects_min_pairs_below_one(pairs, mp):
    with pytest.raises(ValueError):
        check_parity(pairs, tolerance=0.1, min_pairs=mp)


def test_parity_too_few_pairs_is_insufficient_data_and_valid_input_unchanged():
    assert check_parity([], 0.1, 1).status == "insufficient_data"
    rep = check_parity(_P, 0.1, 2)
    assert rep.status == "insufficient_data" and rep.pairs == 1
    assert check_parity(_P, 0.0, 1).status == "pass"
    assert check_parity([("a", 1.0, 1.5)], 0.1, 1).status == "fail"
    assert check_parity([("a", 1.0, 1.1)], 0.1 + 1e-9, 1).status == "pass"


@pytest.mark.parametrize("bad", [{1, 2}, object(), b"raw", {"k": [b"x"]}, {"k": {1}}])
def test_canonical_json_rejects_unsupported_types(bad):
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_canonical_json_tuple_equals_list_and_content_id_no_set_collision():
    assert canonical_json((1, (2, 3))) == canonical_json([1, [2, 3]]) == "[1,[2,3]]"
    assert content_id("x", {"a": (1, 2)}) == content_id("x", {"a": [1, 2]})
    with pytest.raises(TypeError):
        content_id("x", {"a": {1, 2}})
    assert content_id("x", {"a": None}) == content_id("x", {"a": None})


def test_lineage_double_rewrite_of_properties_and_hash_detected():
    import hashlib
    import json

    g = LineageGraph()
    nid = g.add_node("data", {"version": "v1"}, "2024-01-01T00:00:00Z")
    other = g.add_node("data", {"version": "v9"}, "2024-01-01T00:00:00Z")
    assert g.nodes[other].is_immutable() and g.verify_node_immutability(other) is True
    node = g.nodes[nid]
    node.properties = {"version": "EVIL"}
    node.content_hash = hashlib.sha256(json.dumps(
        {"type": node.node_type, "properties": node.properties, "timestamp": node.timestamp},
        sort_keys=True).encode()).hexdigest()
    assert node.is_immutable() is False
    with pytest.raises(ValueError):
        g.verify_node_immutability(nid)
    assert g.verify_node_immutability(other) is True


def test_audit_head_anchor_shape_and_genuine_head_verifies():
    store = build_store(5)
    head = store.head()
    assert {"length", "signature", "anchor"} <= set(head)
    v = ChainVerifier(SECRET)
    assert v.verify_export(store.export(), head=head) is True
    assert v.verify_export(store.export(), head=None) is True
    assert v.verify_export(store.export()[:3], head=head) is False


def test_audit_forged_heads_rejected():
    store = build_store(5)
    records = store.export()
    head = store.head()
    v = ChainVerifier(SECRET)
    prefix = records[:3]
    short = build_store(5)  # same content: head of a 3-entry prefix, original anchor
    forged = {"length": 3, "signature": prefix[-1]["signature"], "anchor": head["anchor"]}
    assert v.verify_export(prefix, head=forged) is False
    zero = {"length": 3, "signature": prefix[-1]["signature"], "anchor": "0" * len(head["anchor"])}
    assert v.verify_export(prefix, head=zero) is False
    zero_full = dict(head, anchor="0" * len(head["anchor"]))
    assert v.verify_export(records, head=zero_full) is False
    assert short.head() == head  # deterministic control
