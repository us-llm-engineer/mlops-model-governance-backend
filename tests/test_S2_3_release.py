"""S2.3 slim suite: release control plane (model stages, dataset versions, supply chain).
Oracle-based with hand-computed expectations; frozen BEFORE the modules exist.
Stage of a version is read through ModelRegistry.history(model_id) (records with
.version_id / .stage) and ModelRegistry.current(model_id, Stage) (record or None).
"""
import copy
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.dataset_versions import DatasetRegistry
from mlops.kernel import Conflict, NotFound, PolicyDenied, ValidationFailed
from mlops.lineage import LineageGraph
from mlops.model_stages import ModelRegistry, ModelVersionRecord, Stage
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule
from mlops.supply_chain import (
    RetentionPolicy,
    build_attestation,
    merkle_root,
    sbom_from_requirements,
    verify_attestation,
)

SECRET = b"s2-3-secret"
BAD = (ValidationFailed, ValueError)
A, B, C = ("a1" * 32, "b2" * 32, "c3" * 32)
H1, H2 = "11" * 32, "22" * 32


def sha(s):
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def _audit():
    return ChainedAuditStore(SECRET)


def _pdp(audit=None):
    rule = Rule(id="acc", version=1, kind="min_metric", params={"metric": "acc", "min": 0.9})
    return PolicyDecisionPoint(PolicyBundle(rules=[rule], name="rel", version=1), audit=audit)


def _stage_of(reg, model_id, version_id):
    return [r for r in reg.history(model_id) if r.version_id == version_id][-1].stage


def _to_prod(reg, vid, model_id="m", ctx=None):
    reg.register("alice", model_id, vid, H1, validated=True)
    reg.transition("alice", model_id, vid, Stage.STAGING)
    reg.transition("alice", model_id, vid, Stage.PRODUCTION, context=ctx)


# ---------------------------------------------------------------- supply chain

def _leaf(x):
    return hashlib.sha256(b"\x00" + bytes.fromhex(x)).hexdigest()


def _node(l, r):
    return hashlib.sha256(b"\x01" + bytes.fromhex(l) + bytes.fromhex(r)).hexdigest()


def test_merkle_root_oracle_and_order():
    # domain-separated: leaf = H(0x00||leaf), node = H(0x01||l||r), odd node promoted unchanged
    assert merkle_root([A]) == _leaf(A)
    assert merkle_root([A]) != A
    assert merkle_root([A, B]) == _node(_leaf(A), _leaf(B))
    assert merkle_root([A, B, C]) == _node(_node(_leaf(A), _leaf(B)), _leaf(C))
    assert merkle_root([B, A]) == _node(_leaf(B), _leaf(A))
    assert merkle_root([B, A]) != merkle_root([A, B])


@pytest.mark.parametrize("leaves", [[], [A, "zz" * 32], [A, "ab" * 31], ["not-hex"]])
def test_merkle_root_rejects_bad_input(leaves):
    with pytest.raises(ValueError):
        merkle_root(leaves)


def _att(secret=SECRET, params=None):
    return build_attestation(secret, {"b.txt": H2, "a.txt": H1}, "builder-1", A, params or {"lr": 0.1})


def test_attestation_shape_and_genuine_verifies():
    att = _att()
    for k in ("subject", "materials_root", "builder_id", "parameters", "provenance_digest", "signature", "format"):
        assert k in att
    assert "no PKI" in att["format"]
    assert att["subject"] == A and att["builder_id"] == "builder-1"
    assert att["materials_root"] == merkle_root([H1, H2])  # sorted by input name: a.txt, b.txt
    assert verify_attestation(att, SECRET) is True
    assert verify_attestation(att, SECRET, artifact_digest=A) is True
    assert verify_attestation(att, SECRET, artifact_digest=B) is False
    assert verify_attestation(att, b"other-secret") is False


@pytest.mark.parametrize("field,new", [
    ("subject", B), ("materials_root", C), ("builder_id", "evil"),
    ("parameters", {"lr": 9}), ("signature", "0" * 64),
])
def test_attestation_any_edit_fails(field, new):
    att = _att()
    forged = copy.deepcopy(att)
    forged[field] = new
    assert verify_attestation(forged, SECRET) is False
    assert verify_attestation(att, SECRET) is True  # original untouched


def test_attestation_malformed_never_raises():
    good = _att()
    missing = [{k: v for k, v in good.items() if k != drop} for drop in good]
    bad_types = [dict(good, signature=123), dict(good, parameters="x"), dict(good, subject=None)]
    for junk in [None, {}, [], "att", 5, *missing, *bad_types]:
        assert verify_attestation(junk, SECRET) is False


def test_sbom_deterministic_sorted_purl_and_duplicates():
    one = sbom_from_requirements([("zlib", "1.2"), ("attrs", "23.1"), ("mid", "0.1")])
    two = sbom_from_requirements([{"name": "mid", "version": "0.1"}, ("attrs", "23.1"), ("zlib", "1.2")])
    assert one == two
    comps = one["components"]
    assert [c["name"] for c in comps] == ["attrs", "mid", "zlib"]
    assert comps[0]["purl"] == "pkg:pypi/attrs@23.1"
    assert comps[2]["version"] == "1.2"
    with pytest.raises(ValidationFailed):
        sbom_from_requirements([("a", "1"), ("a", "2")])


def _recs(*pairs):
    return [{"id": i, "created_ts": t} for i, t in pairs]


def test_retention_max_age_boundary_and_no_mutation():
    recs = _recs(("k950", 950), ("k900", 900), ("d899", 899), ("d100", 100))
    before = copy.deepcopy(recs)
    pol = RetentionPolicy([{"kind": "max_age", "max_age_s": 100}])
    assert sorted(pol.apply(recs, now=1000)) == ["d100", "d899"]  # deletable iff now-created > 100
    assert recs == before


def test_retention_keep_last_and_combined_and_audit():
    recs = _recs(("r1", 10), ("r2", 20), ("r3", 30), ("r4", 40))
    assert sorted(RetentionPolicy([{"kind": "keep_last", "n": 2}]).apply(recs, now=1000)) == ["r1", "r2"]
    combo = _recs(("r1", 100), ("r2", 200), ("r3", 300), ("r4", 950))
    both = RetentionPolicy([{"kind": "max_age", "max_age_s": 100}, {"kind": "keep_last", "n": 2}])
    audit = _audit()
    # max_age alone would delete r1,r2,r3; keep_last protects r3,r4 -> only r1,r2
    assert sorted(both.apply(combo, now=1000, audit=audit)) == ["r1", "r2"]
    assert len(audit.export()) == 2
    audit2 = _audit()
    assert both.apply(_recs(("n1", 990), ("n2", 995)), now=1000, audit=audit2) == []
    assert audit2.export() == []


# -------------------------------------------------------------------- datasets

def test_dataset_register_idempotent_and_id_sensitivity():
    audit, g = _audit(), LineageGraph()
    reg = DatasetRegistry(audit, lineage=g)
    v1 = reg.register("alice", "ds", H1, 10, {"x": "int"})
    assert reg.register("alice", "ds", H1, 10, {"x": "int"}) == v1
    others = [reg.register("alice", "ds", H2, 10, {"x": "int"}),
              reg.register("alice", "ds", H1, 11, {"x": "int"}),
              reg.register("alice", "ds", H1, 10, {"x": "float"})]
    assert len({v1, *others}) == 4
    assert any(n.node_type == "dataset_version" for n in g.nodes.values())
    assert len(audit.export()) >= 1
    assert ChainVerifier(SECRET).verify_export(audit.export(), head=audit.head())


def test_dataset_validation_get_verify_latest():
    audit = _audit()
    reg = DatasetRegistry(audit)
    for bad_hash in ["xyz", "ab" * 31, "zz" * 32]:
        with pytest.raises(BAD):
            reg.register("alice", "ds", bad_hash, 1, {})
    for bad_rows in [-1, True, 1.5, "3"]:
        with pytest.raises(BAD):
            reg.register("alice", "ds", H1, bad_rows, {})
    assert audit.export() == []
    with pytest.raises(NotFound):
        reg.get("ds-missing")
    v1 = reg.register("alice", "ds", H1, 5, {})
    v2 = reg.register("alice", "ds", H2, 6, {})
    assert reg.verify(v1, H1) is True and reg.verify(v1, H2) is False
    latest = reg.latest("ds")
    assert latest == v2 or latest == reg.get(v2)


# ---------------------------------------------------------------- model stages

def test_happy_path_and_write_protection():
    reg = ModelRegistry(_audit(), trust_validated=True)
    rec = reg.register("alice", "m", "v1", H1, validated=True)
    assert isinstance(rec, ModelVersionRecord) and rec.stage == Stage.REGISTERED
    reg.transition("alice", "m", "v1", Stage.STAGING)
    assert _stage_of(reg, "m", "v1") == Stage.STAGING
    reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert reg.current("m", Stage.PRODUCTION).version_id == "v1"
    assert not hasattr(ModelRegistry, "set_production")
    with pytest.raises(AttributeError):
        rec.stage = Stage.PRODUCTION  # frozen dataclass


@pytest.mark.parametrize("path,target", [
    ([], Stage.PRODUCTION),                                   # REGISTERED -> PRODUCTION
    ([Stage.STAGING, Stage.ARCHIVED], Stage.STAGING),         # ARCHIVED -> STAGING
    ([Stage.STAGING, Stage.PRODUCTION], Stage.STAGING),       # PRODUCTION -> STAGING
])
def test_illegal_edges_conflict(path, target):
    reg = ModelRegistry(_audit(), trust_validated=True)
    reg.register("alice", "m", "v1", H1, validated=True)
    for st in path:
        reg.transition("alice", "m", "v1", st)
    before = _stage_of(reg, "m", "v1")
    with pytest.raises(Conflict):
        reg.transition("alice", "m", "v1", target)
    assert _stage_of(reg, "m", "v1") == before


def test_production_requires_validated():
    reg = ModelRegistry(_audit())
    reg.register("alice", "m", "v1", H1, validated=False)
    reg.transition("alice", "m", "v1", Stage.STAGING)
    with pytest.raises((ValidationFailed, Conflict)):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert _stage_of(reg, "m", "v1") == Stage.STAGING
    assert reg.current("m", Stage.PRODUCTION) is None


def test_second_production_archives_first_and_rollback():
    reg = ModelRegistry(_audit(), trust_validated=True)
    _to_prod(reg, "v1")
    with pytest.raises(Conflict):
        reg.rollback("alice", "m")  # no previous production yet
    _to_prod(reg, "v2")
    assert _stage_of(reg, "m", "v1") == Stage.ARCHIVED
    cur = reg.current("m", Stage.PRODUCTION)
    assert cur.version_id == "v2" and cur.previous_production == "v1"
    reg.rollback("alice", "m")
    assert reg.current("m", Stage.PRODUCTION).version_id == "v1"
    assert _stage_of(reg, "m", "v1") == Stage.PRODUCTION
    assert _stage_of(reg, "m", "v2") == Stage.ARCHIVED


def test_policy_gate_on_production():
    audit = _audit()
    reg = ModelRegistry(audit, decision_point=_pdp(audit), trust_validated=True)
    reg.register("alice", "m", "v1", H1, validated=True)
    reg.transition("alice", "m", "v1", Stage.STAGING)
    n = len(audit.export())
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION, context={"acc": 0.5})
    assert _stage_of(reg, "m", "v1") == Stage.STAGING
    assert reg.current("m", Stage.PRODUCTION) is None
    assert any(e["decision"] == "deny" for e in audit.export()[n:])
    reg.transition("alice", "m", "v1", Stage.PRODUCTION, context={"acc": 0.95})
    assert _stage_of(reg, "m", "v1") == Stage.PRODUCTION


def test_each_call_appends_exactly_one_audit_entry():
    audit = _audit()
    reg = ModelRegistry(audit, decision_point=_pdp(), trust_validated=True)  # PDP without its own audit
    steps = [
        lambda: reg.register("alice", "m", "v1", H1, validated=True),
        lambda: reg.transition("alice", "m", "v1", Stage.STAGING),
        lambda: reg.transition("alice", "m", "v1", Stage.PRODUCTION, context={"acc": 0.95}),
        lambda: reg.register("alice", "m", "v2", H2, validated=True),
        lambda: reg.transition("alice", "m", "v2", Stage.STAGING),
    ]
    for i, step in enumerate(steps, 1):
        step()
        assert len(audit.export()) == i
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v2", Stage.PRODUCTION, context={"acc": 0.1})
    assert len(audit.export()) == 6  # denied call audited once
    reg.transition("alice", "m", "v2", Stage.PRODUCTION, context={"acc": 0.99})
    reg.rollback("alice", "m", context={"acc": 0.99})
    assert len(audit.export()) == 8
    assert ChainVerifier(SECRET).verify_export(audit.export(), head=audit.head())


def test_dataset_version_must_exist_and_is_recorded():
    dreg = DatasetRegistry(_audit())
    reg = ModelRegistry(_audit(), datasets=dreg)
    with pytest.raises(NotFound):
        reg.register("alice", "m", "v1", H1, dataset_version="ds-missing")
    dv = dreg.register("alice", "ds", H2, 3, {})
    rec = reg.register("alice", "m", "v1", H1, dataset_version=dv)  # v1 was not registered above
    assert rec.dataset_version == dv


def test_attestation_gating():
    reg = ModelRegistry(_audit(), attestation_secret=SECRET, require_attestation=True)
    good = build_attestation(SECRET, {"a": H2}, "builder", H1, {})
    tampered = dict(good, builder_id="evil")
    for att in (None, tampered):
        with pytest.raises(ValidationFailed):
            reg.register("alice", "m", "v1", H1, attestation=att)
    rec = reg.register("alice", "m", "v1", H1, attestation=good)  # nothing was registered before
    assert rec.attestation_ok is True


@pytest.mark.parametrize("model_id,version_id,artifact_hash", [
    ("", "v1", H1), ("m", "v 1", H1), ("m", "v1", "abc"), ("m", "v1", "zz" * 32),
])
def test_invalid_register_inputs_register_nothing(model_id, version_id, artifact_hash):
    reg = ModelRegistry(_audit())
    with pytest.raises(BAD):
        reg.register("alice", model_id, version_id, artifact_hash)
    rec = reg.register("alice", "m", "v1", H1)  # still succeeds: no residue
    assert rec.stage == Stage.REGISTERED


# ------------------------------------------- fail-closed amendments (constraints)

REJ = (ValidationFailed, ValueError, TypeError)
NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("rules", [
    [{"kind": "keep_lst", "n": 2}],
    [{"kind": "max_age"}],
    [{"kind": "max_age", "max_age_s": -5}],
    [{"kind": "max_age", "max_age_s": True}],
    [{"kind": "max_age", "max_age_s": NAN}],
    [{"kind": "max_age", "max_age_s": INF}],
    [{"kind": "max_age", "max_age_s": "10"}],
    [{"kind": "keep_last"}],
    [{"kind": "keep_last", "n": -1}],
    [{"kind": "keep_last", "n": True}],
    [{"kind": "keep_last", "n": 1.5}],
    [{"kind": "keep_last", "n": "2"}],
    "max_age", None, {"kind": "keep_last", "n": 1}, [],
    ["keep_last"], [None], [{"kind": "keep_last", "n": 1}, "x"],
])
def test_retention_rejects_bad_rules_before_any_effect(rules):
    recs = _recs(("a", 1), ("b", 2))
    before = copy.deepcopy(recs)
    audit = _audit()
    with pytest.raises(REJ):
        RetentionPolicy(rules).apply(recs, now=1000, audit=audit)
    assert audit.export() == []
    assert recs == before


@pytest.mark.parametrize("now", [NAN, INF, True, None, "5"])
def test_retention_rejects_bad_now(now):
    recs = _recs(("a", 1))
    audit = _audit()
    with pytest.raises(REJ):
        RetentionPolicy([{"kind": "max_age", "max_age_s": 10}]).apply(recs, now=now, audit=audit)
    assert audit.export() == [] and recs == _recs(("a", 1))


@pytest.mark.parametrize("recs", [
    [{"created_ts": 1}],
    [{"id": "a"}],
    [{"id": 5, "created_ts": 1}],
    [{"id": "a", "created_ts": NAN}],
    [{"id": "a", "created_ts": True}],
    [{"id": "a", "created_ts": None}],
    [{"id": "a", "created_ts": 1}, {"id": "a", "created_ts": 2}],
    ["a"],
])
def test_retention_rejects_bad_records_without_mutation(recs):
    before = copy.deepcopy(recs)
    audit = _audit()
    with pytest.raises(REJ):
        RetentionPolicy([{"kind": "max_age", "max_age_s": 10}]).apply(recs, now=1000, audit=audit)
    assert recs == before and audit.export() == []


def test_retention_zero_max_age_is_valid():
    recs = _recs(("old", 999), ("same", 1000))
    assert RetentionPolicy([{"kind": "max_age", "max_age_s": 0}]).apply(recs, now=1000) == ["old"]


@pytest.mark.parametrize("kwargs", [
    dict(secret=b""), dict(secret="str-secret"), dict(secret=None),
    dict(inputs={}), dict(inputs={"a": "AB" * 32}), dict(inputs={"a": "ab" * 31}),
    dict(inputs={"a": 5}), dict(builder="") , dict(builder=None),
    dict(artifact="xyz"), dict(artifact="AB" * 32), dict(artifact=None),
    dict(params=[]), dict(params="x"), dict(params=None),
])
def test_build_attestation_rejects_bad_args(kwargs):
    a = dict(secret=SECRET, inputs={"a": H1}, builder="b", artifact=A, params={})
    a.update(kwargs)
    with pytest.raises(REJ):
        build_attestation(a["secret"], a["inputs"], a["builder"], a["artifact"], a["params"])


@pytest.mark.parametrize("secret", ["s2-3-secret", None, b"", 5])
def test_verify_attestation_bad_secret_returns_false(secret):
    assert verify_attestation(_att(), secret) is False


@pytest.mark.parametrize("reqs", [
    [],
    [("a",)], [("a", "1", "x")], [()],
    [(1, "1")], [("a", 1)], [("a", None)], [("", "1")], [("a", "")],
    [{"name": "a"}], [{"version": "1"}], [{}],
    [{"name": 1, "version": "1"}], [{"name": "a", "version": 1}],
    ["ab"], [None], "ab", None,
])
def test_sbom_rejects_malformed_with_validation_failed(reqs):
    with pytest.raises(ValidationFailed):
        sbom_from_requirements(reqs)


def _bad_actor_cases():
    return ["", None, 5, b"alice"]


@pytest.mark.parametrize("actor", _bad_actor_cases())
def test_model_registry_rejects_bad_actor_and_writes_nothing(actor):
    audit = _audit()
    reg = ModelRegistry(audit, trust_validated=True)
    reg.register("alice", "m", "v1", H1, validated=True)
    reg.transition("alice", "m", "v1", Stage.STAGING)
    reg.register("alice", "m", "v2", H2, validated=True)
    reg.transition("alice", "m", "v2", Stage.STAGING)
    reg.transition("alice", "m", "v2", Stage.PRODUCTION)
    reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    n = len(audit.export())
    hist = len(reg.history("m"))
    calls = [
        lambda: reg.register(actor, "m", "v3", H1),
        lambda: reg.transition(actor, "m", "v2", Stage.ARCHIVED),
        lambda: reg.rollback(actor, "m"),
    ]
    for c in calls:
        with pytest.raises(REJ):
            c()
    assert len(audit.export()) == n
    assert len(reg.history("m")) == hist
    assert reg.current("m", Stage.PRODUCTION).version_id == "v1"


@pytest.mark.parametrize("actor", _bad_actor_cases())
def test_dataset_registry_rejects_bad_actor_and_writes_nothing(actor):
    audit = _audit()
    dreg = DatasetRegistry(audit)
    v1 = dreg.register("alice", "ds", H2, 3, {})
    n = len(audit.export())
    with pytest.raises(REJ):
        dreg.register(actor, "ds", H1, 4, {})
    assert len(audit.export()) == n
    assert dreg.latest("ds") == v1  # nothing new registered
