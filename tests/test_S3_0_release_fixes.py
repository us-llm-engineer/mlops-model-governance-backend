"""S3.0 slim suite: fixes for the review notes (H5-H7, M4-M9) in the release control plane.
Complements (does not edit) tests/test_S2_3_release.py. Table-driven; behavioural oracles only.

New API expectations frozen here:
  ModelRegistry(audit, decision_point, datasets, attestation_secret, require_attestation,
                trust_validated=False, require_approver=None)
  ModelRegistry.validate(actor, model_id, version_id, evidence: dict) -> record (audits "model.validate")
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
from mlops.model_stages import ModelRegistry, Stage
from mlops.storage_sqlite import DurableAuditStore
from mlops.supply_chain import (
    RetentionPolicy,
    build_attestation,
    merkle_root,
    sbom_from_requirements,
    verify_attestation,
)

SECRET = b"s3-0-secret"
A, B, C = ("a1" * 32, "b2" * 32, "c3" * 32)
H1, H2, H3 = "11" * 32, "22" * 32, "33" * 32
EVID = {"metric": "acc", "value": 0.95}
CTRL = ["a\x00b", "a\nb", "a\x07b", "a\x1fb", "a\tb", "a\x7fb"]


def _audit():
    return ChainedAuditStore(SECRET)


def _n(audit):
    return len(audit.export())


class FailingAudit:
    def append(self, *a, **k):
        raise OSError("disk full")

    def export(self):
        return []


class SpyPDP:
    audit = None

    def __init__(self):
        self.calls = 0

    def enforce(self, *a, **k):
        self.calls += 1


def _stage_of(reg, model_id, version_id):
    return [r for r in reg.history(model_id) if r.version_id == version_id][-1].stage


def _staged(reg, vid, model="m", actor="alice", h=H1):
    reg.register(actor, model, vid, h)
    reg.validate(actor, model, vid, dict(EVID))
    reg.transition(actor, model, vid, Stage.STAGING)


def _prod(reg, vid, model="m", actor="alice", h=H1):
    _staged(reg, vid, model, actor, h)
    reg.transition(actor, model, vid, Stage.PRODUCTION)


# ------------------------------------------------------------------------- H5

@pytest.mark.parametrize("bad", ["staging", "production", "STAGING", None, 1, Stage])
def test_h5_transition_rejects_non_stage_before_any_effect(bad):
    audit, pdp = _audit(), SpyPDP()
    reg = ModelRegistry(audit, decision_point=pdp)
    _staged(reg, "v1")
    n, hist = _n(audit), reg.history("m")
    with pytest.raises(ValidationFailed):
        reg.transition("alice", "m", "v1", bad)
    assert pdp.calls == 0 and _n(audit) == n
    assert reg.history("m") == hist and _stage_of(reg, "m", "v1") == Stage.STAGING
    assert reg.transition("alice", "m", "v1", Stage.PRODUCTION).stage == Stage.PRODUCTION


def test_h5_every_failure_path_audits_exactly_one_deny():
    audit = _audit()
    reg = ModelRegistry(audit)
    _prod(reg, "v1")
    paths = [
        (NotFound, lambda: reg.transition("alice", "ghost", "v1", Stage.STAGING)),
        (NotFound, lambda: reg.transition("alice", "m", "vX", Stage.STAGING)),
        (NotFound, lambda: reg.rollback("alice", "ghost")),
        (Conflict, lambda: reg.transition("alice", "m", "v1", Stage.STAGING)),   # illegal edge
        (Conflict, lambda: reg.rollback("alice", "m")),                           # no previous
    ]
    for exc, call in paths:
        n = _n(audit)
        with pytest.raises(exc):
            call()
        new = audit.export()[n:]
        assert len(new) == 1 and new[0]["decision"] == "deny", (exc, new)


def test_h5_rollback_without_any_production_is_one_deny():
    audit = _audit()
    reg = ModelRegistry(audit)
    _staged(reg, "v1")
    n = _n(audit)
    with pytest.raises(Conflict):
        reg.rollback("alice", "m")
    new = audit.export()[n:]
    assert len(new) == 1 and new[0]["decision"] == "deny"


# ------------------------------------------------------------------------- H6

@pytest.mark.parametrize("bad", ["no", "yes", 1, 0, None, [], "True"])
def test_h6_validated_must_be_real_bool(bad):
    audit = _audit()
    reg = ModelRegistry(audit, trust_validated=True)
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", H1, validated=bad)
    assert reg.history("m") == [] and _n(audit) == 0


def test_h6_register_validated_true_rejected_unless_trusted():
    audit = _audit()
    reg = ModelRegistry(audit)
    with pytest.raises((ValidationFailed, Conflict)):
        reg.register("alice", "m", "v1", H1, validated=True)
    assert reg.history("m") == [] and _n(audit) == 0
    trusted = ModelRegistry(_audit(), trust_validated=True)
    assert trusted.register("alice", "m", "v1", H1, validated=True).validated is True
    trusted.transition("alice", "m", "v1", Stage.STAGING)
    assert trusted.transition("alice", "m", "v1", Stage.PRODUCTION).stage == Stage.PRODUCTION


def test_h6_validate_step_gates_production():
    audit = _audit()
    reg = ModelRegistry(audit)
    rec = reg.register("alice", "m", "v1", H1)
    assert rec.validated is False
    reg.transition("alice", "m", "v1", Stage.STAGING)
    n = _n(audit)
    with pytest.raises((ValidationFailed, Conflict)):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert _stage_of(reg, "m", "v1") == Stage.STAGING and reg.current("m", Stage.PRODUCTION) is None
    assert _n(audit) - n <= 1  # a denial may be audited, never more than once
    n = _n(audit)
    out = reg.validate("val", "m", "v1", dict(EVID))
    assert out.validated is True and _stage_of(reg, "m", "v1") == Stage.STAGING
    new = audit.export()[n:]
    assert len(new) == 1 and new[0]["action"] == "model.validate" and new[0]["decision"] == "allow"
    assert new[0]["actor"] == "val"
    assert reg.transition("alice", "m", "v1", Stage.PRODUCTION).stage == Stage.PRODUCTION


@pytest.mark.parametrize("actor,mid,vid,evid,exc", [
    ("val", "m", "v1", {}, ValidationFailed),
    ("val", "m", "v1", None, ValidationFailed),
    ("val", "m", "v1", "ok", ValidationFailed),
    ("val", "m", "v1", ["a"], ValidationFailed),
    ("", "m", "v1", EVID, ValidationFailed),
    (None, "m", "v1", EVID, ValidationFailed),
    (5, "m", "v1", EVID, ValidationFailed),
    ("val", "m", "vX", EVID, NotFound),
    ("val", "ghost", "v1", EVID, NotFound),
])
def test_h6_validate_rejects_bad_input_without_effect(actor, mid, vid, evid, exc):
    audit = _audit()
    reg = ModelRegistry(audit)
    reg.register("alice", "m", "v1", H1)
    n = _n(audit)
    with pytest.raises(exc):
        reg.validate(actor, mid, vid, evid)
    assert reg.history("m")[0].validated is False
    assert _n(audit) - n <= (1 if exc is NotFound else 0)


def test_h6_require_approver_gates_production_and_rollback():
    audit = _audit()
    reg = ModelRegistry(audit, require_approver=lambda actor: actor == "boss")
    _staged(reg, "v1", actor="alice")
    n, hist = _n(audit), reg.history("m")
    with pytest.raises(PolicyDenied):
        reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert reg.history("m") == hist and reg.current("m", Stage.PRODUCTION) is None
    new = audit.export()[n:]
    assert len(new) == 1 and new[0]["decision"] == "deny"
    reg.transition("boss", "m", "v1", Stage.PRODUCTION)
    _prod(reg, "v2", actor="boss", h=H2)
    n, hist = _n(audit), reg.history("m")
    with pytest.raises(PolicyDenied):
        reg.rollback("alice", "m")
    assert reg.history("m") == hist and reg.current("m", Stage.PRODUCTION).version_id == "v2"
    new = audit.export()[n:]
    assert len(new) == 1 and new[0]["decision"] == "deny"
    assert reg.rollback("boss", "m").version_id == "v1"


# ------------------------------------------------------------------------- H7

def test_h7_archived_from_production_is_never_rollback_target():
    reg = ModelRegistry(_audit())
    _prod(reg, "v1")
    reg.transition("alice", "m", "v1", Stage.ARCHIVED)
    _prod(reg, "v2", h=H2)
    cur = reg.current("m", Stage.PRODUCTION)
    assert cur.version_id == "v2" and cur.previous_production is None
    with pytest.raises(Conflict):
        reg.rollback("alice", "m")
    assert reg.current("m", Stage.PRODUCTION).version_id == "v2"
    assert _stage_of(reg, "m", "v1") == Stage.ARCHIVED


def test_h7_archived_from_staging_never_previous_production():
    reg = ModelRegistry(_audit())
    _staged(reg, "v1")
    reg.transition("alice", "m", "v1", Stage.ARCHIVED)
    _prod(reg, "v2", h=H2)
    assert reg.current("m", Stage.PRODUCTION).previous_production is None
    with pytest.raises(Conflict):
        reg.rollback("alice", "m")
    assert _stage_of(reg, "m", "v1") == Stage.ARCHIVED


# ------------------------------------------------------------------------- M6

def test_m6_register_audit_failure_leaves_no_state():
    reg = ModelRegistry(FailingAudit())
    with pytest.raises(OSError):
        reg.register("alice", "m", "v1", H1)
    assert reg.history("m") == [] and reg.current("m", Stage.REGISTERED) is None
    good = _audit()
    reg.audit = good
    reg.register("alice", "m", "v1", H1)  # no Conflict: earlier attempt left nothing
    assert _n(good) == 1


@pytest.mark.parametrize("step", ["validate", "staging", "production", "rollback"])
def test_m6_audit_failure_rolls_back_state(step):
    audit = _audit()
    reg = ModelRegistry(audit)
    if step == "validate":
        reg.register("alice", "m", "v1", H1)
        op = lambda: reg.validate("alice", "m", "v1", dict(EVID))
    elif step == "staging":
        reg.register("alice", "m", "v1", H1)
        reg.validate("alice", "m", "v1", dict(EVID))
        op = lambda: reg.transition("alice", "m", "v1", Stage.STAGING)
    elif step == "production":
        _staged(reg, "v1")
        op = lambda: reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    else:
        _prod(reg, "v1")
        _prod(reg, "v2", h=H2)
        op = lambda: reg.rollback("alice", "m")
    before = reg.history("m")
    stages = {s: reg.current("m", s) for s in Stage}
    reg.audit = FailingAudit()
    with pytest.raises(OSError):
        op()
    assert reg.history("m") == before
    assert {s: reg.current("m", s) for s in Stage} == stages
    reg.audit = audit
    op()  # the same call now succeeds: nothing half-applied


# ------------------------------------------------------------------------- M7

def test_m7_per_stage_holds_multiple_and_current_is_latest_staged():
    reg = ModelRegistry(_audit())
    _staged(reg, "v1")
    _staged(reg, "v2", h=H2)
    assert reg.current("m", Stage.STAGING).version_id == "v2"
    reg.transition("alice", "m", "v1", Stage.PRODUCTION)
    assert reg.current("m", Stage.STAGING).version_id == "v2"
    assert reg.current("m", Stage.PRODUCTION).version_id == "v1"
    _staged(reg, "v3", h=H3)
    assert reg.current("m", Stage.STAGING).version_id == "v3"
    reg.transition("alice", "m", "v3", Stage.ARCHIVED)
    assert reg.current("m", Stage.STAGING).version_id == "v2"


@pytest.mark.parametrize("ds", ["ds-anything", "", 123, b"x", ["a"]])
def test_m7_dataset_version_without_registry_or_non_str(ds):
    audit = _audit()
    reg = ModelRegistry(audit)  # datasets=None
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", H1, dataset_version=ds)
    assert reg.history("m") == [] and _n(audit) == 0


@pytest.mark.parametrize("ds", [123, b"x", ["a"], {"a": 1}])
def test_m7_dataset_version_must_be_str_with_registry(ds):
    reg = ModelRegistry(_audit(), datasets=DatasetRegistry())
    with pytest.raises(ValidationFailed):
        reg.register("alice", "m", "v1", H1, dataset_version=ds)
    assert reg.history("m") == []


# ------------------------------------------------------------------------- M8

def _ds(**kw):
    args = dict(actor="alice", name="d", content_hash=H1, rows=3, schema={"a": "int"})
    args.update(kw)
    return args


@pytest.mark.parametrize("field,val", [
    ("name", None), ("name", ""), ("name", 5), ("name", b"d"),
    *[("name", c) for c in CTRL],
    ("schema", None), ("schema", []), ("schema", "x"), ("schema", 5),
    ("schema", {"a": {1, 2}}), ("schema", {"a": object()}), ("schema", {1: "int"}),
])
def test_m8_dataset_rejects_bad_name_and_schema(field, val):
    audit = _audit()
    reg = DatasetRegistry(audit)
    with pytest.raises(ValidationFailed):
        reg.register(**_ds(**{field: val}))
    assert _n(audit) == 0


def test_m8_schema_is_deep_copied_on_register_and_get():
    reg = DatasetRegistry()
    schema = {"cols": {"a": "int"}}
    vid = reg.register(**_ds(schema=schema))
    snapshot = {"cols": {"a": "int"}}
    schema["cols"]["b"] = "str"  # caller mutation after register
    assert reg.get(vid)["schema"] == snapshot
    assert reg.register(**_ds(schema=copy.deepcopy(snapshot))) == vid
    got = reg.get(vid)
    got["schema"]["cols"]["evil"] = "x"  # mutation of returned dict
    assert reg.get(vid)["schema"] == snapshot
    assert reg.get(vid)["version_id"] == vid
    assert reg.register(**_ds(schema=schema)) != vid  # genuinely different schema -> different id


# ------------------------------------------------------------------------- M4

def _att():
    return build_attestation(SECRET, {"b.txt": H2, "a.txt": H1}, "builder-1", A, {"lr": 0.1})


def test_m4_format_and_extra_keys_are_signed():
    att = _att()
    assert verify_attestation(att, SECRET) is True
    f = dict(att, format="in-toto-like/HMAC (with PKI)")
    assert verify_attestation(f, SECRET) is False
    for k, v in [("extra", 1), ("evil", "x"), ("_", None)]:
        assert verify_attestation(dict(att, **{k: v}), SECRET) is False
    assert verify_attestation(att, SECRET) is True


def test_m4_verify_never_raises_on_malformed():
    good = _att()
    junk = [None, {}, [], "x", 5, dict(good, signature=None), dict(good, parameters={"s": {1}}),
            dict(good, format=["a"]), dict(good, subject=[1]), dict(good, provenance_digest=7)]
    for j in junk:
        assert verify_attestation(j, SECRET) is False
    for s in (None, "str", 5, b""):
        assert verify_attestation(good, s) is False


def test_m4_merkle_domain_separation():
    inner = merkle_root([A, B])
    assert merkle_root([A, B, C]) != merkle_root([inner, C])   # leaf vs internal node
    assert merkle_root([A, B]) != merkle_root([inner])
    assert merkle_root([A, B]) != hashlib.sha256((A + B).encode()).hexdigest()
    assert merkle_root([A, B, C]) == merkle_root([A, B, C])    # deterministic
    assert merkle_root([A, B]) != merkle_root([B, A])
    single = merkle_root([A])
    assert single != A and len(single) == 64
    # single leaf root = sha256(0x00 || leaf); leaf encoding (ascii hex vs raw bytes) is not fixed here
    assert single in {
        hashlib.sha256(b"\x00" + A.encode()).hexdigest(),
        hashlib.sha256(b"\x00" + bytes.fromhex(A)).hexdigest(),
    }


# ------------------------------------------------------------------------- M5

def test_m5_rules_are_immutable_after_construction():
    rules = [{"kind": "keep_last", "n": 1}]
    pol = RetentionPolicy(rules)
    with pytest.raises((AttributeError, TypeError)):
        pol.rules.append({"kind": "keep_last", "n": 0})
    rules.append({"kind": "bogus"})
    rules[0]["n"] = 0  # caller mutation must not leak in
    recs = [{"id": "r1", "created_ts": 1}, {"id": "r2", "created_ts": 2}]
    assert pol.apply(recs, now=10) == ["r1"]


def test_m5_apply_audits_with_resource_and_durable_chain_verifies(tmp_path):
    store = DurableAuditStore(str(tmp_path / "a.db"), SECRET)
    pol = RetentionPolicy([{"kind": "keep_last", "n": 1}])
    recs = [{"id": "r1", "created_ts": 1}, {"id": "r2", "created_ts": 2}, {"id": "r3", "created_ts": 3}]
    assert pol.apply(recs, now=10, audit=store) == ["r1", "r2"]
    ex = store.export()
    assert len(ex) == 2 and {e["resource"] for e in ex} == {"r1", "r2"}
    assert all(isinstance(e["resource"], str) and e["resource"] for e in ex)
    assert ChainVerifier(SECRET).verify_export(ex, head=store.head()) is True
    store.close()


def test_m5_apply_never_audits_undeleted_ids():
    audit = _audit()
    pol = RetentionPolicy([{"kind": "max_age", "max_age_s": 100}, {"kind": "keep_last", "n": 2}])
    recs = [{"id": "r1", "created_ts": 100}, {"id": "r2", "created_ts": 200},
            {"id": "r3", "created_ts": 300}, {"id": "r4", "created_ts": 950}]
    out = pol.apply(recs, now=1000, audit=audit)
    assert out == ["r1", "r2"]
    assert sorted(e["resource"] for e in audit.export()) == out
    audit2 = _audit()
    assert pol.apply([{"id": "n1", "created_ts": 999}], now=1000, audit=audit2) == []
    assert audit2.export() == []


# ------------------------------------------------------------------------- M9

@pytest.mark.parametrize("field", ["model_id", "version_id"])
@pytest.mark.parametrize("bad", CTRL)
def test_m9_control_chars_in_ids_rejected(field, bad):
    audit = _audit()
    reg = ModelRegistry(audit)
    kw = {"model_id": "m", "version_id": "v1"}
    kw[field] = bad
    with pytest.raises(ValidationFailed):
        reg.register("alice", kw["model_id"], kw["version_id"], H1)
    assert _n(audit) == 0


@pytest.mark.parametrize("comp", [
    ("a\x00b", "1"), ("a\nb", "1"), ("a\x07b", "1"), ("pkg", "1\x00"), ("pkg", "1\n2"),
    ("a@b", "1"), ("@a", "1"), ("pkg", "1@2"),
    {"name": "a\x1fb", "version": "1"}, {"name": "a@b", "version": "1"},
])
def test_m9_sbom_rejects_control_chars_and_ambiguous_purl(comp):
    with pytest.raises(ValidationFailed):
        sbom_from_requirements([("ok", "1"), comp])


@pytest.mark.parametrize("comps", [
    [("Foo", "1"), ("foo", "2")],
    [{"name": "NumPy", "version": "1"}, {"name": "numpy", "version": "1"}],
    [("foo", "1"), ("FOO", "1")],
])
def test_m9_sbom_case_insensitive_duplicates(comps):
    with pytest.raises(ValidationFailed):
        sbom_from_requirements(comps)
