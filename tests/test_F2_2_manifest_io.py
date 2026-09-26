"""F2.2 slim suite: YAML manifest loading + schema validation (mlops.svc.manifest_io),
contract in the API contract section 2.

Frozen expectations (contract-derived):
  load_manifests(raw, *, max_bytes=1_000_000, max_documents=200) -> list[dict]:
    byte-size cap (len(raw.encode()) > max_bytes) enforced BEFORE any YAML parsing;
    parses with yaml.safe_load_all ONLY (never yaml.load/unsafe_load/full_load); more than
    max_documents yielded -> ValidationFailed without materializing the rest of the
    generator; each document must be a dict or None (blank doc, skipped) else
    ValidationFailed; malformed YAML syntax is caught and re-raised as ValidationFailed,
    never a raw yaml.YAMLError, message never echoes the raw input text.
  check_amplification(doc, *, max_visits=5000) -> None (internal, called by load_manifests
    on every parsed document, BEFORE handing to config_lint): walks the structure counting
    total node visits (not unique nodes); an exponential anchor/alias-amplification
    structure (many-level doubling aliases) must exceed max_visits and raise
    ValidationFailed, while an ordinary document with a few (2-3) reused anchors (no
    exponential doubling) is accepted.
  validate_schema(doc, schema) -> list[str]: schema.check_schema() called first -- a
    broken schema is OUR ValidationFailed, never a raw jsonschema.SchemaError; valid doc ->
    []; invalid doc -> "<dotted.path>: <message>" strings, sorted by path, capped at 100
    even when the document violates the schema hundreds of times.
  MANIFEST_SCHEMA: dict module constant, minimal K8s shape (type object, requires
    apiVersion/kind/metadata, metadata requires name: string).
  load_and_lint(raw, *, image_allowlist=None) -> (docs, findings): loads via
    load_manifests, schema-validates each doc against MANIFEST_SCHEMA (failures become
    config_lint.Finding-shaped dicts with rule_id="schema-invalid", severity="block",
    PREPENDED before the real mlops.config_lint.lint_documents findings).

Scope note: this suite exercises mlops.svc.manifest_io only (section 2 of the F2
contract); sections 1/3/4/5/6 of the same contract are out of scope for this file.
"""
import os
import sys
import time

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import ValidationFailed
from mlops.svc.manifest_io import (
    MANIFEST_SCHEMA,
    check_amplification,
    load_and_lint,
    load_manifests,
    validate_schema,
)


# ---------------------------------------------------------------------------
# load_manifests: happy path
# ---------------------------------------------------------------------------

def test_single_document_happy_path():
    docs = load_manifests("apiVersion: v1\nkind: Pod\nmetadata:\n  name: p1\n")
    assert docs == [{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p1"}}]


def test_multiple_dash_separated_documents():
    raw = "kind: Pod\nname: a\n---\nkind: Service\nname: b\n---\nkind: Job\nname: c\n"
    docs = load_manifests(raw)
    assert len(docs) == 3
    assert [d["kind"] for d in docs] == ["Pod", "Service", "Job"]


def test_blank_document_skipped_not_counted_as_error():
    raw = "kind: Pod\n---\n---\nkind: Service\n"
    docs = load_manifests(raw)
    assert docs == [{"kind": "Pod"}, {"kind": "Service"}]


def test_document_order_preserved():
    raw = "\n---\n".join(f"idx: {i}" for i in range(20))
    docs = load_manifests(raw)
    assert [d["idx"] for d in docs] == list(range(20))


# ---------------------------------------------------------------------------
# load_manifests: byte-size cap, enforced BEFORE parsing
# ---------------------------------------------------------------------------

def test_size_cap_exactly_at_boundary_is_allowed():
    raw = "kind: Pod\n"
    n = len(raw.encode("utf-8"))
    # exactly at the cap must NOT raise (contract: strictly `>` triggers)
    docs = load_manifests(raw, max_bytes=n)
    assert docs == [{"kind": "Pod"}]


def test_size_cap_enforced_before_parsing_with_malformed_oversized_input():
    # Malformed YAML (unbalanced flow collection) that is also over the byte cap.
    # If the size cap were checked after/around a parse attempt we'd see a YAML
    # parse-error message instead of the dedicated size message.
    raw = "{" * 200  # invalid YAML syntax, well over the tiny cap below
    with pytest.raises(ValidationFailed) as exc_info:
        load_manifests(raw, max_bytes=20)
    msg = str(exc_info.value)
    assert "max_bytes" in msg or "20" in msg
    assert "not valid YAML" not in msg


def test_size_cap_error_is_fast():
    raw = "{" * 5000
    t0 = time.time()
    with pytest.raises(ValidationFailed):
        load_manifests(raw, max_bytes=10)
    assert time.time() - t0 < 1.0


# ---------------------------------------------------------------------------
# load_manifests: max_documents cap, lazy short-circuit
# ---------------------------------------------------------------------------

def test_max_documents_cap_exceeded_raises():
    raw = "\n---\n".join(["a: 1"] * 10)
    with pytest.raises(ValidationFailed):
        load_manifests(raw, max_documents=5)


def test_max_documents_cap_plus_one_raises():
    raw = "\n---\n".join(["a: 1"] * 6)
    with pytest.raises(ValidationFailed):
        load_manifests(raw, max_documents=5)


def test_max_documents_cap_exactly_at_boundary_allowed():
    raw = "\n---\n".join(["a: 1"] * 5)
    docs = load_manifests(raw, max_documents=5)
    assert len(docs) == 5


def test_max_documents_short_circuits_without_materializing_all():
    # Far more documents than the cap; a correct implementation must stop pulling
    # from the yaml.safe_load_all generator as soon as the cap is exceeded, rather
    # than parsing all of them first. Fully materializing ~300k tiny docs with
    # PyYAML takes on the order of many seconds; a lazy stop is near-instant.
    n = 300_000
    raw = "\n---\n".join(["{}"] * n)
    t0 = time.time()
    with pytest.raises(ValidationFailed):
        load_manifests(raw, max_bytes=10_000_000, max_documents=5)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"took {elapsed}s -- looks like the generator was fully materialized"


# ---------------------------------------------------------------------------
# load_manifests: document shape validation
# ---------------------------------------------------------------------------

def test_non_dict_document_list_rejected():
    with pytest.raises(ValidationFailed):
        load_manifests("- 1\n- 2\n- 3\n")


def test_non_dict_document_scalar_rejected():
    with pytest.raises(ValidationFailed):
        load_manifests("just-a-string\n")


def test_non_dict_document_int_rejected():
    with pytest.raises(ValidationFailed):
        load_manifests("42\n")


# ---------------------------------------------------------------------------
# load_manifests: malformed YAML syntax
# ---------------------------------------------------------------------------

def test_malformed_yaml_syntax_raises_validationfailed():
    with pytest.raises(ValidationFailed):
        load_manifests("key: [unclosed\n")


def test_malformed_yaml_error_message_does_not_echo_raw_input():
    secret_marker = "TOTALLY-UNIQUE-RAW-TOKEN-98765"
    raw = f"key: [unclosed {secret_marker}\n"
    with pytest.raises(ValidationFailed) as exc_info:
        load_manifests(raw)
    assert secret_marker not in str(exc_info.value)


def test_malformed_yaml_never_lets_raw_yamlerror_escape():
    try:
        load_manifests("a: [1, 2\nb: }\n")
    except ValidationFailed as e:
        assert not isinstance(e, yaml.YAMLError)
    else:
        pytest.fail("expected ValidationFailed for malformed YAML")


# ---------------------------------------------------------------------------
# load_manifests: safe loading only (no code execution via YAML tags)
# ---------------------------------------------------------------------------

def test_unsafe_python_tag_rejected_not_executed(monkeypatch):
    calls = []
    monkeypatch.setattr("os.system", lambda cmd: calls.append(cmd) or 0)
    raw = "exploit: !!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(ValidationFailed):
        load_manifests(raw)
    assert calls == [], "os.system must never be invoked by an untrusted manifest"


def test_only_safe_load_all_used_never_unsafe_loaders(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("yaml.load/unsafe_load/full_load must never be called")

    monkeypatch.setattr(yaml, "load", _boom)
    monkeypatch.setattr(yaml, "unsafe_load", _boom)
    monkeypatch.setattr(yaml, "full_load", _boom)
    docs = load_manifests("kind: Pod\nmetadata:\n  name: ok\n")
    assert docs == [{"kind": "Pod", "metadata": {"name": "ok"}}]


# ---------------------------------------------------------------------------
# check_amplification: the frozen alias-amplification trap
# ---------------------------------------------------------------------------

def _alias_amplification_yaml(levels: int = 14) -> str:
    """Exact pattern from the contract: a{i}: &a{i} [*a{i-1}, *a{i-1}]."""
    lines = ["a0: &a0 0"]
    for i in range(1, levels + 1):
        lines.append(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]")
    return "\n".join(lines)


def test_alias_amplification_14_level_pattern_rejected():
    raw = _alias_amplification_yaml(14)
    doc = yaml.safe_load(raw)  # safe_load_all happily constructs this (shared objects)
    assert isinstance(doc, dict)
    with pytest.raises(ValidationFailed):
        check_amplification(doc)


def test_alias_amplification_rejected_through_load_manifests():
    raw = _alias_amplification_yaml(14)
    with pytest.raises(ValidationFailed):
        load_manifests(raw)


def test_few_reused_anchors_accepted():
    raw = (
        "shared: &shared {a: 1, b: 2}\n"
        "x: *shared\n"
        "y: *shared\n"
        "z: *shared\n"
    )
    doc = yaml.safe_load(raw)
    check_amplification(doc)  # must not raise
    docs = load_manifests(raw)
    assert docs == [doc]


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_broken_schema_raises_validationfailed():
    broken_schema = {"type": 123}  # "type" must be a string or array of strings
    with pytest.raises(ValidationFailed):
        validate_schema({"a": 1}, broken_schema)


def test_validate_schema_broken_schema_no_raw_schemaerror_escapes():
    import jsonschema

    broken_schema = {"properties": "not-an-object"}
    try:
        validate_schema({"a": 1}, broken_schema)
    except ValidationFailed as e:
        assert not isinstance(e, jsonschema.exceptions.SchemaError)
    else:
        pytest.fail("expected ValidationFailed for a broken schema")


def test_validate_schema_valid_doc_empty_list():
    doc = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p1"}}
    assert validate_schema(doc, MANIFEST_SCHEMA) == []


def test_validate_schema_invalid_doc_dotted_path_messages():
    doc = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": 123}}
    errors = validate_schema(doc, MANIFEST_SCHEMA)
    assert len(errors) >= 1
    assert all(isinstance(e, str) and ": " in e for e in errors)
    assert any("metadata" in e and "name" in e for e in errors)


def test_validate_schema_messages_sorted_by_path():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    doc = {"items": [0, "ok", 1, 2, "ok2", 3]}
    errors = validate_schema(doc, schema)
    assert errors == sorted(errors)
    assert len(errors) == 4  # indices 0, 2, 3, 5 are wrong-typed


def test_validate_schema_capped_at_100_with_many_errors():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    doc = {"items": [0] * 600}  # 600 schema violations
    errors = validate_schema(doc, schema)
    assert len(errors) == 100


def test_manifest_schema_constant_shape():
    assert MANIFEST_SCHEMA["type"] == "object"
    assert set(MANIFEST_SCHEMA["required"]) >= {"apiVersion", "kind", "metadata"}
    meta_schema = MANIFEST_SCHEMA["properties"]["metadata"]
    assert "name" in meta_schema.get("required", [])
    assert meta_schema["properties"]["name"]["type"] == "string"


def _rule_id(finding):
    """Findings list mixes schema-invalid dicts with real config_lint.Finding
    dataclass instances -- read rule_id from either shape."""
    return finding["rule_id"] if isinstance(finding, dict) else finding.rule_id


def _severity(finding):
    return finding["severity"] if isinstance(finding, dict) else finding.severity


# ---------------------------------------------------------------------------
# load_and_lint
# ---------------------------------------------------------------------------

def test_load_and_lint_happy_path_returns_docs_and_findings():
    raw = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: p1\n  namespace: prod\nspec:\n  containers: []\n"
    docs, findings = load_and_lint(raw)
    assert isinstance(docs, list) and len(docs) == 1
    assert isinstance(findings, list)


def test_load_and_lint_schema_invalid_finding_prepended():
    # Fails MANIFEST_SCHEMA (missing apiVersion) but has a real pod-spec that
    # config_lint.lint_documents will also flag (no resources/probes/etc).
    raw = (
        "kind: Pod\n"
        "metadata:\n"
        "  name: p1\n"
        "  namespace: prod\n"
        "spec:\n"
        "  containers:\n"
        "  - name: c1\n"
        "    image: myimg:1.0\n"
    )
    docs, findings = load_and_lint(raw)
    assert len(findings) >= 1
    schema_findings = [f for f in findings if _rule_id(f) == "schema-invalid"]
    assert len(schema_findings) >= 1
    for f in schema_findings:
        assert isinstance(f, dict), "schema-invalid findings must be plain dicts"
        assert _severity(f) == "block"
    non_schema_indices = [i for i, f in enumerate(findings) if _rule_id(f) != "schema-invalid"]
    assert non_schema_indices, "expected real config_lint findings for this pod spec too"
    schema_indices = [i for i, f in enumerate(findings) if _rule_id(f) == "schema-invalid"]
    # every schema-invalid finding must be prepended ahead of every non-schema finding
    assert max(schema_indices) < min(non_schema_indices)


def test_load_and_lint_valid_doc_no_schema_findings_only_lint_findings():
    raw = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: p1\n"
    docs, findings = load_and_lint(raw)
    assert all(_rule_id(f) != "schema-invalid" for f in findings)
