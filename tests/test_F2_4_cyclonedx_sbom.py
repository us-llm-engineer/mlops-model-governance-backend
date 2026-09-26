"""F2.4 cyclonedx_sbom: frozen tests for exec/mlops/ext/cyclonedx_sbom.py.

Covers build_cyclonedx_sbom / validate_cyclonedx_sbom / sbom_matches_requirements.
Mirrors supply_chain.sbom_from_requirements's OWN input validation (empty list,
non-str/empty name+version, no control chars/@//,/whitespace in names, no
control chars/@ in versions, case-INSENSITIVE duplicate names -- verified
against exec/mlops/supply_chain.py's own dedup logic and its existing
test_S3_4_mutation_hardening.py::"Flask"/"flask" duplicate case) -- that
validation must run and raise ValidationFailed BEFORE the cyclonedx library is
ever touched.

Determinism is the whole point of this wrapper: cyclonedx-python-lib's
Component defaults to a RANDOM bom_ref and Bom defaults to a wall-clock
metadata.timestamp and a random serialNumber when neither is pinned -- two
calls with identical input must be byte-identical, and no such leftover
nondeterministic field may survive in test-visible output.

NOTE: like the frozen suites this file puts <repo>/exec first on sys.path, so
a mutation harness must copy tests/ AND exec/ into the same scratch root
(PYTHONPATH alone is NOT enough).
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import ValidationFailed  # noqa: E402
from mlops.ext.cyclonedx_sbom import (  # noqa: E402
    build_cyclonedx_sbom,
    sbom_matches_requirements,
    validate_cyclonedx_sbom,
)

VALID_TUPLES = [("attrs", "23.1"), ("zlib", "1.2"), ("mid", "0.1")]
VALID_DICTS = [{"name": "attrs", "version": "23.1"}, {"name": "zlib", "version": "1.2"}, {"name": "mid", "version": "0.1"}]

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _scan_for_nondeterministic_leftovers(obj, path=""):
    """Recursively collect (path, value) pairs that look like a UUID or an ISO
    timestamp anywhere in the structure. Used to catch a leaked bom-ref/
    timestamp/serialNumber default that a mutant failed to pin/strip."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(_scan_for_nondeterministic_leftovers(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_scan_for_nondeterministic_leftovers(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        bare = obj[len("urn:uuid:"):] if obj.startswith("urn:uuid:") else obj
        if _UUID_RE.match(bare) or _ISO_TS_RE.match(obj):
            hits.append((path, obj))
    return hits


# =============================================================== happy path

def test_build_tuples_happy_path_shape():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    names = [c["name"] for c in sbom["components"]]
    assert sorted(names) == ["attrs", "mid", "zlib"]
    for c in sbom["components"]:
        assert c["type"] == "library"
        assert c["bom-ref"] == f"{c['name']}@{c['version']}"


def test_build_dicts_matches_tuples_form():
    sbom_tuples = build_cyclonedx_sbom(VALID_TUPLES)
    sbom_dicts = build_cyclonedx_sbom(VALID_DICTS)
    assert json.dumps(sbom_tuples, sort_keys=True) == json.dumps(sbom_dicts, sort_keys=True)


# =============================================================== input validation (mirrors sbom_from_requirements)

def test_build_empty_list_raises_validation_failed():
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([])


@pytest.mark.parametrize(
    "bad_name",
    ["foo\x01bar", "foo@bar", "foo/bar", "foo bar"],
    ids=["control-char", "at-sign", "slash", "whitespace"],
)
def test_build_bad_name_chars_raise(bad_name):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([(bad_name, "1.0")])


@pytest.mark.parametrize(
    "bad_version",
    ["1.0@2", "1.0\x02"],
    ids=["at-sign", "control-char"],
)
def test_build_bad_version_chars_raise(bad_version):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([("ok", bad_version)])


@pytest.mark.parametrize("bad_name", [123, "", None], ids=["int", "empty-str", "none"])
def test_build_non_str_or_empty_name_raises(bad_name):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([(bad_name, "1.0")])


@pytest.mark.parametrize("bad_version", [123, "", None], ids=["int", "empty-str", "none"])
def test_build_non_str_or_empty_version_raises(bad_version):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([("ok", bad_version)])


@pytest.mark.parametrize(
    "bad_component",
    ["not-a-tuple-or-dict", ("only-one",), {"name": "x"}],
    ids=["plain-string", "short-tuple", "dict-missing-version"],
)
def test_build_invalid_component_shape_raises(bad_component):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([bad_component])


def test_build_duplicate_names_case_insensitive_raises():
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([("Flask", "1"), ("flask", "2")])


def test_build_duplicate_names_exact_case_raises():
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom([("dup", "1"), ("other", "2"), ("dup", "3")])


def test_build_validation_failed_before_any_library_call(monkeypatch):
    """The exception raised for bad input must be OUR ValidationFailed, not any
    cyclonedx-internal exception type -- proof our validation runs first."""
    import mlops.ext.cyclonedx_sbom as mod

    def _boom(*a, **k):
        raise AssertionError("cyclonedx library must not be called for invalid input")

    monkeypatch.setattr(mod, "Component", _boom)
    monkeypatch.setattr(mod, "Bom", _boom)
    with pytest.raises(ValidationFailed) as excinfo:
        build_cyclonedx_sbom([("bad name", "1.0")])
    assert excinfo.type is ValidationFailed


# =============================================================== determinism (the whole point)

def test_build_twice_identical_tuples_is_byte_identical():
    first = build_cyclonedx_sbom(VALID_TUPLES)
    second = build_cyclonedx_sbom(VALID_TUPLES)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_twice_identical_dicts_is_byte_identical():
    first = build_cyclonedx_sbom(VALID_DICTS)
    second = build_cyclonedx_sbom(VALID_DICTS)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_omits_timestamp_and_serial_number_when_not_requested():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    assert "serialNumber" not in sbom
    metadata = sbom.get("metadata")
    if metadata is not None:
        assert "timestamp" not in metadata


def test_build_output_has_no_leftover_nondeterministic_fields():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    hits = _scan_for_nondeterministic_leftovers(sbom)
    assert hits == [], f"found unexpected UUID/timestamp-shaped leftovers: {hits}"


# =============================================================== serial_number

def test_build_accepts_valid_urn_uuid_serial_number():
    sn = "urn:uuid:12345678-1234-5678-1234-567812345678"
    sbom = build_cyclonedx_sbom(VALID_TUPLES, serial_number=sn)
    assert sbom["serialNumber"] == sn


@pytest.mark.parametrize(
    "bad_serial",
    ["urn:invalid", "12345678-1234-5678-1234-567812345678", 12345],
    ids=["bad-urn-scheme", "bare-uuid-string", "non-str"],
)
def test_build_rejects_bad_serial_number(bad_serial):
    with pytest.raises(ValidationFailed):
        build_cyclonedx_sbom(VALID_TUPLES, serial_number=bad_serial)


# =============================================================== validate_cyclonedx_sbom

def test_validate_accepts_its_own_build_output():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    assert validate_cyclonedx_sbom(sbom) is None


def test_validate_rejects_component_missing_type_key_count_only():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    del sbom["components"][0]["type"]
    with pytest.raises(ValidationFailed) as excinfo:
        validate_cyclonedx_sbom(sbom)
    msg = str(excinfo.value)
    assert "is a required property" not in msg
    assert "is not of type" not in msg
    assert "Failed validating" not in msg
    assert any(ch.isdigit() for ch in msg)


def test_validate_rejects_component_version_wrong_type_count_only():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    sbom["components"][0]["version"] = 12345
    with pytest.raises(ValidationFailed) as excinfo:
        validate_cyclonedx_sbom(sbom)
    msg = str(excinfo.value)
    assert "is not of type" not in msg
    assert "Failed validating" not in msg
    assert any(ch.isdigit() for ch in msg)


# =============================================================== sbom_matches_requirements

def test_matches_true_for_sboms_own_input():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    assert sbom_matches_requirements(sbom, VALID_TUPLES) is True


def test_matches_false_after_adding_a_component():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    assert sbom_matches_requirements(sbom, VALID_TUPLES + [("extra", "9.9")]) is False


def test_matches_false_after_changing_a_version():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    changed = [("attrs", "99.0"), ("zlib", "1.2"), ("mid", "0.1")]
    assert sbom_matches_requirements(sbom, changed) is False


def test_matches_is_order_independent():
    sbom = build_cyclonedx_sbom(VALID_TUPLES)
    reordered = list(reversed(VALID_TUPLES))
    assert sbom_matches_requirements(sbom, reordered) is True
