"""
S2.3 Supply Chain: Merkle roots, attestations, SBOMs, and retention policies.

This module provides cryptographic integrity for model artifacts and datasets:

* merkle_root: Deterministic Merkle tree with domain separation.
* build_attestation: HMAC-SHA256 signed attestations with materials root.
* verify_attestation: Safe verification (never raises on malformed input).
* sbom_from_requirements: CycloneDX-like SBOM with duplicate detection.
* RetentionPolicy: Combine max_age and keep_last retention rules.

Stdlib only: hashlib, hmac, json, dataclasses, math, copy, string.
"""

import copy
import hashlib
import hmac
import json
import math
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .kernel import canonical_json, sha256_hex, ValidationFailed

__all__ = [
    "merkle_root",
    "build_attestation",
    "verify_attestation",
    "sbom_from_requirements",
    "RetentionPolicy",
]


# Private validators (shared with other modules)
def _is_finite_real(val: Any) -> bool:
    """Check if val is a real finite number (not bool, str, None, NaN, inf)."""
    if isinstance(val, bool) or val is None or isinstance(val, str):
        return False
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return True
    return False


def _is_hex64(val: Any) -> bool:
    """Check if val is 64-character lowercase hex."""
    return (
        isinstance(val, str)
        and len(val) == 64
        and all(c in "0123456789abcdef" for c in val)
    )


def _is_nonempty_str(val: Any) -> bool:
    """Check if val is a non-empty string."""
    return isinstance(val, str) and len(val) > 0


def _has_no_control_chars(val: str) -> bool:
    """Check if val has no control characters."""
    return all(c not in string.whitespace or c == ' ' for c in val) and all(ord(c) >= 32 or c == ' ' for c in val)


def merkle_root(leaves: List[str]) -> str:
    """Compute Merkle root over hex digest leaves with domain separation.

    Uses b"\x00" prefix for leaves, b"\x01" for nodes.
    Leaves are validated as 64-character lowercase hex strings.
    Order matters (not sorted); odd last node is promoted unchanged.
    Empty list raises ValueError.
    """
    if not leaves:
        raise ValueError("merkle_root requires at least one leaf")

    # Validate all leaves are 64-char lowercase hex
    for leaf in leaves:
        if not _is_hex64(leaf):
            raise ValueError(f"Invalid hex digest: {leaf}")

    # Single leaf: domain-separated hash
    if len(leaves) == 1:
        leaf_hash = hashlib.sha256(b"\x00" + bytes.fromhex(leaves[0])).hexdigest()
        return leaf_hash

    # Build tree bottom-up: pair nodes, promote odd node
    nodes = [hashlib.sha256(b"\x00" + bytes.fromhex(leaf)).hexdigest() for leaf in leaves]
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                # Pair: hash with domain separation
                node_hash = hashlib.sha256(b"\x01" + bytes.fromhex(nodes[i]) + bytes.fromhex(nodes[i + 1])).hexdigest()
                next_level.append(node_hash)
            else:
                # Odd node: promote unchanged
                next_level.append(nodes[i])
        nodes = next_level

    return nodes[0]


def build_attestation(
    secret: bytes,
    inputs: Dict[str, str],
    builder_id: str,
    artifact_digest: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an HMAC-SHA256 signed attestation.

    Args:
        secret: Secret for HMAC signing (must be non-empty bytes)
        inputs: dict of input_name -> sha256 hex digest (non-empty, all values 64-hex)
        builder_id: identifier of the builder (non-empty str)
        artifact_digest: sha256 hex digest of the artifact (64-hex)
        parameters: metadata dict (JSON-safe via canonical_json)

    Returns:
        dict with subject, materials_root, builder_id, parameters,
        provenance_digest, signature, format

    Raises:
        ValidationFailed: Invalid inputs
    """
    # Validate secret
    if not isinstance(secret, bytes) or len(secret) == 0:
        raise ValidationFailed("secret must be non-empty bytes")

    # Validate inputs: non-empty dict with 64-hex values
    if not isinstance(inputs, dict) or len(inputs) == 0:
        raise ValidationFailed("inputs must be non-empty dict")

    for name, digest in inputs.items():
        if not _is_nonempty_str(name):
            raise ValidationFailed(f"input name must be non-empty str: {name}")
        if not _is_hex64(digest):
            raise ValidationFailed(f"input digest must be 64-hex: {digest}")

    # Validate builder_id
    if not _is_nonempty_str(builder_id):
        raise ValidationFailed(f"builder_id must be non-empty str: {builder_id}")

    # Validate artifact_digest
    if not _is_hex64(artifact_digest):
        raise ValidationFailed(f"artifact_digest must be 64-hex: {artifact_digest}")

    # Validate parameters is a dict
    if not isinstance(parameters, dict):
        raise ValidationFailed(f"parameters must be dict, got {type(parameters).__name__}")

    # Sort input names to get deterministic materials_root
    sorted_digests = [inputs[name] for name in sorted(inputs.keys())]
    materials_root_val = merkle_root(sorted_digests)

    format_val = "in-toto-like/HMAC (no PKI, no TEE)"

    # Build body: format, builder_id, materials_root, parameters, subject
    body = {
        "builder_id": builder_id,
        "format": format_val,
        "materials_root": materials_root_val,
        "parameters": parameters,
        "subject": artifact_digest,
    }

    # Compute provenance_digest over canonical JSON of the body
    provenance_json = canonical_json(body)
    provenance_digest = sha256_hex(provenance_json)

    # Sign the provenance digest
    signature = hmac.new(secret, provenance_digest.encode(), hashlib.sha256).hexdigest()

    # Format and all other keys are covered by the signature
    return {
        "builder_id": builder_id,
        "format": format_val,
        "materials_root": materials_root_val,
        "parameters": parameters,
        "provenance_digest": provenance_digest,
        "signature": signature,
        "subject": artifact_digest,
    }


def verify_attestation(
    att: Any,
    secret: bytes,
    artifact_digest: Optional[str] = None,
) -> bool:
    """Verify an attestation signature.

    Returns False on any error (never raises, fail-safe).
    If artifact_digest is provided, checks subject match.
    """
    try:
        # Fail on non-dict or missing required fields
        if not isinstance(att, dict):
            return False

        required = {"subject", "materials_root", "builder_id", "parameters", "provenance_digest", "signature", "format"}
        expected_keys = required

        # Check for exact key match (no extra keys allowed)
        if set(att.keys()) != expected_keys:
            return False

        if not required.issubset(att.keys()):
            return False

        # Validate types strictly
        if not isinstance(att["builder_id"], str):
            return False
        if not isinstance(att["materials_root"], str):
            return False
        if not isinstance(att["subject"], str):
            return False
        if not isinstance(att["provenance_digest"], str):
            return False
        if not isinstance(att["signature"], str) or att["signature"] is None:
            return False
        if not isinstance(att["format"], str):
            return False
        if not isinstance(att["parameters"], dict):
            return False
        if not isinstance(att.get("secret"), bytes) and att.get("secret") is not None:
            # secret is optional but if present must be bytes
            pass

        # Verify subject match if provided
        if artifact_digest is not None and att["subject"] != artifact_digest:
            return False

        # Reconstruct provenance body (deterministic order for digest)
        body = {
            "builder_id": att["builder_id"],
            "format": att["format"],
            "materials_root": att["materials_root"],
            "parameters": att["parameters"],
            "subject": att["subject"],
        }

        # Verify provenance_digest
        provenance_json = canonical_json(body)
        expected_digest = sha256_hex(provenance_json)
        if att["provenance_digest"] != expected_digest:
            return False

        # Verify signature over provenance_digest
        expected_sig = hmac.new(secret, expected_digest.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(att["signature"], expected_sig)
    except Exception:
        return False


def sbom_from_requirements(
    components: List[Union[tuple, Dict[str, str]]],
) -> Dict[str, Any]:
    """Build a CycloneDX-like SBOM from requirement list.

    Components are tuples (name, version) or dicts with "name" and "version".
    Sorted by name; duplicate names (case-sensitive) raise ValidationFailed.
    Names must not have control chars, @, /, or whitespace.

    Args:
        components: List of tuples (name, version) or dicts with name/version keys

    Returns:
        CycloneDX-like SBOM dict

    Raises:
        ValidationFailed: Empty list, invalid format, duplicates, bad types
    """
    if not components:
        raise ValidationFailed("components list must not be empty")

    # Normalize to dict format
    normalized = []
    for comp in components:
        if isinstance(comp, tuple):
            if len(comp) != 2:
                raise ValidationFailed(f"tuple component must have exactly 2 elements, got {len(comp)}")
            name, version = comp
        elif isinstance(comp, dict):
            if "name" not in comp or "version" not in comp:
                raise ValidationFailed(f"dict component missing 'name' or 'version': {comp}")
            name = comp["name"]
            version = comp["version"]
        else:
            raise ValidationFailed(f"Invalid component format: {type(comp).__name__}")

        # Validate name and version are non-empty strings
        if not _is_nonempty_str(name):
            raise ValidationFailed(f"component name must be non-empty str: {name}")
        if not _is_nonempty_str(version):
            raise ValidationFailed(f"component version must be non-empty str: {version}")

        # Validate name has no control chars, @, /, whitespace
        if not _has_no_control_chars(name) or any(c in "@/ \t\n\r" for c in name):
            raise ValidationFailed(f"component name contains invalid chars: {name}")

        # Validate version has no control chars or @ (ambiguous in purl)
        if not _has_no_control_chars(version) or "@" in version:
            raise ValidationFailed(f"component version contains invalid chars: {version}")

        normalized.append({"name": name, "version": version})

    # Check for duplicates by name (case-insensitive)
    names = [c["name"] for c in normalized]
    names_lower = [n.lower() for n in names]
    if len(names_lower) != len(set(names_lower)):
        raise ValidationFailed("Duplicate component names in SBOM (case-insensitive)")

    # Sort by name
    normalized.sort(key=lambda c: c["name"])

    # Build components with purl and sha256
    components_list = []
    for comp in normalized:
        name = comp["name"]
        version = comp["version"]
        purl = f"pkg:pypi/{name}@{version}"
        # Compute sha256 of canonical component dict
        comp_dict = {"name": name, "version": version, "purl": purl}
        comp_sha = sha256_hex(canonical_json(comp_dict))

        components_list.append({
            "name": name,
            "purl": purl,
            "sha256": comp_sha,
            "version": version,
        })

    return {
        "bomFormat": "CycloneDX",
        "components": components_list,
        "specVersion": "1.4",
    }


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention policy combining max_age and keep_last rules.

    Rules are stored as a tuple of frozen copies to prevent mutation.
    Validates rules on initialization to ensure well-formed policy.
    """

    rules: Tuple[Dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate rules on initialization."""
        if not isinstance(self.rules, tuple):
            # Convert list to tuple if needed
            object.__setattr__(self, 'rules', tuple(copy.deepcopy(r) for r in self.rules) if isinstance(self.rules, list) else self.rules)

        if len(self.rules) == 0:
            raise ValidationFailed("rules list must not be empty")

        for i, rule in enumerate(self.rules):
            if not isinstance(rule, dict):
                raise ValidationFailed(f"rule {i} must be a dict, got {type(rule).__name__}")

            kind = rule.get("kind")
            if kind not in {"max_age", "keep_last"}:
                raise ValidationFailed(f"rule {i}: unknown kind '{kind}'")

            if kind == "max_age":
                if "max_age_s" not in rule:
                    raise ValidationFailed(f"rule {i}: max_age requires 'max_age_s' parameter")
                max_age_s = rule["max_age_s"]
                if not _is_finite_real(max_age_s):
                    raise ValidationFailed(f"rule {i}: max_age_s must be real finite number, got {type(max_age_s).__name__}")
                if max_age_s < 0:
                    raise ValidationFailed(f"rule {i}: max_age_s must be >= 0, got {max_age_s}")

            elif kind == "keep_last":
                if "n" not in rule:
                    raise ValidationFailed(f"rule {i}: keep_last requires 'n' parameter")
                n = rule["n"]
                if isinstance(n, bool) or not isinstance(n, int):
                    raise ValidationFailed(f"rule {i}: n must be int, got {type(n).__name__}")
                if n < 0:
                    raise ValidationFailed(f"rule {i}: n must be >= 0, got {n}")

    def apply(
        self,
        records: List[Dict[str, Any]],
        now: float,
        audit: Optional[Any] = None,
    ) -> List[str]:
        """Return list of record IDs to delete.

        A record is deletable only if ALL rules mark it deletable.
        Records are not mutated; one audit entry per deleted ID if audit given.
        Audit resource is the record id (not empty).

        Args:
            records: List of dicts with "id" (unique, non-empty str) and "created_ts" (real)
            now: Current time (real finite)
            audit: Optional audit store

        Returns:
            List of record IDs to delete (sorted)

        Raises:
            ValidationFailed: Invalid records or now
        """
        # Validate now
        if not _is_finite_real(now):
            raise ValidationFailed(f"now must be real finite number, got {type(now).__name__}")

        # Validate records
        if not isinstance(records, list):
            raise ValidationFailed("records must be a list")

        seen_ids = set()
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValidationFailed(f"record {i} must be a dict, got {type(rec).__name__}")

            if "id" not in rec:
                raise ValidationFailed(f"record {i} missing 'id' field")
            rec_id = rec["id"]
            if not _is_nonempty_str(rec_id):
                raise ValidationFailed(f"record {i}: id must be non-empty str, got {type(rec_id).__name__}")
            if rec_id in seen_ids:
                raise ValidationFailed(f"record {i}: duplicate id '{rec_id}'")
            seen_ids.add(rec_id)

            if "created_ts" not in rec:
                raise ValidationFailed(f"record {i} missing 'created_ts' field")
            created_ts = rec["created_ts"]
            if not _is_finite_real(created_ts):
                raise ValidationFailed(f"record {i}: created_ts must be real finite number, got {type(created_ts).__name__}")

        if not records:
            return []

        # Collect which records each rule marks as deletable
        deletable_by_rule = {}
        for rule in self.rules:
            kind = rule.get("kind")
            if kind == "max_age":
                max_age_s = rule.get("max_age_s", 0)
                deletable_by_rule[id(rule)] = set(
                    r["id"] for r in records
                    if (now - r["created_ts"]) > max_age_s
                )
            elif kind == "keep_last":
                n = rule.get("n", 0)
                # Keep the n most recent (highest created_ts)
                sorted_recs = sorted(records, key=lambda r: r["created_ts"], reverse=True)
                keep_ids = set(r["id"] for r in sorted_recs[:n])
                deletable_by_rule[id(rule)] = set(r["id"] for r in records if r["id"] not in keep_ids)

        # Record is deletable only if ALL rules mark it deletable
        if not deletable_by_rule:
            return []

        # Intersection of all sets
        result = set(deletable_by_rule[list(deletable_by_rule.keys())[0]])
        for rule_set in list(deletable_by_rule.values())[1:]:
            result = result.intersection(rule_set)

        result_list = sorted(list(result))

        # Audit each deletion with record id as resource
        if audit is not None:
            for rec_id in result_list:
                audit.append(
                    "system", "retention.delete", rec_id, "allow",
                    meta={"deleted_id": rec_id}
                )

        return result_list
