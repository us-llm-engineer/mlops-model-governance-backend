"""F2.4 cyclonedx_sbom: deterministic SBOM building and validation.

Provides build_cyclonedx_sbom, validate_cyclonedx_sbom, and sbom_matches_requirements
with input validation mirroring supply_chain.sbom_from_requirements, explicit bom_ref
pinning for determinism, and schema validation with error count only.
"""

import json
import re
import string
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from mlops.kernel import ValidationFailed

# Import cyclonedx components
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot5
from cyclonedx.validation.json import JsonStrictValidator
from cyclonedx.schema import SchemaVersion

__all__ = [
    "build_cyclonedx_sbom",
    "validate_cyclonedx_sbom",
    "sbom_matches_requirements",
]


def _is_nonempty_str(val: Any) -> bool:
    """Check if val is a non-empty string."""
    return isinstance(val, str) and len(val) > 0


def _has_no_control_chars(val: str) -> bool:
    """Check if val has no control characters (matching supply_chain logic)."""
    # Allow space but not other whitespace, and no control chars (ord < 32 except space)
    return all(c not in string.whitespace or c == ' ' for c in val) and all(ord(c) >= 32 or c == ' ' for c in val)


def _validate_components(components: List[Union[tuple, Dict[str, str]]]) -> List[Dict[str, str]]:
    """Validate and normalize components to dict format.

    Mirrors the validation logic from supply_chain.sbom_from_requirements.
    Raises ValidationFailed on any validation error.
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
                raise ValidationFailed("dict component missing 'name' or 'version'")
            name = comp["name"]
            version = comp["version"]
        else:
            raise ValidationFailed(f"Invalid component format: {type(comp).__name__}")

        # Validate name and version are non-empty strings
        if not _is_nonempty_str(name):
            raise ValidationFailed("component name must be a non-empty str")
        if not _is_nonempty_str(version):
            raise ValidationFailed("component version must be a non-empty str")

        # Validate name has no control chars, @, /, whitespace
        if not _has_no_control_chars(name) or any(c in "@/ \t\n\r" for c in name):
            raise ValidationFailed("component name contains invalid chars")

        # Validate version has no control chars or @ (ambiguous in purl)
        if not _has_no_control_chars(version) or "@" in version:
            raise ValidationFailed("component version contains invalid chars")

        normalized.append({"name": name, "version": version})

    # Check for duplicates by name (case-insensitive, matching supply_chain logic)
    names = [c["name"] for c in normalized]
    names_lower = [n.lower() for n in names]
    if len(names_lower) != len(set(names_lower)):
        raise ValidationFailed("Duplicate component names in SBOM (case-insensitive)")

    return normalized


def _validate_serial_number(serial_number: Optional[str]) -> Optional[UUID]:
    """Validate serial_number format if provided and convert to UUID object.

    Must be a string matching 'urn:uuid:...' format.
    Returns UUID object if valid, None if not provided.
    """
    if serial_number is None:
        return None

    if not isinstance(serial_number, str):
        raise ValidationFailed(f"serial_number must be str, got {type(serial_number).__name__}")

    # Validate urn:uuid: format with UUID regex
    uuid_pattern = r'^urn:uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    if not re.fullmatch(uuid_pattern, serial_number):
        raise ValidationFailed(f"serial_number must be 'urn:uuid:...' format")

    # Extract UUID part and convert to UUID object
    uuid_str = serial_number[len("urn:uuid:"):]
    return UUID(uuid_str)


def build_cyclonedx_sbom(
    components: List[Union[tuple, Dict[str, str]]],
    *,
    serial_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a CycloneDX 1.5 SBOM dict from components list.

    Validates input (empty list, types, format, duplicates) BEFORE touching
    the cyclonedx library. Always sets explicit bom_ref to ensure determinism.
    Deletes metadata.timestamp to remove nondeterministic wall-clock values.
    Omits serialNumber unless explicitly provided.

    Args:
        components: List of (name, version) tuples or dicts with name/version keys
        serial_number: Optional urn:uuid:... string for SBOM serial number

    Returns:
        CycloneDX 1.5 SBOM dict

    Raises:
        ValidationFailed: Empty list, invalid format, duplicates, bad types
    """
    # Validate serial_number format first (before any library calls)
    serial_number_uuid = _validate_serial_number(serial_number)

    # Validate and normalize components (before any library calls)
    normalized = _validate_components(components)

    # Now safe to use cyclonedx library
    # Sort by name for determinism
    normalized.sort(key=lambda c: c["name"])

    # Build components with explicit bom_ref (library default is RANDOM)
    bom = Bom()
    for comp in normalized:
        name = comp["name"]
        version = comp["version"]

        # TRAP: Always pass bom_ref explicitly, or it becomes a random UUID per process
        component = Component(
            name=name,
            version=version,
            type=ComponentType.LIBRARY,
            bom_ref=f"{name}@{version}",
        )
        bom.components.add(component)

    # Add serial_number if provided
    if serial_number_uuid is not None:
        bom.serial_number = serial_number_uuid

    # Serialize to JSON string with JsonV1Dot5
    json_str = JsonV1Dot5(bom).output_as_string()

    # Parse back to dict
    sbom_dict = json.loads(json_str)

    # Delete nondeterministic timestamp from metadata
    if "metadata" in sbom_dict and sbom_dict["metadata"] is not None:
        if "timestamp" in sbom_dict["metadata"]:
            del sbom_dict["metadata"]["timestamp"]

    # Omit serialNumber unless explicitly provided
    if serial_number is None and "serialNumber" in sbom_dict:
        del sbom_dict["serialNumber"]

    return sbom_dict


def validate_cyclonedx_sbom(sbom: Dict[str, Any]) -> None:
    """Validate SBOM against CycloneDX 1.5 schema.

    Uses JsonStrictValidator with V1_5 schema. Returns normally if valid.
    On validation errors, raises ValidationFailed with error count only
    (never exposes schema error text which may echo document content).

    Args:
        sbom: CycloneDX SBOM dict

    Raises:
        ValidationFailed: If SBOM does not conform to schema (with error count)
    """
    # Re-serialize dict to JSON string for validation
    json_str = json.dumps(sbom, sort_keys=True)

    # Validate with JsonStrictValidator
    validator = JsonStrictValidator(SchemaVersion.V1_5)
    errors = validator.validate_str(json_str)

    # errors is None if valid, list of ValidationError if invalid
    if errors is not None:
        # Count errors
        error_count = len(errors) if isinstance(errors, list) else 1
        raise ValidationFailed(f"{error_count} schema validation error(s)")


def sbom_matches_requirements(
    sbom: Dict[str, Any],
    components: List[Union[tuple, Dict[str, str]]],
) -> bool:
    """Check if SBOM components exactly match the input requirements.

    Order-independent comparison: extracts name/version pairs from both
    SBOM and requirements, then checks if they are the same set.

    Args:
        sbom: CycloneDX SBOM dict (from build_cyclonedx_sbom)
        components: Original requirements list

    Returns:
        True if SBOM matches requirements, False otherwise
    """
    # Extract components from SBOM
    sbom_components = sbom.get("components", [])
    sbom_pairs = {(c["name"], c["version"]) for c in sbom_components}

    # Normalize requirements to (name, version) pairs
    requirements_pairs = set()
    for comp in components:
        if isinstance(comp, tuple):
            requirements_pairs.add((comp[0], comp[1]))
        elif isinstance(comp, dict):
            requirements_pairs.add((comp["name"], comp["version"]))

    # Check if sets match
    return sbom_pairs == requirements_pairs
