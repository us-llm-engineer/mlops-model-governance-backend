"""
S2.3 Dataset Versions: Content-addressed dataset registration and verification.

This module provides immutable dataset version tracking with:

* DatasetRegistry: Register, query, and verify dataset versions.
* Lineage integration: Adds "dataset_version" nodes to the lineage graph.
* Idempotent registration: Same content produces the same version ID.

Stdlib only: dataclasses, typing, copy, string.
"""

import copy
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .kernel import NotFound, ValidationFailed, canonical_json, content_id
from .supply_chain import _is_hex64, _is_nonempty_str

__all__ = ["DatasetRegistry"]


@dataclass(frozen=True)
class DatasetVersionRecord:
    """Immutable dataset version record."""
    name: str
    content_hash: str
    rows: int
    schema: Dict[str, Any]
    version_id: str


class DatasetRegistry:
    """Content-addressed dataset version registry with lineage integration."""

    def __init__(self, audit: Any = None, lineage: Any = None):
        """Initialize the registry.

        Args:
            audit: Optional audit store for logging registrations
            lineage: Optional LineageGraph for adding nodes
        """
        self.audit = audit
        self.lineage = lineage
        self._versions: Dict[str, DatasetVersionRecord] = {}  # version_id -> record
        self._by_name: Dict[str, str] = {}  # name -> latest version_id

    def register(
        self,
        actor: str,
        name: str,
        content_hash: str,
        rows: int,
        schema: Dict[str, Any],
    ) -> str:
        """Register a dataset version.

        Args:
            actor: Principal performing the registration (non-empty str)
            name: Dataset name
            content_hash: SHA256 hex digest of content (validated: 64 lowercase hex)
            rows: Number of rows (validated: int >= 0, not bool)
            schema: Schema dict

        Returns:
            version_id: content-addressed ID

        Raises:
            ValidationFailed: Invalid inputs

        Idempotent: same content produces same version_id.
        """
        # Validate actor FIRST (before any state changes)
        if not _is_nonempty_str(actor):
            raise ValidationFailed(f"actor must be non-empty str, got {type(actor).__name__}")

        # Validate name: non-empty string with no control chars
        if not _is_nonempty_str(name):
            raise ValidationFailed(f"name must be non-empty str, got {type(name).__name__}")
        if any(ord(c) < 32 or ord(c) >= 127 for c in name):
            raise ValidationFailed(f"name contains control characters: {name}")

        # Validate content_hash
        if not _is_hex64(content_hash):
            raise ValidationFailed(f"Invalid content_hash: {content_hash}")

        # Validate rows: must be real int, not bool, not float, not string
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValidationFailed(f"rows must be int, got {type(rows).__name__}")
        if rows < 0:
            raise ValidationFailed(f"rows must be >= 0, got {rows}")

        # Validate schema: must be dict of JSON-safe values
        if not isinstance(schema, dict):
            raise ValidationFailed(f"schema must be dict, got {type(schema).__name__}")
        try:
            canonical_json(schema)
        except Exception as e:
            raise ValidationFailed(f"schema not JSON-safe: {e}")

        # Compute idempotent version_id
        version_id = content_id("ds", {
            "name": name,
            "content_hash": content_hash,
            "rows": rows,
            "schema": schema,
        })

        # If this version already exists, return it (idempotent)
        if version_id in self._versions:
            return version_id

        # Create and store record (deep-copy schema to prevent aliasing)
        record = DatasetVersionRecord(
            name=name,
            content_hash=content_hash,
            rows=rows,
            schema=copy.deepcopy(schema),
            version_id=version_id,
        )
        self._versions[version_id] = record
        self._by_name[name] = version_id

        # Add lineage node if graph is provided
        if self.lineage is not None:
            self.lineage.add_node(
                "dataset_version",
                {
                    "name": name,
                    "content_hash": content_hash,
                    "rows": str(rows),
                },
                version_id,  # Use version_id as timestamp (deterministic)
            )

        # Audit the registration
        if self.audit is not None:
            self.audit.append(
                actor,
                "dataset.register",
                name,
                "allow",
                meta={"version_id": version_id},
            )

        return version_id

    def get(self, version_id: str) -> Dict[str, Any]:
        """Retrieve a dataset version record by ID.

        Args:
            version_id: The version ID

        Returns:
            dict with name, content_hash, rows, schema, version_id (schema is deep-copied)

        Raises:
            NotFound: Version not found
        """
        if version_id not in self._versions:
            raise NotFound(f"Dataset version not found: {version_id}")

        rec = self._versions[version_id]
        return {
            "name": rec.name,
            "content_hash": rec.content_hash,
            "rows": rec.rows,
            "schema": copy.deepcopy(rec.schema),
            "version_id": rec.version_id,
        }

    def latest(self, name: str) -> str:
        """Get the latest version ID for a dataset by name.

        Args:
            name: Dataset name

        Returns:
            version_id: Latest version ID

        Raises:
            NotFound: No version registered for this name
        """
        if name not in self._by_name:
            raise NotFound(f"No dataset versions found for: {name}")
        return self._by_name[name]

    def verify(self, version_id: str, actual_hash: str) -> bool:
        """Verify a dataset version's content hash.

        Args:
            version_id: The version ID to verify
            actual_hash: The actual content hash to compare

        Returns:
            True if hash matches, False otherwise
        """
        if version_id not in self._versions:
            return False

        rec = self._versions[version_id]
        return rec.content_hash == actual_hash
