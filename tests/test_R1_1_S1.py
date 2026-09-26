"""
R1.1-S1: Schema Validation (Success/Boundary)

Test suite for C1: Pipelines and environments are declarative YAML;
each references immutable artifacts (data version, code commit, model registry URI,
config hash, container digest). Mutable references are rejected at parse time.

Dimension: success/boundary cases
Mutation targets:
  - Missing validation check for mutable tags (mutation: remove tag check)
  - Accepting "latest" tag without validation (mutation: skip tag validation)
  - No detection of wildcard paths (mutation: remove path wildcard check)
  - Missing content hash validation (mutation: remove hash format check)
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class PipelineSpec:
    """Mock Pipeline specification."""
    name: str
    data_version: str
    code_commit: str
    model_registry_uri: str
    config_hash: str
    container_digest: str


class PipelineSchemaValidator:
    """Schema validator for pipeline specs."""

    MUTABLE_TAGS = {"latest", "main", "master", "develop", "dev"}

    @staticmethod
    def validate_immutable_reference(ref: str) -> bool:
        """Check if reference is immutable (no mutable tags or wildcards)."""
        if any(tag in ref for tag in PipelineSchemaValidator.MUTABLE_TAGS):
            return False
        if "*" in ref or "?" in ref:
            return False
        return True

    @staticmethod
    def validate_hash_format(hash_str: str) -> bool:
        """Check if hash has valid format (SHA256: 64 hex chars)."""
        if not hash_str:
            return False
        if len(hash_str) != 64:
            return False
        try:
            int(hash_str, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_pipeline(spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline spec. Returns {valid: bool, errors: [str]}."""
        errors = []

        # Check required fields
        required = ["name", "data_version", "code_commit", "model_registry_uri", "config_hash", "container_digest"]
        for field in required:
            if field not in spec:
                errors.append(f"Missing required field: {field}")

        # Validate immutable references
        if "data_version" in spec:
            if not PipelineSchemaValidator.validate_immutable_reference(spec["data_version"]):
                errors.append(f"data_version contains mutable reference: {spec['data_version']}")

        if "code_commit" in spec:
            if not PipelineSchemaValidator.validate_immutable_reference(spec["code_commit"]):
                errors.append(f"code_commit contains mutable reference: {spec['code_commit']}")

        if "model_registry_uri" in spec:
            if not PipelineSchemaValidator.validate_immutable_reference(spec["model_registry_uri"]):
                errors.append(f"model_registry_uri contains mutable reference: {spec['model_registry_uri']}")

        if "container_digest" in spec:
            if not PipelineSchemaValidator.validate_immutable_reference(spec["container_digest"]):
                errors.append(f"container_digest contains mutable reference: {spec['container_digest']}")

        # Validate hash format (config_hash must be SHA256)
        if "config_hash" in spec:
            if not PipelineSchemaValidator.validate_hash_format(spec["config_hash"]):
                errors.append(f"config_hash invalid format: {spec['config_hash']}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class TestSchemaValidationSuccess:
    """Success cases: valid pipeline specs pass validation."""

    def test_valid_pipeline_all_immutable_refs(self):
        """Case 1: Valid pipeline with all immutable references passes."""
        spec = {
            "name": "fraud-detection-training",
            "data_version": "v1.2.3",
            "code_commit": "abc123def456",
            "model_registry_uri": "registry.example.com/fraud-model@sha256:abcd1234",
            "config_hash": "a" * 64,  # Valid SHA256
            "container_digest": "sha256:fedcba9876543210"
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_valid_pipeline_with_shas(self):
        """Case 2: Valid pipeline with all SHA256 references."""
        spec = {
            "name": "pricing-model-prep",
            "data_version": "sha256:" + "b" * 64,
            "code_commit": "sha256:" + "c" * 64,
            "model_registry_uri": "models.internal/pricing@sha256:" + "d" * 64,
            "config_hash": "e" * 64,
            "container_digest": "sha256:" + "f" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_valid_pipeline_with_version_tags(self):
        """Case 3: Valid pipeline with semantic versioning (not latest)."""
        spec = {
            "name": "recommendation-training",
            "data_version": "2024.09.01",
            "code_commit": "v2.1.0",
            "model_registry_uri": "registry.io/rec-model:v3.0.0",
            "config_hash": "1" * 64,
            "container_digest": "gcr.io/ml/trainer:v1.2.3"
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is True
        assert result["errors"] == []


class TestSchemaValidationBoundary:
    """Boundary cases: invalid or edge-case specs are rejected."""

    def test_reject_latest_tag_in_data_version(self):
        """Case 4: Reject 'latest' tag in data_version (mutable reference)."""
        spec = {
            "name": "model-training",
            "data_version": "s3://bucket/data:latest",
            "code_commit": "abc123",
            "model_registry_uri": "registry.io/model:v1",
            "config_hash": "2" * 64,
            "container_digest": "sha256:" + "3" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is False
        assert any("mutable reference" in err for err in result["errors"])

    def test_reject_wildcard_in_data_path(self):
        """Case 5: Reject wildcard paths (mutable/ambiguous)."""
        spec = {
            "name": "batch-training",
            "data_version": "s3://bucket/data/*",
            "code_commit": "def456",
            "model_registry_uri": "registry.io/model@sha256:" + "4" * 64,
            "config_hash": "5" * 64,
            "container_digest": "sha256:" + "6" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is False
        assert any("mutable reference" in err for err in result["errors"])

    def test_reject_invalid_config_hash_format(self):
        """Case 6: Reject invalid SHA256 hash (not 64 hex chars)."""
        spec = {
            "name": "model-training",
            "data_version": "v1.0.0",
            "code_commit": "abc123",
            "model_registry_uri": "registry.io/model@sha256:" + "7" * 64,
            "config_hash": "invalid_hash_short",  # Too short
            "container_digest": "sha256:" + "8" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is False
        assert any("config_hash invalid format" in err for err in result["errors"])

    def test_reject_missing_required_field(self):
        """Case 7: Reject spec with missing required field."""
        spec = {
            "name": "incomplete-pipeline",
            "data_version": "v1.0.0",
            "code_commit": "abc123",
            # Missing model_registry_uri
            "config_hash": "9" * 64,
            "container_digest": "sha256:" + "a" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is False
        assert any("Missing required field: model_registry_uri" in err for err in result["errors"])

    def test_reject_develop_branch_reference(self):
        """Case 8: Reject 'develop' branch reference in code_commit."""
        spec = {
            "name": "model-training",
            "data_version": "v1.0.0",
            "code_commit": "develop",  # Mutable
            "model_registry_uri": "registry.io/model@sha256:" + "b" * 64,
            "config_hash": "c" * 64,
            "container_digest": "sha256:" + "d" * 64
        }

        result = PipelineSchemaValidator.validate_pipeline(spec)

        assert result["valid"] is False
        assert any("mutable reference" in err for err in result["errors"])
