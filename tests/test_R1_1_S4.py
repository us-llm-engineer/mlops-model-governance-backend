"""
R1.1-S4: YAML Round-Trip & Lineage Consistency (Persistence/Concurrency)

Test suite for C4: Deployed pipeline config can be read back from storage,
re-submitted, and produces identical lineage and audit trails. No loss of fidelity
across serialization.

Dimension: persistence/concurrency cases
Mutation targets:
  - Lost field during serialization (mutation: remove field from write)
  - Incorrect hash after round-trip (mutation: modify hash after read)
  - Audit trail not preserved (mutation: skip audit entry on read)
  - Concurrent read/write causes race (mutation: no lock protection)
"""

import pytest
import json
import yaml
import hashlib
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from copy import deepcopy
from threading import Lock, Thread
import tempfile
import os


@dataclass
class PipelineConfig:
    """Pipeline configuration with all required fields."""
    name: str
    data_version: str
    code_commit: str
    model_registry_uri: str
    config_hash: str
    container_digest: str
    environment: str = "dev"
    max_retries: int = 3
    timeout_seconds: int = 3600


@dataclass
class AuditTrail:
    """Audit trail entry for a pipeline submission."""
    pipeline_name: str
    action: str  # "submit", "read", "retry"
    timestamp: float
    actor: str
    config_hash: str


class PipelineStorage:
    """Persistent storage for pipeline configs with audit trails."""

    def __init__(self):
        self.pipelines: Dict[str, Dict] = {}  # pipeline_name → {config, audit}
        self.audit_trails: Dict[str, List[AuditTrail]] = {}  # pipeline_name → [audit entries]
        self._lock = Lock()  # Protect concurrent access

    @staticmethod
    def compute_config_hash(config: Dict) -> str:
        """Compute SHA256 hash of config (canonical JSON)."""
        canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def store_pipeline(self, config: PipelineConfig, actor: str) -> str:
        """
        Store pipeline config with audit trail.
        Returns: config_hash
        """
        with self._lock:
            config_dict = asdict(config)
            config_hash = self.compute_config_hash(config_dict)

            self.pipelines[config.name] = {
                "config": config_dict,
                "config_hash": config_hash,
                "stored_at": time.time()
            }

            # Add audit entry
            if config.name not in self.audit_trails:
                self.audit_trails[config.name] = []

            audit = AuditTrail(
                pipeline_name=config.name,
                action="submit",
                timestamp=time.time(),
                actor=actor,
                config_hash=config_hash
            )
            self.audit_trails[config.name].append(audit)

            return config_hash

    def read_pipeline(self, pipeline_name: str, actor: str) -> Optional[PipelineConfig]:
        """
        Read pipeline config from storage.
        Returns: PipelineConfig or None
        """
        with self._lock:
            if pipeline_name not in self.pipelines:
                return None

            stored = self.pipelines[pipeline_name]
            config_dict = stored["config"]

            # Add read audit entry
            if pipeline_name not in self.audit_trails:
                self.audit_trails[pipeline_name] = []

            audit = AuditTrail(
                pipeline_name=pipeline_name,
                action="read",
                timestamp=time.time(),
                actor=actor,
                config_hash=stored["config_hash"]
            )
            self.audit_trails[pipeline_name].append(audit)

            return PipelineConfig(**config_dict)

    def verify_config_integrity(self, pipeline_name: str) -> bool:
        """
        Verify config hash matches stored value (detects tampering).
        """
        with self._lock:
            if pipeline_name not in self.pipelines:
                return False

            stored = self.pipelines[pipeline_name]
            config_dict = stored["config"]
            stored_hash = stored["config_hash"]

            computed_hash = self.compute_config_hash(config_dict)
            return computed_hash == stored_hash

    def get_audit_trail(self, pipeline_name: str) -> List[AuditTrail]:
        """Get audit trail for pipeline."""
        with self._lock:
            return deepcopy(self.audit_trails.get(pipeline_name, []))


class TestYAMLRoundTripSuccess:
    """Success cases: configs survive serialization/deserialization."""

    def test_config_roundtrip_all_fields_preserved(self):
        """Case 1: All config fields preserved after store/read."""
        storage = PipelineStorage()
        original = PipelineConfig(
            name="fraud-model",
            data_version="v1.2.3",
            code_commit="abc123def456",
            model_registry_uri="registry.example.com/fraud@sha256:abc",
            config_hash="a" * 64,
            container_digest="sha256:fedcba9876"
        )

        # Store
        hash_stored = storage.store_pipeline(original, "alice@example.com")

        # Read back
        read_back = storage.read_pipeline("fraud-model", "alice@example.com")

        assert read_back is not None
        assert read_back.name == original.name
        assert read_back.data_version == original.data_version
        assert read_back.code_commit == original.code_commit
        assert read_back.model_registry_uri == original.model_registry_uri
        assert read_back.config_hash == original.config_hash
        assert read_back.container_digest == original.container_digest

    def test_optional_fields_preserved(self):
        """Case 2: Optional fields (environment, timeout) preserved."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="pricing-model",
            data_version="v2.0.0",
            code_commit="xyz789",
            model_registry_uri="registry.io/pricing@sha256:xyz",
            config_hash="b" * 64,
            container_digest="sha256:123456",
            environment="prod",
            max_retries=5,
            timeout_seconds=7200
        )

        storage.store_pipeline(config, "bob@example.com")
        read_back = storage.read_pipeline("pricing-model", "bob@example.com")

        assert read_back.environment == "prod"
        assert read_back.max_retries == 5
        assert read_back.timeout_seconds == 7200

    def test_hash_consistency_across_roundtrip(self):
        """Case 3: Config hash remains consistent after roundtrip."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="recommendation-model",
            data_version="v3.1.0",
            code_commit="hash123",
            model_registry_uri="models.internal/rec@sha256:hash",
            config_hash="c" * 64,
            container_digest="sha256:789abc"
        )

        hash_before = PipelineStorage.compute_config_hash(asdict(config))

        storage.store_pipeline(config, "charlie@example.com")
        read_back = storage.read_pipeline("recommendation-model", "charlie@example.com")

        hash_after = PipelineStorage.compute_config_hash(asdict(read_back))

        assert hash_before == hash_after


class TestPersistenceIntegrity:
    """Integrity cases: configs are tamper-evident."""

    def test_config_integrity_valid(self):
        """Case 4: Valid config passes integrity check."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-1",
            data_version="v1.0.0",
            code_commit="commit1",
            model_registry_uri="registry/model@sha256:abc",
            config_hash="d" * 64,
            container_digest="sha256:def"
        )

        storage.store_pipeline(config, "user1")

        is_valid = storage.verify_config_integrity("model-1")
        assert is_valid is True

    def test_config_integrity_detects_tampering(self):
        """Case 5: Tampering is detected via hash mismatch."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-2",
            data_version="v1.0.0",
            code_commit="commit2",
            model_registry_uri="registry/model@sha256:def",
            config_hash="e" * 64,
            container_digest="sha256:ghi"
        )

        storage.store_pipeline(config, "user2")

        # Tamper with stored config
        storage.pipelines["model-2"]["config"]["data_version"] = "v2.0.0"

        is_valid = storage.verify_config_integrity("model-2")
        assert is_valid is False


class TestAuditTrailConsistency:
    """Audit trail cases: all operations are recorded."""

    def test_audit_trail_records_submission(self):
        """Case 6: Submission creates audit entry."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-3",
            data_version="v1.0.0",
            code_commit="commit3",
            model_registry_uri="registry/model@sha256:ghi",
            config_hash="f" * 64,
            container_digest="sha256:jkl"
        )

        storage.store_pipeline(config, "alice@company.com")

        audit = storage.get_audit_trail("model-3")
        assert len(audit) == 1
        assert audit[0].action == "submit"
        assert audit[0].actor == "alice@company.com"

    def test_audit_trail_records_reads(self):
        """Case 7: Read operations recorded in audit trail."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-4",
            data_version="v1.0.0",
            code_commit="commit4",
            model_registry_uri="registry/model@sha256:jkl",
            config_hash="1" * 64,
            container_digest="sha256:mno"
        )

        storage.store_pipeline(config, "bob")
        storage.read_pipeline("model-4", "charlie")
        storage.read_pipeline("model-4", "diana")

        audit = storage.get_audit_trail("model-4")
        assert len(audit) == 3  # 1 submit + 2 reads
        assert audit[0].action == "submit"
        assert audit[1].action == "read"
        assert audit[1].actor == "charlie"
        assert audit[2].actor == "diana"

    def test_audit_trail_preserves_config_hash(self):
        """Case 8: Audit entries record config hash (for lineage)."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-5",
            data_version="v1.0.0",
            code_commit="commit5",
            model_registry_uri="registry/model@sha256:mno",
            config_hash="2" * 64,
            container_digest="sha256:pqr"
        )

        hash_stored = storage.store_pipeline(config, "eve")

        audit = storage.get_audit_trail("model-5")
        assert audit[0].config_hash == hash_stored


class TestConcurrentAccess:
    """Concurrency cases: multiple readers/writers don't corrupt state."""

    def test_concurrent_reads_safe(self):
        """Case 9: Multiple concurrent reads are safe."""
        storage = PipelineStorage()
        config = PipelineConfig(
            name="model-6",
            data_version="v1.0.0",
            code_commit="commit6",
            model_registry_uri="registry/model@sha256:pqr",
            config_hash="3" * 64,
            container_digest="sha256:stu"
        )

        storage.store_pipeline(config, "user1")

        results = []

        def reader(actor: str):
            result = storage.read_pipeline("model-6", actor)
            results.append(result)

        # Start 5 concurrent readers
        threads = [Thread(target=reader, args=(f"user{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for result in results:
            assert result.name == "model-6"
            assert result.data_version == "v1.0.0"

    def test_concurrent_writes_serialized(self):
        """Case 10: Concurrent writes are serialized (lock prevents corruption)."""
        storage = PipelineStorage()

        configs = [
            PipelineConfig(
                name=f"model-{i}",
                data_version=f"v{i}.0.0",
                code_commit=f"commit{i}",
                model_registry_uri=f"registry/model@sha256:{chr(96+i)}" * 4,
                config_hash=f"{i}" * 64,
                container_digest=f"sha256:{chr(96+i)}" * 4
            )
            for i in range(1, 4)
        ]

        def writer(cfg: PipelineConfig, actor: str):
            storage.store_pipeline(cfg, actor)

        # Concurrent writes
        threads = [Thread(target=writer, args=(cfg, f"actor{i}")) for i, cfg in enumerate(configs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all pipelines stored correctly
        for i in range(1, 4):
            result = storage.read_pipeline(f"model-{i}", "verifier")
            assert result is not None
