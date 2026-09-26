"""
Claim C6: Artifact Integrity & Content-Addressed Storage.

Each pipeline stage emits an immutable artifact (model file, eval metrics JSON,
packaged binary) stored under a content-addressed tree keyed by SHA256 hash and
filename. Metadata records are linked by hash. Tampering with stored content is
detected on read as a hash mismatch, and re-storing identical content under the
same key is idempotent.

Source test suite: R1.2-S2 (tests/R1_2_S2.py).
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    filename: str
    content_hash: str  # SHA256 of content
    size: int
    artifact_type: str  # model, metrics, binary
    created_at: str
    metadata: Dict[str, Any]


class ArtifactStore:
    """Content-addressed artifact storage."""

    def __init__(self):
        # Simulated storage: (content_hash, filename) -> content
        self.storage: Dict[tuple, bytes] = {}
        self.metadata: Dict[str, ArtifactMetadata] = {}

    def compute_hash(self, content: bytes) -> str:
        """Compute the SHA256 hex digest of content."""
        return hashlib.sha256(content).hexdigest()

    def store_artifact(
        self,
        content: bytes,
        filename: str,
        artifact_type: str,
        created_at: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store an artifact by content hash; return its "<hash>:<filename>" id."""
        if metadata is None:
            metadata = {}

        content_hash = self.compute_hash(content)
        key = (content_hash, filename)

        # IMMUTABLE: if key exists, content must be identical
        if key in self.storage:
            if self.storage[key] != content:
                raise ValueError(
                    f"Artifact with different content exists: {content_hash}/{filename}"
                )
        else:
            # Store new artifact
            self.storage[key] = content

        # Create metadata record
        artifact_metadata = ArtifactMetadata(
            filename=filename,
            content_hash=content_hash,
            size=len(content),
            artifact_type=artifact_type,
            created_at=created_at,
            metadata=metadata
        )

        metadata_id = f"{content_hash}:{filename}"
        self.metadata[metadata_id] = artifact_metadata

        return metadata_id

    def retrieve_artifact(self, artifact_id: str) -> bytes:
        """Retrieve an artifact by id, verifying its hash integrity on read."""
        if artifact_id not in self.metadata:
            raise KeyError(f"Artifact not found: {artifact_id}")

        artifact_meta = self.metadata[artifact_id]
        content_hash = artifact_meta.content_hash
        filename = artifact_meta.filename
        key = (content_hash, filename)

        if key not in self.storage:
            raise KeyError(f"Artifact content not found: {artifact_id}")

        content = self.storage[key]

        # Verify integrity: recompute hash
        actual_hash = self.compute_hash(content)
        if actual_hash != content_hash:
            raise ValueError(
                f"Hash mismatch for {artifact_id}: "
                f"expected {content_hash}, got {actual_hash}"
            )

        return content

    def get_artifact_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Return the metadata record for an artifact; raise KeyError if unknown."""
        if artifact_id not in self.metadata:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self.metadata[artifact_id]

    def list_artifacts_by_type(self, artifact_type: str) -> list:
        """Return (artifact_id, metadata) pairs for all artifacts of a type."""
        return [
            (artifact_id, meta)
            for artifact_id, meta in self.metadata.items()
            if meta.artifact_type == artifact_type
        ]


class PipelineStageExecutor:
    """Execute pipeline stages and store their output artifacts."""

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def execute_training_stage(
        self,
        run_id: str,
        data_version: str,
        training_config: str
    ) -> Dict[str, str]:
        """
        Execute the training stage; return artifact ids for the model and metrics.

        Outputs:
        - model.pkl: trained model (stub)
        - metrics.json: training metrics
        """
        # Stub training
        model_content = f"model_data_for_{run_id}_{data_version}".encode()
        metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "training_config_hash": hashlib.sha256(training_config.encode()).hexdigest()
        }
        metrics_content = json.dumps(metrics).encode()

        model_id = self.artifact_store.store_artifact(
            model_content,
            "model.pkl",
            "model",
            "2024-09-24T00:00:00Z",
            {"run_id": run_id, "data_version": data_version}
        )

        metrics_id = self.artifact_store.store_artifact(
            metrics_content,
            "metrics.json",
            "metrics",
            "2024-09-24T00:00:01Z",
            {"run_id": run_id}
        )

        return {
            "model": model_id,
            "metrics": metrics_id,
        }

    def execute_evaluation_stage(
        self,
        run_id: str,
        model_id: str
    ) -> str:
        """Execute the evaluation stage; return the evaluation artifact id."""
        model_content = self.artifact_store.retrieve_artifact(model_id)

        eval_result = {
            "model_id": model_id,
            "test_accuracy": 0.93,
            "precision": 0.94,
            "recall": 0.92,
        }
        eval_content = json.dumps(eval_result).encode()

        eval_id = self.artifact_store.store_artifact(
            eval_content,
            "eval_result.json",
            "evaluation",
            "2024-09-24T00:00:02Z",
            {"run_id": run_id, "model_id": model_id}
        )

        return eval_id
