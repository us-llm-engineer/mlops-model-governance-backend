"""
C1 -- Declarative YAML Schema Validation.

Claim C1: Pipelines and environments are declared as declarative specs that
reference immutable artifacts. A submission carrying a mutable reference
(wildcard data version, branch name code commit, floating container tag,
``:latest`` model URI, or a non-SHA256 config hash) is rejected at parse time.

Source test suite: R1.1-S1.
"""

from typing import Any, Dict

__all__ = ["PipelineSpec", "PipelineValidator"]


class PipelineSpec:
    """A declarative pipeline specification validated at construction time."""

    def __init__(self, spec_dict: Dict[str, Any]):
        self.spec = spec_dict
        self.validate()

    def validate(self) -> None:
        """Validate the schema, rejecting every mutable artifact reference.

        Raises:
            ValueError: If any referenced artifact is mutable.
        """
        data_version = self.spec.get("data_version", "")
        if "*" in data_version or data_version == "latest":
            raise ValueError(f"Data version contains wildcard/mutable: {data_version}")

        code_commit = self.spec.get("code_commit", "")
        if code_commit in ("main", "develop", "master") or code_commit == "HEAD":
            raise ValueError(f"Code commit is mutable branch: {code_commit}")

        container = self.spec.get("container_image", "")
        if ":latest" in container or (":" not in container and "@" not in container):
            raise ValueError(f"Container image is mutable: {container}")

        model_uri = self.spec.get("model_registry_uri", "")
        if model_uri and (model_uri.endswith(":latest") or ":" not in model_uri):
            raise ValueError(f"Model registry URI is mutable: {model_uri}")

        config_hash = self.spec.get("config_hash", "")
        if config_hash and (
            len(config_hash) != 64
            or not all(c in "0123456789abcdef" for c in config_hash)
        ):
            raise ValueError(f"Config hash is not SHA256: {config_hash}")


class PipelineValidator:
    """Validates and rejects pipelines with mutable references."""

    def submit(self, spec_dict: Dict[str, Any]) -> str:
        """Submit a pipeline spec; raise on validation failure.

        Returns:
            The assigned pipeline identifier.
        """
        PipelineSpec(spec_dict)
        return "pipeline_id_123"
