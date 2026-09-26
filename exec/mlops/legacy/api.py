"""
Platform API contract and error-handling surface (Claim C9 + C10).

Maps to requirement R1.3 (Platform API & CLI) and is lifted from the frozen
reference implementations in:

- ``tests/R1_3_S1.py`` (C9: latency & pagination for metadata endpoints)
- ``tests/R1_3_S2.py`` (C10: structured JSON error handling & validation)

Standard library only. Public names, signatures, defaults, return types,
exception types and error-message substrings mirror the source suites exactly.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ApiResponse:
    """HTTP response stub carrying a body, headers and measured latency."""

    status_code: int
    body: str
    headers: Dict[str, str]
    latency_ms: float


class InMemoryDataStore:
    """In-memory seed data for the platform metadata endpoints."""

    def __init__(self) -> None:
        self.pipelines: List[Dict[str, Any]] = [
            {"id": f"pipe_{i:04d}", "name": f"pipeline_{i}", "status": "running"}
            for i in range(1000)
        ]
        self.models: List[Dict[str, Any]] = [
            {"id": f"model_{i:04d}", "name": f"model_{i}", "accuracy": 0.9 + (i % 10) * 0.001}
            for i in range(500)
        ]
        self.environments: List[Dict[str, Any]] = [
            {"id": f"env_{i:03d}", "name": f"env_{i}", "type": "staging" if i % 2 else "prod"}
            for i in range(50)
        ]
        self.audit_events: List[Dict[str, Any]] = [
            {
                "id": f"event_{i:05d}",
                "actor": f"user_{i % 10}",
                "action": "deploy",
                "timestamp": f"2024-01-{i % 28 + 1:02d}",
            }
            for i in range(5000)
        ]


class ApiServer:
    """Stub API server with pagination, JSON responses and latency tracking."""

    def __init__(self, data_store: InMemoryDataStore) -> None:
        self.data_store = data_store
        self.latencies: List[float] = []

    @staticmethod
    def _clamp(limit: int, offset: int) -> Tuple[int, int]:
        """Clamp pagination params: invalid limit -> 10, negative offset -> 0."""
        if limit <= 0 or limit > 1000:
            limit = 10
        if offset < 0:
            offset = 0
        return limit, offset

    def _paginate(
        self, items_source: List[Dict[str, Any]], limit: int, offset: int
    ) -> ApiResponse:
        """Paginate ``items_source`` and wrap it in a JSON ``ApiResponse``."""
        start = time.time()

        limit, offset = self._clamp(limit, offset)
        items = items_source[offset:offset + limit]
        total = len(items_source)

        response_body = {
            "items": items,
            "limit": limit,
            "offset": offset,
            "total": total,
        }

        latency_ms = (time.time() - start) * 1000
        self.latencies.append(latency_ms)

        return ApiResponse(
            status_code=200,
            body=json.dumps(response_body),
            headers={"Content-Type": "application/json"},
            latency_ms=latency_ms,
        )

    def list_pipelines(self, limit: int = 10, offset: int = 0) -> ApiResponse:
        """List pipelines with pagination."""
        return self._paginate(self.data_store.pipelines, limit, offset)

    def list_models(self, limit: int = 10, offset: int = 0) -> ApiResponse:
        """List models with pagination."""
        return self._paginate(self.data_store.models, limit, offset)

    def list_environments(self, limit: int = 10, offset: int = 0) -> ApiResponse:
        """List environments with pagination."""
        return self._paginate(self.data_store.environments, limit, offset)

    def list_audit(self, limit: int = 10, offset: int = 0) -> ApiResponse:
        """List audit events with pagination."""
        return self._paginate(self.data_store.audit_events, limit, offset)

    def get_p95_latency(self) -> float:
        """Calculate p95 latency from recorded latencies."""
        if not self.latencies:
            return 0

        sorted_latencies = sorted(self.latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[p95_idx]


class ApiValidator:
    """Validate request inputs and report missing fields."""

    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Return True iff ``value`` is a parseable UUID."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_required_fields(data: Dict, required: list) -> tuple:
        """Return ``(is_valid, missing_fields)`` for ``required`` against ``data``."""
        missing = [field for field in required if field not in data]
        return len(missing) == 0, missing


class ErrorResponse:
    """Structured, JSON-serializable error response (never a stack trace)."""

    def __init__(
        self, status_code: int, error_code: str, message: str, details: Dict = None
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_json(self) -> str:
        """Serialize to a JSON body containing error_code, message and details."""
        body: Dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }

        if self.details:
            body["details"] = self.details

        return json.dumps(body)


def _error_dict(error: ErrorResponse) -> Dict[str, Any]:
    """Wrap an ``ErrorResponse`` in the dict shape used by the stub APIs."""
    return {
        "status": error.status_code,
        "body": error.to_json(),
        "headers": {"Content-Type": "application/json"},
    }


class PipelineApi:
    """Stub pipeline API with UUID and required-field validation."""

    def submit_pipeline(self, pipeline_id: str, spec: Dict) -> Dict:
        """Validate and submit a pipeline spec, returning a response dict."""
        if not ApiValidator.validate_uuid(pipeline_id):
            return _error_dict(
                ErrorResponse(
                    status_code=400,
                    error_code="INVALID_PIPELINE_ID",
                    message="Pipeline ID must be a valid UUID",
                    details={"provided": pipeline_id},
                )
            )

        required_fields = ["name", "version", "config"]
        is_valid, missing = ApiValidator.validate_required_fields(spec, required_fields)

        if not is_valid:
            return _error_dict(
                ErrorResponse(
                    status_code=400,
                    error_code="MISSING_REQUIRED_FIELDS",
                    message=f"Missing required fields: {', '.join(missing)}",
                    details={"missing_fields": missing, "required": required_fields},
                )
            )

        return {
            "status": 201,
            "body": json.dumps({"id": pipeline_id, "status": "created"}),
            "headers": {"Content-Type": "application/json"},
        }


class ModelApi:
    """Stub model API with UUID validation and structured errors."""

    def get_model(self, model_id: str) -> Dict:
        """Fetch a model by ID, returning 400 on bad format and 404 otherwise."""
        if not ApiValidator.validate_uuid(model_id):
            return _error_dict(
                ErrorResponse(
                    status_code=400,
                    error_code="INVALID_MODEL_ID_FORMAT",
                    message=f"Model ID must be UUID format, got: {model_id}",
                    details={"provided": model_id},
                )
            )

        return _error_dict(
            ErrorResponse(
                status_code=404,
                error_code="MODEL_NOT_FOUND",
                message=f"Model with ID {model_id} not found",
                details={"model_id": model_id},
            )
        )

    def promote_model(self, model_id: str, promotion_spec: Dict) -> Dict:
        """Validate a model ID and promotion spec, then report a promotion."""
        if not ApiValidator.validate_uuid(model_id):
            return _error_dict(
                ErrorResponse(
                    status_code=400,
                    error_code="INVALID_MODEL_ID_FORMAT",
                    message="Model ID must be UUID format",
                    details={"provided": model_id},
                )
            )

        required = ["from_env", "to_env"]
        is_valid, missing = ApiValidator.validate_required_fields(promotion_spec, required)

        if not is_valid:
            return _error_dict(
                ErrorResponse(
                    status_code=400,
                    error_code="INVALID_PROMOTION_SPEC",
                    message=f"Promotion spec missing: {', '.join(missing)}",
                    details={"missing": missing},
                )
            )

        return {
            "status": 200,
            "body": json.dumps({"id": model_id, "promoted": True}),
            "headers": {"Content-Type": "application/json"},
        }
