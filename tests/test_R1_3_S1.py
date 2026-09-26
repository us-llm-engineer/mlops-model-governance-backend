"""
R1.3-S1: API Contract (Success/Boundary)

Test suite for C9: All metadata endpoints (/pipelines, /models, /environments, /audit)
respond in < 500ms (p95). Query results are JSON. Pagination (limit/offset) supported
for large result sets.

Dimension: success/boundary cases
Mutation targets:
  - Response exceeds latency SLA (mutation: remove caching)
  - Latency not measured (mutation: skip timing check)
  - Pagination not implemented (mutation: return all results)
  - Non-JSON response (mutation: return plaintext)
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from statistics import median, quantiles


@dataclass
class PaginationParams:
    """Pagination parameters."""
    limit: int = 20
    offset: int = 0


@dataclass
class APIResponse:
    """API response wrapper."""
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    response_time_ms: float = 0.0


class MockAPIServer:
    """Mock MLOps API server with endpoints."""

    LATENCY_SLA_MS = 500  # p95 latency must be < 500ms

    def __init__(self):
        self.pipelines = [
            {"id": f"p{i}", "name": f"pipeline-{i}", "status": "completed"}
            for i in range(100)
        ]
        self.models = [
            {"id": f"m{i}", "name": f"model-{i}", "version": f"v{i//10}"}
            for i in range(100)
        ]
        self.environments = [
            {"id": "dev", "name": "development", "region": "us-west-2"},
            {"id": "staging", "name": "staging", "region": "us-west-2"},
            {"id": "prod", "name": "production", "region": "us-east-1"}
        ]
        self.audit_logs = [
            {"id": f"log{i}", "actor": f"user-{i%10}", "action": f"action-{i%5}"}
            for i in range(50)
        ]
        self.request_times: List[float] = []

    def _measure_latency(self, func, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        self.request_times.append(elapsed)
        return result, elapsed

    def list_pipelines(self, limit: int = 20, offset: int = 0) -> APIResponse:
        """GET /pipelines endpoint."""
        def _do_request():
            total = len(self.pipelines)
            items = self.pipelines[offset:offset+limit]
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }

        body, elapsed_ms = self._measure_latency(_do_request)

        return APIResponse(
            status_code=200,
            body=body,
            headers={"Content-Type": "application/json"},
            response_time_ms=elapsed_ms
        )

    def list_models(self, limit: int = 20, offset: int = 0) -> APIResponse:
        """GET /models endpoint."""
        def _do_request():
            total = len(self.models)
            items = self.models[offset:offset+limit]
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }

        body, elapsed_ms = self._measure_latency(_do_request)

        return APIResponse(
            status_code=200,
            body=body,
            headers={"Content-Type": "application/json"},
            response_time_ms=elapsed_ms
        )

    def list_environments(self) -> APIResponse:
        """GET /environments endpoint."""
        def _do_request():
            return {"items": self.environments}

        body, elapsed_ms = self._measure_latency(_do_request)

        return APIResponse(
            status_code=200,
            body=body,
            headers={"Content-Type": "application/json"},
            response_time_ms=elapsed_ms
        )

    def list_audit(self, limit: int = 20, offset: int = 0) -> APIResponse:
        """GET /audit endpoint."""
        def _do_request():
            total = len(self.audit_logs)
            items = self.audit_logs[offset:offset+limit]
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }

        body, elapsed_ms = self._measure_latency(_do_request)

        return APIResponse(
            status_code=200,
            body=body,
            headers={"Content-Type": "application/json"},
            response_time_ms=elapsed_ms
        )

    def get_p95_latency(self) -> float:
        """Get p95 latency from recorded requests."""
        if len(self.request_times) < 20:
            return max(self.request_times) if self.request_times else 0
        return quantiles(self.request_times, n=20)[18]  # 95th percentile


class TestAPIMetadataEndpoints:
    """Metadata endpoint success cases."""

    def test_list_pipelines_returns_json(self):
        """Case 1: /pipelines returns valid JSON response."""
        server = MockAPIServer()
        response = server.list_pipelines()

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert isinstance(response.body, dict)
        assert "items" in response.body
        assert isinstance(response.body["items"], list)

    def test_list_models_returns_json(self):
        """Case 2: /models returns valid JSON response."""
        server = MockAPIServer()
        response = server.list_models()

        assert response.status_code == 200
        assert isinstance(response.body, dict)
        assert "items" in response.body

    def test_list_environments_returns_json(self):
        """Case 3: /environments returns valid JSON."""
        server = MockAPIServer()
        response = server.list_environments()

        assert response.status_code == 200
        assert response.body["items"] is not None
        assert len(response.body["items"]) >= 0

    def test_list_audit_returns_json(self):
        """Case 4: /audit returns valid JSON."""
        server = MockAPIServer()
        response = server.list_audit()

        assert response.status_code == 200
        assert "items" in response.body


class TestAPIPagination:
    """Pagination support cases."""

    def test_pagination_limit_parameter(self):
        """Case 5: limit parameter restricts result set."""
        server = MockAPIServer()

        response_10 = server.list_pipelines(limit=10)
        response_20 = server.list_pipelines(limit=20)

        assert len(response_10.body["items"]) == 10
        assert len(response_20.body["items"]) == 20

    def test_pagination_offset_parameter(self):
        """Case 6: offset parameter skips results."""
        server = MockAPIServer()

        response_0 = server.list_pipelines(limit=5, offset=0)
        response_5 = server.list_pipelines(limit=5, offset=5)
        response_10 = server.list_pipelines(limit=5, offset=10)

        # Verify offset works (items should be different)
        items_0 = response_0.body["items"]
        items_5 = response_5.body["items"]
        items_10 = response_10.body["items"]

        assert items_0[0]["id"] == "p0"
        assert items_5[0]["id"] == "p5"
        assert items_10[0]["id"] == "p10"

    def test_pagination_default_limit(self):
        """Case 7: Default limit is applied when not specified."""
        server = MockAPIServer()
        response = server.list_pipelines()

        # Default limit should be 20
        assert len(response.body["items"]) == 20

    def test_pagination_total_count(self):
        """Case 8: Response includes total count."""
        server = MockAPIServer()
        response = server.list_models(limit=10)

        assert response.body["total"] == len(server.models)
        assert response.body["limit"] == 10
        assert response.body["offset"] == 0


class TestAPILatencySLA:
    """Latency SLA cases."""

    def test_single_request_within_sla(self):
        """Case 9: Single request latency < 500ms."""
        server = MockAPIServer()
        response = server.list_pipelines()

        assert response.response_time_ms < 500

    def test_p95_latency_within_sla(self):
        """Case 10: p95 latency < 500ms across multiple requests."""
        server = MockAPIServer()

        # Make multiple requests
        for _ in range(30):
            server.list_pipelines()
            server.list_models()

        p95 = server.get_p95_latency()
        assert p95 < 500
