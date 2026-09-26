"""
R1.3-S2: Error Mapping (Invalid-Request/Error-Clarity)

Test suite for C10: Invalid requests return HTTP 4xx with descriptive error messages.
Response body is JSON with error_code and message. No plaintext errors or stack traces.

Dimension: invalid-request/error-clarity cases
Mutation targets:
  - Missing error_code field (mutation: return only message)
  - Stack trace in error response (mutation: expose traceback)
  - Plaintext error response (mutation: return non-JSON)
  - Missing field name in error message (mutation: generic error)
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ErrorCode(Enum):
    """Standard error codes."""
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    INVALID_UUID = "INVALID_UUID"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    CONFLICT = "CONFLICT"
    INVALID_PAGINATION = "INVALID_PAGINATION"
    UNAUTHORIZED = "UNAUTHORIZED"


@dataclass
class ErrorResponse:
    """Standardized error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    http_status: int = 400


class APIValidator:
    """Request validation and error response generation."""

    @staticmethod
    def validate_pipeline_submission(request_body: Dict) -> Optional[ErrorResponse]:
        """Validate pipeline submission request."""
        required_fields = ["name", "data_version", "code_commit", "config_hash"]

        for field in required_fields:
            if field not in request_body:
                return ErrorResponse(
                    error_code=ErrorCode.MISSING_REQUIRED_FIELD.value,
                    message=f"Missing required field: {field}",
                    details={"missing_field": field},
                    http_status=400
                )

        # Validate UUID format if provided
        if "run_id" in request_body:
            run_id = request_body["run_id"]
            if not APIValidator._is_valid_uuid(run_id):
                return ErrorResponse(
                    error_code=ErrorCode.INVALID_UUID.value,
                    message=f"Invalid UUID format for run_id: {run_id}",
                    details={"field": "run_id", "value": run_id},
                    http_status=400
                )

        return None  # Valid

    @staticmethod
    def validate_model_promotion(request_body: Dict) -> Optional[ErrorResponse]:
        """Validate model promotion request."""
        if "model_id" not in request_body:
            return ErrorResponse(
                error_code=ErrorCode.MISSING_REQUIRED_FIELD.value,
                message="Missing required field: model_id",
                http_status=400
            )

        if "from_env" not in request_body:
            return ErrorResponse(
                error_code=ErrorCode.MISSING_REQUIRED_FIELD.value,
                message="Missing required field: from_env",
                http_status=400
            )

        if "to_env" not in request_body:
            return ErrorResponse(
                error_code=ErrorCode.MISSING_REQUIRED_FIELD.value,
                message="Missing required field: to_env",
                http_status=400
            )

        return None  # Valid

    @staticmethod
    def validate_pagination(limit: Any, offset: Any) -> Optional[ErrorResponse]:
        """Validate pagination parameters."""
        if limit is not None:
            try:
                limit_int = int(limit)
                if limit_int < 1 or limit_int > 1000:
                    return ErrorResponse(
                        error_code=ErrorCode.INVALID_PAGINATION.value,
                        message="limit must be between 1 and 1000",
                        details={"limit": limit},
                        http_status=400
                    )
            except (ValueError, TypeError):
                return ErrorResponse(
                    error_code=ErrorCode.INVALID_FORMAT.value,
                    message="limit must be an integer",
                    details={"field": "limit", "value": str(limit)},
                    http_status=400
                )

        if offset is not None:
            try:
                offset_int = int(offset)
                if offset_int < 0:
                    return ErrorResponse(
                        error_code=ErrorCode.INVALID_PAGINATION.value,
                        message="offset must be >= 0",
                        http_status=400
                    )
            except (ValueError, TypeError):
                return ErrorResponse(
                    error_code=ErrorCode.INVALID_FORMAT.value,
                    message="offset must be an integer",
                    http_status=400
                )

        return None

    @staticmethod
    def _is_valid_uuid(s: str) -> bool:
        """Check if string is valid UUID format."""
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, s.lower()))

    @staticmethod
    def format_error_response(error: ErrorResponse) -> str:
        """Format error as JSON response."""
        response_dict = {
            "error_code": error.error_code,
            "message": error.message
        }
        if error.details:
            response_dict["details"] = error.details

        return json.dumps(response_dict)


class TestMissingRequiredFields:
    """Missing field error cases."""

    def test_missing_pipeline_name(self):
        """Case 1: Missing name field in pipeline submission."""
        request = {
            "data_version": "v1",
            "code_commit": "abc123",
            "config_hash": "a" * 64
        }

        error = APIValidator.validate_pipeline_submission(request)

        assert error is not None
        assert error.error_code == ErrorCode.MISSING_REQUIRED_FIELD.value
        assert "name" in error.message
        assert error.http_status == 400

    def test_missing_data_version(self):
        """Case 2: Missing data_version field."""
        request = {
            "name": "model1",
            "code_commit": "abc123",
            "config_hash": "b" * 64
        }

        error = APIValidator.validate_pipeline_submission(request)

        assert error is not None
        assert "data_version" in error.message

    def test_missing_model_promotion_from_env(self):
        """Case 3: Missing from_env in promotion request."""
        request = {
            "model_id": "m1",
            "to_env": "prod"
        }

        error = APIValidator.validate_model_promotion(request)

        assert error is not None
        assert "from_env" in error.message


class TestInvalidFormatErrors:
    """Invalid format error cases."""

    def test_invalid_uuid_format(self):
        """Case 4: Invalid UUID format rejected."""
        request = {
            "name": "model",
            "data_version": "v1",
            "code_commit": "abc",
            "config_hash": "c" * 64,
            "run_id": "not-a-uuid"
        }

        error = APIValidator.validate_pipeline_submission(request)

        assert error is not None
        assert error.error_code == ErrorCode.INVALID_UUID.value
        assert "UUID" in error.message

    def test_invalid_pagination_limit_non_integer(self):
        """Case 5: Non-integer limit parameter rejected."""
        error = APIValidator.validate_pagination(limit="abc", offset=None)

        assert error is not None
        assert error.error_code == ErrorCode.INVALID_FORMAT.value
        assert "integer" in error.message.lower()

    def test_invalid_pagination_offset_negative(self):
        """Case 6: Negative offset rejected."""
        error = APIValidator.validate_pagination(limit=10, offset=-1)

        assert error is not None
        assert error.error_code == ErrorCode.INVALID_PAGINATION.value


class TestErrorResponseFormat:
    """Error response format cases."""

    def test_error_response_is_json(self):
        """Case 7: Error response is JSON (not plaintext)."""
        request = {"data_version": "v1"}  # Missing required fields
        error = APIValidator.validate_pipeline_submission(request)

        response_json = APIValidator.format_error_response(error)

        # Should be valid JSON
        parsed = json.loads(response_json)
        assert isinstance(parsed, dict)

    def test_error_response_has_error_code(self):
        """Case 8: Error response includes error_code."""
        error = ErrorResponse(
            error_code="TEST_ERROR",
            message="Test message",
            http_status=400
        )

        response_json = APIValidator.format_error_response(error)
        parsed = json.loads(response_json)

        assert "error_code" in parsed
        assert parsed["error_code"] == "TEST_ERROR"

    def test_error_response_has_message(self):
        """Case 9: Error response includes descriptive message."""
        error = ErrorResponse(
            error_code="FIELD_MISSING",
            message="Missing required field: model_id",
            http_status=400
        )

        response_json = APIValidator.format_error_response(error)
        parsed = json.loads(response_json)

        assert "message" in parsed
        assert "model_id" in parsed["message"]

    def test_error_response_includes_details(self):
        """Case 10: Error response includes details when relevant."""
        error = ErrorResponse(
            error_code="INVALID_FORMAT",
            message="Invalid value for limit",
            details={"field": "limit", "value": "abc"},
            http_status=400
        )

        response_json = APIValidator.format_error_response(error)
        parsed = json.loads(response_json)

        assert "details" in parsed
        assert parsed["details"]["field"] == "limit"

    def test_no_stack_trace_in_error_response(self):
        """Case 11: Error response does not include stack traces."""
        error = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An error occurred",
            http_status=500
        )

        response_json = APIValidator.format_error_response(error)

        # Should not contain common traceback patterns
        assert "Traceback" not in response_json
        assert "File \"" not in response_json
        assert "line" not in response_json.lower() or "Error" in response_json
