"""
CLI-to-API integration and exit-code surface (Claim C11).

Maps to requirement R1.3 (Platform API & CLI) and is lifted from the frozen
reference implementation in ``tests/R1_3_S3.py``.

Commands map directly to API calls, honor a ``--json``/``text`` output flag,
and fail fast with a non-zero exit code on API errors. Standard library only.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple


class CliCommand:
    """Represents a CLI command and its captured execution result."""

    def __init__(self, command: str, args: List[str], api_response: Dict = None) -> None:
        self.command = command
        self.args = args
        self.api_response = api_response or {}
        self.exit_code = None
        self.stdout = ""
        self.stderr = ""

    def execute(self) -> Tuple[int, str, str]:
        """Execute the command stub, returning ``(exit_code, stdout, stderr)``."""
        return self.exit_code, self.stdout, self.stderr


class CliHandler:
    """Handle CLI commands, call the API and format output."""

    def __init__(self, api_client) -> None:
        self.api = api_client

    def pipeline_submit(
        self, filepath: str, output_format: str = "text"
    ) -> Tuple[int, str]:
        """Submit a pipeline JSON file; return ``(exit_code, output)``."""
        try:
            with open(filepath) as f:
                spec = json.load(f)

            api_response = self.api.submit_pipeline(spec)

            if api_response.get("status") != 201:
                error_msg = api_response.get("body", "Unknown error")
                return 1, f"ERROR: Failed to submit pipeline: {error_msg}"

            result = json.loads(api_response["body"])
            if output_format == "json":
                return 0, json.dumps(result)
            else:
                return 0, f"Pipeline submitted: {result['id']}\n"

        except FileNotFoundError:
            return 1, "ERROR: File not found\n"
        except json.JSONDecodeError:
            return 1, "ERROR: Invalid JSON in file\n"
        except Exception as e:
            return 1, f"ERROR: {str(e)}\n"

    def model_list(
        self,
        environment: Optional[str] = None,
        output_format: str = "text",
    ) -> Tuple[int, str]:
        """List models as a text table or JSON; return ``(exit_code, output)``."""
        try:
            api_response = self.api.list_models(environment)

            if api_response.get("status") != 200:
                error_msg = api_response.get("body", "Unknown error")
                return 1, f"ERROR: Failed to list models: {error_msg}"

            models = json.loads(api_response["body"])

            if output_format == "json":
                return 0, json.dumps(models)
            else:
                lines = ["ID                                 | Name              | Accuracy"]
                lines.append("-" * 70)
                for model in models.get("items", []):
                    lines.append(
                        f"{model['id']:36} | {model['name']:17} | {model.get('accuracy', 'N/A')}"
                    )
                return 0, "\n".join(lines) + "\n"

        except Exception as e:
            return 1, f"ERROR: {str(e)}\n"

    def model_promote(
        self,
        model_id: str,
        from_env: str,
        to_env: str,
        output_format: str = "text",
    ) -> Tuple[int, str]:
        """Promote a model between environments; return ``(exit_code, output)``."""
        try:
            spec = {
                "from_env": from_env,
                "to_env": to_env,
            }

            api_response = self.api.promote_model(model_id, spec)

            if api_response.get("status") != 200:
                error_msg = api_response.get("body", "Unknown error")
                return 1, f"ERROR: Promotion failed: {error_msg}"

            result = json.loads(api_response["body"])

            if output_format == "json":
                return 0, json.dumps(result)
            else:
                return 0, f"Model {model_id} promoted from {from_env} to {to_env}\n"

        except Exception as e:
            return 1, f"ERROR: {str(e)}\n"


class StubApiClient:
    """In-memory API client used to exercise the CLI without a live server."""

    def __init__(self) -> None:
        self.models = {
            "model_001": {"id": "model_001", "name": "fraud_detector", "accuracy": 0.95},
            "model_002": {"id": "model_002", "name": "churn_predictor", "accuracy": 0.92},
        }

    def submit_pipeline(self, spec: Dict) -> Dict:
        """Submit a pipeline spec, requiring a ``name`` field."""
        if "name" not in spec:
            return {
                "status": 400,
                "body": json.dumps({"error_code": "MISSING_NAME"}),
            }

        return {
            "status": 201,
            "body": json.dumps({"id": "pipe_123", "status": "created"}),
        }

    def list_models(self, environment: Optional[str] = None) -> Dict:
        """List all models (environment filter is a no-op stub)."""
        items = list(self.models.values())
        return {
            "status": 200,
            "body": json.dumps({"items": items, "total": len(items)}),
        }

    def promote_model(self, model_id: str, spec: Dict) -> Dict:
        """Promote a known model, validating spec and existence."""
        if "from_env" not in spec or "to_env" not in spec:
            return {
                "status": 400,
                "body": json.dumps({"error_code": "INVALID_SPEC"}),
            }

        if model_id not in self.models:
            return {
                "status": 404,
                "body": json.dumps({"error_code": "NOT_FOUND"}),
            }

        return {
            "status": 200,
            "body": json.dumps({"id": model_id, "promoted": True}),
        }


# Backwards/alternate name for the same stub client.
CliApiClient = StubApiClient
