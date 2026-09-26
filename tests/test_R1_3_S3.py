"""
R1.3-S3: CLI Integration (Operator-Journey)

Test suite for C11: CLI commands map directly to API calls. Output is human-readable
(tables, JSON if --json flag). Commands fail fast with non-zero exit codes on API errors.

Dimension: operator-journey/integration cases
Mutation targets:
  - CLI success but API call fails (mutation: skip error check)
  - API error not propagated to CLI exit code (mutation: return 0 on error)
  - Output not human-readable (mutation: return raw JSON)
  - Missing JSON flag support (mutation: remove --json flag)
"""

import pytest
import json
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from io import StringIO


@dataclass
class CLIResult:
    """Result of CLI command execution."""
    exit_code: int
    stdout: str
    stderr: str


class CLIClient:
    """MLOps CLI client."""

    def __init__(self, api_client):
        self.api = api_client

    def pipeline_list(self, json_format: bool = False, limit: int = 20) -> CLIResult:
        """mlops pipeline list [--json] [--limit N]"""
        try:
            response = self.api.list_pipelines(limit=limit)

            if json_format:
                output = json.dumps(response, indent=2)
            else:
                # Human-readable table format
                output = self._format_table(response, ["id", "name", "status"])

            return CLIResult(exit_code=0, stdout=output, stderr="")

        except Exception as e:
            return CLIResult(
                exit_code=1,
                stdout="",
                stderr=f"Error: {str(e)}"
            )

    def model_list(self, json_format: bool = False) -> CLIResult:
        """mlops model list [--json]"""
        try:
            response = self.api.list_models()

            if json_format:
                output = json.dumps(response, indent=2)
            else:
                output = self._format_table(response, ["id", "name", "version"])

            return CLIResult(exit_code=0, stdout=output, stderr="")

        except Exception as e:
            return CLIResult(exit_code=1, stdout="", stderr=f"Error: {str(e)}")

    def model_promote(self, model_id: str, from_env: str, to_env: str,
                     json_format: bool = False) -> CLIResult:
        """mlops model promote <model_id> --from <env> --to <env> [--json]"""
        try:
            response = self.api.promote_model(model_id, from_env, to_env)

            if json_format:
                output = json.dumps(response, indent=2)
            else:
                output = f"✓ Model {model_id} promoted from {from_env} to {to_env}"

            return CLIResult(exit_code=0, stdout=output, stderr="")

        except Exception as e:
            return CLIResult(exit_code=1, stdout="", stderr=f"Error: {str(e)}")

    def pipeline_submit(self, pipeline_file: str, json_format: bool = False) -> CLIResult:
        """mlops pipeline submit <file> [--json]"""
        try:
            with open(pipeline_file, 'r') as f:
                pipeline_config = json.load(f)

            response = self.api.submit_pipeline(pipeline_config)

            if json_format:
                output = json.dumps(response, indent=2)
            else:
                output = f"✓ Pipeline submitted: {response['run_id']}"

            return CLIResult(exit_code=0, stdout=output, stderr="")

        except Exception as e:
            return CLIResult(exit_code=1, stdout="", stderr=f"Error: {str(e)}")

    @staticmethod
    def _format_table(data: Dict, columns: List[str]) -> str:
        """Format data as human-readable table."""
        items = data.get("items", [])

        if not items:
            return "No items found."

        # Simple table format
        header = " | ".join(columns)
        separator = "-" * len(header)
        rows = []
        rows.append(header)
        rows.append(separator)

        for item in items:
            values = [str(item.get(col, "-")) for col in columns]
            rows.append(" | ".join(values))

        return "\n".join(rows)


class MockAPIClient:
    """Mock API client for testing."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.call_log = []

    def list_pipelines(self, limit: int = 20) -> Dict:
        self.call_log.append(("list_pipelines", {"limit": limit}))
        if self.fail:
            raise RuntimeError("API error: list_pipelines failed")
        return {
            "items": [
                {"id": "p1", "name": "fraud-detection", "status": "completed"},
                {"id": "p2", "name": "pricing-model", "status": "running"}
            ]
        }

    def list_models(self) -> Dict:
        self.call_log.append(("list_models", {}))
        if self.fail:
            raise RuntimeError("API error: list_models failed")
        return {
            "items": [
                {"id": "m1", "name": "fraud-v2", "version": "v2.0.0"},
                {"id": "m2", "name": "pricing-v1", "version": "v1.5.0"}
            ]
        }

    def promote_model(self, model_id: str, from_env: str, to_env: str) -> Dict:
        self.call_log.append(("promote_model", {"model_id": model_id, "from_env": from_env, "to_env": to_env}))
        if self.fail:
            raise RuntimeError("API error: promotion failed")
        return {"model_id": model_id, "status": "promoted"}

    def submit_pipeline(self, config: Dict) -> Dict:
        self.call_log.append(("submit_pipeline", {}))
        if self.fail:
            raise RuntimeError("API error: submission failed")
        return {"run_id": "run-12345", "status": "queued"}


class TestCLISuccessJourney:
    """CLI success cases."""

    def test_pipeline_list_command(self):
        """Case 1: mlops pipeline list succeeds."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.pipeline_list()

        assert result.exit_code == 0
        assert "fraud-detection" in result.stdout
        assert "pricing-model" in result.stdout

    def test_model_list_command(self):
        """Case 2: mlops model list succeeds."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.model_list()

        assert result.exit_code == 0
        assert "fraud-v2" in result.stdout

    def test_model_promote_command(self):
        """Case 3: mlops model promote command succeeds."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.model_promote("m1", "dev", "prod")

        assert result.exit_code == 0
        assert "promoted" in result.stdout

    def test_pipeline_submit_command(self, tmp_path):
        """Case 4: mlops pipeline submit command succeeds."""
        # Create temp pipeline file
        pipeline_file = tmp_path / "pipeline.json"
        pipeline_file.write_text(json.dumps({
            "name": "test-pipeline",
            "data_version": "v1",
            "code_commit": "abc123",
            "config_hash": "x" * 64
        }))

        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.pipeline_submit(str(pipeline_file))

        assert result.exit_code == 0
        assert "run-" in result.stdout


class TestCLIJSONFormat:
    """JSON output format cases."""

    def test_json_flag_outputs_json(self):
        """Case 5: --json flag produces JSON output."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.pipeline_list(json_format=True)

        assert result.exit_code == 0
        # Should be valid JSON
        parsed = json.loads(result.stdout)
        assert "items" in parsed

    def test_default_format_is_human_readable(self):
        """Case 6: Default output is human-readable (not JSON)."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.pipeline_list(json_format=False)

        assert result.exit_code == 0
        # Should contain table-like output with pipes
        assert "|" in result.stdout

    def test_model_list_json_format(self):
        """Case 7: Model list with --json flag."""
        api = MockAPIClient()
        cli = CLIClient(api)

        result = cli.model_list(json_format=True)

        parsed = json.loads(result.stdout)
        assert len(parsed["items"]) > 0


class TestCLIErrorHandling:
    """Error handling cases."""

    def test_cli_propagates_api_error(self):
        """Case 8: CLI returns non-zero exit code on API error."""
        api = MockAPIClient(fail=True)
        cli = CLIClient(api)

        result = cli.pipeline_list()

        assert result.exit_code != 0
        assert result.exit_code == 1

    def test_cli_outputs_error_to_stderr(self):
        """Case 9: Error message goes to stderr."""
        api = MockAPIClient(fail=True)
        cli = CLIClient(api)

        result = cli.model_list()

        assert result.exit_code == 1
        assert len(result.stderr) > 0
        assert "Error" in result.stderr

    def test_cli_promotion_error_fails_fast(self):
        """Case 10: Promotion error caught immediately."""
        api = MockAPIClient(fail=True)
        cli = CLIClient(api)

        result = cli.model_promote("m1", "dev", "prod")

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Error" in result.stderr
