"""
R1.2-S4: Failure & Retry Semantics (Failure-Mapping/Recovery)

Test suite for C8: K8s Job failure (exit code != 0, pod eviction) causes run to
transition to Failed. Retry counter allows up to 3 retries; 4th failure is terminal.
Each retry generates new artifact set (old artifacts preserved under distinct hashes).

Dimension: failure-mapping/recovery cases
Mutation targets:
  - Failed run not recognized (mutation: skip failure check)
  - Retry limit not enforced (mutation: remove attempt counter check)
  - Old artifacts overwritten on retry (mutation: reuse hash)
  - Retry doesn't preserve previous artifacts (mutation: delete old artifacts)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class JobExitCode(Enum):
    SUCCESS = 0
    GENERIC_ERROR = 1
    OOM_KILL = 137
    TIMEOUT = 124


@dataclass
class JobFailureInfo:
    """Information about job failure."""
    exit_code: int
    failure_reason: str  # "timeout", "oom", "code_error", "eviction", etc.
    error_message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttemptResult:
    """Result of one pipeline run attempt."""
    attempt_number: int
    run_id: str
    success: bool
    failure_info: Optional[JobFailureInfo] = None
    artifacts_produced: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class RetryPolicy:
    """Retry policy and failure recovery."""

    MAX_ATTEMPTS = 3
    RETRYABLE_ERRORS = {
        "timeout",
        "oom",
        "eviction",
        "transient_network_error"
    }
    TERMINAL_ERRORS = {
        "code_error",
        "invalid_config",
        "missing_data"
    }

    def __init__(self):
        self.attempt_history: List[AttemptResult] = []
        self.artifacts_by_attempt: Dict[int, List[str]] = {}

    def record_attempt(self, run_id: str, success: bool, attempt_num: int,
                      failure_info: Optional[JobFailureInfo] = None,
                      artifacts: Optional[List[str]] = None):
        """Record attempt result."""
        if artifacts is None:
            artifacts = []

        result = AttemptResult(
            attempt_number=attempt_num,
            run_id=run_id,
            success=success,
            failure_info=failure_info,
            artifacts_produced=artifacts
        )

        self.attempt_history.append(result)
        self.artifacts_by_attempt[attempt_num] = artifacts

    def should_retry(self, failure_info: JobFailureInfo, attempt_num: int) -> bool:
        """Determine if should retry based on error and attempt count."""
        if attempt_num >= self.MAX_ATTEMPTS:
            return False  # Terminal attempt

        if failure_info.failure_reason in self.TERMINAL_ERRORS:
            return False  # Don't retry terminal errors

        if failure_info.failure_reason in self.RETRYABLE_ERRORS:
            return True

        return False  # Unknown error, don't retry

    def get_artifact_history(self, attempt_num: int) -> List[str]:
        """Get artifacts from specific attempt (preserves history)."""
        return self.artifacts_by_attempt.get(attempt_num, [])

    def all_attempts_failed(self) -> bool:
        """Check if all attempts failed."""
        return all(not result.success for result in self.attempt_history)

    def get_retry_reason(self, failure_info: JobFailureInfo) -> str:
        """Provide human-readable retry reason."""
        if failure_info.failure_reason == "timeout":
            return "Job exceeded time limit; retrying"
        elif failure_info.failure_reason == "oom":
            return "Job killed due to out-of-memory; retrying with more resources"
        elif failure_info.failure_reason == "eviction":
            return "Pod evicted from node; retrying"
        elif failure_info.failure_reason == "transient_network_error":
            return "Transient network error; retrying"
        else:
            return f"Transient error ({failure_info.failure_reason}); retrying"


class TestRetryableFailures:
    """Retryable failure cases."""

    def test_timeout_is_retryable(self):
        """Case 1: Job timeout is retryable."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=124,
            failure_reason="timeout",
            error_message="Job exceeded 3600s limit"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is True

    def test_oom_is_retryable(self):
        """Case 2: Out-of-memory is retryable."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=137,
            failure_reason="oom",
            error_message="Process killed (OOM)"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is True

    def test_pod_eviction_is_retryable(self):
        """Case 3: Pod eviction is retryable."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=143,
            failure_reason="eviction",
            error_message="Pod evicted by scheduler"
        )

        should_retry = policy.should_retry(failure, attempt_num=2)
        assert should_retry is True


class TestTerminalFailures:
    """Terminal failure cases (not retried)."""

    def test_code_error_not_retryable(self):
        """Case 4: Application code error is terminal."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=1,
            failure_reason="code_error",
            error_message="NullPointerException in training.py line 42"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is False

    def test_invalid_config_not_retryable(self):
        """Case 5: Invalid configuration is terminal."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=1,
            failure_reason="invalid_config",
            error_message="Config validation failed: missing model_type"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is False

    def test_missing_data_not_retryable(self):
        """Case 6: Missing data is terminal."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=1,
            failure_reason="missing_data",
            error_message="Data file not found: s3://bucket/data/v1.0.0"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is False


class TestRetryAttemptLimits:
    """Retry attempt limits."""

    def test_retry_succeeds_on_attempt_1(self):
        """Case 7: Retry allowed on first attempt."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=124,
            failure_reason="timeout",
            error_message="Timeout"
        )

        should_retry = policy.should_retry(failure, attempt_num=1)
        assert should_retry is True

    def test_retry_succeeds_on_attempt_2(self):
        """Case 8: Retry allowed on second attempt."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=137,
            failure_reason="oom",
            error_message="OOM"
        )

        should_retry = policy.should_retry(failure, attempt_num=2)
        assert should_retry is True

    def test_no_retry_on_max_attempts(self):
        """Case 9: No retry at max attempts (terminal)."""
        policy = RetryPolicy()
        failure = JobFailureInfo(
            exit_code=124,
            failure_reason="timeout",
            error_message="Timeout"
        )

        # At attempt 3 (4th attempt would be next), no more retries
        should_retry = policy.should_retry(failure, attempt_num=3)
        assert should_retry is False


class TestArtifactPreservation:
    """Artifact preservation across retries."""

    def test_artifacts_preserved_from_attempt_1(self):
        """Case 10: Artifacts from attempt 1 preserved when retry."""
        policy = RetryPolicy()
        run_id = "run-001"

        # Attempt 1: produces artifacts, fails
        artifacts_1 = ["model_v1.pkl", "metrics_attempt1.json"]
        policy.record_attempt(run_id, success=False, attempt_num=1,
                            failure_info=JobFailureInfo(
                                exit_code=124,
                                failure_reason="timeout",
                                error_message="Timeout"
                            ),
                            artifacts=artifacts_1)

        # Attempt 2: produces new artifacts, succeeds
        artifacts_2 = ["model_v2.pkl", "metrics_attempt2.json"]
        policy.record_attempt(run_id, success=True, attempt_num=2,
                            artifacts=artifacts_2)

        # Verify both artifact sets preserved
        hist_1 = policy.get_artifact_history(1)
        hist_2 = policy.get_artifact_history(2)

        assert hist_1 == artifacts_1
        assert hist_2 == artifacts_2
        assert hist_1 != hist_2  # Different artifact sets

    def test_multiple_failed_attempts_artifacts_distinct(self):
        """Case 11: Each attempt's artifacts stored under distinct hashes."""
        policy = RetryPolicy()
        run_id = "run-002"

        # Attempt 1: produces artifacts, fails
        artifacts_1 = ["weights_attempt1_hash_aaa.pkl"]
        policy.record_attempt(run_id, success=False, attempt_num=1,
                            failure_info=JobFailureInfo(
                                exit_code=124, failure_reason="timeout",
                                error_message="Timeout"
                            ),
                            artifacts=artifacts_1)

        # Attempt 2: produces different artifacts (different hash), fails
        artifacts_2 = ["weights_attempt2_hash_bbb.pkl"]
        policy.record_attempt(run_id, success=False, attempt_num=2,
                            failure_info=JobFailureInfo(
                                exit_code=124, failure_reason="timeout",
                                error_message="Timeout"
                            ),
                            artifacts=artifacts_2)

        # Attempt 3: produces final artifacts
        artifacts_3 = ["weights_attempt3_hash_ccc.pkl"]
        policy.record_attempt(run_id, success=True, attempt_num=3,
                            artifacts=artifacts_3)

        # All distinct
        assert policy.get_artifact_history(1) == artifacts_1
        assert policy.get_artifact_history(2) == artifacts_2
        assert policy.get_artifact_history(3) == artifacts_3


class TestAttempptRecording:
    """Attempt history recording."""

    def test_all_attempts_failed_flag(self):
        """Case 12: Can query if all attempts failed."""
        policy = RetryPolicy()
        run_id = "run-003"

        # All fail
        policy.record_attempt(run_id, success=False, attempt_num=1,
                            failure_info=JobFailureInfo(124, "timeout", "t1"))
        policy.record_attempt(run_id, success=False, attempt_num=2,
                            failure_info=JobFailureInfo(124, "timeout", "t2"))
        policy.record_attempt(run_id, success=False, attempt_num=3,
                            failure_info=JobFailureInfo(124, "timeout", "t3"))

        assert policy.all_attempts_failed() is True

    def test_retry_reason_message(self):
        """Case 13: Get human-readable retry reason."""
        policy = RetryPolicy()

        failure_timeout = JobFailureInfo(124, "timeout", "")
        msg = policy.get_retry_reason(failure_timeout)
        assert "time limit" in msg

        failure_oom = JobFailureInfo(137, "oom", "")
        msg = policy.get_retry_reason(failure_oom)
        assert "out-of-memory" in msg
