"""
Claim C8: Failure/Retry Semantics & Artifact Versioning.

A K8s Job failure (non-zero exit code or pod eviction) moves a run to Failed.
A retry counter allows up to 3 retries (4 total attempts); the run becomes
terminal once retries are exhausted. Every attempt persists its own artifact
set under a distinct hash keyed by (run_id, attempt_number, name, hash), so
failed-attempt artifacts are retained and older versions are never overwritten.

Source test suite: R1.2-S4 (tests/R1_2_S4.py).
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt_number: int
    job_id: str
    exit_code: Optional[int]
    output_artifacts: Dict[str, str]  # artifact_name -> content_hash
    created_at: str


class ArtifactRegistry:
    """Store artifacts keyed by run_id, attempt_number, and content hash."""

    def __init__(self):
        self.artifacts: Dict[tuple, bytes] = {}  # (run_id, attempt, name, hash) -> content
        self.metadata: Dict[str, Dict] = {}  # "run_id:attempt:name" -> metadata

    def store_attempt_artifacts(
        self,
        run_id: str,
        attempt_number: int,
        artifacts: Dict[str, bytes]  # name -> content
    ) -> Dict[str, str]:  # name -> content_hash
        """Store an attempt's artifacts, versioned by attempt; return name -> hash."""
        artifact_hashes = {}

        for name, content in artifacts.items():
            content_hash = hashlib.sha256(content).hexdigest()
            key = (run_id, attempt_number, name, content_hash)

            self.artifacts[key] = content

            # Store metadata
            metadata_key = f"{run_id}:{attempt_number}:{name}"
            self.metadata[metadata_key] = {
                "name": name,
                "hash": content_hash,
                "size": len(content),
            }

            artifact_hashes[name] = content_hash

        return artifact_hashes

    def list_artifacts_for_run(self, run_id: str) -> List[Dict]:
        """List all artifacts for a run across all attempts."""
        result = []
        for run, attempt, name, content_hash in self.artifacts.keys():
            if run == run_id:
                result.append({
                    "attempt": attempt,
                    "name": name,
                    "hash": content_hash,
                })
        return result


class RunWithRetries:
    """Pipeline run with retry support (max 3 retries)."""

    MAX_RETRIES = 3

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.state = "pending"  # pending, running, completed, failed (terminal)
        self.attempt = 1
        self.max_attempts = 1 + self.MAX_RETRIES  # 1 initial + 3 retries = 4 total
        self.attempts: List[RetryAttempt] = []
        self.current_job_id: Optional[str] = None

    def start_job(self, job_id: str):
        """Start the K8s job for the current attempt."""
        if self.state not in ("pending", "failed"):
            raise ValueError(f"Cannot start job in state {self.state}")

        self.current_job_id = job_id
        self.state = "running"

    def complete_job(self, exit_code: int, output_artifacts: Dict[str, str]):
        """Complete the job; transition state based on exit code and record it."""
        if self.state != "running":
            raise ValueError(f"Cannot complete job in state {self.state}")

        if exit_code == 0:
            # Success
            self.state = "completed"
            attempt = RetryAttempt(
                attempt_number=self.attempt,
                job_id=self.current_job_id,
                exit_code=exit_code,
                output_artifacts=output_artifacts,
                created_at="2024-09-24T00:00:00Z"
            )
            self.attempts.append(attempt)
        else:
            # Failure: transition to failed (not terminal yet if retries available)
            self.state = "failed"
            attempt = RetryAttempt(
                attempt_number=self.attempt,
                job_id=self.current_job_id,
                exit_code=exit_code,
                output_artifacts=output_artifacts,
                created_at="2024-09-24T00:00:00Z"
            )
            self.attempts.append(attempt)

    def can_retry(self) -> bool:
        """Return True if a failed run still has retries remaining."""
        if self.state != "failed":
            return False

        # Can retry if we haven't exhausted max retries
        return self.attempt < self.max_attempts

    def retry(self) -> bool:
        """Retry a failed run; increment attempt and reset to pending."""
        if not self.can_retry():
            return False

        # Increment attempt
        self.attempt += 1

        # Reset state to pending
        self.state = "pending"
        self.current_job_id = None

        return True

    def is_terminal(self) -> bool:
        """Return True if the run is completed or failed with no retries left."""
        # Terminal if: completed, or failed with no more retries
        if self.state == "completed":
            return True

        if self.state == "failed" and not self.can_retry():
            return True

        return False


class RetryManager:
    """Manage pipeline runs with retry semantics."""

    def __init__(self):
        self.runs: Dict[str, RunWithRetries] = {}
        self.artifact_registry = ArtifactRegistry()

    def submit_run(self, run_id: str) -> RunWithRetries:
        """Submit a pipeline run."""
        run = RunWithRetries(run_id)
        self.runs[run_id] = run
        return run

    def handle_job_completion(
        self,
        run_id: str,
        exit_code: int,
        output_artifacts: Dict[str, bytes]
    ) -> bool:
        """
        Handle job completion, storing artifacts and completing the job.

        Returns True if the run is terminal, False if it can be retried.
        """
        run = self.runs[run_id]

        # Store artifacts (versioned by attempt)
        artifact_hashes = self.artifact_registry.store_attempt_artifacts(
            run_id,
            run.attempt,
            output_artifacts
        )

        # Complete job
        run.complete_job(exit_code, artifact_hashes)

        # Check if terminal
        return run.is_terminal()
