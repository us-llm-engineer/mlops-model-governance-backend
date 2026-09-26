"""
R1.2-S1: State Machine Logic (Success/Boundary)

Test suite for C5: Pipeline run is a state machine. Submitting a pipeline creates
a run in Pending state. K8s Job success → Completed; Job failure/timeout → Failed.
Retries increment counter. No skipped transitions.

Dimension: success/boundary cases
Mutation targets:
  - Allow transition to invalid state (mutation: remove state validation)
  - Skip state transition (mutation: keep current state on event)
  - Lost retry counter (mutation: reset counter instead of incrementing)
  - Invalid state transition allowed (mutation: remove transition guard)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class RunState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class PipelineRun:
    """Pipeline execution run."""
    run_id: str
    pipeline_name: str
    state: RunState = RunState.PENDING
    attempt_number: int = 1
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    k8s_job_name: Optional[str] = None


class StateMachine:
    """Pipeline run state machine."""

    # Valid state transitions
    VALID_TRANSITIONS = {
        RunState.PENDING: {RunState.RUNNING, RunState.ARCHIVED},
        RunState.RUNNING: {RunState.COMPLETED, RunState.FAILED},
        RunState.COMPLETED: {RunState.ARCHIVED},
        RunState.FAILED: {RunState.PENDING, RunState.ARCHIVED},  # PENDING for retry
        RunState.ARCHIVED: set()  # Terminal
    }

    def __init__(self):
        self.runs: Dict[str, PipelineRun] = {}
        self.k8s_job_statuses: Dict[str, JobStatus] = {}

    def create_run(self, pipeline_name: str) -> PipelineRun:
        """Create new pipeline run in PENDING state."""
        run_id = f"run-{time.time()}"
        run = PipelineRun(run_id=run_id, pipeline_name=pipeline_name)
        self.runs[run_id] = run
        return run

    def submit_job_for_run(self, run_id: str) -> str:
        """Submit K8s Job, transition to RUNNING."""
        if run_id not in self.runs:
            raise ValueError(f"Unknown run: {run_id}")

        run = self.runs[run_id]

        # Validate transition
        if not self._can_transition(run.state, RunState.RUNNING):
            raise ValueError(f"Cannot transition from {run.state.value} to RUNNING")

        # Create job
        job_name = f"{run.run_id}-job"
        run.k8s_job_name = job_name
        run.state = RunState.RUNNING
        run.started_at = time.time()
        self.k8s_job_statuses[job_name] = JobStatus.RUNNING

        return job_name

    def complete_job(self, run_id: str, success: bool):
        """Handle K8s Job completion."""
        if run_id not in self.runs:
            raise ValueError(f"Unknown run: {run_id}")

        run = self.runs[run_id]

        if run.state != RunState.RUNNING:
            raise ValueError(f"Cannot complete run in {run.state.value} state")

        if success:
            target_state = RunState.COMPLETED
            job_status = JobStatus.SUCCEEDED
        else:
            target_state = RunState.FAILED
            job_status = JobStatus.FAILED

        run.state = target_state
        run.completed_at = time.time()
        if run.k8s_job_name:
            self.k8s_job_statuses[run.k8s_job_name] = job_status

    def retry_run(self, run_id: str) -> PipelineRun:
        """Retry failed run (transitions FAILED → PENDING, increments attempt)."""
        if run_id not in self.runs:
            raise ValueError(f"Unknown run: {run_id}")

        run = self.runs[run_id]

        if run.state != RunState.FAILED:
            raise ValueError(f"Cannot retry run in {run.state.value} state")

        if not self._can_transition(run.state, RunState.PENDING):
            raise ValueError(f"Cannot retry from {run.state.value}")

        run.state = RunState.PENDING
        run.attempt_number += 1
        run.k8s_job_name = None
        run.started_at = None
        run.completed_at = None

        return run

    def archive_run(self, run_id: str):
        """Archive run (terminal state)."""
        if run_id not in self.runs:
            raise ValueError(f"Unknown run: {run_id}")

        run = self.runs[run_id]

        if run.state == RunState.ARCHIVED:
            return  # Already archived

        if not self._can_transition(run.state, RunState.ARCHIVED):
            raise ValueError(f"Cannot archive from {run.state.value}")

        run.state = RunState.ARCHIVED

    def _can_transition(self, from_state: RunState, to_state: RunState) -> bool:
        """Check if transition is valid."""
        valid = self.VALID_TRANSITIONS.get(from_state, set())
        return to_state in valid


class TestStateMachineSuccess:
    """Success cases: valid state transitions work."""

    def test_create_run_starts_in_pending(self):
        """Case 1: New run created in PENDING state."""
        sm = StateMachine()
        run = sm.create_run("my-pipeline")

        assert run.state == RunState.PENDING
        assert run.attempt_number == 1
        assert run.k8s_job_name is None

    def test_pending_to_running_transition(self):
        """Case 2: PENDING → RUNNING transition valid."""
        sm = StateMachine()
        run = sm.create_run("pipeline-1")

        sm.submit_job_for_run(run.run_id)

        assert run.state == RunState.RUNNING
        assert run.k8s_job_name is not None
        assert run.started_at is not None

    def test_running_to_completed_on_job_success(self):
        """Case 3: RUNNING → COMPLETED on job success."""
        sm = StateMachine()
        run = sm.create_run("pipeline-2")
        sm.submit_job_for_run(run.run_id)

        sm.complete_job(run.run_id, success=True)

        assert run.state == RunState.COMPLETED
        assert run.completed_at is not None

    def test_running_to_failed_on_job_failure(self):
        """Case 4: RUNNING → FAILED on job failure."""
        sm = StateMachine()
        run = sm.create_run("pipeline-3")
        sm.submit_job_for_run(run.run_id)

        sm.complete_job(run.run_id, success=False)

        assert run.state == RunState.FAILED


class TestStateMachineBoundary:
    """Boundary cases: invalid transitions are rejected."""

    def test_cannot_submit_job_from_completed_state(self):
        """Case 5: Cannot transition COMPLETED → RUNNING."""
        sm = StateMachine()
        run = sm.create_run("pipeline-4")
        sm.submit_job_for_run(run.run_id)
        sm.complete_job(run.run_id, success=True)

        with pytest.raises(ValueError, match="Cannot transition"):
            sm.submit_job_for_run(run.run_id)

    def test_cannot_complete_non_running_run(self):
        """Case 6: Cannot complete run not in RUNNING state."""
        sm = StateMachine()
        run = sm.create_run("pipeline-5")

        with pytest.raises(ValueError, match="Cannot complete"):
            sm.complete_job(run.run_id, success=True)

    def test_cannot_retry_completed_run(self):
        """Case 7: Cannot retry COMPLETED run (only FAILED)."""
        sm = StateMachine()
        run = sm.create_run("pipeline-6")
        sm.submit_job_for_run(run.run_id)
        sm.complete_job(run.run_id, success=True)

        with pytest.raises(ValueError, match="Cannot retry"):
            sm.retry_run(run.run_id)

    def test_cannot_archive_from_running(self):
        """Case 8: Cannot archive RUNNING run (must complete first)."""
        sm = StateMachine()
        run = sm.create_run("pipeline-7")
        sm.submit_job_for_run(run.run_id)

        with pytest.raises(ValueError, match="Cannot archive"):
            sm.archive_run(run.run_id)


class TestRetryCounter:
    """Retry counter cases."""

    def test_retry_increments_attempt_number(self):
        """Case 9: Each retry increments attempt_number."""
        sm = StateMachine()
        run = sm.create_run("pipeline-8")
        assert run.attempt_number == 1

        sm.submit_job_for_run(run.run_id)
        sm.complete_job(run.run_id, success=False)
        sm.retry_run(run.run_id)

        assert run.attempt_number == 2
        assert run.state == RunState.PENDING

    def test_multiple_retries_track_attempts(self):
        """Case 10: Multiple retries track correct attempt count."""
        sm = StateMachine()
        run = sm.create_run("pipeline-9")

        for attempt in range(1, 4):
            assert run.attempt_number == attempt
            sm.submit_job_for_run(run.run_id)
            sm.complete_job(run.run_id, success=False)
            if attempt < 3:
                sm.retry_run(run.run_id)

        assert run.attempt_number == 3
