"""
Claim C5: Pipeline Run State Machine.

A pipeline run is an explicit state machine. Submitting a pipeline creates a
run in Pending state; a K8s Job is submitted on the transition to Running; Job
success moves the run to Completed, Job failure/timeout moves it to Failed; a
retry increments an attempt counter and returns the run to Pending. Invalid
transitions are rejected and concurrent state changes are serialized by a lock.

Source test suite: R1.2-S1 (tests/R1_2_S1.py).
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class RunState(Enum):
    """Lifecycle states of a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class JobStatus:
    """K8s Job status stub."""

    job_id: str
    state: str  # pending, running, success, failure, timeout, evicted
    exit_code: Optional[int] = None


@dataclass
class PipelineRun:
    """Pipeline run with state machine."""

    run_id: str
    state: RunState
    attempt: int
    job_id: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class StateMachine:
    """Enforce valid state transitions."""

    # Valid transitions: from_state -> [to_states]
    VALID_TRANSITIONS = {
        RunState.PENDING: [RunState.RUNNING],
        RunState.RUNNING: [RunState.COMPLETED, RunState.FAILED],
        RunState.COMPLETED: [RunState.ARCHIVED],
        RunState.FAILED: [RunState.PENDING],  # Retry allowed
        RunState.ARCHIVED: [],
    }

    def can_transition(self, from_state: RunState, to_state: RunState) -> bool:
        """Return True if the transition from_state -> to_state is valid."""
        return to_state in self.VALID_TRANSITIONS.get(from_state, [])

    def assert_transition(self, from_state: RunState, to_state: RunState):
        """Assert a transition is valid; raise ValueError otherwise."""
        if not self.can_transition(from_state, to_state):
            raise ValueError(
                f"Invalid transition: {from_state.value} -> {to_state.value}"
            )


class JobSimulator:
    """Mock K8s Job status."""

    def __init__(self):
        self.jobs: Dict[str, JobStatus] = {}

    def submit_job(self, run_id: str) -> str:
        """Submit a K8s Job for a run; return the generated job_id."""
        job_id = f"job-{run_id}"
        self.jobs[job_id] = JobStatus(job_id=job_id, state="pending")
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Return the status of a job; raise KeyError if unknown."""
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self.jobs[job_id]

    def simulate_job_success(self, job_id: str):
        """Mark a job as successfully completed."""
        self.jobs[job_id].state = "success"
        self.jobs[job_id].exit_code = 0

    def simulate_job_failure(self, job_id: str, exit_code: int = 1):
        """Mark a job as failed with the given non-zero exit code."""
        self.jobs[job_id].state = "failure"
        self.jobs[job_id].exit_code = exit_code

    def simulate_job_timeout(self, job_id: str):
        """Mark a job as timed out with exit code 124."""
        self.jobs[job_id].state = "timeout"
        self.jobs[job_id].exit_code = 124  # timeout exit code


class PipelineRunManager:
    """Manage pipeline runs with state machine enforcement."""

    def __init__(self):
        self.runs: Dict[str, PipelineRun] = {}
        self.state_machine = StateMachine()
        self.job_simulator = JobSimulator()
        self.lock = threading.RLock()

    def submit_run(self, run_id: str) -> PipelineRun:
        """Submit a pipeline; create a run in Pending state with attempt=1."""
        with self.lock:
            run = PipelineRun(
                run_id=run_id,
                state=RunState.PENDING,
                attempt=1
            )
            self.runs[run_id] = run
            return run

    def transition_to_running(self, run_id: str):
        """Transition a run Pending -> Running and submit its K8s Job."""
        with self.lock:
            run = self.runs[run_id]

            # Validate transition
            self.state_machine.assert_transition(run.state, RunState.RUNNING)

            # Submit job
            job_id = self.job_simulator.submit_job(run_id)

            # Update run
            run.state = RunState.RUNNING
            run.job_id = job_id
            run.started_at = datetime.utcnow()

    def handle_job_success(self, run_id: str):
        """Handle Job success; transition the run to Completed."""
        with self.lock:
            run = self.runs[run_id]

            # Validate transition
            self.state_machine.assert_transition(run.state, RunState.COMPLETED)

            # Update job status
            self.job_simulator.simulate_job_success(run.job_id)

            # Update run
            run.state = RunState.COMPLETED
            run.completed_at = datetime.utcnow()

    def handle_job_failure(self, run_id: str):
        """Handle Job failure; transition the run to Failed."""
        with self.lock:
            run = self.runs[run_id]

            # Validate transition
            self.state_machine.assert_transition(run.state, RunState.FAILED)

            # Update job status
            self.job_simulator.simulate_job_failure(run.job_id)

            # Update run
            run.state = RunState.FAILED
            run.completed_at = datetime.utcnow()

    def retry_run(self, run_id: str):
        """Retry a failed run: increment attempt and return it to Pending."""
        with self.lock:
            run = self.runs[run_id]

            if run.state != RunState.FAILED:
                raise ValueError(f"Cannot retry non-failed run: {run.state.value}")

            # Increment attempt
            run.attempt += 1

            # Transition back to Pending
            self.state_machine.assert_transition(run.state, RunState.PENDING)
            run.state = RunState.PENDING
            run.job_id = None
            run.started_at = None

    def get_run(self, run_id: str) -> PipelineRun:
        """Return the run with the given id, or None if absent."""
        return self.runs.get(run_id)
