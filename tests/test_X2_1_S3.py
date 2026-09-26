"""X2.1-S3 (C51): writable-state gate on log writes (Running/Completed only)."""
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.state_machine import PipelineRunManager, RunState
from mlops.legacy.artifacts import ArtifactStore
from mlops.audit import ChainedAuditStore
from mlops.experiments import ExperimentTracker, ExperimentError, RunNotWritableError

SECRET = b"x2-1-s3-secret"


def make_tracker():
    run_manager = PipelineRunManager()
    tracker = ExperimentTracker(run_manager, ArtifactStore(), ChainedAuditStore(SECRET))
    return tracker, run_manager


def test_log_metric_before_run_exists_at_all_raises():
    tracker, _ = make_tracker()
    with pytest.raises(RunNotWritableError):
        tracker.log_metric("never-started", "acc", 0.5)


def test_log_metric_while_pending_before_transition_to_running_raises():
    tracker, run_manager = make_tracker()
    # created directly via the run_manager (bypassing tracker.start_run), so the
    # run exists but sits in Pending -- never transitioned to Running.
    run_manager.submit_run("run-pending")
    assert run_manager.get_run("run-pending").state == RunState.PENDING
    with pytest.raises(RunNotWritableError):
        tracker.log_metric("run-pending", "acc", 0.5)


def test_log_metric_accepted_while_running():
    tracker, run_manager = make_tracker()
    tracker.start_run("run-live", {"lr": 0.01})
    assert run_manager.get_run("run-live").state == RunState.RUNNING
    tracker.log_metric("run-live", "acc", 0.42)  # must not raise
    values = [e.value for e in tracker.entries("run-live") if e.kind == "metric"]
    assert values == [0.42]


def test_log_metric_still_accepted_once_run_reaches_completed():
    tracker, run_manager = make_tracker()
    tracker.start_run("run-done", {})
    run_manager.handle_job_success("run-done")
    assert run_manager.get_run("run-done").state == RunState.COMPLETED
    tracker.log_metric("run-done", "final_acc", 0.99)  # must not raise
    values = [e.value for e in tracker.entries("run-done") if e.kind == "metric"]
    assert values == [0.99]


def test_log_metric_rejected_once_run_is_failed():
    tracker, run_manager = make_tracker()
    tracker.start_run("run-fail", {})
    run_manager.handle_job_failure("run-fail")
    assert run_manager.get_run("run-fail").state == RunState.FAILED
    with pytest.raises(RunNotWritableError):
        tracker.log_metric("run-fail", "acc", 0.1)


def test_log_metric_rejected_once_run_is_archived():
    tracker, run_manager = make_tracker()
    tracker.start_run("run-arch", {})
    run_manager.handle_job_success("run-arch")
    # PipelineRunManager exposes no dedicated archive transition method; the
    # state machine's own VALID_TRANSITIONS lists Completed -> Archived, so we
    # reach that terminal state the same way any archiver would: setting the
    # run's public .state field directly.
    run_manager.get_run("run-arch").state = RunState.ARCHIVED
    with pytest.raises(RunNotWritableError):
        tracker.log_metric("run-arch", "acc", 0.1)


def test_run_not_writable_error_is_an_experiment_error():
    assert issubclass(RunNotWritableError, ExperimentError)
