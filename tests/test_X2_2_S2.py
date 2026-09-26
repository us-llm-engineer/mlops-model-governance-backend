"""X2.2-S2 (C54): pivot best-run selection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exec"))

from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.legacy.artifacts import ArtifactStore  # noqa: E402
from mlops.state_machine import PipelineRunManager  # noqa: E402
from mlops.experiments import ExperimentTracker  # noqa: E402


def make_tracker():
    return ExperimentTracker(PipelineRunManager(), ArtifactStore(), ChainedAuditStore(b"secret"))


def test_best_run_max_returns_highest():
    t = make_tracker()
    t.start_run("run-a", {})
    t.start_run("run-b", {})
    t.start_run("run-c", {})
    t.log_metric("run-a", "acc", 0.70)
    t.log_metric("run-b", "acc", 0.95)
    t.log_metric("run-c", "acc", 0.40)

    assert t.best_run("acc", "max") == "run-b"


def test_best_run_min_returns_lowest():
    t = make_tracker()
    t.start_run("run-a", {})
    t.start_run("run-b", {})
    t.start_run("run-c", {})
    t.log_metric("run-a", "loss", 0.90)
    t.log_metric("run-b", "loss", 0.50)
    t.log_metric("run-c", "loss", 0.70)

    assert t.best_run("loss", "min") == "run-b"


def test_tie_broken_by_earliest_start_run_call():
    t = make_tracker()
    # Controlled start_run ordering: run-a first, then run-b, then run-c.
    t.start_run("run-a", {})
    t.start_run("run-b", {})
    t.start_run("run-c", {})
    # run-a and run-c tie for the max value; run-a started first.
    t.log_metric("run-a", "acc", 0.90)
    t.log_metric("run-b", "acc", 0.10)
    t.log_metric("run-c", "acc", 0.90)

    assert t.best_run("acc", "max") == "run-a"

    # Reversed start order (fresh tracker) flips which run wins the tie,
    # proving the winner tracks start_run order rather than name or value.
    t2 = make_tracker()
    t2.start_run("run-c", {})
    t2.start_run("run-b", {})
    t2.start_run("run-a", {})
    t2.log_metric("run-a", "acc", 0.90)
    t2.log_metric("run-b", "acc", 0.10)
    t2.log_metric("run-c", "acc", 0.90)

    assert t2.best_run("acc", "max") == "run-c"


def test_best_run_none_when_metric_never_logged():
    t = make_tracker()
    t.start_run("run-a", {})
    t.start_run("run-b", {})
    t.log_metric("run-a", "acc", 0.5)
    t.log_metric("run-b", "acc", 0.6)

    assert t.best_run("never_logged_metric", "max") is None
