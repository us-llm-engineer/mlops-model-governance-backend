"""X2.1-S4 (C52): concurrent metric writes to the same run never lose an update."""
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.state_machine import PipelineRunManager
from mlops.legacy.artifacts import ArtifactStore
from mlops.audit import ChainedAuditStore
from mlops.experiments import ExperimentTracker

SECRET = b"x2-1-s4-secret"


def make_tracker():
    return ExperimentTracker(PipelineRunManager(), ArtifactStore(), ChainedAuditStore(SECRET))


def _run_concurrent_writers(num_threads: int):
    """Start one run, then have num_threads threads each call log_metric once
    on that SAME run_id, released together via a Barrier (no sleeps). Returns
    the tracker so the caller can inspect entries()."""
    tracker = make_tracker()
    run_id = "shared-run"
    tracker.start_run(run_id, {})

    barrier = threading.Barrier(num_threads)
    errors = []

    def writer(i):
        try:
            barrier.wait()  # maximize actual concurrent contention
            tracker.log_metric(run_id, f"metric_{i}", float(i), step=i)
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    return tracker, run_id


def test_eight_threads_concurrent_writes_no_lost_updates():
    num_threads = 8
    tracker, run_id = _run_concurrent_writers(num_threads)
    entries = [e for e in tracker.entries(run_id) if e.kind == "metric"]
    assert len(entries) == num_threads
    names = {e.name for e in entries}
    assert names == {f"metric_{i}" for i in range(num_threads)}
    by_name = {e.name: e.value for e in entries}
    for i in range(num_threads):
        assert by_name[f"metric_{i}"] == float(i)


def test_seventeen_threads_concurrent_writes_no_lost_updates():
    # a different, odd thread count than the first case, to catch bugs that
    # only manifest at certain sizes (e.g. off-by-one chunking, power-of-two
    # assumptions in a broken lock-free implementation)
    num_threads = 17
    tracker, run_id = _run_concurrent_writers(num_threads)
    entries = [e for e in tracker.entries(run_id) if e.kind == "metric"]
    assert len(entries) == num_threads
    names = {e.name for e in entries}
    assert names == {f"metric_{i}" for i in range(num_threads)}
    by_name = {e.name: e.value for e in entries}
    for i in range(num_threads):
        assert by_name[f"metric_{i}"] == float(i)
