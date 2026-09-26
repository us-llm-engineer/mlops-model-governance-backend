"""X2.3-S2: C58 retry/attempt isolation -- compare_attempts over a real RetryManager
keeps each attempt's artifacts separately addressable and never merges them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.experiment_registry import ExperimentRegistry
from mlops.lineage import LineageGraph
from mlops.legacy.retry import RetryManager


def _build_registry():
    return ExperimentRegistry(ChainedAuditStore(secret=b"s2-secret"), LineageGraph())


def _run_two_attempts(retry_mgr, run_id):
    """First attempt fails, second (retried) attempt succeeds, with distinct artifacts."""
    run = retry_mgr.submit_run(run_id)
    run.start_job("job-1")
    terminal_after_fail = retry_mgr.handle_job_completion(
        run_id, exit_code=1, output_artifacts={"model.bin": b"attempt-1-bytes"}
    )
    assert terminal_after_fail is False
    assert run.can_retry() is True

    assert run.retry() is True
    run.start_job("job-2")
    terminal_after_success = retry_mgr.handle_job_completion(
        run_id, exit_code=0, output_artifacts={"model.bin": b"attempt-2-bytes"}
    )
    assert terminal_after_success is True
    return run


def test_compare_attempts_keys_by_attempt_number_with_nonempty_lists():
    retry_mgr = RetryManager()
    registry = _build_registry()
    _run_two_attempts(retry_mgr, "run-x2s2-a")

    result = registry.compare_attempts(retry_mgr, "run-x2s2-a")

    assert set(result.keys()) == {1, 2}
    assert len(result[1]) > 0
    assert len(result[2]) > 0


def test_attempt_1_artifacts_never_appear_under_attempt_2_and_vice_versa():
    retry_mgr = RetryManager()
    registry = _build_registry()
    _run_two_attempts(retry_mgr, "run-x2s2-b")

    result = registry.compare_attempts(retry_mgr, "run-x2s2-b")

    hashes_1 = {item["hash"] for item in result[1]}
    hashes_2 = {item["hash"] for item in result[2]}
    assert hashes_1.isdisjoint(hashes_2)


def test_single_attempt_run_returns_dict_with_only_key_one():
    retry_mgr = RetryManager()
    registry = _build_registry()

    run = retry_mgr.submit_run("run-x2s2-c")
    run.start_job("job-1")
    terminal = retry_mgr.handle_job_completion(
        "run-x2s2-c", exit_code=0, output_artifacts={"model.bin": b"only-attempt-bytes"}
    )
    assert terminal is True

    result = registry.compare_attempts(retry_mgr, "run-x2s2-c")

    assert set(result.keys()) == {1}
    assert len(result[1]) > 0


def test_artifact_content_hashes_differ_between_attempts_for_same_name():
    retry_mgr = RetryManager()
    registry = _build_registry()
    _run_two_attempts(retry_mgr, "run-x2s2-d")

    result = registry.compare_attempts(retry_mgr, "run-x2s2-d")

    names_1 = {item["name"]: item["hash"] for item in result[1]}
    names_2 = {item["name"]: item["hash"] for item in result[2]}
    assert names_1["model.bin"] != names_2["model.bin"]


def test_compare_attempts_only_reflects_requested_run_id():
    retry_mgr = RetryManager()
    registry = _build_registry()
    _run_two_attempts(retry_mgr, "run-x2s2-e1")

    run2 = retry_mgr.submit_run("run-x2s2-e2")
    run2.start_job("job-1")
    retry_mgr.handle_job_completion(
        "run-x2s2-e2", exit_code=0, output_artifacts={"other.bin": b"unrelated-bytes"}
    )

    result_e1 = registry.compare_attempts(retry_mgr, "run-x2s2-e1")
    all_names_e1 = {item["name"] for items in result_e1.values() for item in items}
    assert "other.bin" not in all_names_e1
