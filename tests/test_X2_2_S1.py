"""X2.2-S1 (C53): backfill core.

Resolved ambiguity: the contract's conflict rule is "if name already has a
non-backfilled 'metric' entry for this run, raise BackfillConflictError".
Backfilled values are appended as kind="backfill", never kind="metric", so a
prior *backfilled* entry for the same name is never itself a "non-backfilled
metric entry" and therefore never conflicts with a later backfill of the same
name. We resolve and test this as: re-backfilling a metric that has only
prior backfilled entries is ALLOWED (not a self-conflict); only a live
(kind="metric", backfilled=False) entry blocks a backfill.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exec"))

from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.legacy.artifacts import ArtifactStore  # noqa: E402
from mlops.state_machine import PipelineRunManager  # noqa: E402
from mlops.experiments import (  # noqa: E402
    ExperimentTracker,
    BackfillConflictError,
)

import pytest  # noqa: E402


def make_tracker():
    return ExperimentTracker(PipelineRunManager(), ArtifactStore(), ChainedAuditStore(b"secret"))


def start(tracker, run_id, params):
    tracker.start_run(run_id, params)


def test_backfill_never_logged_metric_succeeds_and_appends():
    t = make_tracker()
    start(t, "r1", {"lr": 0.1})

    def replay_fn(ctx):
        return 0.42

    val = t.backfill_metric("r1", "f1_score", replay_fn)
    assert val == 0.42

    entries = t.entries("r1")
    backfill_entries = [e for e in entries if e.kind == "backfill" and e.name == "f1_score"]
    assert len(backfill_entries) == 1
    e = backfill_entries[0]
    assert e.value == 0.42
    assert e.backfilled is True


def test_replay_fn_receives_run_context_dict():
    t = make_tracker()
    start(t, "r2", {"lr": 0.05, "batch_size": 32})
    t.log_checkpoint("r2", 1, b"ckpt-one")
    t.log_checkpoint("r2", 2, b"ckpt-two")

    captured = {}

    def capturing_replay_fn(ctx):
        captured.update(ctx)
        return 1.0

    t.backfill_metric("r2", "recall", capturing_replay_fn)

    assert captured["params"] == {"lr": 0.05, "batch_size": 32}
    assert captured["checkpoints"] == {1: b"ckpt-one", 2: b"ckpt-two"}


def test_backfill_conflicts_with_live_metric_and_does_not_append():
    t = make_tracker()
    start(t, "r3", {"lr": 0.1})
    t.log_metric("r3", "acc", 0.9)

    before = list(t.entries("r3"))

    with pytest.raises(BackfillConflictError):
        t.backfill_metric("r3", "acc", lambda ctx: 0.99)

    after = t.entries("r3")
    assert after == before
    acc_entries = [e for e in after if e.name == "acc"]
    assert len(acc_entries) == 1
    assert acc_entries[0].value == 0.9
    assert acc_entries[0].backfilled is False


def test_backfill_over_prior_backfill_is_allowed_not_self_conflict():
    t = make_tracker()
    start(t, "r4", {"lr": 0.1})

    t.backfill_metric("r4", "recall", lambda ctx: 0.5)
    # Same metric name, only a prior *backfilled* entry exists -> allowed.
    val2 = t.backfill_metric("r4", "recall", lambda ctx: 0.6)
    assert val2 == 0.6

    backfill_entries = [e for e in t.entries("r4") if e.kind == "backfill" and e.name == "recall"]
    assert len(backfill_entries) == 2
    assert [e.value for e in backfill_entries] == [0.5, 0.6]
    assert all(e.backfilled for e in backfill_entries)


def test_backfill_calls_replay_fn_exactly_once_no_pipeline_rerun():
    t = make_tracker()
    start(t, "r5", {"lr": 0.1})

    submit_calls_before = len(t.run_manager.runs)
    call_count = {"n": 0}

    def counting_replay_fn(ctx):
        call_count["n"] += 1
        return 7.0

    t.backfill_metric("r5", "custom_metric", counting_replay_fn)

    assert call_count["n"] == 1
    # No new run was created/resubmitted as part of backfill.
    assert len(t.run_manager.runs) == submit_calls_before
    assert t.run_manager.get_run("r5").attempt == 1
