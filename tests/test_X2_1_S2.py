"""X2.1-S2 (C50): checkpoint logging via the real ArtifactStore, keyed by (run_id, step)."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.state_machine import PipelineRunManager
from mlops.legacy.artifacts import ArtifactStore
from mlops.audit import ChainedAuditStore
from mlops.experiments import ExperimentTracker

SECRET = b"x2-1-s2-secret"


def make_tracker():
    store = ArtifactStore()
    tracker = ExperimentTracker(PipelineRunManager(), store, ChainedAuditStore(SECRET))
    return tracker, store


def test_checkpoint_id_has_hash_colon_filename_shape_and_is_content_addressed():
    tracker, store = make_tracker()
    tracker.start_run("run-a", {})
    content = b"model-weights-v1"
    artifact_id = tracker.log_checkpoint("run-a", step=0, content=content)
    assert ":" in artifact_id
    hash_part, _, filename_part = artifact_id.partition(":")
    assert hash_part == store.compute_hash(content)
    assert len(hash_part) == 64
    int(hash_part, 16)  # valid hex
    assert filename_part  # non-empty filename component


def test_get_checkpoint_returns_byte_identical_content():
    tracker, _ = make_tracker()
    tracker.start_run("run-b", {})
    content = bytes(range(256)) * 4  # deterministic non-trivial payload
    tracker.log_checkpoint("run-b", step=5, content=content)
    fetched = tracker.get_checkpoint("run-b", step=5)
    assert fetched == content
    assert isinstance(fetched, bytes)


def test_two_different_steps_get_distinct_ids_and_distinct_content():
    tracker, _ = make_tracker()
    tracker.start_run("run-c", {})
    id0 = tracker.log_checkpoint("run-c", step=0, content=b"epoch-0-weights")
    id1 = tracker.log_checkpoint("run-c", step=1, content=b"epoch-1-weights")
    assert id0 != id1
    assert tracker.get_checkpoint("run-c", 0) == b"epoch-0-weights"
    assert tracker.get_checkpoint("run-c", 1) == b"epoch-1-weights"


def test_identical_content_at_two_different_run_step_pairs_both_retrievable_independently():
    tracker, _ = make_tracker()
    tracker.start_run("run-d", {})
    tracker.start_run("run-e", {})
    same_content = b"shared-checkpoint-bytes"
    id_d = tracker.log_checkpoint("run-d", step=0, content=same_content)
    id_e = tracker.log_checkpoint("run-e", step=0, content=same_content)
    # both independently retrievable with the exact same content
    assert tracker.get_checkpoint("run-d", 0) == same_content
    assert tracker.get_checkpoint("run-e", 0) == same_content
    # keyed by (run_id, step) via distinct filenames, so ids need not collide
    # even though content hash is identical
    hash_d = id_d.split(":", 1)[0]
    hash_e = id_e.split(":", 1)[0]
    assert hash_d == hash_e  # same content -> same content hash


def test_identical_content_same_run_different_steps_both_independently_retrievable():
    tracker, _ = make_tracker()
    tracker.start_run("run-f", {})
    same_content = b"reused-bytes-across-steps"
    id0 = tracker.log_checkpoint("run-f", step=0, content=same_content)
    id1 = tracker.log_checkpoint("run-f", step=1, content=same_content)
    assert id0 != id1  # distinct (run_id, step) -> distinct filenames -> distinct ids
    assert tracker.get_checkpoint("run-f", 0) == same_content
    assert tracker.get_checkpoint("run-f", 1) == same_content
