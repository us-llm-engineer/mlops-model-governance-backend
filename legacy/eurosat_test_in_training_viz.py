"""Mock-data tests for the in-training visualization classes in eurosat_cnn.py --
WeightTrajectoryTracker (In situ TensorView, arXiv:1806.07382) and LeftRuleAnomalyDetector
(DeepTracker, arXiv:1808.08531). Run entirely on synthetic data / a tiny real model, no dataset
download and no GPU needed, so these can catch a regression before the next multi-minute Colab
retrain rather than after."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this file's own dir (legacy/)

from eurosat_cnn import (
    LeftRuleAnomalyDetector,
    WeightTrajectoryTracker,
    per_image_correctness,
)


class _TinyModel(nn.Module):
    """Minimal stand-in with a named 'stem.0.weight' parameter, matching what
    WeightTrajectoryTracker's default param_name expects from BalancedAttnCNN."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(4, 4, bias=False))

    def forward(self, x):
        return self.stem(x)


# ---------------------------------------------------------------------------
# WeightTrajectoryTracker
# ---------------------------------------------------------------------------

def test_tracker_records_the_exact_first_n_coords_each_step():
    model = _TinyModel()
    with torch.no_grad():
        model.stem[0].weight.copy_(torch.tensor(
            [[1.0, 2.0, 3.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    tracker = WeightTrajectoryTracker(model, param_name="stem.0.weight", n_coords=3)
    tracker.step(model)
    assert tracker.points == [(1.0, 2.0, 3.0)]

    with torch.no_grad():
        model.stem[0].weight[0, 0] = 5.0
    tracker.step(model)
    assert tracker.points == [(1.0, 2.0, 3.0), (5.0, 2.0, 3.0)]


def test_tracker_rejects_a_param_name_not_in_the_model():
    model = _TinyModel()
    with pytest.raises(AssertionError):
        WeightTrajectoryTracker(model, param_name="does.not.exist")


def test_health_signal_is_zero_for_a_frozen_trajectory():
    """A trajectory that never moves (the paper's gradient-vanishing signature) must report a
    health signal of exactly 0 -- not NaN, not a small positive number."""
    model = _TinyModel()
    tracker = WeightTrajectoryTracker(model, param_name="stem.0.weight", n_coords=3)
    for _ in range(10):
        tracker.step(model)  # weights never change between steps
    assert tracker.health_signal(window=10) == pytest.approx(0.0)


def test_health_signal_matches_hand_computed_path_length():
    """3 points on a simple staircase -> path length is exactly 2 * sqrt(1^2+1^2+1^2)."""
    model = _TinyModel()
    tracker = WeightTrajectoryTracker(model, param_name="stem.0.weight", n_coords=3)
    coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
    for x, y, z in coords:
        with torch.no_grad():
            model.stem[0].weight[0, 0] = x
            model.stem[0].weight[0, 1] = y
            model.stem[0].weight[0, 2] = z
        tracker.step(model)
    expected = 2 * (3 ** 0.5)
    assert tracker.health_signal(window=3) == pytest.approx(expected)


def test_health_signal_is_nan_with_fewer_than_two_points():
    model = _TinyModel()
    tracker = WeightTrajectoryTracker(model, param_name="stem.0.weight", n_coords=3)
    assert np.isnan(tracker.health_signal())
    tracker.step(model)
    assert np.isnan(tracker.health_signal())


def test_health_signal_window_only_uses_the_most_recent_points():
    """A large jump early in the run must NOT leak into a health_signal computed with a window
    that excludes it -- otherwise the signal would misreport a currently-frozen trajectory as
    healthy just because of ancient history."""
    model = _TinyModel()
    tracker = WeightTrajectoryTracker(model, param_name="stem.0.weight", n_coords=3)
    with torch.no_grad():
        model.stem[0].weight[0, :3] = torch.tensor([0.0, 0.0, 0.0])
    tracker.step(model)
    with torch.no_grad():
        model.stem[0].weight[0, :3] = torch.tensor([100.0, 100.0, 100.0])  # huge jump
    tracker.step(model)
    for _ in range(5):
        tracker.step(model)  # frozen thereafter
    assert tracker.health_signal(window=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LeftRuleAnomalyDetector
# ---------------------------------------------------------------------------

def test_no_violations_possible_before_the_window_fills():
    """With k=3, a violation needs 3 prior constant checkpoints -- the 1st through 3rd updates
    can never produce one, regardless of what correctness values are fed in."""
    labels = np.array([0, 0, 1, 1])
    det = LeftRuleAnomalyDetector(labels, k=3)
    for t, correct in enumerate([
        np.array([1, 1, 1, 1]),
        np.array([0, 0, 0, 0]),  # would be a "violation" if the window were already full
        np.array([1, 1, 1, 1]),
    ]):
        result = det.update(t, correct)
        assert result["total_violations"] == 0


def test_violation_fires_exactly_when_a_constant_streak_breaks():
    """4 constant-correct checkpoints for image 0, then a flip on the 5th -- must be flagged as
    exactly 1 violation, attributed to image 0's class, on that 5th update only."""
    labels = np.array([0, 1])  # image 0 -> class 0, image 1 -> class 1
    det = LeftRuleAnomalyDetector(labels, k=3)
    history = [
        np.array([1, 1]),
        np.array([1, 1]),
        np.array([1, 1]),
        np.array([1, 1]),
        np.array([0, 1]),  # image 0 flips off a streak of 3 constant priors -> violation
    ]
    results = [det.update(t, c) for t, c in enumerate(history)]
    # k=3 means the window is the 3 checkpoints immediately preceding "current" -- so a change
    # at index i can only be flagged once i >= k, comparing against indices [i-k, i-1]. Here the
    # flip lands at index 4 (window = indices 1,2,3, all constant-1), so the violation fires at
    # index 4, not 3.
    assert [r["total_violations"] for r in results] == [0, 0, 0, 0, 1]
    assert results[4]["class_violation_counts"][0] == 1
    assert results[4]["class_violation_counts"][1] == 0


def test_no_violation_when_the_window_itself_is_not_constant():
    """If the k preceding values were already mixed (not all-equal), the rule makes no
    prediction, so a change on the current step is NOT a violation."""
    labels = np.array([0])
    det = LeftRuleAnomalyDetector(labels, k=3)
    history = [np.array([1]), np.array([0]), np.array([1]), np.array([0])]
    results = [det.update(t, c) for t, c in enumerate(history)]
    assert all(r["total_violations"] == 0 for r in results)


def test_class_violation_counts_cover_every_class_even_at_zero():
    labels = np.array([0, 0, 2, 2])  # class 1 deliberately absent from these 4 images
    det = LeftRuleAnomalyDetector(labels, k=1)
    det.update(0, np.array([1, 1, 1, 1]))
    result = det.update(1, np.array([1, 1, 0, 1]))  # image 2 (class 2) flips
    assert result["class_violation_counts"] == {0: 0, 2: 1}


def test_log_accumulates_one_entry_per_update_in_order():
    labels = np.array([0, 1])
    det = LeftRuleAnomalyDetector(labels, k=1)
    for t in range(4):
        det.update(t, np.array([1, 1]))
    assert [entry["t"] for entry in det.log] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# per_image_correctness
# ---------------------------------------------------------------------------

def test_per_image_correctness_matches_a_known_fixed_order():
    """A model that always predicts class 0: build a 4-image dataset with known labels and
    confirm the returned bool array matches (correct where label==0) in the DataLoader's
    unshuffled iteration order."""
    class _AlwaysZero(nn.Module):
        def forward(self, x):
            n = x.shape[0]
            out = torch.zeros(n, 2)
            out[:, 0] = 10.0  # argmax -> class 0 always
            return out

    class _FixedDS(torch.utils.data.Dataset):
        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return torch.zeros(1), int(self.labels[i])

    labels = np.array([0, 1, 0, 1])
    dl = torch.utils.data.DataLoader(_FixedDS(labels), batch_size=2, shuffle=False)
    correct = per_image_correctness(_AlwaysZero(), dl, torch.device("cpu"))
    assert correct.tolist() == [True, False, True, False]
