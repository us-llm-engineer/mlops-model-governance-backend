"""Mock-data tests for the in-training visualization classes in eurosat_cnn.py --
WeightTrajectoryTracker (In situ TensorView, arXiv:1806.07382). Run entirely on synthetic data /
a tiny real model, no dataset download and no GPU needed, so these can catch a regression before
the next multi-minute Colab retrain rather than after."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this file's own dir (legacy/)

from eurosat_cnn import WeightTrajectoryTracker


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
