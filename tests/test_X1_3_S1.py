"""X1.3-S1: C45 parity checker (pass/fail, tolerance boundary, mismatch listing)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.feature_parity import ParityReport, check_parity, ParityGate


def test_all_within_tolerance_passes_with_empty_mismatches():
    r = check_parity([("a", 1.0, 1.05), ("b", 2.0, 1.95), ("c", 3.0, 3.0)], tolerance=0.1, min_pairs=3)
    assert isinstance(r, ParityReport)
    assert r.status == "pass"
    assert r.pairs == 3
    assert r.mismatches == []


def test_boundary_equality_passes_and_just_over_fails():
    ok = check_parity([("a", 1.0, 1.5), ("b", 2.0, 2.0)], tolerance=0.5, min_pairs=2)
    assert ok.status == "pass"
    bad = check_parity([("a", 1.0, 1.75), ("b", 2.0, 2.0)], tolerance=0.5, min_pairs=2)
    assert bad.status == "fail"
    assert bad.mismatches == ["a"]


def test_fail_lists_exactly_mismatching_entities_and_max_diff():
    pairs = [("a", 1.0, 1.0), ("b", 10.0, 12.0), ("c", 5.0, 5.25), ("d", 7.0, 3.0), ("e", 0.0, 0.0)]
    r = check_parity(pairs, tolerance=0.5, min_pairs=1)
    assert r.status == "fail"
    assert sorted(r.mismatches) == ["b", "d"]
    assert r.max_abs_diff == pytest.approx(4.0)
    assert r.pairs == 5


def test_difference_is_absolute_in_both_directions():
    r = check_parity([("hi", 5.0, 1.0), ("lo", 1.0, 5.0)], tolerance=1.0, min_pairs=2)
    assert r.status == "fail"
    assert sorted(r.mismatches) == ["hi", "lo"]
    assert r.max_abs_diff == pytest.approx(4.0)


def test_int_values_and_mixed_int_float():
    assert check_parity([("a", 3, 3), ("b", 4, 4)], tolerance=0, min_pairs=2).status == "pass"
    r = check_parity([("a", 3, 4), ("b", 4, 4.0)], tolerance=0, min_pairs=2)
    assert r.status == "fail"
    assert r.mismatches == ["a"]
    assert r.max_abs_diff == pytest.approx(1.0)
