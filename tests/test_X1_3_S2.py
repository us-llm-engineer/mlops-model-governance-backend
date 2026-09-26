"""X1.3-S2: C46 insufficient_data when fewer than min_pairs pairs."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.feature_parity import ParityReport, check_parity, ParityGate


def test_below_min_pairs_all_matching_is_insufficient_not_pass():
    r = check_parity([("a", 1.0, 1.0), ("b", 2.0, 2.0)], tolerance=0.1, min_pairs=3)
    assert r.status == "insufficient_data"
    assert r.pairs == 2
    assert r.mismatches == []


def test_zero_pairs_is_insufficient():
    r = check_parity([], tolerance=0.1, min_pairs=1)
    assert r.status == "insufficient_data"
    assert r.pairs == 0


def test_exactly_min_pairs_is_evaluated():
    assert check_parity([("a", 1, 1), ("b", 2, 2)], tolerance=0.1, min_pairs=2).status == "pass"
    assert check_parity([("a", 1, 1), ("b", 2, 9)], tolerance=0.1, min_pairs=2).status == "fail"


def test_single_mismatch_below_min_pairs_is_neither_pass_nor_fail():
    r = check_parity([("a", 1.0, 99.0)], tolerance=0.1, min_pairs=5)
    assert r.status == "insufficient_data"
    assert r.status not in ("pass", "fail")
    assert r.pairs == 1
    assert r.mismatches == []


@pytest.mark.parametrize("kwargs", [
    dict(tolerance=0.1, min_pairs=-1),
    dict(tolerance=-0.1, min_pairs=1),
    dict(tolerance=-1, min_pairs=-1),
])
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        check_parity([("a", 1.0, 1.0)], **kwargs)
