"""X1.3-S3: C48 promotion gate blocks on failing / insufficient / missing parity."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.feature_parity import ParityReport, check_parity, ParityGate


def _r(status):
    return ParityReport(status=status, pairs=10, mismatches=[] if status != "fail" else ["e1"],
                        max_abs_diff=0.0 if status != "fail" else 3.0)


def test_all_pass_allows_promotion():
    d = ParityGate({"fv_a": _r("pass"), "fv_b": _r("pass")}).allow_promotion(["fv_a", "fv_b"])
    assert d.passed is True


def test_failing_report_blocks_naming_version_and_status():
    d = ParityGate({"fv_a": _r("pass"), "fv_bad": _r("fail")}).allow_promotion(["fv_a", "fv_bad"])
    assert d.passed is False
    assert "fv_bad" in d.reason and "fail" in d.reason
    assert "fv_a" not in d.reason.replace("fv_bad", "")


def test_insufficient_data_blocks_naming_version_and_status():
    d = ParityGate({"fv_x": _r("insufficient_data")}).allow_promotion(["fv_x"])
    assert d.passed is False
    assert "fv_x" in d.reason and "insufficient_data" in d.reason


def test_missing_report_blocks_naming_version():
    d = ParityGate({"fv_a": _r("pass")}).allow_promotion(["fv_a", "fv_ghost"])
    assert d.passed is False
    assert "fv_ghost" in d.reason


def test_empty_version_list_is_blocked():
    # Documented decision: nothing to vouch for -> blocked (fail closed).
    assert ParityGate({"fv_a": _r("pass")}).allow_promotion([]).passed is False
    assert ParityGate({}).allow_promotion([]).passed is False


def test_mixed_list_blocked_if_any_one_bad_in_any_position():
    reports = {"fv_1": _r("pass"), "fv_2": _r("pass"), "fv_3": _r("insufficient_data")}
    for order in (["fv_3", "fv_1", "fv_2"], ["fv_1", "fv_3", "fv_2"], ["fv_1", "fv_2", "fv_3"]):
        d = ParityGate(reports).allow_promotion(order)
        assert d.passed is False
        assert "fv_3" in d.reason


def test_extra_reports_for_unlisted_versions_do_not_matter():
    reports = {"fv_a": _r("pass"), "fv_other": _r("fail")}
    assert ParityGate(reports).allow_promotion(["fv_a"]).passed is True
