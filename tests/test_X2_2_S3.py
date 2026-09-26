"""X2.2-S3 (C55): statistical pass-through.

compare_run_to_baseline must equal StatisticalJudge().judge(cand, base)
called directly on the SAME two lists -- exact dict equality, not just
matching keys. mann_whitney_u itself is grounded with a small,
hand-computable example independent of the pass-through wrapper.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exec"))

from mlops.audit import ChainedAuditStore  # noqa: E402
from mlops.lineage import LineageGraph  # noqa: E402
from mlops.legacy.metrics import StatisticalJudge, mann_whitney_u  # noqa: E402
from mlops.experiment_registry import ExperimentRegistry  # noqa: E402


def make_registry():
    return ExperimentRegistry(
        rbac=None,
        audit=ChainedAuditStore(b"secret"),
        lineage=LineageGraph(),
    )


def test_compare_matches_direct_judge_when_p_below_alpha_promote_true():
    reg = make_registry()
    judge = StatisticalJudge()
    cand = [1.0, 2.0, 3.0, 4.0, 5.0]
    base = [10.0, 11.0, 12.0, 13.0, 14.0]

    via_registry = reg.compare_run_to_baseline(judge, cand, base)
    direct = judge.judge(cand, base)

    assert via_registry == direct
    assert direct["p_value"] < 0.05
    assert direct["promote"] is True


def test_compare_matches_direct_judge_when_p_at_or_above_alpha_promote_false():
    reg = make_registry()
    judge = StatisticalJudge()
    cand = [5.0, 6.0, 7.0, 8.0, 9.0]
    base = [5.0, 6.0, 7.0, 8.0, 9.0]

    via_registry = reg.compare_run_to_baseline(judge, cand, base)
    direct = judge.judge(cand, base)

    assert via_registry == direct
    assert direct["p_value"] >= 0.05
    assert direct["promote"] is False


def test_compare_matches_direct_judge_for_a_third_distinct_sample_pair():
    reg = make_registry()
    judge = StatisticalJudge()
    cand = [2.0, 2.5, 3.0]
    base = [2.9, 3.1, 3.4, 3.6]

    via_registry = reg.compare_run_to_baseline(judge, cand, base)
    direct = judge.judge(cand, base)

    assert via_registry == direct
    # Pin down exact pass-through equality field by field too.
    assert via_registry["u_statistic"] == direct["u_statistic"]
    assert via_registry["p_value"] == direct["p_value"]
    assert via_registry["promote"] == direct["promote"]


def test_mann_whitney_u_hand_checkable_tiny_example():
    # sample_a = [1, 2], sample_b = [3, 4]: fully separated, a stochastically
    # less than b. Combined ranks: 1->1, 2->2, 3->3, 4->4 (no ties).
    # rank_sum_a = 1 + 2 = 3; U1 = rank_sum_a - n1*(n1+1)/2 = 3 - 3 = 0.
    # mean_u = n1*n2/2 = 2; std_u = sqrt(n1*n2*(n1+n2+1)/12) = sqrt(20/12).
    # z = (0 - 2) / sqrt(20/12); p = 0.5*(1 + erf(z/sqrt(2))).
    n1, n2 = 2, 2
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (0.0 - mean_u) / std_u
    expected_p = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    u, p = mann_whitney_u([1.0, 2.0], [3.0, 4.0])

    assert u == 0.0
    assert p == pytest.approx(expected_p, abs=1e-9)
