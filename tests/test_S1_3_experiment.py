"""S1.3: honesty guards for the gate experiment (ungated / static / policy)
and the enforcement-location model (C81-C84)."""
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.enforcement_model import EnforcementModel, Location
from mlops.feature_parity import check_parity
from mlops.ml_test_score import evaluate_ml_test_score
from mlops.gate_experiment import (
    PolicyArm,
    StaticGateArm,
    UngatedArm,
    format_report,
    generate_workload,
    run_experiment,
    run_replicates,
    sensitivity_sweep,
)

SIGNALS = ("parity", "contract", "ml_test_score")
SEEDS = list(range(1, 9))


def _frac(values, pred):
    return sum(1 for v in values if pred(v)) / len(values)


def _mean(xs):
    return sum(xs) / len(xs)


@pytest.fixture(scope="module")
def big_sep():
    return generate_workload(seed=7, n=6000, defect_rate=0.3, separable=True)


@pytest.fixture(scope="module")
def big_nonsep():
    return generate_workload(seed=7, n=6000, defect_rate=0.3, separable=False)


@pytest.fixture(scope="module")
def sep_rep():
    return run_replicates(SEEDS, n=400, defect_rate=0.3, separable=True)


@pytest.fixture(scope="module")
def nonsep_rep():
    return run_replicates(SEEDS, n=400, defect_rate=0.3, separable=False)


@pytest.fixture(scope="module")
def sweep():
    return sensitivity_sweep([0.0, 0.5, 1.0], SEEDS, n=400, defect_rate=0.3, separable=True)


# (1) determinism per seed, variation across seeds
def test_workload_deterministic_per_seed_and_differs_across_seeds():
    a = generate_workload(seed=3, n=200)
    b = generate_workload(seed=3, n=200)
    c = generate_workload(seed=4, n=200)
    assert [(r.id, r.defective, r.metrics, r.signals, r.evidence) for r in a] == \
           [(r.id, r.defective, r.metrics, r.signals, r.evidence) for r in b]
    assert [r.metrics for r in a] != [r.metrics for r in c]
    assert [r.signals for r in a] != [r.signals for r in c]


# (2) no defects => nothing to escape for any arm
def test_zero_defect_rate_means_no_escapes_in_any_arm():
    wl = generate_workload(seed=5, n=300, defect_rate=0.0)
    assert not any(r.defective for r in wl)
    res = run_experiment(seed=5, n=300, defect_rate=0.0)
    for arm in ("ungated", "static", "policy"):
        assert res[arm].escaped_defects == 0
        assert res[arm].false_negatives == 0


# (3) DIAGNOSTIC: the generator matches its documented distributions
def test_workload_distribution_diagnostic_separable(big_sep):
    d = [r for r in big_sep if r.defective]
    h = [r for r in big_sep if not r.defective]
    assert 0.27 < len(d) / len(big_sep) < 0.33
    d_acc = [r.metrics["accuracy"] for r in d]
    h_acc = [r.metrics["accuracy"] for r in h]
    assert _mean(d_acc) == pytest.approx(0.930, abs=0.003)
    assert _mean(h_acc) == pytest.approx(0.947, abs=0.002)
    assert _frac(d_acc, lambda a: a >= 0.935) == pytest.approx(0.41, abs=0.04)
    # In separable mode with real checkers: most defects fail at least one signal,
    # most healthy pass all signals. Exact rates vary by checker (parity/mlts/contract)
    # but the key is that defects are much more likely to fail.
    defect_any_fail = _frac(d, lambda r: any(r.signals[s] == "fail" for s in SIGNALS))
    healthy_any_fail = _frac(h, lambda r: any(r.signals[s] == "fail" for s in SIGNALS))
    assert defect_any_fail > 0.90  # Most defects should fail at least one signal
    assert healthy_any_fail < 0.15  # Most healthy should pass all signals


# (4) non-separable: signals carry no information about defectiveness
def test_non_separable_signals_uninformative(big_nonsep):
    d = [r for r in big_nonsep if r.defective]
    h = [r for r in big_nonsep if not r.defective]
    # In non-separable mode, evidence is not corrupted based on defectiveness,
    # so signal failure rates should be similar for defects and healthy.
    # Each checker has its own baseline failure rate (not corrupted by p_parity/p_mlts/p_contract).
    defect_any_fail = _frac(d, lambda r: any(r.signals[s] == "fail" for s in SIGNALS))
    healthy_any_fail = _frac(h, lambda r: any(r.signals[s] == "fail" for s in SIGNALS))
    # Rates should be similar (within ~5 percentage points)
    assert abs(defect_any_fail - healthy_any_fail) < 0.05


# (5) ungated promotes everything
def test_ungated_escapes_every_defective_release():
    wl = generate_workload(seed=11, n=500)
    arm = UngatedArm()
    assert all(arm.decide(r).promote for r in wl)
    res = run_experiment(seed=11, n=500)
    assert res["ungated"].escaped_defects == sum(r.defective for r in wl)
    assert res["ungated"].false_positives == 0


# (6) accounting identities, checked against an independent recount
@pytest.mark.parametrize("separable", [True, False])
def test_accounting_identities_per_arm(separable):
    n, seed = 500, 21
    wl = generate_workload(seed=seed, n=n, defect_rate=0.3, separable=separable)
    n_def = sum(r.defective for r in wl)
    assert 0 < n_def < n
    res = run_experiment(seed=seed, n=n, defect_rate=0.3, separable=separable)
    arms = {"ungated": UngatedArm(), "static": StaticGateArm(), "policy": PolicyArm()}
    for name, arm in arms.items():
        decisions = [(r, arm.decide(r).promote) for r in wl]
        blocked_def = sum(1 for r, p in decisions if r.defective and not p)
        escaped = sum(1 for r, p in decisions if r.defective and p)
        healthy_blocked = sum(1 for r, p in decisions if not r.defective and not p)
        m = res[name]
        assert m.false_negatives == m.escaped_defects == escaped
        assert m.false_positives == healthy_blocked
        assert blocked_def + m.escaped_defects == n_def


# (7) separable ordering
def test_separable_ordering_policy_beats_static_beats_ungated(sep_rep):
    res, _ = sep_rep
    esc = {a: res[a]["escaped_defects"]["mean"] for a in ("ungated", "static", "policy")}
    assert esc["policy"] < esc["static"] < esc["ungated"]
    assert res["policy"]["false_positives"]["mean"] < res["static"]["false_positives"]["mean"]


# (8) honesty: no advantage when signals are uninformative
def test_non_separable_policy_advantage_vanishes(nonsep_rep):
    res, _ = nonsep_rep
    pol = res["policy"]["escaped_defects"]["mean"]
    sta = res["static"]["escaped_defects"]["mean"]
    assert sta > 0
    assert pol >= 0.85 * sta, f"policy {pol} suspiciously better than static {sta}"
    pol_rate = res["policy"]["escape_rate"]["mean"]
    sta_rate = res["static"]["escape_rate"]["mean"]
    assert pol_rate >= sta_rate, f"policy escape rate {pol_rate} below static {sta_rate}"


# (8b) NO LEAKAGE: arms never read release.defective
@pytest.mark.parametrize("separable", [True, False])
def test_arms_do_not_read_defective_label(separable):
    wl = generate_workload(seed=31, n=150, defect_rate=0.3, separable=separable)
    flipped = [dataclasses.replace(r, defective=not r.defective) for r in wl]
    assert all(a.defective != b.defective for a, b in zip(wl, flipped))
    for arm in (UngatedArm(), StaticGateArm(), PolicyArm()):
        for a, b in zip(wl, flipped):
            assert arm.decide(a) == arm.decide(b)


# (8c) EVIDENCE DERIVATION: signals are recomputable from the latent evidence
def test_signals_derive_from_real_checkers_on_evidence():
    wl = generate_workload(seed=32, n=200, defect_rate=0.3)
    seen = set()
    for r in wl:
        rep = check_parity(r.evidence["parity_pairs"], tolerance=0.05, min_pairs=8)
        parity = rep.status if rep.status in ("pass", "fail") else "fail"
        mlts = "pass" if evaluate_ml_test_score(r.evidence["ml_test_evidence"]).final >= 5.0 else "fail"
        contract = "fail" if r.evidence["contract_fails"] else "pass"
        assert r.signals == {"parity": parity, "contract": contract, "ml_test_score": mlts}
        seen.update([("parity", parity), ("mlts", mlts), ("contract", contract)])
    # both outcomes actually occur, so the equality is not trivially all-pass
    for k in ("parity", "mlts", "contract"):
        assert (k, "pass") in seen and (k, "fail") in seen


# (9) bootstrap CIs: sane, non-degenerate, deterministic
def test_bootstrap_ci_nondegenerate_contains_mean_and_deterministic(sep_rep):
    res, _ = sep_rep
    widths = []
    for arm in res.values():
        for metric in ("escaped_defects", "false_positives"):
            lo, hi = arm[metric]["ci_95"]
            assert lo <= arm[metric]["mean"] <= hi
            widths.append(hi - lo)
    assert any(w > 0 for w in widths)
    again, _ = run_replicates(SEEDS, n=400, defect_rate=0.3, separable=True)
    assert again == res


# (10) report
def test_format_report_mentions_all_arms(sep_rep):
    res, diag = sep_rep
    text = format_report(res, diag, title="t")
    assert isinstance(text, str)
    for label in ("Ungated", "Static Gate", "Policy Gate"):
        assert label in text


# (10b) rates
def test_ungated_rates_and_single_run_rates_are_count_ratios(sep_rep):
    res, _ = sep_rep
    assert res["ungated"]["escape_rate"]["mean"] > 0.99
    assert res["ungated"]["false_positive_rate"]["mean"] == 0
    for seed in (1, 2):
        wl = generate_workload(seed=seed, n=400, defect_rate=0.3, separable=True)
        nd = sum(r.defective for r in wl)
        nh = len(wl) - nd
        one, _ = run_replicates([seed], n=400, defect_rate=0.3, separable=True)
        counts = run_experiment(seed=seed, n=400, defect_rate=0.3, separable=True)
        for arm in ("ungated", "static", "policy"):
            assert one[arm]["escape_rate"]["mean"] == pytest.approx(counts[arm].escaped_defects / nd)
            assert one[arm]["false_positive_rate"]["mean"] == pytest.approx(counts[arm].false_positives / nh)


# (10c) sensitivity sweep
def test_sensitivity_sweep_null_and_threshold(sweep):
    per = sweep["per_multiplier"]
    zero = per[0.0]
    assert zero["policy_escape_mean"] >= zero["static_escape_mean"]
    for m in (0.5, 1.0):
        assert per[m]["policy_escape_ci"][1] < per[m]["static_escape_ci"][0]
        assert per[m]["policy_escape_mean"] < per[m]["static_escape_mean"]
    expected = next((m for m in sorted(per)
                     if per[m]["policy_escape_ci"][1] < per[m]["static_escape_ci"][0]), None)
    assert sweep["policy_wins_at"] == expected
    assert expected is not None and expected <= 0.5


# (10d) honest framing
def test_report_and_runner_state_simulation_and_conditional(sep_rep):
    res, diag = sep_rep
    assert isinstance(format_report(res, diag, title="t"), str)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec", "mlops", "gate_experiment.py")).read().lower()
    assert "simulation" in src and "conditional" in src


# (11) enforcement model
def _gate(release):
    return any(v == "fail" for v in release.signals.values())


def _model(locs, bypass, seed=1, latency=None):
    return EnforcementModel(
        locations=list(locs),
        bypass_prob={l: bypass for l in locs},
        latency_s=dict(latency or {}),
        seed=seed,
    )


ALL3 = [Location.BUILD, Location.REGISTRY, Location.SERVING]


def test_enforcement_deterministic_and_layering_monotone():
    wl = generate_workload(seed=9, n=2000, defect_rate=0.3)
    a = _model(ALL3, 0.3, seed=5).evaluate(wl, _gate)
    b = _model(ALL3, 0.3, seed=5).evaluate(wl, _gate)
    assert a == b

    def esc(locs):
        return _model(locs, 0.3, seed=5).evaluate(wl, _gate)["escaped"]

    e3 = esc(ALL3)
    e2 = esc([Location.BUILD, Location.REGISTRY])
    e1 = esc([Location.BUILD])
    e0 = esc([])
    assert e3 <= e2 <= e1 <= e0
    assert e3 < e1 < e0
    assert e0 == sum(r.defective for r in wl)


@pytest.mark.parametrize("locs", [ALL3, [Location.BUILD], [Location.REGISTRY, Location.SERVING]])
def test_zero_bypass_escapes_equal_gate_misses(locs):
    wl = generate_workload(seed=13, n=1500, defect_rate=0.3)
    misses = sum(1 for r in wl if r.defective and not _gate(r))
    assert misses > 0
    out = _model(locs, 0.0).evaluate(wl, _gate)
    assert out["escaped"] == misses
    assert sum(out["bypass_events"].values()) == 0


# (12) latency
def test_build_only_adds_no_latency_and_serving_adds_some():
    wl = generate_workload(seed=15, n=600, defect_rate=0.3)
    build = _model([Location.BUILD], 0.0).evaluate(wl, _gate)
    serving = _model([Location.SERVING], 0.0, latency={Location.SERVING: 0.2}).evaluate(wl, _gate)
    assert build["mean_added_latency_s"] == 0.0
    assert serving["mean_added_latency_s"] > 0.0


def test_doubling_serving_latency_increases_reported_latency():
    wl = generate_workload(seed=15, n=600, defect_rate=0.3)
    lo = _model([Location.SERVING], 0.0, latency={Location.SERVING: 0.2}).evaluate(wl, _gate)
    hi = _model([Location.SERVING], 0.0, latency={Location.SERVING: 0.4}).evaluate(wl, _gate)
    assert hi["mean_added_latency_s"] > lo["mean_added_latency_s"]


def test_mean_added_latency_never_exceeds_configured_serving_latency():
    wl = generate_workload(seed=15, n=600, defect_rate=0.3)
    out = _model([Location.SERVING], 0.0, latency={Location.SERVING: 0.2}).evaluate(wl, _gate)
    assert out["mean_added_latency_s"] <= 0.2 + 1e-9


def test_serving_latency_is_fraction_of_releases_reaching_serving():
    wl = generate_workload(seed=15, n=600, defect_rate=0.3)
    out = _model([Location.SERVING], 0.0, latency={Location.SERVING: 0.2}).evaluate(wl, _gate)
    reached = sum(1 for r in wl if not _gate(r))
    assert 0 < reached < len(wl)
    assert out["mean_added_latency_s"] <= 0.2
    assert out["mean_added_latency_s"] == pytest.approx(0.2 * reached / len(wl))


def test_build_latency_is_release_pipeline_not_request_path():
    wl = generate_workload(seed=15, n=600, defect_rate=0.3)
    out = _model([Location.BUILD], 0.0, latency={Location.BUILD: 1.0}).evaluate(wl, _gate)
    assert out["mean_added_latency_s"] == 0.0
    assert out["mean_release_latency_s"] > 0.0


def test_false_positives_count_only_blocked_healthy_releases():
    wl = generate_workload(seed=17, n=800, defect_rate=0.3)
    n_healthy_low = sum(1 for r in wl if not r.defective and r.metrics["accuracy"] < 0.94)
    strict = _model([Location.BUILD], 0.0).evaluate(wl, lambda r: r.metrics["accuracy"] < 0.94)
    assert strict["false_positives"] == n_healthy_low > 0
    only_defective = _model([Location.BUILD], 0.0).evaluate(wl, lambda r: r.defective)
    assert only_defective["false_positives"] == 0
    assert only_defective["escaped"] == 0
    assert only_defective["blocked"] == sum(r.defective for r in wl)


def test_full_bypass_equals_no_enforcement_and_adds_no_latency():
    wl = generate_workload(seed=19, n=600, defect_rate=0.3)
    lat = {l: 0.2 for l in ALL3}
    out = _model(ALL3, 1.0, latency=lat).evaluate(wl, _gate)
    none = _model([], 0.0).evaluate(wl, _gate)
    assert out["escaped"] == none["escaped"] == sum(r.defective for r in wl)
    assert out["mean_added_latency_s"] == 0.0
    assert out["mean_release_latency_s"] == 0.0
    assert out["false_positives"] == 0
