"""
S1 Gate Experiment: Three-arm release gate comparison (Claims C81-C84).

Research question: Do policy-as-code gates beat no gates and static gates on
identical workloads? Cross-seed replicate aggregation with meaningful bootstrap CIs.

Workload: healthy acc ~ N(0.947, 0.020), defective ~ N(0.930, 0.022).
Signals: latent evidence corrupted with p_parity, p_mlts, p_contract (defects only).
Arms consume real feature_parity and ml_test_score checkers; arms never read release.defective.

Stdlib only: random, dataclasses, statistics.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .feature_parity import check_parity
from .ml_test_score import evaluate_ml_test_score, ITEMS
from .policy_engine import Rule, PolicyBundle, PolicyDecisionPoint

__all__ = [
    "Release",
    "Decision",
    "generate_workload",
    "UngatedArm",
    "StaticGateArm",
    "PolicyArm",
    "run_experiment",
    "run_replicates",
    "sensitivity_sweep",
    "default_policy_bundle",
    "format_report",
    "workload_diagnostic",
]


@dataclass
class Release:
    """A single release candidate with defectiveness and measurable signals."""

    id: str
    defective: bool
    metrics: Dict[str, float]
    signals: Dict[str, str]
    evidence: Dict[str, Any] = field(default_factory=dict)  # Raw latent evidence


@dataclass
class Decision:
    """A gate decision for a single release."""

    promote: bool
    gate_latency_s: float
    dev_actions: int


def generate_workload(
    seed: int,
    n: int,
    defect_rate: float = 0.3,
    separable: bool = True,
    p_parity: float = 0.35,
    p_mlts: float = 0.12,
    p_contract: float = 0.4,
) -> List[Release]:
    """Generate deterministic workload with latent evidence and real checkers.

    Healthy accuracy ~ N(0.947, 0.020); defective ~ N(0.930, 0.022).

    Latent evidence:
    - Feature parity: 12 pairs [(entity, online, offline)] with
      offline = online + Normal(0, 0.01). Defective releases corrupt
      each pair independently with prob p_parity by ~0.25 skew.
    - ML Test Score: Evidence dict built from ITEMS. Healthy: automated (1.0)
      with prob 0.9, manual (0.5) with 0.05, missing (0) with 0.05.
      Defective: raise missing probability by p_mlts.
    - Contract signal: Bernoulli evidence check, fails with prob 0.02 (healthy)
      or p_contract (defective).
    - Accuracy unchanged: defective N(0.930, 0.022), healthy N(0.947, 0.020).

    Non-separable mode: evidence is unaffected by label.
    Arms never read release.defective; signals derived from checkers only.
    """
    rng = random.Random(seed)
    releases = []

    for i in range(n):
        is_defective = rng.random() < defect_rate
        release_id = f"rel_{seed}_{i:04d}"

        # Accuracy: overlapping distributions
        if is_defective:
            accuracy = 0.930 + rng.gauss(0, 0.022)
        else:
            accuracy = 0.947 + rng.gauss(0, 0.020)
        accuracy = max(0.0, min(1.0, accuracy))

        # === FEATURE PARITY EVIDENCE ===
        # Generate 12 latent pairs: (entity, online, offline)
        parity_pairs = []
        for j in range(12):
            online_val = 100.0 + rng.gauss(0, 10.0)
            offline_val = online_val + rng.gauss(0, 0.01)
            entity_name = f"entity_{j}"

            # Corrupt if defective (separable mode only)
            if separable and is_defective:
                if rng.random() < p_parity:
                    # Introduce skew of ~0.25
                    offline_val += 0.25 * rng.gauss(0, 1.0)

            parity_pairs.append((entity_name, online_val, offline_val))

        # === ML TEST SCORE EVIDENCE ===
        # Build evidence dict from ITEMS
        # Healthy: automated 0.9, manual 0.05, missing 0.05
        # Defective: raise missing probability by p_mlts (reduce automated, increase missing)
        ml_test_evidence = {}

        # Adjust probabilities for defects
        auto_prob = 0.9
        if separable and is_defective:
            auto_prob = 0.9 - p_mlts
        manual_prob = 0.05
        missing_prob = 0.05 + (p_mlts if (separable and is_defective) else 0.0)

        for item in ITEMS:
            item_id = item["id"]
            prob = rng.random()
            if prob < auto_prob:
                ml_test_evidence[item_id] = 1.0  # Automated
            elif prob < auto_prob + manual_prob:
                ml_test_evidence[item_id] = 0.5  # Manual
            else:
                ml_test_evidence[item_id] = 0.0  # Missing

        # === CONTRACT SIGNAL EVIDENCE ===
        # Bernoulli: contract check fails with prob 0.02 (healthy) or p_contract (defective)
        contract_fail_prob = 0.02
        if separable and is_defective:
            contract_fail_prob = p_contract
        contract_fails = rng.random() < contract_fail_prob

        # === EVALUATE WITH REAL CHECKERS ===
        # Feature parity check
        parity_report = check_parity(parity_pairs, tolerance=0.05, min_pairs=8)
        parity_signal = parity_report.status if parity_report.status in ("pass", "fail") else "fail"

        # ML Test Score check - use floor of 5.0 (requires min category score of 5,
        # roughly expecting 5 of 7 items automated; discriminates against releases
        # with multiple missing/manual items; healthy avg ~6.5, defective ~5.6)
        ml_test_report = evaluate_ml_test_score(ml_test_evidence)
        ml_test_signal = "pass" if ml_test_report.final >= 5.0 else "fail"

        # Contract check
        contract_signal = "pass" if not contract_fails else "fail"

        # === METRICS ===
        metrics = {
            "accuracy": accuracy,
            "latency_p99_ms": 15.0 + rng.gauss(0, 2.0),
            "error_rate": 0.001 + rng.gauss(0, 0.0005),
        }

        # === SIGNALS AND EVIDENCE ===
        signals = {
            "parity": parity_signal,
            "contract": contract_signal,
            "ml_test_score": ml_test_signal,
        }

        evidence = {
            "parity_pairs": parity_pairs,
            "ml_test_evidence": ml_test_evidence,
            "contract_fails": contract_fails,
        }

        releases.append(Release(
            id=release_id,
            defective=is_defective,
            metrics=metrics,
            signals=signals,
            evidence=evidence,
        ))

    return releases


def default_policy_bundle(accuracy_floor: float = 0.90) -> PolicyBundle:
    """Create default policy bundle.

    Signal rules (parity, contract, ml_test) block on fail.
    Accuracy: hard floor at 0.90 (block); below 0.935 as warn only.
    """
    rules = [
        Rule(
            id="parity_check",
            version=1,
            kind="status_equals",
            params={"key": "parity_status", "value": "pass"},
            severity="block",
        ),
        Rule(
            id="contract_check",
            version=1,
            kind="status_equals",
            params={"key": "contract_status", "value": "pass"},
            severity="block",
        ),
        Rule(
            id="ml_test_check",
            version=1,
            kind="status_equals",
            params={"key": "ml_test_score", "value": "pass"},
            severity="block",
        ),
        Rule(
            id="accuracy_floor",
            version=1,
            kind="min_metric",
            params={"metric": "accuracy", "min": accuracy_floor},
            severity="block",
        ),
        Rule(
            id="accuracy_threshold",
            version=1,
            kind="min_metric",
            params={"metric": "accuracy", "min": 0.935},
            severity="warn",
        ),
    ]
    return PolicyBundle(rules=rules, name="default_release_gate", version=1)


class UngatedArm:
    """No gates: always promote."""

    def decide(self, release: Release) -> Decision:
        return Decision(promote=True, gate_latency_s=0.1, dev_actions=0)


class StaticGateArm:
    """Single-threshold static gate on accuracy."""

    def __init__(self, min_accuracy: float = 0.935):
        self.min_accuracy = min_accuracy

    def decide(self, release: Release) -> Decision:
        promote = release.metrics["accuracy"] >= self.min_accuracy
        gate_latency_s = 0.5
        dev_actions = 0 if promote else 1
        return Decision(promote=promote, gate_latency_s=gate_latency_s, dev_actions=dev_actions)


class PolicyArm:
    """Policy-based gate using PolicyDecisionPoint."""

    def __init__(self, decision_point=None, default_latency_s: float = 1.0):
        if decision_point is None:
            bundle = default_policy_bundle()
            decision_point = PolicyDecisionPoint(bundle=bundle)
        self.decision_point = decision_point
        self.default_latency_s = default_latency_s

    def decide(self, release: Release) -> Decision:
        context = {
            "accuracy": release.metrics["accuracy"],
            "parity_status": release.signals["parity"],
            "contract_status": release.signals["contract"],
            "ml_test_score": release.signals["ml_test_score"],
        }
        try:
            policy_decision = self.decision_point.decide("promote", context, actor="experiment")
            promote = policy_decision.allow
            dev_actions = 0 if promote else 1
            return Decision(promote=promote, gate_latency_s=self.default_latency_s, dev_actions=dev_actions)
        except Exception:
            return Decision(promote=False, gate_latency_s=self.default_latency_s, dev_actions=1)


@dataclass
class SingleRunMetrics:
    """Metrics from a single run (one seed)."""

    escaped_defects: int
    false_positives: int
    false_negatives: int
    median_lead_time_s: float
    rollback_time_s: float
    developer_actions: int


def _run_single_experiment(
    seed: int,
    n: int,
    defect_rate: float,
    separable: bool,
    p_parity: float = 0.35,
    p_mlts: float = 0.12,
    p_contract: float = 0.4,
) -> Tuple[Dict[str, SingleRunMetrics], int, int]:
    """Run one replicate for the given seed.

    Returns:
        (results_dict, num_defects_in_workload, num_healthy_in_workload)
    """
    arms = {
        "ungated": UngatedArm(),
        "static": StaticGateArm(min_accuracy=0.935),
        "policy": PolicyArm(),
    }

    workload = generate_workload(
        seed=seed, n=n, defect_rate=defect_rate, separable=separable,
        p_parity=p_parity, p_mlts=p_mlts, p_contract=p_contract
    )

    # Count defects and healthy in this replicate
    num_defects = sum(1 for r in workload if r.defective)
    num_healthy = len(workload) - num_defects

    results = {}
    for arm_name, arm in arms.items():
        escaped_defects = 0  # Defective releases that were promoted (false negatives)
        false_positives = 0  # Healthy releases that were blocked
        total_lead_time_s = 0.0
        total_rollback_time_s = 0.0
        total_dev_actions = 0

        for release in workload:
            decision = arm.decide(release)
            total_lead_time_s += decision.gate_latency_s
            total_dev_actions += decision.dev_actions

            if decision.promote:
                if release.defective:
                    # Defective promoted = false negative = escaped defect
                    escaped_defects += 1
                    total_rollback_time_s += 5.0
            else:
                if not release.defective:
                    # Healthy blocked = false positive
                    false_positives += 1
                    total_dev_actions += 2

        median_lead_time = total_lead_time_s / n if n > 0 else 0.0

        results[arm_name] = SingleRunMetrics(
            escaped_defects=escaped_defects,
            false_positives=false_positives,
            false_negatives=escaped_defects,  # FN = defectives escaped
            median_lead_time_s=median_lead_time,
            rollback_time_s=total_rollback_time_s,
            developer_actions=total_dev_actions,
        )

    return results, num_defects, num_healthy


def run_experiment(
    seed: int,
    n: int,
    defect_rate: float = 0.3,
    separable: bool = True,
    p_parity: float = 0.35,
    p_mlts: float = 0.12,
    p_contract: float = 0.4,
) -> Dict[str, Dict[str, float]]:
    """Run a single experiment."""
    results, _, _ = _run_single_experiment(seed, n, defect_rate, separable, p_parity, p_mlts, p_contract)
    return results


def workload_diagnostic(
    seeds: List[int],
    n: int,
    defect_rate: float,
    separable: bool,
    p_parity: float = 0.35,
    p_mlts: float = 0.12,
    p_contract: float = 0.4,
) -> str:
    """Print diagnostic statistics about the workload."""
    all_healthy_accs = []
    all_defect_accs = []
    defect_at_or_above_threshold = 0
    healthy_below_threshold = 0
    total_defects = 0
    total_healthy = 0

    # Signal failure rates
    defect_signal_fails = {"parity": 0, "contract": 0, "ml_test_score": 0}
    healthy_signal_fails = {"parity": 0, "contract": 0, "ml_test_score": 0}

    for seed in seeds:
        workload = generate_workload(
            seed=seed, n=n, defect_rate=defect_rate, separable=separable,
            p_parity=p_parity, p_mlts=p_mlts, p_contract=p_contract
        )
        for release in workload:
            if release.defective:
                all_defect_accs.append(release.metrics["accuracy"])
                total_defects += 1
                if release.metrics["accuracy"] >= 0.935:
                    defect_at_or_above_threshold += 1
                for sig_name in ["parity", "contract", "ml_test_score"]:
                    if release.signals[sig_name] == "fail":
                        defect_signal_fails[sig_name] += 1
            else:
                all_healthy_accs.append(release.metrics["accuracy"])
                total_healthy += 1
                if release.metrics["accuracy"] < 0.935:
                    healthy_below_threshold += 1
                for sig_name in ["parity", "contract", "ml_test_score"]:
                    if release.signals[sig_name] == "fail":
                        healthy_signal_fails[sig_name] += 1

    frac_defect_overlap = defect_at_or_above_threshold / total_defects if total_defects > 0 else 0
    frac_healthy_miss = healthy_below_threshold / total_healthy if total_healthy > 0 else 0

    lines = []
    lines.append(f"  Workload: {'separable' if separable else 'non-separable'}, defect_rate={defect_rate:.1%}")
    lines.append(f"  Defects with acc >= 0.935: {frac_defect_overlap:.1%} ({defect_at_or_above_threshold}/{total_defects})")
    lines.append(f"  Healthy with acc < 0.935: {frac_healthy_miss:.1%} ({healthy_below_threshold}/{total_healthy})")

    if total_defects > 0 and separable:
        lines.append(f"  Defect signal fails - parity: {defect_signal_fails['parity']/total_defects:.1%}, " +
                     f"contract: {defect_signal_fails['contract']/total_defects:.1%}, " +
                     f"ml_test_score: {defect_signal_fails['ml_test_score']/total_defects:.1%}")
        lines.append(f"  Healthy signal fails - parity: {healthy_signal_fails['parity']/total_healthy:.1%}, " +
                     f"contract: {healthy_signal_fails['contract']/total_healthy:.1%}, " +
                     f"ml_test_score: {healthy_signal_fails['ml_test_score']/total_healthy:.1%}")
    elif total_defects > 0:
        lines.append(f"  Signal fails (independent) - parity: {defect_signal_fails['parity']/total_defects:.1%}, " +
                     f"contract: {defect_signal_fails['contract']/total_defects:.1%}, " +
                     f"ml_test_score: {defect_signal_fails['ml_test_score']/total_defects:.1%}")
    else:
        lines.append(f"  No defects in workload")

    return "\n".join(lines)


def run_replicates(
    seeds: List[int],
    n: int,
    defect_rate: float = 0.3,
    separable: bool = True,
    p_parity: float = 0.35,
    p_mlts: float = 0.12,
    p_contract: float = 0.4,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Run replicates over multiple seeds and aggregate metrics with CIs.

    Computes escape_rate and false_positive_rate PER REPLICATE, then averages
    across replicates with bootstrap CIs. This avoids deflation from dividing
    by total defects/healthy across all seeds.

    Returns:
        Tuple of (aggregated_results, diagnostic_string)
    """
    all_results = {name: [] for name in ["ungated", "static", "policy"]}
    metric_names = ["escaped_defects", "false_positives", "false_negatives",
                     "median_lead_time_s", "rollback_time_s", "developer_actions"]

    # Run all replicates and track per-replicate counts
    per_replicate_defect_counts = []
    per_replicate_healthy_counts = []

    for seed in seeds:
        run_results, num_defects, num_healthy = _run_single_experiment(
            seed, n, defect_rate, separable, p_parity, p_mlts, p_contract
        )
        per_replicate_defect_counts.append(num_defects)
        per_replicate_healthy_counts.append(num_healthy)
        for arm_name, metrics in run_results.items():
            all_results[arm_name].append((metrics, num_defects, num_healthy))

    # Aggregate with CIs
    agg_results = {}
    for arm_name, metrics_list_with_counts in all_results.items():
        arm_agg = {}

        # Compute per-replicate counts for absolute metrics
        for metric_name in metric_names:
            values = [m[0].__getattribute__(metric_name) for m in metrics_list_with_counts]
            mean_val = statistics.mean(values)

            # Bootstrap CI
            rng = random.Random(42)
            bootstrap_means = []
            for _ in range(1000):
                sample = [rng.choice(values) for _ in range(len(values))]
                bootstrap_means.append(statistics.mean(sample))

            bootstrap_means.sort()
            idx_lo = int(0.025 * len(bootstrap_means))
            idx_hi = int(0.975 * len(bootstrap_means))
            ci_lo = bootstrap_means[idx_lo]
            ci_hi = bootstrap_means[idx_hi]

            arm_agg[metric_name] = {
                "mean": mean_val,
                "ci_95": (ci_lo, ci_hi),
            }

        # Compute per-replicate rates
        escape_rates = []
        fp_rates = []
        for (metrics, num_defects, num_healthy) in metrics_list_with_counts:
            # Avoid division by zero
            if num_defects > 0:
                escape_rates.append(metrics.escaped_defects / num_defects)
            if num_healthy > 0:
                fp_rates.append(metrics.false_positives / num_healthy)

        # Aggregate rates
        if escape_rates:
            mean_escape = statistics.mean(escape_rates)
            rng = random.Random(42)
            bootstrap_escape = []
            for _ in range(1000):
                sample = [rng.choice(escape_rates) for _ in range(len(escape_rates))]
                bootstrap_escape.append(statistics.mean(sample))
            bootstrap_escape.sort()
            idx_lo = int(0.025 * len(bootstrap_escape))
            idx_hi = int(0.975 * len(bootstrap_escape))
            arm_agg["escape_rate"] = {
                "mean": mean_escape,
                "ci_95": (bootstrap_escape[idx_lo], bootstrap_escape[idx_hi]),
            }
        else:
            arm_agg["escape_rate"] = {"mean": 0.0, "ci_95": (0.0, 0.0)}

        if fp_rates:
            mean_fp = statistics.mean(fp_rates)
            rng = random.Random(42)
            bootstrap_fp = []
            for _ in range(1000):
                sample = [rng.choice(fp_rates) for _ in range(len(fp_rates))]
                bootstrap_fp.append(statistics.mean(sample))
            bootstrap_fp.sort()
            idx_lo = int(0.025 * len(bootstrap_fp))
            idx_hi = int(0.975 * len(bootstrap_fp))
            arm_agg["false_positive_rate"] = {
                "mean": mean_fp,
                "ci_95": (bootstrap_fp[idx_lo], bootstrap_fp[idx_hi]),
            }
        else:
            arm_agg["false_positive_rate"] = {"mean": 0.0, "ci_95": (0.0, 0.0)}

        agg_results[arm_name] = arm_agg

    # Extract just the metrics for diagnostic
    all_results_clean = {
        name: [m[0] for m in metrics_list]
        for name, metrics_list in all_results.items()
    }

    diagnostic = workload_diagnostic(seeds, n, defect_rate, separable, p_parity, p_mlts, p_contract)
    return agg_results, diagnostic


def sensitivity_sweep(
    multipliers: List[float],
    seeds: List[int],
    n: int,
    defect_rate: float = 0.3,
    separable: bool = True,
) -> Dict[str, Any]:
    """Run sensitivity sweep over evidence signal strengths.

    Scales p_parity, p_mlts, p_contract together by multipliers.
    Returns per multiplier: static and policy escape rates with CIs,
    and the smallest multiplier where policy's escape-rate CI is
    strictly below static's.

    Results conditional on assumed evidence detection strengths.
    """
    results = {}

    for multiplier in multipliers:
        p_parity = 0.35 * multiplier
        p_mlts = 0.12 * multiplier
        p_contract = 0.4 * multiplier

        agg, _ = run_replicates(
            seeds=seeds,
            n=n,
            defect_rate=defect_rate,
            separable=separable,
            p_parity=p_parity,
            p_mlts=p_mlts,
            p_contract=p_contract,
        )

        static_escape = agg["static"]["escape_rate"]
        policy_escape = agg["policy"]["escape_rate"]

        results[multiplier] = {
            "static_escape_mean": static_escape["mean"],
            "static_escape_ci": static_escape["ci_95"],
            "policy_escape_mean": policy_escape["mean"],
            "policy_escape_ci": policy_escape["ci_95"],
        }

    # Find smallest multiplier where policy's CI is strictly below static's
    policy_wins_at = None
    for m in sorted(multipliers):
        static_ci = results[m]["static_escape_ci"]
        policy_ci = results[m]["policy_escape_ci"]
        # Strictly below: policy's upper bound < static's lower bound
        if policy_ci[1] < static_ci[0]:
            policy_wins_at = m
            break

    return {
        "per_multiplier": results,
        "policy_wins_at": policy_wins_at,
    }


def format_report(results: Dict[str, Dict[str, Any]], diagnostic: str = "", title: str = "") -> str:
    """Format replicate results as a table."""
    lines = []
    lines.append("=" * 130)
    lines.append(f"Gate Experiment Report{f' - {title}' if title else ''}")
    lines.append("=" * 130)
    if diagnostic:
        lines.append(diagnostic)
    lines.append("")

    lines.append(f"{'Metric':<30} {'Ungated':<30} {'Static Gate':<30} {'Policy Gate':<30}")
    lines.append("-" * 130)

    metrics_order = [
        ("Escaped Defects", "escaped_defects", "{:.1f}"),
        ("Escape Rate", "escape_rate", "{:.1%}"),
        ("False Positives", "false_positives", "{:.1f}"),
        ("FP Rate", "false_positive_rate", "{:.1%}"),
        ("Median Lead Time (s)", "median_lead_time_s", "{:.2f}"),
        ("Developer Actions", "developer_actions", "{:.1f}"),
    ]

    for metric_label, metric_key, fmt in metrics_order:
        values = {}
        for arm_name in ["ungated", "static", "policy"]:
            if arm_name not in results or metric_key not in results[arm_name]:
                values[arm_name] = "N/A"
                continue
            metric_data = results[arm_name][metric_key]
            mean_val = metric_data["mean"]
            ci_lo, ci_hi = metric_data["ci_95"]
            mean_str = fmt.format(mean_val)
            lo_str = fmt.format(ci_lo)
            hi_str = fmt.format(ci_hi)
            values[arm_name] = f"{mean_str} [{lo_str}-{hi_str}]"

        lines.append(f"{metric_label:<30} {values['ungated']:<30} {values['static']:<30} {values['policy']:<30}")

    lines.append("=" * 130)
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Running gate experiment across 30 replicates...\n")

    seeds = list(range(1, 31))

    # Sensitivity sweep over evidence signal strengths
    print("Sensitivity Sweep: Evidence Signal Strength Multipliers\n")
    print("Multiplier | Static Escape Rate | Policy Escape Rate | Policy Wins?")
    print("-" * 75)

    sweep_result = sensitivity_sweep(
        multipliers=[0.0, 0.25, 0.5, 1.0, 1.5],
        seeds=seeds,
        n=400,
        defect_rate=0.3,
        separable=True,
    )

    for mult in [0.0, 0.25, 0.5, 1.0, 1.5]:
        sr = sweep_result["per_multiplier"][mult]
        static_mean = sr["static_escape_mean"]
        static_ci = sr["static_escape_ci"]
        policy_mean = sr["policy_escape_mean"]
        policy_ci = sr["policy_escape_ci"]
        static_str = f"{static_mean:.3f} [{static_ci[0]:.3f}-{static_ci[1]:.3f}]"
        policy_str = f"{policy_mean:.3f} [{policy_ci[0]:.3f}-{policy_ci[1]:.3f}]"
        wins = "Yes" if policy_ci[1] < static_ci[0] else "No"
        print(f"{mult:9.2f} | {static_str:18} | {policy_str:18} | {wins}")

    policy_wins_at = sweep_result["policy_wins_at"]
    print()
    if policy_wins_at is not None:
        print(f"Policy beats static starting at multiplier: {policy_wins_at}")
    else:
        print("Policy does not strictly beat static at any tested multiplier.")

    print("\nNote: Results are conditional on the assumed evidence detection strengths.")
    print("This is a simulation, not evidence about real pipelines.\n")

    # Separable workload (default multiplier = 1.0)
    sep_results, sep_diag = run_replicates(
        seeds=seeds,
        n=400,
        defect_rate=0.3,
        separable=True,
        p_parity=0.35,
        p_mlts=0.12,
        p_contract=0.4,
    )
    print(format_report(sep_results, sep_diag, title="Separable Signals (30 replicates)"))

    # Non-separable workload
    nonsep_results, nonsep_diag = run_replicates(
        seeds=seeds,
        n=400,
        defect_rate=0.3,
        separable=False,
        p_parity=0.35,
        p_mlts=0.12,
        p_contract=0.4,
    )
    print(format_report(nonsep_results, nonsep_diag, title="Non-Separable Signals (30 replicates)"))

    # No defects
    nodedef_results, nodedef_diag = run_replicates(
        seeds=seeds,
        n=400,
        defect_rate=0.0,
        separable=True,
        p_parity=0.35,
        p_mlts=0.12,
        p_contract=0.4,
    )
    print(format_report(nodedef_results, nodedef_diag, title="No Defects (30 replicates)"))
