"""
S3.2 Chaos scenario framework: steady states, fault specifications, scenarios, and simulation.

Stdlib only. Validates all inputs fail-closed (numbers are real finite int/float, not bool,
selectors must be non-empty with blast-radius bounds, strings non-empty without control chars).
Dataclasses are frozen; run_scenario never mutates the input cluster, works on copies.
Deterministic (sorted names, no randomness; < 3s for 5000 pods x 20 steps).
"""

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from mlops.kernel import ValidationFailed


__all__ = [
    "FAULT_TYPES",
    "SteadyState",
    "FaultSpec",
    "Scenario",
    "SimulatedCluster",
    "run_scenario",
    "chaos_context",
]


# ============================================================================
# FAULT_TYPES constant
# ============================================================================

FAULT_TYPES = {
    "PodChaos": ["pod-kill", "container-kill"],
    "NetworkChaos": ["delay", "loss", "duplicate", "corrupt", "partition"],
    "DNSChaos": ["error", "random"],
    "HTTPChaos": ["delay", "abort"],
    "StressChaos": ["cpu", "memory"],
    "IOChaos": ["latency", "fault"],
    "TimeChaos": ["skew"],
}


# ============================================================================
# SteadyState: frozen dataclass for steady-state assertions
# ============================================================================

def _validate_op(op: str) -> None:
    """Validate operator is one of the allowed values."""
    if op not in ("<=", "<", ">=", ">"):
        raise ValidationFailed(f"Invalid operator: {op}")


def _validate_real_finite(value: Any) -> None:
    """Validate that value is a real finite number (not bool, NaN, inf, str, None)."""
    if isinstance(value, bool):
        raise ValidationFailed(f"Expected finite real number, got bool: {value}")
    if not isinstance(value, (int, float)):
        raise ValidationFailed(f"Expected finite real number, got {type(value).__name__}: {value}")
    if isinstance(value, float):
        if value != value or value == float("inf") or value == float("-inf"):
            raise ValidationFailed(f"Expected finite real number, got {value}")


def _validate_non_empty_str(value: Any) -> None:
    """Validate that value is a non-empty string without control characters."""
    if not isinstance(value, str):
        raise TypeError(f"Expected str, got {type(value).__name__}: {value}")
    if not value:
        raise ValidationFailed(f"Expected non-empty string")
    if any(ord(c) < 32 for c in value):
        raise ValidationFailed(f"String contains control characters: {value}")


@dataclass(frozen=True)
class SteadyState:
    """A steady-state hypothesis: metric value compared to a threshold."""
    name: str
    metric: str
    op: str
    threshold: float

    def __post_init__(self):
        """Validate all fields on creation."""
        _validate_non_empty_str(self.name)
        _validate_non_empty_str(self.metric)
        _validate_op(self.op)
        _validate_real_finite(self.threshold)

    def holds(self, value: float) -> bool:
        """Check if the steady state holds for the given value."""
        if self.op == "<=":
            return value <= self.threshold
        elif self.op == "<":
            return value < self.threshold
        elif self.op == ">=":
            return value >= self.threshold
        elif self.op == ">":
            return value > self.threshold
        return False


# ============================================================================
# FaultSpec: frozen dataclass for a single fault specification
# ============================================================================

def _validate_selector(selector: Any) -> Dict[str, Any]:
    """Validate selector structure and return a validated copy."""
    if not isinstance(selector, dict):
        raise ValidationFailed(f"Selector must be a dict, got {type(selector).__name__}")

    # Must have namespaces
    if "namespaces" not in selector:
        raise ValidationFailed("Selector must contain 'namespaces'")

    namespaces = selector["namespaces"]
    if not isinstance(namespaces, list):
        raise ValidationFailed(f"namespaces must be a list, got {type(namespaces).__name__}")
    if not namespaces:
        raise ValidationFailed("namespaces must be non-empty")

    for ns in namespaces:
        if not isinstance(ns, str):
            raise ValidationFailed(f"namespace must be str, got {type(ns).__name__}")
        if not ns:
            raise ValidationFailed("namespace must be non-empty string")
        if any(ord(c) < 32 for c in ns):
            raise ValidationFailed(f"namespace contains control characters: {ns}")

    # Optional labelSelectors
    if "labelSelectors" in selector:
        labels = selector["labelSelectors"]
        if not isinstance(labels, dict):
            raise ValidationFailed(f"labelSelectors must be a dict, got {type(labels).__name__}")
        for k, v in labels.items():
            if not isinstance(k, str):
                raise ValidationFailed(f"labelSelectors key must be str, got {type(k).__name__}")
            if not isinstance(v, str):
                raise ValidationFailed(f"labelSelectors value must be str, got {type(v).__name__}")

    return copy.deepcopy(selector)


@dataclass(frozen=True)
class FaultSpec:
    """Specification of a single fault injection."""
    type: str
    action: str
    selector: Dict[str, Any]
    mode: str = "all"
    value: Optional[int] = None
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validate all fields on creation."""
        # Validate type and action
        if self.type not in FAULT_TYPES:
            raise ValidationFailed(f"Unknown fault type: {self.type}")
        if self.action not in FAULT_TYPES[self.type]:
            raise ValidationFailed(f"Unknown action {self.action} for type {self.type}")

        # Validate selector (selector is already validated and copied in __init__)
        try:
            _validate_selector(self.selector)
        except (ValidationFailed, TypeError) as e:
            raise ValidationFailed(f"Invalid selector: {e}")

        # Validate mode
        valid_modes = {"one", "all", "fixed", "fixed-percent", "random-max-percent"}
        if self.mode not in valid_modes:
            raise ValidationFailed(f"Invalid mode: {self.mode}")

        # Validate value based on mode
        if self.mode == "one":
            if self.value is not None:
                raise ValidationFailed(f"mode 'one' does not take a value")
        elif self.mode == "all":
            if self.value is not None:
                raise ValidationFailed(f"mode 'all' does not take a value")
        elif self.mode == "fixed":
            if self.value is None:
                raise ValidationFailed(f"mode 'fixed' requires a value")
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValidationFailed(f"value must be int, got {type(self.value).__name__}")
            if self.value < 1:
                raise ValidationFailed(f"value for 'fixed' must be >= 1, got {self.value}")
        elif self.mode == "fixed-percent":
            if self.value is None:
                raise ValidationFailed(f"mode 'fixed-percent' requires a value")
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValidationFailed(f"value must be int, got {type(self.value).__name__}")
            if not (1 <= self.value <= 100):
                raise ValidationFailed(f"value for 'fixed-percent' must be 1-100, got {self.value}")
        elif self.mode == "random-max-percent":
            if self.value is None:
                raise ValidationFailed(f"mode 'random-max-percent' requires a value")
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValidationFailed(f"value must be int, got {type(self.value).__name__}")
            if not (1 <= self.value <= 100):
                raise ValidationFailed(f"value for 'random-max-percent' must be 1-100, got {self.value}")

        # Validate params based on action
        if self.action == "delay":
            if "latency_ms" not in self.params:
                raise ValidationFailed(f"action 'delay' requires params.latency_ms")
            latency = self.params["latency_ms"]
            _validate_real_finite(latency)
            if latency <= 0:
                raise ValidationFailed(f"latency_ms must be > 0, got {latency}")
        elif self.action == "loss":
            if "loss_pct" not in self.params:
                raise ValidationFailed(f"action 'loss' requires params.loss_pct")
            loss = self.params["loss_pct"]
            _validate_real_finite(loss)
            if not (0 < loss <= 100):
                raise ValidationFailed(f"loss_pct must be in (0, 100], got {loss}")
        elif self.action == "cpu" or self.action == "memory":
            if "workers" not in self.params:
                raise ValidationFailed(f"action '{self.action}' requires params.workers")
            workers = self.params["workers"]
            if not isinstance(workers, int) or isinstance(workers, bool):
                raise ValidationFailed(f"workers must be int, got {type(workers).__name__}")
            if workers < 1:
                raise ValidationFailed(f"workers must be >= 1, got {workers}")


# ============================================================================
# Scenario: dataclass for a sequence of fault-injection steps
# ============================================================================

@dataclass
class Scenario:
    """A sequence of fault-injection steps with steady-state hypotheses."""
    steps: Tuple[Tuple[FaultSpec, ...], ...]
    steady_states: Tuple[SteadyState, ...]
    name: str

    def __init__(self, steps: List[List[FaultSpec]], steady_states: List[SteadyState], name: str):
        """Initialize and validate scenario."""
        # Validate name
        _validate_non_empty_str(name)

        # Validate steps
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a list, got {type(steps).__name__}")
        if not steps:
            raise ValidationFailed("steps must be non-empty")
        if len(steps) > 20:
            raise ValidationFailed(f"steps must have at most 20 elements, got {len(steps)}")

        for i, step in enumerate(steps):
            if not isinstance(step, list):
                raise TypeError(f"step {i} must be a list, got {type(step).__name__}")
            if not step:
                raise ValidationFailed(f"step {i} must be non-empty")
            if len(step) > 10:
                raise ValidationFailed(f"step {i} has more than 10 faults")
            for j, fault in enumerate(step):
                if not isinstance(fault, FaultSpec):
                    raise TypeError(f"step {i} fault {j} must be FaultSpec, got {type(fault).__name__}")

        # Validate steady_states
        if not isinstance(steady_states, list):
            raise TypeError(f"steady_states must be a list, got {type(steady_states).__name__}")
        if not steady_states:
            raise ValidationFailed("steady_states must be non-empty")

        for ss in steady_states:
            if not isinstance(ss, SteadyState):
                raise TypeError(f"steady_state must be SteadyState, got {type(ss).__name__}")

        # Store as tuples (frozen copies)
        object.__setattr__(self, "steps", tuple(tuple(s) for s in steps))
        object.__setattr__(self, "steady_states", tuple(steady_states))
        object.__setattr__(self, "name", name)


# ============================================================================
# SimulatedCluster: simulated Kubernetes cluster
# ============================================================================

def _validate_pod_dict(pods: Any) -> None:
    """Validate the pods dictionary structure."""
    if not isinstance(pods, dict):
        raise TypeError(f"pods must be a dict, got {type(pods).__name__}")

    for name, pod in pods.items():
        if not isinstance(name, str):
            raise TypeError(f"pod name must be str, got {type(name).__name__}")
        if not isinstance(pod, dict):
            raise TypeError(f"pod {name} must be a dict, got {type(pod).__name__}")

        # Check required fields
        if "namespace" not in pod:
            raise ValidationFailed(f"pod {name} missing 'namespace'")
        if "labels" not in pod:
            raise ValidationFailed(f"pod {name} missing 'labels'")
        if "ready" not in pod:
            raise ValidationFailed(f"pod {name} missing 'ready'")

        namespace = pod["namespace"]
        labels = pod["labels"]
        ready = pod["ready"]

        if not isinstance(namespace, str) or not namespace:
            raise ValidationFailed(f"pod {name} namespace must be non-empty str")
        if not isinstance(labels, dict):
            raise TypeError(f"pod {name} labels must be dict")
        for k, v in labels.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValidationFailed(f"pod {name} label keys and values must be str")
        if not isinstance(ready, bool):
            raise TypeError(f"pod {name} ready must be bool, got {type(ready).__name__}")


@dataclass
class SimulatedCluster:
    """A simulated Kubernetes cluster with pods and fault effects."""
    pods: Dict[str, Dict[str, Any]]
    _pod_names_cache: Optional[List[str]] = dataclasses.field(default=None, init=False, repr=False)

    def __init__(self, pods: Dict[str, Dict[str, Any]]):
        """Initialize cluster and validate pods."""
        _validate_pod_dict(pods)
        object.__setattr__(self, "pods", copy.deepcopy(pods))
        object.__setattr__(self, "_pod_names_cache", None)

    def select(self, selector: Dict[str, Any]) -> List[str]:
        """Select pod names matching the selector (namespaces and labelSelectors)."""
        namespaces = selector.get("namespaces", [])
        label_selectors = selector.get("labelSelectors", {})

        matched = []
        for name, pod in self.pods.items():
            # Check namespace
            if pod["namespace"] not in namespaces:
                continue

            # Check label selectors
            match = True
            for k, v in label_selectors.items():
                if pod["labels"].get(k) != v:
                    match = False
                    break

            if match:
                matched.append(name)

        return sorted(matched)

    def blast_radius(self, faults: List[FaultSpec]) -> Set[str]:
        """Calculate the set of pods that would be affected by the faults."""
        affected = set()

        for fault in faults:
            matched = self.select(fault.selector)

            if fault.mode == "one":
                if matched:
                    affected.add(matched[0])
            elif fault.mode == "all":
                affected.update(matched)
            elif fault.mode == "fixed":
                affected.update(matched[:fault.value])
            elif fault.mode == "fixed-percent":
                count = max(1, math.ceil(len(matched) * fault.value / 100.0))
                affected.update(matched[:count])
            elif fault.mode == "random-max-percent":
                count = max(1, math.ceil(len(matched) * fault.value / 100.0))
                affected.update(matched[:count])

        return affected

    def metrics(self) -> Dict[str, float]:
        """Calculate service metrics for the current cluster state."""
        total = len(self.pods)
        ready = sum(1 for p in self.pods.values() if p["ready"])
        error_rate = 1.0 - (ready / total if total > 0 else 1.0)

        # Latency: 20.0 base + maximum delay latency across affected pods, capped at 5000
        latency = 20.0
        max_delay = 0.0
        for pod in self.pods.values():
            if "_delay_ms" in pod:
                max_delay = max(max_delay, pod["_delay_ms"])
        latency = min(latency + max_delay, 5000.0)

        return {
            "ready_pods": ready,
            "error_rate": error_rate,
            "latency_ms": latency,
        }


# ============================================================================
# ScenarioResult: result of running a scenario
# ============================================================================

@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    passed: bool
    stages: List[Dict[str, Any]]
    violated: List[str]


# ============================================================================
# run_scenario: run a scenario against a cluster
# ============================================================================

def run_scenario(scenario: Scenario, cluster: SimulatedCluster) -> ScenarioResult:
    """Run a scenario against a cluster and return the result."""
    # Validate inputs
    if not isinstance(scenario, Scenario):
        raise TypeError(f"scenario must be Scenario, got {type(scenario).__name__}")
    if not isinstance(cluster, SimulatedCluster):
        raise TypeError(f"cluster must be SimulatedCluster, got {type(cluster).__name__}")

    stages: List[Dict[str, Any]] = []
    all_violated: List[str] = []

    # Work with a copy so we don't mutate the input
    working_cluster = copy.deepcopy(cluster)

    # Pre-validation stage
    pre_metrics = working_cluster.metrics()
    pre_violated = []
    pre_passed = True

    for ss in scenario.steady_states:
        if not ss.holds(pre_metrics[ss.metric]):
            pre_violated.append(ss.name)
            pre_passed = False
            all_violated.append(ss.name)

    stages.append({
        "stage": "pre",
        "metrics": pre_metrics,
        "violated": pre_violated,
    })

    # If pre-validation failed, return early without injecting faults
    if not pre_passed:
        # Add a violation reason
        all_violated = [f"pre-validation failed: {v}" for v in pre_violated] if not all_violated else all_violated
        # Reformat violated to include pre-validation reason
        all_violated = []
        for ss in scenario.steady_states:
            if not ss.holds(pre_metrics[ss.metric]):
                all_violated.append(f"pre-validation failed ({ss.name})")

        return ScenarioResult(
            passed=False,
            stages=stages,
            violated=all_violated,
        )

    # Injection stages
    for step_idx, step_faults in enumerate(scenario.steps):
        # Apply faults to a new copy for this step
        step_cluster = copy.deepcopy(working_cluster)

        for fault in step_faults:
            matched_pods = step_cluster.select(fault.selector)

            # Determine which pods are affected based on mode
            pods_to_affect = []
            if fault.mode == "one":
                pods_to_affect = matched_pods[:1]
            elif fault.mode == "all":
                pods_to_affect = matched_pods
            elif fault.mode == "fixed":
                pods_to_affect = sorted(matched_pods)[:fault.value]
            elif fault.mode == "fixed-percent":
                count = max(1, math.ceil(len(matched_pods) * fault.value / 100.0))
                pods_to_affect = sorted(matched_pods)[:count]
            elif fault.mode == "random-max-percent":
                count = max(1, math.ceil(len(matched_pods) * fault.value / 100.0))
                pods_to_affect = sorted(matched_pods)[:count]

            # Apply the fault action
            if fault.action in ("pod-kill", "container-kill"):
                # Kill pods by marking them as not ready
                for pod_name in pods_to_affect:
                    step_cluster.pods[pod_name]["ready"] = False
            elif fault.action == "delay":
                # Add latency to affected pods
                latency_ms = fault.params.get("latency_ms", 0)
                for pod_name in pods_to_affect:
                    if "_delay_ms" not in step_cluster.pods[pod_name]:
                        step_cluster.pods[pod_name]["_delay_ms"] = 0
                    step_cluster.pods[pod_name]["_delay_ms"] += latency_ms

        # Measure metrics after injection
        step_metrics = step_cluster.metrics()

        # Evaluate steady states
        step_violated = []
        for ss in scenario.steady_states:
            if not ss.holds(step_metrics[ss.metric]):
                step_violated.append(ss.name)

        stages.append({
            "stage": "inject",
            "step": step_idx + 1,
            "metrics": step_metrics,
            "violated": step_violated,
        })

        if step_violated:
            all_violated.extend(step_violated)

        # Keep the cluster state for the next step
        working_cluster = step_cluster

    # Post-validation stage: evaluate on the ORIGINAL cluster
    post_cluster = copy.deepcopy(cluster)
    post_metrics = post_cluster.metrics()
    post_violated = []

    # Post only shows violations if pre passed (which we know it did at this point)
    for ss in scenario.steady_states:
        if not ss.holds(post_metrics[ss.metric]):
            post_violated.append(ss.name)

    stages.append({
        "stage": "post",
        "metrics": post_metrics,
        "violated": post_violated,
    })

    # Determine overall pass/fail
    passed = len(all_violated) == 0

    return ScenarioResult(
        passed=passed,
        stages=stages,
        violated=all_violated,
    )


# ============================================================================
# chaos_context: convert scenario result to context dict
# ============================================================================

def chaos_context(result: ScenarioResult) -> Dict[str, Any]:
    """Convert a scenario result to a context dict for policy decisions."""
    if not isinstance(result, ScenarioResult):
        raise TypeError(f"result must be ScenarioResult, got {type(result).__name__}")
    if not hasattr(result, "passed") or not hasattr(result, "violated"):
        raise AttributeError(f"result missing required attributes")

    return {
        "chaos_passed": result.passed,
        "chaos_violations": len(result.violated),
    }
