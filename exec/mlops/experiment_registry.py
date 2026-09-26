"""
X2.2/X2.3 -- Experiment Registry (supports C53-C60 surfaces).

The registry ties the pieces of the X2 round together: model-version
registration against the lineage graph, feature-version lookup from a trained
model, statistical and quality-gate comparisons, and per-attempt artifact
inspection from the retry subsystem. Every registration decision (allow or
deny) is recorded exactly once in the caller-supplied audit store, and the
lineage graph is only touched on an authorized, fully-resolvable registration.

Constructor compatibility (deliberate deviation from the API contract)
----------------------------------------------------------------------
the API contract nominally lists the parameters as
``ExperimentRegistry(rbac, audit, lineage, retry_mgr)``. The frozen tests are
authoritative and call the constructor BOTH ways::

    ExperimentRegistry(audit, lineage)                      # 2 positional
    ExperimentRegistry(rbac=None, audit=aid, lineage=lg)    # keywords

so this module defines::

    def __init__(self, audit=None, lineage=None, rbac=None, retry_mgr=None)

which is the only ordering that satisfies both call shapes. ``rbac`` and
``retry_mgr`` are accepted and stored for contract compatibility; the
authorization decision itself is the role/scope check below, audited directly
against ``self.audit``.

Stdlib only: hashlib, typing.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from .audit import now_iso
from .features import FeaturePermissionError
from .lineage import LineageGraph
from .rbac import Role

__all__ = ["ExperimentRegistry"]

# Roles permitted to register a model version (mirrors FeatureRegistry).
_ALLOWED_ROLES = (Role.DEPLOYER, Role.APPROVER)

# Edge type linking a feature_version node to the model_version consuming it.
_MODEL_EDGE = "depends"


def _default_namespace(run_id: str) -> str:
    """Return the run's namespace: the prefix before the first "/"."""
    return run_id.split("/", 1)[0] if "/" in run_id else "default"


class ExperimentRegistry:
    """Model-version registration, lineage lookup and comparison pass-throughs.

    Parameters are stored under their public names (``audit``, ``lineage``,
    ``rbac``, ``retry_mgr``) and the constructor order is the test-authoritative
    ``(audit, lineage, rbac, retry_mgr)`` rather than the contract's nominal
    ``(rbac, audit, lineage, retry_mgr)``.

    Enforcement is opt-in: a decision_point is used only when explicitly set by
    the caller. No decision point means no policy check.

    With a decision_point set, model registration is checked (after RBAC) before
    any lineage mutation: the audit chain records an RBAC "allow" entry followed
    by (if policy denies) a policy "deny" entry, and lineage remains unchanged.
    """

    def __init__(
        self,
        audit: Any = None,
        lineage: Any = None,
        rbac: Any = None,
        retry_mgr: Any = None,
        decision_point: Any = None,
    ) -> None:
        self.audit = audit
        self.lineage = lineage
        self.rbac = rbac
        self.retry_mgr = retry_mgr
        self.decision_point = decision_point

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register_model_version(
        self,
        principal: Any,
        run_id: str,
        version_tag: str,
        feature_version_ids: Any = (),
        artifact_hash: str = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """Register a model version, returning its content-addressed id.

        Audit order: RBAC check first, then policy (if decision_point is set),
        then lineage mutation.

        An unauthorized principal produces exactly one ``deny`` audit entry and
        the lineage graph is never touched. An authorized registration produces
        exactly one ``allow`` audit entry BEFORE feature lookup.

        If a decision_point is set, it is consulted after the RBAC ``allow`` entry
        and before any lineage mutation: a policy denial logs a policy ``deny`` entry
        and leaves lineage unchanged.

        Args:
            artifact_hash: Optional artifact content hash; when provided, it is
                          included in the model version id so different content
                          produces different ids.
            context: Optional policy context dict (unused if no decision_point).
        """
        namespace = _default_namespace(run_id)

        if principal.role not in _ALLOWED_ROLES or principal.scope != namespace:
            self.audit.append(
                principal.name, "experiment.register", run_id, "deny"
            )
            raise FeaturePermissionError(
                "principal %s may not register models in namespace %s"
                % (principal.name, namespace)
            )

        self.audit.append(principal.name, "experiment.register", run_id, "allow")

        # Enforce policy BEFORE touching lineage (fail-closed)
        if self.decision_point is not None:
            policy_context = context or {}
            self.decision_point.enforce(
                "register", policy_context, actor=principal.name
            )

        # Compute model version id including artifact_hash when provided
        if artifact_hash:
            hash_input = run_id + version_tag + artifact_hash
        else:
            hash_input = run_id + version_tag
        mv_id = "mv_" + hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        graph = self.lineage
        if not isinstance(graph, LineageGraph):
            if feature_version_ids:
                raise KeyError("no lineage graph to resolve feature versions")
            return mv_id

        # Resolve every feature version BEFORE mutating the graph, so an
        # unknown id leaves the graph unchanged.
        feature_node_ids: List[str] = []
        for fv in feature_version_ids:
            node = self._find_node_with_value("feature_version", fv)
            if node is None:
                raise KeyError("unknown feature version: %s" % (fv,))
            feature_node_ids.append(node.node_id)

        model_node = self._find_node_with_value("model_version", mv_id)
        if model_node is None:
            model_node_id = graph.add_node(
                "model_version",
                {
                    "model_version_id": mv_id,
                    "run_id": str(run_id),
                    "version_tag": str(version_tag),
                },
                now_iso(),
            )
        else:
            model_node_id = model_node.node_id

        seen = set()
        for node_id in feature_node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            self._add_edge_once(node_id, model_node_id, _MODEL_EDGE)

        return mv_id

    # ------------------------------------------------------------------ #
    # lineage read access
    # ------------------------------------------------------------------ #
    def feature_versions_for_model(self, model_version_id: str) -> List[str]:
        """Return feature version ids feeding the model, in trace order.

        Traces the lineage backward from the matching ``model_version`` node,
        keeps only ``feature_version`` nodes, and maps each to its
        ``version_id`` property, deduped while preserving trace order. Returns
        ``[]`` when the model node is absent or no features are linked.
        """
        graph = self.lineage
        if not isinstance(graph, LineageGraph):
            return []

        result: List[str] = []
        seen = set()
        for node in list(graph.nodes.values()):
            if node.node_type != "model_version":
                continue
            if model_version_id not in node.properties.values():
                continue
            for node_id in graph.trace_lineage_backward(node.node_id):
                traced = graph.nodes.get(node_id)
                if traced is None or traced.node_type != "feature_version":
                    continue
                version_id = traced.properties.get("version_id")
                if version_id is None or version_id in seen:
                    continue
                seen.add(version_id)
                result.append(version_id)
        return result

    # ------------------------------------------------------------------ #
    # comparison pass-throughs
    # ------------------------------------------------------------------ #
    def compare_run_to_baseline(
        self,
        judge: Any,
        candidate_run_metric_samples: Any,
        baseline_run_metric_samples: Any,
    ) -> dict:
        """Return exactly ``judge.judge(candidate, baseline)`` (pure pass-through)."""
        return judge.judge(
            candidate_run_metric_samples, baseline_run_metric_samples
        )

    def evaluate_final_accuracy(
        self, gate: Any, model_family: str, run_final_accuracy: float
    ) -> Any:
        """Return exactly ``gate.evaluate(model_family, accuracy)`` (pure pass-through)."""
        return gate.evaluate(model_family, run_final_accuracy)

    def compare_attempts(
        self, retry_mgr: Any, run_id: str
    ) -> Dict[int, List[dict]]:
        """Group a run's artifacts by attempt, never merging across attempts.

        Returns ``{attempt_number: [{"name": ..., "hash": ...}, ...]}`` built
        from ``retry_mgr.artifact_registry.list_artifacts_for_run(run_id)``.
        An absent run yields ``{}``.
        """
        grouped: Dict[int, List[dict]] = {}
        for item in retry_mgr.artifact_registry.list_artifacts_for_run(run_id):
            grouped.setdefault(item["attempt"], []).append(
                {"name": item["name"], "hash": item["hash"]}
            )
        return grouped

    # ------------------------------------------------------------------ #
    # lineage helpers
    # ------------------------------------------------------------------ #
    def _find_node_with_value(self, node_type: str, value: Any):
        """Return the first node of ``node_type`` holding ``value`` in its properties."""
        graph = self.lineage
        if not isinstance(graph, LineageGraph):
            return None
        for node in graph.nodes.values():
            if node.node_type == node_type and value in node.properties.values():
                return node
        return None

    def _add_edge_once(self, from_id: str, to_id: str, edge_type: str) -> None:
        """Add a directed edge unless an identical one already exists."""
        graph = self.lineage
        for a, b, t in graph.edges:
            if a == from_id and b == to_id and t == edge_type:
                return
        graph.add_edge(from_id, to_id, edge_type)