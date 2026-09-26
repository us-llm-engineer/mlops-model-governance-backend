"""
X1.1 -- Content-Addressed Feature Registry (C37, C38, C39, C40).

A feature definition is published against a namespace by a principal holding an
appropriate role. Every published version is content-addressed, immutable, and
recorded in a tamper-evident audit chain. When a lineage graph is supplied, each
publish links its source dataset to the feature version, and models can be
linked to the feature versions they consume so lineage can be traced backward
from a model.

Stdlib only: json, re, hashlib, copy.
"""

import copy
import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from .audit import ChainedAuditStore, now_iso
from .lineage import LineageGraph
from .rbac import Principal, Role

__all__ = [
    "FeatureError",
    "MutableReferenceError",
    "ImmutableVersionError",
    "FeaturePermissionError",
    "ContractViolation",
    "FeatureRegistry",
]

_SOURCE_RE = re.compile(r"^dataset:.+@sha256:[0-9a-f]{64}$")

_ALLOWED_ROLES = (Role.DEPLOYER, Role.APPROVER)

# Edge type used to link a feature version to the model version consuming it.
_MODEL_EDGE = "depends"


class FeatureError(Exception):
    """Base class for every feature-registry error."""


class MutableReferenceError(FeatureError):
    """Raised when a source reference is not an immutable digest."""


class ImmutableVersionError(FeatureError):
    """Raised when an attempt is made to mutate a published version."""


class FeaturePermissionError(FeatureError):
    """Raised when a principal may not publish into a namespace."""


class ContractViolation(FeatureError):
    """Raised by the feature store when a row violates a data contract."""


class FeatureRegistry:
    """Content-addressed, immutable feature-definition registry."""

    def __init__(
        self,
        audit: ChainedAuditStore,
        lineage: Optional[LineageGraph] = None,
    ) -> None:
        self._audit = audit
        self._lineage = lineage
        self._defs: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[tuple, List[str]] = {}

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def publish(
        self, principal: Principal, namespace: str, definition: dict
    ) -> str:
        """Publish a feature definition, returning its content-addressed id.

        The permission check runs first: an unauthorized principal is denied and
        audited without the definition ever being validated or stored.
        """
        resource = "%s/%s" % (namespace, definition.get("name"))

        if principal.scope != namespace or principal.role not in _ALLOWED_ROLES:
            self._audit.append(
                principal.name, "feature.publish", resource, "deny"
            )
            raise FeaturePermissionError(
                "principal %s may not publish into %s" % (principal.name, namespace)
            )

        source = definition.get("source")
        if not isinstance(source, str) or not _SOURCE_RE.match(source):
            self._audit.append(
                principal.name, "feature.publish", resource, "deny"
            )
            raise MutableReferenceError(
                "source reference is not an immutable sha256 digest: %r" % (source,)
            )

        version_id = "fv_" + hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()[:16]

        self._defs[version_id] = copy.deepcopy(definition)

        key = (namespace, definition.get("name"))
        ordered = self._versions.setdefault(key, [])
        if version_id not in ordered:
            ordered.append(version_id)

        self._audit.append(principal.name, "feature.publish", resource, "allow")

        self._update_lineage(version_id, source)

        return version_id

    # ------------------------------------------------------------------ #
    # read access
    # ------------------------------------------------------------------ #
    def get(self, version_id: str) -> dict:
        """Return a fresh deep copy of the stored definition for ``version_id``."""
        if version_id not in self._defs:
            raise FeatureError("unknown feature version: %s" % version_id)
        return copy.deepcopy(self._defs[version_id])

    def versions(self, namespace: str, name: str) -> List[str]:
        """Return a copy of the unique version ids in first-publish order."""
        return list(self._versions.get((namespace, name), []))

    def latest(self, namespace: str, name: str) -> str:
        """Return the newest version id for ``namespace/name``."""
        ordered = self._versions.get((namespace, name))
        if not ordered:
            raise FeatureError("no versions for %s/%s" % (namespace, name))
        return ordered[-1]

    # ------------------------------------------------------------------ #
    # immutability
    # ------------------------------------------------------------------ #
    def mutate(self, version_id: str, **changes: Any) -> None:
        """Always refuse: published versions are immutable. Audited each time."""
        self._audit.append("system", "feature.mutate", version_id, "deny")
        raise ImmutableVersionError(
            "feature version %s is immutable" % version_id
        )

    # ------------------------------------------------------------------ #
    # lineage
    # ------------------------------------------------------------------ #
    def _has_lineage(self) -> bool:
        return isinstance(self._lineage, LineageGraph)

    def _find_node(self, node_type: str, **properties: Any):
        graph = self._lineage
        for node in graph.nodes.values():
            if node.node_type != node_type:
                continue
            if all(node.properties.get(k) == v for k, v in properties.items()):
                return node
        return None

    def _find_node_with_value(self, node_type: str, value: Any):
        graph = self._lineage
        for node in graph.nodes.values():
            if node.node_type == node_type and value in node.properties.values():
                return node
        return None

    def _update_lineage(self, version_id: str, source: str) -> None:
        if not self._has_lineage():
            return
        graph = self._lineage

        dataset = self._find_node("dataset", ref=source)
        if dataset is None:
            dataset_id = graph.add_node("dataset", {"ref": source}, now_iso())
        else:
            dataset_id = dataset.node_id

        feature_version = self._find_node_with_value("feature_version", version_id)
        if feature_version is None:
            feature_version_id = graph.add_node(
                "feature_version",
                {"version_id": version_id, "source": source},
                now_iso(),
            )
        else:
            feature_version_id = feature_version.node_id

        self._add_edge_once(dataset_id, feature_version_id, "source")

    def link_model(self, version_ids: List[str], model_id: str) -> None:
        """Link each feature version to a model version. No-op without lineage."""
        if not self._has_lineage():
            return
        graph = self._lineage

        model = self._find_node_with_value("model_version", model_id)
        if model is None:
            model_node_id = graph.add_node(
                "model_version", {"model_id": model_id}, now_iso()
            )
        else:
            model_node_id = model.node_id

        for version_id in version_ids:
            feature_version = self._find_node_with_value(
                "feature_version", version_id
            )
            if feature_version is None:
                continue
            self._add_edge_once(feature_version.node_id, model_node_id, _MODEL_EDGE)

    def feature_versions_for_model(self, model_id: str) -> List[str]:
        """Return feature version ids feeding the model, in trace order."""
        if not self._has_lineage():
            return []
        graph = self._lineage

        result: List[str] = []
        seen = set()
        for node in list(graph.nodes.values()):
            if node.node_type != "model_version":
                continue
            if model_id not in node.properties.values():
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

    def _add_edge_once(self, from_id: str, to_id: str, edge_type: str) -> None:
        graph = self._lineage
        for a, b, t in graph.edges:
            if a == from_id and b == to_id and t == edge_type:
                return
        graph.add_edge(from_id, to_id, edge_type)
