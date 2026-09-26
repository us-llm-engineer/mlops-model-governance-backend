"""
C4 -- Persistence Round-Trip and Lineage Consistency.

Claim C4: A persisted pipeline config can be read back with full fidelity
(including microsecond timestamp precision), lineage node IDs are
deterministic and stable across serialization, and audit trails preserve
their insertion order through a newline-delimited JSON round-trip.

Source test suite: R1.1-S4.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict

__all__ = [
    "PipelineConfig",
    "LineageNode",
    "PersistenceLayer",
    "LineageTracker",
    "AuditStore",
]


@dataclass
class PipelineConfig:
    """An immutable pipeline configuration."""

    name: str
    data_version: str
    code_commit: str
    container_image: str
    config_hash: str
    created_at: str
    created_by: str


@dataclass
class LineageNode:
    """A content-addressed lineage node."""

    node_id: str
    node_type: str
    content_hash: str
    timestamp: str
    metadata: Dict[str, Any]


class PersistenceLayer:
    """Store and retrieve pipeline configs with full fidelity."""

    def __init__(self):
        self.storage: Dict[str, str] = {}

    def serialize_config(self, config: PipelineConfig) -> str:
        """Serialize a config to JSON, preserving timestamps verbatim."""
        config_dict = asdict(config)
        config_dict["created_at"] = config.created_at
        return json.dumps(config_dict, sort_keys=True, indent=2)

    def deserialize_config(self, json_str: str) -> PipelineConfig:
        """Deserialize a config from JSON, validating required fields.

        Raises:
            ValueError: If any required field is missing.
        """
        data = json.loads(json_str)

        required = {"name", "data_version", "code_commit", "container_image",
                    "config_hash", "created_at", "created_by"}
        if not required.issubset(data.keys()):
            missing = required - set(data.keys())
            raise ValueError(f"Missing required fields: {missing}")

        return PipelineConfig(
            name=data["name"],
            data_version=data["data_version"],
            code_commit=data["code_commit"],
            container_image=data["container_image"],
            config_hash=data["config_hash"],
            created_at=data["created_at"],
            created_by=data["created_by"],
        )

    def store_config(self, config: PipelineConfig) -> str:
        """Store a config, returning its deterministic identifier."""
        config_id = hashlib.sha256(
            f"{config.name}:{config.created_at}".encode()
        ).hexdigest()

        json_str = self.serialize_config(config)
        self.storage[config_id] = json_str
        return config_id

    def retrieve_config(self, config_id: str) -> PipelineConfig:
        """Retrieve a config by identifier.

        Raises:
            KeyError: If the identifier is unknown.
        """
        if config_id not in self.storage:
            raise KeyError(f"Config not found: {config_id}")

        json_str = self.storage[config_id]
        return self.deserialize_config(json_str)


class LineageTracker:
    """Track lineage with content-addressed, deterministic nodes."""

    def __init__(self):
        self.lineage: Dict[str, LineageNode] = {}
        self.graph_edges: list = []

    def add_node(self, node_type: str, content_hash: str, metadata: Dict[str, Any]) -> str:
        """Add a lineage node and return its deterministic node identifier."""
        node_id = hashlib.sha256(
            f"{node_type}:{content_hash}:{metadata.get('timestamp', '')}".encode()
        ).hexdigest()

        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            content_hash=content_hash,
            timestamp=metadata.get("timestamp", datetime.utcnow().isoformat()),
            metadata=metadata,
        )
        self.lineage[node_id] = node
        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a directed edge between two existing nodes.

        Raises:
            ValueError: If either endpoint is not present in the lineage.
        """
        if from_node_id not in self.lineage or to_node_id not in self.lineage:
            raise ValueError("One or both nodes not found in lineage")

        self.graph_edges.append((from_node_id, to_node_id))

    def serialize_lineage(self) -> str:
        """Serialize the lineage graph to JSON."""
        graph_data = {
            "nodes": {
                node_id: {
                    "type": node.node_type,
                    "content_hash": node.content_hash,
                    "timestamp": node.timestamp,
                    "metadata": node.metadata,
                }
                for node_id, node in self.lineage.items()
            },
            "edges": [
                {"from": f_id, "to": t_id}
                for f_id, t_id in self.graph_edges
            ],
        }
        return json.dumps(graph_data, sort_keys=True, indent=2)

    def deserialize_lineage(self, json_str: str) -> None:
        """Reconstruct lineage nodes and edges from a serialized graph."""
        graph_data = json.loads(json_str)

        for node_id, node_data in graph_data.get("nodes", {}).items():
            node = LineageNode(
                node_id=node_id,
                node_type=node_data["type"],
                content_hash=node_data["content_hash"],
                timestamp=node_data["timestamp"],
                metadata=node_data["metadata"],
            )
            self.lineage[node_id] = node

        for edge in graph_data.get("edges", []):
            self.graph_edges.append((edge["from"], edge["to"]))


class AuditStore:
    """Store audit events with guaranteed insertion ordering."""

    def __init__(self):
        self.events: list = []

    def append_event(self, event: Dict[str, Any]) -> None:
        """Append an audit event, preserving insertion order."""
        self.events.append(event)

    def serialize_audit(self) -> str:
        """Serialize events as ordered newline-delimited JSON."""
        lines = [json.dumps(event) for event in self.events]
        return "\n".join(lines)

    def deserialize_audit(self, jsonl_str: str) -> None:
        """Deserialize newline-delimited JSON, preserving order."""
        self.events = [
            json.loads(line)
            for line in jsonl_str.strip().split("\n")
            if line
        ]
