"""
Claim C7: Lineage Graph Persistence & Tamper Detection.

A lineage graph links data_version -> training_run_id -> model_version ->
eval_result -> promotion_decision. Each node is content-addressed (its id is
the SHA256 of its serialized content) and therefore immutable; querying a
model's lineage traces back to the exact data, training config, and evaluation
that justified it. Retroactively modifying a node is detected as a hash
mismatch, and edges are validated so no orphaned nodes can enter the graph.

Source test suite: R1.2-S3 (tests/R1_2_S3.py).
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LineageGraphNode:
    """Content-addressed lineage node."""

    node_id: str
    node_type: str  # data, training, model, evaluation, promotion
    content_hash: str  # SHA256 of immutable content
    timestamp: str
    properties: Dict[str, str]

    def is_immutable(self) -> bool:
        """Return True if the node's content still matches its stored hash.

        Claim C75: Returns True for untouched node, False after content tamper.

        Detects tampering even if both properties AND content_hash are rewritten
        together by comparing against the node_id (which is the original content
        hash and should never change).
        """
        # Recompute hash using the same formula as LineageGraph.add_node
        content = json.dumps(
            {"type": self.node_type, "properties": self.properties, "timestamp": self.timestamp},
            sort_keys=True
        )
        computed_hash = hashlib.sha256(content.encode()).hexdigest()
        # Check against both content_hash and node_id (should be the same for untampered)
        return computed_hash == self.content_hash and computed_hash == self.node_id


# Backwards-compatible alias for the node dataclass.
LineageNode = LineageGraphNode


class LineageGraph:
    """Directed acyclic graph of immutable lineage nodes."""

    def __init__(self):
        self.nodes: Dict[str, LineageGraphNode] = {}
        self.edges: List[tuple] = []  # (from_node_id, to_node_id, edge_type)
        self.immutable_content: Dict[str, str] = {}  # node_id -> original content

    def add_node(
        self,
        node_type: str,
        properties: Dict[str, str],
        timestamp: str
    ) -> str:
        """Add an immutable node; return its deterministic content-addressed id."""
        # Compute deterministic node ID: hash(type + properties + timestamp)
        content = json.dumps(
            {"type": node_type, "properties": properties, "timestamp": timestamp},
            sort_keys=True
        )
        node_id = hashlib.sha256(content.encode()).hexdigest()
        content_hash = node_id  # Node ID is its content hash

        # Store immutable content
        self.immutable_content[node_id] = content

        node = LineageGraphNode(
            node_id=node_id,
            node_type=node_type,
            content_hash=content_hash,
            timestamp=timestamp,
            properties=properties
        )

        self.nodes[node_id] = node
        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str, edge_type: str = "depends"):
        """Add a directed edge; validate that both endpoint nodes exist."""
        # Validate both nodes exist
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node not found: {from_node_id}")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node not found: {to_node_id}")

        self.edges.append((from_node_id, to_node_id, edge_type))

    def trace_lineage_backward(self, node_id: str) -> List[str]:
        """Trace backward from a node to all ancestors (including itself)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")

        visited = set()
        trace = []

        def dfs(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            trace.append(current_id)

            # Find all incoming edges
            for from_id, to_id, _ in self.edges:
                if to_id == current_id:
                    dfs(from_id)

        dfs(node_id)
        return trace

    def verify_node_immutability(self, node_id: str) -> bool:
        """Recompute a node's hash; raise ValueError if its content was modified.

        Detects tampering even if both properties AND content_hash are rewritten
        together by comparing against the original node_id and immutable_content.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")

        node = self.nodes[node_id]

        # Recompute hash from stored content (same as add_node)
        content = json.dumps(
            {"type": node.node_type, "properties": node.properties, "timestamp": node.timestamp},
            sort_keys=True
        )
        computed_hash = hashlib.sha256(content.encode()).hexdigest()

        # Verify against both the node's content_hash and the node_id (original hash)
        if computed_hash != node.content_hash or computed_hash != node.node_id:
            raise ValueError(
                f"Node {node_id} content has been modified: "
                f"expected {node.node_id}, got {computed_hash}"
            )

        # Also verify against immutable_content if it exists
        if node_id in self.immutable_content:
            if content != self.immutable_content[node_id]:
                raise ValueError(
                    f"Node {node_id} content diverges from immutable_content"
                )

        return True

    def get_node_lineage_chain(self, node_id: str) -> Dict:
        """Return the full lineage chain (target, path, and node details)."""
        trace = self.trace_lineage_backward(node_id)

        chain = {
            "target_node": node_id,
            "lineage_path": trace,
            "nodes": {
                nid: {
                    "type": self.nodes[nid].node_type,
                    "content_hash": self.nodes[nid].content_hash,
                    "timestamp": self.nodes[nid].timestamp,
                    "properties": self.nodes[nid].properties,
                }
                for nid in trace
            }
        }

        return chain


class PipelineLineageBuilder:
    """Build a lineage graph for a pipeline run."""

    def __init__(self, lineage_graph: LineageGraph):
        self.graph = lineage_graph

    def build_lineage_for_run(
        self,
        run_id: str,
        data_version: str,
        data_hash: str,
        training_config: str,
        model_artifact_hash: str,
        eval_metrics: str,
        eval_hash: str,
        promotion_decision: str
    ) -> Dict[str, str]:
        """
        Build the complete lineage for a pipeline run.

        Nodes: data -> training -> model -> evaluation -> promotion.
        """
        node_ids = {}

        # Data node
        data_node_id = self.graph.add_node(
            "data",
            {"version": data_version, "hash": data_hash},
            "2024-09-24T00:00:00Z"
        )
        node_ids["data"] = data_node_id

        # Training node
        training_node_id = self.graph.add_node(
            "training",
            {"run_id": run_id, "config": training_config},
            "2024-09-24T00:01:00Z"
        )
        node_ids["training"] = training_node_id
        self.graph.add_edge(data_node_id, training_node_id, "input")

        # Model node
        model_node_id = self.graph.add_node(
            "model",
            {"artifact_hash": model_artifact_hash, "run_id": run_id},
            "2024-09-24T00:02:00Z"
        )
        node_ids["model"] = model_node_id
        self.graph.add_edge(training_node_id, model_node_id, "produces")

        # Evaluation node
        eval_node_id = self.graph.add_node(
            "evaluation",
            {"metrics": eval_metrics, "eval_hash": eval_hash},
            "2024-09-24T00:03:00Z"
        )
        node_ids["evaluation"] = eval_node_id
        self.graph.add_edge(model_node_id, eval_node_id, "evaluated_by")

        # Promotion node
        promotion_node_id = self.graph.add_node(
            "promotion",
            {"decision": promotion_decision, "run_id": run_id},
            "2024-09-24T00:04:00Z"
        )
        node_ids["promotion"] = promotion_node_id
        self.graph.add_edge(eval_node_id, promotion_node_id, "justifies")

        return node_ids
