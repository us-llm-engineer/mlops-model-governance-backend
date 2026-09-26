"""
R1.2-S3: Lineage DAG (Persistence/Graph-Consistency)

Test suite for C7: Lineage graph: data_version → training_run_id → model_version →
eval_result → promotion_decision. Each node content-addressed (immutable). Querying
lineage traces back to exact data/config/eval. Retroactive config modification detected.

Dimension: persistence/graph-consistency cases
Mutation targets:
  - Edge not recorded in graph (mutation: skip edge insertion)
  - Node not content-addressed (mutation: store by mutable reference)
  - Retroactive modification not detected (mutation: remove hash validation)
  - Lineage path doesn't include required node (mutation: skip node)
"""

import pytest
import json
import hashlib
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class LineageNode:
    """Immutable node in lineage DAG."""
    node_id: str
    node_type: str  # "data", "training_run", "model", "eval", "promotion"
    content_hash: str  # SHA256 of node content
    metadata: Dict
    created_at: float = field(default_factory=time.time)


@dataclass
class LineageEdge:
    """Edge in lineage DAG."""
    from_node_id: str
    to_node_id: str
    edge_type: str  # "produces", "trains_on", "evaluates", "promotes"
    created_at: float = field(default_factory=time.time)


class LineageDAG:
    """Directed acyclic graph for pipeline lineage (immutable, content-addressed)."""

    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self.node_index: Dict[str, str] = {}  # content_hash → node_id (for deduplication)

    @staticmethod
    def compute_content_hash(content: Dict) -> str:
        """Compute SHA256 of content."""
        canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def add_node(self, node_type: str, content: Dict) -> str:
        """Add immutable node. Returns node_id."""
        content_hash = self.compute_content_hash(content)

        # Check if node already exists (content-addressed deduplication)
        if content_hash in self.node_index:
            return self.node_index[content_hash]

        node_id = f"{node_type}-{content_hash[:8]}"
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            content_hash=content_hash,
            metadata=content
        )

        self.nodes[node_id] = node
        self.node_index[content_hash] = node_id

        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str, edge_type: str):
        """Add edge in DAG. Validates nodes exist."""
        if from_node_id not in self.nodes:
            raise ValueError(f"From node not found: {from_node_id}")
        if to_node_id not in self.nodes:
            raise ValueError(f"To node not found: {to_node_id}")

        # Check for cycles (simplified: no reverse edges allowed)
        # In real impl, would do full cycle detection

        edge = LineageEdge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_type=edge_type
        )
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def trace_lineage(self, to_node_id: str) -> List[Tuple[str, str]]:
        """
        Trace lineage backwards from node to sources.
        Returns list of (node_id, node_type) in reverse order.
        """
        if to_node_id not in self.nodes:
            return []

        lineage = []
        visited = set()

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            lineage.append((node_id, node.node_type))

            # Find edges pointing to this node
            for edge in self.edges:
                if edge.to_node_id == node_id:
                    visit(edge.from_node_id)

        visit(to_node_id)
        return lineage

    def verify_node_immutability(self, node_id: str, expected_content: Dict) -> bool:
        """Verify node content hasn't changed (detect tampering)."""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        expected_hash = self.compute_content_hash(expected_content)

        return node.content_hash == expected_hash

    def get_edges_to_node(self, to_node_id: str) -> List[LineageEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.to_node_id == to_node_id]

    def get_edges_from_node(self, from_node_id: str) -> List[LineageEdge]:
        """Get all edges leaving a node."""
        return [e for e in self.edges if e.from_node_id == from_node_id]


class TestLineageNodeCreation:
    """Lineage node creation cases."""

    def test_create_data_node(self):
        """Case 1: Create data node with immutable hash."""
        dag = LineageDAG()
        data_content = {"dataset": "fraud_2024_q3", "version": "20240915", "rows": 1000000}

        node_id = dag.add_node("data", data_content)
        node = dag.get_node(node_id)

        assert node is not None
        assert node.node_type == "data"
        assert node.content_hash == LineageDAG.compute_content_hash(data_content)

    def test_create_training_run_node(self):
        """Case 2: Create training run node."""
        dag = LineageDAG()
        training_content = {
            "run_id": "run-20240915-001",
            "model_type": "xgboost",
            "hyperparams": {"max_depth": 8, "learning_rate": 0.1},
            "config_hash": "a" * 64
        }

        node_id = dag.add_node("training_run", training_content)
        node = dag.get_node(node_id)

        assert node.node_type == "training_run"
        assert node.metadata["run_id"] == "run-20240915-001"

    def test_identical_content_produces_same_node_id(self):
        """Case 3: Identical content is deduplicated (same content hash)."""
        dag = LineageDAG()
        content = {"model": "fraud", "version": "v2"}

        node_id_1 = dag.add_node("model", content)
        node_id_2 = dag.add_node("model", content)

        assert node_id_1 == node_id_2


class TestLineageEdgesAndConnectivity:
    """Lineage edge creation and connectivity cases."""

    def test_add_edge_data_to_training(self):
        """Case 4: Add edge from data to training run."""
        dag = LineageDAG()
        data_id = dag.add_node("data", {"dataset": "d1"})
        train_id = dag.add_node("training_run", {"run": "r1"})

        dag.add_edge(data_id, train_id, "trains_on")

        edges = dag.get_edges_from_node(data_id)
        assert len(edges) == 1
        assert edges[0].to_node_id == train_id
        assert edges[0].edge_type == "trains_on"

    def test_edge_requires_both_nodes_exist(self):
        """Case 5: Cannot add edge if nodes don't exist."""
        dag = LineageDAG()
        data_id = dag.add_node("data", {"d": "1"})

        with pytest.raises(ValueError, match="To node not found"):
            dag.add_edge(data_id, "nonexistent-node", "trains_on")

    def test_full_lineage_chain(self):
        """Case 6: Create and traverse full lineage chain."""
        dag = LineageDAG()

        # Build chain: data → training → model → eval → promotion
        data_id = dag.add_node("data", {"version": "v1"})
        train_id = dag.add_node("training_run", {"run": "t1"})
        model_id = dag.add_node("model", {"name": "fraud-v1"})
        eval_id = dag.add_node("eval", {"metrics": {"auc": 0.95}})
        promo_id = dag.add_node("promotion", {"env": "prod"})

        dag.add_edge(data_id, train_id, "trains_on")
        dag.add_edge(train_id, model_id, "produces")
        dag.add_edge(model_id, eval_id, "evaluates")
        dag.add_edge(eval_id, promo_id, "promotes")

        lineage = dag.trace_lineage(promo_id)

        # Should trace back through all nodes
        node_ids = [n[0] for n in lineage]
        assert promo_id in node_ids
        assert eval_id in node_ids
        assert model_id in node_ids
        assert train_id in node_ids
        assert data_id in node_ids


class TestLineageImmutability:
    """Immutability and tampering detection cases."""

    def test_node_immutability_verification_passes(self):
        """Case 7: Immutability check passes for unchanged content."""
        dag = LineageDAG()
        content = {"model": "test", "version": "1"}
        node_id = dag.add_node("model", content)

        is_valid = dag.verify_node_immutability(node_id, content)
        assert is_valid is True

    def test_retroactive_modification_detected(self):
        """Case 8: Retroactive modification (change in metadata) detected."""
        dag = LineageDAG()
        original_content = {"run": "r1", "loss": 0.05}
        node_id = dag.add_node("training_run", original_content)

        # Try to verify with modified content
        modified_content = {"run": "r1", "loss": 0.01}  # Changed loss

        is_valid = dag.verify_node_immutability(node_id, modified_content)
        assert is_valid is False

    def test_lineage_traces_to_original_data(self):
        """Case 9: Lineage always traces to original data version."""
        dag = LineageDAG()
        data_v1 = dag.add_node("data", {"version": "v1", "hash": "abc"})
        train = dag.add_node("training_run", {"run": "t1"})
        model = dag.add_node("model", {"name": "m1"})

        dag.add_edge(data_v1, train, "trains_on")
        dag.add_edge(train, model, "produces")

        # Trace from model should reach data_v1
        lineage = dag.trace_lineage(model)
        node_types = [n[1] for n in lineage]

        assert "data" in node_types
        assert lineage[-1][0] == data_v1  # Should reach data node


class TestLineageQueries:
    """Lineage query cases."""

    def test_query_edges_to_node(self):
        """Case 10: Query incoming edges to a node."""
        dag = LineageDAG()
        data = dag.add_node("data", {"d": "1"})
        train = dag.add_node("training_run", {"t": "1"})
        model = dag.add_node("model", {"m": "1"})

        dag.add_edge(data, train, "trains_on")
        dag.add_edge(train, model, "produces")

        incoming_to_model = dag.get_edges_to_node(model)

        assert len(incoming_to_model) == 1
        assert incoming_to_model[0].from_node_id == train
