"""F3.5 lineage_graph: frozen tests for exec/mlops/ext/lineage_graph.py.

Covers to_networkx / ancestors / blast_radius / has_cycle / shortest_path_len /
to_dot, all operating on the REAL exec/mlops/lineage.py::LineageGraph (nodes:
Dict[node_id -> LineageGraphNode], edges: List[(from_id, to_id, edge_type)] --
read directly from lineage.py, not guessed).

Attribute-name contract this suite fixes for the implementer (api-contract.md
section 5 names `node_type` and `edge_type` literally but leaves the
is_immutable() attribute name open -- this suite pins it to "immutable"):
  - to_networkx node attrs: "node_type" (str, == LineageGraphNode.node_type)
                              and "immutable" (bool, == node.is_immutable())
  - to_networkx edge attrs: "edge_type" (str, == the 3rd tuple element added
                              via LineageGraph.add_edge)

ancestors() cross-check note: nx.ancestors(g, node_id) excludes node_id itself
by definition, while LineageGraph.trace_lineage_backward(node_id) INCLUDES the
starting node (see lineage.py's dfs()). The two are therefore compared as
  ancestors(graph, node_id) == set(graph.trace_lineage_backward(node_id)) - {node_id}
which is the only self-consistent way to call them "the same set" -- this is
not a guess, it falls directly out of reading both methods' real behavior.

Cycle note: LineageGraph.add_edge (lineage.py) validates ONLY that both
endpoint nodes already exist in self.nodes -- it performs no reachability or
cycle check. A cyclic structure is therefore reachable through the ordinary
public add_node/add_edge API, and the has_cycle=True case below is built that
way (no monkeypatching or private access needed).

NOTE: like the frozen suites this file puts <repo>/exec first on sys.path, so
a mutation harness must copy tests/ AND exec/ into the same scratch root
(PYTHONPATH alone is NOT enough).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

import networkx as nx  # noqa: E402

from mlops.kernel import NotFound  # noqa: E402
from mlops.lineage import LineageGraph  # noqa: E402
from mlops.ext.lineage_graph import (  # noqa: E402
    ancestors,
    blast_radius,
    has_cycle,
    shortest_path_len,
    to_dot,
    to_networkx,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_graph():
    """5-node linear lineage chain with 4 edges of MIXED edge_type:

        data --input--> training --produces--> model
             --evaluated_by--> evaluation --justifies--> promotion

    Satisfies the >=4 nodes / >=3 edges / mixed edge_type requirement on its
    own (5 nodes, 4 edges, 4 distinct edge_type values).
    """
    g = LineageGraph()
    data_id = g.add_node("data", {"version": "v1", "hash": "h1"}, "2024-01-01T00:00:00Z")
    training_id = g.add_node("training", {"run_id": "r1", "config": "c1"}, "2024-01-01T00:01:00Z")
    model_id = g.add_node("model", {"artifact_hash": "a1", "run_id": "r1"}, "2024-01-01T00:02:00Z")
    eval_id = g.add_node("evaluation", {"metrics": "m1", "eval_hash": "e1"}, "2024-01-01T00:03:00Z")
    promo_id = g.add_node("promotion", {"decision": "promote", "run_id": "r1"}, "2024-01-01T00:04:00Z")

    g.add_edge(data_id, training_id, "input")
    g.add_edge(training_id, model_id, "produces")
    g.add_edge(model_id, eval_id, "evaluated_by")
    g.add_edge(eval_id, promo_id, "justifies")

    ids = {
        "data": data_id,
        "training": training_id,
        "model": model_id,
        "evaluation": eval_id,
        "promotion": promo_id,
    }
    return g, ids


@pytest.fixture
def diamond_graph():
    """Diamond DAG: A->B, A->C, B->D, C->D (used for blast_radius/ancestors)."""
    g = LineageGraph()
    a = g.add_node("data", {"name": "A"}, "2024-02-01T00:00:00Z")
    b = g.add_node("training", {"name": "B"}, "2024-02-01T00:01:00Z")
    c = g.add_node("training", {"name": "C"}, "2024-02-01T00:01:01Z")
    d = g.add_node("model", {"name": "D"}, "2024-02-01T00:02:00Z")

    g.add_edge(a, b, "input")
    g.add_edge(a, c, "input")
    g.add_edge(b, d, "produces")
    g.add_edge(c, d, "produces")

    ids = {"A": a, "B": b, "C": c, "D": d}
    return g, ids


@pytest.fixture
def cyclic_graph():
    """A -> B -> C -> A, built purely through the public add_node/add_edge
    API. LineageGraph.add_edge does not reject this (see module docstring)."""
    g = LineageGraph()
    a = g.add_node("data", {"name": "cyc-A"}, "2024-03-01T00:00:00Z")
    b = g.add_node("training", {"name": "cyc-B"}, "2024-03-01T00:01:00Z")
    c = g.add_node("model", {"name": "cyc-C"}, "2024-03-01T00:02:00Z")

    g.add_edge(a, b, "input")
    g.add_edge(b, c, "produces")
    g.add_edge(c, a, "loops_back")
    return g


def _expected_ancestors(g, node_id):
    """trace_lineage_backward includes node_id itself; nx.ancestors does not."""
    return set(g.trace_lineage_backward(node_id)) - {node_id}


# ---------------------------------------------------------------------------
# to_networkx
# ---------------------------------------------------------------------------

class TestToNetworkx:
    def test_returns_digraph_instance(self, chain_graph):
        g, _ids = chain_graph
        nxg = to_networkx(g)
        assert isinstance(nxg, nx.DiGraph)

    def test_node_and_edge_counts(self, chain_graph):
        g, _ids = chain_graph
        nxg = to_networkx(g)
        assert nxg.number_of_nodes() == len(g.nodes) == 5
        assert nxg.number_of_edges() == len(g.edges) == 4

    def test_node_ids_match_graph_nodes(self, chain_graph):
        g, _ids = chain_graph
        nxg = to_networkx(g)
        assert set(nxg.nodes) == set(g.nodes.keys())

    def test_node_type_and_immutability_attributes(self, chain_graph):
        g, _ids = chain_graph
        nxg = to_networkx(g)
        for node_id, node in g.nodes.items():
            assert nxg.nodes[node_id]["node_type"] == node.node_type
            assert nxg.nodes[node_id]["immutable"] is True
            assert nxg.nodes[node_id]["immutable"] == node.is_immutable()

    def test_immutability_attribute_false_after_tamper(self, chain_graph):
        g, ids = chain_graph
        # Tamper with a node's properties in place; content_hash/node_id stay
        # stale, so is_immutable() must flip to False and to_networkx must
        # reflect that (i.e. it must call is_immutable(), not cache True).
        g.nodes[ids["model"]].properties["artifact_hash"] = "TAMPERED"
        nxg = to_networkx(g)
        assert nxg.nodes[ids["model"]]["immutable"] is False
        assert nxg.nodes[ids["data"]]["immutable"] is True

    def test_edge_type_attributes_mixed(self, chain_graph):
        g, ids = chain_graph
        nxg = to_networkx(g)
        assert nxg.edges[ids["data"], ids["training"]]["edge_type"] == "input"
        assert nxg.edges[ids["training"], ids["model"]]["edge_type"] == "produces"
        assert nxg.edges[ids["model"], ids["evaluation"]]["edge_type"] == "evaluated_by"
        assert nxg.edges[ids["evaluation"], ids["promotion"]]["edge_type"] == "justifies"

    def test_edge_direction_not_reversed(self, chain_graph):
        g, ids = chain_graph
        nxg = to_networkx(g)
        assert nxg.has_edge(ids["data"], ids["training"])
        assert not nxg.has_edge(ids["training"], ids["data"])


# ---------------------------------------------------------------------------
# ancestors() cross-checked against LineageGraph.trace_lineage_backward
# ---------------------------------------------------------------------------

class TestAncestorsCrossCheck:
    def test_leaf_node_promotion(self, chain_graph):
        g, ids = chain_graph
        expected = _expected_ancestors(g, ids["promotion"])
        assert expected == {ids["data"], ids["training"], ids["model"], ids["evaluation"]}
        assert ancestors(g, ids["promotion"]) == expected

    def test_root_node_data_has_no_ancestors(self, chain_graph):
        g, ids = chain_graph
        expected = _expected_ancestors(g, ids["data"])
        assert expected == set()
        assert ancestors(g, ids["data"]) == expected

    def test_diamond_node_d_all_upstream(self, diamond_graph):
        g, ids = diamond_graph
        expected = _expected_ancestors(g, ids["D"])
        assert expected == {ids["A"], ids["B"], ids["C"]}
        assert ancestors(g, ids["D"]) == expected

    def test_diamond_node_b_only_root(self, diamond_graph):
        g, ids = diamond_graph
        expected = _expected_ancestors(g, ids["B"])
        assert expected == {ids["A"]}
        assert ancestors(g, ids["B"]) == expected


# ---------------------------------------------------------------------------
# blast_radius() -- new capability, no existing equivalent to cross-check
# ---------------------------------------------------------------------------

class TestBlastRadius:
    def test_blast_radius_diamond_root(self, diamond_graph):
        g, ids = diamond_graph
        assert blast_radius(g, ids["A"]) == {ids["B"], ids["C"], ids["D"]}

    def test_blast_radius_diamond_leaf_is_empty(self, diamond_graph):
        g, ids = diamond_graph
        assert blast_radius(g, ids["D"]) == set()

    def test_blast_radius_diamond_mid_node(self, diamond_graph):
        g, ids = diamond_graph
        assert blast_radius(g, ids["B"]) == {ids["D"]}

    def test_blast_radius_chain_mid_node(self, chain_graph):
        g, ids = chain_graph
        assert blast_radius(g, ids["training"]) == {
            ids["model"], ids["evaluation"], ids["promotion"],
        }


# ---------------------------------------------------------------------------
# NotFound must be raised BEFORE any networkx call for unknown node_ids
# ---------------------------------------------------------------------------

class TestNotFoundBeforeNetworkx:
    def test_ancestors_unknown_node_raises_exact_notfound_type(self, chain_graph):
        g, _ids = chain_graph
        with pytest.raises(NotFound) as excinfo:
            ancestors(g, "nonexistent-node-id")
        assert type(excinfo.value) is NotFound
        assert not isinstance(excinfo.value, nx.NetworkXException)

    def test_blast_radius_unknown_node_raises_exact_notfound_type(self, chain_graph):
        g, _ids = chain_graph
        with pytest.raises(NotFound) as excinfo:
            blast_radius(g, "nonexistent-node-id")
        assert type(excinfo.value) is NotFound
        assert not isinstance(excinfo.value, nx.NetworkXException)

    def test_shortest_path_len_unknown_from_raises_notfound(self, chain_graph):
        g, ids = chain_graph
        with pytest.raises(NotFound):
            shortest_path_len(g, "nonexistent-node-id", ids["model"])

    def test_shortest_path_len_unknown_to_raises_notfound(self, chain_graph):
        g, ids = chain_graph
        with pytest.raises(NotFound):
            shortest_path_len(g, ids["model"], "nonexistent-node-id")


# ---------------------------------------------------------------------------
# has_cycle()
# ---------------------------------------------------------------------------

class TestHasCycle:
    def test_false_for_normal_dag(self, chain_graph):
        g, _ids = chain_graph
        assert has_cycle(g) is False

    def test_true_for_manually_constructed_cycle(self, cyclic_graph):
        assert has_cycle(cyclic_graph) is True


# ---------------------------------------------------------------------------
# shortest_path_len()
# ---------------------------------------------------------------------------

class TestShortestPathLen:
    def test_known_hop_counts(self, chain_graph):
        g, ids = chain_graph
        assert shortest_path_len(g, ids["data"], ids["training"]) == 1
        assert shortest_path_len(g, ids["data"], ids["promotion"]) == 4

    def test_none_when_reverse_direction_has_no_path(self, chain_graph):
        g, ids = chain_graph
        assert shortest_path_len(g, ids["promotion"], ids["data"]) is None

    def test_none_for_leaf_with_no_outgoing_edges(self, diamond_graph):
        g, ids = diamond_graph
        assert shortest_path_len(g, ids["D"], ids["A"]) is None

    def test_same_node_is_zero_hops(self, chain_graph):
        g, ids = chain_graph
        assert shortest_path_len(g, ids["model"], ids["model"]) == 0


# ---------------------------------------------------------------------------
# to_dot()
# ---------------------------------------------------------------------------

class TestToDot:
    def test_starts_with_digraph(self, chain_graph):
        g, _ids = chain_graph
        dot = to_dot(g)
        assert isinstance(dot, str)
        assert dot.strip().startswith("digraph")

    def test_contains_every_node_id(self, chain_graph):
        g, _ids = chain_graph
        dot = to_dot(g)
        for node_id in g.nodes:
            assert node_id in dot

    def test_contains_every_edge_as_an_arrow_line(self, chain_graph):
        g, _ids = chain_graph
        dot = to_dot(g)
        lines = dot.splitlines()
        for from_id, to_id, _edge_type in g.edges:
            assert any(
                from_id in line and to_id in line and "->" in line
                for line in lines
            ), f"edge {from_id}->{to_id} missing from DOT output"

    def test_balanced_braces(self, chain_graph):
        g, _ids = chain_graph
        dot = to_dot(g)
        assert dot.count("{") == dot.count("}")
        assert dot.count("{") >= 1
