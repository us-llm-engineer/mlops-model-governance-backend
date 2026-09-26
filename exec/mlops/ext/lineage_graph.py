"""NetworkX integration for LineageGraph.

Implements graph analysis functions (ancestors, blast_radius, has_cycle,
shortest_path_len, to_dot) by converting a LineageGraph to a NetworkX DiGraph
with properly attributed nodes and edges.

Node attributes (set by to_networkx):
  - node_type (str): The node's type (data, training, model, evaluation, promotion)
  - immutable (bool): Result of node.is_immutable()

Edge attributes (set by to_networkx):
  - edge_type (str): The edge type as added to the LineageGraph
"""

import networkx as nx

from mlops.kernel import NotFound
from mlops.lineage import LineageGraph


def _require_node(graph: LineageGraph, node_id: str) -> None:
    """Require that node_id exists in the graph; raise NotFound if not.

    This must be called before any networkx operation to catch unknown
    node_ids with our NotFound exception, not networkx's NodeNotFound.
    """
    if node_id not in graph.nodes:
        raise NotFound(f"Node not found: {node_id}")


def to_networkx(graph: LineageGraph) -> nx.DiGraph:
    """Convert a LineageGraph to a NetworkX DiGraph with node and edge attributes.

    Node attributes:
      - node_type (str): The node's type from LineageGraphNode.node_type
      - immutable (bool): Result of node.is_immutable()

    Edge attributes:
      - edge_type (str): The edge type as stored in the edge tuple

    Args:
        graph: A LineageGraph instance.

    Returns:
        A NetworkX DiGraph with all nodes and edges from the input graph,
        with attributes set as described above.
    """
    nxg = nx.DiGraph()

    # Add all nodes with their attributes
    for node_id, node in graph.nodes.items():
        nxg.add_node(node_id, node_type=node.node_type, immutable=node.is_immutable())

    # Add all edges with their attributes
    for from_id, to_id, edge_type in graph.edges:
        nxg.add_edge(from_id, to_id, edge_type=edge_type)

    return nxg


def ancestors(graph: LineageGraph, node_id: str) -> set:
    """Return the set of ancestors of a node (excluding the node itself).

    This wraps nx.ancestors(graph, node_id) and will raise NotFound if the
    node doesn't exist in the graph.

    Args:
        graph: A LineageGraph instance.
        node_id: The node ID to find ancestors for.

    Returns:
        A set of node IDs that are ancestors of the given node (does not
        include the node itself).

    Raises:
        NotFound: If node_id is not in the graph.
    """
    _require_node(graph, node_id)
    nxg = to_networkx(graph)
    return nx.ancestors(nxg, node_id)


def blast_radius(graph: LineageGraph, node_id: str) -> set:
    """Return the set of nodes that would be affected by a change to this node.

    This is the set of descendants (nodes reachable FROM the given node in the
    forward direction), excluding the node itself.

    Args:
        graph: A LineageGraph instance.
        node_id: The node ID to find the blast radius for.

    Returns:
        A set of node IDs that are descendants of the given node (nodes that
        depend on or are produced from this node), excluding the node itself.

    Raises:
        NotFound: If node_id is not in the graph.
    """
    _require_node(graph, node_id)
    nxg = to_networkx(graph)
    return nx.descendants(nxg, node_id)


def has_cycle(graph: LineageGraph) -> bool:
    """Return True if the graph contains any cycles.

    Args:
        graph: A LineageGraph instance.

    Returns:
        True if the graph is NOT a directed acyclic graph (i.e., has cycles),
        False otherwise.
    """
    nxg = to_networkx(graph)
    return not nx.is_directed_acyclic_graph(nxg)


def shortest_path_len(graph: LineageGraph, from_id: str, to_id: str) -> int | None:
    """Return the shortest path length between two nodes, or None if no path exists.

    If from_id == to_id, returns 0.

    Args:
        graph: A LineageGraph instance.
        from_id: The starting node ID.
        to_id: The ending node ID.

    Returns:
        The length (number of edges) of the shortest path, or None if no path exists.
        Returns 0 if from_id == to_id.

    Raises:
        NotFound: If either from_id or to_id is not in the graph.
    """
    _require_node(graph, from_id)
    _require_node(graph, to_id)

    if from_id == to_id:
        return 0

    nxg = to_networkx(graph)
    try:
        return nx.shortest_path_length(nxg, from_id, to_id)
    except nx.NetworkXNoPath:
        return None


def to_dot(graph: LineageGraph) -> str:
    """Return a DOT format representation of the graph.

    This is a hand-written digraph emitter (no pydot/pygraphviz dependencies).
    The output is a valid GraphViz digraph string.

    Args:
        graph: A LineageGraph instance.

    Returns:
        A string containing the digraph in DOT format.
    """
    lines = ["digraph {"]

    # Add all nodes
    for node_id in graph.nodes.keys():
        # Escape node ID if necessary for DOT (quoted if it contains special chars)
        safe_id = node_id
        lines.append(f'  "{safe_id}";')

    # Add all edges
    for from_id, to_id, edge_type in graph.edges:
        safe_from = from_id
        safe_to = to_id
        lines.append(f'  "{safe_from}" -> "{safe_to}";')

    lines.append("}")
    return "\n".join(lines)
