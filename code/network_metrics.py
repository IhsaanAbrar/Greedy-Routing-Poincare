"""Structural measurements for connected, simple, undirected graphs."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

import networkx as nx


class NetworkMetrics(TypedDict):
    """Stable JSON-serializable field schema returned by this module."""

    number_of_vertices: int
    number_of_edges: int
    average_degree: float
    maximum_degree: int
    population_degree_variance: float
    average_clustering_coefficient: float
    diameter: int
    average_shortest_path_length: float


@dataclass(frozen=True)
class AllPairsShortestPathData:
    """Reusable exact unweighted distances tied to one graph topology."""

    distances: Mapping[Hashable, Mapping[Hashable, int]]
    nodes: frozenset[Hashable]
    edges: frozenset[frozenset[Hashable]]
    diameter: int
    ordered_distance_sum: int


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise ValueError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise ValueError("graph must be undirected")
    if graph.is_multigraph():
        raise ValueError("graph must be simple, not a multigraph")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must be non-empty")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
    if not nx.is_connected(graph):
        raise ValueError("graph must be connected")


def _edge_snapshot(graph: nx.Graph) -> frozenset[frozenset[Hashable]]:
    return frozenset(frozenset((left, right)) for left, right in graph.edges())


def prepare_all_pairs_shortest_paths(graph: nx.Graph) -> AllPairsShortestPathData:
    """Compute exact unweighted all-pairs distances once for safe reuse."""

    _validate_graph(graph)
    node_snapshot = frozenset(graph.nodes())
    mutable_distances: dict[Hashable, Mapping[Hashable, int]] = {}
    diameter = 0
    ordered_distance_sum = 0
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        row = {target: int(distance) for target, distance in lengths.items()}
        if frozenset(row) != node_snapshot:
            raise RuntimeError("all-pairs traversal did not reach every graph node")
        mutable_distances[source] = MappingProxyType(row)
        diameter = max(diameter, max(row.values(), default=0))
        ordered_distance_sum += sum(row.values())
    if frozenset(mutable_distances) != node_snapshot:
        raise RuntimeError("all-pairs traversal omitted a graph source")

    return AllPairsShortestPathData(
        distances=MappingProxyType(mutable_distances),
        nodes=node_snapshot,
        edges=_edge_snapshot(graph),
        diameter=diameter,
        ordered_distance_sum=ordered_distance_sum,
    )


def validate_shortest_path_data(
    graph: nx.Graph,
    shortest_paths: AllPairsShortestPathData,
) -> AllPairsShortestPathData:
    """Reject cached distances prepared for a different graph topology."""

    _validate_graph(graph)
    if not isinstance(shortest_paths, AllPairsShortestPathData):
        raise ValueError(
            "shortest_paths must come from prepare_all_pairs_shortest_paths"
        )
    if shortest_paths.nodes != frozenset(graph.nodes()):
        raise ValueError("shortest-path data nodes do not match the graph")
    if shortest_paths.edges != _edge_snapshot(graph):
        raise ValueError("shortest-path data topology does not match the graph")
    return shortest_paths


def calculate_network_metrics(
    graph: nx.Graph,
    *,
    shortest_paths: AllPairsShortestPathData | None = None,
) -> NetworkMetrics:
    """Calculate the experiment's graph-level structural measurements.

    Definitions for a graph with ``n`` vertices and degrees ``d_i``:

    - ``average_degree`` is ``sum(d_i) / n`` (equivalently ``2|E| / n``).
    - ``maximum_degree`` is ``max(d_i)``.
    - ``population_degree_variance`` is
      ``sum((d_i - average_degree)**2) / n`` (``ddof=0``).
    - ``average_clustering_coefficient`` is the mean of NetworkX's local
      clustering coefficients.
    - ``diameter`` is the largest unweighted shortest-path distance.
    - ``average_shortest_path_length`` is the mean unweighted distance over
      distinct vertex pairs.  For an undirected graph, averaging ordered or
      unordered pairs gives the same value.

    Embedding distortion is intentionally excluded because it is measured only
    after coordinates have been constructed.
    """

    path_data = (
        prepare_all_pairs_shortest_paths(graph)
        if shortest_paths is None
        else validate_shortest_path_data(graph, shortest_paths)
    )

    number_of_vertices = graph.number_of_nodes()
    degrees = [degree for _, degree in graph.degree()]
    average_degree = sum(degrees) / number_of_vertices
    population_degree_variance = (
        sum((degree - average_degree) ** 2 for degree in degrees)
        / number_of_vertices
    )

    return {
        "number_of_vertices": number_of_vertices,
        "number_of_edges": graph.number_of_edges(),
        "average_degree": float(average_degree),
        "maximum_degree": int(max(degrees)),
        "population_degree_variance": float(population_degree_variance),
        "average_clustering_coefficient": float(nx.average_clustering(graph)),
        "diameter": path_data.diameter,
        "average_shortest_path_length": (
            float(path_data.ordered_distance_sum)
            / (number_of_vertices * (number_of_vertices - 1))
            if number_of_vertices > 1
            else 0.0
        ),
    }
