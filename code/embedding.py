"""Deterministic development embedding and embedding-quality measurements."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from numbers import Integral

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from network_metrics import (
    AllPairsShortestPathData,
    prepare_all_pairs_shortest_paths,
    validate_shortest_path_data,
)
from poincare_distance import poincare_distance, validate_disk_point


DEFAULT_DISK_EPSILON = 1e-6
DEFAULT_NUMERICAL_TOLERANCE = 1e-12
DEFAULT_EMBEDDING_RADIUS = 0.85
DEFAULT_FORCE_ITERATIONS = 50
# Backward-compatible alias for the earlier development API name.
DEFAULT_SPRING_ITERATIONS = DEFAULT_FORCE_ITERATIONS
MAX_RANDOM_SEED = 2**32 - 1
DEVELOPMENT_EMBEDDING_ID = "dense_fruchterman_reingold_rescaled_v1"
FORCE_LAYOUT_IMPLEMENTATION = "project_dense_fruchterman_reingold_v1"
FORCE_CONVERGENCE_THRESHOLD = 1e-4


def development_embedding_metadata() -> dict[str, str | float]:
    """Describe the provisional embedding without hiding methodological choices."""

    return {
        "embedding_id": DEVELOPMENT_EMBEDDING_ID,
        "layout_library": "numpy_and_networkx_adjacency",
        "networkx_version": nx.__version__,
        "layout_algorithm": "fruchterman_reingold",
        "layout_method": FORCE_LAYOUT_IMPLEMENTATION,
        "initialization_rng": "numpy_randomstate_mt19937",
        "edge_weighting": "unweighted",
        "convergence_threshold": FORCE_CONVERGENCE_THRESHOLD,
        "centering": "arithmetic_mean",
        "rescaling": "uniform_max_norm_to_embedding_radius",
        "scientific_status": "development_only_not_approved_for_final_results",
    }


@dataclass(frozen=True)
class EmbeddingDistortion:
    """Scale-fitted relative distortion over all unordered vertex pairs."""

    fitted_scale_alpha: float
    mean_relative_distortion: float
    rmse_relative_distortion: float
    unordered_pair_count: int

    @property
    def alpha(self) -> float:
        """Short alias for the fitted geometric scale."""

        return self.fitted_scale_alpha

    @property
    def pair_count(self) -> int:
        """Short alias for the number of unordered vertex pairs."""

        return self.unordered_pair_count

    def as_dict(self) -> dict[str, float | int]:
        """Return a serialization-friendly representation."""

        return {
            "fitted_scale_alpha": self.fitted_scale_alpha,
            "mean_relative_distortion": self.mean_relative_distortion,
            "rmse_relative_distortion": self.rmse_relative_distortion,
            "unordered_pair_count": self.unordered_pair_count,
        }


def _require_positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    validated = float(value)
    if not isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    validated = int(value)
    if validated <= 0:
        raise ValueError(f"{name} must be positive")
    return validated


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    validated = int(seed)
    if not 0 <= validated <= MAX_RANDOM_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_RANDOM_SEED}")
    return validated


def _as_finite_point(point: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite two-dimensional vector") from exc
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional vector")
    return np.array(array, dtype=float, copy=True)


def _validate_undirected_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise ValueError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise ValueError("graph must be undirected")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")


def _stable_node_order(graph: nx.Graph) -> tuple[Hashable, ...]:
    nodes = tuple(graph.nodes())
    if all(isinstance(node, Integral) and not isinstance(node, bool) for node in nodes):
        return tuple(sorted(nodes))
    if all(isinstance(node, str) for node in nodes):
        return tuple(sorted(nodes))
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                type(node).__module__,
                type(node).__qualname__,
                repr(node),
            ),
        )
    )


def _topology_copy_with_stable_order(
    graph: nx.Graph, ordered_nodes: tuple[Hashable, ...]
) -> nx.Graph:
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    ordered_edges = sorted(
        (
            (u, v)
            if node_index[u] <= node_index[v]
            else (v, u)
            for u, v in graph.edges()
        ),
        key=lambda edge: (node_index[edge[0]], node_index[edge[1]]),
    )
    topology = nx.Graph()
    topology.add_nodes_from(ordered_nodes)
    topology.add_edges_from(ordered_edges)
    return topology


def _dependency_free_force_layout(
    graph: nx.Graph,
    ordered_nodes: tuple[Hashable, ...],
    *,
    seed: int,
    iterations: int,
) -> NDArray[np.float64]:
    """Return a deterministic dense Fruchterman-Reingold layout.

    NetworkX's public spring-layout dispatcher selects a SciPy implementation
    at 500 nodes even when its force method is requested. This small, versioned
    implementation keeps one algorithm across graph sizes and depends only on
    the project's required NumPy installation. It follows the standard dense
    Fruchterman-Reingold update with unweighted adjacency and linear cooling.
    """

    node_count = len(ordered_nodes)
    adjacency = nx.to_numpy_array(
        graph,
        nodelist=ordered_nodes,
        dtype=float,
        weight=None,
    )
    random_state = np.random.RandomState(seed)
    positions = np.asarray(random_state.random_sample((node_count, 2)), dtype=float)
    optimal_distance = sqrt(1.0 / node_count)
    temperature = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1]))) * 0.1
    cooling_step = temperature / (iterations + 1)

    for _ in range(iterations):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        np.clip(distances, 0.01, None, out=distances)
        force = optimal_distance**2 / distances**2 - (
            adjacency * distances / optimal_distance
        )
        displacement = np.einsum("ijk,ij->ik", delta, force)
        displacement_norm = np.linalg.norm(displacement, axis=-1)
        np.clip(displacement_norm, 0.01, None, out=displacement_norm)
        position_delta = displacement * (temperature / displacement_norm)[:, None]
        positions += position_delta
        temperature -= cooling_step
        if np.linalg.norm(position_delta) / node_count < FORCE_CONVERGENCE_THRESHOLD:
            break

    if not np.isfinite(positions).all():
        raise RuntimeError("force layout returned a non-finite coordinate")
    return positions


def project_to_unit_disk(
    point: ArrayLike, *, disk_epsilon: float = DEFAULT_DISK_EPSILON
) -> NDArray[np.float64]:
    """Project a finite 2D point radially to the configured safe disk radius.

    Points whose norm is at least ``1 - disk_epsilon`` are projected. Points
    already below that radius are returned unchanged as independent arrays.
    """

    epsilon = _require_positive_finite("disk_epsilon", disk_epsilon)
    if epsilon >= 1.0:
        raise ValueError("disk_epsilon must satisfy 0 < disk_epsilon < 1")

    array = _as_finite_point(point, name="point")
    safe_radius = min(1.0 - epsilon, float(np.nextafter(1.0, 0.0)))
    norm = hypot(float(array[0]), float(array[1]))
    if norm < safe_radius:
        return array
    if norm == 0.0:
        return np.zeros(2, dtype=float)

    # Normalising through the largest component avoids overflow for large but
    # finite input coordinates.
    max_component = float(np.max(np.abs(array)))
    scaled = array / max_component
    direction = scaled / hypot(float(scaled[0]), float(scaled[1]))
    projected = direction * safe_radius
    if not hypot(float(projected[0]), float(projected[1])) < 1.0:
        projected *= float(np.nextafter(1.0, 0.0))
    return np.asarray(projected, dtype=float)


def embed_graph_in_poincare_disk(
    graph: nx.Graph,
    *,
    seed: int,
    embedding_radius: float = DEFAULT_EMBEDDING_RADIUS,
    iterations: int = DEFAULT_FORCE_ITERATIONS,
    disk_epsilon: float = DEFAULT_DISK_EPSILON,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> dict[Hashable, NDArray[np.float64]]:
    """Create a deterministic force-directed development embedding.

    A project-owned dense Fruchterman-Reingold layout is generated from a
    stable topology ordering, centred at the origin, and uniformly rescaled
    into the Poincare disk. This is a Euclidean force-directed development
    embedding, not a canonical hyperbolic embedding; routing conclusions
    remain conditional on this construction.
    """

    _validate_undirected_graph(graph)
    validated_seed = _validate_seed(seed)
    validated_iterations = _require_positive_int("iterations", iterations)
    epsilon = _require_positive_finite("disk_epsilon", disk_epsilon)
    if epsilon >= 1.0:
        raise ValueError("disk_epsilon must satisfy 0 < disk_epsilon < 1")
    radius = _require_positive_finite("embedding_radius", embedding_radius)
    if not radius < 1.0 - epsilon:
        raise ValueError("embedding_radius must be less than 1 - disk_epsilon")
    validated_tolerance = _require_positive_finite("tolerance", tolerance)

    ordered_nodes = _stable_node_order(graph)
    if len(ordered_nodes) == 1:
        return {ordered_nodes[0]: np.zeros(2, dtype=float)}

    topology = _topology_copy_with_stable_order(graph, ordered_nodes)
    coordinate_matrix = _dependency_free_force_layout(
        topology,
        ordered_nodes,
        seed=validated_seed,
        iterations=validated_iterations,
    )
    if coordinate_matrix.shape != (len(ordered_nodes), 2):
        raise RuntimeError("force layout did not return one 2D point per node")
    if not np.isfinite(coordinate_matrix).all():
        raise RuntimeError("force layout returned a non-finite coordinate")

    coordinate_matrix -= np.mean(coordinate_matrix, axis=0)
    norms = np.linalg.norm(coordinate_matrix, axis=1)
    maximum_norm = float(np.max(norms))
    if not isfinite(maximum_norm) or maximum_norm <= validated_tolerance:
        raise RuntimeError("force layout collapsed to a single coordinate")
    coordinate_matrix *= radius / maximum_norm

    coordinates: dict[Hashable, NDArray[np.float64]] = {}
    for node, coordinate in zip(ordered_nodes, coordinate_matrix, strict=True):
        projected = project_to_unit_disk(coordinate, disk_epsilon=epsilon)
        coordinates[node] = validate_disk_point(projected, name=f"coordinate[{node!r}]")
    return coordinates


def calculate_embedding_distortion(
    graph: nx.Graph,
    coordinates: Mapping[Hashable, ArrayLike],
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
    shortest_paths: AllPairsShortestPathData | None = None,
) -> EmbeddingDistortion:
    r"""Measure scale-fitted relative distortion over all unordered pairs.

    For graph distance ``g_ij`` and Poincare distance ``h_ij``, define
    ``q_ij = h_ij / g_ij`` and fit

    ``alpha = sum(q_ij) / sum(q_ij**2)``.

    The primary value is ``mean(abs(alpha*h_ij - g_ij) / g_ij)``; the
    corresponding relative RMSE is also returned. Routing-pair samples are not
    involved.
    """

    _validate_undirected_graph(graph)
    if graph.number_of_nodes() < 2:
        raise ValueError("embedding distortion requires at least two graph nodes")
    validated_tolerance = _require_positive_finite("tolerance", tolerance)
    path_data = (
        prepare_all_pairs_shortest_paths(graph)
        if shortest_paths is None
        else validate_shortest_path_data(graph, shortest_paths)
    )
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinates must be a mapping from every node to a point")

    graph_nodes = set(graph.nodes())
    coordinate_nodes = set(coordinates)
    if coordinate_nodes != graph_nodes:
        missing = graph_nodes - coordinate_nodes
        extra = coordinate_nodes - graph_nodes
        raise ValueError(
            "coordinates must cover exactly the graph nodes "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    ordered_nodes = _stable_node_order(graph)
    validated_coordinates = {
        node: validate_disk_point(coordinates[node], name=f"coordinate[{node!r}]")
        for node in ordered_nodes
    }
    coordinate_owners: dict[tuple[float, float], Hashable] = {}
    for node, coordinate in validated_coordinates.items():
        coordinate_key = (float(coordinate[0]), float(coordinate[1]))
        if coordinate_key in coordinate_owners:
            other = coordinate_owners[coordinate_key]
            raise ValueError(
                "distinct graph nodes must not have duplicate coordinates "
                f"({other!r} and {node!r})"
            )
        coordinate_owners[coordinate_key] = node
    shortest_path_lengths = path_data.distances

    ratios: list[float] = []
    for left_index, left in enumerate(ordered_nodes[:-1]):
        for right in ordered_nodes[left_index + 1 :]:
            graph_distance = shortest_path_lengths[left][right]
            if graph_distance <= 0:
                raise RuntimeError("distinct connected vertices must have positive distance")
            geometric_distance = poincare_distance(
                validated_coordinates[left],
                validated_coordinates[right],
                tolerance=validated_tolerance,
            )
            ratio = geometric_distance / float(graph_distance)
            if not isfinite(ratio) or ratio < 0.0:
                raise ValueError("embedding produced an invalid pairwise distance ratio")
            ratios.append(ratio)

    ratio_array = np.asarray(ratios, dtype=float)
    squared_sum = float(np.dot(ratio_array, ratio_array))
    if squared_sum <= 0.0 or not isfinite(squared_sum):
        raise ValueError("cannot fit a positive scale to zero geometric distances")
    alpha = float(np.sum(ratio_array) / squared_sum)
    if alpha <= 0.0 or not isfinite(alpha):
        raise ValueError("fitted embedding scale must be positive and finite")

    relative_errors = alpha * ratio_array - 1.0
    mean_relative_distortion = float(np.mean(np.abs(relative_errors)))
    rmse_relative_distortion = sqrt(float(np.mean(np.square(relative_errors))))
    if not (
        isfinite(mean_relative_distortion)
        and isfinite(rmse_relative_distortion)
        and mean_relative_distortion >= 0.0
        and rmse_relative_distortion >= 0.0
    ):
        raise ValueError("embedding distortion is not finite and non-negative")

    return EmbeddingDistortion(
        fitted_scale_alpha=alpha,
        mean_relative_distortion=mean_relative_distortion,
        rmse_relative_distortion=rmse_relative_distortion,
        unordered_pair_count=len(ratios),
    )
