"""Shared embedding inputs, validation, and development embedding utilities."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
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
EMBEDDING_NODE_LABEL_CONTRACT = (
    "homogeneous_non_boolean_integers_or_homogeneous_strings_v1"
)
FINAL_EXPERIMENT_NODE_LABEL_CONTRACT = (
    "non_boolean_integer_ids_zero_through_n_minus_one_v1"
)


def _validate_supported_node_labels(nodes: tuple[Hashable, ...]) -> str:
    """Return the stable label kind or reject labels with unstable identities."""

    if all(
        isinstance(node, Integral) and not isinstance(node, bool)
        for node in nodes
    ):
        return "integer"
    if all(isinstance(node, str) for node in nodes):
        return "string"
    raise ValueError(
        "node labels must be homogeneous non-boolean integers or homogeneous "
        "strings; mixed labels and custom objects are unsupported"
    )


@dataclass(frozen=True, eq=False)
class EmbeddingInput:
    """One immutable ordered graph-distance matrix shared by embedding families."""

    node_order: tuple[Hashable, ...]
    distance_matrix: NDArray[np.float64] = field(repr=False)
    configuration_fingerprint: str
    input_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_order, tuple) or not self.node_order:
            raise ValueError("EmbeddingInput.node_order must be a non-empty tuple")
        _validate_supported_node_labels(self.node_order)
        try:
            if len(set(self.node_order)) != len(self.node_order):
                raise ValueError("EmbeddingInput.node_order values must be unique")
        except TypeError as exc:
            raise ValueError(
                "EmbeddingInput.node_order values must be hashable"
            ) from exc
        if not isinstance(self.distance_matrix, np.ndarray):
            raise ValueError("EmbeddingInput.distance_matrix must be a NumPy array")
        if (
            self.distance_matrix.dtype != np.dtype(np.float64)
            or self.distance_matrix.ndim != 2
            or self.distance_matrix.shape
            != (len(self.node_order), len(self.node_order))
        ):
            raise ValueError(
                "EmbeddingInput.distance_matrix must be a matching float64 square matrix"
            )
        if (
            not np.isfinite(self.distance_matrix).all()
            or np.any(self.distance_matrix < 0.0)
            or not np.array_equal(
                self.distance_matrix,
                self.distance_matrix.T,
            )
            or not np.array_equal(
                np.diag(self.distance_matrix),
                np.zeros(len(self.node_order), dtype=np.float64),
            )
        ):
            raise ValueError(
                "EmbeddingInput.distance_matrix must retain validated distance invariants"
            )
        if self.distance_matrix.flags.writeable:
            raise ValueError("EmbeddingInput.distance_matrix must be read-only")
        if (
            not isinstance(self.configuration_fingerprint, str)
            or not self.configuration_fingerprint
        ):
            raise ValueError(
                "EmbeddingInput.configuration_fingerprint must be a non-empty string"
            )
        if (
            not isinstance(self.input_fingerprint, str)
            or len(self.input_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.input_fingerprint
            )
        ):
            raise ValueError(
                "EmbeddingInput.input_fingerprint must be a lowercase SHA-256 digest"
            )


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


def _canonicalize_vector_sign(
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.array(vector, dtype=np.float64, copy=True)
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def canonical_eigenvectors_for_indices(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    selected_indices: tuple[int, ...],
    *,
    relative_tolerance: float,
) -> tuple[tuple[NDArray[np.float64], ...], tuple[int, ...]]:
    """Return deterministic vectors even when selected eigenvalues repeat.

    Eigenvectors from a symmetric eigensolver are sign-ambiguous, and an
    orthonormal basis of a repeated eigenspace is rotation-ambiguous. Values
    within ``relative_tolerance * max(1, max(abs(eigenvalues)))`` are treated
    as one numerical eigenspace. Its basis is then reconstructed by projecting
    stable standard node axes and applying deterministic reorthogonalisation.
    Only vectors needed by ``selected_indices`` are constructed.
    """

    tolerance = _require_positive_finite(
        "relative_tolerance",
        relative_tolerance,
    )
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if (
        values.ndim != 1
        or vectors.ndim != 2
        or vectors.shape != (len(values), len(values))
        or not np.isfinite(values).all()
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("eigenvalues and eigenvectors have invalid shapes or values")
    indices = tuple(selected_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(values)
            for index in indices
        )
    ):
        raise ValueError("selected_indices must contain unique valid integers")
    scale = max(1.0, float(np.max(np.abs(values))))
    grouping_threshold = tolerance * scale
    if np.any(values[:-1] < values[1:] - grouping_threshold):
        raise ValueError("eigenvalues must be sorted in non-increasing order")

    vector_by_index: dict[int, NDArray[np.float64]] = {}
    selected_group_sizes: list[int] = []
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and float(values[start] - values[end]) <= grouping_threshold
        ):
            end += 1
        selected_in_group = tuple(
            index for index in indices if start <= index < end
        )
        if not selected_in_group:
            start = end
            continue
        group_size = end - start
        if group_size == 1:
            vector_by_index[selected_in_group[0]] = _canonicalize_vector_sign(
                vectors[:, start]
            )
            start = end
            continue

        selected_group_sizes.append(group_size)
        group_vectors = vectors[:, start:end]
        basis: list[NDArray[np.float64]] = []
        basis_threshold = max(
            tolerance,
            64.0 * np.finfo(np.float64).eps * len(values),
        )
        for axis in range(len(values)):
            candidate = group_vectors @ group_vectors[axis, :]
            for _ in range(2):
                for previous in basis:
                    candidate -= float(np.dot(previous, candidate)) * previous
            norm = float(np.linalg.norm(candidate))
            if norm <= basis_threshold:
                continue
            candidate /= norm
            if candidate[axis] < 0.0:
                candidate *= -1.0
            basis.append(candidate)
            if len(basis) == len(selected_in_group):
                break
        if len(basis) != len(selected_in_group):
            raise RuntimeError(
                "could not construct a deterministic repeated-eigenspace basis"
            )
        for index, vector in zip(selected_in_group, basis, strict=True):
            vector_by_index[index] = vector
        start = end

    return (
        tuple(vector_by_index[index] for index in indices),
        tuple(selected_group_sizes),
    )


def validate_distance_matrix(
    distance_matrix: ArrayLike,
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> NDArray[np.float64]:
    """Validate and return a symmetric float64 copy of a distance matrix.

    Symmetry and the zero diagonal are checked within ``tolerance``. Values
    must otherwise be finite and non-negative. A matrix that passes the
    symmetry check is averaged with its transpose only to remove roundoff-level
    asymmetry before symmetric eigendecomposition.
    """

    validated_tolerance = _require_positive_finite("tolerance", tolerance)
    try:
        matrix = np.asarray(distance_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_matrix must be a finite square numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if matrix.shape[0] < 2:
        raise ValueError("distance_matrix must contain at least two points")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "distance_matrix must contain only finite connected-pair distances"
        )
    if np.any(matrix < 0.0):
        raise ValueError("distance_matrix distances must be non-negative")
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=validated_tolerance,
    ):
        raise ValueError(
            "distance_matrix must be symmetric within the configured tolerance"
        )
    if not np.allclose(
        np.diag(matrix),
        0.0,
        rtol=0.0,
        atol=validated_tolerance,
    ):
        raise ValueError(
            "distance_matrix diagonal must be zero within the configured tolerance"
        )

    validated = np.array((matrix + matrix.T) * 0.5, dtype=np.float64, copy=True)
    np.fill_diagonal(validated, 0.0)
    return validated


def _node_fingerprint_identity(node: Hashable) -> tuple[str, str]:
    if isinstance(node, Integral) and not isinstance(node, bool):
        return ("integer", str(int(node)))
    if isinstance(node, str):
        return ("string", node)
    raise ValueError("unsupported node label reached fingerprint serialization")


def embedding_input_from_distance_matrix(
    distance_matrix: ArrayLike,
    node_order: tuple[Hashable, ...] | list[Hashable],
    *,
    configuration_fingerprint: str,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> EmbeddingInput:
    """Build an immutable, fingerprinted input shared by Hydra and MDS.

    Low-level mathematical fixtures may use homogeneous string labels. The
    approved experiment uses only non-boolean integer IDs; mixed labels and
    custom object labels are rejected before a fingerprint can be produced.
    """

    if not isinstance(configuration_fingerprint, str) or not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be a non-empty string")
    ordered_nodes = tuple(node_order)
    if not ordered_nodes:
        raise ValueError("node_order must not be empty")
    _validate_supported_node_labels(ordered_nodes)
    try:
        unique_nodes = set(ordered_nodes)
    except TypeError as exc:
        raise ValueError("node_order values must be hashable") from exc
    if len(unique_nodes) != len(ordered_nodes):
        raise ValueError("node_order values must be unique")

    validated = validate_distance_matrix(distance_matrix, tolerance=tolerance)
    if validated.shape[0] != len(ordered_nodes):
        raise ValueError(
            "node_order length must match the distance_matrix dimension"
        )
    canonical_nodes = [
        _node_fingerprint_identity(node) for node in ordered_nodes
    ]
    digest = sha256()
    digest.update(b"greedy-routing-embedding-input-v2\0")
    digest.update(EMBEDDING_NODE_LABEL_CONTRACT.encode("ascii"))
    digest.update(b"\0")
    digest.update(configuration_fingerprint.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            canonical_nodes,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    canonical_matrix = np.asarray(validated, dtype="<f8", order="C")
    digest.update(str(canonical_matrix.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_matrix.tobytes(order="C"))
    validated.setflags(write=False)
    return EmbeddingInput(
        node_order=ordered_nodes,
        distance_matrix=validated,
        configuration_fingerprint=configuration_fingerprint,
        input_fingerprint=digest.hexdigest(),
    )


def _validate_undirected_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise ValueError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise ValueError("graph must be undirected")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")


def _stable_node_order(graph: nx.Graph) -> tuple[Hashable, ...]:
    nodes = tuple(graph.nodes())
    label_kind = _validate_supported_node_labels(nodes)
    if label_kind == "integer":
        return tuple(sorted(nodes))
    if label_kind == "string":
        return tuple(sorted(nodes))
    raise RuntimeError("validated node-label kind is unsupported")


def stable_node_order(graph: nx.Graph) -> tuple[Hashable, ...]:
    """Return the public deterministic node ordering used by every embedding."""

    _validate_undirected_graph(graph)
    return _stable_node_order(graph)


def prepare_embedding_input(
    graph: nx.Graph,
    shortest_paths: AllPairsShortestPathData,
    *,
    configuration_fingerprint: str,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> EmbeddingInput:
    """Construct one reusable matrix from already-computed shortest paths."""

    path_data = validate_shortest_path_data(graph, shortest_paths)
    ordered_nodes = stable_node_order(graph)
    matrix = np.asarray(
        [
            [path_data.distances[source][target] for target in ordered_nodes]
            for source in ordered_nodes
        ],
        dtype=np.float64,
    )
    return embedding_input_from_distance_matrix(
        matrix,
        ordered_nodes,
        configuration_fingerprint=configuration_fingerprint,
        tolerance=tolerance,
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
