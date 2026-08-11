"""Independent spectral reconstructions used to audit Iteration 2 embeddings.

The functions accept only an APSP matrix and node order.  They do not import or
call either production embedding implementation or its intermediate helpers.
Comparisons are made in the geometry-invariant pairwise distance domain.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IndependentEmbedding:
    family: str
    coordinates: Mapping[Hashable, tuple[float, float]]
    effective_rank: int
    retained_eigenvalues: tuple[float, ...]
    eigenvalue_absolute_threshold: float


@dataclass(frozen=True)
class EmbeddingOracleAgreement:
    family: str
    pair_count: int
    maximum_pairwise_distance_error: float
    mean_pairwise_distance_error: float
    reference_effective_rank: int
    production_effective_rank: int
    finite: bool
    inside_open_disk: bool | None


def _validated(
    distance_matrix: Sequence[Sequence[float]],
    node_order: Sequence[Hashable],
) -> tuple[NDArray[np.float64], tuple[Hashable, ...]]:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    nodes = tuple(node_order)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] != len(nodes)
        or len(nodes) < 3
        or len(set(nodes)) != len(nodes)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0)
        or not np.all(np.diag(matrix) == 0.0)
    ):
        raise ValueError("independent embedding requires an exact finite APSP matrix")
    return np.array(matrix, copy=True), nodes


def _mapping(
    coordinates: NDArray[np.float64],
    node_order: Sequence[Hashable],
) -> Mapping[Hashable, tuple[float, float]]:
    return MappingProxyType(
        {
            node: (float(point[0]), float(point[1]))
            for node, point in zip(node_order, coordinates, strict=True)
        }
    )


def _canonical_selected_eigenvectors(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    selected_indices: Sequence[int],
    *,
    relative_tolerance: float,
) -> tuple[NDArray[np.float64], ...]:
    """Independently fix signs and repeated-eigenspace basis ambiguity."""

    count = len(eigenvalues)
    indices = tuple(index % count for index in selected_indices)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = float(relative_tolerance) * scale
    result: dict[int, NDArray[np.float64]] = {}
    start = 0
    while start < count:
        end = start + 1
        while (
            end < count
            and float(eigenvalues[start] - eigenvalues[end]) <= threshold
        ):
            end += 1
        requested = tuple(index for index in indices if start <= index < end)
        if not requested:
            start = end
            continue
        if end - start == 1:
            vector = np.array(eigenvectors[:, start], copy=True)
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector *= -1.0
            result[requested[0]] = vector
            start = end
            continue
        eigenspace = eigenvectors[:, start:end]
        projector = eigenspace @ eigenspace.T
        basis: list[NDArray[np.float64]] = []
        basis_floor = max(
            float(relative_tolerance),
            64.0 * np.finfo(np.float64).eps * count,
        )
        for axis in range(count):
            candidate = np.array(projector[:, axis], copy=True)
            for _ in range(2):
                for previous in basis:
                    candidate -= np.dot(previous, candidate) * previous
            norm = float(np.linalg.norm(candidate))
            if norm <= basis_floor:
                continue
            candidate /= norm
            if candidate[axis] < 0.0:
                candidate *= -1.0
            basis.append(candidate)
            if len(basis) == len(requested):
                break
        if len(basis) != len(requested):
            raise ValueError(
                "independent repeated-eigenspace basis is underdetermined"
            )
        for index, vector in zip(requested, basis, strict=True):
            result[index] = vector
        start = end
    return tuple(result[index] for index in indices)


def independent_classical_mds(
    distance_matrix: Sequence[Sequence[float]],
    node_order: Sequence[Hashable],
    *,
    relative_tolerance: float = 1e-12,
) -> IndependentEmbedding:
    matrix, nodes = _validated(distance_matrix, node_order)
    count = len(nodes)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (matrix * matrix) @ centering
    gram = (gram + gram.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    threshold = float(relative_tolerance) * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    selected = np.flatnonzero(eigenvalues > threshold)[:2]
    if not len(selected):
        raise ValueError("independent MDS reconstruction has rank zero")
    canonical = _canonical_selected_eigenvectors(
        eigenvalues,
        eigenvectors,
        tuple(int(index) for index in selected),
        relative_tolerance=relative_tolerance,
    )
    coordinates = np.column_stack(
        [
            vector * sqrt(float(eigenvalues[index]))
            for index, vector in zip(selected, canonical, strict=True)
        ]
    )
    if len(selected) == 1:
        coordinates = np.column_stack(
            (coordinates[:, 0], np.zeros(count, dtype=np.float64))
        )
    coordinates -= coordinates.mean(axis=0)
    if (
        not np.isfinite(coordinates).all()
        or float(np.max(np.linalg.norm(coordinates, axis=1))) <= 0.0
        or float(np.linalg.norm(np.mean(coordinates, axis=0))) > 1e-12
    ):
        raise ValueError("independent MDS reconstruction is invalid")
    if len(selected) == 1 and not np.all(coordinates[:, 1] == 0.0):
        raise ValueError("independent rank-one MDS did not zero column one")
    return IndependentEmbedding(
        family="mds",
        coordinates=_mapping(coordinates, nodes),
        effective_rank=len(selected),
        retained_eigenvalues=tuple(float(eigenvalues[i]) for i in selected),
        eigenvalue_absolute_threshold=threshold,
    )


def independent_hydra(
    distance_matrix: Sequence[Sequence[float]],
    node_order: Sequence[Hashable],
    *,
    relative_tolerance: float = 1e-12,
) -> IndependentEmbedding:
    matrix, nodes = _validated(distance_matrix, node_order)
    transformed = np.cosh(matrix)
    if not np.isfinite(transformed).all():
        raise ValueError("independent Hydra cosh transform overflowed")
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    threshold = float(relative_tolerance) * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    leading = float(eigenvalues[0])
    if leading <= threshold:
        raise ValueError("independent Hydra has no time-like direction")
    canonical = _canonical_selected_eigenvectors(
        eigenvalues,
        eigenvectors,
        (0, len(eigenvalues) - 2, len(eigenvalues) - 1),
        relative_tolerance=relative_tolerance,
    )
    leading_vector = np.asarray(canonical[0], dtype=np.float64)
    if leading_vector.sum() < 0.0:
        leading_vector *= -1.0
    if np.any(leading_vector <= 0.0):
        raise ValueError("independent Hydra time direction is not positive")
    spatial_columns: list[NDArray[np.float64]] = []
    retained: list[float] = []
    for index, vector in zip(
        (-2, -1),
        canonical[1:],
        strict=True,
    ):
        value = -float(eigenvalues[index])
        if value > threshold:
            spatial_columns.append(vector * sqrt(value))
            retained.append(float(eigenvalues[index]))
    if not spatial_columns:
        raise ValueError("independent Hydra has spatial rank zero")
    if len(spatial_columns) == 1:
        spatial_columns.append(np.zeros_like(spatial_columns[0]))
    spatial = np.column_stack(spatial_columns)
    time = sqrt(leading) * leading_vector
    minimum_time = min(1.0, float(np.min(time)))
    radial = np.sqrt(np.maximum((time - minimum_time) / (time + minimum_time), 0.0))
    norms = np.linalg.norm(spatial, axis=1)
    coordinates = np.zeros((len(nodes), 2), dtype=np.float64)
    for index, norm in enumerate(norms):
        if norm > 0.0:
            coordinates[index] = radial[index] * spatial[index] / norm
        elif radial[index] > relative_tolerance:
            raise ValueError("independent Hydra direction is undefined")
    radii = np.linalg.norm(coordinates, axis=1)
    if (
        not np.isfinite(coordinates).all()
        or np.any(radii >= 1.0)
        or not np.any(np.linalg.norm(coordinates - coordinates[0], axis=1) > 0.0)
    ):
        raise ValueError("independent Hydra left the open disk")
    if len(retained) == 1 and not np.all(coordinates[:, 1] == 0.0):
        raise ValueError("independent rank-one Hydra did not zero column one")
    return IndependentEmbedding(
        family="hydra",
        coordinates=_mapping(coordinates, nodes),
        effective_rank=len(retained),
        retained_eigenvalues=(leading, *retained),
        eigenvalue_absolute_threshold=threshold,
    )


def pairwise_euclidean(
    coordinates: Mapping[Hashable, Sequence[float]],
    node_order: Sequence[Hashable],
) -> NDArray[np.float64]:
    values = np.asarray([coordinates[node] for node in node_order], dtype=np.float64)
    difference = values[:, None, :] - values[None, :, :]
    matrix = np.linalg.norm(difference, axis=2)
    return matrix[np.triu_indices(len(values), 1)]


def pairwise_poincare(
    coordinates: Mapping[Hashable, Sequence[float]],
    node_order: Sequence[Hashable],
) -> NDArray[np.float64]:
    values = np.asarray([coordinates[node] for node in node_order], dtype=np.float64)
    squared = np.sum(values * values, axis=1)
    if np.any(squared >= 1.0):
        raise ValueError("Poincare comparison requires open-disk coordinates")
    difference = values[:, None, :] - values[None, :, :]
    numerator = np.linalg.norm(difference, axis=2)
    denominator = np.sqrt(
        (1.0 - squared)[:, None] * (1.0 - squared)[None, :]
    )
    matrix = 2.0 * np.arcsinh(numerator / denominator)
    return matrix[np.triu_indices(len(values), 1)]


def independent_distortion_diagnostic(
    graph_distances: Sequence[float],
    geometric_distances: Sequence[float],
) -> dict[str, float | None]:
    """Independently reconstruct the frozen scale-fitted diagnostics."""

    graph = np.asarray(tuple(graph_distances), dtype=np.float64)
    geometry = np.asarray(tuple(geometric_distances), dtype=np.float64)
    if (
        graph.ndim != 1
        or geometry.shape != graph.shape
        or not len(graph)
        or not np.isfinite(graph).all()
        or not np.isfinite(geometry).all()
        or np.any(graph <= 0.0)
        or np.any(geometry < 0.0)
    ):
        raise ValueError("distortion oracle requires finite paired distances")
    denominator = float(sum(float(value) ** 2 for value in geometry))
    if denominator <= 0.0:
        raise ValueError("distortion oracle rejects complete pairwise collapse")
    numerator = float(
        sum(
            float(left) * float(right)
            for left, right in zip(graph, geometry, strict=True)
        )
    )
    fitted_scale = numerator / denominator
    residual = fitted_scale * geometry - graph
    relative = residual / graph

    def midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0 + 1.0
            for position in range(start, end):
                ranks[order[position]] = rank
            start = end
        return ranks

    graph_ranks = midranks(graph)
    geometry_ranks = midranks(geometry)
    correlation = (
        None
        if float(np.std(graph_ranks)) == 0.0
        or float(np.std(geometry_ranks)) == 0.0
        else float(np.corrcoef(graph_ranks, geometry_ranks)[0, 1])
    )
    return {
        "fitted_scale": fitted_scale,
        "scale_fitted_mean_absolute_distance_error": float(
            np.mean(np.abs(residual))
        ),
        "scale_fitted_rmse_distance": float(
            np.sqrt(np.mean(np.square(residual)))
        ),
        "scale_fitted_mean_relative_error": float(np.mean(np.abs(relative))),
        "scale_fitted_rmse_relative_error": float(
            np.sqrt(np.mean(np.square(relative)))
        ),
        "graph_distance_rank_correlation": correlation,
        "normalized_distance_stress": float(
            np.linalg.norm(residual) / np.linalg.norm(graph)
        ),
    }


def audit_embedding(
    *,
    family: str,
    distance_matrix: Sequence[Sequence[float]],
    node_order: Sequence[Hashable],
    production_coordinates: Mapping[Hashable, Sequence[float]],
    production_effective_rank: int,
    relative_tolerance: float = 1e-12,
    reference: IndependentEmbedding | None = None,
) -> EmbeddingOracleAgreement:
    if family == "mds":
        reference = (
            independent_classical_mds(
                distance_matrix,
                node_order,
                relative_tolerance=relative_tolerance,
            )
            if reference is None
            else reference
        )
        if reference.family != "mds":
            raise ValueError("embedding oracle reference family mismatch")
        production_distances = pairwise_euclidean(
            production_coordinates, node_order
        )
        reference_distances = pairwise_euclidean(
            reference.coordinates, node_order
        )
        inside: bool | None = None
    elif family == "hydra":
        reference = (
            independent_hydra(
                distance_matrix,
                node_order,
                relative_tolerance=relative_tolerance,
            )
            if reference is None
            else reference
        )
        if reference.family != "hydra":
            raise ValueError("embedding oracle reference family mismatch")
        production_distances = pairwise_poincare(
            production_coordinates, node_order
        )
        reference_distances = pairwise_poincare(
            reference.coordinates, node_order
        )
        inside = all(
            float(point[0]) ** 2 + float(point[1]) ** 2 < 1.0
            for point in production_coordinates.values()
        )
    else:
        raise ValueError("family must be hydra or mds")
    errors = np.abs(production_distances - reference_distances)
    maximum = float(np.max(errors))
    mean = float(np.mean(errors))
    if not all(isfinite(value) for value in (maximum, mean)):
        raise ValueError("embedding oracle comparison is non-finite")
    return EmbeddingOracleAgreement(
        family=family,
        pair_count=len(errors),
        maximum_pairwise_distance_error=maximum,
        mean_pairwise_distance_error=mean,
        reference_effective_rank=reference.effective_rank,
        production_effective_rank=int(production_effective_rank),
        finite=True,
        inside_open_disk=inside,
    )
