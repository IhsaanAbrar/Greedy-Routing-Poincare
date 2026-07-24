"""Deterministic classical two-dimensional multidimensional scaling."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from embedding import EmbeddingInput, canonical_eigenvectors_for_indices
from experiment_config import (
    MDS_BASE_EMBEDDING_ID,
    MDS_CONDITION_IDS,
    MDS_DIMENSION,
    MDS_EMBEDDING_FAMILY,
    MDS_EUCLIDEAN_TOLERANCE_POLICY,
    MDS_MAXIMUM_RADII,
)


MDS_EMBEDDING_VERSION = MDS_BASE_EMBEDDING_ID
MDS_CONDITION_BY_RADIUS = MappingProxyType(
    dict(zip(MDS_MAXIMUM_RADII, MDS_CONDITION_IDS, strict=True))
)


class ClassicalMDSError(RuntimeError):
    """Raised when classical MDS cannot provide two valid positive dimensions."""


@dataclass(frozen=True)
class ClassicalMDSMetadata:
    embedding_family: str
    embedding_version: str
    dimension: int
    effective_rank: int
    node_order: tuple[Hashable, ...]
    retained_eigenvalues: tuple[float, float]
    eigenvalue_relative_tolerance: float
    eigenvalue_absolute_threshold: float
    canonicalized_eigenspace_group_sizes: tuple[int, ...]
    discarded_significant_negative_count: int
    discarded_significant_negative_sum: float
    most_negative_eigenvalue: float
    tiny_negative_eigenvalue_count: int
    centroid_residual: float
    unscaled_maximum_norm: float
    coincident_coordinate_group_count: int
    coincident_vertex_count: int
    coincident_vertex_pair_count: int
    coincident_coordinate_groups: tuple[tuple[Hashable, ...], ...]
    configuration_fingerprint: str
    embedding_input_fingerprint: str


@dataclass(frozen=True)
class ClassicalMDSResult:
    coordinates: Mapping[Hashable, tuple[float, float]]
    coordinate_matrix: NDArray[np.float64]
    metadata: ClassicalMDSMetadata

    def __post_init__(self) -> None:
        if self.coordinate_matrix.flags.writeable:
            raise ValueError("ClassicalMDSResult.coordinate_matrix must be read-only")


@dataclass(frozen=True)
class MDSConditionMetadata:
    embedding_family: str
    embedding_version: str
    coordinate_condition_id: str
    dimension: int
    effective_rank: int
    node_order: tuple[Hashable, ...]
    retained_eigenvalues: tuple[float, float]
    canonicalized_eigenspace_group_sizes: tuple[int, ...]
    discarded_significant_negative_count: int
    discarded_significant_negative_sum: float
    most_negative_eigenvalue: float
    centroid_residual: float
    unscaled_maximum_norm: float
    scale_factor: float
    requested_maximum_radius: float
    achieved_maximum_radius: float
    nested_sensitivity_condition: bool
    coincident_coordinate_group_count: int
    coincident_vertex_count: int
    coincident_vertex_pair_count: int
    coincident_coordinate_groups: tuple[tuple[Hashable, ...], ...]
    configuration_fingerprint: str
    embedding_input_fingerprint: str


@dataclass(frozen=True)
class MDSConditionResult:
    coordinates: Mapping[Hashable, tuple[float, float]]
    metadata: MDSConditionMetadata


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    validated = float(value)
    if not isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _coincident_coordinate_metadata(
    coordinates: NDArray[np.float64],
    node_order: Sequence[Hashable],
) -> tuple[int, int, int, tuple[tuple[Hashable, ...], ...]]:
    """Count exact coordinate coincidences without modifying coordinates."""

    groups_by_point: dict[tuple[float, float], list[Hashable]] = {}
    for node, point in zip(node_order, coordinates, strict=True):
        key = (float(point[0]), float(point[1]))
        groups_by_point.setdefault(key, []).append(node)
    groups = tuple(
        tuple(nodes)
        for nodes in groups_by_point.values()
        if len(nodes) > 1
    )
    coincident_sizes = tuple(len(group) for group in groups)
    return (
        len(coincident_sizes),
        sum(coincident_sizes),
        sum(size * (size - 1) // 2 for size in coincident_sizes),
        groups,
    )


def scale_equivariant_euclidean_routing_tolerance(
    numerical_tolerance: float,
    maximum_radius: float,
) -> float:
    """Scale the Euclidean tie/progress tolerance with an MDS condition.

    Uniform MDS transformations multiply every Euclidean distance by the same
    radius-dependent factor. Scaling the absolute routing tolerance by
    ``radius / max(approved_radii)`` therefore preserves the same comparisons
    under every nested radius without changing the shared routing algorithm.
    """

    tolerance = _positive_finite("numerical_tolerance", numerical_tolerance)
    if isinstance(maximum_radius, bool) or not isinstance(
        maximum_radius,
        (int, float),
    ):
        raise ValueError("maximum_radius must be numeric")
    radius = float(maximum_radius)
    if radius not in MDS_CONDITION_BY_RADIUS:
        raise ValueError(
            f"maximum_radius must be one of {MDS_MAXIMUM_RADII}"
        )
    if (
        MDS_EUCLIDEAN_TOLERANCE_POLICY
        != "scale_by_radius_over_maximum_approved_radius_v1"
    ):
        raise RuntimeError("unsupported MDS Euclidean tolerance policy")
    return tolerance * radius / max(MDS_MAXIMUM_RADII)


def classical_mds(
    embedding_input: EmbeddingInput,
    *,
    dimension: int = MDS_DIMENSION,
    eigenvalue_relative_tolerance: float = 1e-12,
    centroid_tolerance: float = 1e-12,
) -> ClassicalMDSResult:
    """Compute one unscaled two-dimensional Torgerson MDS embedding."""

    if not isinstance(embedding_input, EmbeddingInput):
        raise ValueError("embedding_input must be an EmbeddingInput")
    if dimension != MDS_DIMENSION:
        raise ValueError("classical experiment MDS dimension must equal 2")
    if len(embedding_input.node_order) < dimension + 1:
        raise ValueError("classical MDS dimension 2 requires at least three points")
    relative_tolerance = _positive_finite(
        "eigenvalue_relative_tolerance", eigenvalue_relative_tolerance
    )
    validated_centroid_tolerance = _positive_finite(
        "centroid_tolerance", centroid_tolerance
    )

    distance_matrix = embedding_input.distance_matrix
    point_count = distance_matrix.shape[0]
    centering = (
        np.eye(point_count, dtype=np.float64)
        - np.full((point_count, point_count), 1.0 / point_count)
    )
    gram = -0.5 * centering @ np.square(distance_matrix) @ centering
    gram = (gram + gram.T) * 0.5
    if not np.isfinite(gram).all():
        raise ClassicalMDSError("classical MDS double-centred matrix is non-finite")

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError as exc:
        raise ClassicalMDSError(
            "classical MDS symmetric eigendecomposition did not converge"
        ) from exc
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    if not np.isfinite(eigenvalues).all():
        raise ClassicalMDSError("classical MDS eigenvalues are non-finite")
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    absolute_threshold = relative_tolerance * eigenvalue_scale
    positive_indices = np.flatnonzero(eigenvalues > absolute_threshold)
    if len(positive_indices) == 0:
        raise ClassicalMDSError(
            "classical MDS requires at least one eigenvalue above the "
            f"positive threshold {absolute_threshold:.17g}; rank 0 is unusable"
        )

    retained_indices = positive_indices[:dimension]
    effective_rank = int(len(retained_indices))
    retained_index_tuple = tuple(int(index) for index in retained_indices)
    retained_values = tuple(
        float(eigenvalues[index]) for index in retained_indices
    )
    try:
        retained_vectors, repeated_group_sizes = (
            canonical_eigenvectors_for_indices(
                eigenvalues,
                eigenvectors,
                retained_index_tuple,
                relative_tolerance=relative_tolerance,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise ClassicalMDSError(
            "classical MDS could not canonicalize its retained eigenspaces"
        ) from exc
    columns = [
        vector * np.sqrt(eigenvalues[index])
        for index, vector in zip(
            retained_index_tuple,
            retained_vectors,
            strict=True,
        )
    ]
    coordinates = np.column_stack(columns)
    if effective_rank == 1:
        coordinates = np.column_stack(
            (coordinates[:, 0], np.zeros(point_count, dtype=np.float64))
        )
    coordinates -= np.mean(coordinates, axis=0)
    centroid = np.mean(coordinates, axis=0)
    centroid_residual = float(np.linalg.norm(centroid))
    if (
        not isfinite(centroid_residual)
        or centroid_residual > validated_centroid_tolerance
    ):
        raise ClassicalMDSError(
            "classical MDS centroid residual exceeds the configured tolerance"
        )
    norms = np.linalg.norm(coordinates, axis=1)
    maximum_norm = float(np.max(norms))
    if not isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ClassicalMDSError("classical MDS coordinates collapsed")

    significant_negative = eigenvalues[eigenvalues < -absolute_threshold]
    tiny_negative = eigenvalues[
        (eigenvalues < 0.0) & (eigenvalues >= -absolute_threshold)
    ]
    most_negative = float(np.min(eigenvalues))
    coordinates = np.asarray(coordinates, dtype=np.float64)
    (
        coincident_group_count,
        coincident_vertex_count,
        coincident_pair_count,
        coincident_groups,
    ) = _coincident_coordinate_metadata(
        coordinates,
        embedding_input.node_order,
    )
    coordinates.setflags(write=False)
    coordinate_mapping = MappingProxyType(
        {
            node: (float(point[0]), float(point[1]))
            for node, point in zip(
                embedding_input.node_order,
                coordinates,
                strict=True,
            )
        }
    )
    metadata = ClassicalMDSMetadata(
        embedding_family=MDS_EMBEDDING_FAMILY,
        embedding_version=MDS_EMBEDDING_VERSION,
        dimension=MDS_DIMENSION,
        effective_rank=effective_rank,
        node_order=embedding_input.node_order,
        retained_eigenvalues=(
            retained_values[0],
            retained_values[1] if effective_rank == 2 else 0.0,
        ),
        eigenvalue_relative_tolerance=relative_tolerance,
        eigenvalue_absolute_threshold=absolute_threshold,
        canonicalized_eigenspace_group_sizes=repeated_group_sizes,
        discarded_significant_negative_count=int(len(significant_negative)),
        discarded_significant_negative_sum=float(
            np.sum(np.abs(significant_negative))
        ),
        most_negative_eigenvalue=most_negative,
        tiny_negative_eigenvalue_count=int(len(tiny_negative)),
        centroid_residual=centroid_residual,
        unscaled_maximum_norm=maximum_norm,
        coincident_coordinate_group_count=coincident_group_count,
        coincident_vertex_count=coincident_vertex_count,
        coincident_vertex_pair_count=coincident_pair_count,
        coincident_coordinate_groups=coincident_groups,
        configuration_fingerprint=embedding_input.configuration_fingerprint,
        embedding_input_fingerprint=embedding_input.input_fingerprint,
    )
    return ClassicalMDSResult(
        coordinates=coordinate_mapping,
        coordinate_matrix=coordinates,
        metadata=metadata,
    )


def transform_mds_to_radius(
    base_embedding: ClassicalMDSResult,
    maximum_radius: float,
    *,
    tolerance: float = 1e-12,
) -> MDSConditionResult:
    """Uniformly transform one base MDS embedding to an approved disk radius."""

    if not isinstance(base_embedding, ClassicalMDSResult):
        raise ValueError("base_embedding must be a ClassicalMDSResult")
    if isinstance(maximum_radius, bool) or not isinstance(
        maximum_radius, (int, float)
    ):
        raise ValueError("maximum_radius must be numeric")
    radius = float(maximum_radius)
    if radius not in MDS_CONDITION_BY_RADIUS:
        raise ValueError(
            f"maximum_radius must be one of {MDS_MAXIMUM_RADII}"
        )
    validated_tolerance = _positive_finite("tolerance", tolerance)
    scale_factor = radius / base_embedding.metadata.unscaled_maximum_norm
    transformed = np.asarray(
        base_embedding.coordinate_matrix * scale_factor,
        dtype=np.float64,
    )
    norms = np.linalg.norm(transformed, axis=1)
    achieved_radius = float(np.max(norms))
    if not np.isclose(
        achieved_radius,
        radius,
        rtol=0.0,
        atol=validated_tolerance,
    ):
        raise ClassicalMDSError(
            "uniform MDS transformation did not achieve the requested radius"
        )
    if np.any(norms >= 1.0):
        raise ClassicalMDSError("transformed MDS coordinate left the open unit disk")
    (
        coincident_group_count,
        coincident_vertex_count,
        coincident_pair_count,
        coincident_groups,
    ) = _coincident_coordinate_metadata(
        transformed,
        base_embedding.metadata.node_order,
    )

    coordinate_mapping = MappingProxyType(
        {
            node: (float(point[0]), float(point[1]))
            for node, point in zip(
                base_embedding.metadata.node_order,
                transformed,
                strict=True,
            )
        }
    )
    base = base_embedding.metadata
    return MDSConditionResult(
        coordinates=coordinate_mapping,
        metadata=MDSConditionMetadata(
            embedding_family=MDS_EMBEDDING_FAMILY,
            embedding_version=MDS_EMBEDDING_VERSION,
            coordinate_condition_id=MDS_CONDITION_BY_RADIUS[radius],
            dimension=MDS_DIMENSION,
            effective_rank=base.effective_rank,
            node_order=base.node_order,
            retained_eigenvalues=base.retained_eigenvalues,
            canonicalized_eigenspace_group_sizes=(
                base.canonicalized_eigenspace_group_sizes
            ),
            discarded_significant_negative_count=(
                base.discarded_significant_negative_count
            ),
            discarded_significant_negative_sum=(
                base.discarded_significant_negative_sum
            ),
            most_negative_eigenvalue=base.most_negative_eigenvalue,
            centroid_residual=base.centroid_residual,
            unscaled_maximum_norm=base.unscaled_maximum_norm,
            scale_factor=scale_factor,
            requested_maximum_radius=radius,
            achieved_maximum_radius=achieved_radius,
            nested_sensitivity_condition=True,
            coincident_coordinate_group_count=coincident_group_count,
            coincident_vertex_count=coincident_vertex_count,
            coincident_vertex_pair_count=coincident_pair_count,
            coincident_coordinate_groups=coincident_groups,
            configuration_fingerprint=base.configuration_fingerprint,
            embedding_input_fingerprint=base.embedding_input_fingerprint,
        ),
    )


def create_mds_sensitivity_conditions(
    base_embedding: ClassicalMDSResult,
    *,
    radii: Sequence[float] = MDS_MAXIMUM_RADII,
    tolerance: float = 1e-12,
) -> tuple[MDSConditionResult, ...]:
    """Create the four nested conditions from one already-computed base embedding."""

    requested = tuple(float(radius) for radius in radii)
    if requested != MDS_MAXIMUM_RADII:
        raise ValueError(
            f"radii must equal the approved nested sequence {MDS_MAXIMUM_RADII}"
        )
    return tuple(
        transform_mds_to_radius(
            base_embedding,
            radius,
            tolerance=tolerance,
        )
        for radius in requested
    )
