"""Balanced native and matched-radius coordinate construction for Iteration 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import atanh, hypot, isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from embedding import EmbeddingInput
from hydra_embedding import HydraEmbeddingResult, embed_hydra
from iteration2_config import (
    BOUNDARY_ROUNDOFF_TOLERANCE,
    DATA_GENERATION_HASH,
    HYDRA_CENTERING_MAX_ITERATIONS,
    HYDRA_CENTERING_TOLERANCE,
    HYDRA_DIMENSION,
    HYDRA_KAPPA,
    ISOMETRY_ABSOLUTE_TOLERANCE,
    ISOMETRY_RELATIVE_TOLERANCE,
    MATCHED_RADII,
    MATCHED_RADIUS_LABELS,
    MDS_CENTROID_TOLERANCE,
    ROUTING_ROUNDOFF_MULTIPLIER,
    SPECTRAL_RELATIVE_TOLERANCE,
)
from mds_embedding import ClassicalMDSResult, classical_mds
from iteration2_embedding_oracle import (
    audit_embedding,
    independent_distortion_diagnostic,
    independent_hydra,
)
from iteration2_runtime_guard import scientific_operation_boundary


@dataclass(frozen=True)
class CoordinateCondition:
    condition_id: str
    embedding_family: str
    condition_kind: str
    coordinates: Mapping[Hashable, tuple[float, float]]
    original_maximum_radius: float
    transformed_maximum_radius: float
    target_maximum_radius: float | None
    scale_factor: float
    sensitivity_transformation: bool
    standard_native_embedding: bool
    poincare_routable: bool
    coincidence_groups: tuple[tuple[Hashable, ...], ...]


@dataclass(frozen=True)
class PairwiseDistanceChange:
    condition_id: str
    embedding_family: str
    unordered_pair_count: int
    exact_coincident_pair_count: int
    original_maximum_radius: float
    transformed_maximum_radius: float
    scale_factor: float
    original_euclidean_mean: float
    transformed_euclidean_mean: float
    euclidean_mean_absolute_change: float
    euclidean_maximum_scale_error: float
    original_poincare_status: str
    original_poincare_mean: float | None
    transformed_poincare_mean: float
    poincare_mean_absolute_change: float | None
    poincare_maximum_absolute_change: float | None


@dataclass(frozen=True)
class Iteration2Embeddings:
    hydra_native: CoordinateCondition
    mds_native: CoordinateCondition
    matched_conditions: tuple[CoordinateCondition, ...]
    distance_changes: tuple[PairwiseDistanceChange, ...]
    hydra_result: HydraEmbeddingResult
    mds_result: ClassicalMDSResult
    hydra_uncentered_reference: Mapping[Hashable, tuple[float, float]]
    diagnostics: tuple[dict[str, object], ...]
    independent_validation: tuple[dict[str, object], ...]

    @property
    def routable_conditions(self) -> tuple[CoordinateCondition, ...]:
        return (self.hydra_native, *self.matched_conditions)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "data_generation_hash": DATA_GENERATION_HASH,
            "native": {
                "hydra": _condition_metadata(self.hydra_native),
                "mds": _condition_metadata(self.mds_native),
            },
            "matched": [
                _condition_metadata(condition)
                for condition in self.matched_conditions
            ],
            "distance_changes": [
                asdict(change) for change in self.distance_changes
            ],
            "embedding_diagnostics": list(self.diagnostics),
            "independent_embedding_validation": list(
                self.independent_validation
            ),
        }


def _matrix(
    coordinates: Mapping[Hashable, Sequence[float]],
    node_order: Sequence[Hashable],
) -> NDArray[np.float64]:
    if set(coordinates) != set(node_order):
        raise ValueError("coordinates must cover the exact node order")
    values = np.asarray(
        [coordinates[node] for node in node_order],
        dtype=np.float64,
    )
    if values.shape != (len(node_order), 2) or not np.isfinite(values).all():
        raise ValueError("coordinates must be finite float64 two-vectors")
    return values


def _coincidences(
    values: NDArray[np.float64],
    node_order: Sequence[Hashable],
) -> tuple[tuple[Hashable, ...], ...]:
    groups: dict[tuple[float, float], list[Hashable]] = defaultdict(list)
    for node, point in zip(node_order, values, strict=True):
        groups[(float(point[0]), float(point[1]))].append(node)
    return tuple(
        tuple(group)
        for group in groups.values()
        if len(group) > 1
    )


def _mapping(
    values: NDArray[np.float64],
    node_order: Sequence[Hashable],
) -> Mapping[Hashable, tuple[float, float]]:
    return MappingProxyType(
        {
            node: (float(point[0]), float(point[1]))
            for node, point in zip(node_order, values, strict=True)
        }
    )


def _maximum_radius(values: NDArray[np.float64]) -> float:
    maximum = float(np.max(np.linalg.norm(values, axis=1)))
    if not isfinite(maximum) or maximum <= 0.0:
        raise ValueError("complete coordinate collapse is forbidden")
    return maximum


def _condition_metadata(condition: CoordinateCondition) -> dict[str, object]:
    return {
        "condition_id": condition.condition_id,
        "embedding_family": condition.embedding_family,
        "condition_kind": condition.condition_kind,
        "original_maximum_radius": condition.original_maximum_radius,
        "transformed_maximum_radius": (
            condition.transformed_maximum_radius
        ),
        "target_maximum_radius": condition.target_maximum_radius,
        "scale_factor": condition.scale_factor,
        "sensitivity_transformation": condition.sensitivity_transformation,
        "standard_native_embedding": condition.standard_native_embedding,
        "poincare_routable": condition.poincare_routable,
        "coincident_coordinate_group_count": len(
            condition.coincidence_groups
        ),
        "coincident_vertex_count": sum(
            len(group) for group in condition.coincidence_groups
        ),
        "coincident_vertex_pair_count": sum(
            len(group) * (len(group) - 1) // 2
            for group in condition.coincidence_groups
        ),
        "coincident_coordinate_groups": [
            list(group) for group in condition.coincidence_groups
        ],
    }


def native_condition(
    condition_id: str,
    embedding_family: str,
    coordinates: Mapping[Hashable, Sequence[float]],
    node_order: Sequence[Hashable],
) -> CoordinateCondition:
    values = _matrix(coordinates, node_order)
    maximum = _maximum_radius(values)
    poincare_routable = bool(np.all(np.linalg.norm(values, axis=1) < 1.0))
    return CoordinateCondition(
        condition_id=condition_id,
        embedding_family=embedding_family,
        condition_kind="native",
        coordinates=_mapping(np.array(values, copy=True), node_order),
        original_maximum_radius=maximum,
        transformed_maximum_radius=maximum,
        target_maximum_radius=None,
        scale_factor=1.0,
        sensitivity_transformation=False,
        standard_native_embedding=True,
        poincare_routable=poincare_routable,
        coincidence_groups=_coincidences(values, node_order),
    )


def uniformly_map_to_radius(
    *,
    embedding_family: str,
    condition_id: str,
    coordinates: Mapping[Hashable, Sequence[float]],
    node_order: Sequence[Hashable],
    target_radius: float,
) -> CoordinateCondition:
    """Apply the same deterministic maximum-norm scaling to either family."""

    if embedding_family not in ("hydra", "mds"):
        raise ValueError("embedding_family must be hydra or mds")
    if target_radius not in MATCHED_RADII:
        raise ValueError(f"target_radius must be one of {MATCHED_RADII}")
    values = _matrix(coordinates, node_order)
    original_maximum = _maximum_radius(values)
    scale_factor = target_radius / original_maximum
    transformed = np.asarray(values * scale_factor, dtype=np.float64)
    transformed_maximum = _maximum_radius(transformed)
    tolerance = (
        ROUTING_ROUNDOFF_MULTIPLIER
        * np.finfo(np.float64).eps
        * target_radius
    )
    if not np.isclose(
        transformed_maximum,
        target_radius,
        rtol=0.0,
        atol=tolerance,
    ):
        raise RuntimeError("uniform scaling did not reach the target radius")
    if np.any(np.linalg.norm(transformed, axis=1) >= 1.0):
        raise RuntimeError("matched-radius coordinate left the open unit disk")
    if len(set(map(tuple, transformed))) <= 1:
        raise RuntimeError("uniform scaling produced complete coordinate collapse")
    return CoordinateCondition(
        condition_id=condition_id,
        embedding_family=embedding_family,
        condition_kind="matched_radius_sensitivity",
        coordinates=_mapping(transformed, node_order),
        original_maximum_radius=original_maximum,
        transformed_maximum_radius=transformed_maximum,
        target_maximum_radius=target_radius,
        scale_factor=scale_factor,
        sensitivity_transformation=True,
        standard_native_embedding=False,
        poincare_routable=True,
        coincidence_groups=_coincidences(transformed, node_order),
    )


def euclidean_routing_tolerance(condition: CoordinateCondition) -> float:
    """Return a scale-equivariant float64 roundoff guard."""

    if not isinstance(condition, CoordinateCondition):
        raise ValueError("condition must be a CoordinateCondition")
    scale = condition.transformed_maximum_radius
    return ROUTING_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * scale


def poincare_routing_tolerance(condition: CoordinateCondition) -> float:
    """Return a curvature-scale-aware float64 roundoff guard."""

    if not condition.poincare_routable:
        raise ValueError("Poincare routing is undefined for this condition")
    radius = condition.transformed_maximum_radius
    if not 0.0 < radius < 1.0:
        raise ValueError("Poincare condition must have maximum radius in (0,1)")
    distance_scale_bound = max(1.0, 4.0 * atanh(radius))
    return (
        ROUTING_ROUNDOFF_MULTIPLIER
        * np.finfo(np.float64).eps
        * distance_scale_bound
    )


def _pairwise_euclidean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    differences = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return distances[np.triu_indices(len(values), 1)]


def _pairwise_poincare(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms >= 1.0):
        raise ValueError("Poincare distance is undefined outside the disk")
    boundary = (1.0 - norms) * (1.0 + norms)
    differences = values[:, None, :] - values[None, :, :]
    difference_norms = np.linalg.norm(differences, axis=2)
    denominator = np.sqrt(boundary[:, None] * boundary[None, :])
    distances = 2.0 * np.arcsinh(difference_norms / denominator)
    return distances[np.triu_indices(len(values), 1)]


def pairwise_distance_change(
    native: CoordinateCondition,
    transformed: CoordinateCondition,
    node_order: Sequence[Hashable],
) -> PairwiseDistanceChange:
    if native.embedding_family != transformed.embedding_family:
        raise ValueError("distance change requires one embedding family")
    before = _matrix(native.coordinates, node_order)
    after = _matrix(transformed.coordinates, node_order)
    euclidean_before = _pairwise_euclidean(before)
    euclidean_after = _pairwise_euclidean(after)
    expected = euclidean_before * transformed.scale_factor
    maximum_scale_error = float(np.max(np.abs(euclidean_after - expected)))
    nonzero = euclidean_before > 0.0
    coincident = int(np.count_nonzero(~nonzero))

    poincare_after = _pairwise_poincare(after)
    if native.poincare_routable:
        poincare_before = _pairwise_poincare(before)
        poincare_status = "defined"
        poincare_original_mean: float | None = float(
            np.mean(poincare_before)
        )
        poincare_mean_change: float | None = float(
            np.mean(np.abs(poincare_after - poincare_before))
        )
        poincare_max_change: float | None = float(
            np.max(np.abs(poincare_after - poincare_before))
        )
    else:
        poincare_status = "undefined_native_coordinates_outside_open_disk"
        poincare_original_mean = None
        poincare_mean_change = None
        poincare_max_change = None
    return PairwiseDistanceChange(
        condition_id=transformed.condition_id,
        embedding_family=transformed.embedding_family,
        unordered_pair_count=len(euclidean_before),
        exact_coincident_pair_count=coincident,
        original_maximum_radius=native.original_maximum_radius,
        transformed_maximum_radius=transformed.transformed_maximum_radius,
        scale_factor=transformed.scale_factor,
        original_euclidean_mean=float(np.mean(euclidean_before)),
        transformed_euclidean_mean=float(np.mean(euclidean_after)),
        euclidean_mean_absolute_change=float(
            np.mean(np.abs(euclidean_after - euclidean_before))
        ),
        euclidean_maximum_scale_error=maximum_scale_error,
        original_poincare_status=poincare_status,
        original_poincare_mean=poincare_original_mean,
        transformed_poincare_mean=float(np.mean(poincare_after)),
        poincare_mean_absolute_change=poincare_mean_change,
        poincare_maximum_absolute_change=poincare_max_change,
    )


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _scale_fitted_diagnostic(
    graph_distances: NDArray[np.float64],
    geometric_distances: NDArray[np.float64],
) -> dict[str, object]:
    if (
        graph_distances.shape != geometric_distances.shape
        or not len(graph_distances)
        or not np.isfinite(graph_distances).all()
        or not np.isfinite(geometric_distances).all()
        or np.any(graph_distances <= 0.0)
        or np.any(geometric_distances < 0.0)
    ):
        raise ValueError("distortion requires finite paired positive graph distances")
    denominator = float(np.dot(geometric_distances, geometric_distances))
    if denominator <= 0.0:
        raise ValueError("complete pairwise collapse is not diagnosable")
    alpha = float(np.dot(graph_distances, geometric_distances) / denominator)
    residual = alpha * geometric_distances - graph_distances
    relative = residual / graph_distances
    graph_ranks = _average_ranks(graph_distances)
    geometry_ranks = _average_ranks(geometric_distances)
    correlation = (
        None
        if float(np.std(graph_ranks)) == 0.0
        or float(np.std(geometry_ranks)) == 0.0
        else float(np.corrcoef(graph_ranks, geometry_ranks)[0, 1])
    )
    stress = float(np.linalg.norm(residual) / np.linalg.norm(graph_distances))
    result: dict[str, object] = {
        "status": "defined",
        "applicability": "applicable",
        "na_reason": None,
        "fitted_scale": alpha,
        "scale_fitted_mean_absolute_distance_error": float(
            np.mean(np.abs(residual))
        ),
        "scale_fitted_rmse_distance": float(
            np.sqrt(np.mean(np.square(residual)))
        ),
        "scale_fitted_mean_relative_error": float(
            np.mean(np.abs(relative))
        ),
        "scale_fitted_rmse_relative_error": float(
            np.sqrt(np.mean(np.square(relative)))
        ),
        "graph_distance_rank_correlation": correlation,
        "graph_distance_rank_correlation_status": (
            "defined" if correlation is not None else "undefined"
        ),
        "graph_distance_rank_correlation_na_reason": (
            None if correlation is not None else "zero_rank_variance"
        ),
        "normalized_distance_stress": stress,
    }
    oracle = independent_distortion_diagnostic(
        graph_distances, geometric_distances
    )
    for key, expected in oracle.items():
        observed = result[key]
        if expected is None:
            if observed is not None:
                raise RuntimeError("distortion oracle missingness disagreement")
        elif observed is None or not np.isclose(
            float(observed), float(expected), rtol=5e-13, atol=5e-15
        ):
            raise RuntimeError(f"distortion oracle disagreed for {key}")
    result["independent_oracle_agreement"] = True
    return result


def _spectral_diagnostic(
    distance_matrix: NDArray[np.float64],
    family: str,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if family == "hydra":
        eigenvalues = np.linalg.eigvalsh(np.cosh(distance_matrix))[::-1]
        threshold = SPECTRAL_RELATIVE_TOLERANCE * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        )
        selected = {
            index
            for index in (0, len(eigenvalues) - 2, len(eigenvalues) - 1)
            if (
                eigenvalues[index] > threshold
                if index == 0
                else eigenvalues[index] < -threshold
            )
        }
    else:
        count = len(distance_matrix)
        centering = np.eye(count) - np.ones((count, count)) / count
        gram = -0.5 * centering @ np.square(distance_matrix) @ centering
        eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2.0)[::-1]
        threshold = SPECTRAL_RELATIVE_TOLERANCE * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        )
        selected = set(
            int(index)
            for index in np.flatnonzero(eigenvalues > threshold)[:2]
        )
    threshold = SPECTRAL_RELATIVE_TOLERANCE * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    retained = tuple(
        float(value)
        for index, value in enumerate(eigenvalues)
        if index in selected
    )
    rejected = tuple(
        float(value)
        for index, value in enumerate(eigenvalues)
        if index not in selected
    )
    return retained, rejected, threshold


def _embedding_diagnostic(
    embedding_input: EmbeddingInput,
    condition: CoordinateCondition,
    *,
    hydra: HydraEmbeddingResult,
    mds: ClassicalMDSResult,
) -> dict[str, object]:
    nodes = embedding_input.node_order
    values = _matrix(condition.coordinates, nodes)
    radii = np.linalg.norm(values, axis=1)
    graph_matrix = np.asarray(
        embedding_input.distance_matrix,
        dtype=np.float64,
    )
    graph_pairs = graph_matrix[np.triu_indices(len(nodes), 1)]
    euclidean_pairs = _pairwise_euclidean(values)
    euclidean = _scale_fitted_diagnostic(graph_pairs, euclidean_pairs)
    poincare: dict[str, object] | None = None
    if condition.poincare_routable:
        poincare = _scale_fitted_diagnostic(
            graph_pairs,
            _pairwise_poincare(values),
        )
    retained, rejected, threshold = _spectral_diagnostic(
        graph_matrix,
        condition.embedding_family,
    )
    affected = tuple(
        node for group in condition.coincidence_groups for node in group
    )
    if condition.embedding_family == "hydra":
        effective_rank = hydra.metadata.effective_spatial_rank
        frechet_iterations: int | None = (
            hydra.metadata.centering_iteration_count
        )
        frechet_residual: float | None = (
            hydra.metadata.final_frechet_mean_residual
        )
        absolute_invariance: float | None = (
            hydra.metadata.maximum_pairwise_distance_error
        )
        relative_invariance: float | None = (
            hydra.metadata.maximum_pairwise_normalized_error
        )
        centroid_residual: float | None = None
        boundary_count = hydra.metadata.boundary_correction_count
        boundary_magnitude: float | None = (
            0.0
            if boundary_count == 0
            else hydra.metadata.boundary_roundoff_tolerance
        )
        if poincare is None:
            algorithmic_objective_value = None
        else:
            target = np.cosh(graph_pairs)
            reconstructed = np.cosh(_pairwise_poincare(values))
            algorithmic_objective_value = float(
                np.linalg.norm(reconstructed - target)
                / np.linalg.norm(target)
            )
        algorithmic_objective_name = "hydra_lorentz_cosh_strain"
    else:
        effective_rank = mds.metadata.effective_rank
        frechet_iterations = None
        frechet_residual = None
        absolute_invariance = None
        relative_invariance = None
        centroid_residual = mds.metadata.centroid_residual
        boundary_count = 0
        boundary_magnitude = 0.0
        count = len(graph_matrix)
        centering = np.eye(count) - np.ones((count, count)) / count
        target_gram = (
            -0.5 * centering @ np.square(graph_matrix) @ centering
        )
        reconstructed_gram = values @ values.T
        algorithmic_objective_name = "classical_mds_gram_strain"
        algorithmic_objective_value = float(
            np.linalg.norm(reconstructed_gram - target_gram)
            / np.linalg.norm(target_gram)
        )
    return {
        "condition_id": condition.condition_id,
        "embedding_family": condition.embedding_family,
        "native_or_transformed": (
            "native" if condition.standard_native_embedding else "transformed"
        ),
        "dimension_requested": 2,
        "effective_rank": effective_rank,
        "curvature": -1.0 if condition.embedding_family == "hydra" else None,
        "eigenvalue_threshold": threshold,
        "retained_eigenvalues": retained,
        "rejected_eigenvalues": rejected,
        "maximum_euclidean_radius": float(np.max(radii)),
        "minimum_radius": float(np.min(radii)),
        "q1_radius": float(np.quantile(radii, 0.25)),
        "median_radius": float(np.median(radii)),
        "mean_radius": float(np.mean(radii)),
        "q3_radius": float(np.quantile(radii, 0.75)),
        "radial_standard_deviation": float(np.std(radii, ddof=0)),
        "boundary_correction_count": boundary_count,
        "maximum_boundary_correction_magnitude_or_bound": boundary_magnitude,
        "frechet_iterations": frechet_iterations,
        "frechet_final_residual": frechet_residual,
        "centering_invariance_maximum_absolute_error": absolute_invariance,
        "centering_invariance_maximum_normalized_error": relative_invariance,
        "mds_centroid_residual": centroid_residual,
        "coincident_coordinate_groups": condition.coincidence_groups,
        "affected_vertices": affected,
        "coincident_pairs": sum(
            len(group) * (len(group) - 1) // 2
            for group in condition.coincidence_groups
        ),
        "complete_collapse": False,
        "finite_coordinates": bool(np.isfinite(values).all()),
        "disk_membership_margin": (
            1.0 - float(np.max(radii))
            if condition.poincare_routable
            else None
        ),
        "euclidean_distortion": euclidean,
        "poincare_distortion": poincare,
        "poincare_distortion_status": (
            "defined" if poincare is not None else "not_applicable"
        ),
        "poincare_distortion_applicability": (
            "applicable" if poincare is not None else "not_applicable"
        ),
        "poincare_distortion_na_reason": (
            None
            if poincare is not None
            else "coordinates_outside_open_unit_disk"
        ),
        "algorithmic_objective_name": algorithmic_objective_name,
        "algorithmic_objective_value": algorithmic_objective_value,
        "algorithmic_objective_applicability": (
            "native_embedding_optimized_objective"
            if condition.standard_native_embedding
            else "post_transformation_diagnostic_not_optimized"
        ),
        "distance_stress_is_algorithmic_objective": False,
        "distortion_cross_geometry_comparability": (
            "not_directly_commensurate_across_embedding_objectives"
        ),
    }


@scientific_operation_boundary("scientific_embedding_coordinates")
def create_iteration2_embeddings(
    embedding_input: EmbeddingInput,
) -> Iteration2Embeddings:
    """Create two independent native bases and the fully crossed radius grid."""

    if not isinstance(embedding_input, EmbeddingInput):
        raise ValueError("embedding_input must be an EmbeddingInput")
    hydra = embed_hydra(
        embedding_input,
        dimension=HYDRA_DIMENSION,
        kappa=HYDRA_KAPPA,
        centering_tolerance=HYDRA_CENTERING_TOLERANCE,
        centering_max_iterations=HYDRA_CENTERING_MAX_ITERATIONS,
        eigenvalue_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
        pairwise_isometry_absolute_tolerance=ISOMETRY_ABSOLUTE_TOLERANCE,
        pairwise_isometry_tolerance=ISOMETRY_RELATIVE_TOLERANCE,
        boundary_roundoff_tolerance=BOUNDARY_ROUNDOFF_TOLERANCE,
    )
    if (
        hydra.metadata.radial_rescaling_after_centering
        or hydra.metadata.maximum_pairwise_normalized_error > 1.0
        or hydra.metadata.final_frechet_mean_residual
        > HYDRA_CENTERING_TOLERANCE
    ):
        raise RuntimeError("native Hydra preservation/isometry check failed")
    mds = classical_mds(
        embedding_input,
        dimension=2,
        eigenvalue_relative_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
        centroid_tolerance=MDS_CENTROID_TOLERANCE,
    )
    nodes = embedding_input.node_order
    hydra_native = native_condition(
        "hydra_native", "hydra", hydra.coordinates, nodes
    )
    mds_native = native_condition(
        "mds_native", "mds", mds.coordinates, nodes
    )
    if not hydra_native.poincare_routable:
        raise RuntimeError("native Hydra must lie in the open disk")
    matched: list[CoordinateCondition] = []
    changes: list[PairwiseDistanceChange] = []
    for family, native in (
        ("hydra", hydra_native),
        ("mds", mds_native),
    ):
        for radius, label in zip(
            MATCHED_RADII,
            MATCHED_RADIUS_LABELS,
            strict=True,
        ):
            condition = uniformly_map_to_radius(
                embedding_family=family,
                condition_id=f"{family}_scaled_{label}",
                coordinates=native.coordinates,
                node_order=nodes,
                target_radius=radius,
            )
            matched.append(condition)
            changes.append(pairwise_distance_change(native, condition, nodes))
    diagnostics = tuple(
        _embedding_diagnostic(
            embedding_input,
            condition,
            hydra=hydra,
            mds=mds,
        )
        for condition in (hydra_native, mds_native, *matched)
    )
    hydra_reference = independent_hydra(
        embedding_input.distance_matrix,
        nodes,
        relative_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
    )
    validations = (
        audit_embedding(
            family="hydra",
            distance_matrix=embedding_input.distance_matrix,
            node_order=nodes,
            production_coordinates=hydra.coordinates,
            production_effective_rank=hydra.metadata.effective_spatial_rank,
            relative_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
            reference=hydra_reference,
        ),
        audit_embedding(
            family="mds",
            distance_matrix=embedding_input.distance_matrix,
            node_order=nodes,
            production_coordinates=mds.coordinates,
            production_effective_rank=mds.metadata.effective_rank,
            relative_tolerance=SPECTRAL_RELATIVE_TOLERANCE,
        ),
    )
    if any(
        item.maximum_pairwise_distance_error > 1e-8
        or item.reference_effective_rank != item.production_effective_rank
        for item in validations
    ):
        raise RuntimeError("independent embedding reconstruction disagreed")
    return Iteration2Embeddings(
        hydra_native=hydra_native,
        mds_native=mds_native,
        matched_conditions=tuple(matched),
        distance_changes=tuple(changes),
        hydra_result=hydra,
        mds_result=mds,
        hydra_uncentered_reference=hydra_reference.coordinates,
        diagnostics=diagnostics,
        independent_validation=tuple(
            asdict(item) for item in validations
        ),
    )


def maximum_radius(
    coordinates: Mapping[Hashable, Sequence[float]],
) -> float:
    """Small public helper used by validators and tests."""

    maximum = max(
        hypot(float(point[0]), float(point[1]))
        for point in coordinates.values()
    )
    if not isfinite(maximum):
        raise ValueError("coordinate radius is non-finite")
    return maximum
