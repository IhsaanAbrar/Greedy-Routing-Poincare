"""Standard two-dimensional Hydra embedding with hyperbolic centring."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from math import atanh, isfinite, sqrt
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from embedding import EmbeddingInput, canonical_eigenvectors_for_indices
from experiment_config import (
    HYDRA_CONDITION_ID,
    HYDRA_CURVATURE,
    HYDRA_DIMENSION,
    HYDRA_EMBEDDING_FAMILY,
    HYDRA_KAPPA,
)


HYDRA_EMBEDDING_VERSION = "standard_hydra_spectral_2d_v1"
MAX_BOUNDARY_ROUNDOFF_TOLERANCE = 1e-10


class HydraEmbeddingError(RuntimeError):
    """Raised when standard Hydra cannot return a valid deterministic embedding."""


class HydraCenteringError(HydraEmbeddingError):
    """Raised when the hyperbolic Frechet mean does not converge safely."""


@dataclass(frozen=True)
class HydraEmbeddingMetadata:
    embedding_family: str
    embedding_version: str
    coordinate_condition_id: str
    dimension: int
    kappa: float
    curvature: float
    node_order: tuple[Hashable, ...]
    centering_tolerance: float
    centering_max_iterations: int
    centering_iteration_count: int
    final_frechet_mean_residual: float
    maximum_coordinate_norm: float
    boundary_roundoff_tolerance: float
    boundary_correction_occurred: bool
    boundary_correction_count: int
    pairwise_isometry_tolerance: float
    maximum_pairwise_distance_error: float
    eigenvalue_tolerance: float
    canonicalized_eigenspace_group_sizes: tuple[int, ...]
    leading_eigenvalue: float
    retained_negative_eigenvalues: tuple[float, float]
    radial_rescaling_after_centering: bool
    configuration_fingerprint: str
    embedding_input_fingerprint: str


@dataclass(frozen=True)
class HydraEmbeddingResult:
    coordinates: Mapping[Hashable, tuple[float, float]]
    metadata: HydraEmbeddingMetadata


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    validated = float(value)
    if not isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mobius_add(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return curvature-minus-one Mobius addition for one or many right points."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_squared = float(np.dot(left_array, left_array))
    right_squared = np.sum(right_array * right_array, axis=-1)
    inner = np.sum(right_array * left_array, axis=-1)
    denominator = 1.0 + 2.0 * inner + left_squared * right_squared
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise HydraEmbeddingError("Mobius addition encountered an invalid denominator")
    left_factor = 1.0 + 2.0 * inner + right_squared
    right_factor = 1.0 - left_squared
    return (
        left_factor[..., np.newaxis] * left_array
        + right_factor * right_array
    ) / denominator[..., np.newaxis]


def _log_map(
    base: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    translated = _mobius_add(-base, points)
    translated_norms = np.linalg.norm(translated, axis=1)
    if np.any(~np.isfinite(translated_norms)) or np.any(translated_norms >= 1.0):
        raise HydraCenteringError("Frechet log map left the open Poincare disk")
    base_squared = float(np.dot(base, base))
    factors = np.empty_like(translated_norms)
    zero = translated_norms == 0.0
    factors[zero] = 1.0 - base_squared
    factors[~zero] = (
        (1.0 - base_squared)
        * np.arctanh(translated_norms[~zero])
        / translated_norms[~zero]
    )
    return translated * factors[:, np.newaxis]


def _exp_map(
    base: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm == 0.0:
        return np.array(base, dtype=np.float64, copy=True)
    base_squared = float(np.dot(base, base))
    if not base_squared < 1.0:
        raise HydraCenteringError("Frechet exponential-map base is outside the disk")
    conformal_factor = 2.0 / (1.0 - base_squared)
    radial = np.tanh(conformal_factor * tangent_norm * 0.5)
    step = radial * tangent / tangent_norm
    candidate = _mobius_add(base, step)
    if candidate.shape != (2,) or not np.isfinite(candidate).all():
        raise HydraCenteringError("Frechet exponential map returned an invalid point")
    return np.asarray(candidate, dtype=np.float64)


def _distances_from(
    base: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    translated = _mobius_add(-base, points)
    norms = np.linalg.norm(translated, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms >= 1.0):
        raise HydraCenteringError("Frechet objective received an invalid disk point")
    return 2.0 * np.arctanh(norms)


def _frechet_residual(
    base: NDArray[np.float64],
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    tangent = np.mean(_log_map(base, points), axis=0)
    conformal_factor = 2.0 / (1.0 - float(np.dot(base, base)))
    residual = conformal_factor * float(np.linalg.norm(tangent))
    return tangent, residual


def hyperbolic_frechet_mean(
    points: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int, float]:
    """Compute the unweighted Poincare Frechet mean deterministically."""

    validated_tolerance = _positive_finite("tolerance", tolerance)
    validated_max_iterations = _positive_int("max_iterations", max_iterations)
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2 or not len(point_array):
        raise ValueError("points must be a non-empty n-by-2 array")
    if not np.isfinite(point_array).all():
        raise ValueError("points must be finite")
    if np.any(np.linalg.norm(point_array, axis=1) >= 1.0):
        raise ValueError("points must lie strictly inside the Poincare disk")

    mean = np.zeros(2, dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    for iteration in range(validated_max_iterations + 1):
        logarithms = _log_map(mean, point_array)
        conformal_factor = 2.0 / (1.0 - float(np.dot(mean, mean)))
        orthonormal_logs = logarithms * conformal_factor
        average_log = np.mean(orthonormal_logs, axis=0)
        residual = float(np.linalg.norm(average_log))
        if not isfinite(residual):
            raise HydraCenteringError("Frechet-mean residual became non-finite")
        if residual <= validated_tolerance:
            return mean, iteration, residual
        if iteration == validated_max_iterations:
            break

        hessian = np.zeros((2, 2), dtype=np.float64)
        for logarithm in orthonormal_logs:
            radius = float(np.linalg.norm(logarithm))
            if radius <= np.finfo(np.float64).eps:
                hessian += identity
                continue
            direction = logarithm / radius
            radial_projection = np.outer(direction, direction)
            tangential_weight = radius / np.tanh(radius)
            hessian += (
                radial_projection
                + tangential_weight * (identity - radial_projection)
            )
        hessian /= len(orthonormal_logs)
        try:
            orthonormal_direction = np.linalg.solve(hessian, average_log)
        except np.linalg.LinAlgError as exc:
            raise HydraCenteringError(
                "Frechet-mean Hessian is numerically singular"
            ) from exc
        descent = float(np.dot(average_log, orthonormal_direction))
        if not isfinite(descent) or descent <= 0.0:
            raise HydraCenteringError(
                "Frechet-mean Newton direction is not a descent direction"
            )
        coordinate_direction = orthonormal_direction / conformal_factor
        current_distances = _distances_from(mean, point_array)
        current_objective = float(
            0.5 * np.mean(current_distances * current_distances)
        )
        accepted = False
        step_size = 1.0
        for _ in range(48):
            candidate = _exp_map(mean, coordinate_direction * step_size)
            if float(np.linalg.norm(candidate)) >= 1.0:
                step_size *= 0.5
                continue
            _, candidate_residual = _frechet_residual(candidate, point_array)
            if candidate_residual <= validated_tolerance:
                return candidate, iteration + 1, candidate_residual
            candidate_distances = _distances_from(candidate, point_array)
            candidate_objective = float(
                0.5 * np.mean(candidate_distances * candidate_distances)
            )
            if (
                isfinite(candidate_objective)
                and candidate_objective
                <= current_objective - 1e-4 * step_size * descent
            ):
                mean = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            raise HydraCenteringError(
                "Frechet-mean line search failed to decrease the objective"
            )

    _, final_residual = _frechet_residual(mean, point_array)
    raise HydraCenteringError(
        "Frechet mean did not converge within "
        f"{validated_max_iterations} iterations "
        f"(residual={final_residual:.17g}, tolerance={validated_tolerance:.17g})"
    )


def _pairwise_poincare_distances(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    norms = np.linalg.norm(points, axis=1)
    boundary = (1.0 - norms) * (1.0 + norms)
    if np.any(boundary <= 0.0):
        raise HydraEmbeddingError("pairwise distance received a point outside the disk")
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    difference_norms = np.linalg.norm(differences, axis=-1)
    ratio = (
        difference_norms
        / np.sqrt(boundary[:, np.newaxis] * boundary[np.newaxis, :])
    )
    distances = 2.0 * np.arcsinh(ratio)
    np.fill_diagonal(distances, 0.0)
    return distances


def _standard_hydra_coordinates(
    distance_matrix: NDArray[np.float64],
    *,
    eigenvalue_tolerance: float,
) -> tuple[
    NDArray[np.float64],
    float,
    tuple[float, float],
    tuple[int, ...],
]:
    """Implement Algorithm 1 of Keller-Ressel and Nargang for d=2, kappa=1."""

    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.cosh(distance_matrix)
    if not np.isfinite(transformed).all():
        raise HydraEmbeddingError("cosh(distance_matrix) overflowed")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    except np.linalg.LinAlgError as exc:
        raise HydraEmbeddingError(
            "Hydra symmetric eigendecomposition did not converge"
        ) from exc
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    try:
        selected_vectors, repeated_group_sizes = (
            canonical_eigenvectors_for_indices(
                eigenvalues,
                eigenvectors,
                (0, len(eigenvalues) - 2, len(eigenvalues) - 1),
                relative_tolerance=eigenvalue_tolerance,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise HydraEmbeddingError(
            "Hydra could not canonicalize its selected eigenspaces"
        ) from exc

    leading_eigenvalue = float(eigenvalues[0])
    scale = max(1.0, abs(leading_eigenvalue))
    if leading_eigenvalue <= eigenvalue_tolerance * scale:
        raise HydraEmbeddingError("Hydra requires a positive leading eigenvalue")
    leading_vector = np.asarray(selected_vectors[0], dtype=np.float64)
    if float(np.sum(leading_vector)) < 0.0:
        leading_vector *= -1.0
    if np.any(leading_vector <= 0.0):
        raise HydraEmbeddingError(
            "Hydra leading eigenvector is not strictly positive"
        )

    retained_negative_values = (
        float(eigenvalues[-2]),
        float(eigenvalues[-1]),
    )
    spatial_columns: list[NDArray[np.float64]] = []
    for index, vector in zip(
        (-2, -1),
        selected_vectors[1:],
        strict=True,
    ):
        weight = sqrt(max(-float(eigenvalues[index]), 0.0))
        spatial_columns.append(vector * weight)
    spatial = np.column_stack(spatial_columns)
    time = sqrt(leading_eigenvalue) * leading_vector
    if np.any(~np.isfinite(time)) or np.any(time <= 0.0):
        raise HydraEmbeddingError("Hydra returned an invalid positive-time coordinate")

    minimum_time = min(1.0, float(np.min(time)))
    radial_ratio = (time - minimum_time) / (time + minimum_time)
    if np.any(radial_ratio < -eigenvalue_tolerance):
        raise HydraEmbeddingError("Hydra radial projection became negative")
    radial = np.sqrt(np.maximum(radial_ratio, 0.0))
    spatial_norms = np.linalg.norm(spatial, axis=1)
    coordinates = np.zeros((len(time), HYDRA_DIMENSION), dtype=np.float64)
    for index in range(len(time)):
        if spatial_norms[index] > eigenvalue_tolerance:
            coordinates[index] = (
                radial[index] * spatial[index] / spatial_norms[index]
            )
        elif radial[index] <= eigenvalue_tolerance:
            coordinates[index] = (0.0, 0.0)
        else:
            raise HydraEmbeddingError(
                "Hydra directional projection is undefined for a non-origin point"
            )
    if not np.isfinite(coordinates).all():
        raise HydraEmbeddingError("Hydra produced non-finite Poincare coordinates")
    if np.any(np.linalg.norm(coordinates, axis=1) >= 1.0):
        raise HydraEmbeddingError("Hydra radial projection left the open unit disk")
    return (
        coordinates,
        leading_eigenvalue,
        retained_negative_values,
        repeated_group_sizes,
    )


def embed_hydra(
    embedding_input: EmbeddingInput,
    *,
    dimension: int = HYDRA_DIMENSION,
    kappa: float = HYDRA_KAPPA,
    centering_tolerance: float = 1e-10,
    centering_max_iterations: int = 256,
    eigenvalue_tolerance: float = 1e-12,
    pairwise_isometry_tolerance: float = 1e-9,
    boundary_roundoff_tolerance: float = 1e-12,
) -> HydraEmbeddingResult:
    """Embed one shared graph-distance matrix using standard centred Hydra."""

    if not isinstance(embedding_input, EmbeddingInput):
        raise ValueError("embedding_input must be an EmbeddingInput")
    if dimension != HYDRA_DIMENSION:
        raise ValueError("standard experiment Hydra dimension must equal 2")
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)):
        raise ValueError("kappa must be numeric")
    if float(kappa) != HYDRA_KAPPA:
        raise ValueError(
            "standard experiment Hydra uses kappa=1 (sectional curvature -1)"
        )
    if len(embedding_input.node_order) < dimension + 1:
        raise ValueError("Hydra dimension 2 requires at least three points")
    validated_centering_tolerance = _positive_finite(
        "centering_tolerance", centering_tolerance
    )
    validated_iterations = _positive_int(
        "centering_max_iterations", centering_max_iterations
    )
    validated_eigenvalue_tolerance = _positive_finite(
        "eigenvalue_tolerance", eigenvalue_tolerance
    )
    validated_isometry_tolerance = _positive_finite(
        "pairwise_isometry_tolerance", pairwise_isometry_tolerance
    )
    validated_boundary_tolerance = _positive_finite(
        "boundary_roundoff_tolerance", boundary_roundoff_tolerance
    )
    if validated_boundary_tolerance > MAX_BOUNDARY_ROUNDOFF_TOLERANCE:
        raise ValueError(
            "boundary_roundoff_tolerance must remain at most "
            f"{MAX_BOUNDARY_ROUNDOFF_TOLERANCE} to prevent hidden rescaling"
        )

    (
        uncentered,
        leading,
        retained_negative,
        repeated_group_sizes,
    ) = _standard_hydra_coordinates(
        embedding_input.distance_matrix,
        eigenvalue_tolerance=validated_eigenvalue_tolerance,
    )
    distances_before = _pairwise_poincare_distances(uncentered)
    mean, iteration_count, _ = hyperbolic_frechet_mean(
        uncentered,
        tolerance=validated_centering_tolerance,
        max_iterations=validated_iterations,
    )
    centered = np.asarray(_mobius_add(-mean, uncentered), dtype=np.float64)

    correction_count = 0
    for index, norm in enumerate(np.linalg.norm(centered, axis=1)):
        if not isfinite(float(norm)):
            raise HydraEmbeddingError("centred Hydra coordinate norm is non-finite")
        if norm >= 1.0:
            if norm > 1.0 + validated_boundary_tolerance:
                raise HydraEmbeddingError(
                    "centred Hydra coordinate exceeds the disk beyond roundoff"
                )
            centered[index] *= float(np.nextafter(1.0, 0.0)) / float(norm)
            correction_count += 1

    centered_norms = np.linalg.norm(centered, axis=1)
    if np.any(centered_norms >= 1.0):
        raise HydraEmbeddingError("centred Hydra coordinates are not inside the disk")
    _, final_residual = _frechet_residual(np.zeros(2), centered)
    if final_residual > validated_centering_tolerance * 2.0:
        raise HydraCenteringError(
            "centred coordinates do not place the Frechet mean at the origin "
            f"(residual={final_residual:.17g})"
        )

    distances_after = _pairwise_poincare_distances(centered)
    absolute_errors = np.abs(distances_after - distances_before)
    allowed_errors = validated_isometry_tolerance * np.maximum(
        1.0, distances_before
    )
    if np.any(absolute_errors > allowed_errors):
        raise HydraEmbeddingError(
            "Frechet centring failed the pairwise Poincare-distance isometry check"
        )
    maximum_distance_error = float(np.max(absolute_errors))
    maximum_norm = float(np.max(centered_norms))
    coordinates = MappingProxyType(
        {
            node: (float(point[0]), float(point[1]))
            for node, point in zip(
                embedding_input.node_order,
                centered,
                strict=True,
            )
        }
    )
    return HydraEmbeddingResult(
        coordinates=coordinates,
        metadata=HydraEmbeddingMetadata(
            embedding_family=HYDRA_EMBEDDING_FAMILY,
            embedding_version=HYDRA_EMBEDDING_VERSION,
            coordinate_condition_id=HYDRA_CONDITION_ID,
            dimension=HYDRA_DIMENSION,
            kappa=HYDRA_KAPPA,
            curvature=HYDRA_CURVATURE,
            node_order=embedding_input.node_order,
            centering_tolerance=validated_centering_tolerance,
            centering_max_iterations=validated_iterations,
            centering_iteration_count=iteration_count,
            final_frechet_mean_residual=final_residual,
            maximum_coordinate_norm=maximum_norm,
            boundary_roundoff_tolerance=validated_boundary_tolerance,
            boundary_correction_occurred=correction_count > 0,
            boundary_correction_count=correction_count,
            pairwise_isometry_tolerance=validated_isometry_tolerance,
            maximum_pairwise_distance_error=maximum_distance_error,
            eigenvalue_tolerance=validated_eigenvalue_tolerance,
            canonicalized_eigenspace_group_sizes=repeated_group_sizes,
            leading_eigenvalue=leading,
            retained_negative_eigenvalues=retained_negative,
            radial_rescaling_after_centering=False,
            configuration_fingerprint=(
                embedding_input.configuration_fingerprint
            ),
            embedding_input_fingerprint=embedding_input.input_fingerprint,
        ),
    )
